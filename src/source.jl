"""
    SourceTerm

A forcing term added to the heat equation:

```
du/dt = alpha * laplacian(u) + q(x, y, t)
```

Without one, `Heat2D` is a closed system that can only redistribute the heat it
started with. A twin of a real installation almost always needs one, because
real systems have heaters, pumps, losses and weather acting on them
continuously.

Three forms, covering the cases the course needs:

| Type | `q` | Use |
|:-----|:----|:----|
| [`NoSource`](@ref) | `0` | closed system; compiles away entirely |
| [`UniformSource`](@ref) | `rate` everywhere | ambient gain or loss over the whole domain |
| [`PatternSource`](@ref) | `rate * pattern[i,j]` | heaters, pipes, or any fixed spatial layout |

`PatternSource` splits *where* the forcing acts from *how strong it is*. That
split is what makes it drivable: the pattern is fixed geometry that can live on
the GPU untouched, while the rate is one scalar per step that can come from a
measurement series. See [`TimeSeries`](@ref).

An additive source does not affect the stability limit of the explicit scheme,
so [`cfl_number`](@ref) is unchanged by it. It can still make the solution grow
without bound — that is physics, not instability, and the two are worth
distinguishing in a report.
"""
abstract type SourceTerm end

"""
    NoSource()

No forcing. The default, and free: the stencil generated for `NoSource` is
identical to one written without any source support at all.
"""
struct NoSource <: SourceTerm end

"""
    UniformSource(rate)

Adds `rate` to every cell per unit of simulated time. `rate` may be a number, or
a callable of time for a source that varies (see [`TimeSeries`](@ref)).

Negative rates are losses, which is the usual way to represent a domain leaking
heat to an ambient environment.
"""
struct UniformSource{T} <: SourceTerm
    rate::T
end

"""
    PatternSource(pattern, rate = 1)

Adds `rate * pattern[i, j]` to each cell per unit of simulated time.

`pattern` is fixed spatial geometry — where the heaters are — and `rate` scales
all of it together. `rate` may be a number or a callable of time.

The pattern moves to the device automatically with [`to_backend`](@ref), so a
driven model runs on a GPU without any further handling.
"""
struct PatternSource{T,A<:AbstractMatrix} <: SourceTerm
    pattern::A
    rate::T
end

PatternSource(pattern::AbstractMatrix) = PatternSource(pattern, true)

# ---------------------------------------------------------------------------
# Applying a source inside the stencil
# ---------------------------------------------------------------------------
#
# Written as "add to an already-computed value" rather than "return q", so that
# NoSource is a literal identity function. A `return zero(T)` version would still
# emit an addition per cell in the hot loop.

@inline apply_source(::NoSource, value, i, j, dt) = value

@inline apply_source(source::UniformSource, value, i, j, dt) =
    value + dt * source.rate

@inline function apply_source(source::PatternSource, value, i, j, dt)
    @inbounds return value + dt * source.rate * source.pattern[i, j]
end

# ---------------------------------------------------------------------------
# Resolving a time-varying source to something a kernel can take
# ---------------------------------------------------------------------------

"""
    resolve(source, t, ::Type{T}) -> SourceTerm

Evaluate any time-dependent rate at simulated time `t` and convert to the field
element type, producing an `isbits` value the kernel can be launched with.

This happens once per step on the host. A GPU kernel cannot call an arbitrary
Julia closure, and would not want to re-evaluate the same scalar in every one of
a million threads even if it could.
"""
# As for boundaries, the split is "number" versus "something to call", so that a
# callable struct such as TimeSeries works without subtyping Function.
@inline resolve(source::NoSource, t, ::Type{T}) where {T} = source

@inline resolve(source::UniformSource{<:Number}, t, ::Type{T}) where {T} =
    UniformSource(convert(T, source.rate))

@inline resolve(source::UniformSource, t, ::Type{T}) where {T} =
    UniformSource(convert(T, source.rate(t)))

@inline resolve(source::PatternSource{<:Number}, t, ::Type{T}) where {T} =
    PatternSource(source.pattern, convert(T, source.rate))

@inline resolve(source::PatternSource, t, ::Type{T}) where {T} =
    PatternSource(source.pattern, convert(T, source.rate(t)))

"""
    is_driven(source) -> Bool

Whether the source's rate depends on simulated time.
"""
is_driven(::SourceTerm) = false
is_driven(::UniformSource) = true
is_driven(::UniformSource{<:Number}) = false
is_driven(::PatternSource) = true
is_driven(::PatternSource{<:Number}) = false

source_array(::SourceTerm) = nothing
source_array(source::PatternSource) = source.pattern

with_array(source::SourceTerm, ::Any) = source
with_array(source::PatternSource, pattern) = PatternSource(pattern, source.rate)

# A `PatternSource` holding a device array still cannot be passed to a kernel as
# it stands: `MtlMatrix`/`CuArray` are *host-side* wrappers around a buffer
# reference, and a kernel argument has to be isbits. Adapt is the mechanism the
# GPU packages use to rewrite such wrappers into plain device pointers at launch
# time, and this line is what extends it to our struct.
#
# Without it the kernel fails to compile with "passing non-bitstype argument",
# which is the standard symptom of a custom struct carrying an array onto a GPU.
Adapt.@adapt_structure PatternSource

# Compact display. Without these, showing a model with a PatternSource prints the
# whole grid-sized pattern array.
describe_rate(rate::Number) = repr(rate)
describe_rate(rate) = string("driven by ", rate)

Base.show(io::IO, ::NoSource) = print(io, "NoSource()")

Base.show(io::IO, source::UniformSource) =
    print(io, "UniformSource(", describe_rate(source.rate), ")")

function Base.show(io::IO, source::PatternSource)
    nx, ny = size(source.pattern)
    active = count(!iszero, source.pattern)
    print(io, "PatternSource(", nx, "x", ny, " pattern, ", active, " active cells, ",
          describe_rate(source.rate), ")")
    return nothing
end

# ---------------------------------------------------------------------------
# Driving a source or a boundary from sampled data
# ---------------------------------------------------------------------------

"""
    TimeSeries(times, values; extrapolate = :clamp)

A callable that interpolates sampled data linearly, so a recorded or synthetic
series can drive a boundary or a source directly:

```julia
outdoor = TimeSeries(hours, temperatures)
model = Heat2D(nx = 64, boundary = Dirichlet(outdoor), dt = 0.1f0)
```

`times` must be sorted and strictly increasing. Outside the sampled range,
`:clamp` holds the first or last value and `:error` throws — choose `:error`
when running past the end of your data should be a failure rather than a silently
flat extrapolation.

Interpolating is a modelling decision worth stating in a report: hourly weather
data driving a model with `dt = 0.1` means 36 000 simulated steps between two
real measurements, and the smoothness in between is an assumption, not data.
"""
struct TimeSeries{T<:Real,V<:Real}
    times::Vector{T}
    values::Vector{V}
    extrapolate::Symbol

    function TimeSeries(times::AbstractVector{T}, values::AbstractVector{V};
                        extrapolate::Symbol = :clamp) where {T<:Real,V<:Real}
        length(times) == length(values) ||
            throw(ArgumentError("times and values must have equal length, got $(length(times)) and $(length(values))"))
        length(times) >= 2 ||
            throw(ArgumentError("a TimeSeries needs at least two samples, got $(length(times))"))
        issorted(times) && allunique(times) ||
            throw(ArgumentError("times must be sorted and strictly increasing"))
        extrapolate in (:clamp, :error) ||
            throw(ArgumentError("extrapolate must be :clamp or :error, got :$extrapolate"))
        return new{T,V}(collect(times), collect(values), extrapolate)
    end
end

function (series::TimeSeries)(t)
    times, values = series.times, series.values
    if t <= first(times)
        series.extrapolate === :error && t < first(times) &&
            throw(ArgumentError("time $t is before the series start $(first(times))"))
        return float(first(values))
    elseif t >= last(times)
        series.extrapolate === :error && t > last(times) &&
            throw(ArgumentError("time $t is past the series end $(last(times))"))
        return float(last(values))
    end

    # searchsortedlast gives the sample at or before t; t is strictly interior
    # here, so idx is in 1:length-1 and idx+1 is always valid.
    idx = searchsortedlast(times, t)
    t0, t1 = times[idx], times[idx + 1]
    v0, v1 = values[idx], values[idx + 1]
    theta = (t - t0) / (t1 - t0)
    return float(v0 + theta * (v1 - v0))
end

Base.length(series::TimeSeries) = length(series.times)
Base.extrema(series::TimeSeries) = (first(series.times), last(series.times))

function Base.show(io::IO, series::TimeSeries)
    t0, t1 = extrema(series)
    print(io, "TimeSeries(", length(series), " samples, t ∈ [", t0, ", ", t1, "])")
    return nothing
end
