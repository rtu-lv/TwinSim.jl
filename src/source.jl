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

"""
    ProportionalSource(target, gain)

Forcing proportional to how far the cell is from a target:

```
q = gain * (target - u[i, j])
```

Two readings of the same term, and both matter in this course:

- **Physics**: Newton's law of cooling. The cell exchanges heat with a reservoir
  at `target`, at a rate set by the coupling `gain`.
- **Control**: a per-cell proportional controller (a thermostat) driving the
  state towards a setpoint.

`target` may be a number or a callable of time, so a setpoint schedule works.
**`gain` must be a plain number.** It is a tuning constant, not a signal, and
keeping it constant is what makes the stability limit decidable at construction —
see [`stability_number`](@ref).

Unlike the other sources this one **depends on the state**, which changes the
stability limit of the explicit scheme. That is why `Heat2D` checks
`stability_number` rather than `cfl_number`.
"""
struct ProportionalSource{T,G<:Number} <: SourceTerm
    target::T
    gain::G

    function ProportionalSource(target::T, gain::G) where {T,G<:Number}
        gain >= 0 || throw(ArgumentError(
            "gain must be non-negative, got $gain. A negative gain is positive " *
            "feedback: it drives cells away from the target and diverges for any dt."))
        return new{T,G}(target, gain)
    end
end

"""
    CombinedSource(sources...)
    source_a + source_b

Several forcings acting at once — heaters *and* ambient loss, which is what a
real installation has.

The rates are summed and applied once, rather than each source updating the
value in turn. For purely additive sources the two are the same; as soon as one
of them depends on the state (as [`ProportionalSource`](@ref) does) they are not,
and applying them in sequence would make the result depend on the order.
"""
struct CombinedSource{S<:Tuple} <: SourceTerm
    sources::S
end

CombinedSource(sources::SourceTerm...) = CombinedSource(sources)

Base.:+(a::SourceTerm, b::SourceTerm) = CombinedSource(a, b)
Base.:+(a::CombinedSource, b::SourceTerm) = CombinedSource((a.sources..., b))
Base.:+(a::SourceTerm, b::CombinedSource) = CombinedSource((a, b.sources...))
Base.:+(a::CombinedSource, b::CombinedSource) = CombinedSource((a.sources..., b.sources...))
Base.:+(a::NoSource, b::SourceTerm) = b
Base.:+(a::SourceTerm, b::NoSource) = a
Base.:+(a::NoSource, b::NoSource) = a

# ---------------------------------------------------------------------------
# Applying a source inside the stencil
# ---------------------------------------------------------------------------
#
# `source_rate` returns q; `apply_source` is what the stencil calls.
#
# The split exists so that sources can be *summed* before being applied, which a
# state-dependent term requires. `apply_source` keeps a dedicated NoSource method
# that is the literal identity: going through `value + dt * zero(T)` would emit a
# real addition per cell, because floating-point `x + 0.0` is not `x` when
# `x === -0.0` and the compiler is not free to fold it.

@inline source_rate(::NoSource, value, i, j) = zero(value)
@inline source_rate(source::UniformSource, value, i, j) = source.rate

@inline function source_rate(source::PatternSource, value, i, j)
    @inbounds return source.rate * source.pattern[i, j]
end

@inline source_rate(source::ProportionalSource, value, i, j) =
    source.gain * (source.target - value)

@inline source_rate(source::CombinedSource, value, i, j) =
    sum_rates(source.sources, value, i, j)

# Peeled recursively rather than with `for inner in sources`. Iterating a tuple
# compiles to a dynamic `getindex`, which a GPU kernel cannot do — it shows up as
# "unsupported call to an unknown function (call to ijl_get_nth_field_checked)".
# This form is fully unrolled at compile time and each source keeps its own
# specialisation.
@inline sum_rates(::Tuple{}, value, i, j) = zero(value)
@inline sum_rates(sources::Tuple, value, i, j) =
    source_rate(first(sources), value, i, j) + sum_rates(Base.tail(sources), value, i, j)

@inline apply_source(::NoSource, value, i, j, dt) = value
@inline apply_source(source::SourceTerm, value, i, j, dt) =
    value + dt * source_rate(source, value, i, j)

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

@inline resolve(source::ProportionalSource{<:Number}, t, ::Type{T}) where {T} =
    ProportionalSource(convert(T, source.target), convert(T, source.gain))

@inline resolve(source::ProportionalSource, t, ::Type{T}) where {T} =
    ProportionalSource(convert(T, source.target(t)), convert(T, source.gain))

@inline resolve(source::CombinedSource, t, ::Type{T}) where {T} =
    CombinedSource(map(inner -> resolve(inner, t, T), source.sources))

"""
    is_driven(source) -> Bool

Whether the source's rate depends on simulated time.
"""
is_driven(::SourceTerm) = false
is_driven(::UniformSource) = true
is_driven(::UniformSource{<:Number}) = false
is_driven(::PatternSource) = true
is_driven(::PatternSource{<:Number}) = false
is_driven(::ProportionalSource) = true
is_driven(::ProportionalSource{<:Number}) = false
is_driven(source::CombinedSource) = any(is_driven, source.sources)

"""
    feedback_coefficient(source) -> Real

The largest `|d q / d u|` the source contributes: how strongly the forcing
responds to the state it is forcing.

Zero for every source that does not read the state. [`ProportionalSource`](@ref)
contributes its gain, and that term tightens the stability limit of the explicit
scheme — see [`stability_number`](@ref).
"""
feedback_coefficient(::SourceTerm) = 0
feedback_coefficient(source::ProportionalSource) = source.gain
feedback_coefficient(source::CombinedSource) =
    sum(feedback_coefficient, source.sources; init = 0)

"""
    move_source_to_device(source, device) -> (source, bytes)

Copy any grid-sized arrays the source carries into `device` memory, returning the
relocated source and the number of bytes transferred. Recurses through
[`CombinedSource`](@ref), which may hold more than one pattern.
"""
move_source_to_device(source::SourceTerm, device) = (source, 0)

function move_source_to_device(source::PatternSource, device)
    pattern = source.pattern
    device_pattern = KernelAbstractions.allocate(device, eltype(pattern), size(pattern))
    copyto!(device_pattern, pattern)
    return (PatternSource(device_pattern, source.rate),
            sizeof(eltype(pattern)) * length(pattern))
end

function move_source_to_device(source::CombinedSource, device)
    moved = map(inner -> move_source_to_device(inner, device), source.sources)
    return (CombinedSource(map(first, moved)), sum(last, moved; init = 0))
end

"""
    check_source_shape(source, dims)

Verify that every pattern the source carries matches the field.
"""
check_source_shape(::SourceTerm, dims) = nothing

function check_source_shape(source::PatternSource, dims)
    size(source.pattern) == dims || throw(DimensionMismatch(
        "source pattern is $(size(source.pattern)) but the field is $dims"))
    return nothing
end

function check_source_shape(source::CombinedSource, dims)
    foreach(inner -> check_source_shape(inner, dims), source.sources)
    return nothing
end

"""
    ControlSignal(value)

A mutable scalar that can be used anywhere a drive is accepted, and written to
from a `run!` callback:

```julia
power = ControlSignal(0.0f0)
model = Heat2D(nx = 96, source = PatternSource(layout, power))

run!(sim; callback_every = 10, callback = function (m, progress)
    mean_temp = sum(Array(state(m))) / length(m.field)
    power[] = clamp(0.5f0 * (20 - mean_temp), 0, 5)      # close the loop
    return nothing
end)
```

This is how a controller that reacts to the *state* is written, as opposed to
[`ProportionalSource`](@ref), which reacts per cell inside the kernel. The
difference is real and worth stating in a report:

- `ProportionalSource` acts on every cell independently, every step, using that
  cell's own value. It is a distributed thermostat, and it changes the stability
  limit.
- `ControlSignal` acts on one scalar shared by the whole domain, updated as often
  as the callback runs, using whatever reduction of the state you choose. It is a
  central controller with a sampling rate, and it does not affect stability
  because the kernel still sees a constant.

Because a `Heat2D` is immutable, this is also the supported way to change a
forcing mid-run without rebuilding the model.
"""
mutable struct ControlSignal{T}
    value::T
end

(signal::ControlSignal)(t) = signal.value

Base.getindex(signal::ControlSignal) = signal.value
Base.setindex!(signal::ControlSignal, value) =
    (signal.value = convert(typeof(signal.value), value))
Base.show(io::IO, signal::ControlSignal) = print(io, "ControlSignal(", signal.value, ")")

# A `PatternSource` holding a device array still cannot be passed to a kernel as
# it stands: `MtlMatrix`/`CuArray` are *host-side* wrappers around a buffer
# reference, and a kernel argument has to be isbits. Adapt is the mechanism the
# GPU packages use to rewrite such wrappers into plain device pointers at launch
# time, and this line is what extends it to our struct.
#
# Without it the kernel fails to compile with "passing non-bitstype argument",
# which is the standard symptom of a custom struct carrying an array onto a GPU.
Adapt.@adapt_structure PatternSource

# And again for the wrapper: a CombinedSource holding a PatternSource is no more
# isbits than the PatternSource was. Adapt already recurses through tuples, so
# this one line covers a combination of any depth.
Adapt.@adapt_structure CombinedSource

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

Base.show(io::IO, source::ProportionalSource) =
    print(io, "ProportionalSource(target ", describe_rate(source.target),
          ", gain ", source.gain, ")")

function Base.show(io::IO, source::CombinedSource)
    print(io, "CombinedSource(")
    for (k, inner) in enumerate(source.sources)
        k == 1 || print(io, " + ")
        show(io, inner)
    end
    print(io, ")")
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
