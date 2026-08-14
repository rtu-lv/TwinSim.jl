"""
    MetricRecorder(; name = f, ...)

A `run!` callback that computes named reductions of the state and keeps only
those, discarding the field itself.

```julia
recorder = MetricRecorder(
    mean = m -> sum(state(m)) / length(m.field),
    coldest = m -> minimum(state(m)),
)

run!(sim; callback = recorder, callback_every = 50)

recorder[:mean]      # one value per recorded step
recorder.times       # the simulated time each was taken at
```

This is the in-situ pattern, and the reason for it is arithmetic. A 2048x2048
`Float32` field is 16 MB; retaining it every 50 steps of a 100 000-step run is
32 GB. Retaining two numbers instead is 32 kB. A twin that has decided in
advance what it needs to keep can run for as long as the plant does; one that
stores state and post-processes cannot.

Each metric receives the model, synchronised to host memory before the callback
runs, so `state(m)` works on any backend. It must return a single number.

Metrics are evaluated in the order given. Keep them cheap: they run inside the
timed loop, and `run!` excludes callback time from `compute_seconds` precisely
so that an expensive one is visible as a gap between `compute_seconds` and
`elapsed_seconds` rather than silently inflating the throughput figure.
"""
struct MetricRecorder{F<:NamedTuple}
    metrics::F
    steps::Vector{Int}
    times::Vector{Float64}
    values::Dict{Symbol,Vector{Float64}}
end

function MetricRecorder(; metrics...)
    named = values(metrics)
    isempty(named) && throw(ArgumentError(
        "MetricRecorder needs at least one metric, e.g. MetricRecorder(total = sum_state)"))
    store = Dict{Symbol,Vector{Float64}}(name => Float64[] for name in keys(named))
    return MetricRecorder(named, Int[], Float64[], store)
end

function (recorder::MetricRecorder)(model, progress)
    push!(recorder.steps, progress.step)
    push!(recorder.times, progress.simulated_time)
    for name in keys(recorder.metrics)
        value = recorder.metrics[name](model)
        push!(recorder.values[name], Float64(value))
    end
    return nothing
end

Base.getindex(recorder::MetricRecorder, name::Symbol) = recorder.values[name]
Base.keys(recorder::MetricRecorder) = keys(recorder.metrics)
Base.length(recorder::MetricRecorder) = length(recorder.steps)

"""
    empty!(recorder) -> recorder

Discard everything recorded so far, keeping the metric definitions.
"""
function Base.empty!(recorder::MetricRecorder)
    empty!(recorder.steps)
    empty!(recorder.times)
    foreach(empty!, Base.values(recorder.values))
    return recorder
end

function Base.show(io::IO, recorder::MetricRecorder)
    print(io, "MetricRecorder(", join(keys(recorder.metrics), ", "), "; ",
          length(recorder), " samples)")
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", recorder::MetricRecorder)
    println(io, "MetricRecorder with ", length(recorder), " samples")
    if isempty(recorder)
        print(io, "  (nothing recorded yet)")
        return nothing
    end
    @printf(io, "  %-14s %12s %12s %12s\n", "metric", "first", "last", "range")
    for name in keys(recorder.metrics)
        series = recorder[name]
        @printf(io, "  %-14s %12.4g %12.4g %12.4g\n",
                name, first(series), last(series), maximum(series) - minimum(series))
    end
    print(io, "  simulated time ", first(recorder.times), " .. ", last(recorder.times))
    return nothing
end

Base.isempty(recorder::MetricRecorder) = isempty(recorder.steps)
