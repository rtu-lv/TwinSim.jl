# The runtime that makes a simulation a twin.
#
# Every twin runtime has the same four stages. This file provides the loop that
# drives them and records what happened; the stages themselves are the work, and
# all four default to doing nothing so they can be filled in one at a time.

"""
    TwinLoop(; assimilate, validate, decide, steps_per_window = 1)

The four stages of a twin runtime, in the order they must run:

1. **assimilate** `(model, observation, t)` — bring the simulated state towards
   what was measured. [`nudge!`](@ref) is the supplied implementation.
2. **advance** — the loop runs the model forward `steps_per_window` steps. This
   stage is not configurable; it is `run!`.
3. **validate** `(model, observation, t)` — decide whether the state can be
   trusted. Returns anything; a `NamedTuple` is usual.
4. **decide** `(model, checks, t)` — produce the output somebody acts on.

The check stage sits between simulating and deciding rather than at the end, and
that ordering is the design. A twin that emits a decision without first
establishing whether its own state is trustworthy is worse than one that emits
nothing, because it is confidently wrong and looks the same as being right.

Every stage defaults to a no-op, so a loop with the skeleton in place runs from
the start and each stage can be filled in separately.

```julia
loop = TwinLoop(
    assimilate = (m, obs, t) -> nudge!(m, [Sensor(32, 32, obs)]; gain = 0.5, radius = 4),
    validate   = (m, obs, t) -> (; residual = m.field[32, 32] - obs),
    decide     = (m, checks, t) -> abs(checks.residual) > 2.0 ? :alarm : :ok,
    steps_per_window = 50,
)

log = twin_run!(loop, model, observed)
```

## Where the check sits, and why it is a keyword rather than a fixed order

`check` selects what the `validate` stage sees, using the standard
data-assimilation vocabulary:

- `:forecast` (default) — **advance, validate, assimilate, decide**. `validate`
  runs on the state *before* the correction, so `observation - prediction` is the
  **innovation**: what the model did not expect.
- `:analysis` — **assimilate, advance, validate, decide**. `validate` runs on the
  corrected state, so the residual says how well the correction fitted.

`:forecast` is the default because the innovation is the quantity a monitoring
twin actually wants, and because it is what a filter means by a residual.
`:analysis` is the right choice when the question is "is my correction fitting?"
rather than "has something changed?".

How much the choice matters depends on how hard you assimilate, and the two
orderings differ by only one window's correction — so do not expect it to rescue
a badly-posed detector. In `examples/monitoring_twin.jl`, at an assimilation gain
of 0.6 the peak innovation after a heater failure is 2.6 under `:forecast` and
2.1 under `:analysis`: a real difference, and a small one.

The effect that dominates both is **assimilation strength itself**. In the same
example, the peak innovation after the failure is:

| assimilation gain | peak innovation |
|------------------:|----------------:|
| 0.0 (no correction) | 23.2 |
| 0.15 | 8.3 |
| 0.6 | 2.6 |

An assimilating twin chases the failing plant and stops being surprised by it,
whichever order the stages run in. That is the trap worth understanding: a twin
tuned to track well is, by construction, a twin that reports small residuals —
and residual size is exactly what a monitoring rule keys on. If a twin must both
track and monitor, the monitor should watch how hard the assimilation is having
to pull, not how small the residual ends up.
"""
Base.@kwdef struct TwinLoop{A,V,D}
    assimilate::A = (model, observation, t) -> nothing
    validate::V = (model, observation, t) -> nothing
    decide::D = (model, checks, t) -> nothing
    steps_per_window::Int = 1
    check::Symbol = :forecast

    function TwinLoop(assimilate::A, validate::V, decide::D,
                      steps_per_window::Integer, check::Symbol) where {A,V,D}
        steps_per_window > 0 || throw(ArgumentError(
            "steps_per_window must be positive, got $steps_per_window"))
        check in (:forecast, :analysis) || throw(ArgumentError(
            "check must be :forecast or :analysis, got :$check"))
        return new{A,V,D}(assimilate, validate, decide, Int(steps_per_window), check)
    end
end

"""
    TwinLog

What a [`twin_run!`](@ref) produced, one entry per observation window.

- `times`, `observations` — the input stream as it was consumed;
- `checks` — whatever `validate` returned;
- `decisions` — whatever `decide` returned;
- `metrics` — the [`RunMetrics`](@ref) of each window's `run!`.

`decisions` is the series a stakeholder would act on; `checks` is the evidence
behind it. Keeping them separate is what lets a report say *why* an alarm was
raised rather than only that it was.
"""
struct TwinLog{O,C,D}
    times::Vector{Float64}
    observations::Vector{O}
    checks::Vector{C}
    decisions::Vector{D}
    metrics::Vector{RunMetrics}
end

Base.length(log::TwinLog) = length(log.times)

"""
    compute_seconds(log) -> Float64

Total time the twin spent simulating, across every window.
"""
compute_seconds(log::TwinLog) = sum(m -> m.compute_seconds, log.metrics; init = 0.0)

"""
    simulated_time(log) -> Float64

Total simulated time advanced, across every window.
"""
simulated_time(log::TwinLog) = sum(m -> m.simulated_time, log.metrics; init = 0.0)

"""
    realtime_ratio(log, seconds_per_time_unit = 1.0) -> Float64

How many times faster than the real system the twin ran. The headroom available
for assimilation, scenarios and uncertainty — and the number that decides whether
a twin can be predictive at all.
"""
function realtime_ratio(log::TwinLog, seconds_per_time_unit::Real = 1.0)
    spent = compute_seconds(log)
    spent > 0 || return Inf
    return simulated_time(log) * seconds_per_time_unit / spent
end

function Base.show(io::IO, ::MIME"text/plain", log::TwinLog)
    println(io, "TwinLog with ", length(log), " windows")
    isempty(log.times) && return nothing
    @printf(io, "  simulated time    %.4g\n", simulated_time(log))
    @printf(io, "  compute time      %.4f s\n", compute_seconds(log))
    @printf(io, "  faster than real  %.0fx\n", realtime_ratio(log))
    counts = Dict{Any,Int}()
    for decision in log.decisions
        counts[decision] = get(counts, decision, 0) + 1
    end
    print(io, "  decisions         ",
          join(("$k x$v" for (k, v) in sort(collect(counts); by = last, rev = true)), ", "))
    return nothing
end

"""
    twin_run!(loop, model, observations; backend, times, callback) -> TwinLog

Drive `model` through `observations`, running the four stages of `loop` once per
observation.

`observations` may be an [`ObservationSeries`](@ref) — in which case its sample
times are used — or any vector, with `times` supplied separately or defaulting to
`0, 1, 2, ...` scaled by the model's step.

Each window advances `loop.steps_per_window` steps, so the observation cadence
and the model's time step are decoupled: a twin fed hourly measurements while
stepping every 0.02 hours runs 50 steps per observation.

The state stays on the backend across windows, so a GPU twin is not billed for a
transfer per observation. Assimilation needs host memory, so a window whose
`assimilate` stage does anything will sync — which is a real cost and shows up in
`transfer_seconds`.
"""
function twin_run!(loop::TwinLoop, model, observations;
                   backend::AbstractBackend = CPUBackend(),
                   times = nothing,
                   callback = nothing)
    values, sample_times = observation_stream(observations, times, model, loop.steps_per_window)
    length(values) == length(sample_times) || throw(DimensionMismatch(
        "got $(length(values)) observations and $(length(sample_times)) times"))
    check_cadence(model, sample_times, loop.steps_per_window)

    log_times = Float64[]
    log_observations = Any[]
    log_checks = Any[]
    log_decisions = Any[]
    log_metrics = RunMetrics[]

    forecast_check = loop.check === :forecast

    for (k, observation) in enumerate(values)
        t = sample_times[k]

        if forecast_check
            # advance, then check the forecast against the observation. The
            # residual is the innovation: what the model failed to predict.
            metrics = run!(model; backend = backend, steps = loop.steps_per_window)
            checks = loop.validate(model, observation, t)
            loop.assimilate(model, observation, t)
        else
            # assimilate first, then check the corrected state. The residual says
            # how well the correction fitted, not whether anything is wrong.
            loop.assimilate(model, observation, t)
            metrics = run!(model; backend = backend, steps = loop.steps_per_window)
            checks = loop.validate(model, observation, t)
        end

        # Deciding always comes last, after the state has been checked. A twin
        # that emits a decision without knowing whether its own state is
        # trustworthy is worse than one that emits nothing.
        decision = loop.decide(model, checks, t)

        push!(log_times, t)
        push!(log_observations, observation)
        push!(log_checks, checks)
        push!(log_decisions, decision)
        push!(log_metrics, metrics)

        callback === nothing || callback(model, (; step = k, time = t, checks, decision))
    end

    return TwinLog(log_times,
                   narrow(log_observations),
                   narrow(log_checks),
                   narrow(log_decisions),
                   log_metrics)
end

# `Any[]` while collecting, narrowed afterwards: the stage return types are not
# known until they have run, and a log of concrete type is far nicer to work with.
# `identity.` is the idiom for this — broadcasting recomputes the element type.
narrow(v::Vector) = isempty(v) ? v : identity.(v)

"""
    check_cadence(model, sample_times, steps_per_window)

Warn when the observation cadence and the simulated window length disagree.

There are two clocks in a twin run and they are easy to desynchronise. The stages
receive the *observation* time, but a time-varying boundary or source is evaluated
at the *model's* simulated time, which advances by `steps_per_window * dt` per
window. If observations arrive hourly while each window advances half an hour,
the drive is being sampled at half the rate the observations imply, and every
comparison between prediction and measurement is made at the wrong instant.

The symptom is a twin that appears to work but has a systematically large
innovation — easily mistaken for a modelling error, and expensive to find.
"""
function check_cadence(model, sample_times, steps_per_window)
    length(sample_times) >= 2 || return nothing

    window = Float64(timestep(model)) * steps_per_window
    spacing = sample_times[2] - sample_times[1]
    spacing > 0 || return nothing

    if !isapprox(spacing, window; rtol = 1e-3)
        @warn """
              Observation cadence and simulated window length disagree.

              Observations are $(spacing) apart, but each window advances
              $(steps_per_window) steps of dt=$(timestep(model)) = $(window) of simulated time.

              A time-varying boundary or source is evaluated at the model's clock, so it
              will drift out of step with the observations. Set steps_per_window to
              $(round(Int, spacing / Float64(timestep(model)))) to match, or pass `times`
              that follow the model's clock.
              """ maxlog = 1
    end
    return nothing
end

observation_stream(series::ObservationSeries, ::Nothing, model, per_window) =
    (series.values, series.times)
observation_stream(series::ObservationSeries, times, model, per_window) =
    (series.values, collect(Float64, times))

function observation_stream(values::AbstractVector, ::Nothing, model, per_window)
    # Default cadence: one observation per window, so the k-th observation is at
    # the simulated time the k-th window starts.
    window = Float64(timestep(model)) * per_window
    return (values, [window * (k - 1) for k in 1:length(values)])
end

observation_stream(values::AbstractVector, times, model, per_window) =
    (values, collect(Float64, times))
