# Synthetic observation data with known faults in it.
#
# A monitoring twin can only be evaluated against data whose truth you know, and
# nobody has a live district heating feed. Generating the data here — rather than
# leaving every student to write their own — means the detection results in
# different reports are comparable, and that the ground truth is recorded rather
# than remembered.

"""
    Anomaly

A fault injected into a synthetic series. The four kinds behave very differently
under a threshold test, which is the point of having more than one:

| Type | Shape | What catches it |
|:-----|:------|:----------------|
| [`Spike`](@ref) | one sample far out | any magnitude test, but easy to confuse with noise |
| [`LevelShift`](@ref) | step, persists | magnitude test, once past the threshold |
| [`Drift`](@ref) | slow ramp | magnitude test *eventually* — the delay is the problem |
| [`Stuck`](@ref) | sensor freezes | depends entirely on what you compare against |

`Stuck` is worth assigning deliberately, because whether it is detectable is not
a property of the fault alone:

- Against a **range or magnitude check on the values**, it is invisible. A frozen
  sensor reports perfectly plausible numbers.
- Against a **model prediction**, it shows up as soon as the model expects the
  signal to have moved and it has not — so how fast it is caught depends on how
  fast the true signal was changing, not on how large the fault is.
- Against a **variance check over a window**, it is obvious, and that detector
  will not see any of the other three.

A student who reports "my detector caught 3 of 4 anomalies" without working out
which one it missed, and what a detector would have to look at to catch it, has
missed the lesson.
"""
abstract type Anomaly end

"""
    Spike(at, magnitude)

A single sample displaced by `magnitude`, at index `at`.
"""
struct Spike{T} <: Anomaly
    at::Int
    magnitude::T
end

"""
    LevelShift(from, magnitude)

A step of `magnitude` applied from index `from` to the end of the series: a
sensor that loses calibration and never recovers.
"""
struct LevelShift{T} <: Anomaly
    from::Int
    magnitude::T
end

"""
    Drift(from, rate)

A ramp growing by `rate` per sample from index `from`: a sensor degrading
slowly. The hardest of the four to threshold, because the right threshold
depends on how long you are willing to wait.
"""
struct Drift{T} <: Anomaly
    from::Int
    rate::T
end

"""
    Stuck(from, count)

The value freezes at whatever it was at `from`, for `count` samples.

Deliberately invisible to a residual-magnitude test: the values stay in range and
look ordinary. Detecting it requires noticing an *absence* of variation.
"""
struct Stuck <: Anomaly
    from::Int
    count::Int
end

anomaly_range(a::Spike, n) = a.at:a.at
anomaly_range(a::LevelShift, n) = a.from:n
anomaly_range(a::Drift, n) = a.from:n
anomaly_range(a::Stuck, n) = a.from:min(a.from + a.count - 1, n)

function apply_anomaly!(values, a::Spike)
    values[a.at] += a.magnitude
    return values
end

function apply_anomaly!(values, a::LevelShift)
    for k in a.from:length(values)
        values[k] += a.magnitude
    end
    return values
end

function apply_anomaly!(values, a::Drift)
    for k in a.from:length(values)
        values[k] += a.rate * (k - a.from + 1)
    end
    return values
end

function apply_anomaly!(values, a::Stuck)
    frozen = values[a.from]
    for k in a.from:min(a.from + a.count - 1, length(values))
        values[k] = frozen
    end
    return values
end

"""
    ObservationSeries

A synthetic measurement series together with the truth about it. Callable, so it
drives a boundary or a source directly:

```julia
observed = synthetic_series(anomalies = [LevelShift(70, 6.0)], seed = 1)
model = Heat2D(nx = 64, boundary = Dirichlet(observed))
```

Fields:

- `times`, `values` — the series as an instrument would report it;
- `clean` — the same series without the faults, which a real deployment never has;
- `anomalous` — one flag per sample, the ground truth for [`detection_report`](@ref);
- `anomalies` — the injected faults;
- `seed` — everything needed to regenerate it.
"""
struct ObservationSeries{T<:Real}
    times::Vector{Float64}
    values::Vector{T}
    clean::Vector{T}
    anomalous::Vector{Bool}
    anomalies::Vector{Anomaly}
    seed::Int
    interpolant::SampledSeries{Float64,T}
end

(series::ObservationSeries)(t) = series.interpolant(t)

Base.length(series::ObservationSeries) = length(series.times)
Base.extrema(series::ObservationSeries) = (first(series.times), last(series.times))

function Base.show(io::IO, series::ObservationSeries)
    print(io, "ObservationSeries(", length(series), " samples, ",
          count(series.anomalous), " anomalous, seed ", series.seed, ")")
    return nothing
end

"""
    synthetic_series(; samples = 120, step = 1.0, baseline = -4.0, amplitude = 6.0,
                       period = 24.0, phase = 9.0, noise = 0.4,
                       anomalies = Anomaly[], seed = 1)

Build an [`ObservationSeries`](@ref): a daily cycle, plus Gaussian noise, plus
whatever faults you inject.

```julia
observed = synthetic_series(samples = 120,
                            anomalies = [LevelShift(70, 6.0)],
                            seed = 42)
```

Reproducible from `seed` alone, and the ground truth travels with the data
rather than being written down separately.

The defaults describe an outdoor temperature in a Latvian winter: a mean around
-4 degrees, a 6-degree daily swing peaking mid-afternoon, and instrument noise.
State in the report that the series is synthetic — the generator records the
seed so that "synthetic, seed 42, one level shift at sample 70" is a complete
and checkable description.
"""
function synthetic_series(; samples::Integer = 120,
                          step::Real = 1.0,
                          baseline::Real = -4.0,
                          amplitude::Real = 6.0,
                          period::Real = 24.0,
                          phase::Real = 9.0,
                          noise::Real = 0.4,
                          anomalies::AbstractVector = Anomaly[],
                          seed::Integer = 1)
    samples >= 2 || throw(ArgumentError("need at least two samples, got $samples"))
    step > 0 || throw(ArgumentError("step must be positive, got $step"))
    noise >= 0 || throw(ArgumentError("noise must be non-negative, got $noise"))
    period > 0 || throw(ArgumentError("period must be positive, got $period"))

    for a in anomalies
        first(anomaly_range(a, samples)) in 1:samples || throw(ArgumentError(
            "anomaly $a starts outside the series (1:$samples)"))
    end

    times = collect(0.0:float(step):(step * (samples - 1)))
    rng = Random.Xoshiro(hash((seed, :synthetic_series)))

    clean = [baseline + amplitude * sin(2pi * (t - phase) / period) for t in times]
    values = clean .+ noise .* randn(rng, samples)

    flagged = falses(samples)
    for a in anomalies
        apply_anomaly!(values, a)
        flagged[anomaly_range(a, samples)] .= true
    end

    interpolant = SampledSeries(times, values)
    return ObservationSeries(times, values, clean, collect(flagged),
                             collect(Anomaly, anomalies), Int(seed), interpolant)
end
