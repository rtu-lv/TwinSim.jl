# A monitoring twin: a model that notices when the real system stops matching it.
#
#   julia --project=. examples/monitoring_twin.jl
#
# The plant is a heated slab with two heaters. Part way through the run one
# heater fails. The twin does not know that — it keeps assuming both are healthy —
# and the job is to notice from a single temperature sensor.
#
# This is the fourth stage of the reference architecture, the one that turns a
# simulation into something an operator acts on.

using Printf
using Random
using TwinSim

const GRID = 64
const DT = 0.02f0             # hours per step
const STEPS_PER_HOUR = round(Int, 1 / DT)
const HOURS = 140
const FAILURE_HOUR = 70
const PROBE = (20, 20)        # the one cell an instrument actually reads

# ---------------------------------------------------------------------------
# Shared environment and geometry.
# ---------------------------------------------------------------------------
sample_times = collect(0.0:1.0:float(HOURS - 1))
outdoor = SampledSeries(sample_times, [-4.0 + 6.0 * sin(2pi * (t - 9) / 24) for t in sample_times])

function heater_layout(n)
    pattern = zeros(Float32, n, n)
    for (cx, cy) in ((20, 20), (44, 44)), j in (cy - 5):(cy + 5), i in (cx - 5):(cx + 5)
        pattern[i, j] = 1.0f0
    end
    return pattern
end

layout = heater_layout(GRID)

build(power) = Heat2D(nx = GRID, ny = GRID, initial = 5.0f0,
                      alpha = 0.5f0, dt = DT,
                      boundary = Dirichlet(outdoor),
                      source = PatternSource(layout, power))

# ---------------------------------------------------------------------------
# Phase 1: the plant runs, a heater fails, and an instrument records it.
#
# In a deployment this phase is the world, and all you receive is the series.
# ---------------------------------------------------------------------------
plant_power = ControlSignal(1.0f0)
plant = build(plant_power)
noise = Xoshiro(7)

readings = Float64[]
truth = falses(HOURS)

for hour in 1:HOURS
    if hour >= FAILURE_HOUR
        plant_power[] = 0.35f0          # one heater degrades and stays degraded
        truth[hour] = true
    end
    run!(plant; steps = STEPS_PER_HOUR)
    push!(readings, Float64(plant.field[PROBE...]) + 0.15 * randn(noise))
end

@printf("Plant run: %d hourly readings, heater failure injected at hour %d\n",
        length(readings), FAILURE_HOUR)
@printf("Sensor at cell %s, measurement noise sd 0.15\n\n", string(PROBE))

# ---------------------------------------------------------------------------
# Phase 2: the twin sees only the readings.
# ---------------------------------------------------------------------------
function monitor(check::Symbol)
    twin = build(ControlSignal(1.0f0))          # believes both heaters are healthy

    loop = TwinLoop(
        # Nothing to assimilate here: the point is to compare, not to correct.
        # A twin that chases this sensor would absorb the fault it is looking for.
        validate = (model, observation, t) ->
            (; innovation = observation - Float64(model.field[PROBE...])),
        decide = (model, checks, t) -> abs(checks.innovation) > 0.5 ? :alarm : :ok,
        steps_per_window = STEPS_PER_HOUR,
        check = check,
    )

    log = twin_run!(loop, twin, readings; times = 0:(HOURS - 1))
    return log, [c.innovation for c in log.checks]
end

log, innovation = monitor(:forecast)

healthy = innovation[1:(FAILURE_HOUR - 1)]
faulty = innovation[FAILURE_HOUR:end]
healthy_mean = sum(healthy) / length(healthy)
healthy_sd = sqrt(sum(abs2, healthy .- healthy_mean) / (length(healthy) - 1))

@printf("Innovation while healthy: mean %+.3f, sd %.3f\n", healthy_mean, healthy_sd)
@printf("Innovation after failure: %.2f to %.2f\n\n", minimum(faulty), maximum(faulty))

# ---------------------------------------------------------------------------
# The threshold is a decision with a cost on each side.
# ---------------------------------------------------------------------------
println("Threshold sweep:\n")
@printf("%10s %8s %8s %8s %10s %10s\n",
        "threshold", "sigmas", "delay", "false+", "FP rate", "recall")
for threshold in (0.25, 0.5, 1.0, 2.0, 5.0, 15.0)
    report = detection_report(innovation, truth, threshold)
    @printf("%10.2f %8.1f %8s %8d %10.3f %10.2f\n",
            threshold, threshold / healthy_sd,
            report.delay === missing ? "never" : string(report.delay),
            report.false_positives, report.false_positive_rate, report.recall)
end

println("""

There is no best row, and that is the point. A low threshold detects the failure
within the hour and raises false alarms during normal operation; a high one never
cries wolf and lets the building cool for hours first. Choosing between them
needs a number this program does not have: what an unnecessary callout costs the
operator, against what a cold building costs the tenants.

State that trade-off in stakeholder terms in the report. "I picked 1.0 because it
had the best F1 score" is not an answer to it.
""")

# ---------------------------------------------------------------------------
# What happens when the twin also tracks the plant.
#
# The twin above only compares. A twin that also *corrects* itself towards the
# sensor is the usual arrangement — and it quietly destroys its own ability to
# notice anything.
# ---------------------------------------------------------------------------
function monitored_with_assimilation(gain, check)
    twin = build(ControlSignal(1.0f0))
    loop = TwinLoop(
        assimilate = gain == 0 ? (m, o, t) -> nothing :
                     (m, o, t) -> nudge!(m, [Sensor(PROBE..., Float32(o))];
                                         gain = gain, radius = 4),
        validate = (model, observation, t) ->
            (; innovation = observation - Float64(model.field[PROBE...])),
        decide = (model, checks, t) -> abs(checks.innovation) > 0.5 ? :alarm : :ok,
        steps_per_window = STEPS_PER_HOUR,
        check = check,
    )
    corrected = twin_run!(loop, twin, readings; times = 0:(HOURS - 1))
    return [c.innovation for c in corrected.checks]
end

println("Peak innovation after the failure, against assimilation strength:\n")
@printf("%18s %14s %14s\n", "assimilation gain", ":forecast", ":analysis")
for gain in (0.0, 0.15, 0.6)
    peaks = map((:forecast, :analysis)) do check
        maximum(abs, monitored_with_assimilation(gain, check)[FAILURE_HOUR:end])
    end
    @printf("%18.2f %14.1f %14.1f\n", gain, peaks[1], peaks[2])
end

println("""

Read the columns first: choosing `:forecast` over `:analysis` is worth a little,
because the two differ by exactly one window's correction. Now read the rows,
which move by an order of magnitude.

A twin that assimilates hard tracks the failing plant and stops being surprised
by it. Its residuals stay small precisely because it is doing its tracking job
well — and residual size is exactly what the monitoring rule keys on. Tuning the
twin to fit better makes it blinder.

This is the trap in combining the two roles, and it is not fixed by reordering
the stages. If a twin must both track and monitor, the monitor should watch **how
hard the assimilation is having to pull** — the size of the correction being
applied — rather than how small the residual ends up afterwards.

Implementing that alternative detector, and comparing it against this one on the
same data, is a good extension for Lab Work 4.
""")

@printf("\n%d windows, %.4f s of compute, %.0fx faster than the plant it watches\n",
        length(log), compute_seconds(log), realtime_ratio(log, 3600.0))
