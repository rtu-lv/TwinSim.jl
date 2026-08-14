# A model driven by its environment, which is what separates a twin from a
# simulation: the state depends on *when* you run it, not only on how many steps.
#
#   julia --project=. examples/driven_heat.jl
#
# The setting is Case A from the course: a slab losing heat through its edge to
# the outdoor air, with heaters inside it. Two things vary over time — the
# outdoor temperature, which nobody controls, and the heater output, which the
# operator does. That split is the shape of a control-support twin.

using Printf
using VisuTwinSim

const HOURS = 72          # three days
const DT = 0.02f0         # hours per step
const GRID = 96

# ---------------------------------------------------------------------------
# The environment: a daily cycle, sampled hourly, as a forecast would arrive.
# Synthetic, and said so — an unstated synthetic input is a reporting defect.
# ---------------------------------------------------------------------------
sample_times = collect(0.0:1.0:HOURS)
outdoor_samples = [-4.0 + 6.0 * sin(2pi * (t - 9) / 24) for t in sample_times]
outdoor = TimeSeries(sample_times, outdoor_samples)

@printf("Outdoor series: %d hourly samples, %.1f to %.1f degrees\n",
        length(outdoor), minimum(outdoor_samples), maximum(outdoor_samples))
@printf("The model takes %d steps between consecutive samples, so the values in\n",
        round(Int, 1 / DT))
@printf("between are interpolation, not measurement.\n\n")

# ---------------------------------------------------------------------------
# The heaters: fixed geometry, one scalar of control authority.
# ---------------------------------------------------------------------------
function building_layout(n)
    pattern = zeros(Float32, n, n)
    for (cx, cy, power) in ((24, 24, 1.0f0), (72, 30, 0.8f0), (48, 64, 1.4f0), (26, 70, 0.6f0))
        for j in (cy - 4):(cy + 4), i in (cx - 4):(cx + 4)
            checkbounds(Bool, pattern, i, j) || continue
            pattern[i, j] = power
        end
    end
    return pattern
end

layout = building_layout(GRID)

# Weather compensation: heat harder when it is colder outside. This is the
# control law the twin exists to evaluate.
heating_demand(t) = Float32(max(0.0, 0.06 * (16.0 - outdoor(t))))

model = Heat2D(nx = GRID, ny = GRID, initial = 6.0f0,
               alpha = 0.5f0, dt = DT,
               boundary = Dirichlet(outdoor),                # the edge follows the weather
               source = PatternSource(layout, heating_demand))

println(sprint(show, MIME"text/plain"(), model), "\n")

# ---------------------------------------------------------------------------
# Run it, sampling once per simulated hour through a callback. The full field is
# discarded each time; only the summaries survive. That is the in-situ pattern:
# decide up front what the twin has to retain, and keep only that.
# ---------------------------------------------------------------------------
near_edge, centre, mean_temp = Float64[], Float64[], Float64[]

function sample(m, progress)
    u = Array(state(m))
    push!(near_edge, u[3, GRID ÷ 2])
    push!(centre, u[GRID ÷ 2, GRID ÷ 2])
    push!(mean_temp, sum(u) / length(u))
    return nothing
end

metrics = run!(Simulation(model; stop = UntilTime(HOURS));
               callback = sample, callback_every = round(Int, 1 / DT))

@printf("%6s %10s %12s %10s %10s\n", "hour", "outdoor", "near edge", "centre", "mean")
for k in 1:4:length(mean_temp)
    @printf("%6d %10.2f %12.2f %10.2f %10.2f\n",
            k, outdoor(Float64(k)), near_edge[k], centre[k], mean_temp[k])
end

last_day = (length(mean_temp) - 23):length(mean_temp)
swing(v) = maximum(v[last_day]) - minimum(v[last_day])

println()
@printf("Over the final 24 hours:\n")
@printf("  outdoor swings   %.2f degrees\n", maximum(outdoor_samples) - minimum(outdoor_samples))
@printf("  near-edge swings %.2f degrees\n", swing(near_edge))
@printf("  centre swings    %.2f degrees\n", swing(centre))
@printf("  the mean drifts  %.2f degrees per day\n",
        mean_temp[end] - mean_temp[end - 23])
@printf("\nSimulated %.0f hours in %.3f s of compute (%.0fx faster than real time)\n",
        metrics.simulated_time, metrics.compute_seconds,
        metrics.simulated_time * 3600 / metrics.compute_seconds)

println("""

Three timescales are visible at once, and separating them is the point:

  * The **daily cycle** penetrates only a short distance. Near the edge the slab
    follows the weather closely; at the centre the same forcing arrives heavily
    damped and delayed. That distance is the thermal penetration depth, and it
    is set by alpha and the period, not by the grid.
  * The **bulk trend** is far slower. A slab this size has a diffusive time
    constant of order L^2/alpha, which here is thousands of hours — so over
    three days the mean drifts steadily and never reaches equilibrium. That is
    the physics, not a badly tuned control law.
  * The **compute time** is a rounding error against either, which is the
    headroom a real twin spends on assimilation, scenarios and uncertainty.

The consequence for a twin is model spin-up. Because the slab takes months to
forget its initial condition, a twin started from a guess stays wrong for months.
It has to be *initialised from measurements* and then kept on track by
assimilation — which is what `nudge!` and `examples/digital_twin.jl` are for.
Running a slow model from a cold start and trusting its absolute values is one
of the easier ways to produce a confident, useless twin.

Assumptions worth stating in a report, none of them innocent:

  * The outdoor series is synthetic and hourly; the model steps 50 times between
    samples.
  * The control law reacts to the true outdoor temperature instantly. A real
    controller sees a delayed, noisy measurement, and reacts on a schedule.
  * A Dirichlet edge holds the boundary at the outdoor temperature exactly,
    which assumes perfect thermal contact with the air.
  * The source rate is a function of time alone. A controller that responds to
    the slab's own state has to close that loop between `run!` calls, the same
    way measurements are assimilated.
""")
