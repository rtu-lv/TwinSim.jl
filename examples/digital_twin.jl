# A minimal digital twin: a simulation corrected by measurements, paced against
# the wall clock, and able to survive being restarted.
#
#   julia --project=. examples/digital_twin.jl
#
# The "plant" here is another simulation, which is the usual way to develop a
# twin before connecting it to real instrumentation. The twin does not know the
# plant has heaters in it; all it ever sees is a handful of point measurements.

using Printf
using VisuTwinSim

const GRID = 64
const WINDOW = 50        # steps between measurements
const WINDOWS = 40

"""Add a heater with spatial extent. Real sources are never single cells, and
the difference matters: a field made of delta spikes cannot be reconstructed
from sparse samples at all, however good the assimilation scheme is."""
function add_source!(model, x0, y0, amplitude, sigma)
    u = state(model)
    nx, ny = size(u)
    for j in 1:ny, i in 1:nx
        u[i, j] += Float32(amplitude * exp(-((i - x0)^2 + (j - y0)^2) / (2 * sigma^2)))
    end
    return model
end

heat_plant!(plant) = (add_source!(plant, 20, 44, 3.0, 6.0); add_source!(plant, 45, 20, 2.0, 5.0))

sensor_grid(stride) = [(i, j) for i in stride:stride:GRID, j in stride:stride:GRID]

"""Run plant and twin side by side, assimilating every WINDOW steps."""
function track(; stride, radius, gain = 0.7)
    plant = Heat2D(nx = GRID, ny = GRID)
    open_loop = Heat2D(nx = GRID, ny = GRID)
    twin = Heat2D(nx = GRID, ny = GRID)
    probes = sensor_grid(stride)

    rms(model) = sqrt(sum(abs2, state(model) .- state(plant)) / length(plant.field))

    history = NTuple{3,Float64}[]
    for window in 1:WINDOWS
        heat_plant!(plant)
        run!(plant; steps = WINDOW)
        run!(open_loop; steps = WINDOW)
        run!(twin; steps = WINDOW)

        measurements = [Sensor(i, j, plant.field[i, j]) for (i, j) in probes]
        nudge!(twin, measurements; gain = gain, radius = radius)

        push!(history, (window * WINDOW, rms(open_loop), rms(twin)))
    end
    return (; probes = length(probes), history, twin, plant)
end

# ---------------------------------------------------------------------------
# How well does the twin track, and what actually controls that?
# ---------------------------------------------------------------------------
println("Grid $(GRID)x$(GRID) = $(GRID^2) cells, assimilating every $WINDOW steps.\n")
@printf("%9s %9s %14s %14s %10s\n", "sensors", "radius", "open-loop RMS", "twin RMS", "improvement")

for stride in (16, 8, 4)
    for radius in (0, stride / 2, stride)
        result = track(; stride, radius)
        _, open_rms, twin_rms = last(result.history)
        @printf("%9d %9.1f %14.3f %14.3f %9.1fx\n",
                result.probes, radius, open_rms, twin_rms, open_rms / twin_rms)
    end
end

println("""

The localisation radius matters more than the sensor count. With `radius = 0`
each measurement corrects exactly one cell out of $(GRID^2), so even a dense
sensor grid barely helps — diffusion cannot spread those point corrections
faster than the error grows. Giving each measurement an influence region of
about half the sensor spacing is what turns the same data into a tracking twin.

That is the core idea behind optimal interpolation and the localisation step of
an ensemble Kalman filter: a measurement is evidence about a *region*, and how
large that region is comes from the physics, not from the instrument.
""")

# ---------------------------------------------------------------------------
# Convergence of the best configuration over time.
# ---------------------------------------------------------------------------
best = track(; stride = 8, radius = 4)
println("Tracking with $(best.probes) sensors and radius 4:\n")
@printf("%8s %14s %14s\n", "step", "open-loop RMS", "twin RMS")
for (step, open_rms, twin_rms) in best.history[1:5:end]
    @printf("%8d %14.3f %14.3f\n", step, open_rms, twin_rms)
end

# ---------------------------------------------------------------------------
# Running against the wall clock, and checkpointing along the way.
# ---------------------------------------------------------------------------
checkpoint_dir = mktempdir()
twin = best.twin

println("\nRunning the twin in real time (1 unit of simulated time per second):")
metrics = run!(Simulation(twin; stop = UntilTime(twin.params.dt * 30));
               realtime_factor = 1.0,
               callback = checkpoint_callback(joinpath(checkpoint_dir, "twin_%05d.vts")),
               callback_every = 10)

@printf("  %d steps, simulated time %.2f, wall clock %.2f s\n",
        metrics.steps, metrics.simulated_time, metrics.elapsed_seconds)
@printf("  of which %.4f s was compute — the rest was the twin waiting for the world\n",
        metrics.compute_seconds)

written = sort(filter(endswith(".vts"), readdir(checkpoint_dir)))
println("  checkpoints: ", join(written, ", "))

restored = load_state(joinpath(checkpoint_dir, last(written)))
resumed = Heat2D(restored.field; boundary = restored.boundary, alpha = 0.15f0, dt = 0.1f0)
@printf("  restarted from step %d, total heat %.3f (twin had %.3f)\n",
        restored.step, sum_state(resumed), sum_state(twin))
