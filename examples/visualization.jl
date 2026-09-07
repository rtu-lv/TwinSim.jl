# Getting results out of a run, in the two forms a report needs.
#
#   julia --project=. -e 'using Pkg; Pkg.add("CairoMakie")'
#   julia --project=. examples/visualization.jl
#
# Runs without CairoMakie too, and says what it skipped. That is the point of the
# split: the machine that runs the big cases is often not the one with a plotting
# stack, and CSV export needs nothing.

using Printf
using TwinSim

# CairoMakie renders headless, which is what a cluster node or a CI job needs.
# GLMakie opens a window instead, for exploring interactively.
#
# The first run after the package changes spends a minute or two precompiling the
# Makie extension before printing anything. That is Makie, not the simulation —
# the whole workload below takes about 1.5 seconds. Note also that `--project=.`
# does not hide your *global* environment: if CairoMakie is installed there, this
# will find and load it even though it is not a dependency of this project.
println("Loading (first run precompiles the Makie extension; this can take a minute)...")
try
    @eval using CairoMakie
catch
    @warn "CairoMakie not available — figures will be skipped, CSV export will still run"
end

const CAN_PLOT = Base.invokelatest(plotting_available)
outdir = mktempdir()
@printf("Writing to %s\n", outdir)
@printf("Plotting backend loaded: %s\n\n", CAN_PLOT ? "yes" : "no")

# ---------------------------------------------------------------------------
# 1. A field, and metrics recorded during the run.
# ---------------------------------------------------------------------------
model = Heat2D(nx = 128, ny = 128, alpha = 0.15f0, dt = 0.1f0)
initialize_gaussian!(model.field; amplitude = 100.0f0, sigma = 8.0f0)

recorder = MetricRecorder(
    total = m -> sum(state(m)),
    peak = m -> maximum(state(m)),
    centre = m -> center_value(m),
)

metrics = run!(Simulation(model; stop = Steps(2000));
               callback = recorder, callback_every = 50)

@printf("Ran %d steps in %.4f s; retained %d samples of %d metrics.\n",
        metrics.steps, metrics.compute_seconds, length(recorder), length(keys(recorder)))
@printf("State retained: %d numbers. Full state would have been %d.\n\n",
        length(recorder) * length(keys(recorder)),
        length(recorder) * length(model.field))

write_csv(joinpath(outdir, "metrics.csv"), recorder)
println("wrote metrics.csv")

if CAN_PLOT
    save(joinpath(outdir, "field.png"),
         plot_field(model; title = "Heat2D after $(metrics.steps) steps"))
    save(joinpath(outdir, "metrics.png"), plot_series(recorder; title = "in-situ metrics"))
    println("wrote field.png, metrics.png")
end

# ---------------------------------------------------------------------------
# 2. Detection: the figure Lab Work 4 asks for at its session 8 entry check.
# ---------------------------------------------------------------------------
observed = synthetic_series(samples = 140, noise = 0.4, seed = 11,
                            anomalies = [LevelShift(80, 5.0)])
residuals = observed.values .- observed.clean

write_csv(joinpath(outdir, "observations.csv"), observed)
println("\nwrote observations.csv")

println("\nDetection at three thresholds:")
@printf("%10s %8s %8s %10s\n", "threshold", "delay", "false+", "recall")
for threshold in (1.0, 2.0, 4.0)
    report = detection_report(residuals, observed.anomalous, threshold)
    @printf("%10.1f %8s %8d %10.2f\n", threshold,
            report.delay === missing ? "never" : string(report.delay),
            report.false_positives, report.recall)
end

if CAN_PLOT
    save(joinpath(outdir, "detection.png"),
         plot_detection(observed.times, residuals, observed.anomalous;
                        thresholds = [1.0, 2.0, 4.0],
                        title = "residuals, true fault shaded"))
    println("wrote detection.png")
end

# ---------------------------------------------------------------------------
# 3. A scenario sweep.
# ---------------------------------------------------------------------------
results = parameter_sweep(Float32[0.05, 0.10, 0.15, 0.20, 0.25];
                          stop = Steps(500), threaded = true) do alpha
    scenario = Heat2D(nx = 96, ny = 96, alpha = alpha)
    initialize_peak!(scenario.field, 100.0f0)
    scenario
end

println("\n", sweep_table(results))
write_csv(joinpath(outdir, "sweep.csv"), results)
println("wrote sweep.csv")

if CAN_PLOT
    save(joinpath(outdir, "sweep.png"), plot_sweep(results; y = :centre,
                                                   title = "centre temperature vs alpha"))
    println("wrote sweep.png")
end

# ---------------------------------------------------------------------------
# 4. Animation — the in-situ pattern applied to pictures.
# ---------------------------------------------------------------------------
if CAN_PLOT
    animated = Heat2D(nx = 128, ny = 128, alpha = 0.15f0, dt = 0.1f0)
    initialize_gaussian!(animated.field; amplitude = 100.0f0, sigma = 6.0f0)

    path = joinpath(outdir, "diffusion.mp4")
    animate_field(animated, path;
                  frames = 60, steps_per_frame = 20, framerate = 20,
                  colorrange = (0.0, 60.0),          # fixed: see below
                  title = "heat diffusion")
    @printf("\nwrote diffusion.mp4 (%.1f kB) — 60 frames, 1200 steps\n", filesize(path) / 1024)
    @printf("Storing every frame's field instead would have been %.1f MB\n",
            60 * length(animated.field) * 4 / 1024^2)
end

println("""

Two things worth carrying into a report:

  * **Fix `colorrange` for animations.** Left to rescale per frame, Makie makes a
    decaying peak look perfectly constant — the animation shows the colour map
    adapting, not the physics. Any figure whose axes move between frames is
    telling you about its own normalisation.
  * **Ship the CSV, not just the picture.** `write_csv` needs no dependencies and
    runs anywhere the simulation does. A figure nobody can regenerate from data
    is an assertion; the data plus the script that drew it is evidence, and it is
    what the reproducibility requirement in Lab Work 4 actually asks for.
""")

@printf("All output in: %s\n", outdir)
