# Getting results out of the package, in two tiers.
#
# Tier 1, here: CSV export, which needs no dependencies and therefore works
# everywhere — on a headless cluster node, in a container, in a CI job. Data that
# has left the process can be plotted with anything, including tools that are not
# Julia, and it is what a reproducibility package should contain anyway.
#
# Tier 2, in ext/TwinSimMakieExt.jl: real figures, available once a Makie
# backend is loaded. Plotting is a heavy dependency and most runs do not need it,
# so it stays optional — the same arrangement as the GPU backends.

"""
    write_csv(path, data) -> path

Write recorded results to a CSV file.

Accepts a [`MetricRecorder`](@ref), a [`TwinLog`](@ref) or an
[`ObservationSeries`](@ref). No dependencies are involved, so this works on any
machine that can run the simulation — which is the point, since the machine that
runs the big cases is rarely the one with a plotting stack installed.

```julia
write_csv("out/metrics.csv", recorder)
write_csv("out/twin.csv", log)
write_csv("out/observations.csv", observed)
```

Include the file in the reproducibility package. A figure nobody can regenerate
from data is an assertion, not evidence.
"""
function write_csv end

function write_csv(path::AbstractString, recorder::MetricRecorder)
    names = collect(keys(recorder.metrics))
    open(path, "w") do io
        println(io, join(vcat(["step", "simulated_time"], string.(names)), ","))
        for k in 1:length(recorder)
            row = vcat([string(recorder.steps[k]), string(recorder.times[k])],
                       [string(recorder[name][k]) for name in names])
            println(io, join(row, ","))
        end
    end
    return path
end

function write_csv(path::AbstractString, log::TwinLog)
    # `checks` is whatever validate returned. A NamedTuple gets one column per
    # field; anything else is stringified into a single column.
    # `keys` needs an instance, not the element type.
    check_names = (!isempty(log.checks) && first(log.checks) isa NamedTuple) ?
                  collect(keys(first(log.checks))) : Symbol[]

    open(path, "w") do io
        header = vcat(["window", "time", "observation"], string.(check_names),
                      ["decision", "steps", "compute_seconds"])
        isempty(check_names) && insert!(header, 4, "checks")
        println(io, join(header, ","))

        for k in 1:length(log)
            row = [string(k), string(log.times[k]), csv_field(log.observations[k])]
            if isempty(check_names)
                push!(row, csv_field(log.checks[k]))
            else
                append!(row, [csv_field(getproperty(log.checks[k], name)) for name in check_names])
            end
            push!(row, csv_field(log.decisions[k]))
            push!(row, string(log.metrics[k].steps))
            push!(row, string(log.metrics[k].compute_seconds))
            println(io, join(row, ","))
        end
    end
    return path
end

function write_csv(path::AbstractString, series::ObservationSeries)
    open(path, "w") do io
        println(io, "index,time,value,clean,anomalous")
        for k in 1:length(series)
            println(io, join((k, series.times[k], series.values[k],
                              series.clean[k], series.anomalous[k]), ","))
        end
    end
    return path
end

function write_csv(path::AbstractString, results::AbstractVector{<:NamedTuple})
    isempty(results) && throw(ArgumentError("nothing to write"))
    hasproperty(first(results), :observation) || throw(ArgumentError(
        "expected sweep results; got a vector of NamedTuples without an `observation` field"))
    names = collect(keys(first(results).observation))

    open(path, "w") do io
        println(io, join(vcat(["scenario"], string.(names), ["mlups", "compute_seconds"]), ","))
        for result in results
            row = vcat([csv_field(scenario_label(result.scenario))],
                       [csv_field(getproperty(result.observation, name)) for name in names],
                       [string(mlups(result.metrics)), string(result.metrics.compute_seconds)])
            println(io, join(row, ","))
        end
    end
    return path
end

# Quote anything containing a comma or a quote, so a stringified NamedTuple does
# not silently split into extra columns.
function csv_field(value)
    text = string(value)
    if occursin(',', text) || occursin('"', text) || occursin('\n', text)
        return string('"', replace(text, '"' => "\"\""), '"')
    end
    return text
end

# ---------------------------------------------------------------------------
# Tier 2: figures, filled in by the Makie extension
# ---------------------------------------------------------------------------
#
# Declared here so they are exported, documented and discoverable from
# `names(TwinSim)` whether or not a plotting backend is installed. The
# fallbacks below are the only methods in this module; the extension adds more
# specific ones rather than overwriting these.

const PLOTTING_HINT = """
    Plotting requires a Makie backend, which is an optional dependency:

        using Pkg; Pkg.add("CairoMakie")
        using CairoMakie          # or GLMakie for an interactive window
        using TwinSim

    CairoMakie works headless, so it is the one to use on a cluster node or in CI.

    Without a plotting backend, `write_csv` exports the same data for plotting
    elsewhere — that path has no dependencies and always works.
    """

"""
    plot_field(model; title, colormap, colorrange) -> Figure

Heatmap of the model's current state, with a colour bar. Works on any backend;
a device-resident field is copied to the host first.

Requires a Makie backend — see [`write_csv`](@ref) for the dependency-free path.
"""
plot_field(args...; kwargs...) = throw(ArgumentError(PLOTTING_HINT))

"""
    plot_series(recorder; metrics, title) -> Figure

Recorded metrics against simulated time, one panel per metric.

This is the figure Lab Work 4 asks for at its session 8 entry check.
"""
plot_series(args...; kwargs...) = throw(ArgumentError(PLOTTING_HINT))

"""
    plot_detection(times, residuals, truth; thresholds, title) -> Figure

Residuals over time, with the true anomalous interval shaded and each candidate
threshold drawn as a horizontal line.

The figure to argue a threshold choice from: it shows at a glance how much of the
normal signal a given threshold would clip, and how far into the fault it sits.
"""
plot_detection(args...; kwargs...) = throw(ArgumentError(PLOTTING_HINT))

"""
    plot_sweep(results; x, y, title) -> Figure

Scenario sweep results: an observed quantity against the scenario that produced it.
"""
plot_sweep(args...; kwargs...) = throw(ArgumentError(PLOTTING_HINT))

"""
    animate_field(model, path; backend, stop, every, framerate, colorrange) -> path

Run the model and record the field evolving, to `.mp4` or `.gif`.

The in-situ pattern applied to pictures: frames are rendered *during* the run and
the field is discarded, rather than storing every state and post-processing. A
2048² `Float32` field held every 10 steps of a 10 000-step run is 16 GB of state;
the animation is a few megabytes.

Fix `colorrange` when comparing animations — Makie rescales per frame otherwise,
which makes a decaying peak look constant.
"""
animate_field(args...; kwargs...) = throw(ArgumentError(PLOTTING_HINT))

"""
    frame_callback(path_pattern; every, colorrange, kwargs...) -> callback

A `run!` callback that writes one image per interval, for streaming visualisation
of a long run without holding the frames in memory.

```julia
run!(sim; callback = frame_callback("out/frame_%05d.png"), callback_every = 100)
```
"""
frame_callback(args...; kwargs...) = throw(ArgumentError(PLOTTING_HINT))

"""
    plotting_available() -> Bool

Whether a Makie backend is loaded and the plotting functions will work.

```julia
figure = plotting_available() ? plot_series(recorder) : write_csv("metrics.csv", recorder)
```
"""
plotting_available() = plotting_backend_loaded(Val(:makie))

# The extension adds `plotting_backend_loaded(::Val{:makie})`, which is more
# specific than this. Redefining `plotting_available` itself from the extension
# is a method overwrite, and Julia refuses to precompile a module that does one.
plotting_backend_loaded(::Val) = false
