module VisuTwinSimMakieExt

using Makie
using Printf
using VisuTwinSim
using VisuTwinSim: MetricRecorder, TwinLog, ObservationSeries, Heat2D, Field2D

# Triggered by `Makie`, not by a specific backend, so `using CairoMakie` (headless,
# for reports and CI) and `using GLMakie` (interactive) both work — both load
# Makie as a dependency.

VisuTwinSim.plotting_backend_loaded(::Val{:makie}) = true

# The field may live on a GPU; every plot needs it on the host.
host_field(model::Heat2D) = Array(state(model))
host_field(field::Field2D) = Array(state(field))
host_field(data::AbstractMatrix) = Array(data)

# ---------------------------------------------------------------------------
# Fields
# ---------------------------------------------------------------------------

function VisuTwinSim.plot_field(model::Union{Heat2D,Field2D,AbstractMatrix};
                                title = "",
                                colormap = :inferno,
                                colorrange = nothing,
                                axis_labels = ("x", "y"),
                                size = (620, 520))
    data = host_field(model)
    range = colorrange === nothing ? auto_colorrange(data) : colorrange

    figure = Figure(; size = size)
    axis = Axis(figure[1, 1];
                title = title,
                xlabel = axis_labels[1], ylabel = axis_labels[2],
                aspect = Makie.DataAspect())

    # Passed as-is: Makie's heatmap reads the first index as x and the second as
    # y, which is the package's own convention for Field2D. No transpose.
    plot = heatmap!(axis, data; colormap = colormap, colorrange = range)
    Colorbar(figure[1, 2], plot)
    return figure
end

function auto_colorrange(data)
    low, high = extrema(data)
    low == high && return (low - 1, high + 1)   # Makie rejects a zero-width range
    return (low, high)
end

# ---------------------------------------------------------------------------
# Recorded metrics
# ---------------------------------------------------------------------------

function VisuTwinSim.plot_series(recorder::MetricRecorder;
                                 metrics = collect(keys(recorder.metrics)),
                                 title = "",
                                 xlabel = "simulated time",
                                 size = nothing)
    isempty(recorder) && throw(ArgumentError("nothing recorded yet"))
    names = collect(metrics)
    height = 200 * length(names) + 60
    figure = Figure(; size = something(size, (720, height)))

    for (row, name) in enumerate(names)
        haskey(recorder.values, name) ||
            throw(ArgumentError("no metric named :$name; recorded: $(join(keys(recorder.metrics), ", "))"))
        axis = Axis(figure[row, 1];
                    ylabel = string(name),
                    xlabel = row == length(names) ? xlabel : "",
                    title = row == 1 ? title : "")
        lines!(axis, recorder.times, recorder[name])
        row == length(names) || hidexdecorations!(axis; grid = false)
    end
    return figure
end

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

function VisuTwinSim.plot_detection(times, residuals, truth;
                                    thresholds = Float64[],
                                    title = "residuals and detection thresholds",
                                    xlabel = "time",
                                    ylabel = "residual",
                                    size = (760, 420))
    length(times) == length(residuals) == length(truth) || throw(DimensionMismatch(
        "times, residuals and truth must have equal length"))

    figure = Figure(; size = size)
    axis = Axis(figure[1, 1]; title = title, xlabel = xlabel, ylabel = ylabel)

    # Shade the intervals where the fault was actually present, so a threshold
    # can be judged against the truth rather than against the eye.
    for (from, to) in true_intervals(truth)
        vspan!(axis, times[from], times[to]; color = (:red, 0.12))
    end

    lines!(axis, times, residuals; color = :black, label = "residual")

    palette = [:royalblue, :darkorange, :seagreen, :purple, :crimson]
    for (k, threshold) in enumerate(thresholds)
        colour = palette[mod1(k, length(palette))]
        hlines!(axis, [threshold, -threshold];
                color = colour, linestyle = :dash,
                label = @sprintf("threshold %.3g", threshold))
    end

    # Opaque background and a frame: the threshold lines run the full width of
    # the axis and strike through legend text otherwise.
    isempty(thresholds) || axislegend(axis; position = :lt, merge = true,
                                      framevisible = true,
                                      backgroundcolor = (:white, 0.92))
    return figure
end

VisuTwinSim.plot_detection(residuals, truth; kwargs...) =
    VisuTwinSim.plot_detection(1:length(residuals), residuals, truth; kwargs...)

"""Contiguous runs of `true`, as (from, to) index pairs."""
function true_intervals(truth)
    intervals = Tuple{Int,Int}[]
    start = 0
    for (k, flag) in enumerate(truth)
        if flag && start == 0
            start = k
        elseif !flag && start != 0
            push!(intervals, (start, k - 1))
            start = 0
        end
    end
    start == 0 || push!(intervals, (start, length(truth)))
    return intervals
end

# ---------------------------------------------------------------------------
# Scenario sweeps
# ---------------------------------------------------------------------------

function VisuTwinSim.plot_sweep(results::AbstractVector;
                                y = nothing,
                                title = "scenario sweep",
                                xlabel = "scenario",
                                size = (720, 420))
    isempty(results) && throw(ArgumentError("no scenarios to plot"))
    name = y === nothing ? first(keys(first(results).observation)) : y

    labels = [VisuTwinSim.scenario_label(r.scenario) for r in results]
    values = [Float64(getproperty(r.observation, name)) for r in results]

    figure = Figure(; size = size)
    axis = Axis(figure[1, 1];
                title = title, xlabel = xlabel, ylabel = string(name),
                xticks = (1:length(labels), labels),
                xticklabelrotation = length(labels) > 4 ? pi / 6 : 0.0)
    scatterlines!(axis, 1:length(values), values)
    return figure
end

# ---------------------------------------------------------------------------
# Animation and streaming frames
# ---------------------------------------------------------------------------

function VisuTwinSim.animate_field(model::Heat2D, path::AbstractString;
                                   backend = CPUBackend(),
                                   frames::Integer = 100,
                                   steps_per_frame::Integer = 10,
                                   framerate::Integer = 20,
                                   colormap = :inferno,
                                   colorrange = nothing,
                                   title = "")
    frames > 0 || throw(ArgumentError("frames must be positive, got $frames"))
    steps_per_frame > 0 || throw(ArgumentError("steps_per_frame must be positive"))

    # Fixed by default from the initial state: letting Makie rescale per frame
    # makes a decaying peak look constant, which is exactly the thing the
    # animation is supposed to show.
    range = colorrange === nothing ? auto_colorrange(host_field(model)) : colorrange

    observable = Makie.Observable(host_field(model))
    figure = Figure(; size = (620, 520))
    axis = Axis(figure[1, 1]; title = title, aspect = Makie.DataAspect())
    plot = heatmap!(axis, observable; colormap = colormap, colorrange = range)
    Colorbar(figure[1, 2], plot)

    Makie.record(figure, path, 1:frames; framerate = framerate) do _
        run!(model; backend = backend, steps = steps_per_frame)
        observable[] = host_field(model)      # the frame is kept; the state is not
    end
    return path
end

function VisuTwinSim.frame_callback(path_pattern::AbstractString;
                                    every::Integer = 1,
                                    colorrange = nothing,
                                    kwargs...)
    return function (model, progress)
        progress.step % every == 0 || return nothing
        figure = VisuTwinSim.plot_field(model;
                                        title = @sprintf("step %d, t = %.4g",
                                                         progress.step, progress.simulated_time),
                                        colorrange = colorrange, kwargs...)
        Makie.save(Printf.format(Printf.Format(path_pattern), progress.step), figure)
        return nothing
    end
end

end # module VisuTwinSimMakieExt
