"""
    Simulation(model; backend = CPUBackend(), stop = Steps(1))

A model, the backend it runs on, and when it should stop. `stop` accepts any
[`StopCondition`](@ref) or a plain integer step count.
"""
struct Simulation{M,B<:AbstractBackend,S<:StopCondition}
    model::M
    backend::B
    stop::S
end

function Simulation(model; backend::AbstractBackend = CPUBackend(), stop = Steps(1))
    return Simulation(model, backend, as_stop_condition(stop))
end

function Base.show(io::IO, ::MIME"text/plain", sim::Simulation)
    println(io, "Simulation")
    println(io, "  backend  ", backend_name(sim.backend))
    println(io, "  stop     ", sim.stop)
    print(io, "  model    ", summary(sim.model))
    return nothing
end

# ---------------------------------------------------------------------------
# Host <-> device movement
# ---------------------------------------------------------------------------

"""
    to_backend(model, backend) -> model

Return a model whose buffers live in `backend`'s memory, or the original model
when it is already there.

Calling this once and reusing the result keeps the state device-resident across
several `run!` calls, so a long run is not billed for an upload and a download
every time. `run!` calls it internally, which is why a model already on the
device costs no transfer.
"""
to_backend(model, backend::AbstractBackend) = first(upload(backend, model))

function upload(backend::AbstractBackend, model::AbstractModel)
    device = ka_device(backend)
    device === nothing && return (model, 0.0, 0)
    KernelAbstractions.get_backend(state(model)) == device && return (model, 0.0, 0)

    start = time_ns()
    device_model, bytes = move_to_device(model, device)
    KernelAbstractions.synchronize(device)
    return (device_model, (time_ns() - start) / 1e9, bytes)
end

"""
    sync_to_host!(host, backend, device_model) -> (seconds, bytes)

Copy the live buffer back into `host`. A no-op — and free — when the two models
are the same object, which is the case for every CPU backend.
"""
function sync_to_host!(host::AbstractModel, backend::AbstractBackend,
                       device_model::AbstractModel)
    host === device_model && return (0.0, 0)
    start = time_ns()
    bytes = sync_state!(host, device_model)
    device_synchronize(backend)
    return ((time_ns() - start) / 1e9, bytes)
end

# ---------------------------------------------------------------------------
# The run loop
# ---------------------------------------------------------------------------

"""
    run!(sim::Simulation; callback, callback_every, realtime_factor, step_limit) -> RunMetrics
    run!(model; backend = CPUBackend(), steps = 1, kwargs...) -> RunMetrics

Advance the simulation and return a [`RunMetrics`](@ref) describing both the
result and the cost of producing it.

Keyword arguments:

- `callback` – called as `callback(model, state)` every `callback_every` steps,
  with `model` synchronised to host memory. Return `:stop` to end the run early.
  Use it for live visualisation, logging or writing checkpoints.
- `callback_every` – callback interval in steps (default `1`).
- `realtime_factor` – pace the loop so that `realtime_factor` units of simulated
  time elapse per second of wall clock. `1.0` runs the twin in real time;
  `nothing` (the default) runs as fast as possible.
- `step_limit` – hard safety bound for open-ended stop conditions.

Timing notes: `compute_seconds` excludes callbacks, host transfers and real-time
pacing, and is measured after a device synchronise so GPU numbers reflect kernel
execution rather than launch time. A [`Converged`](@ref) stop condition adds a
grid-wide reduction that *is* counted as compute.
"""
function run!(sim::Simulation{<:AbstractModel};
              callback = nothing,
              callback_every::Integer = 1,
              realtime_factor::Union{Nothing,Real} = nothing,
              step_limit::Integer = 100_000_000)
    callback_every > 0 || throw(ArgumentError("callback_every must be positive"))

    host = sim.model
    backend = sim.backend
    T = eltype(host)

    metrics = RunMetrics(backend = backend_name(backend),
                         precision = T,
                         cells = cells(host),
                         bytes_per_cell = bytes_per_cell(host),
                         flops_per_cell = flops_per_cell(host))

    wall_start = time_ns()
    model, upload_seconds, upload_bytes = upload(backend, host)
    metrics.transfer_seconds += upload_seconds
    metrics.transferred_bytes += upload_bytes

    interval = change_interval(sim.stop)
    snapshot = tracks_change(sim.stop) ? copy(state(model)) : nothing

    dt = Float64(timestep(host))
    # Continue from where the model left off, so a driven model advanced in
    # windows sees a monotonic clock instead of replaying its first window.
    started_at = clock(host)[]
    progress = RunState(started_at)
    excluded_ns = 0  # callback + transfer + pacing time, subtracted from compute

    loop_start = time_ns()
    while true
        reason = stop_reason(sim.stop, progress)
        if reason !== nothing
            metrics.stopped_by = reason
            break
        end
        if progress.step >= step_limit
            metrics.stopped_by = :step_limit
            @warn "run! hit step_limit=$step_limit before any stop condition fired" sim.stop
            break
        end

        # The drive is evaluated at the time *entering* the step, so the first
        # step sees t = 0 and the series is sampled at the same instants the
        # state is reported at.
        step!(backend, model, oftype(dt, progress.simulated_time))
        progress.step += 1
        progress.simulated_time += dt

        if snapshot !== nothing && progress.step % interval == 0
            current = state(model)
            progress.max_change = Float64(maximum(abs, current .- snapshot))
            copyto!(snapshot, current)
        end

        if callback !== nothing && progress.step % callback_every == 0
            pause = time_ns()
            seconds, bytes = sync_to_host!(host, backend, model)
            metrics.transfer_seconds += seconds
            metrics.transferred_bytes += bytes
            verdict = callback(host, progress)
            excluded_ns += time_ns() - pause
            if verdict === :stop
                metrics.stopped_by = :callback
                break
            end
        end

        if realtime_factor !== nothing
            # Paced against what *this* run has advanced: the wall clock below
            # also starts at this call, so an absolute simulated time would make
            # a chained run think it was already far behind.
            target = advanced(progress) / realtime_factor
            achieved = (time_ns() - loop_start) / 1e9
            if target > achieved
                pause = time_ns()
                sleep(target - achieved)
                excluded_ns += time_ns() - pause
            end
        end

        progress.elapsed_seconds = (time_ns() - loop_start) / 1e9
    end

    # Only after synchronising does the elapsed time mean anything on a GPU:
    # kernel launches are asynchronous, so without this we would be timing the
    # launch queue rather than the kernels.
    device_synchronize(backend)
    metrics.compute_seconds = max((time_ns() - loop_start - excluded_ns) / 1e9, 0.0)

    seconds, bytes = sync_to_host!(host, backend, model)
    metrics.transfer_seconds += seconds
    metrics.transferred_bytes += bytes

    metrics.steps = progress.step
    # The metrics report what *this* run advanced; the model keeps the total.
    metrics.simulated_time = progress.simulated_time - started_at
    clock(host)[] = progress.simulated_time
    metrics.total_state = Float64(sum_state(model))
    metrics.elapsed_seconds = (time_ns() - wall_start) / 1e9
    return metrics
end

function run!(model::AbstractModel; backend::AbstractBackend = CPUBackend(), steps = 1, kwargs...)
    return run!(Simulation(model; backend, stop = as_stop_condition(steps)); kwargs...)
end

# Deliberately the most generic method in the package, so a backend extension
# *adds* a `step!` method rather than overwriting this one. Overwriting a method
# from an extension invalidates the compiled code that already called it.
function step!(backend::AbstractBackend, model)
    throw(ArgumentError("no step! implementation for $(typeof(backend)) applied to $(typeof(model)). " *
                        "Available backends: $(join(available_backends(), ", "))"))
end
