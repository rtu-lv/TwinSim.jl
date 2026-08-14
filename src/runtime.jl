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

upload(::CPUBackend, model::Heat2D) = (model, 0.0, 0)

function upload(backend::KernelBackend, model::Heat2D{T}) where {T}
    KernelAbstractions.get_backend(model.field.current) == backend.device &&
        return (model, 0.0, 0)

    device = backend.device
    start = time_ns()
    current = KernelAbstractions.allocate(device, T, size(model.field))
    next = KernelAbstractions.allocate(device, T, size(model.field))
    copyto!(current, model.field.current)
    KernelAbstractions.synchronize(device)
    elapsed = (time_ns() - start) / 1e9

    device_model = Heat2D(Field2D(current, next), model.params, model.boundary)
    return (device_model, elapsed, sizeof(T) * length(model.field))
end

"""
    sync_to_host!(host, backend, device_model) -> (seconds, bytes)

Copy the live buffer back into `host`. A no-op — and free — when the two models
are the same object, which is the case for every CPU backend.
"""
function sync_to_host!(host::Heat2D, backend::AbstractBackend, device_model::Heat2D)
    host === device_model && return (0.0, 0)
    start = time_ns()
    copyto!(host.field.current, device_model.field.current)
    device_synchronize(backend)
    bytes = sizeof(eltype(host)) * length(host.field)
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
function run!(sim::Simulation{<:Heat2D};
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
                         cells = length(host.field),
                         bytes_per_cell = bytes_per_cell(host),
                         flops_per_cell = flops_per_cell(host))

    wall_start = time_ns()
    model, upload_seconds, upload_bytes = upload(backend, host)
    metrics.transfer_seconds += upload_seconds
    metrics.transferred_bytes += upload_bytes

    interval = change_interval(sim.stop)
    snapshot = tracks_change(sim.stop) ? copy(model.field.current) : nothing

    dt = Float64(host.params.dt)
    state = RunState()
    excluded_ns = 0  # callback + transfer + pacing time, subtracted from compute

    loop_start = time_ns()
    while true
        reason = stop_reason(sim.stop, state)
        if reason !== nothing
            metrics.stopped_by = reason
            break
        end
        if state.step >= step_limit
            metrics.stopped_by = :step_limit
            @warn "run! hit step_limit=$step_limit before any stop condition fired" sim.stop
            break
        end

        step!(backend, model)
        state.step += 1
        state.simulated_time += dt

        if snapshot !== nothing && state.step % interval == 0
            current = model.field.current
            state.max_change = Float64(maximum(abs, current .- snapshot))
            copyto!(snapshot, current)
        end

        if callback !== nothing && state.step % callback_every == 0
            pause = time_ns()
            seconds, bytes = sync_to_host!(host, backend, model)
            metrics.transfer_seconds += seconds
            metrics.transferred_bytes += bytes
            verdict = callback(host, state)
            excluded_ns += time_ns() - pause
            if verdict === :stop
                metrics.stopped_by = :callback
                break
            end
        end

        if realtime_factor !== nothing
            target = state.simulated_time / realtime_factor
            achieved = (time_ns() - loop_start) / 1e9
            if target > achieved
                pause = time_ns()
                sleep(target - achieved)
                excluded_ns += time_ns() - pause
            end
        end

        state.elapsed_seconds = (time_ns() - loop_start) / 1e9
    end

    # Only after synchronising does the elapsed time mean anything on a GPU:
    # kernel launches are asynchronous, so without this we would be timing the
    # launch queue rather than the kernels.
    device_synchronize(backend)
    metrics.compute_seconds = max((time_ns() - loop_start - excluded_ns) / 1e9, 0.0)

    seconds, bytes = sync_to_host!(host, backend, model)
    metrics.transfer_seconds += seconds
    metrics.transferred_bytes += bytes

    metrics.steps = state.step
    metrics.simulated_time = state.simulated_time
    metrics.total_state = Float64(sum_state(model))
    metrics.elapsed_seconds = (time_ns() - wall_start) / 1e9
    return metrics
end

function run!(model::Heat2D; backend::AbstractBackend = CPUBackend(), steps = 1, kwargs...)
    return run!(Simulation(model; backend, stop = as_stop_condition(steps)); kwargs...)
end

# Deliberately the most generic method in the package, so a backend extension
# *adds* a `step!` method rather than overwriting this one. Overwriting a method
# from an extension invalidates the compiled code that already called it.
function step!(backend::AbstractBackend, model)
    throw(ArgumentError("no step! implementation for $(typeof(backend)) applied to $(typeof(model)). " *
                        "Available backends: $(join(available_backends(), ", "))"))
end
