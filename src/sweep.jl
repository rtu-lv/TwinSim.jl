"""
    parameter_sweep(build, scenarios; backend, stop, observe, threaded) -> Vector{NamedTuple}

Run one simulation per scenario and reduce each to a result.

`build(scenario)` returns a fresh model; `observe(model, metrics)` returns
whatever the sweep should record. Each entry of the result carries the scenario,
the observation, and the run's [`RunMetrics`](@ref).

```julia
alphas = [0.05f0, 0.1f0, 0.15f0, 0.2f0]

results = parameter_sweep(alphas; threaded = true) do alpha
    model = Heat2D(nx = 128, ny = 128, alpha = alpha)
    initialize_peak!(model.field, 100.0f0)
    model
end

results[1].scenario     # 0.05f0
results[1].observation  # what `observe` returned
results[1].metrics      # cost of that run
```

Scenarios are independent, which makes this the second embarrassingly parallel
workload in the package after [`random_walk_ensemble`](@ref) — and a more
realistic one, since each task is large enough that the parallelism actually
pays. `threaded = true` runs them across `Threads.nthreads()`.

Threading is refused for GPU backends rather than silently ignored. Several host
threads submitting to one device do not get more of it, and the failure mode is
confusing: no speedup, and occasionally a driver-level error. For a GPU sweep,
either run the scenarios in sequence — the device is already parallel — or batch
them into one larger problem.

`observe` defaults to a summary that is cheap and always available; supply your
own to keep something specific.
"""
function parameter_sweep(build, scenarios;
                         backend::AbstractBackend = CPUBackend(),
                         stop = Steps(100),
                         observe = default_observation,
                         threaded::Bool = false)
    if threaded && is_gpu(backend)
        throw(ArgumentError("""
            threaded = true is not supported for GPU backends ($(backend_name(backend))).
            Several host threads submitting to one device share the same device; the
            scenarios do not run any faster and the driver may error. Run the sweep in
            sequence, or batch the scenarios into a single larger problem.
            """))
    end

    cases = collect(scenarios)
    results = Vector{Any}(undef, length(cases))

    if threaded
        Threads.@threads for k in eachindex(cases)
            results[k] = run_scenario(build, cases[k], backend, stop, observe)
        end
    else
        for k in eachindex(cases)
            results[k] = run_scenario(build, cases[k], backend, stop, observe)
        end
    end

    return identity.(results)
end

function run_scenario(build, scenario, backend, stop, observe)
    model = build(scenario)
    metrics = run!(Simulation(model; backend = backend, stop = as_stop_condition(stop)))
    return (; scenario, observation = observe(model, metrics), metrics)
end

"""
    default_observation(model, metrics)

What a sweep records when nothing else is asked for: the total, the centre value,
and why the run stopped.
"""
default_observation(model, metrics) =
    (; total = metrics.total_state,
       centre = Float64(center_value(model)),
       steps = metrics.steps,
       stopped_by = metrics.stopped_by)

"""
    sweep_table(results; columns = nothing) -> String

Render sweep results as a fixed-width table, for pasting into a report.

```julia
println(sweep_table(results))
```
"""
function sweep_table(results; columns = nothing)
    isempty(results) && return "(no scenarios)"

    first_observation = first(results).observation
    names = columns === nothing ? collect(keys(first_observation)) : collect(columns)

    io = IOBuffer()
    @printf(io, "%-24s", "scenario")
    foreach(name -> @printf(io, " %14s", name), names)
    @printf(io, " %12s\n", "MLUP/s")

    for result in results
        @printf(io, "%-24s", scenario_label(result.scenario))
        for name in names
            value = getproperty(result.observation, name)
            value isa Real ? @printf(io, " %14.6g", value) : @printf(io, " %14s", value)
        end
        @printf(io, " %12.0f\n", mlups(result.metrics))
    end
    return String(take!(io))
end

scenario_label(scenario) = string(scenario)
scenario_label(scenario::NamedTuple) =
    join(("$k=$(v)" for (k, v) in pairs(scenario)), " ")
