@testset "parameter_sweep" begin
    alphas = Float32[0.05, 0.1, 0.15, 0.2]

    build(alpha) = begin
        model = Heat2D(nx = 48, ny = 48, alpha = alpha)
        initialize_peak!(model.field, 100.0f0)
        model
    end

    results = parameter_sweep(build, alphas; stop = Steps(200))
    @test length(results) == 4
    @test [r.scenario for r in results] == alphas
    @test all(r -> r.metrics.steps == 200, results)
    @test all(r -> isapprox(r.observation.total, 100.0; rtol = 1e-4), results)  # Neumann conserves

    # A faster diffusion spreads the peak further, so the centre is cooler.
    centres = [r.observation.centre for r in results]
    @test issorted(centres; rev = true)

    # Scenario order is preserved, threaded or not.
    threaded = parameter_sweep(build, alphas; stop = Steps(200), threaded = true)
    @test [r.scenario for r in threaded] == alphas
    @test [r.observation.centre for r in threaded] ≈ centres

    @test isempty(parameter_sweep(build, Float32[]))
end

@testset "sweep scenarios can be any object" begin
    scenarios = [(; nx = 32, alpha = 0.1f0), (; nx = 48, alpha = 0.2f0)]

    results = parameter_sweep(scenarios; stop = Steps(50)) do scenario
        model = Heat2D(nx = scenario.nx, ny = scenario.nx, alpha = scenario.alpha)
        initialize_peak!(model.field, 50.0f0)
        model
    end

    @test length(results) == 2
    @test results[1].metrics.cells == 32 * 32
    @test results[2].metrics.cells == 48 * 48
    @test occursin("nx=32", sweep_table(results))
end

@testset "sweep observations are configurable" begin
    results = parameter_sweep([0.1f0, 0.2f0];
                              stop = Steps(100),
                              observe = (model, metrics) ->
                                  (; peak = Float64(maximum(state(model))),
                                     spread = Float64(maximum(state(model)) -
                                                      minimum(state(model))))) do alpha
        model = Heat2D(nx = 32, ny = 32, alpha = alpha)
        initialize_peak!(model.field, 100.0f0)
        model
    end

    @test Set(keys(results[1].observation)) == Set((:peak, :spread))
    @test results[1].observation.peak > results[2].observation.peak

    table = sweep_table(results)
    @test occursin("peak", table)
    @test occursin("MLUP/s", table)
    @test occursin("spread", table)

    narrow = sweep_table(results; columns = [:peak])
    @test occursin("peak", narrow)
    @test !occursin("spread", narrow)

    @test sweep_table([]) == "(no scenarios)"
end

@testset "sweep with different stop conditions" begin
    results = parameter_sweep([0.05f0, 0.1f0]; stop = UntilTime(5.0)) do alpha
        model = Heat2D(nx = 24, ny = 24, alpha = alpha)
        initialize_peak!(model.field, 100.0f0)
        model
    end
    @test all(r -> r.observation.stopped_by === :time, results)
    @test all(r -> r.observation.steps == 50, results)

    # An integer is accepted as a step count, as elsewhere.
    @test parameter_sweep(a -> Heat2D(nx = 16, alpha = a), [0.1f0]; stop = 7)[1].metrics.steps == 7
end

@testset "threading a GPU sweep is refused, not ignored" begin
    for (name, backend) in GPU_BACKENDS
        err = try
            parameter_sweep(a -> Heat2D(nx = 16, alpha = a), [0.1f0];
                            backend = backend, threaded = true)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("threaded", sprint(showerror, err))
    end
end

@testset "sweep runs on a GPU backend in sequence" begin
    for (name, backend) in GPU_BACKENDS
        alphas = Float32[0.05, 0.1, 0.15]
        build(alpha) = begin
            model = Heat2D(nx = 32, ny = 32, alpha = alpha)
            initialize_peak!(model.field, 100.0f0)
            model
        end

        gpu = parameter_sweep(build, alphas; backend = backend, stop = Steps(100))
        cpu = parameter_sweep(build, alphas; backend = CPUBackend(), stop = Steps(100))

        @test [r.scenario for r in gpu] == alphas
        for (g, c) in zip(gpu, cpu)
            @test g.observation.centre ≈ c.observation.centre rtol = 1e-4
            @test g.observation.total ≈ c.observation.total rtol = 1e-4
        end
    end
end
