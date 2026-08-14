# Note on naming: callbacks receive `(model, progress)`. Calling the second
# argument `state` would shadow the exported `state(model)` accessor inside the
# closure, which is a confusing error to hit in a lab.

change_interval_of(condition) = VisuTwinSim.change_interval(condition)

@testset "Steps validates its argument" begin
    # The original outer-constructor check never ran: the compiler-generated
    # Steps(::Int) is more specific than Steps(::Integer), so every literal
    # bypassed the validation.
    @test Steps(5).count == 5
    @test Steps(0).count == 0
    @test_throws ArgumentError Steps(-1)
    @test_throws ArgumentError Steps(-5)
    @test_throws ArgumentError Steps(Int32(-3))
    @test Steps(Int32(7)).count == 7
end

@testset "stop conditions" begin
    @test_throws ArgumentError UntilTime(-1.0)
    @test_throws ArgumentError WallClock(0)
    @test_throws ArgumentError Converged(0.0)
    @test_throws ArgumentError Converged(1e-6; check_every = 0)

    build() = (m = Heat2D(nx = 32, ny = 32); initialize_peak!(m.field, 100.0f0); m)

    @testset "Steps" begin
        metrics = run!(Simulation(build(); stop = Steps(37)))
        @test metrics.steps == 37
        @test metrics.stopped_by === :steps
    end

    @testset "integer shorthand" begin
        @test run!(Simulation(build(); stop = 12)).steps == 12
    end

    @testset "UntilTime is independent of dt" begin
        coarse = Heat2D(nx = 32, ny = 32, dt = 0.1f0)
        initialize_peak!(coarse.field, 100.0f0)
        fine = Heat2D(nx = 32, ny = 32, dt = 0.05f0)
        initialize_peak!(fine.field, 100.0f0)

        a = run!(Simulation(coarse; stop = UntilTime(4.0)))
        b = run!(Simulation(fine; stop = UntilTime(4.0)))
        @test a.steps == 40
        @test b.steps == 80
        @test a.simulated_time ≈ 4.0 rtol = 1e-5
        @test b.simulated_time ≈ 4.0 rtol = 1e-5
        @test a.stopped_by === :time
        @test b.stopped_by === :time
    end

    @testset "Converged reaches steady state" begin
        model = build()
        metrics = run!(Simulation(model; stop = AnyOf(Converged(1.0f-5), Steps(500_000))))
        @test metrics.stopped_by === :converged
        # The Neumann steady state is the uniform mean.
        field = Array(state(model))
        @test maximum(field) - minimum(field) < 1e-1
        @test metrics.total_state ≈ 100.0 rtol = 1e-3
    end

    @testset "WallClock" begin
        metrics = run!(Simulation(build(); stop = AnyOf(WallClock(0.05), Steps(50_000_000))))
        @test metrics.stopped_by === :wallclock
        @test metrics.steps > 0
    end

    @testset "AnyOf reports which condition fired" begin
        @test run!(Simulation(build(); stop = AnyOf(Steps(10), Converged(1.0f-12)))).stopped_by === :steps
        @test change_interval_of(AnyOf(Steps(10), Converged(1.0f-6; check_every = 4))) == 4
        @test change_interval_of(Steps(10)) == typemax(Int)
    end
end

@testset "callbacks" begin
    build() = (m = Heat2D(nx = 24, ny = 24); initialize_peak!(m.field, 100.0f0); m)

    @testset "fire on schedule" begin
        seen = Int[]
        run!(Simulation(build(); stop = Steps(50));
             callback = (model, progress) -> (push!(seen, progress.step); nothing),
             callback_every = 10)
        @test seen == [10, 20, 30, 40, 50]
    end

    @testset "see host-visible state" begin
        totals = Float64[]
        run!(Simulation(build(); stop = Steps(20));
             callback = (model, progress) -> (push!(totals, sum(Array(state(model)))); nothing),
             callback_every = 5)
        @test length(totals) == 4
        @test all(t -> isapprox(t, 100.0; rtol = 1e-4), totals)
    end

    @testset "can stop the run" begin
        metrics = run!(Simulation(build(); stop = Steps(1000));
                       callback = (model, progress) -> progress.step >= 30 ? :stop : nothing,
                       callback_every = 10)
        @test metrics.steps == 30
        @test metrics.stopped_by === :callback
    end

    @testset "see device state on a GPU too" begin
        for (name, backend) in GPU_BACKENDS
            totals = Float64[]
            model = build()
            run!(Simulation(model; backend = backend, stop = Steps(20));
                 callback = (m, progress) -> (push!(totals, sum(Array(state(m)))); nothing),
                 callback_every = 5)
            @test length(totals) == 4
            @test all(t -> isapprox(t, 100.0; rtol = 1e-3), totals)
        end
    end

    @test_throws ArgumentError run!(Simulation(build(); stop = Steps(1)); callback_every = 0)
end

@testset "real-time pacing" begin
    model = Heat2D(nx = 16, ny = 16, dt = 0.1f0)
    initialize_peak!(model.field, 100.0f0)
    # 30 steps of dt = 0.1 is 3.0 units of simulated time; at 30x that is 0.1 s.
    metrics = run!(Simulation(model; stop = Steps(30)); realtime_factor = 30.0)
    @test metrics.elapsed_seconds > 0.08
    @test metrics.elapsed_seconds < 2.0
    # Time spent waiting must not be billed as compute.
    @test metrics.compute_seconds < metrics.elapsed_seconds
end

@testset "step_limit guards an unreachable condition" begin
    model = Heat2D(nx = 16, ny = 16)
    initialize_peak!(model.field, 100.0f0)
    metrics = @test_logs (:warn,) run!(Simulation(model; stop = Converged(1.0f-30));
                                       step_limit = 100)
    @test metrics.stopped_by === :step_limit
    @test metrics.steps == 100
end

@testset "zero steps is a valid run" begin
    model = Heat2D(nx = 8, ny = 8)
    initialize_peak!(model.field, 100.0f0)
    metrics = run!(Simulation(model; stop = Steps(0)))
    @test metrics.steps == 0
    @test metrics.total_state ≈ 100.0
    @test center_value(model) == 100.0f0
end

@testset "Simulation displays its configuration" begin
    text = sprint(show, MIME"text/plain"(), Simulation(Heat2D(nx = 8); stop = Steps(5)))
    @test occursin("Simulation", text)
    @test occursin("cpu", text)
end
