@testset "source term types" begin
    @test Heat2D(nx = 8).source === NoSource()
    @test conserves_state(Heat2D(nx = 8))

    # A source adds heat the domain did not start with, so nothing is conserved
    # even under an insulating boundary.
    driven = Heat2D(nx = 8; source = UniformSource(0.1f0))
    @test !conserves_state(driven)
    @test conserves_state(driven.boundary)      # the boundary itself still is

    pattern = zeros(Float32, 8, 8)
    @test Heat2D(nx = 8; source = PatternSource(pattern)).source isa PatternSource
    @test_throws DimensionMismatch Heat2D(nx = 8; source = PatternSource(zeros(Float32, 4, 4)))
end

@testset "uniform source adds exactly rate*dt per cell per step" begin
    # The total after n steps must be cells * rate * dt * n, exactly. This pins
    # down the source scaling: a source applied without the dt factor, or applied
    # twice, or skipped on the first step, all fail here.
    for bc in (Neumann(), Periodic())
        nx, ny, rate, dt, steps = 16, 12, 0.5f0, 0.1f0, 200
        model = Heat2D(nx = nx, ny = ny; dt = dt, boundary = bc,
                       source = UniformSource(rate))
        metrics = run!(model; steps = steps)
        @test metrics.total_state ≈ nx * ny * rate * dt * steps rtol = 1e-5
    end

    # A negative rate is a loss.
    model = Heat2D(nx = 10, ny = 10; source = UniformSource(-0.2f0))
    initialize_peak!(model.field, 100.0f0)
    metrics = run!(model; steps = 100)
    @test metrics.total_state ≈ 100 - 10 * 10 * 0.2 * 0.1 * 100 rtol = 1e-4
end

@testset "pattern source is scaled by the pattern" begin
    pattern = zeros(Float32, 12, 12)
    pattern[3, 4] = 1.0f0
    pattern[9, 8] = 2.0f0

    model = Heat2D(nx = 12, ny = 12; source = PatternSource(pattern, 1.0f0))
    metrics = run!(model; steps = 150)
    # Total injected is dt * steps * sum(pattern) * rate.
    @test metrics.total_state ≈ 0.1 * 150 * 3.0 rtol = 1e-4

    # Halving the rate halves the injected total.
    half = Heat2D(nx = 12, ny = 12; source = PatternSource(pattern, 0.5f0))
    @test run!(half; steps = 150).total_state ≈ 0.1 * 150 * 3.0 * 0.5 rtol = 1e-4

    # Zero pattern is the same as no source at all.
    blank = Heat2D(nx = 12, ny = 12; source = PatternSource(zeros(Float32, 12, 12), 5.0f0))
    @test run!(blank; steps = 50).total_state ≈ 0.0 atol = 1e-6
end

@testset "Dirichlet cells do not receive the source" begin
    # A pinned edge must stay pinned. If the source were applied to fixed cells,
    # the boundary would drift away from the value the model says it holds.
    model = Heat2D(nx = 16, ny = 16; boundary = Dirichlet(4.0f0),
                   source = UniformSource(10.0f0))
    run!(model; steps = 100)
    u = Array(state(model))
    @test all(≈(4.0f0), u[1, :])
    @test all(≈(4.0f0), u[end, :])
    @test all(≈(4.0f0), u[:, 1])
    @test all(≈(4.0f0), u[:, end])
    @test u[8, 8] > 4.0f0        # the interior did receive it
end

@testset "time-varying boundary" begin
    @test !is_driven(Heat2D(nx = 8))
    @test !is_driven(Dirichlet(1.0f0))
    @test is_driven(Dirichlet(t -> 1.0f0))
    @test is_driven(Heat2D(nx = 8; boundary = Dirichlet(t -> 1.0f0)))

    # The edge must track the drive. dt = 0.1, so after n steps the last update
    # was evaluated at t = (n-1)*dt.
    schedule(t) = 10.0f0 * t
    model = Heat2D(nx = 12, ny = 12; dt = 0.1f0, boundary = Dirichlet(schedule))
    run!(model; steps = 50)
    @test Array(state(model))[1, 1] ≈ schedule(49 * 0.1f0) rtol = 1e-4

    # Continuing the run continues the clock rather than restarting it.
    run!(model; steps = 50)
    @test Array(state(model))[1, 1] ≈ schedule(49 * 0.1f0) rtol = 1e-4
end

@testset "time-varying source" begin
    @test is_driven(UniformSource(t -> 1.0f0))
    @test !is_driven(UniformSource(1.0f0))
    @test is_driven(PatternSource(zeros(Float32, 4, 4), t -> 1.0f0))
    @test !is_driven(PatternSource(zeros(Float32, 4, 4), 1.0f0))

    # A heater switched off half way injects half as much.
    gated(t) = t < 5.0 ? 1.0f0 : 0.0f0
    model = Heat2D(nx = 10, ny = 10; dt = 0.1f0, source = UniformSource(gated))
    metrics = run!(model; steps = 100)          # t runs 0.0 .. 9.9, on for 50 steps
    @test metrics.total_state ≈ 10 * 10 * 1.0 * 0.1 * 50 rtol = 1e-4
end

@testset "TimeSeries" begin
    series = TimeSeries([0.0, 5.0, 10.0], [0.0, 100.0, 0.0])
    @test series(0.0) ≈ 0.0
    @test series(2.5) ≈ 50.0
    @test series(5.0) ≈ 100.0
    @test series(7.5) ≈ 50.0
    @test series(10.0) ≈ 0.0
    @test length(series) == 3
    @test extrema(series) == (0.0, 10.0)

    # Clamping outside the sampled range.
    @test series(-1.0) ≈ 0.0
    @test series(99.0) ≈ 0.0

    strict = TimeSeries([0.0, 1.0], [5.0, 6.0]; extrapolate = :error)
    @test strict(0.5) ≈ 5.5
    @test strict(0.0) ≈ 5.0          # endpoints are inside the range
    @test strict(1.0) ≈ 6.0
    @test_throws ArgumentError strict(-0.001)
    @test_throws ArgumentError strict(1.001)

    @test_throws ArgumentError TimeSeries([0.0], [1.0])                    # too short
    @test_throws ArgumentError TimeSeries([0.0, 1.0], [1.0])               # length mismatch
    @test_throws ArgumentError TimeSeries([1.0, 0.0], [1.0, 2.0])          # unsorted
    @test_throws ArgumentError TimeSeries([0.0, 0.0], [1.0, 2.0])          # duplicated
    @test_throws ArgumentError TimeSeries([0.0, 1.0], [1.0, 2.0]; extrapolate = :hold)

    # Driving a boundary from sampled data.
    model = Heat2D(nx = 10, ny = 10; dt = 0.1f0,
                   boundary = Dirichlet(TimeSeries([0.0, 10.0], [0.0, 100.0])))
    run!(model; steps = 50)                      # last update at t = 4.9 -> 49.0
    @test Array(state(model))[1, 1] ≈ 49.0f0 rtol = 1e-3
end

@testset "forcing agrees across backends" begin
    pattern = zeros(Float32, 24, 24)
    pattern[6, 6] = 1.0f0
    pattern[18, 12] = 2.5f0

    configurations = [
        "uniform"          => (; source = UniformSource(0.3f0)),
        "uniform driven"   => (; source = UniformSource(t -> 0.3f0 * (1 + sin(t)))),
        "pattern"          => (; source = PatternSource(copy(pattern), 0.7f0)),
        "pattern driven"   => (; source = PatternSource(copy(pattern), t -> t < 3 ? 1.0f0 : 0.2f0)),
        "driven boundary"  => (; boundary = Dirichlet(t -> 5.0f0 * sin(t))),
        "boundary+source"  => (; boundary = Dirichlet(t -> 2.0f0),
                                 source = PatternSource(copy(pattern), 0.5f0)),
    ]

    for (name, kwargs) in configurations
        @testset "$name" begin
            reference = Heat2D(nx = 24, ny = 24; kwargs...)
            initialize_peak!(reference.field, 50.0f0)
            run!(reference; backend = CPUBackend(), steps = 200)
            expected = Array(state(reference))

            for backend in (CPUBackend(threaded = true), KernelBackend())
                model = Heat2D(nx = 24, ny = 24; kwargs...)
                initialize_peak!(model.field, 50.0f0)
                run!(model; backend = backend, steps = 200)
                @test Array(state(model)) == expected
            end

            for (gpu_name, backend) in GPU_BACKENDS
                model = Heat2D(nx = 24, ny = 24; kwargs...)
                initialize_peak!(model.field, 50.0f0)
                metrics = run!(model; backend = backend, steps = 200)
                @test Array(state(model)) ≈ expected rtol = 1e-4 atol = 1e-5
                @test metrics.transferred_bytes > 0
            end
        end
    end
end

@testset "pattern source travels to the device" begin
    for (name, backend) in GPU_BACKENDS
        pattern = zeros(Float32, 16, 16)
        pattern[4, 4] = 3.0f0
        model = Heat2D(nx = 16, ny = 16; source = PatternSource(pattern, 1.0f0))

        resident = to_backend(model, backend)
        @test !(resident.source.pattern isa Array)      # moved, not left on the host
        @test Array(resident.source.pattern) == pattern

        # A resident model costs no transfer, pattern included.
        metrics = run!(resident; backend = backend, steps = 50)
        @test metrics.transferred_bytes == 0

        reference = Heat2D(nx = 16, ny = 16; source = PatternSource(pattern, 1.0f0))
        run!(reference; backend = CPUBackend(), steps = 50)
        @test Array(state(resident)) ≈ Array(state(reference)) rtol = 1e-4 atol = 1e-5

        # The upload of a non-resident model must account for the pattern too.
        fresh = Heat2D(nx = 16, ny = 16; source = PatternSource(pattern, 1.0f0))
        cold = run!(fresh; backend = backend, steps = 1)
        @test cold.transferred_bytes >= 3 * sizeof(Float32) * 16 * 16   # field up+down, pattern up
    end
end

@testset "an additive source does not change the stability limit" begin
    # The source is additive and independent of u, so it does not enter the
    # von Neumann analysis. Same CFL number, same limit.
    plain = Heat2D(nx = 8, dt = 0.1f0)
    forced = Heat2D(nx = 8, dt = 0.1f0; source = UniformSource(1000.0f0))
    @test cfl_number(plain) == cfl_number(forced)
    @test is_stable(forced)

    # It can still make the solution grow without bound. That is physics, not
    # instability, and the field stays finite and smooth.
    run!(forced; steps = 500)
    @test all(isfinite, state(forced))
    @test sum_state(forced) > 0

    @test_throws ArgumentError Heat2D(nx = 8, dt = 10.0f0; source = UniformSource(1.0f0))
end

@testset "step! without a time argument evaluates the drive at zero" begin
    # Documented behaviour: calling step! directly on a driven model without a
    # time silently gives constant forcing. run! always passes the real clock.
    model = Heat2D(nx = 8, ny = 8; boundary = Dirichlet(t -> Float32(t)))
    step!(CPUBackend(), model)
    @test Array(state(model))[1, 1] == 0.0f0

    step!(CPUBackend(), model, 7.0)
    @test Array(state(model))[1, 1] == 7.0f0
end
