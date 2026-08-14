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
    #
    # The obvious check here — that the edge equals schedule(49*dt) again — is
    # vacuous: it holds whether the clock continued or restarted, because both
    # end 49 steps after their own start. Assert on the *clock* instead.
    @test simulated_time(model) ≈ 5.0 rtol = 1e-4
    run!(model; steps = 50)
    @test simulated_time(model) ≈ 10.0 rtol = 1e-4
    @test Array(state(model))[1, 1] ≈ schedule(99 * 0.1f0) rtol = 1e-4
end

@testset "the simulated clock persists across run! calls" begin
    # Regression: run! used to restart the clock every call, so a driven model
    # advanced in windows replayed the same slice of its drive forever.
    seen = Float64[]
    probe(t) = (push!(seen, Float64(t)); 0.0f0)
    model = Heat2D(nx = 8, ny = 8, dt = 0.1f0; boundary = Dirichlet(probe))

    run!(model; steps = 3)
    run!(model; steps = 3)
    run!(model; steps = 3)

    @test length(seen) == 9
    @test issorted(seen)
    @test seen[1] ≈ 0.0
    @test seen[4] ≈ 0.3 rtol = 1e-4      # the second call continues
    @test seen[7] ≈ 0.6 rtol = 1e-4
    @test simulated_time(model) ≈ 0.9 rtol = 1e-4

    # Two runs of 50 see the same drive as one run of 100.
    chained = Heat2D(nx = 8, ny = 8, dt = 0.1f0; boundary = Dirichlet(t -> Float32(10t)))
    run!(chained; steps = 50); run!(chained; steps = 50)
    single = Heat2D(nx = 8, ny = 8, dt = 0.1f0; boundary = Dirichlet(t -> Float32(10t)))
    run!(single; steps = 100)
    @test Array(state(chained)) == Array(state(single))

    # metrics report what the individual run advanced, not the absolute clock.
    @test run!(chained; steps = 10).simulated_time ≈ 1.0 rtol = 1e-4
    @test simulated_time(chained) ≈ 11.0 rtol = 1e-4

    reset_clock!(chained, 2.5)
    @test simulated_time(chained) ≈ 2.5
    reset_clock!(chained)
    @test simulated_time(chained) == 0.0
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

@testset "SampledSeries" begin
    series = SampledSeries([0.0, 5.0, 10.0], [0.0, 100.0, 0.0])
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

    strict = SampledSeries([0.0, 1.0], [5.0, 6.0]; extrapolate = :error)
    @test strict(0.5) ≈ 5.5
    @test strict(0.0) ≈ 5.0          # endpoints are inside the range
    @test strict(1.0) ≈ 6.0
    @test_throws ArgumentError strict(-0.001)
    @test_throws ArgumentError strict(1.001)

    @test_throws ArgumentError SampledSeries([0.0], [1.0])                    # too short
    @test_throws ArgumentError SampledSeries([0.0, 1.0], [1.0])               # length mismatch
    @test_throws ArgumentError SampledSeries([1.0, 0.0], [1.0, 2.0])          # unsorted
    @test_throws ArgumentError SampledSeries([0.0, 0.0], [1.0, 2.0])          # duplicated
    @test_throws ArgumentError SampledSeries([0.0, 1.0], [1.0, 2.0]; extrapolate = :hold)

    # Driving a boundary from sampled data.
    model = Heat2D(nx = 10, ny = 10; dt = 0.1f0,
                   boundary = Dirichlet(SampledSeries([0.0, 10.0], [0.0, 100.0])))
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

# ---------------------------------------------------------------------------
# State-dependent forcing, and what it does to stability
# ---------------------------------------------------------------------------

@testset "ProportionalSource relaxes towards its target" begin
    model = Heat2D(nx = 16, ny = 16, initial = 0.0f0;
                   source = ProportionalSource(20.0f0, 1.0f0))
    run!(model; steps = 500)
    u = Array(state(model))
    @test all(≈(20.0f0; atol = 1e-3), u)          # every cell reaches the target

    # Approach is exponential with rate `gain`: after time t the remaining gap is
    # exp(-gain*t) of the original. Uniform field, so diffusion does nothing.
    gap(steps, gain) = begin
        m = Heat2D(nx = 8, ny = 8, initial = 0.0f0, dt = 0.01f0;
                   source = ProportionalSource(10.0f0, gain))
        run!(m; steps = steps)
        10.0 - Array(state(m))[4, 4]
    end
    @test gap(100, 1.0f0) ≈ 10 * (1 - 0.01)^100 rtol = 1e-3   # discrete, not exp()
    @test gap(100, 2.0f0) ≈ 10 * (1 - 0.02)^100 rtol = 1e-3

    @test_throws ArgumentError ProportionalSource(1.0f0, -0.5f0)   # positive feedback
end

@testset "a state-dependent source tightens the stability limit" begin
    # This is the hole that `cfl_number` alone could not see: the diffusion term
    # is comfortably stable, and the model still diverges because of the gain.
    alpha, dt = 0.15f0, 0.1f0
    diffusion_only = Heat2DParams(alpha = alpha, dt = dt, dx = 1.0f0, dy = 1.0f0)
    @test cfl_number(diffusion_only) ≈ 0.03f0
    @test is_stable(diffusion_only)

    # |G| <= 1 requires dt*g + 4*(cx+cy) <= 2, i.e. cfl + dt*g/4 <= 1/2.
    limit = (0.5 - cfl_number(diffusion_only)) * 4 / dt          # 18.8
    @test limit ≈ 18.8 rtol = 1e-3

    for gain in Float32[1, 10, 18]
        model = Heat2D(nx = 8, alpha = alpha, dt = dt;
                       source = ProportionalSource(1.0f0, gain))
        @test is_stable(model)
        @test stability_number(model) ≈ cfl_number(diffusion_only) + dt * gain / 4
        # cfl_number keeps its conventional meaning: diffusion only.
        @test cfl_number(model) ≈ cfl_number(diffusion_only)
    end

    # Above the limit the constructor refuses, where before it accepted happily
    # and produced NaN.
    for gain in Float32[19, 25, 40]
        @test_throws ArgumentError Heat2D(nx = 8, alpha = alpha, dt = dt;
                                          source = ProportionalSource(1.0f0, gain))
    end

    # And the refusal is justified: forced through, it really does diverge.
    diverging = Heat2D(nx = 16, alpha = alpha, dt = dt, check_stability = false;
                       source = ProportionalSource(20.0f0, 40.0f0))
    initialize_peak!(diverging.field, 5.0f0)
    run!(diverging; steps = 400)
    @test !all(isfinite, state(diverging))

    # max_stable_dt accounts for the gain, and its answer is exactly admissible.
    source = ProportionalSource(1.0f0, 10.0f0)
    best = max_stable_dt(Heat2DParams(alpha = alpha, dt = dt, dx = 1.0f0, dy = 1.0f0), source)
    @test stability_number(Heat2DParams(alpha = alpha, dt = best, dx = 1.0f0, dy = 1.0f0),
                           source) ≈ 0.5 rtol = 1e-5
    @test is_stable(Heat2D(nx = 8, alpha = alpha, dt = best; source = source))

    # The error message has to say the source is implicated, or it sends the
    # student to change dt when lowering the gain is the better fix.
    message = try
        Heat2D(nx = 8, alpha = alpha, dt = dt; source = ProportionalSource(1.0f0, 40.0f0))
        ""
    catch err
        sprint(showerror, err)
    end
    @test occursin("feedback coefficient", message)
    @test occursin("gain", message)
end

@testset "additive sources still do not affect stability" begin
    params = Heat2DParams(alpha = 0.15f0, dt = 0.1f0, dx = 1.0f0, dy = 1.0f0)
    for source in (NoSource(), UniformSource(1000.0f0),
                   PatternSource(ones(Float32, 8, 8), 1000.0f0))
        @test feedback_coefficient(source) == 0
        @test stability_number(params, source) ≈ cfl_number(params)
    end
end

@testset "sources combine additively" begin
    pattern = zeros(Float32, 10, 10)
    pattern[5, 5] = 1.0f0

    combined = UniformSource(0.2f0) + PatternSource(pattern, 1.0f0)
    @test combined isa CombinedSource
    @test length(combined.sources) == 2
    @test feedback_coefficient(combined) == 0

    # The combination injects the sum of what each injects alone.
    total(source) = run!(Heat2D(nx = 10, ny = 10; source = source); steps = 100).total_state
    @test total(combined) ≈ total(UniformSource(0.2f0)) + total(PatternSource(pattern, 1.0f0)) rtol = 1e-4

    # NoSource is the identity of the sum, and stays free.
    @test NoSource() + UniformSource(1.0f0) === UniformSource(1.0f0)
    @test UniformSource(1.0f0) + NoSource() === UniformSource(1.0f0)
    @test NoSource() + NoSource() === NoSource()

    # Flattening rather than nesting.
    triple = UniformSource(1.0f0) + PatternSource(pattern, 1.0f0) + UniformSource(2.0f0)
    @test length(triple.sources) == 3

    # Feedback coefficients add, so a combination can be rejected when neither
    # part would be on its own.
    pair = ProportionalSource(1.0f0, 10.0f0) + ProportionalSource(2.0f0, 12.0f0)
    @test feedback_coefficient(pair) == 22.0f0
    @test_throws ArgumentError Heat2D(nx = 8, dt = 0.1f0; source = pair)
    @test is_stable(Heat2D(nx = 8, dt = 0.1f0; source = ProportionalSource(1.0f0, 10.0f0)))
end

@testset "order of a combination does not matter" begin
    # Applying sources in sequence rather than summing their rates would make
    # this fail as soon as one of them reads the state.
    pattern = ones(Float32, 12, 12)
    a = PatternSource(pattern, 0.4f0)
    b = ProportionalSource(5.0f0, 2.0f0)

    forward = Heat2D(nx = 12, ny = 12; source = a + b)
    backward = Heat2D(nx = 12, ny = 12; source = b + a)
    initialize_peak!(forward.field, 30.0f0)
    initialize_peak!(backward.field, 30.0f0)
    run!(forward; steps = 200)
    run!(backward; steps = 200)
    @test Array(state(forward)) == Array(state(backward))
end

@testset "ControlSignal closes the loop from a callback" begin
    pattern = zeros(Float32, 16, 16)
    pattern[8, 8] = 1.0f0

    power = ControlSignal(0.0f0)
    @test power[] == 0.0f0
    power[] = 2.5
    @test power[] === 2.5f0                      # converted to the stored type
    @test power(123.0) === 2.5f0                 # callable, ignores time

    model = Heat2D(nx = 16, ny = 16; source = PatternSource(pattern, power))
    @test is_driven(model)
    @test feedback_coefficient(model.source) == 0    # a signal is not feedback

    # Off, then on: the total only grows during the second half.
    power[] = 0.0f0
    run!(model; steps = 50)
    @test sum_state(model) ≈ 0.0f0 atol = 1e-6
    power[] = 5.0f0
    run!(model; steps = 50)
    @test sum_state(model) ≈ 50 * 0.1 * 5.0 rtol = 1e-4

    # Driven from inside a callback, which is the intended use.
    controlled = Heat2D(nx = 16, ny = 16, initial = 0.0f0;
                        source = PatternSource(ones(Float32, 16, 16), power))
    power[] = 0.0f0
    samples = Float64[]
    run!(Simulation(controlled; stop = Steps(300));
         callback_every = 10,
         callback = function (m, progress)
             mean_temp = sum(Array(state(m))) / length(m.field)
             power[] = clamp(0.5f0 * (10 - mean_temp), 0.0f0, 2.0f0)
             push!(samples, mean_temp)
             return nothing
         end)
    @test length(samples) == 30
    @test issorted(samples)                       # monotone approach to the setpoint
    @test last(samples) > 5.0                     # and it got most of the way there
    @test last(samples) < 10.5                    # without overshooting badly
end

@testset "state-dependent forcing agrees across backends" begin
    pattern = zeros(Float32, 24, 24)
    pattern[6, 6] = 1.0f0
    pattern[18, 12] = 2.5f0

    configurations = [
        "proportional"        => (; source = ProportionalSource(12.0f0, 2.0f0)),
        "proportional driven" => (; source = ProportionalSource(t -> 12.0f0 + 4sin(t), 2.0f0)),
        "pattern+relaxation"  => (; source = PatternSource(copy(pattern), 0.6f0) +
                                             ProportionalSource(0.0f0, 1.5f0)),
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
                run!(model; backend = backend, steps = 200)
                @test Array(state(model)) ≈ expected rtol = 1e-4 atol = 1e-5
            end
        end
    end
end
