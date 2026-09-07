@testset "Heat2DParams promotion" begin
    # Mixed argument types used to be a MethodError, because every @kwdef
    # default was Float32 and the struct requires one shared element type.
    params = Heat2DParams(alpha = 0.2, dt = 0.05, dx = 1.0, dy = 1.0)
    @test eltype(params) == Float64

    mixed = Heat2DParams(0.2, 0.1f0, 1.0f0, 1)
    @test eltype(mixed) == Float64
    @test mixed.alpha == 0.2

    @test eltype(Heat2DParams()) == Float32
    @test eltype(Heat2DParams(alpha = 1, dt = 1, dx = 1, dy = 1)) == Int
end

@testset "model element type follows the field" begin
    @test eltype(Heat2D(nx = 8, initial = 0.0f0)) == Float32
    @test eltype(Heat2D(nx = 8, initial = 0.0)) == Float64
    # The failing call from the guidelines: one Float64 keyword on a Float32 field.
    model = Heat2D(nx = 8, alpha = 0.2)
    @test eltype(model) == Float32
    @test model.params.alpha === 0.2f0
    @test eltype(Heat2D(nx = 8, initial = 0.0, alpha = 0.2).params) == Float64
end

@testset "CFL stability" begin
    stable = Heat2DParams(alpha = 0.15f0, dt = 0.1f0, dx = 1.0f0, dy = 1.0f0)
    @test cfl_number(stable) ≈ 0.03f0
    @test is_stable(stable)

    unstable = Heat2DParams(alpha = 0.15f0, dt = 10.0f0, dx = 1.0f0, dy = 1.0f0)
    @test !is_stable(unstable)

    # A configuration that would diverge is rejected at construction rather than
    # silently producing NaN a few dozen steps later.
    @test_throws ArgumentError Heat2D(nx = 16, dt = 10.0f0)
    @test_throws ArgumentError Heat2D(nx = 16, alpha = 100.0f0)

    # ...unless the lab is deliberately demonstrating the instability.
    diverging = Heat2D(nx = 16, dt = 10.0f0, check_stability = false)
    @test !is_stable(diverging)
    initialize_peak!(diverging.field, 100.0f0)
    run!(diverging; steps = 200)
    @test !all(isfinite, state(diverging))

    # max_stable_dt is exactly the boundary of the admissible region.
    params = Heat2DParams(alpha = 0.25f0, dt = 0.01f0, dx = 0.5f0, dy = 2.0f0)
    @test is_stable(Heat2DParams(alpha = 0.25f0, dt = max_stable_dt(params), dx = 0.5f0, dy = 2.0f0))
    @test cfl_number(Heat2DParams(alpha = 0.25f0, dt = max_stable_dt(params), dx = 0.5f0, dy = 2.0f0)) ≈ 0.5f0
end

@testset "boundary conditions" begin
    @test conserves_state(Neumann())
    @test conserves_state(Periodic())
    @test !conserves_state(Dirichlet(0.0f0))
    @test Heat2D(nx = 8).boundary === Neumann()      # conserving by default

    # Dirichlet values are converted to the field element type so a Float64
    # literal cannot drag the kernel out of Float32.
    model = Heat2D(nx = 8; boundary = Dirichlet(5.0))
    @test model.boundary isa Dirichlet{Float32}
    @test model.boundary.value === 5.0f0
end

@testset "conservation follows the boundary condition" begin
    # Long enough for heat to reach the edge — the previous 25-step test could
    # not have detected a leak.
    for bc in (Neumann(), Periodic())
        model = Heat2D(nx = 40, ny = 40; boundary = bc)
        initialize_peak!(model.field, 100.0f0)
        metrics = run!(model; steps = 20_000)
        @test metrics.total_state ≈ 100.0 rtol = 1e-4
        # Fully mixed: every cell approaches the mean.
        @test maximum(state(model)) - minimum(state(model)) < 1e-2
    end

    leaky = Heat2D(nx = 40, ny = 40; boundary = Dirichlet(0.0f0))
    initialize_peak!(leaky.field, 100.0f0)
    metrics = run!(leaky; steps = 20_000)
    @test metrics.total_state < 90.0          # heat genuinely leaves the domain
    @test all(iszero, state(leaky)[1, :])     # the edge is held at the fixed value
end

@testset "Dirichlet holds a non-zero edge" begin
    model = Heat2D(nx = 24, ny = 24; boundary = Dirichlet(10.0f0))
    # Long enough to actually relax: the slowest mode of a 24-cell domain decays
    # at roughly 5e-4 per step, so 5000 steps still leaves the centre ~10% short.
    metrics = run!(model; steps = 30_000)
    edges = state(model)
    @test all(≈(10.0f0), edges[1, :])
    @test all(≈(10.0f0), edges[end, :])
    @test all(≈(10.0f0), edges[:, 1])
    # Started cold, so the interior fills up towards the boundary value.
    @test center_value(model) ≈ 10.0f0 atol = 1e-2
end

@testset "type stability and allocations" begin
    model = Heat2D(nx = 32, ny = 32)
    params = model.params
    @test @inferred(cfl_number(params)) isa Float32
    @test @inferred(TwinSim.diffusion_coefficients(params)) isa NTuple{2,Float32}
    @test @inferred(TwinSim.bytes_per_cell(model)) == 8
    @test @inferred(step!(CPUBackend(), model)) === model

    # The stepping loop must not allocate: a per-step allocation shows up as GC
    # pressure that dominates the measurement on large grids.
    backend = CPUBackend()
    step!(backend, model)
    @test @allocated(step!(backend, model)) == 0
end
