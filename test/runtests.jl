using Test
using VisuTwinSim

@testset "Field2D" begin
    field = Field2D(8, 6)
    @test size(field) == (8, 6)
    initialize_peak!(field, 42.0f0)
    @test center_value(field) == 42.0f0
    @test sum_state(field) == 42.0f0
end

@testset "Heat2D CPU" begin
    model = Heat2D(nx = 32, ny = 32)
    initialize_peak!(model.field, 100.0f0)
    metrics = run!(model; backend = CPUBackend(), steps = 25)
    @test metrics.backend == :cpu
    @test metrics.steps == 25
    @test isapprox(metrics.last_reduction, 100.0; atol = 1.0f-3)
    @test center_value(model) < 100.0f0
end

@testset "Simulation wrapper" begin
    model = Heat2D(nx = 16, ny = 16)
    initialize_peak!(model.field, 10.0f0)
    sim = Simulation(model; backend = CPUBackend(), stop = Steps(4))
    metrics = run!(sim)
    @test metrics.steps == 4
    @test isapprox(sum_state(model), 10.0f0; atol = 1.0f-4)
end

@testset "Random walk ensemble" begin
    positions = random_walk_ensemble(trajectories = 128, steps = 16, seed = 7)
    @test length(positions) == 128
    @test all(isfinite, positions)
end
