using Statistics

@testset "random_walk_ensemble" begin
    positions = random_walk_ensemble(trajectories = 128, steps = 16, seed = 7)
    @test length(positions) == 128
    @test eltype(positions) == Int
    # Every walk of 16 steps ends on an even offset within [-16, 16].
    @test all(p -> -16 <= p <= 16, positions)
    @test all(iseven, positions)

    @test_throws ArgumentError random_walk_ensemble(trajectories = 0)
    @test_throws ArgumentError random_walk_ensemble(steps = -1)
    @test random_walk_ensemble(trajectories = 5, steps = 0) == zeros(Int, 5)
end

@testset "results are reproducible and thread-count independent" begin
    # A shared RNG would make results depend on how work happened to be
    # scheduled, which is the classic way parallel Monte Carlo stops being
    # reproducible.
    a = random_walk_ensemble(trajectories = 500, steps = 300, seed = 11, threaded = true)
    b = random_walk_ensemble(trajectories = 500, steps = 300, seed = 11, threaded = false)
    @test a == b

    c = random_walk_ensemble(trajectories = 500, steps = 300, seed = 12, threaded = true)
    @test a != c

    # Each trajectory's result depends only on (seed, index), so a longer run
    # reproduces the prefix of a shorter one.
    short = random_walk_ensemble(trajectories = 100, steps = 300, seed = 11)
    @test short == a[1:100]
end

@testset "statistics match the analytical random walk" begin
    steps = 4096
    positions = random_walk_ensemble(trajectories = 20_000, steps = steps, seed = 3)
    # A symmetric walk has mean 0 and variance equal to the number of steps.
    @test abs(mean(positions)) < 4 * sqrt(steps / 20_000) * 3
    @test var(positions) ≈ steps rtol = 0.05
end

@testset "step counts that are not a multiple of 64" begin
    # The implementation consumes 64 random bits at a time; the remainder path
    # is the easy thing to get wrong.
    for steps in (1, 63, 64, 65, 127, 128, 129)
        positions = random_walk_ensemble(trajectories = 400, steps = steps, seed = 5)
        @test all(p -> -steps <= p <= steps, positions)
        @test all(p -> iseven(p) == iseven(steps), positions)
        @test abs(mean(positions)) < 6 * sqrt(steps / 400)
    end
end
