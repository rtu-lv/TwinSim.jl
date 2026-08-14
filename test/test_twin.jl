@testset "nudge! pulls the state towards measurements" begin
    model = Heat2D(nx = 16, ny = 16)
    @test model.field[4, 5] == 0.0f0

    nudge!(model, [Sensor(4, 5, 100.0f0)]; gain = 0.25)
    @test model.field[4, 5] ≈ 25.0f0

    nudge!(model, [Sensor(4, 5, 100.0f0)]; gain = 1.0)     # direct insertion
    @test model.field[4, 5] ≈ 100.0f0

    nudge!(model, [Sensor(4, 5, 0.0f0)]; gain = 0.0)       # ignore the sensor
    @test model.field[4, 5] ≈ 100.0f0

    @test_throws ArgumentError nudge!(model, [Sensor(1, 1, 1.0f0)]; gain = 1.5)
    @test_throws ArgumentError nudge!(model, [Sensor(1, 1, 1.0f0)]; gain = -0.1)
    @test_throws BoundsError nudge!(model, [Sensor(99, 1, 1.0f0)])
end

@testset "assimilation keeps the twin near the observed system" begin
    # A "real" system held hot at one cell, and a twin that starts cold. Without
    # assimilation the twin never learns about the heat; with it, it tracks.
    truth = 80.0f0
    probe = (6, 6)

    blind = Heat2D(nx = 24, ny = 24)
    tracking = Heat2D(nx = 24, ny = 24)
    sensors = [Sensor(probe..., truth)]

    for _ in 1:20
        run!(blind; steps = 25)
        run!(tracking; steps = 25)
        nudge!(tracking, sensors; gain = 0.5)
    end

    @test sum_state(blind) == 0.0f0                      # nothing ever entered
    @test tracking.field[probe...] > 0.5f0 * truth       # stays near the measurement
    @test sum_state(tracking) > 10.0f0                   # heat spread into the domain
end

@testset "checkpoint round trip" begin
    path = joinpath(mktempdir(), "state.vts")

    model = Heat2D(nx = 20, ny = 12; boundary = Dirichlet(3.5f0))
    initialize_peak!(model.field, 100.0f0)
    run!(model; steps = 100)

    save_state(path, model; step = 100, simulated_time = 10.0)
    restored = load_state(path)

    @test Array(state(restored.field)) == Array(state(model))
    @test size(restored.field) == (20, 12)
    @test restored.step == 100
    @test restored.simulated_time == 10.0
    @test restored.boundary isa Dirichlet{Float32}
    @test restored.boundary.value == 3.5f0

    # A restarted run continues exactly where the original stopped.
    resumed = Heat2D(restored.field; boundary = restored.boundary)
    run!(resumed; steps = 50)
    run!(model; steps = 50)
    @test Array(state(resumed)) == Array(state(model))
end

@testset "checkpoints carry precision and boundary" begin
    dir = mktempdir()
    for (bc, T) in ((Neumann(), Float32), (Periodic(), Float64), (Dirichlet(2.0), Float64))
        path = joinpath(dir, "cp_$(nameof(typeof(bc)))_$T.vts")
        model = Heat2D(nx = 9, ny = 7, initial = zero(T); boundary = bc)
        initialize_gaussian!(model.field; sigma = 2)
        save_state(path, model)
        restored = load_state(path)
        @test eltype(restored.field) == T
        @test typeof(restored.boundary) == typeof(Heat2D(nx = 4, initial = zero(T); boundary = bc).boundary)
        @test Array(state(restored.field)) == Array(state(model))
    end
end

@testset "corrupt checkpoints are rejected" begin
    dir = mktempdir()
    bad = joinpath(dir, "bad.vts")
    write(bad, "definitely not a checkpoint file")
    @test_throws ArgumentError load_state(bad)

    truncated = joinpath(dir, "truncated.vts")
    model = Heat2D(nx = 8, ny = 8)
    save_state(truncated, model)
    open(truncated, "a") do io
        write(io, zeros(UInt8, 16))          # trailing junk
    end
    @test_throws ArgumentError load_state(truncated)
end

@testset "checkpoint_callback writes on schedule" begin
    dir = mktempdir()
    model = Heat2D(nx = 12, ny = 12)
    initialize_peak!(model.field, 100.0f0)
    run!(Simulation(model; stop = Steps(100));
         callback = checkpoint_callback(joinpath(dir, "state_%04d.vts")),
         callback_every = 25)

    written = sort(filter(endswith(".vts"), readdir(dir)))
    @test written == ["state_0025.vts", "state_0050.vts", "state_0075.vts", "state_0100.vts"]
    @test load_state(joinpath(dir, "state_0050.vts")).step == 50
end

@testset "GPU state can be checkpointed" begin
    for (name, backend) in GPU_BACKENDS
        path = joinpath(mktempdir(), "gpu.vts")
        model = Heat2D(nx = 16, ny = 16)
        initialize_peak!(model.field, 100.0f0)
        resident = to_backend(model, backend)
        run!(resident; backend = backend, steps = 50)

        save_state(path, resident; step = 50)      # copies to the host on the way out
        restored = load_state(path)
        @test Array(state(restored.field)) ≈ Array(state(resident))
    end
end
