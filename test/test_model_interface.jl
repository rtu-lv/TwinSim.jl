# A model defined here, in the test suite, using only the public interface —
# standing in for a contributed component defined in someone else's package.
#
# Deliberately unlike Heat2D: one dimension, no Field2D, no boundary condition,
# no source term, an analytical solution. If the runtime had any remaining
# assumption that a model looks like heat diffusion, this would fail.

import TwinSim: step!, state, timestep, clock, flops_per_cell, move_to_device

"""Exponential decay, du/dt = -rate * u. Exact solution u0*(1 - rate*dt)^n."""
struct Decay{T,A<:AbstractVector{T}} <: AbstractModel
    u::A
    rate::T
    dt::T
    clock::Base.RefValue{Float64}
end

Decay(n::Integer = 1000; rate = 0.1f0, dt = 0.05f0, initial = 1.0f0) =
    Decay(fill(initial, n), rate, dt, Ref(0.0))

state(m::Decay) = m.u
timestep(m::Decay) = m.dt
clock(m::Decay) = m.clock
flops_per_cell(::Decay) = 3

function step!(::CPUBackend, m::Decay{T}, t = zero(T)) where {T}
    @inbounds @simd for i in eachindex(m.u)
        m.u[i] -= m.dt * m.rate * m.u[i]
    end
    return m
end

@testset "a foreign model satisfies the interface" begin
    model = Decay(100)
    @test model isa AbstractModel
    @test check_model_interface(model; verbose = false)

    @test state(model) === model.u
    @test timestep(model) == 0.05f0
    @test clock(model) isa Base.RefValue{Float64}
    @test cells(model) == 100
    @test bytes_per_cell(model) == 2 * sizeof(Float32)   # default
    @test flops_per_cell(model) == 3                     # overridden
    @test sum_state(model) ≈ 100.0f0                     # default
    @test eltype(model) == Float32
end

@testset "check_model_interface reports what is missing" begin
    struct Incomplete <: AbstractModel end
    @test !check_model_interface(Incomplete(); verbose = false)

    # And it says so out loud, naming the methods rather than failing later.
    buffer = IOBuffer()
    check_model_interface(Incomplete(); verbose = true, io = buffer)
    text = String(take!(buffer))
    @test occursin("does not satisfy", text)
    @test occursin("state(model) failed", text)
end

@testset "the runtime works on a foreign model" begin
    model = Decay(1000; rate = 0.1f0, dt = 0.05f0)
    metrics = run!(model; steps = 200)

    # Exact: each step multiplies by (1 - dt*rate).
    expected = 1000 * Float64(1 - 0.05f0 * 0.1f0)^200
    @test metrics.total_state ≈ expected rtol = 1e-4

    @test metrics.cells == 1000
    @test metrics.steps == 200
    @test metrics.simulated_time ≈ 10.0 rtol = 1e-5
    @test metrics.precision === Float32
    @test mlups(metrics) > 0
    @test arithmetic_intensity(metrics) ≈ 3 / 8      # 3 flops, 8 bytes
end

@testset "the clock persists for a foreign model" begin
    model = Decay(50)
    run!(model; steps = 30)
    @test simulated_time(model) ≈ 1.5 rtol = 1e-5
    run!(model; steps = 30)
    @test simulated_time(model) ≈ 3.0 rtol = 1e-5
    reset_clock!(model)
    @test simulated_time(model) == 0.0
end

@testset "stop conditions work on a foreign model" begin
    @test run!(Simulation(Decay(50); stop = Steps(17))).steps == 17
    @test run!(Simulation(Decay(50); stop = UntilTime(2.0))).steps == 40
    @test run!(Simulation(Decay(50); stop = ForDuration(1.0))).steps == 20

    converged = run!(Simulation(Decay(50); stop = AnyOf(Converged(1.0f-6), Steps(100_000))))
    @test converged.stopped_by === :converged

    timed = run!(Simulation(Decay(50); stop = AnyOf(WallClock(0.02), Steps(10_000_000))))
    @test timed.stopped_by === :wallclock
end

@testset "callbacks and recorders work on a foreign model" begin
    recorder = MetricRecorder(total = m -> sum_state(m), peak = m -> maximum(state(m)))
    run!(Simulation(Decay(200); stop = Steps(100)); callback = recorder, callback_every = 25)

    @test length(recorder) == 4
    @test issorted(recorder[:total]; rev = true)      # decay
    @test issorted(recorder[:peak]; rev = true)

    stopped = run!(Simulation(Decay(200); stop = Steps(1000));
                   callback = (m, p) -> p.step >= 40 ? :stop : nothing,
                   callback_every = 10)
    @test stopped.steps == 40
    @test stopped.stopped_by === :callback
end

@testset "sweeps and the twin loop work on a foreign model" begin
    results = parameter_sweep(Float32[0.05, 0.1, 0.2]; stop = Steps(100),
                              observe = (m, _) -> (; total = Float64(sum_state(m)))) do rate
        Decay(100; rate = rate)
    end
    @test length(results) == 3
    # A faster decay rate leaves less behind.
    @test issorted([r.observation.total for r in results]; rev = true)

    loop = TwinLoop(validate = (m, o, t) -> (; total = Float64(sum_state(m))),
                    decide = (m, c, t) -> c.total > 50 ? :high : :low,
                    steps_per_window = 20)
    log = twin_run!(loop, Decay(100), ones(6))
    @test length(log) == 6
    @test all(d -> d in (:high, :low), log.decisions)
end

@testset "a CPU-only model refuses GPU backends clearly" begin
    for (name, backend) in GPU_BACKENDS
        err = try
            run!(Decay(50); backend = backend, steps = 1)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        message = sprint(showerror, err)
        # It must name the model and the method to implement, not fail obscurely.
        @test occursin("Decay", message)
        @test occursin("move_to_device", message)
    end
end

@testset "Heat2D satisfies its own interface" begin
    model = Heat2D(nx = 16, ny = 16)
    @test model isa AbstractModel
    @test check_model_interface(model; verbose = false)
    @test timestep(model) == 0.1f0
    @test clock(model) === model.clock
    @test cells(model) == 256
    @test flops_per_cell(model) == 10          # overridden, not the default 0
    @test bytes_per_cell(model) == 8
end
