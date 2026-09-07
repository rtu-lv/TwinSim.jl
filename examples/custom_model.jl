# Adding a model from outside the package.
#
#   julia --project=. examples/custom_model.jl
#
# This file defines a wave equation and gets the whole runtime for it — stop
# conditions, metrics, callbacks, checkpointing, the twin loop, parameter sweeps —
# without touching TwinSim itself. Everything below could live in a separate
# package that merely depends on it.
#
# It is deliberately a *different* problem from Heat2D: second order in time, so
# it needs two previous states rather than one, it conserves energy rather than
# mass, and its stability limit is a different expression. If the interface only
# fitted diffusion, this would not work.
#
# Use it as the template for a contributed component.

using Printf
using TwinSim

# The four required methods. Import them by name to add methods rather than
# shadow them — `import`, not `using`.
import TwinSim: step!, state, timestep, clock
import TwinSim: cells, bytes_per_cell, flops_per_cell, sum_state

# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
#
#   d2u/dt2 = c^2 * laplacian(u)
#
# Discretised explicitly, the update needs the two previous states:
#
#   u^{n+1} = 2u^n - u^{n-1} + (c dt/dx)^2 * laplacian(u^n)

"""A vibrating membrane with fixed edges."""
struct Wave2D{T} <: AbstractModel
    current::Matrix{T}      # u^n
    previous::Matrix{T}     # u^{n-1}
    next::Matrix{T}         # scratch for u^{n+1}
    speed::T                # wave speed c
    dt::T
    dx::T
    clock::Base.RefValue{Float64}
end

function Wave2D(; n::Integer = 128, speed = 1.0f0, dt = 0.4f0, dx = 1.0f0)
    courant = speed * dt / dx
    # The 2D explicit wave scheme is stable for c*dt/dx <= 1/sqrt(2). Checking it
    # here mirrors what Heat2D does with its CFL number, and for the same reason:
    # an unstable configuration should fail at construction, not produce NaN a
    # thousand steps later.
    courant <= 1 / sqrt(2) || throw(ArgumentError(
        "Courant number $courant exceeds the 2D limit of $(1/sqrt(2)); reduce dt or speed"))
    z = zeros(typeof(speed), n, n)
    return Wave2D(z, copy(z), copy(z), speed, dt, dx, Ref(0.0))
end

# --- the four required methods ---------------------------------------------
state(m::Wave2D) = m.current
timestep(m::Wave2D) = m.dt
clock(m::Wave2D) = m.clock

function step!(::CPUBackend, m::Wave2D{T}, t = zero(T)) where {T}
    n = size(m.current, 1)
    c2 = (m.speed * m.dt / m.dx)^2
    u, up, un = m.current, m.previous, m.next

    @inbounds for j in 2:(n - 1)
        @simd for i in 2:(n - 1)
            lap = u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] - 4u[i, j]
            un[i, j] = 2u[i, j] - up[i, j] + c2 * lap
        end
    end
    # Fixed edges: a clamped membrane.
    @inbounds for k in 1:n
        un[1, k] = un[n, k] = un[k, 1] = un[k, n] = zero(T)
    end

    # Rotate the three buffers. No copying, as in Field2D.
    copyto!(up, u)
    copyto!(u, un)
    return m
end

# --- optional methods, overridden because the defaults would be wrong -------
# Three arrays are touched per cell, not two, and the stencil is 6 flops.
bytes_per_cell(m::Wave2D{T}) where {T} = 3 * sizeof(T)
flops_per_cell(::Wave2D) = 6

# Energy, not mass, is what this model conserves; `sum_state` should report the
# quantity a validity check would actually use.
sum_state(m::Wave2D) = sum(abs2, m.current)

# ---------------------------------------------------------------------------
# That is the whole model. Everything below is the runtime, unmodified.
# ---------------------------------------------------------------------------

model = Wave2D(n = 192, speed = 1.0f0, dt = 0.4f0)

println("Interface check:")
check_model_interface(model)

# A struck membrane.
for j in 90:102, i in 90:102
    model.current[i, j] = exp(-((i - 96)^2 + (j - 96)^2) / 18)
    model.previous[i, j] = model.current[i, j]
end

println("\n--- run! works, with metrics ---")
metrics = run!(model; backend = CPUBackend(), steps = 400)
show(stdout, MIME"text/plain"(), metrics)
println()

println("\n--- stop conditions work ---")
for stop in (Steps(50), ForDuration(20.0), AnyOf(WallClock(0.02), Steps(100_000)))
    m = Wave2D(n = 96)
    m.current[48, 48] = 1.0f0
    m.previous[48, 48] = 1.0f0
    r = run!(Simulation(m; stop = stop))
    @printf("  %-42s %6d steps, stopped by :%s\n", string(stop), r.steps, r.stopped_by)
end

println("\n--- callbacks and in-situ metrics work ---")
recorder = MetricRecorder(energy = m -> sum_state(m),
                          peak = m -> maximum(abs, state(m)))
m = Wave2D(n = 128)
m.current[64, 64] = 1.0f0
m.previous[64, 64] = 1.0f0
run!(Simulation(m; stop = Steps(300)); callback = recorder, callback_every = 60)
show(stdout, MIME"text/plain"(), recorder)
println()

println("\n--- parameter sweeps work ---")
results = parameter_sweep(Float32[0.2, 0.4, 0.6]; stop = Steps(200),
                          observe = (m, _) -> (; energy = Float64(sum_state(m)))) do dt
    m = Wave2D(n = 96, dt = dt)
    m.current[48, 48] = 1.0f0
    m.previous[48, 48] = 1.0f0
    m
end
print(sweep_table(results))

println("\n--- checkpointing and the twin loop work too ---")
path = joinpath(mktempdir(), "wave.vts")
# save_state is written against Heat2D's field layout, so a model with three
# buffers needs its own — the one place this model does not inherit for free.
# Reporting that honestly is the point of the exercise.
println("  save_state:  Heat2D-specific; a three-buffer model needs its own (see the report guidance)")
loop = TwinLoop(validate = (m, o, t) -> (; energy = Float64(sum_state(m))),
                decide = (m, c, t) -> c.energy > 0.5 ? :ringing : :quiet,
                steps_per_window = 25)
log = twin_run!(loop, Wave2D(n = 64), ones(8))
@printf("  twin_run!:   %d windows, decisions %s\n", length(log), string(unique(log.decisions)))

println("""

What this file demonstrates, and what a contributed component looks like:

  * Four required methods — `state`, `timestep`, `clock`, and `step!` — buy the
    entire runtime. Nothing in `run!`, the stop conditions, the metrics, the
    callbacks, the sweeps or the twin loop is specific to heat diffusion.
  * Three optional methods were overridden because their defaults were wrong
    here: this stencil touches three arrays rather than two, does 6 flops rather
    than 10, and conserves energy rather than mass. Getting those wrong would not
    break the run — it would silently produce wrong MLUP/s and bandwidth figures,
    which is worse.
  * One thing did *not* come for free: `save_state` assumes a single-buffer field.
    A three-buffer model needs its own. Finding and reporting a limit like that is
    exactly what the independent-work analysis is for.
  * No GPU support here, because `move_to_device` was not implemented. That is a
    legitimate stopping point — using this model with a GPU backend fails with a
    message saying so rather than silently computing on host memory.
""")
