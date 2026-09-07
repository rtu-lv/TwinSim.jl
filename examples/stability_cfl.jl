# What "the time step is too large" actually looks like.
#
#   julia --project=. examples/stability_cfl.jl
#
# The explicit scheme is stable only while
#     CFL = alpha * dt * (1/dx^2 + 1/dy^2) <= 1/2.
# Above that the solution does not just get inaccurate, it diverges — and it
# does so exponentially, so a run that looks fine for fifty steps is NaN by two
# hundred.

using Printf
using TwinSim

alpha, dx = 0.15f0, 1.0f0

println("alpha = $alpha, dx = dy = $dx")
println("largest stable dt = ", max_stable_dt(Heat2DParams(alpha = alpha, dt = 0.1f0, dx = dx, dy = dx)))
println()
@printf("%8s %8s %10s   %s\n", "dt", "CFL", "stable?", "max |u| after N steps")
@printf("%8s %8s %10s   %8s %12s %12s %12s\n", "", "", "", "50", "100", "200", "400")

for dt in Float32[0.1, 1.0, 1.6, 1.67, 1.7, 2.0]
    params = Heat2DParams(alpha = alpha, dt = dt, dx = dx, dy = dx)
    @printf("%8.3f %8.4f %10s  ", dt, cfl_number(params), is_stable(params) ? "yes" : "NO")

    # check_stability = false is required here: the constructor rejects these on
    # purpose, and this script exists precisely to show why.
    model = Heat2D(nx = 64, ny = 64; alpha = alpha, dt = dt, dx = dx, dy = dx,
                   check_stability = false)
    initialize_peak!(model.field, 100.0f0)

    previous = 0
    for target in (50, 100, 200, 400)
        run!(model; steps = target - previous)
        previous = target
        peak = maximum(abs, state(model))
        @printf("%13.4g", peak)
    end
    println()
end

println("""

The transition is sharp. At CFL = 0.5005 (dt = 1.67) the run is still finite
after 400 steps but already growing; a little above it, every extra step
multiplies the error again and the field saturates to Inf and then NaN.

By default `Heat2D` refuses to build an unstable configuration:

    julia> Heat2D(nx = 64, dt = 2.0f0)
    ERROR: ArgumentError: Unstable configuration: CFL number is 0.6 ...

which turns a silent NaN twenty seconds into a run into an error at the point
where the mistake was actually made.

Questions:
  * Halving dx while keeping dt fixed multiplies the CFL number by four. What
    does that imply about the cost of refining a grid with an explicit scheme?
  * An implicit scheme is unconditionally stable but needs a linear solve per
    step. At which grid size does that trade become worth it?
""")
