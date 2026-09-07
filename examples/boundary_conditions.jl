# Why the boundary condition is not a detail.
#
#   julia --project=. examples/boundary_conditions.jl
#
# The same initial condition under three boundary conditions. Two conserve heat,
# one does not — and the one that does not is the behaviour the package had
# before boundary conditions were made explicit.

using Printf
using TwinSim

const STEPS = (0, 500, 2_000, 10_000, 50_000)

function evolve(bc)
    model = Heat2D(nx = 64, ny = 64; boundary = bc)
    initialize_peak!(model.field, 100.0f0)
    totals = Float64[]
    previous = 0
    for target in STEPS
        run!(model; steps = target - previous)
        previous = target
        push!(totals, Float64(sum_state(model)))
    end
    return totals
end

@printf("%-14s %9s", "boundary", "")
foreach(s -> @printf("%12d", s), STEPS)
println()
@printf("%-14s %9s", "", "conserves")
foreach(_ -> @printf("%12s", "total heat"), STEPS)
println()

for bc in (Neumann(), Periodic(), Dirichlet(0.0f0))
    @printf("%-14s %9s", nameof(typeof(bc)), conserves_state(bc) ? "yes" : "no")
    foreach(t -> @printf("%12.4f", t), evolve(bc))
    println()
end

println("""

Dirichlet holds the edge cells at a fixed value, so heat that reaches the
boundary leaves the domain permanently. Neumann mirrors the ghost cell, making
the flux across the edge exactly zero, and Periodic wraps the domain — both keep
the total constant to within floating-point rounding.

Two consequences for the course:

  * "Total heat is conserved" is only a valid test under a conserving boundary
    condition, and only a *meaningful* one once heat has had time to reach the
    edge. A short run passes it no matter what the boundary does.
  * Picking a boundary condition is a modelling decision about the physical
    system: an insulated vessel is Neumann, a plate clamped to a heat sink is
    Dirichlet. A digital twin that gets this wrong drifts away from its
    real counterpart no matter how accurate the interior scheme is.
""")
