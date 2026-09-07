# The reference run: one hot cell in a cold insulated plate.
#
#   julia --project=. examples/heat2d_cpu.jl

using TwinSim

model = Heat2D(nx = 512, ny = 512, alpha = 0.15f0, dt = 0.1f0, boundary = Neumann())
initialize_peak!(model.field, 100.0f0)

println(sprint(show, MIME"text/plain"(), model), "\n")

metrics = run!(model; backend = CPUBackend(), steps = 500)

show(stdout, MIME"text/plain"(), metrics)
println("\n")
println("centre value  ", center_value(model))
println("total heat    ", sum_state(model), "  (started at 100.0)")
println()
println("The total is unchanged because Neumann boundaries are insulating.")
println("Re-run with `boundary = Dirichlet(0.0f0)` and it will decay instead.")
