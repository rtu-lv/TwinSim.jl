using VisuTwinSim

model = Heat2D(nx = 128, ny = 128, alpha = 0.15f0, dt = 0.1f0)
initialize_peak!(model.field, 100.0f0)

metrics = run!(model; backend = CPUBackend(), steps = 250)

println("backend = ", metrics.backend)
println("steps = ", metrics.steps)
println("heat_sum = ", metrics.last_reduction)
println("center = ", center_value(model))
