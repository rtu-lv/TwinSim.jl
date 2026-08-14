using VisuTwinSim
using CUDA

model = Heat2D(nx = 512, ny = 512, alpha = 0.15f0, dt = 0.1f0)
initialize_peak!(model.field, 100.0f0)

metrics = run!(model; backend = CUDABackend(), steps = 1_000)

println("backend = ", metrics.backend)
println("steps = ", metrics.steps)
println("heat_sum = ", metrics.last_reduction)
println("center = ", center_value(model))
