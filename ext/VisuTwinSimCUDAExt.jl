module VisuTwinSimCUDAExt

using CUDA
using VisuTwinSim

function heat2d_kernel!(next, current, nx::Int, ny::Int, alpha, dt, dx, dy)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= nx && j <= ny
        @inbounds begin
            if i == 1 || j == 1 || i == nx || j == ny
                next[i, j] = current[i, j]
            else
                cx = alpha * dt / (dx * dx)
                cy = alpha * dt / (dy * dy)
                next[i, j] = current[i, j] +
                    cx * (current[i - 1, j] - 2 * current[i, j] + current[i + 1, j]) +
                    cy * (current[i, j - 1] - 2 * current[i, j] + current[i, j + 1])
            end
        end
    end

    return nothing
end

function VisuTwinSim.run!(sim::VisuTwinSim.Simulation{<:VisuTwinSim.Heat2D,<:VisuTwinSim.CUDABackend})
    CUDA.functional() || throw(ArgumentError("CUDA.jl is installed, but no functional CUDA device is available"))

    model = sim.model
    field = model.field
    current = CuArray(field.current)
    next = similar(current)
    nx, ny = size(field)
    params = model.params

    threads = (16, 16)
    blocks = (cld(nx, threads[1]), cld(ny, threads[2]))
    metrics = VisuTwinSim.RunMetrics(backend = :cuda)

    for _ in 1:sim.stop.count
        @cuda threads=threads blocks=blocks heat2d_kernel!(
            next,
            current,
            nx,
            ny,
            params.alpha,
            params.dt,
            params.dx,
            params.dy)
        current, next = next, current
        metrics.steps += 1
        metrics.simulated_time += Float64(params.dt)
    end

    CUDA.synchronize()
    field.current .= Array(current)
    metrics.last_reduction = Float64(sum(current))
    return metrics
end

end # module VisuTwinSimCUDAExt
