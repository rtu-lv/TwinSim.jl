module VisuTwinSim

using Random

export AbstractBackend,
    CPUBackend,
    CUDABackend,
    Field2D,
    Heat2D,
    Heat2DParams,
    RunMetrics,
    Simulation,
    Steps,
    initialize_peak!,
    run!,
    sum_state,
    center_value,
    random_walk_ensemble

abstract type AbstractBackend end

"""
    CPUBackend()

Reference backend that runs models with plain Julia arrays and loops. This is the
backend used for correctness tests and for explaining the algorithm before moving
to GPU execution.
"""
struct CPUBackend <: AbstractBackend end

"""
    CUDABackend()

CUDA backend marker. The implementation is loaded by Julia's package extension
system when CUDA.jl is available in the active environment.
"""
struct CUDABackend <: AbstractBackend end

"""
    Field2D(nx, ny; initial = 0.0f0)

Double-buffered two-dimensional scalar field. The first index is `x`, the second
index is `y`, matching the notation used in the course examples.
"""
struct Field2D{T,A<:AbstractMatrix{T}}
    current::A
    next::A
end

function Field2D(nx::Integer, ny::Integer; initial::T = 0.0f0) where {T}
    nx > 0 || throw(ArgumentError("nx must be positive"))
    ny > 0 || throw(ArgumentError("ny must be positive"))
    data = fill(initial, nx, ny)
    return Field2D(data, similar(data))
end

Field2D(current::AbstractMatrix{T}) where {T} = Field2D(current, similar(current))

Base.size(field::Field2D) = size(field.current)
Base.size(field::Field2D, dim::Integer) = size(field.current, dim)
Base.getindex(field::Field2D, i::Integer, j::Integer) = field.current[i, j]
Base.setindex!(field::Field2D, value, i::Integer, j::Integer) = (field.current[i, j] = value)

function swapbuffers!(field::Field2D)
    field.current, field.next
end

"""
    Heat2DParams(; alpha = 0.15f0, dt = 0.1f0, dx = 1.0f0, dy = 1.0f0)

Parameters for an explicit finite-difference heat diffusion model.
"""
Base.@kwdef struct Heat2DParams{T}
    alpha::T = 0.15f0
    dt::T = 0.1f0
    dx::T = 1.0f0
    dy::T = 1.0f0
end

struct Heat2D{F<:Field2D,P<:Heat2DParams}
    field::F
    params::P
end

Heat2D(field::Field2D; kwargs...) = Heat2D(field, Heat2DParams(; kwargs...))
Heat2D(; nx::Integer = 128, ny::Integer = nx, initial = 0.0f0, kwargs...) =
    Heat2D(Field2D(nx, ny; initial); kwargs...)

struct Steps
    count::Int
end

Steps(count::Integer) = count >= 0 ? Steps(Int(count)) : throw(ArgumentError("step count must be non-negative"))

Base.@kwdef mutable struct RunMetrics
    steps::Int = 0
    simulated_time::Float64 = 0.0
    last_reduction::Float64 = 0.0
    backend::Symbol = :unknown
end

struct Simulation{M,B<:AbstractBackend}
    model::M
    backend::B
    stop::Steps
end

Simulation(model; backend::AbstractBackend = CPUBackend(), stop::Steps = Steps(1)) =
    Simulation(model, backend, stop)

function initialize_peak!(field::Field2D, value = 100.0f0; x::Integer = cld(size(field, 1), 2), y::Integer = cld(size(field, 2), 2))
    field[x, y] = value
    return field
end

sum_state(field::Field2D) = sum(field.current)
sum_state(model::Heat2D) = sum_state(model.field)

function center_value(field::Field2D)
    x = cld(size(field, 1), 2)
    y = cld(size(field, 2), 2)
    return field[x, y]
end

center_value(model::Heat2D) = center_value(model.field)

function step!(::CPUBackend, model::Heat2D)
    field = model.field
    current = field.current
    next = field.next
    nx, ny = size(field)
    params = model.params
    cx = params.alpha * params.dt / (params.dx * params.dx)
    cy = params.alpha * params.dt / (params.dy * params.dy)

    @inbounds for j in 1:ny, i in 1:nx
        if i == 1 || j == 1 || i == nx || j == ny
            next[i, j] = current[i, j]
        else
            next[i, j] = current[i, j] +
                cx * (current[i - 1, j] - 2 * current[i, j] + current[i + 1, j]) +
                cy * (current[i, j - 1] - 2 * current[i, j] + current[i, j + 1])
        end
    end

    field.current .= next
    return model
end

function run!(sim::Simulation{<:Heat2D,<:CPUBackend})
    metrics = RunMetrics(backend = :cpu)
    for _ in 1:sim.stop.count
        step!(sim.backend, sim.model)
        metrics.steps += 1
        metrics.simulated_time += Float64(sim.model.params.dt)
    end
    metrics.last_reduction = Float64(sum_state(sim.model))
    return metrics
end

run!(model::Heat2D; backend::AbstractBackend = CPUBackend(), steps::Integer = 1) =
    run!(Simulation(model; backend, stop = Steps(steps)))

function run!(::Simulation{<:Heat2D,<:CUDABackend})
    throw(ArgumentError("CUDABackend requires CUDA.jl in the active environment. Run `import Pkg; Pkg.add(\"CUDA\")`, then retry."))
end

"""
    random_walk_ensemble(; trajectories, steps, seed)

Small embarrassingly parallel teaching example. It returns final one-dimensional
positions for independent random walks and is useful before introducing GPU
ensemble simulation.
"""
function random_walk_ensemble(; trajectories::Integer = 10_000, steps::Integer = 1_000, seed::Integer = 1)
    trajectories > 0 || throw(ArgumentError("trajectories must be positive"))
    steps >= 0 || throw(ArgumentError("steps must be non-negative"))
    rng = MersenneTwister(seed)
    positions = zeros(Int, trajectories)
    for _ in 1:steps
        @inbounds for i in eachindex(positions)
            positions[i] += rand(rng, Bool) ? 1 : -1
        end
    end
    return positions
end

end # module VisuTwinSim
