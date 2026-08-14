"""
    AbstractBackend

Execution backend for a [`Simulation`](@ref). The course works through the
backends in this order:

1. `CPUBackend()` – plain Julia loops, one thread. The reference implementation.
2. `CPUBackend(threaded = true)` – the same loops split across `Threads.nthreads()`.
3. `KernelBackend()` – the portable KernelAbstractions kernel on the CPU.
4. `CUDADevice()` / `MetalDevice()` / `ROCmDevice()` – the *same* kernel on a GPU.

Steps 3 and 4 run identical kernel source, which is the point: the only thing
that changes between a CPU run and a GPU run is the device the kernel is
launched on.
"""
abstract type AbstractBackend end

"""
    CPUBackend(; threaded = false)

Reference backend built from ordinary Julia arrays and loops. With
`threaded = true` the outer (column) loop is distributed with `Threads.@threads`;
start Julia with `julia -t auto` for that to do anything.
"""
struct CPUBackend <: AbstractBackend
    threaded::Bool
end

CPUBackend(; threaded::Bool = false) = CPUBackend(threaded)

"""
    KernelBackend(device = KernelAbstractions.CPU())

Backend that executes the portable KernelAbstractions kernel on `device`.
`KernelBackend()` runs it on the CPU, which is how the GPU code path stays
testable on machines without a GPU.

Prefer the [`CUDADevice`](@ref), [`MetalDevice`](@ref) and [`ROCmDevice`](@ref)
helpers over constructing this with a vendor device by hand.
"""
struct KernelBackend{D} <: AbstractBackend
    device::D
end

KernelBackend() = KernelBackend(KernelAbstractions.CPU())

# Vendor GPU packages export their own `CUDABackend` / `MetalBackend` / `ROCBackend`
# types. Naming ours `*Device` keeps `using VisuTwinSim, CUDA` free of the name
# clash that an exported `CUDABackend` here would cause.

"""
    CUDADevice()

`KernelBackend` running on an NVIDIA GPU. Requires `using CUDA` first; the
device is registered through a package extension.
"""
CUDADevice() = KernelBackend(gpu_device(Val(:cuda)))

"""
    MetalDevice()

`KernelBackend` running on an Apple Silicon GPU. Requires `using Metal` first.
"""
MetalDevice() = KernelBackend(gpu_device(Val(:metal)))

"""
    ROCmDevice()

`KernelBackend` running on an AMD GPU. Requires `using AMDGPU` first.
"""
ROCmDevice() = KernelBackend(gpu_device(Val(:rocm)))

const GPU_PACKAGES = (cuda = "CUDA", metal = "Metal", rocm = "AMDGPU")

"""
    gpu_device(::Val{:cuda|:metal|:rocm})

Hook filled in by the package extensions. The fallback below is deliberately the
*only* definition in this module: extensions add methods for their own `Val`,
they never overwrite an existing one. Overwriting a method from an extension
triggers method invalidation and a redefinition warning.
"""
function gpu_device(::Val{S}) where {S}
    pkg = get(GPU_PACKAGES, S, string(S))
    throw(ArgumentError("""
        The $S backend is not loaded.

            using Pkg; Pkg.add("$pkg")
            using $pkg          # registers the device with VisuTwinSim
            using VisuTwinSim

        `using $pkg` must happen in the same session; VisuTwinSim picks the device
        up through a package extension. Currently available: $(join(available_backends(), ", ")).
        """))
end

"""
    available_backends() -> Vector{String}

Names of the backends usable in this session. Useful as a first command in a lab
so students can see what their machine actually offers.
"""
function available_backends()
    names = ["CPUBackend()", "CPUBackend(threaded = true)", "KernelBackend()"]
    for (key, helper) in ((:cuda, "CUDADevice()"), (:metal, "MetalDevice()"), (:rocm, "ROCmDevice()"))
        if hasmethod(gpu_device, Tuple{Val{key}})
            push!(names, helper)
        end
    end
    return names
end

backend_name(backend::CPUBackend) = backend.threaded ? :cpu_threaded : :cpu
backend_name(::KernelBackend{KernelAbstractions.CPU}) = :ka_cpu

function backend_name(backend::KernelBackend)
    label = replace(String(nameof(typeof(backend.device))), "Backend" => "")
    return Symbol(lowercase(label))
end

"""
    is_gpu(backend) -> Bool

Whether the backend runs on a device with separate memory, i.e. whether a run
pays for host/device transfers.
"""
is_gpu(::CPUBackend) = false
is_gpu(backend::KernelBackend) = !(backend.device isa KernelAbstractions.CPU)

"""
    device_synchronize(backend)

Block until all work queued on the backend has finished. Timing a GPU run
without this measures kernel *launch* time, not kernel execution time — one of
the classic first mistakes when benchmarking GPU code.
"""
device_synchronize(::CPUBackend) = nothing
device_synchronize(backend::KernelBackend) = KernelAbstractions.synchronize(backend.device)

"""
    workgroup_size(backend) -> Tuple

Default workgroup (CUDA "block") shape for two-dimensional kernels. 16x16 = 256
work items is a reasonable default everywhere; labs can override it to measure
the effect of the launch configuration.
"""
workgroup_size(::AbstractBackend) = (16, 16)
