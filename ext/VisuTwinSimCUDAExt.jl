module VisuTwinSimCUDAExt

using CUDA
using VisuTwinSim

# The whole NVIDIA backend. There is no kernel here, because there does not need
# to be one: `VisuTwinSim.heat2d_kernel!` is written with KernelAbstractions and
# CUDA.jl supplies a device it can be launched on.
#
# This fills a hole rather than replacing a definition — `gpu_device(::Val)` has
# exactly one method in the main module, and it errors. Defining a *new* method
# for `Val{:cuda}` adds a specialisation; redefining an existing method from an
# extension would invalidate already-compiled callers and emit an overwrite
# warning at load time.
function VisuTwinSim.gpu_device(::Val{:cuda})
    CUDA.functional() || throw(ArgumentError(
        "CUDA.jl is loaded but no usable CUDA device was found. " *
        "Check `CUDA.versioninfo()`; on a machine without an NVIDIA GPU use " *
        "KernelBackend() or MetalDevice() instead."))
    return CUDA.CUDABackend()
end

end # module VisuTwinSimCUDAExt
