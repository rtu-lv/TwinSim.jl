module VisuTwinSimMetalExt

using Metal
using VisuTwinSim

# Apple Silicon GPU support, so the GPU code path is reachable on a Mac.
#
# Note for labs: Metal defaults to Float32 and has limited Float64 support, so
# double-precision models will not run here. That constraint is worth showing
# rather than hiding — it is the same trade-off that consumer NVIDIA cards make.
function VisuTwinSim.gpu_device(::Val{:metal})
    Metal.functional() || throw(ArgumentError(
        "Metal.jl is loaded but no usable Metal device was found. " *
        "Metal requires Apple Silicon and macOS 13+; use KernelBackend() otherwise."))
    return Metal.MetalBackend()
end

end # module VisuTwinSimMetalExt
