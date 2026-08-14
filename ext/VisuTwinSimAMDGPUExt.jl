module VisuTwinSimAMDGPUExt

using AMDGPU
using VisuTwinSim

# AMD ROCm support. Same portable kernel, third vendor.
function VisuTwinSim.gpu_device(::Val{:rocm})
    AMDGPU.functional() || throw(ArgumentError(
        "AMDGPU.jl is loaded but no usable ROCm device was found. " *
        "Check `AMDGPU.versioninfo()`."))
    return AMDGPU.ROCBackend()
end

end # module VisuTwinSimAMDGPUExt
