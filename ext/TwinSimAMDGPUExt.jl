module TwinSimAMDGPUExt

using AMDGPU
using TwinSim

# AMD ROCm support. Same portable kernel, third vendor.
function TwinSim.gpu_device(::Val{:rocm})
    AMDGPU.functional() || throw(ArgumentError(
        "AMDGPU.jl is loaded but no usable ROCm device was found. " *
        "Check `AMDGPU.versioninfo()`."))
    return AMDGPU.ROCBackend()
end

end # module TwinSimAMDGPUExt
