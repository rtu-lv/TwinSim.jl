using Test
using VisuTwinSim

# GPU backends are optional. Any vendor package present in the test environment
# is picked up automatically; everything else is skipped with a note rather than
# failing, so the same suite runs on a laptop, in CI and on a cluster node.
const GPU_BACKENDS = Pair{String,Any}[]

for (pkg, helper) in (("CUDA", CUDADevice), ("Metal", MetalDevice), ("AMDGPU", ROCmDevice))
    Base.identify_package(pkg) === nothing && continue
    try
        @eval using $(Symbol(pkg))
        # invokelatest: the extension's gpu_device method is defined in a newer
        # world age than this file was lowered in.
        push!(GPU_BACKENDS, pkg => Base.invokelatest(helper))
    catch err
        @info "GPU backend $pkg present but not usable, skipping" exception = err
    end
end

if isempty(GPU_BACKENDS)
    @info "No GPU backend available; GPU tests will be skipped. " *
          "Add CUDA, Metal or AMDGPU to test/Project.toml to exercise them."
else
    @info "Testing GPU backends: $(join(first.(GPU_BACKENDS), ", "))"
end

@testset "VisuTwinSim" begin
    include("test_field.jl")
    include("test_model.jl")
    include("test_analytical.jl")
    include("test_source.jl")
    include("test_backends.jl")
    include("test_runtime.jl")
    include("test_twin.jl")
    include("test_ensemble.jl")
end
