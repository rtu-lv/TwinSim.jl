# The same model on a GPU. Load whichever vendor package your machine has:
#
#   julia --project=. -e 'using Pkg; Pkg.add("CUDA")'    # NVIDIA
#   julia --project=. -e 'using Pkg; Pkg.add("Metal")'   # Apple Silicon
#   julia --project=. -e 'using Pkg; Pkg.add("AMDGPU")'  # AMD
#
#   julia --project=. examples/heat2d_gpu.jl
#
# Note what is *not* here: a GPU kernel. `VisuTwinSim.heat2d_kernel!` is written
# once with KernelAbstractions and this script only picks a different device.

using VisuTwinSim

# `Base.invokelatest` matters here: `using CUDA` defines the extension's device
# method in a newer world age than this script was lowered in, so calling
# CUDADevice() directly would still see the "backend not loaded" fallback.
backend = try
    if Base.identify_package("CUDA") !== nothing
        @eval using CUDA
        Base.invokelatest(CUDADevice)
    elseif Base.identify_package("Metal") !== nothing
        @eval using Metal
        Base.invokelatest(MetalDevice)
    elseif Base.identify_package("AMDGPU") !== nothing
        @eval using AMDGPU
        Base.invokelatest(ROCmDevice)
    else
        error("no GPU package installed")
    end
catch err
    @warn "Falling back to the portable CPU kernel" exception = err
    KernelBackend()
end

println("running on: ", backend_name(backend), "\n")

model = Heat2D(nx = 2048, ny = 2048, alpha = 0.15f0, dt = 0.1f0)
initialize_peak!(model.field, 100.0f0)

# Move the state onto the device once, so repeated runs are not billed for a
# transfer each time. `run!` would do this internally, but then it would also
# copy back and forth on every call.
resident = to_backend(model, backend)

run!(resident; backend = backend, steps = 10)              # warm up / compile
metrics = run!(resident; backend = backend, steps = 1000)

show(stdout, MIME"text/plain"(), metrics)
println("\n")
println("centre value  ", center_value(resident))
println("total heat    ", sum_state(resident))
