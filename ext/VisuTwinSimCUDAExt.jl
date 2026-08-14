module VisuTwinSimCUDAExt

using CUDA
using VisuTwinSim
using VisuTwinSim: heat_update, diffusion_coefficients, resolve, swapbuffers!

# ---------------------------------------------------------------------------
# The portable path: no kernel here at all.
# ---------------------------------------------------------------------------
#
# `VisuTwinSim.heat2d_kernel!` is written once with KernelAbstractions and CUDA.jl
# supplies a device it can be launched on. That is the whole NVIDIA backend.
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

# ---------------------------------------------------------------------------
# The teaching path: the same stencil, launched by hand.
# ---------------------------------------------------------------------------
#
# Everything below has an exact counterpart in the portable kernel. Read them
# side by side:
#
#   raw CUDA                                    KernelAbstractions
#   ------------------------------------------  --------------------------------
#   function f(...) ... end + @cuda             @kernel function f(...) ... end
#   (blockIdx().x-1)*blockDim().x + threadIdx() @index(Global, NTuple)
#   threads = (16, 16)                          workgroup_size(backend)
#   blocks  = cld.(size, threads)               ndrange = (nx, ny)
#   CUDA.synchronize()                          KernelAbstractions.synchronize
#
# The bounds guard is needed for the same reason in both: the launch is rounded
# up to whole blocks, so on a grid that is not a multiple of the block size some
# threads have no cell and must do nothing rather than write out of bounds.

"""
    raw_heat2d_kernel!(next, current, nx, ny, cx, cy, bc, source, dt)

Hand-written CUDA kernel. The body is one call to the shared `heat_update`; only
the index arithmetic is written out, because that is the part a portable kernel
hides and a CUDA lecture is about.
"""
function raw_heat2d_kernel!(next, current, nx, ny, cx, cy, bc, source, dt)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if i <= nx && j <= ny
        @inbounds next[i, j] = heat_update(current, i, j, nx, ny, cx, cy, bc, source, dt)
    end
    return nothing
end

function VisuTwinSim.step!(backend::RawCUDABackend, model::VisuTwinSim.Heat2D{T},
                           t = zero(T)) where {T}
    field = model.field
    nx, ny = size(field)
    cx, cy = diffusion_coefficients(model.params)
    dt = model.params.dt

    bc = resolve(model.boundary, t, T)
    source = resolve(model.source, t, T)

    threads = backend.threads
    blocks = (cld(nx, threads[1]), cld(ny, threads[2]))

    @cuda threads = threads blocks = blocks raw_heat2d_kernel!(
        field.next, field.current, nx, ny, cx, cy, bc, source, dt)

    swapbuffers!(field)
    return model
end
end # module VisuTwinSimCUDAExt
