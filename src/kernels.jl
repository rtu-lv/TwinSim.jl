# One stencil, every backend.
#
# `interior_update` is the hot path: no branches, no index arithmetic beyond
# +/-1. `heat_update` wraps it with the boundary handling and is only called on
# edge cells, so the boundary condition never costs anything in the interior.
#
# Both are plain functions with no CPU-only constructs, which is what lets the
# serial loop, the threaded loop and the GPU kernel share them verbatim.

"""
    interior_update(u, i, j, cx, cy)

Five-point explicit heat stencil for a cell that is guaranteed to have all four
neighbours. Branch-free by construction.
"""
@inline function interior_update(u, i, j, cx, cy)
    @inbounds begin
        uij = u[i, j]
        return uij +
               cx * (u[i - 1, j] - 2 * uij + u[i + 1, j]) +
               cy * (u[i, j - 1] - 2 * uij + u[i, j + 1])
    end
end

"""
    heat_update(u, i, j, nx, ny, cx, cy, bc)

Stencil for any cell, including the edges, with the boundary condition applied.
Dispatch on `bc` keeps the fixed-value test out of the code generated for the
conserving boundary conditions.
"""
@inline function heat_update(u, i, j, nx, ny, cx, cy, bc::Dirichlet)
    if i == 1 || j == 1 || i == nx || j == ny
        return convert(eltype(u), bc.value)
    end
    return interior_update(u, i, j, cx, cy)
end

@inline function heat_update(u, i, j, nx, ny, cx, cy, bc::Union{Neumann,Periodic})
    @inbounds begin
        im1 = neighbor_index(bc, i, nx, -1)
        ip1 = neighbor_index(bc, i, nx, 1)
        jm1 = neighbor_index(bc, j, ny, -1)
        jp1 = neighbor_index(bc, j, ny, 1)
        uij = u[i, j]
        return uij +
               cx * (u[im1, j] - 2 * uij + u[ip1, j]) +
               cy * (u[i, jm1] - 2 * uij + u[i, jp1])
    end
end

"""
    heat2d_kernel!(next, current, nx, ny, cx, cy, bc)

The portable KernelAbstractions kernel. This exact function runs on the CPU, on
CUDA, on Metal and on ROCm — switching backend changes the launch, not the
source.
"""
@kernel function heat2d_kernel!(next, @Const(current), nx, ny, cx, cy, bc)
    i, j = @index(Global, NTuple)
    # KernelAbstractions rounds the launch up to whole workgroups, so the guard
    # is required whenever the grid is not a multiple of the workgroup size.
    if i <= nx && j <= ny
        @inbounds next[i, j] = heat_update(current, i, j, nx, ny, cx, cy, bc)
    end
end

# ---------------------------------------------------------------------------
# CPU reference backend
# ---------------------------------------------------------------------------

"""
    update_column!(next, current, j, nx, ny, cx, cy, bc)

Update one column of the grid. Interior columns are split into
`boundary / interior / boundary` so the innermost loop is a straight
`@simd`-able run with no per-cell branch — the branch that sat in the hot loop
of the original implementation.
"""
@inline function update_column!(next, current, j, nx, ny, cx, cy, bc)
    if j == 1 || j == ny || nx < 3
        @inbounds for i in 1:nx
            next[i, j] = heat_update(current, i, j, nx, ny, cx, cy, bc)
        end
    else
        @inbounds next[1, j] = heat_update(current, 1, j, nx, ny, cx, cy, bc)
        @inbounds @simd for i in 2:(nx - 1)
            next[i, j] = interior_update(current, i, j, cx, cy)
        end
        @inbounds next[nx, j] = heat_update(current, nx, j, nx, ny, cx, cy, bc)
    end
    return nothing
end

"""
    step!(backend, model) -> model

Advance `model` by one time step on `backend`. Buffers are swapped, never
copied.
"""
function step!(backend::CPUBackend, model::Heat2D)
    field = model.field
    current, next = field.current, field.next
    nx, ny = size(field)
    cx, cy = diffusion_coefficients(model.params)
    bc = model.boundary

    if backend.threaded
        Threads.@threads for j in 1:ny
            update_column!(next, current, j, nx, ny, cx, cy, bc)
        end
    else
        for j in 1:ny
            update_column!(next, current, j, nx, ny, cx, cy, bc)
        end
    end

    swapbuffers!(field)
    return model
end

function step!(backend::KernelBackend, model::Heat2D)
    field = model.field
    nx, ny = size(field)
    cx, cy = diffusion_coefficients(model.params)
    kernel! = heat2d_kernel!(backend.device, workgroup_size(backend))
    kernel!(field.next, field.current, nx, ny, cx, cy, model.boundary; ndrange = (nx, ny))
    swapbuffers!(field)
    return model
end
