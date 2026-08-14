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
@inline function interior_update(u, i, j, cx, cy, source, dt)
    @inbounds begin
        uij = u[i, j]
        diffused = uij +
                   cx * (u[i - 1, j] - 2 * uij + u[i + 1, j]) +
                   cy * (u[i, j - 1] - 2 * uij + u[i, j + 1])
        return apply_source(source, diffused, i, j, dt)
    end
end

"""
    heat_update(u, i, j, nx, ny, cx, cy, bc)

Stencil for any cell, including the edges, with the boundary condition applied.
Dispatch on `bc` keeps the fixed-value test out of the code generated for the
conserving boundary conditions.
"""
@inline function heat_update(u, i, j, nx, ny, cx, cy, bc::Dirichlet, source, dt)
    if i == 1 || j == 1 || i == nx || j == ny
        # A pinned cell is pinned. Adding the source here would let forcing
        # override the boundary condition, and the edge would drift away from
        # the value the model says it is held at.
        return convert(eltype(u), bc.value)
    end
    return interior_update(u, i, j, cx, cy, source, dt)
end

@inline function heat_update(u, i, j, nx, ny, cx, cy, bc::Union{Neumann,Periodic}, source, dt)
    @inbounds begin
        im1 = neighbor_index(bc, i, nx, -1)
        ip1 = neighbor_index(bc, i, nx, 1)
        jm1 = neighbor_index(bc, j, ny, -1)
        jp1 = neighbor_index(bc, j, ny, 1)
        uij = u[i, j]
        diffused = uij +
                   cx * (u[im1, j] - 2 * uij + u[ip1, j]) +
                   cy * (u[i, jm1] - 2 * uij + u[i, jp1])
        return apply_source(source, diffused, i, j, dt)
    end
end

"""
    heat2d_kernel!(next, current, nx, ny, cx, cy, bc)

The portable KernelAbstractions kernel. This exact function runs on the CPU, on
CUDA, on Metal and on ROCm — switching backend changes the launch, not the
source.
"""
@kernel function heat2d_kernel!(next, @Const(current), nx, ny, cx, cy, bc, source, dt)
    i, j = @index(Global, NTuple)
    # KernelAbstractions rounds the launch up to whole workgroups, so the guard
    # is required whenever the grid is not a multiple of the workgroup size.
    if i <= nx && j <= ny
        @inbounds next[i, j] = heat_update(current, i, j, nx, ny, cx, cy, bc, source, dt)
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
# A column lying on the domain edge: every cell needs the boundary-aware stencil.
@inline function update_edge_column!(next, current, j, nx, ny, cx, cy, bc, source, dt)
    @inbounds for i in 1:nx
        next[i, j] = heat_update(current, i, j, nx, ny, cx, cy, bc, source, dt)
    end
    return nothing
end

# A column away from the top and bottom edges: only its first and last cells
# touch the boundary, so the run between them is a branch-free `@simd` loop.
@inline function update_interior_column!(next, current, j, nx, ny, cx, cy, bc, source, dt)
    nx < 3 && return update_edge_column!(next, current, j, nx, ny, cx, cy, bc, source, dt)

    @inbounds next[1, j] = heat_update(current, 1, j, nx, ny, cx, cy, bc, source, dt)
    @inbounds @simd for i in 2:(nx - 1)
        next[i, j] = interior_update(current, i, j, cx, cy, source, dt)
    end
    @inbounds next[nx, j] = heat_update(current, nx, j, nx, ny, cx, cy, bc, source, dt)
    return nothing
end

"""
    step!(backend, model, t = 0) -> model

Advance `model` by one time step on `backend`, at simulated time `t`. Buffers
are swapped, never copied.

`t` is only read by a time-varying boundary or source; for a model with neither,
it is ignored. `run!` passes the running simulated time, so a driven model
stepped through `run!` sees the correct clock. Calling `step!` directly on a
driven model without a `t` evaluates the drive at time zero, which is a way to
get a silently constant forcing.
"""
function step!(backend::CPUBackend, model::Heat2D{T}, t = zero(T)) where {T}
    field = model.field
    current, next = field.current, field.next
    nx, ny = size(field)
    cx, cy = diffusion_coefficients(model.params)
    dt = model.params.dt

    # Time-dependent values are evaluated once here, not once per cell.
    bc = resolve(model.boundary, t, T)
    source = resolve(model.source, t, T)

    # The two edge columns are handled outside the loop rather than with a
    # `j == 1 || j == ny` test inside it.
    #
    # This is not micro-optimisation. With the test inside the loop, a Dirichlet
    # boundary ran at 1300 MLUP/s against Neumann's 5200 — yet timed on their
    # own, Dirichlet's edge column is the *faster* of the two (35 ns against
    # 1087 ns), because it only writes a constant. The work was never the
    # problem: a branch whose two sides generate very different code, sitting in
    # the loop that has to vectorise, costs far more than the branch itself.
    if ny < 3
        for j in 1:ny
            update_edge_column!(next, current, j, nx, ny, cx, cy, bc, source, dt)
        end
    else
        update_edge_column!(next, current, 1, nx, ny, cx, cy, bc, source, dt)
        update_edge_column!(next, current, ny, nx, ny, cx, cy, bc, source, dt)
        if backend.threaded
            Threads.@threads for j in 2:(ny - 1)
                update_interior_column!(next, current, j, nx, ny, cx, cy, bc, source, dt)
            end
        else
            for j in 2:(ny - 1)
                update_interior_column!(next, current, j, nx, ny, cx, cy, bc, source, dt)
            end
        end
    end

    swapbuffers!(field)
    return model
end

function step!(backend::KernelBackend, model::Heat2D{T}, t = zero(T)) where {T}
    field = model.field
    nx, ny = size(field)
    cx, cy = diffusion_coefficients(model.params)
    dt = model.params.dt

    # Resolved on the host: a GPU kernel cannot call an arbitrary Julia closure,
    # and re-evaluating one scalar in every thread would be waste even if it could.
    bc = resolve(model.boundary, t, T)
    source = resolve(model.source, t, T)

    kernel! = heat2d_kernel!(backend.device, workgroup_size(backend))
    kernel!(field.next, field.current, nx, ny, cx, cy, bc, source, dt; ndrange = (nx, ny))
    swapbuffers!(field)
    return model
end
