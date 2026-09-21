# One stencil, every backend.
#
# `interior_update` is the hot path: no branches, no index arithmetic beyond
# +/-1. `heat_update` wraps it with the boundary handling. The CPU loops call
# `heat_update` only on edge cells, so the boundary condition never costs
# anything in the interior; the GPU kernel calls it for every cell, because a
# single kernel body is simpler to launch than separate edge and interior ones.
#
# Both are plain functions with no CPU-only constructs, which is what lets the
# serial loop, the threaded loop and the GPU kernel share them verbatim.
#
# Reading order for the update rule itself: `interior_update` is the formula,
# `heat_update` adds the boundary, `step!(::CPUBackend, ...)` is the loop that
# reads the old buffer (`current`) and writes the new one (`next`).

"""
    interior_update(u, i, j, cx, cy, source, dt)

Five-point explicit heat stencil for a cell that is guaranteed to have all four
neighbours. Branch-free by construction.

`u` is the old buffer; the return value is the new value of cell `(i, j)`. `cx`
and `cy` are the stencil weights from [`diffusion_coefficients`](@ref), with
`cx` acting along the first index and `cy` along the second.

`source` and `dt` are only used by the forcing term. For a model without one
(`NoSource`, the default) `apply_source` returns `diffused` unchanged, so the
update is exactly the three-line formula below.
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
    heat_update(u, i, j, nx, ny, cx, cy, bc, source, dt)

Stencil for any cell, including the edges, with the boundary condition applied.
There is one method per kind of boundary:

- [`Dirichlet`](@ref): an edge cell is set to the boundary value and the
  stencil is not evaluated for it; every other cell has all four neighbours and
  goes to [`interior_update`](@ref).
- [`Neumann`](@ref) and [`Periodic`](@ref): the same formula as
  `interior_update`, with each neighbour index passed through `neighbor_index`,
  which mirrors or wraps an index that would fall outside the grid.

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
    heat2d_kernel!(next, current, nx, ny, cx, cy, bc, source, dt)

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

# The grid is updated one column (fixed `j`) at a time, because Julia stores
# arrays column by column and the inner loop then walks memory in order. There
# are two kinds of column, and keeping them apart is what leaves the innermost
# loop without a per-cell boundary test.

"""
    update_edge_column!(next, current, j, nx, ny, cx, cy, bc, source, dt)

Update column `j` when it lies on the domain edge (`j == 1` or `j == ny`): every
cell in it needs the boundary-aware [`heat_update`](@ref).
"""
@inline function update_edge_column!(next, current, j, nx, ny, cx, cy, bc, source, dt)
    @inbounds for i in 1:nx
        next[i, j] = heat_update(current, i, j, nx, ny, cx, cy, bc, source, dt)
    end
    return nothing
end

"""
    update_interior_column!(next, current, j, nx, ny, cx, cy, bc, source, dt)

Update column `j` when it lies away from the domain edge. Only its first and
last cells touch the boundary, so the column is split into
`boundary / interior / boundary` and the run in between is a branch-free
`@simd` loop over [`interior_update`](@ref).
"""
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
    undriven_step_time(model) -> Real

The time a `step!` call without a `t` argument uses: zero for a model nothing
drives, and an error for a driven one. See [`step!`](@ref).
"""
function undriven_step_time(model::Heat2D{T}) where {T}
    is_driven(model) && throw(ArgumentError("""
        step!(backend, model) was called without a time on a model whose boundary
        or source depends on time. step! does not advance the model's clock, so the
        drive would be evaluated at t = 0 on every call.

        Use run!(model; steps = n), which keeps the clock, or pass the time
        explicitly: step!(backend, model, t)."""))
    return zero(T)
end

"""
    step!(backend, model) -> model
    step!(backend, model, t) -> model

Advance `model` by one time step on `backend`, at simulated time `t`. The step
reads `field.current`, writes `field.next` and then swaps the two buffers, so
nothing is copied and no cell ever reads a value written during the same step.

`t` is only read by a time-varying boundary or source. A model with neither may
be stepped without it. A driven model may not: `step!` does not advance the
model's clock, so there is no meaningful default, and leaving `t` out throws
rather than evaluating the drive at time zero on every call. Use [`run!`](@ref),
which keeps the clock and passes it in, or pass `t` explicitly.
"""
function step!(backend::CPUBackend, model::Heat2D{T}, t = undriven_step_time(model)) where {T}
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
    # This is not micro-optimisation. Measured with the test inside the loop, a
    # Dirichlet boundary ran at 1300 MLUP/s against Neumann's 5200 — yet timed
    # on their own, Dirichlet's edge column is the *faster* of the two (35 ns
    # against 1087 ns), because it only writes a constant. The work was never
    # the problem: a branch whose two sides generate very different code,
    # sitting in the loop that has to vectorise, costs far more than the branch
    # itself.
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

function step!(backend::KernelBackend, model::Heat2D{T}, t = undriven_step_time(model)) where {T}
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
