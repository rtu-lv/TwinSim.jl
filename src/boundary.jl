"""
    BoundaryCondition

How the stencil treats cells on the edge of the grid. The choice is not
cosmetic: it decides whether the simulation conserves the quantity it
transports, which is the first invariant a digital twin has to get right.

| Condition      | Physical reading            | Conserves total heat |
|:---------------|:----------------------------|:---------------------|
| [`Neumann`](@ref)   | insulated box, zero flux    | yes                  |
| [`Periodic`](@ref)  | domain wraps around         | yes                  |
| [`Dirichlet`](@ref) | edge held at a fixed value  | no — heat leaves      |

All three are implemented by changing how a neighbour index is computed, so a
single stencil serves every case and every backend.
"""
abstract type BoundaryCondition end

"""
    Neumann()

Zero-flux (insulated) boundary: the ghost cell outside the grid mirrors the
cell inside it, so no heat crosses the edge. This is the default because it
makes "total heat is conserved" a true invariant that tests can assert.
"""
struct Neumann <: BoundaryCondition end

"""
    Periodic()

The grid wraps around in both directions. Also conserving, and the boundary
condition that makes the discrete problem translation invariant.
"""
struct Periodic <: BoundaryCondition end

"""
    Dirichlet(value = 0)

Edge cells are held at `value` for the whole run. Interior heat drains into the
boundary and disappears, so the total is *not* conserved — run
`examples/boundary_conditions.jl` to watch it decay.
"""
struct Dirichlet{T} <: BoundaryCondition
    value::T
end

Dirichlet() = Dirichlet(0.0f0)

# Neighbour index lookups. `clamp` gives the mirrored ghost cell that makes the
# flux across the edge zero; `mod1` wraps. Both are branch-light and work
# unchanged inside a GPU kernel.
@inline neighbor_index(::Neumann, i, n, offset) = clamp(i + offset, 1, n)
@inline neighbor_index(::Dirichlet, i, n, offset) = clamp(i + offset, 1, n)
@inline neighbor_index(::Periodic, i, n, offset) = mod1(i + offset, n)

"""
    conserves_state(bc) -> Bool

Whether `bc` keeps the sum over the grid constant. Used by the test suite to
decide which invariant to assert.
"""
conserves_state(::Neumann) = true
conserves_state(::Periodic) = true
conserves_state(::Dirichlet) = false

"""
    adapt_boundary(bc, ::Type{T})

Convert any stored value to the field element type so the boundary condition
stays `isbits` and does not drag a `Float64` into an otherwise `Float32` kernel.
"""
adapt_boundary(bc::Union{Neumann,Periodic}, ::Type{T}) where {T} = bc
adapt_boundary(bc::Dirichlet, ::Type{T}) where {T} = Dirichlet(convert(T, bc.value))
