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

`Neumann` and `Periodic` are implemented by changing how a neighbour index is
computed (`neighbor_index`), so the stencil formula itself is the same for both.
`Dirichlet` does not touch the indices: its edge cells are simply set to the
boundary value, and every other cell has all four neighbours inside the grid.
The three cases are the methods of `heat_update` in `src/kernels.jl`.
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

Edge cells are held at `value`. Interior heat drains into the boundary and
disappears, so the total is *not* conserved — run
`examples/boundary_conditions.jl` to watch it decay.

`value` may also be a **callable of simulated time**, which is how a twin is
driven by its environment:

```julia
Dirichlet(t -> 5.0f0 + 10.0f0 * sin(2pi * t / 24))   # a daily cycle
Dirichlet(SampledSeries(hours, outdoor_temperatures))   # measured or forecast data
```

The callable is evaluated once per step on the host, and the resulting scalar is
what the kernel receives. See [`SampledSeries`](@ref).
"""
struct Dirichlet{T} <: BoundaryCondition
    value::T
end

Dirichlet() = Dirichlet(0.0f0)

# Dispatch on "not a number" rather than "is a Function": a callable struct such
# as SampledSeries does not subtype Function, and neither do most user-defined
# callables. A Dirichlet value is either a plain number or something to call.
#
# (Comments like this one sit *above* the docstring on purpose. A comment placed
# between a docstring and its definition detaches the docstring.)

"""
    is_driven(bc) -> Bool

Whether the boundary value depends on simulated time.
"""
is_driven(::BoundaryCondition) = false
is_driven(::Dirichlet) = true
is_driven(::Dirichlet{<:Number}) = false

"""
    resolve(bc, t, ::Type{T}) -> BoundaryCondition

Evaluate any time-dependent boundary value at simulated time `t` and convert to
the field element type, producing an `isbits` value fit to be a kernel argument.
"""
@inline resolve(bc::Union{Neumann,Periodic}, t, ::Type{T}) where {T} = bc
@inline resolve(bc::Dirichlet{<:Number}, t, ::Type{T}) where {T} = Dirichlet(convert(T, bc.value))
@inline resolve(bc::Dirichlet, t, ::Type{T}) where {T} = Dirichlet(convert(T, bc.value(t)))

"""
    neighbor_index(bc, i, n, offset) -> Int

Index of the neighbour `offset` (`-1` or `+1`) away from `i` along an axis of
length `n`, for the boundary conditions that are expressed through indexing.

- `Neumann`: `clamp` keeps the index inside `1:n`, so the neighbour "outside"
  an edge cell is the edge cell itself. The difference across the edge is then
  zero, which is the zero-flux condition.
- `Periodic`: `mod1` wraps the index round to the opposite edge.

Both are branch-light and work unchanged inside a GPU kernel. `Dirichlet` has no
method: a pinned edge cell is never updated by the stencil, so it never looks up
a neighbour.
"""
@inline neighbor_index(::Neumann, i, n, offset) = clamp(i + offset, 1, n)
@inline neighbor_index(::Periodic, i, n, offset) = mod1(i + offset, n)

function Base.show(io::IO, bc::Dirichlet)
    print(io, "Dirichlet(")
    is_driven(bc) ? print(io, "driven by ", bc.value) : print(io, bc.value)
    print(io, ")")
    return nothing
end

"""
    conserves_state(bc) -> Bool

Whether `bc` keeps the sum over the grid constant, which decides whether
"the total is conserved" is an invariant that may be asserted at all.
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
adapt_boundary(bc::Dirichlet{<:Number}, ::Type{T}) where {T} = Dirichlet(convert(T, bc.value))
# A time-varying value stays callable; it is converted when it is evaluated.
adapt_boundary(bc::Dirichlet, ::Type{T}) where {T} = bc
