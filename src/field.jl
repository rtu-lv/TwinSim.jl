"""
    Field2D(nx, ny; initial = 0.0f0)
    Field2D(matrix)

Double-buffered two-dimensional scalar field. The first index is `x`, the
second is `y`, matching the notation used in the course examples.

The struct is **mutable** so that [`swapbuffers!`](@ref) can exchange the two
buffers by rebinding fields. An immutable version forces a full array copy after
every step, which roughly doubles the memory traffic of a bandwidth-bound
stencil — measurable, and exactly the mistake this package is meant to teach
students to avoid.
"""
mutable struct Field2D{T,A<:AbstractMatrix{T}}
    current::A
    next::A
end

function Field2D(nx::Integer, ny::Integer; initial::T = 0.0f0) where {T}
    nx > 0 || throw(ArgumentError("nx must be positive, got $nx"))
    ny > 0 || throw(ArgumentError("ny must be positive, got $ny"))
    # Both buffers are initialised: leaving `next` as uninitialised memory means
    # any cell the stencil fails to write shows up as garbage rather than as an
    # obvious zero, which is a miserable bug to chase in a lab.
    return Field2D(fill(initial, nx, ny), fill(initial, nx, ny))
end

Field2D(current::AbstractMatrix) = Field2D(current, fill!(similar(current), zero(eltype(current))))

Base.summary(field::Field2D{T}) where {T} =
    string(size(field, 1), "x", size(field, 2), " Field2D{", T, "}")

function Base.show(io::IO, field::Field2D)
    print(io, summary(field))
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", field::Field2D)
    println(io, summary(field), ", current buffer:")
    show(io, MIME"text/plain"(), field.current)
    return nothing
end

Base.size(field::Field2D) = size(field.current)
Base.size(field::Field2D, dim::Integer) = size(field.current, dim)
Base.eltype(::Field2D{T}) where {T} = T
Base.length(field::Field2D) = length(field.current)
Base.getindex(field::Field2D, i::Integer, j::Integer) = field.current[i, j]
Base.setindex!(field::Field2D, value, i::Integer, j::Integer) = (field.current[i, j] = value)
Base.axes(field::Field2D) = axes(field.current)

# Single-cell access that also works on device memory.
#
# GPU array libraries disable scalar indexing on purpose: reading one element of
# a device array looks free in source and costs a full round trip at run time, so
# doing it in a loop silently destroys performance. Diagnostics like
# `center_value` genuinely need one element though, so they go through these
# helpers, which perform an explicit one-element copy instead of tripping the
# scalar-indexing error.
@inline scalar_at(data::Array, i::Integer, j::Integer) = data[i, j]
scalar_at(data::AbstractMatrix, i::Integer, j::Integer) = only(Array(@view data[i:i, j:j]))

@inline set_scalar!(data::Array, value, i::Integer, j::Integer) = (data[i, j] = value)

function set_scalar!(data::AbstractMatrix, value, i::Integer, j::Integer)
    copyto!(@view(data[i:i, j:j]), [convert(eltype(data), value)])
    return value
end

"""
    state(field) -> AbstractMatrix

The live buffer. On a GPU backend this is a device array; call `Array(state(f))`
to bring it back to the host.
"""
state(field::Field2D) = field.current

"""
    swapbuffers!(field) -> field

Exchange the `current` and `next` buffers. This is a rebinding of two
references, not a copy: it is O(1) regardless of grid size.
"""
function swapbuffers!(field::Field2D)
    field.current, field.next = field.next, field.current
    return field
end

"""
    initialize_peak!(field, value = 100.0f0; x, y) -> field

Put a single hot cell in the field, by default at the centre. The classic
"drop of heat in a cold plate" initial condition.
"""
function initialize_peak!(field::Field2D, value = 100.0f0;
                          x::Integer = cld(size(field, 1), 2),
                          y::Integer = cld(size(field, 2), 2))
    checkbounds(field.current, x, y)
    set_scalar!(field.current, value, x, y)
    return field
end

"""
    initialize_gaussian!(field; amplitude = 1, sigma = ..., x0, y0, dx = 1, dy = 1) -> field

Seed the field with a Gaussian bump. Diffusion has a closed-form solution for a
Gaussian initial condition — it stays Gaussian and its variance grows as
`sigma^2 + 2*alpha*t` — which is what `test/runtests.jl` uses to check the
discretisation against the analytical solution rather than against itself.
"""
function initialize_gaussian!(field::Field2D{T};
                              amplitude = one(T),
                              sigma = T(max(size(field, 1), size(field, 2)) / 16),
                              x0 = (size(field, 1) + 1) / 2,
                              y0 = (size(field, 2) + 1) / 2,
                              dx = one(T),
                              dy = one(T)) where {T}
    nx, ny = size(field)
    data = field.current
    inv_two_sigma_sq = T(1 / (2 * sigma^2))
    @inbounds for j in 1:ny, i in 1:nx
        rx = T((i - x0) * dx)
        ry = T((j - y0) * dy)
        data[i, j] = T(amplitude) * exp(-(rx * rx + ry * ry) * inv_two_sigma_sq)
    end
    fill!(field.next, zero(T))
    return field
end

"""
    sum_state(field) -> Number

Total of the live buffer. For a conserving boundary condition this is the
quantity that must stay constant.
"""
sum_state(field::Field2D) = sum(field.current)

"""
    center_value(field) -> Number

Value at the centre cell.
"""
function center_value(field::Field2D)
    x = cld(size(field, 1), 2)
    y = cld(size(field, 2), 2)
    return scalar_at(field.current, x, y)
end
