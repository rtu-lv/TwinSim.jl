"""
    Heat2DParams(; alpha = 0.15f0, dt = 0.1f0, dx = 1.0f0, dy = 1.0f0)

Parameters of the explicit (FTCS) finite-difference heat equation.

All four values share one element type. Arguments of mixed types are promoted,
so `Heat2DParams(alpha = 0.2)` widens the `Float32` defaults to `Float64`
instead of failing to find a constructor.
"""
Base.@kwdef struct Heat2DParams{T}
    alpha::T = 0.15f0
    dt::T = 0.1f0
    dx::T = 1.0f0
    dy::T = 1.0f0
end

# The auto-generated `Heat2DParams(::T, ::T, ::T, ::T) where T` is more specific
# than this method, so same-type arguments never reach it and there is no
# recursion; mixed-type arguments land here and get promoted.
Heat2DParams(alpha, dt, dx, dy) = Heat2DParams(promote(alpha, dt, dx, dy)...)

Base.eltype(::Heat2DParams{T}) where {T} = T

"""
    cfl_number(params) -> Real

Stability parameter of the explicit scheme, `alpha * dt * (1/dx^2 + 1/dy^2)`.

The two-dimensional FTCS discretisation is stable only for values `<= 1/2`.
Above that the solution does not merely lose accuracy, it grows without bound
and reaches `NaN` within a few dozen steps.
"""
cfl_number(params::Heat2DParams) =
    params.alpha * params.dt * (inv(params.dx^2) + inv(params.dy^2))

"""
    is_stable(params) -> Bool

Whether `cfl_number(params) <= 1/2`.
"""
is_stable(params::Heat2DParams) = cfl_number(params) <= 0.5

"""
    max_stable_dt(params) -> Real

Largest `dt` that keeps the scheme stable for the given `alpha`, `dx` and `dy`.
"""
max_stable_dt(params::Heat2DParams) =
    oftype(params.dt, 0.5 / (params.alpha * (inv(params.dx^2) + inv(params.dy^2))))

cfl_number(model) = cfl_number(model.params)

"""
    stability_number(model) -> Real
    stability_number(params, source) -> Real

The quantity that must stay at or below `1/2` for the explicit scheme to be
stable. This — not [`cfl_number`](@ref) — is what `Heat2D` checks.

```
stability_number = alpha*dt*(1/dx^2 + 1/dy^2) + dt*g/4
                   \\_______ diffusion _______/   \\_ feedback _/
```

where `g` is [`feedback_coefficient`](@ref), the largest `|dq/du|` the source
contributes. For every source that does not read the state, `g` is zero and this
reduces exactly to the CFL number.

A [`ProportionalSource`](@ref) does read the state, and its gain enters the von
Neumann analysis: the amplification factor becomes `1 - dt*g - D` where
`D <= 4*(cx + cy)`, so `|G| <= 1` requires `dt*g + 4*(cx + cy) <= 2`. Dividing by
four gives the form above.

The practical consequence: a strong enough controller destabilises the scheme on
its own, at a time step the diffusion alone would tolerate comfortably. Reported
by `cfl_number` this looks perfectly safe, which is why the model checks this
instead.
"""
stability_number(params::Heat2DParams, source::SourceTerm) =
    cfl_number(params) + params.dt * feedback_coefficient(source) / 4

stability_number(params::Heat2DParams) = cfl_number(params)
stability_number(model) = stability_number(model.params, model.source)

is_stable(model) = stability_number(model) <= 0.5

"""
    max_stable_dt(model) -> Real
    max_stable_dt(params, source) -> Real

The largest `dt` that keeps the scheme stable, accounting for any state-dependent
forcing.
"""
function max_stable_dt(params::Heat2DParams, source::SourceTerm)
    diffusion = params.alpha * (inv(params.dx^2) + inv(params.dy^2))
    feedback = feedback_coefficient(source) / 4
    return oftype(params.dt, 0.5 / (diffusion + feedback))
end

max_stable_dt(model) = max_stable_dt(model.params, model.source)

"""
    diffusion_coefficients(params) -> (cx, cy)

The two per-axis stencil weights, computed once on the host rather than once per
cell inside the kernel.
"""
@inline function diffusion_coefficients(params::Heat2DParams{T}) where {T}
    return (params.alpha * params.dt / (params.dx * params.dx),
            params.alpha * params.dt / (params.dy * params.dy))
end

"""
    Heat2D(field; boundary = Neumann(), check_stability = true, kwargs...)
    Heat2D(; nx = 128, ny = nx, initial = 0.0f0, boundary = Neumann(), kwargs...)

Two-dimensional heat diffusion model: a double-buffered [`Field2D`](@ref), a set
of [`Heat2DParams`](@ref) and a [`BoundaryCondition`](@ref).

The parameter element type follows the field element type, so
`Heat2D(nx = 128, initial = 0.0)` gives a consistently `Float64` model and
`initial = 0.0f0` (the default) gives a `Float32` one.

`check_stability = false` disables the CFL check, which is how the lab on
numerical stability produces a deliberately diverging run.
"""
struct Heat2D{T,F<:Field2D{T},P<:Heat2DParams{T},B<:BoundaryCondition,S<:SourceTerm}
    field::F
    params::P
    boundary::B
    source::S

    # Written out rather than relying on the auto-generated constructor: `T` only
    # appears inside the other type parameters, so spelling the inference out
    # keeps the error message readable when a field and its parameters disagree.
    function Heat2D(field::Field2D{T}, params::Heat2DParams{T},
                    boundary::BoundaryCondition,
                    source::SourceTerm = NoSource()) where {T}
        return new{T,typeof(field),typeof(params),typeof(boundary),typeof(source)}(
            field, params, boundary, source)
    end
end

function Heat2D(field::Field2D{T};
                boundary::BoundaryCondition = Neumann(),
                source::SourceTerm = NoSource(),
                check_stability::Bool = true,
                kwargs...) where {T}
    params = Heat2DParams{T}(; kwargs...)
    check_source_shape(source, size(field))

    number = stability_number(params, source)
    if check_stability && number > 0.5
        feedback = feedback_coefficient(source)
        detail = iszero(feedback) ? "" : """

            The source contributes to this. Its feedback coefficient is $feedback, adding
            dt*g/4 = $(params.dt * feedback / 4) on top of the diffusion term
            $(cfl_number(params)). A state-dependent source such as ProportionalSource
            destabilises the scheme on its own: reducing the gain is as valid a fix as
            reducing dt."""

        throw(ArgumentError("""
            Unstable configuration: stability number is $number, which exceeds the
            explicit-scheme limit of 0.5. The run would diverge to NaN.$detail

            Fix it by reducing dt to at most $(max_stable_dt(params, source)), reducing
            alpha, coarsening the grid$(iszero(feedback) ? "" : ", or lowering the gain").
            To explore the instability on purpose, construct the model with
            `check_stability = false`.
            """))
    end
    return Heat2D(field, params, adapt_boundary(boundary, T), source)
end


function Heat2D(; nx::Integer = 128, ny::Integer = nx, initial = 0.0f0, kwargs...)
    return Heat2D(Field2D(nx, ny; initial); kwargs...)
end

Base.size(model::Heat2D) = size(model.field)
Base.size(model::Heat2D, dim::Integer) = size(model.field, dim)
Base.eltype(::Heat2D{T}) where {T} = T

state(model::Heat2D) = state(model.field)
sum_state(model::Heat2D) = sum_state(model.field)
center_value(model::Heat2D) = center_value(model.field)
"""
    conserves_state(model) -> Bool

Whether the total over the grid must stay constant. True only for a conserving
boundary condition *and* no forcing: a source adds heat the domain did not start
with, so a driven model has no conservation invariant to test against.
"""
conserves_state(model::Heat2D) =
    conserves_state(model.boundary) && model.source isa NoSource

"""
    is_driven(model) -> Bool

Whether anything in the model depends on simulated time. A driven model's
results depend on *when* it was run, not only on how many steps.
"""
is_driven(model::Heat2D) = is_driven(model.boundary) || is_driven(model.source)

"""
    bytes_per_cell(model) -> Int

Compulsory memory traffic per cell update: read the old value, write the new
one. Neighbour reads are assumed to be cache hits, which is the convention used
when quoting stencil bandwidth.
"""
bytes_per_cell(::Heat2D{T}) where {T} = 2 * sizeof(T)

"""
    flops_per_cell(model) -> Int

Floating-point operations per cell update for the five-point stencil:
per axis one multiply for `2u`, two additions for the second difference and one
multiply by the coefficient (4), twice over (8), plus two additions to combine
with the centre value.
"""
flops_per_cell(::Heat2D) = 10

Base.summary(model::Heat2D{T}) where {T} =
    string(size(model, 1), "x", size(model, 2), " Heat2D{", T, "} (", nameof(typeof(model.boundary)), ")")

function Base.show(io::IO, model::Heat2D)
    print(io, summary(model))
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", model::Heat2D)
    nx, ny = size(model)
    println(io, "Heat2D{", eltype(model), "} ", nx, "x", ny)
    println(io, "  alpha = ", model.params.alpha, ", dt = ", model.params.dt,
                ", dx = ", model.params.dx, ", dy = ", model.params.dy)
    println(io, "  boundary  ", model.boundary,
                conserves_state(model.boundary) ? " (conserving)" : " (not conserving)")
    model.source isa NoSource ||
        println(io, "  source    ", model.source, is_driven(model.source) ? " (time-varying)" : "")
    is_driven(model.boundary) && println(io, "  boundary is time-varying")
    print(io, "  CFL       ", cfl_number(model), is_stable(model) ? " (stable)" : " (UNSTABLE)")
    return nothing
end
