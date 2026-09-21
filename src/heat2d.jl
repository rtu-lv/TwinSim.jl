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
    check_parameters(params)

Reject parameters no heat model can run with, naming the offending one. Called
by the `Heat2D` constructor before the stability check, because the stability
number of a nonsensical configuration is itself nonsense: a negative `dt` gives
a negative number that passes `<= 1/2`, and a `NaN` compares false with
everything, so neither would otherwise be caught.

`check_stability = false` does not switch this off. An unstable `dt` is a valid
experiment; a negative one is a typing error.
"""
function check_parameters(params::Heat2DParams)
    # Written as "is it good?" rather than "is it bad?" so that NaN fails too:
    # every comparison with NaN is false.
    params.alpha >= 0 && isfinite(params.alpha) ||
        throw(ArgumentError("alpha must be finite and non-negative, got $(params.alpha)"))
    params.dt > 0 && isfinite(params.dt) ||
        throw(ArgumentError("dt must be finite and positive, got $(params.dt)"))
    params.dx > 0 && isfinite(params.dx) ||
        throw(ArgumentError("dx must be finite and positive, got $(params.dx)"))
    params.dy > 0 && isfinite(params.dy) ||
        throw(ArgumentError("dy must be finite and positive, got $(params.dy)"))
    return nothing
end

# ---------------------------------------------------------------------------
# Stability of the explicit scheme
# ---------------------------------------------------------------------------

"""
    cfl_number(params) -> Real
    cfl_number(model) -> Real

Stability parameter of the explicit scheme, `alpha * dt * (1/dx^2 + 1/dy^2)`,
which is `cx + cy` in terms of [`diffusion_coefficients`](@ref).

The two-dimensional FTCS discretisation is stable only for values `<= 1/2`.
Above that the solution does not merely lose accuracy: the fastest-varying
pattern on the grid is multiplied by `|1 - 4*(cx + cy)| > 1` every step, so it
grows exponentially and eventually overflows to `Inf` and `NaN`. How soon
depends on how far above the limit the run is: about 490 steps at `1.1` times
the largest stable `dt`, fewer than 100 at twice it.
"""
cfl_number(params::Heat2DParams) =
    params.alpha * params.dt * (inv(params.dx^2) + inv(params.dy^2))

cfl_number(model) = cfl_number(model.params)

"""
    is_stable(params) -> Bool
    is_stable(model) -> Bool

Whether the configuration is within the stability limit of the explicit scheme:
`cfl_number(params) <= 1/2` for bare parameters, and
`stability_number(model) <= 1/2` for a model, which also accounts for a
state-dependent source.
"""
is_stable(params::Heat2DParams) = cfl_number(params) <= 0.5
is_stable(model) = stability_number(model) <= 0.5

"""
    diffusion_coefficients(params) -> (cx, cy)

The two per-axis stencil weights, `alpha*dt/dx^2` and `alpha*dt/dy^2`, computed
once on the host rather than once per cell inside the kernel.
"""
@inline function diffusion_coefficients(params::Heat2DParams)
    return (params.alpha * params.dt / (params.dx * params.dx),
            params.alpha * params.dt / (params.dy * params.dy))
end

# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

"""
    Heat2D(; nx = 128, ny = nx, initial = 0.0f0, kwargs...)
    Heat2D(field; kwargs...)

Two-dimensional heat diffusion model: a double-buffered [`Field2D`](@ref), a set
of [`Heat2DParams`](@ref), a [`BoundaryCondition`](@ref) and an optional
[`SourceTerm`](@ref).

Keyword arguments:

- `nx`, `ny` – grid size in cells (first form only).
- `initial` – value every cell starts with (first form only). Its type sets the
  element type of the whole model: `0.0f0` (the default) gives `Float32`,
  `0.0` gives `Float64`. An integer such as `20` is read as `20.0`, so it also
  gives `Float64`; write `20f0` for a `Float32` model.
- `alpha`, `dt`, `dx`, `dy` – diffusivity, time step and grid spacings, see
  [`Heat2DParams`](@ref). They are converted to the field element type.
- `boundary` – [`Neumann`](@ref) (the default), [`Periodic`](@ref) or
  [`Dirichlet`](@ref).
- `source` – a [`SourceTerm`](@ref); [`NoSource`](@ref) by default.
- `check_stability` – `true` by default, so a configuration above the stability
  limit is rejected with an `ArgumentError`. `false` disables that check, which
  is how a stability experiment produces a deliberately diverging run.

Parameters that are invalid rather than unstable (a negative or zero `dt`, a
negative or `NaN` `alpha`, a non-positive grid spacing) are always rejected.
"""
struct Heat2D{T,F<:Field2D{T},P<:Heat2DParams{T},B<:BoundaryCondition,S<:SourceTerm} <: AbstractModel
    field::F
    params::P
    boundary::B
    source::S
    # The model's own simulated clock, so that successive `run!` calls continue
    # rather than restarting. A `Ref` keeps the struct immutable while letting the
    # clock advance — and lets a device-resident copy share the same clock object.
    #
    # Without a persistent clock a driven model advanced in windows would replay
    # the same slice of its drive forever: `run!(m; steps=3)` three times would
    # evaluate the boundary at t = 0.0, 0.1, 0.2 on all three calls. Every
    # windowed pattern in the package — `twin_run!` above all — depends on it.
    clock::Base.RefValue{Float64}

    # Written out rather than relying on the auto-generated constructor: `T` only
    # appears inside the other type parameters, so spelling the inference out
    # keeps the error message readable when a field and its parameters disagree.
    function Heat2D(field::Field2D{T}, params::Heat2DParams{T},
                    boundary::BoundaryCondition,
                    source::SourceTerm = NoSource(),
                    clock::Base.RefValue{Float64} = Ref(0.0)) where {T}
        return new{T,typeof(field),typeof(params),typeof(boundary),typeof(source)}(
            field, params, boundary, source, clock)
    end
end

function Heat2D(field::Field2D{T};
                boundary::BoundaryCondition = Neumann(),
                source::SourceTerm = NoSource(),
                check_stability::Bool = true,
                kwargs...) where {T}
    T <: AbstractFloat || throw(ArgumentError(
        "Heat2D needs a floating-point field, got element type $T. " *
        "Use a literal such as `initial = 20f0` rather than `initial = 20`."))
    params = Heat2DParams{T}(; kwargs...)
    check_parameters(params)
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
    # `float` turns an integer such as `initial = 20` into 20.0 and leaves a
    # floating-point value, and with it the chosen precision, untouched.
    return Heat2D(Field2D(nx, ny; initial = float(initial)); kwargs...)
end

# ---------------------------------------------------------------------------
# Stability with a state-dependent source
# ---------------------------------------------------------------------------
#
# For a model without a source, or with one that does not read the state, this
# section reduces to the CFL number above and can be skipped on a first reading.

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

"""
    max_stable_dt(params) -> Real
    max_stable_dt(params, source) -> Real
    max_stable_dt(model) -> Real

The largest `dt` that keeps the scheme stable for the given `alpha`, `dx` and
`dy`, accounting for any state-dependent forcing.

The result is guaranteed to pass the check it describes: a model built with
`dt = max_stable_dt(...)` is accepted by the constructor. The exact quotient does
not always have that property, because rounding it to the parameter type can
land just above the limit (`cfl_number` of `0.50000006f0`), so the value is
stepped down to the nearest representable `dt` that the check accepts.
"""
function max_stable_dt(params::Heat2DParams, source::SourceTerm = NoSource())
    rate = params.alpha * (inv(params.dx^2) + inv(params.dy^2)) +
           feedback_coefficient(source) / 4
    # Nothing diffuses and nothing feeds back: every dt is stable.
    rate > 0 || return oftype(params.dt, Inf)

    dt = oftype(params.dt, 0.5 / rate)
    while stability_number(Heat2DParams(params.alpha, dt, params.dx, params.dy), source) > 0.5
        dt = prevfloat(dt)
    end
    return dt
end

max_stable_dt(model) = max_stable_dt(model.params, model.source)

Base.size(model::Heat2D) = size(model.field)
Base.size(model::Heat2D, dim::Integer) = size(model.field, dim)
Base.eltype(::Heat2D{T}) where {T} = T

# The AbstractModel contract. Everything else the runtime needs has a default.
state(model::Heat2D) = state(model.field)
timestep(model::Heat2D) = model.params.dt
clock(model::Heat2D) = model.clock
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
    # The verdict is about `stability_number`, which is what the constructor
    # checks. It differs from the CFL number only for a state-dependent source,
    # and then both are shown so the verdict is not read against the wrong one.
    verdict = is_stable(model) ? " (stable)" : " (UNSTABLE)"
    if stability_number(model) == cfl_number(model)
        print(io, "  CFL       ", cfl_number(model), verdict)
    else
        println(io, "  CFL       ", cfl_number(model))
        print(io, "  stability ", stability_number(model), verdict, ", including the source feedback")
    end
    return nothing
end

"""
    move_to_device(model::Heat2D, device) -> (model, bytes)

Relocate the field and any grid-sized source arrays onto `device`, sharing the
clock so host and device copies stay on the same simulated time.
"""
function move_to_device(model::Heat2D{T}, device) where {T}
    current = KernelAbstractions.allocate(device, T, size(model.field))
    next = KernelAbstractions.allocate(device, T, size(model.field))
    copyto!(current, model.field.current)

    # A PatternSource carries a grid-sized array of its own, which has to travel
    # with the field or the kernel would index host memory from the device.
    # CombinedSource may hold several, so the move recurses.
    source, source_bytes = move_source_to_device(model.source, device)
    bytes = sizeof(T) * length(model.field) + source_bytes

    device_model = Heat2D(Field2D(current, next), model.params, model.boundary,
                          source, model.clock)
    return (device_model, bytes)
end
