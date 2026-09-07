"""
    AbstractModel

The contract a simulation model implements in order to use the runtime.

`Heat2D` is the reference implementation, but nothing in `run!`, the stop
conditions, the metrics, the callbacks, the checkpointing or the twin loop is
specific to it. A model that satisfies this interface gets all of that for free,
which is the point: contributing a new model should mean writing the *physics*,
not re-implementing the runtime around it.

## Required

Four methods. Without them the runtime cannot step, measure or schedule.

| Method | Meaning |
|:-------|:--------|
| `step!(backend, model, t)` | advance one time step, at simulated time `t` |
| `state(model)` | the live state array (host or device) |
| `timestep(model)` | the time step, `dt` |
| `clock(model)` | a `Ref{Float64}` holding the simulated clock |

`clock` returns a `Ref` rather than a value because the runtime advances it, and
because a device-resident copy of the model must share the same clock object.

## Optional

Everything else has a default. Override where the default is wrong for your model.

| Method | Default | Override when |
|:-------|:--------|:--------------|
| `cells(model)` | `length(state(model))` | state is not one array per cell |
| `bytes_per_cell(model)` | `2 * sizeof(eltype)` | your stencil moves more per cell |
| `flops_per_cell(model)` | `0` | you want a meaningful GFLOP/s figure |
| `sum_state(model)` | `sum(state(model))` | the total is not a plain sum |
| `move_to_device(model, device)` | errors | you want GPU support |
| `sync_state!(host, device)` | `copyto!` of `state` | state is more than one array |

Leaving `move_to_device` unimplemented is a legitimate choice: the model is then
CPU-only, and using it with a GPU backend fails with a message saying so rather
than silently computing on the wrong memory.

## A minimal model

```julia
using TwinSim
import TwinSim: step!, state, timestep, clock

struct Decay{T} <: AbstractModel
    u::Vector{T}
    rate::T
    dt::T
    clock::Base.RefValue{Float64}
end

Decay(n; rate = 0.1f0, dt = 0.05f0) = Decay(ones(Float32, n), rate, dt, Ref(0.0))

state(m::Decay) = m.u
timestep(m::Decay) = m.dt
clock(m::Decay) = m.clock

function step!(::CPUBackend, m::Decay, t = 0.0)
    @. m.u -= m.dt * m.rate * m.u
    return m
end

run!(Decay(1000); steps = 100)          # metrics, stop conditions, callbacks: all work
```

Call [`check_model_interface`](@ref) to confirm a model satisfies the contract
before wiring it into anything larger.
"""
abstract type AbstractModel end

"""
    state(model) -> AbstractArray

The live state array. On a GPU backend this is device memory; wrap in `Array` to
bring it to the host.

**Required** for a custom model.
"""
function state end

"""
    timestep(model) -> Real

The model's time step. Used to advance the simulated clock and by the
time-based stop conditions.

**Required** for a custom model.
"""
function timestep end

"""
    clock(model) -> Base.RefValue{Float64}

The model's simulated clock, as a `Ref` so the runtime can advance it and a
device-resident copy can share it.

**Required** for a custom model. The usual implementation stores a
`Base.RefValue{Float64}` field initialised to `Ref(0.0)` and returns it.
"""
function clock end

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

"""
    cells(model) -> Int

Number of cells updated per step. Sets the denominator of the throughput figures.
"""
cells(model::AbstractModel) = length(state(model))

"""
    bytes_per_cell(model) -> Int

Compulsory memory traffic per cell update. The default assumes one read and one
write of a single value, which is right for an explicit stencil over one field.
"""
bytes_per_cell(model::AbstractModel) = 2 * sizeof(eltype(state(model)))

"""
    flops_per_cell(model) -> Int

Floating-point operations per cell update. Defaults to `0`, which makes
`gflops(metrics)` report zero rather than a wrong number — an honest default,
since only the model's author knows the real count.
"""
flops_per_cell(::AbstractModel) = 0

sum_state(model::AbstractModel) = sum(state(model))
simulated_time(model::AbstractModel) = clock(model)[]

function reset_clock!(model::AbstractModel, t::Real = 0.0)
    clock(model)[] = Float64(t)
    return model
end

Base.eltype(model::AbstractModel) = eltype(state(model))

"""
    move_to_device(model, device) -> (model, bytes)

Return a copy of `model` whose arrays live in `device` memory, together with the
number of bytes transferred.

The default refuses rather than guessing: a generic implementation cannot know
which of a model's fields are grid-sized arrays and which are parameters, and
silently leaving state in host memory while launching device kernels produces
wrong answers rather than an error.
"""
function move_to_device(model::AbstractModel, device)
    throw(ArgumentError("""
        $(nameof(typeof(model))) does not support GPU backends.

        It is a CPU-only model, which is a perfectly valid thing to be. To add GPU
        support, implement

            TwinSim.move_to_device(model::$(nameof(typeof(model))), device)

        returning a copy of the model with its arrays allocated on `device`
        (see `KernelAbstractions.allocate`) and the number of bytes copied.
        """))
end

"""
    sync_state!(host, device_model) -> bytes

Copy the live state from a device-resident model back into the host model,
returning the bytes moved. The default copies `state` and is right whenever the
state is a single array.
"""
function sync_state!(host::AbstractModel, device_model::AbstractModel)
    copyto!(state(host), state(device_model))
    return sizeof(eltype(state(host))) * length(state(host))
end

"""
    check_model_interface(model; verbose = true, io = stdout) -> Bool

Check that `model` satisfies the [`AbstractModel`](@ref) contract, reporting what
is missing rather than failing later inside `run!`.

```julia
check_model_interface(MyModel(...))
```

Intended for anyone writing a model against this interface: run it first, and the
errors you get will name the method to implement instead of surfacing as an
unrelated failure three layers down.
"""
function check_model_interface(model; verbose::Bool = true, io::IO = stdout)
    problems = String[]

    for (name, call) in (("state", () -> state(model)),
                         ("timestep", () -> timestep(model)),
                         ("clock", () -> clock(model)))
        try
            call()
        catch err
            push!(problems, "$name(model) failed: $(sprint(showerror, err))")
        end
    end

    if isempty(problems)
        clock(model) isa Base.RefValue{Float64} ||
            push!(problems, "clock(model) must return a Base.RefValue{Float64}, got $(typeof(clock(model)))")
        timestep(model) isa Real ||
            push!(problems, "timestep(model) must return a Real, got $(typeof(timestep(model)))")
        state(model) isa AbstractArray ||
            push!(problems, "state(model) must return an AbstractArray, got $(typeof(state(model)))")
    end

    hasmethod(step!, Tuple{CPUBackend,typeof(model),Float64}) ||
        hasmethod(step!, Tuple{CPUBackend,typeof(model)}) ||
        push!(problems, "no step!(::CPUBackend, ::$(typeof(model)), t) method")

    if verbose
        if isempty(problems)
            printstyled(io, "$(nameof(typeof(model))) satisfies the AbstractModel interface\n"; color = :green)
            println(io, "  cells           ", cells(model))
            println(io, "  bytes per cell  ", bytes_per_cell(model))
            println(io, "  flops per cell  ", flops_per_cell(model),
                    flops_per_cell(model) == 0 ? "   (default; override for a GFLOP/s figure)" : "")
            println(io, "  GPU support     ",
                    hasmethod(move_to_device, Tuple{typeof(model),Any}) &&
                    which(move_to_device, Tuple{typeof(model),Any}).sig !==
                        which(move_to_device, Tuple{AbstractModel,Any}).sig ?
                    "yes" : "no (CPU-only)")
        else
            printstyled(io, "$(nameof(typeof(model))) does not satisfy the AbstractModel interface\n";
                        color = :red)
            foreach(p -> println(io, "  - ", p), problems)
        end
    end
    return isempty(problems)
end
