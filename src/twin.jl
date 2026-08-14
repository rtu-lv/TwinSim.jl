# The pieces that make this a *twin* rather than a batch simulation: getting
# measurements in, and getting state on and off disk.

"""
    Sensor(i, j, value)

A single point measurement of the real system, located at grid cell `(i, j)`.
"""
struct Sensor{T}
    i::Int
    j::Int
    value::T
end

"""
    nudge!(model, sensors; gain = 0.5, radius = 0) -> model

Pull the simulated state towards measured values:

    u[i, j] += gain * w(distance) * (measured - u[i, j])

`gain = 0` ignores the sensors, `gain = 1` with `radius = 0` overwrites the
measured cells outright (direct insertion), and values in between trade trust in
the model against trust in the instrument.

`radius` is the **localisation radius**. With `radius = 0` each sensor corrects
exactly one cell, which is the honest but nearly useless scheme: a handful of
point corrections cannot fix a field that is wrong everywhere, and diffusion
spreads them far too slowly to keep up. With `radius > 0` each correction is
applied to the neighbourhood with Gaussian weight `exp(-d^2 / 2 radius^2)`,
truncated at `3 * radius`. That is the essential idea behind optimal
interpolation and the localisation step of an ensemble Kalman filter: a
measurement tells you about a *region*, not a point, and how large that region
is depends on the physics.

Overlapping corrections are accumulated before being applied, so the result does
not depend on the order of `sensors`.

The update writes individual cells, so it runs on host memory. With a GPU
backend, assimilate between `run!` calls — which is also how a real twin works,
since measurements arrive far more slowly than time steps.
"""
function nudge!(model::Heat2D{T}, sensors; gain::Real = 0.5, radius::Real = 0) where {T}
    0 <= gain <= 1 || throw(ArgumentError("gain must be in [0, 1], got $gain"))
    radius >= 0 || throw(ArgumentError("radius must be non-negative, got $radius"))
    data = model.field.current
    data isa Array ||
        throw(ArgumentError("nudge! needs host memory; call sync_to_host! or assimilate between run! calls"))

    nx, ny = size(data)
    for sensor in sensors
        checkbounds(data, sensor.i, sensor.j)
    end

    if iszero(radius)
        g = convert(T, gain)
        for sensor in sensors
            @inbounds data[sensor.i, sensor.j] +=
                g * (convert(T, sensor.value) - data[sensor.i, sensor.j])
        end
        return model
    end

    # Accumulate weighted increments first, so that overlapping sensor
    # neighbourhoods combine additively instead of each seeing the previous
    # sensor's correction.
    increment = zeros(T, nx, ny)
    weight = zeros(T, nx, ny)
    reach = ceil(Int, 3 * radius)
    inv_two_r2 = 1 / (2 * radius^2)

    for sensor in sensors
        measured = convert(T, sensor.value)
        for j in max(1, sensor.j - reach):min(ny, sensor.j + reach),
            i in max(1, sensor.i - reach):min(nx, sensor.i + reach)

            d2 = (i - sensor.i)^2 + (j - sensor.j)^2
            w = convert(T, exp(-d2 * inv_two_r2))
            @inbounds increment[i, j] += w * (measured - data[i, j])
            @inbounds weight[i, j] += w
        end
    end

    g = convert(T, gain)
    @inbounds for idx in eachindex(data)
        weight[idx] > 0 || continue
        data[idx] += g * increment[idx] / weight[idx]
    end
    return model
end

# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
#
# A small explicit binary format rather than a serialisation library: it is
# stable across Julia versions, it is readable from C++ or Python in a few
# lines, and the header is short enough to show on a slide.
#
#   offset  size  contents
#   0       8     magic "VTSIM01\n"
#   8       1     element type tag (1 = Float32, 2 = Float64)
#   9       1     boundary tag (1 = Neumann, 2 = Periodic, 3 = Dirichlet)
#   10      8     nx                     (Int64, little endian)
#   18      8     ny                     (Int64)
#   26      8     step                   (Int64)
#   34      8     simulated time         (Float64)
#   42      8     Dirichlet value        (Float64, 0 otherwise)
#   50      *     nx*ny values, column major

const CHECKPOINT_MAGIC = b"VTSIM01\n"

const ELTYPE_TAGS = Dict{DataType,UInt8}(Float32 => 0x01, Float64 => 0x02)
const TAG_ELTYPES = Dict{UInt8,DataType}(0x01 => Float32, 0x02 => Float64)

boundary_tag(::Neumann) = 0x01
boundary_tag(::Periodic) = 0x02
boundary_tag(::Dirichlet) = 0x03

boundary_from_tag(tag::UInt8, value::Float64, ::Type{T}) where {T} =
    tag == 0x01 ? Neumann() :
    tag == 0x02 ? Periodic() :
    tag == 0x03 ? Dirichlet(convert(T, value)) :
    throw(ArgumentError("unknown boundary tag $tag in checkpoint"))

"""
    save_state(path, model; step = 0, simulated_time = 0.0)

Write the live buffer and enough metadata to restart from it. GPU-resident
models are copied to the host first.

See [`load_state`](@ref) for reading it back.
"""
function save_state(path::AbstractString, model::Heat2D{T};
                    step::Integer = 0, simulated_time::Real = model.clock[]) where {T}
    haskey(ELTYPE_TAGS, T) ||
        throw(ArgumentError("checkpoints support Float32 and Float64, got $T"))
    data = model.field.current
    host_data = data isa Array ? data : Array(data)
    nx, ny = size(model.field)
    dirichlet_value = model.boundary isa Dirichlet ? Float64(model.boundary.value) : 0.0

    open(path, "w") do io
        write(io, CHECKPOINT_MAGIC)
        write(io, ELTYPE_TAGS[T])
        write(io, boundary_tag(model.boundary))
        write(io, Int64(nx))
        write(io, Int64(ny))
        write(io, Int64(step))
        write(io, Float64(simulated_time))
        write(io, dirichlet_value)
        write(io, host_data)
    end
    return path
end

"""
    load_state(path) -> (; field, boundary, step, simulated_time)

Read a checkpoint written by [`save_state`](@ref). Rebuild a model with

    cp = load_state("state.vts")
    model = Heat2D(cp.field; boundary = cp.boundary, alpha = ..., dt = ...)

Parameters are deliberately *not* stored: a restart usually wants to change
them, and silently reusing the old ones hides that choice.
"""
function load_state(path::AbstractString)
    open(path, "r") do io
        magic = read(io, length(CHECKPOINT_MAGIC))
        magic == CHECKPOINT_MAGIC ||
            throw(ArgumentError("$path is not a VisuTwinSim checkpoint (bad magic)"))
        tag = read(io, UInt8)
        haskey(TAG_ELTYPES, tag) || throw(ArgumentError("unknown element type tag $tag in $path"))
        T = TAG_ELTYPES[tag]
        btag = read(io, UInt8)
        nx = Int(read(io, Int64))
        ny = Int(read(io, Int64))
        step = Int(read(io, Int64))
        simulated_time = read(io, Float64)
        dirichlet_value = read(io, Float64)

        data = Array{T}(undef, nx, ny)
        read!(io, data)
        eof(io) || throw(ArgumentError("$path contains more data than its header describes"))

        return (field = Field2D(data),
                boundary = boundary_from_tag(btag, dirichlet_value, T),
                step = step,
                simulated_time = simulated_time)
    end
end

"""
    checkpoint_callback(path_pattern; every = 100)

Build a `run!` callback that writes a checkpoint every `every` steps.
`path_pattern` is formatted with the step number:

    run!(sim; callback = checkpoint_callback("out/state_%06d.vts"), callback_every = 100)
"""
function checkpoint_callback(path_pattern::AbstractString; every::Integer = 1)
    return function (model, state)
        state.step % every == 0 || return nothing
        save_state(Printf.format(Printf.Format(path_pattern), state.step), model;
                   step = state.step, simulated_time = state.simulated_time)
        return nothing
    end
end
