"""
    StopCondition

When a run should end. A fixed step count is only the simplest case: a digital
twin usually runs until it reaches a point in time, until the state stops
changing, or until it runs out of wall-clock budget.

Combine several with [`AnyOf`](@ref), which stops as soon as any one of them
fires — the usual way to put a safety bound on an open-ended condition.
"""
abstract type StopCondition end

"""
    Steps(count)

Stop after exactly `count` steps.

The validation lives in an inner constructor. As an outer method it would be
shadowed by the compiler-generated `Steps(::Int)` for every `Int` argument,
which is why `Steps(-5)` used to be accepted.
"""
struct Steps <: StopCondition
    count::Int

    function Steps(count::Integer)
        count >= 0 || throw(ArgumentError("step count must be non-negative, got $count"))
        return new(Int(count))
    end
end

"""
    UntilTime(time)

Stop once the simulated time reaches `time`. Unlike a step count this is
independent of `dt`, so refining the time step no longer changes what is being
compared.
"""
struct UntilTime{T<:Real} <: StopCondition
    time::T

    function UntilTime(time::T) where {T<:Real}
        time >= 0 || throw(ArgumentError("stop time must be non-negative, got $time"))
        return new{T}(time)
    end
end

"""
    WallClock(seconds)

Stop after `seconds` of real time. Useful for "how far can this backend get in
one second?" comparisons.
"""
struct WallClock <: StopCondition
    seconds::Float64

    function WallClock(seconds::Real)
        seconds > 0 || throw(ArgumentError("wall-clock budget must be positive, got $seconds"))
        return new(Float64(seconds))
    end
end

"""
    Converged(tol; check_every = 10)

Stop once the largest change of any cell over `check_every` steps drops below
`tol`, i.e. once the field has reached steady state.

The check is a reduction over the whole grid, so `check_every` trades detection
latency against cost. Pair it with `Steps` inside [`AnyOf`](@ref) unless you are
certain the run converges.
"""
struct Converged{T<:Real} <: StopCondition
    tol::T
    check_every::Int

    function Converged(tol::T; check_every::Integer = 10) where {T<:Real}
        tol > 0 || throw(ArgumentError("tolerance must be positive, got $tol"))
        check_every > 0 || throw(ArgumentError("check_every must be positive, got $check_every"))
        return new{T}(tol, Int(check_every))
    end
end

"""
    AnyOf(conditions...)

Stop as soon as any of `conditions` fires. `AnyOf(Converged(1e-6), Steps(100_000))`
is the standard safe form of an open-ended run.
"""
struct AnyOf{C<:Tuple} <: StopCondition
    conditions::C
end

AnyOf(conditions::StopCondition...) = AnyOf(conditions)

"""
    RunState

Progress passed to the stop conditions on every iteration.
"""
mutable struct RunState
    step::Int
    simulated_time::Float64
    elapsed_seconds::Float64
    max_change::Float64
end

RunState() = RunState(0, 0.0, 0.0, Inf)

"""
    stop_reason(condition, state) -> Union{Symbol,Nothing}

`nothing` while the run should continue, otherwise a symbol naming the condition
that fired. The symbol ends up in `metrics.stopped_by`, so a run that hit its
step cap is distinguishable from one that actually converged.
"""
stop_reason(c::Steps, state::RunState) = state.step >= c.count ? :steps : nothing
stop_reason(c::UntilTime, state::RunState) = state.simulated_time >= c.time ? :time : nothing
stop_reason(c::WallClock, state::RunState) = state.elapsed_seconds >= c.seconds ? :wallclock : nothing
stop_reason(c::Converged, state::RunState) = state.max_change <= c.tol ? :converged : nothing

function stop_reason(c::AnyOf, state::RunState)
    for condition in c.conditions
        reason = stop_reason(condition, state)
        reason === nothing || return reason
    end
    return nothing
end

"""
    change_interval(condition) -> Int

How often the run loop must measure the change between successive states.
`typemax(Int)` means never, which is the case for every condition except
[`Converged`](@ref).
"""
change_interval(::StopCondition) = typemax(Int)
change_interval(c::Converged) = c.check_every
change_interval(c::AnyOf) = minimum(change_interval, c.conditions; init = typemax(Int))

tracks_change(condition::StopCondition) = change_interval(condition) < typemax(Int)

# Convenience: `stop = 250` reads better than `stop = Steps(250)` in a lab script.
as_stop_condition(condition::StopCondition) = condition
as_stop_condition(count::Integer) = Steps(count)
