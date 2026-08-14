"""
    random_walk_ensemble(; trajectories = 10_000, steps = 1_000, seed = 1, threaded = true)

Final positions of `trajectories` independent one-dimensional random walks.

The embarrassingly parallel warm-up before GPU ensemble simulation, and it is
written the way such a problem should be written:

- **Trajectory-outer.** Each walk is independent, so the outer loop is the one
  to parallelise. The original step-outer ordering created a false dependency
  between trajectories at every step.
- **One RNG per trajectory**, seeded from `(seed, trajectory)`. Results are
  therefore identical no matter how many threads run — a shared RNG would make
  them depend on the scheduler, which is the standard way parallel Monte Carlo
  loses reproducibility.
- **64 steps per random number.** A step needs one bit, but a generator produces
  64 at a time. `count_ones` turns those 64 bits into a displacement directly:
  `2 * popcount - 64` sums `+1` for each set bit and `-1` for each clear one.

Set `threaded = false` to measure the serial baseline.
"""
function random_walk_ensemble(; trajectories::Integer = 10_000,
                              steps::Integer = 1_000,
                              seed::Integer = 1,
                              threaded::Bool = true)
    trajectories > 0 || throw(ArgumentError("trajectories must be positive, got $trajectories"))
    steps >= 0 || throw(ArgumentError("steps must be non-negative, got $steps"))

    positions = Vector{Int}(undef, trajectories)
    if threaded
        Threads.@threads for t in 1:trajectories
            @inbounds positions[t] = walk_one(steps, seed, t)
        end
    else
        for t in 1:trajectories
            @inbounds positions[t] = walk_one(steps, seed, t)
        end
    end
    return positions
end

"""
    walk_one(steps, seed, trajectory) -> Int

One walk, with a private RNG derived from `(seed, trajectory)` so the result
does not depend on which thread ran it.
"""
function walk_one(steps::Integer, seed::Integer, trajectory::Integer)
    rng = Random.Xoshiro(hash((seed, trajectory)))
    position = 0
    full, remainder = divrem(steps, 64)
    for _ in 1:full
        position += 2 * count_ones(rand(rng, UInt64)) - 64
    end
    if remainder > 0
        mask = (UInt64(1) << remainder) - UInt64(1)
        position += 2 * count_ones(rand(rng, UInt64) & mask) - remainder
    end
    return position
end
