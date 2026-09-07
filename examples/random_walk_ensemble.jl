# The embarrassingly parallel warm-up before GPU ensemble simulation.
#
#   julia --project=. -t auto examples/random_walk_ensemble.jl

using Printf
using Statistics
using TwinSim

trajectories, steps = 200_000, 2_000

println("threads = ", Threads.nthreads())
println("$trajectories trajectories x $steps steps\n")

serial_time = @elapsed serial = random_walk_ensemble(; trajectories, steps, seed = 42, threaded = false)
threaded_time = @elapsed threaded = random_walk_ensemble(; trajectories, steps, seed = 42, threaded = true)

@printf("serial    %8.3f s\n", serial_time)
@printf("threaded  %8.3f s   (%.2fx on %d threads)\n",
        threaded_time, serial_time / threaded_time, Threads.nthreads())
println()
println("identical results: ", serial == threaded)
println()

@printf("mean      %10.4f   (expected 0)\n", mean(threaded))
@printf("variance  %10.1f   (expected %d)\n", var(threaded), steps)
@printf("max |x|   %10d   (bounded by %d)\n", maximum(abs, threaded), steps)

println("""

Two properties worth pointing at in the lab:

  * The results are bit-identical with and without threading, because each
    trajectory has its own RNG seeded from (seed, index) rather than drawing
    from one shared stream. With a shared RNG the answer would depend on how the
    scheduler happened to interleave the work — reproducibility is a design
    decision here, not a given.
  * A step needs one random bit but a generator produces 64 at a time, so this
    draws once per 64 steps and turns the bits into a displacement with
    `2 * count_ones(bits) - 64`. That is the same idea as a warp-level
    population count on a GPU.

Next step for the GPU lecture: one trajectory per thread with a counter-based
RNG, where the seed-per-trajectory scheme above maps directly onto the thread
index.
""")
