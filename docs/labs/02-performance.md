# Lab 02: Measuring Performance

Lab 01 established that the model is correct. This lab asks what it costs, and
introduces the habit that the rest of the course depends on: never quote a
performance number you did not measure on the machine in front of you.

## Every run is already a measurement

```julia
using TwinSim

model = Heat2D(nx = 2048, ny = 2048)
initialize_peak!(model.field, 100.0f0)
metrics = run!(model; backend = CPUBackend(threaded = true), steps = 500)
```

`metrics` reports throughput (`mlups`), achieved bandwidth (`bandwidth_gbs`),
compute rate (`gflops`) and arithmetic intensity (`arithmetic_intensity`), with
host/device transfer time kept separate from compute time.

## The number that matters

The five-point stencil moves 2 values per cell update (read the old value, write
the new one; the four neighbours are cache hits) and performs about 10
floating-point operations. In `Float32`:

```
arithmetic intensity = 10 FLOP / 8 bytes = 1.25 FLOP/byte
```

Every machine in existence can do far more than 1.25 FLOP per byte it can fetch,
so this kernel is **memory bound** everywhere. The consequence:

- `bandwidth_gbs` should be compared against the machine's memory bandwidth and
  can get respectably close to it;
- `gflops` will stay far below the machine's peak no matter what you do, and
  optimising for it is a category error.

Switch the model to `Float64` (`initial = 0.0`) and the intensity halves to
0.625 FLOP/byte. On a memory-bound kernel that shows up almost exactly as half
the throughput. Verify this — it is the cleanest demonstration in the course that
precision is a performance decision.

## Exercise 1: the backend table

```bash
julia --project=. -t auto examples/backend_comparison.jl
```

Reference results across three devices, Float32, MLUP/s (best CPU backend at
each size):

| grid | CPU | Metal | CUDA |
|:-----|----:|------:|-----:|
| 256² | 4 877 | 3 529 | 17 730 |
| 512² | 6 217 | 7 034 | 57 100 |
| 1024² | 8 309 | 7 534 | 88 386 |
| 2048² | 15 389 | 7 342 | 97 019 |
| 4096² | 15 050 | 7 011 | 56 381 |

CPU and Metal are an Apple M2 Max (8 Julia threads); CUDA is an RTX 4070 SUPER
in an i5-13600K host. The CUDA column is therefore *not* comparable to the CPU
column — different machine. Against its own host CPU it runs 7.5x to 16.2x
faster.

Full per-backend breakdown, Apple M2 Max:

| grid | `cpu` | `cpu x8` | `ka-cpu` | `metal` |
|:-----|------:|---------:|---------:|--------:|
| 256² | 4 877 | 1 520 | 560 | 3 529 |
| 512² | 4 753 | 6 217 | 2 185 | 7 034 |
| 1024² | 5 097 | 8 309 | 2 953 | 7 534 |
| 2048² | 4 354 | 15 389 | 3 854 | 7 342 |
| 4096² | 3 412 | 15 050 | 3 700 | 7 011 |

Questions:

1. At 256² the threaded backend is three times *slower* than the serial one.
   How much work is there per thread at that size, and how does it compare to
   the cost of a thread barrier?
2. Where is the CPU/GPU crossover on your machine? Is it the same for Float32
   and Float64?
3. `ka-cpu` runs the GPU kernel on the CPU and is consistently slower than the
   hand-written CPU loop. What does the hand-written version do that the
   portable kernel cannot? (Look at `update_column!` in `src/kernels.jl`.)
4. Metal peaks at 1.13x over the M2 Max CPU and *loses* at large grids, while
   CUDA beats its host CPU by up to 16x. Both are GPUs with many more cores than
   their CPUs. Since this kernel is memory bound, what does each GPU's memory
   arrangement predict — and which number in the bandwidth table confirms it?

## Exercise 2: reading a bandwidth number that is impossible

NVIDIA RTX 4070 SUPER — 48 MB L2 cache, 504 GB/s rated memory bandwidth:

| grid | buffers | MLUP/s | achieved GB/s |
|:-----|--------:|-------:|--------------:|
| 256² | 0.5 MB | 17 615 | 140.9 |
| 512² | 2 MB | 56 236 | 449.9 |
| 1024² | 8 MB | 87 732 | 701.9 |
| 2048² | 32 MB | 97 169 | 777.4 |
| 4096² | 128 MB | 56 362 | 450.9 |

The 2048² row claims 777 GB/s from a card rated at 504 GB/s. Explain it.

Then explain why 4096² drops back to 451 GB/s, and why that number is the more
"honest" one. What does the ratio 451/504 tell you about the quality of the
kernel's memory access pattern?

This is the exercise to spend time on. It shows that a performance figure is
meaningless without knowing which level of the memory hierarchy served it, and
it gives a way to *measure* a cache size rather than look it up.

## Exercise 3: benchmarking is itself a skill

`examples/backend_comparison.jl` carries three corrections, each of which was
found by getting a wrong answer first:

- **GPUs need a timed warm-up.** Between GPU measurements the script runs CPU
  backends and the GPU drops to a low power state. A fixed step-count warm-up on
  a small grid finishes before the clocks ramp back up, and the resulting table
  is not even monotonic in grid size.
- **CPUs need a short warm-up.** A long all-core warm-up drives the package into
  thermal throttling, so repeating the script gives monotonically worse numbers.
- **Report the best of several runs, not the mean.** Interference — a GC pause,
  another process, a power transition — can only make a run slower. During
  development one contended run reported a seventh of the real throughput.

Exercise: comment out the `GC.gc()` call, or set `REPEATS = 1`, and see how much
the table moves. Then run the script while something else is loading the machine.

Questions:

4. Why is *best-of-N* the right statistic here, when for most measurements you
   would want a mean and a standard deviation?
5. `run!` synchronises the device before it stops the clock. What would the GPU
   numbers look like without that, and why?
6. The transfer column is charged once per `run!`. At what steps-per-call does
   the transfer stop mattering on your machine?

## Exercise 4: where the CPU time actually goes

The serial CPU backend in this package is about 5.8x faster than a
straightforward implementation of the same equation (measured: 909 → 5 238
MLUP/s at 1024²). Three changes account for it:

- the double buffer is *swapped* rather than copied, removing a full array copy
  per step;
- the boundary test is hoisted out of the innermost loop, so the interior is a
  branch-free run (`interior_update` in `src/kernels.jl`);
- that branch-free run can then be vectorised with `@simd`.

Exercise: undo them one at a time and measure each. Which is worth the most? Does
the ranking change between Float32 and Float64, or between 512² and 4096²?
