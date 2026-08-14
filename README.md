# VisuTwinSim.jl

Julia teaching and research package for VisuTwin simulation concepts, with a CPU
reference backend and portable GPU backends for CUDA, Metal and ROCm.

Built for the study course **High-Performance Computing in Simulation and
Digital Twin Systems** and for examples accompanying CUDA Julia material.

## Goals

- Keep the student-facing API high level.
- Use pure Julia for the reference implementation.
- Run on a GPU without making any particular vendor's GPU mandatory.
- Mirror the concepts of VisuTwin Sim Core: model, backend, runtime, metrics.
- Make performance part of the result rather than an afterthought: every `run!`
  returns a measurement that can go straight onto a roofline plot.

## Quick Start

```julia
using Pkg
Pkg.activate(".")
Pkg.test()
```

```julia
using VisuTwinSim

model = Heat2D(nx = 512, ny = 512, alpha = 0.15f0, dt = 0.1f0)
initialize_peak!(model.field, 100.0f0)

metrics = run!(model; backend = CPUBackend(), steps = 500)
```

```
RunMetrics
  backend            cpu (Float32)
  grid               262144 cells
  steps              500  (simulated time 50)
  stopped by         steps
  wall time          0.0266 s
    compute          0.0266 s
  throughput         4930.8 MLUP/s
  bandwidth          39.45 GB/s (compulsory traffic)
  compute rate       49.31 GFLOP/s
  arith. intensity   1.250 FLOP/byte
  total state        99.99999237
```

`available_backends()` lists what the current session can actually use.

## Concepts

| Concept | Type | Purpose |
|:--------|:-----|:--------|
| Model | `Heat2D`, `Heat2DParams` | what is simulated |
| Boundary | `Neumann`, `Periodic`, `Dirichlet` | how the domain edge behaves |
| Forcing | `NoSource`, `UniformSource`, `PatternSource`, `ProportionalSource`, `TimeSeries`, `ControlSignal` | what drives it from outside |
| Backend | `CPUBackend`, `KernelBackend`, `CUDADevice`, `MetalDevice`, `ROCmDevice` | where it runs |
| Runtime | `Simulation`, `Steps`, `UntilTime`, `Converged`, `WallClock`, `AnyOf` | how long it runs |
| Metrics | `RunMetrics`, `mlups`, `bandwidth_gbs`, `arithmetic_intensity` | what it cost |
| Twin | `Sensor`, `nudge!`, `save_state`, `load_state` | connecting it to a real system |

### Backends

The backends form a progression, and steps 3–5 run **identical kernel source**:

```julia
CPUBackend()                    # plain Julia loops, one thread
CPUBackend(threaded = true)     # the same loops across Threads.nthreads()
KernelBackend()                 # the portable KernelAbstractions kernel, on CPU
CUDADevice()                    # the same kernel on NVIDIA   (needs `using CUDA`)
MetalDevice()                   # the same kernel on Apple    (needs `using Metal`)
ROCmDevice()                    # the same kernel on AMD      (needs `using AMDGPU`)
```

`KernelBackend()` exists so the GPU code path stays testable on machines without
a GPU — the same kernel, executed on the CPU.

GPU backends are loaded through package extensions, so CUDA, Metal and AMDGPU
are all optional and none of them is a hard dependency.

> The GPU helpers are named `CUDADevice` / `MetalDevice` / `ROCmDevice`, not
> `CUDABackend` / `MetalBackend`. Those names are already exported by CUDA.jl and
> Metal.jl, so `using VisuTwinSim, CUDA` followed by `CUDABackend()` would be an
> ambiguity error rather than a working program.

### Boundary conditions decide whether the model conserves anything

```julia
Heat2D(nx = 128, boundary = Neumann())        # insulated, conserving (default)
Heat2D(nx = 128, boundary = Periodic())       # wraps around, conserving
Heat2D(nx = 128, boundary = Dirichlet(0.0f0)) # edge pinned, heat leaves the domain
```

Measured total heat starting from 100.0 (`examples/boundary_conditions.jl`):

| steps | 0 | 500 | 2 000 | 10 000 | 50 000 |
|:------|--:|----:|------:|-------:|-------:|
| Neumann | 100.0000 | 100.0000 | 100.0000 | 100.0002 | 100.0014 |
| Periodic | 100.0000 | 100.0000 | 100.0000 | 100.0002 | 100.0000 |
| Dirichlet | 100.0000 | 100.0000 | 99.9767 | 74.2460 | 3.8898 |

"Total heat is conserved" is only a valid check under a conserving boundary
condition, and only a meaningful one once heat has had time to reach the edge.

### Driving the model from its environment

Without forcing, `Heat2D` is a closed system: it can only redistribute the heat
it started with. A twin of a real installation needs a **source term** and,
usually, a boundary that changes over time.

```
du/dt = alpha * laplacian(u) + q(x, y, t)
```

| Source | `q` | Use |
|:-------|:----|:----|
| `NoSource()` | `0` | closed system (default); compiles away entirely |
| `UniformSource(rate)` | `rate` everywhere | ambient gain or loss |
| `PatternSource(pattern, rate)` | `rate * pattern[i,j]` | heaters, pipes, any fixed layout |
| `ProportionalSource(target, gain)` | `gain * (target - u[i,j])` | Newton cooling; a per-cell thermostat |

Sources add with `+`, because a real installation has heaters *and* ambient loss:

```julia
source = PatternSource(layout, demand) + ProportionalSource(outdoor, 0.3f0)
```

Their rates are summed and applied once, not applied in sequence — with a
state-dependent term in the mix those differ, and sequencing would make the
result depend on the order you wrote them in.

Anywhere a value is accepted, a **callable of simulated time** is too:

```julia
outdoor = TimeSeries(hours, temperatures)          # interpolates sampled data
demand(t) = max(0.0f0, 0.06f0 * (16 - outdoor(t))) # a weather-compensated control law

model = Heat2D(nx = 96, ny = 96, dt = 0.02f0,
               boundary = Dirichlet(outdoor),              # the edge follows the weather
               source = PatternSource(layout, demand))     # the heaters follow the operator
```

`PatternSource` separates *where* the forcing acts from *how strong it is*: the
pattern is fixed geometry that stays on the GPU untouched, while the rate is one
scalar per step that can come from a measurement series. Time-dependent values
are evaluated once per step on the host, because a GPU kernel cannot call a
Julia closure and would not want to re-evaluate one scalar in a million threads.

### Closing the loop

There are two different ways for forcing to respond to the state, and the
difference is worth a paragraph in any report that uses one:

```julia
# Per cell, every step, inside the kernel — a distributed thermostat.
ProportionalSource(target, gain)

# One scalar for the whole domain, updated as often as the callback runs —
# a central controller with a sampling rate.
power = ControlSignal(0.0f0)
model = Heat2D(nx = 96, source = PatternSource(layout, power))

run!(sim; callback_every = 10, callback = function (m, progress)
    power[] = clamp(0.5f0 * (20 - mean(state(m))), 0, 5)
    return nothing
end)
```

`ControlSignal` is also the supported way to change a forcing mid-run: `Heat2D`
is immutable, so without it you would have to rebuild the model.

Three consequences worth stating in a report:

- A driven model has **no conservation invariant** — `conserves_state` returns
  `false` as soon as a source is present, whatever the boundary condition.
- An additive source (`UniformSource`, `PatternSource`, a `ControlSignal`) does
  not change the stability limit. It can still make the solution grow without
  bound; that is physics, not instability, and the two should not be confused.
- A **state-dependent** source does change the limit, and the model checks for
  it — see below.

`examples/driven_heat.jl` runs Case A and separates three timescales in one
table — the daily cycle penetrating a short distance from the edge, the bulk
trend over thousands of hours, and the compute time that is negligible against
both.

### Stability is checked, not discovered

`Heat2D` rejects configurations that would diverge, instead of producing `NaN`
several thousand steps later:

```julia
julia> Heat2D(nx = 64, dt = 2.0f0)
ERROR: ArgumentError: Unstable configuration: stability number is 0.6, which
exceeds the explicit-scheme limit of 0.5. The run would diverge to NaN.
...
```

The quantity checked is `stability_number`, not `cfl_number`:

```
stability_number = alpha*dt*(1/dx^2 + 1/dy^2) + dt*g/4
                   \_______ diffusion _______/   \_ feedback _/
```

`g` is the source's `feedback_coefficient` — the largest `|dq/du|` it
contributes, which is zero for every source that does not read the state. With
no state-dependent forcing this is exactly the CFL number, and `cfl_number`
keeps its conventional diffusion-only meaning.

The second term is not cosmetic. A `ProportionalSource` enters the von Neumann
analysis, so a strong enough controller destabilises the scheme **at a time step
the diffusion alone tolerates comfortably**:

| gain | `cfl_number` | `stability_number` | outcome |
|-----:|-------------:|-------------------:|:--------|
| 5 | 0.030 | 0.155 | stable |
| 18 | 0.030 | 0.480 | stable |
| 19 | 0.030 | 0.505 | **rejected** |
| 40 | 0.030 | 1.030 | **rejected** — diverges to NaN if forced through |

Every row has the same, perfectly safe-looking CFL number. Checking only that
would accept the last one. The error message names the source's contribution, so
that lowering the gain is visible as a fix alongside reducing `dt`.

Pass `check_stability = false` to explore the instability deliberately; see
`examples/stability_cfl.jl`.

## Measured performance

Regenerate all of this on your own machine with:

```bash
julia --project=. -t auto examples/backend_comparison.jl
```

### CPU vs Metal vs CUDA

Throughput in **MLUP/s** (million lattice updates per second), Float32, 500 steps,
best of three runs:

| grid | CPU | Metal | CUDA |
|:-----|----:|------:|-----:|
| 256² | 4 877 | 3 529 | 17 730 |
| 512² | 6 217 | 7 034 | 57 100 |
| 1024² | 8 309 | 7 534 | 88 386 |
| 2048² | 15 389 | 7 342 | 97 019 |
| 4096² | 15 050 | 7 011 | 56 381 |

The same runs as **achieved memory bandwidth in GB/s**, which is the meaningful
figure for a memory-bound stencil:

| grid | CPU | Metal | CUDA |
|:-----|----:|------:|-----:|
| 256² | 39.0 | 28.2 | 141.8 |
| 512² | 49.7 | 56.3 | 456.8 |
| 1024² | 66.5 | 60.3 | 707.1 |
| 2048² | 123.1 | 58.7 | 776.1 |
| 4096² | 120.4 | 56.1 | 451.0 |

Hardware, and an important caveat about reading across the columns:

| column | device | host |
|:-------|:-------|:-----|
| CPU | Apple M2 Max, 8 Julia threads | same machine as Metal |
| Metal | Apple M2 Max integrated GPU | same machine as CPU |
| CUDA | NVIDIA RTX 4070 SUPER, 48 MB L2, 504 GB/s rated | Intel i5-13600K, 20 threads |

**CPU and Metal share a machine, CUDA does not.** The CUDA column is therefore
not a fair comparison against the CPU column — it is a different host. Measured
against *its own* CPU (i5-13600K, best of serial and 20-thread), the speedups
are:

| grid | host CPU | CUDA | speedup |
|:-----|---------:|-----:|--------:|
| 256² | 1 892 | 17 730 | 9.4x |
| 512² | 3 939 | 57 100 | 14.5x |
| 1024² | 5 440 | 88 386 | 16.2x |
| 2048² | 12 912 | 97 019 | 7.5x |
| 4096² | 5 422 | 56 381 | 10.4x |

The CPU column above is the better of the serial and threaded backends at each
size, since which one wins changes with the grid. The full breakdown of all four
backends per machine is further down.

Four things in these tables are worth a lecture each:

- **The discrete GPU wins by roughly an order of magnitude; the integrated one
  does not.** Metal never reaches even 1.2x over the M2 Max CPU and falls to
  0.5x at large grids. Both sit on the same unified memory, so the GPU has no
  bandwidth advantage to exploit — it has the same memory as its competitor.
  A discrete card with its own dedicated GDDR6 is a different proposition, and
  that difference, not the core count, is what the CUDA column is showing.
- **Threading loses on small grids.** At 256² the threaded backend is 3x *slower*
  than the serial one; synchronisation costs more than the work saved.
- **The GPU exceeds its own rated bandwidth between 512² and 2048².** It cannot:
  777 GB/s against a 504 GB/s rating means the two buffers (32 MB at 2048²) fit
  inside the 48 MB L2 cache and the data never reaches DRAM. At 4096² the
  buffers total 128 MB, no longer fit, and throughput falls back to 451 GB/s —
  89% of the DRAM rating, which is about what a well-behaved stencil should get.
- **Bandwidth is the metric, not GFLOP/s.** This stencil does 1.25 FLOP per byte
  in Float32, so it is memory bound everywhere. Switching to `Float64` halves
  the arithmetic intensity and roughly halves the throughput.

### Full breakdown

All four backends, per machine, MLUP/s:

**Apple M2 Max, 8 Julia threads**

| grid | `cpu` | `cpu x8` | `ka-cpu` | `metal` |
|:-----|------:|---------:|---------:|--------:|
| 256² | 4 877 | 1 520 | 560 | 3 529 |
| 512² | 4 753 | 6 217 | 2 185 | 7 034 |
| 1024² | 5 097 | 8 309 | 2 953 | 7 534 |
| 2048² | 4 354 | 15 389 | 3 854 | 7 342 |
| 4096² | 3 412 | 15 050 | 3 700 | 7 011 |

**Intel i5-13600K, 20 Julia threads, RTX 4070 SUPER**

| grid | `cpu` | `cpu x20` | `ka-cpu` | `cuda` |
|:-----|------:|----------:|---------:|-------:|
| 256² | 1 892 | 856 | 615 | 17 730 |
| 512² | 3 939 | 2 250 | 1 345 | 57 100 |
| 1024² | 5 440 | 5 199 | 2 034 | 88 386 |
| 2048² | 3 531 | 12 912 | 3 157 | 97 019 |
| 4096² | 2 823 | 5 422 | 3 057 | 56 381 |

The i5-13600K is a hybrid design (performance and efficiency cores). Julia's
`@threads` splits the columns statically and evenly, so every step waits for the
chunk that landed on the slowest core — which is why its threaded backend only
overtakes the serial one at 2048², and why its numbers are less regular than the
homogeneous M2 Max. A dynamic or size-weighted decomposition is the fix, and a
good exercise.

Benchmarking notes are in `examples/backend_comparison.jl`: GPUs need a *timed*
warm-up (a step-count warm-up measures power-state transitions), CPUs need a
short one (a long one measures thermal throttling), and the script reports the
best of three runs because interference can only ever make a run slower.

## Package Structure

```text
src/backends.jl      backend types and device resolution
src/boundary.jl      Neumann / Periodic / Dirichlet
src/field.jl         Field2D, double buffering
src/heat2d.jl        model, parameters, CFL stability
src/kernels.jl       the shared stencil and the KernelAbstractions kernel
src/metrics.jl       RunMetrics and derived performance figures
src/stopping.jl      stop conditions
src/runtime.jl       Simulation, run!, host/device movement
src/twin.jl          sensors, assimilation, checkpoints
src/ensemble.jl      random walk ensemble
ext/                 CUDA / Metal / AMDGPU device registration
examples/            runnable course examples
test/                package tests
docs/labs/           lab notes
```

## Examples

| Script | Shows |
|:-------|:------|
| `heat2d_cpu.jl` | the reference run and its metrics |
| `heat2d_gpu.jl` | the same model on whichever GPU is available |
| `backend_comparison.jl` | the performance table above, plus benchmarking method |
| `boundary_conditions.jl` | conservation and why the boundary condition decides it |
| `driven_heat.jl` | a model driven by weather and a control law; timescale separation |
| `stability_cfl.jl` | what exceeding the CFL limit actually does |
| `digital_twin.jl` | assimilation, real-time pacing, checkpoint and restart |
| `random_walk_ensemble.jl` | reproducible parallel Monte Carlo |

## Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

The suite validates against the analytical solution rather than against previous
output. The sharpest of those checks uses the fact that for this stencil the
second spatial moment grows by *exactly* `2 * alpha * dt` per step, with no
discretisation error — which pins down `alpha`, `dt`, `dx` and `dy`
simultaneously, and (with `dx != dy`) catches a swapped axis.

GPU backends are picked up automatically if present. To exercise them, add the
vendor package to `test/Project.toml`, or run the suite from an environment that
has it:

```bash
julia --project=/path/to/env -t 4 test/runtests.jl
```

Currently verified: 230 tests passing on CPU, on Metal (Apple M2 Max) and on
CUDA (RTX 4070 SUPER).

## Relationship to VisuTwin Sim Core

`VisuTwinSim.jl` is the high-level teaching/research interface. The C++23
`visutwin-sim` project can remain the lower-level production/runtime
implementation. The two can later be connected through a C ABI, CxxWrap,
Arrow/Parquet state exchange, or generated kernels.

The checkpoint format in `src/twin.jl` is deliberately a documented 50-byte
header plus raw column-major data, so it can be read from C++ or Python without
a Julia dependency — the simplest available bridge between the two projects.
