# Lab 01: Heat Diffusion

This lab introduces the same model through three levels:

1. A scalar update equation.
2. A CPU implementation with ordinary Julia arrays.
3. A GPU implementation, which turns out to be the *same* implementation.

## The equation

The explicit (FTCS) discretisation of `du/dt = alpha * laplacian(u)` on a
uniform grid:

```
u'[i,j] = u[i,j]
        + alpha*dt/dx^2 * (u[i-1,j] - 2u[i,j] + u[i+1,j])
        + alpha*dt/dy^2 * (u[i,j-1] - 2u[i,j] + u[i,j+1])
```

Each cell reads its four neighbours and writes one value. That ratio — a handful
of arithmetic operations per value moved — is what makes this kernel memory
bound, and it is why Lab 02 measures bandwidth rather than FLOP/s.

## The starting model

```julia
using VisuTwinSim

model = Heat2D(nx = 128, ny = 128)
initialize_peak!(model.field, 100.0f0)
metrics = run!(model; backend = CPUBackend(), steps = 250)
```

Expected checks:

- the centre temperature decreases as the peak spreads;
- **under the default `Neumann` boundary**, the total stays at 100.0 for as long
  as you care to run;
- larger grids make the CPU backend take proportionally longer;
- the GPU backend becomes useful once the grid is large enough to hide the
  kernel launch cost.

## Boundary conditions, and a warning about the conservation check

The second check above only holds because the default boundary condition is
insulating. Run `examples/boundary_conditions.jl`:

| steps | 0 | 500 | 2 000 | 10 000 | 50 000 |
|:------|--:|----:|------:|-------:|-------:|
| Neumann | 100.0000 | 100.0000 | 100.0000 | 100.0002 | 100.0014 |
| Periodic | 100.0000 | 100.0000 | 100.0000 | 100.0002 | 100.0000 |
| Dirichlet | 100.0000 | 100.0000 | 99.9767 | 74.2460 | 3.8898 |

Two things to take from this:

- With a `Dirichlet` boundary the edge cells are pinned to a fixed value, so heat
  reaching the edge leaves the domain permanently. That is a perfectly valid
  model of a plate clamped to a heat sink — it is simply not a conserving one.
- A conservation test that runs for 500 steps on this grid passes under *every*
  boundary condition, because the heat has not reached the edge yet. A test that
  cannot fail is not a test. Check conservation only after the disturbance has
  had time to cross the domain.

## The GPU version

```julia
using VisuTwinSim
using CUDA          # or: using Metal / using AMDGPU

model = Heat2D(nx = 2048, ny = 2048)
initialize_peak!(model.field, 100.0f0)

resident = to_backend(model, CUDADevice())   # move once, not once per run
metrics = run!(resident; backend = CUDADevice(), steps = 1000)
```

Look at `src/kernels.jl` before running this. There is exactly one stencil in the
package, and one `@kernel` wrapping it; the CPU loops and every GPU backend all
go through it. Choosing a backend changes the launch, not the arithmetic.

Note the helper is `CUDADevice()`, not `CUDABackend()`. CUDA.jl exports a type of
its own called `CUDABackend`, so a package exporting that name would make
`using VisuTwinSim, CUDA` ambiguous.

## Stability

The scheme is stable only while

```
CFL = alpha * dt * (1/dx^2 + 1/dy^2) <= 1/2
```

`Heat2D` refuses to build a configuration that violates this. To see why, build
one anyway with `check_stability = false` and run `examples/stability_cfl.jl`:

`max |u|` after N steps, starting from a peak of 100 on a 64x64 grid with
`alpha = 0.15`, `dx = dy = 1` (largest stable `dt` is 1.667):

| dt | CFL | stable? | after 50 | after 100 | after 200 | after 400 |
|---:|----:|:--------|---------:|----------:|----------:|----------:|
| 0.100 | 0.030 | yes | 13.16 | 5.828 | 2.762 | 1.352 |
| 1.000 | 0.300 | yes | 1.058 | 0.5296 | 0.265 | 0.1326 |
| 1.600 | 0.480 | yes | 0.6662 | 0.3301 | 0.1654 | 0.0828 |
| 1.670 | 0.501 | **no** | 1.400 | 0.7891 | 0.5119 | 0.4725 |
| 1.700 | 0.510 | **no** | 5.186 | 16.62 | 413.1 | 5.272e5 |
| 2.000 | 0.600 | **no** | 1.491e7 | 1.516e14 | 3.115e28 | NaN |

The transition is sharp. Below the limit the peak decays monotonically, as
diffusion should. At `dt = 1.67` — 0.2% over the limit — the run is still finite
after 400 steps but has stopped decaying. Two percent over, and it reaches 1e5;
twenty percent over, and it is `NaN` before 400 steps.

The `dt = 1.67` row is the dangerous one: a run that looks merely inaccurate is
already unstable, and nothing about its first hundred steps says so.

## Questions

- What happens if `dt` is too large — and *how much* too large does it have to be
  before you would notice within the first fifty steps?
- Why does a `Dirichlet` boundary lose heat while a `Neumann` boundary does not?
  Write down the flux across the edge in each case.
- At what grid size does GPU execution overtake the threaded CPU on your machine?
- How does the host/device transfer affect the measurement, and what does
  `to_backend` change about it?
- Halving `dx` at fixed `dt` multiplies the CFL number by four. What does that
  imply about the cost of refining a grid with an explicit scheme?
