# Lab 01: Heat Diffusion

This lab introduces the same model through three levels:

1. A scalar update equation.
2. A CPU implementation with ordinary Julia arrays.
3. A CUDA implementation with CUDA.jl.

The starting model:

```julia
using VisuTwinSim

model = Heat2D(nx = 128, ny = 128)
initialize_peak!(model.field, 100.0f0)
metrics = run!(model; backend = CPUBackend(), steps = 250)
```

Expected checks:

- total heat should remain close to the initial value;
- center temperature should decrease;
- larger grids should make the CPU backend slower;
- the CUDA backend should become useful when the grid is large enough.

CUDA version:

```julia
using VisuTwinSim
using CUDA

model = Heat2D(nx = 1024, ny = 1024)
initialize_peak!(model.field, 100.0f0)
metrics = run!(model; backend = CUDABackend(), steps = 1000)
```

Questions:

- What happens if `dt` is too large?
- Why does the boundary stay unchanged?
- At what grid size does GPU execution become faster than CPU execution?
- How does memory transfer affect the measurement?
