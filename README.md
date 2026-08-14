# VisuTwinSim.jl

Julia teaching and research package for VisuTwin simulation concepts, with a CPU reference backend and an optional CUDA.jl backend.

This package is intended for the study course **High-Performance Computing in Simulation and Digital Twin Systems** and for examples accompanying CUDA Julia material.

## Goals

- Keep the student-facing API high level.
- Use pure Julia for the reference implementation.
- Use CUDA.jl for GPU acceleration without making CUDA mandatory.
- Mirror the concepts of VisuTwin Sim Core: model, backend, runtime, metrics.
- Provide examples that can grow into lectures, labs, and book chapters.

## Quick Start

From this directory:

```julia
using Pkg
Pkg.activate(".")
Pkg.test()
```

Run the CPU example:

```bash
julia --project=. examples/heat2d_cpu.jl
```

Run the CUDA example after adding CUDA.jl:

```julia
using Pkg
Pkg.add("CUDA")
```

```bash
julia --project=. examples/heat2d_cuda.jl
```

## Example

```julia
using VisuTwinSim

model = Heat2D(nx = 128, ny = 128)
initialize_peak!(model.field, 100.0f0)

metrics = run!(model; backend = CPUBackend(), steps = 250)

println(metrics)
println(center_value(model))
```

## Package Structure

```text
src/VisuTwinSim.jl          - CPU backend, models, runtime API
ext/VisuTwinSimCUDAExt.jl   - optional CUDA.jl backend
examples/                  - runnable course examples
test/                      - package tests
docs/labs/                 - lab notes
```

## Relationship to VisuTwin Sim Core

`VisuTwinSim.jl` is the high-level teaching/research interface. The C++23 `visutwin-sim` project can remain the lower-level production/runtime implementation. The two can later be connected through a C ABI, CxxWrap, Arrow/Parquet state exchange, or generated kernels.
