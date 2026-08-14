"""
    VisuTwinSim

Teaching and research package for VisuTwin simulation concepts, built for the
course *High-Performance Computing in Simulation and Digital Twin Systems*.

The package is organised around four ideas that stay separate throughout:

- **model** — what is being simulated ([`Heat2D`](@ref)) and under which
  [`BoundaryCondition`](@ref);
- **backend** — where it runs ([`CPUBackend`](@ref), [`KernelBackend`](@ref),
  [`CUDADevice`](@ref), [`MetalDevice`](@ref), [`ROCmDevice`](@ref));
- **runtime** — how long it runs ([`Simulation`](@ref), [`StopCondition`](@ref))
  and what observes it (`run!` callbacks);
- **metrics** — what it cost ([`RunMetrics`](@ref)).

Every `run!` returns a measurement, because in this course the performance of a
simulation is part of its result rather than an afterthought.
"""
module VisuTwinSim

using Adapt
using KernelAbstractions
using Printf
using Random

export
    # backends
    AbstractBackend,
    CPUBackend,
    KernelBackend,
    CUDADevice,
    MetalDevice,
    ROCmDevice,
    RawCUDABackend,
    available_backends,
    backend_name,
    is_gpu,
    to_backend,
    # fields and boundary conditions
    Field2D,
    BoundaryCondition,
    Neumann,
    Periodic,
    Dirichlet,
    swapbuffers!,
    initialize_peak!,
    initialize_gaussian!,
    state,
    sum_state,
    center_value,
    conserves_state,
    # forcing
    SourceTerm,
    NoSource,
    UniformSource,
    PatternSource,
    ProportionalSource,
    CombinedSource,
    ControlSignal,
    TimeSeries,
    is_driven,
    feedback_coefficient,
    # model
    Heat2D,
    Heat2DParams,
    cfl_number,
    stability_number,
    is_stable,
    max_stable_dt,
    reset_clock!,
    # runtime
    Simulation,
    StopCondition,
    Steps,
    UntilTime,
    ForDuration,
    WallClock,
    Converged,
    AnyOf,
    RunState,
    run!,
    step!,
    # metrics
    RunMetrics,
    mlups,
    bandwidth_gbs,
    gflops,
    arithmetic_intensity,
    cell_updates,
    # digital twin
    Sensor,
    nudge!,
    save_state,
    load_state,
    checkpoint_callback,
    # observation data and monitoring
    Anomaly,
    Spike,
    LevelShift,
    Drift,
    Stuck,
    ObservationSeries,
    synthetic_series,
    flag_exceedances,
    DetectionReport,
    detection_report,
    threshold_sweep,
    MetricRecorder,
    # twin runtime
    TwinLoop,
    TwinLog,
    twin_run!,
    realtime_ratio,
    compute_seconds,
    simulated_time,
    # scenarios
    parameter_sweep,
    sweep_table,
    # ensemble
    random_walk_ensemble

include("backends.jl")
include("boundary.jl")
include("source.jl")
include("field.jl")
include("metrics.jl")
include("heat2d.jl")
include("kernels.jl")
include("stopping.jl")
include("runtime.jl")
include("twin.jl")
include("series.jl")
include("detection.jl")
include("recorder.jl")
include("twinloop.jl")
include("sweep.jl")
include("ensemble.jl")

end # module VisuTwinSim
