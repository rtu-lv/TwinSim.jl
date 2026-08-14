"""
    RunMetrics

What a run cost, not just what it produced. This is a course on high-performance
computing, so every `run!` returns a measurement that can be put on a roofline
plot without further instrumentation.

Timing fields:

- `compute_seconds` – time in the stepping loop, after a device synchronise.
- `transfer_seconds` – host/device copies, kept separate so the GPU's compute
  advantage and its transfer tax can be discussed independently.
- `elapsed_seconds` – wall clock for the whole `run!`, including both of the above
  plus callbacks and real-time pacing.

Derived quantities are functions, not stored fields, so they cannot go stale:
[`mlups`](@ref), [`bandwidth_gbs`](@ref), [`gflops`](@ref),
[`arithmetic_intensity`](@ref).
"""
Base.@kwdef mutable struct RunMetrics
    backend::Symbol = :unknown
    precision::DataType = Float32
    cells::Int = 0
    steps::Int = 0
    simulated_time::Float64 = 0.0
    elapsed_seconds::Float64 = 0.0
    compute_seconds::Float64 = 0.0
    transfer_seconds::Float64 = 0.0
    transferred_bytes::Int = 0
    bytes_per_cell::Int = 0
    flops_per_cell::Int = 0
    total_state::Float64 = 0.0
    stopped_by::Symbol = :none
end

"""
    cell_updates(metrics) -> Int

Total cell updates performed, `steps * cells`. The unit of work for a stencil.
"""
cell_updates(metrics::RunMetrics) = metrics.steps * metrics.cells

"""
    mlups(metrics) -> Float64

Million lattice updates per second, the standard throughput figure for stencil
codes. Based on `compute_seconds`, so it is comparable between backends
regardless of transfer cost.
"""
function mlups(metrics::RunMetrics)
    metrics.compute_seconds > 0 || return NaN
    return cell_updates(metrics) / metrics.compute_seconds / 1e6
end

"""
    moved_bytes(metrics) -> Float64

Compulsory memory traffic: one read and one write of each cell per step. Real
traffic is higher when the cache cannot hold three rows of the grid, so
comparing `bandwidth_gbs` against the machine's peak shows how well the stencil
is reusing cached neighbours.
"""
moved_bytes(metrics::RunMetrics) = float(cell_updates(metrics)) * metrics.bytes_per_cell

"""
    bandwidth_gbs(metrics) -> Float64

Achieved memory bandwidth in GB/s. A five-point stencil is bandwidth bound, so
this — not GFLOP/s — is the number to compare against the hardware limit.
"""
function bandwidth_gbs(metrics::RunMetrics)
    metrics.compute_seconds > 0 || return NaN
    return moved_bytes(metrics) / metrics.compute_seconds / 1e9
end

"""
    gflops(metrics) -> Float64

Achieved floating-point rate. Expected to be far below peak for this model; that
gap is the lesson.
"""
function gflops(metrics::RunMetrics)
    metrics.compute_seconds > 0 || return NaN
    return float(cell_updates(metrics)) * metrics.flops_per_cell / metrics.compute_seconds / 1e9
end

"""
    arithmetic_intensity(metrics) -> Float64

FLOP per byte of compulsory traffic. Fixes the model's position on the x-axis of
a roofline plot; it depends only on the algorithm and the precision, never on
the machine.
"""
function arithmetic_intensity(metrics::RunMetrics)
    metrics.bytes_per_cell > 0 || return NaN
    return metrics.flops_per_cell / metrics.bytes_per_cell
end

"""
    transfer_gbs(metrics) -> Float64

Host/device transfer rate, or `NaN` when nothing was transferred.
"""
function transfer_gbs(metrics::RunMetrics)
    metrics.transfer_seconds > 0 || return NaN
    return metrics.transferred_bytes / metrics.transfer_seconds / 1e9
end

function Base.show(io::IO, ::MIME"text/plain", metrics::RunMetrics)
    println(io, "RunMetrics")
    @printf(io, "  backend            %s (%s)\n", metrics.backend, metrics.precision)
    @printf(io, "  grid               %d cells\n", metrics.cells)
    @printf(io, "  steps              %d  (simulated time %.4g)\n", metrics.steps, metrics.simulated_time)
    metrics.stopped_by === :none || @printf(io, "  stopped by         %s\n", metrics.stopped_by)
    @printf(io, "  wall time          %.4f s\n", metrics.elapsed_seconds)
    @printf(io, "    compute          %.4f s\n", metrics.compute_seconds)
    if metrics.transferred_bytes > 0
        @printf(io, "    host<->device    %.4f s  (%.2f MiB, %.2f GB/s)\n",
                metrics.transfer_seconds,
                metrics.transferred_bytes / 1024^2,
                transfer_gbs(metrics))
    end
    @printf(io, "  throughput         %.1f MLUP/s\n", mlups(metrics))
    @printf(io, "  bandwidth          %.2f GB/s (compulsory traffic)\n", bandwidth_gbs(metrics))
    @printf(io, "  compute rate       %.2f GFLOP/s\n", gflops(metrics))
    @printf(io, "  arith. intensity   %.3f FLOP/byte\n", arithmetic_intensity(metrics))
    @printf(io, "  total state        %.10g", metrics.total_state)
    return nothing
end

function Base.show(io::IO, metrics::RunMetrics)
    @printf(io, "RunMetrics(%s, %d steps, %.1f MLUP/s)", metrics.backend, metrics.steps, mlups(metrics))
    return nothing
end
