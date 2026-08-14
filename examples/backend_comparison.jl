# Measure every available backend across a range of grid sizes.
#
#   julia --project=. -t auto examples/backend_comparison.jl
#
# This is the script behind the table in the README. Run it on your own machine
# before quoting any of those numbers — the CPU/GPU crossover depends entirely
# on the hardware.

using Printf
using VisuTwinSim

const GRIDS = (256, 512, 1024, 2048, 4096)
const STEPS = 500
# Warm-up policy, and why it differs by backend.
#
# GPUs need a *timed* warm-up. Between two GPU measurements this script runs
# three CPU backends, during which the GPU drops to a low power state; a
# step-count warm-up on a small grid finishes before the clocks ramp back up, so
# the small-grid rows come out slower than the large ones and the table stops
# being monotonic. That is measuring power management, not the kernel.
#
# CPUs need the opposite. A long warm-up on all cores drives the package into
# thermal throttling, and the measured run is then slower than it would be cold
# — repeating the script gives monotonically worse numbers. One pass to force
# compilation and populate the caches is enough.
const GPU_WARMUP_SECONDS = 0.4
const CPU_WARMUP_STEPS = 5

# Report the best of several timed runs. Interference — a GC pause, another
# process, a power-state transition — can only ever make a run slower, so the
# fastest observation is the closest one to the machine's actual capability.
# Reporting a mean instead lets one unlucky run dominate the table; on a shared
# machine this model measured a seventh of its real throughput that way.
const REPEATS = 3

function collect_backends()
    # Pair{String,Any} on purpose: the GPU backend types are not known until the
    # vendor package is loaded, so a narrower element type cannot hold them.
    backends = Pair{String,Any}["cpu" => CPUBackend(),
                                "cpu x$(Threads.nthreads())" => CPUBackend(threaded = true),
                                "ka-cpu" => KernelBackend()]
    for (pkg, helper) in (("CUDA", CUDADevice), ("Metal", MetalDevice), ("AMDGPU", ROCmDevice))
        Base.identify_package(pkg) === nothing && continue
        try
            @eval using $(Symbol(pkg))
            # invokelatest is required: loading the vendor package defines the
            # extension's gpu_device method in a newer world age than the one
            # this function was compiled in, so a direct call would still
            # dispatch to the "backend not loaded" fallback.
            push!(backends, lowercase(pkg) => Base.invokelatest(helper))
        catch err
            @info "skipping $pkg" exception = err
        end
    end
    return backends
end

function measure(backend, n, steps)
    model = Heat2D(nx = n, ny = n)
    initialize_peak!(model.field, 100.0f0)

    # The one-time cost of getting the state onto the device, measured on its
    # own before the state is made resident.
    transfer_ms = run!(deepcopy(model); backend = backend, steps = 1).transfer_seconds * 1000

    # Keep the state on the device across the warm-up and the measured run, so
    # the measured run is not billed for an upload it does not need.
    resident = to_backend(model, backend)
    if is_gpu(backend)
        deadline = time() + GPU_WARMUP_SECONDS
        while time() < deadline
            run!(resident; backend = backend, steps = 20)
        end
    else
        run!(resident; backend = backend, steps = CPU_WARMUP_STEPS)
    end

    best = nothing
    for _ in 1:REPEATS
        # Collect the previous iteration's garbage before starting the clock, so
        # the timed run does not absorb a GC pause that belongs to the warm-up.
        GC.gc()
        metrics = run!(resident; backend = backend, steps = steps)
        if best === nothing || metrics.compute_seconds < best.compute_seconds
            best = metrics
        end
    end
    return (best, transfer_ms)
end

backends = collect_backends()
println("Julia threads: ", Threads.nthreads())
println("Steps per measurement: ", STEPS, "\n")

for n in GRIDS
    @printf("grid %d x %d  (%.1f MiB per buffer, Float32)\n", n, n, n * n * 4 / 1024^2)
    @printf("  %-12s %10s %10s %10s %10s\n", "backend", "MLUP/s", "GB/s", "GFLOP/s", "xfer ms")
    for (name, backend) in backends
        metrics, transfer_ms = measure(backend, n, STEPS)
        @printf("  %-12s %10.0f %10.1f %10.1f %10.2f\n",
                name, mlups(metrics), bandwidth_gbs(metrics), gflops(metrics), transfer_ms)
    end
    println()
end

println("""
Things worth explaining in the lab:

  * Threading loses to the serial loop on small grids. Splitting 256x256 into
    per-thread chunks costs more in synchronisation than it saves in work.
  * The bandwidth column matters more than the GFLOP/s column. This stencil does
    1.25 FLOP per byte in Float32, so it is memory bound on every machine here;
    the GFLOP/s number will stay far below the hardware peak no matter what.
  * The transfer column is charged once per `run!`. A twin that steps a few
    hundred times per call amortises it; one that returns to the host every step
    does not, which is why `to_backend` exists.
  * Switch the model to Float64 (`initial = 0.0`) and the arithmetic intensity
    halves. On a memory-bound kernel that shows up directly as half the MLUP/s.
  * Watch what the GPU bandwidth does as the grid grows. If it *exceeds* the
    card's rated memory bandwidth, the working set is fitting in L2 and the data
    never reaches DRAM at all; the row where it drops back below the rating is
    the row where the two buffers stopped fitting. On a 48 MB L2 that happens
    between 2048^2 and 4096^2 in Float32. Compare the numbers here against
    `nvidia-smi --query-gpu=memory.bus_width,clocks.max.memory --format=csv`.
""")
