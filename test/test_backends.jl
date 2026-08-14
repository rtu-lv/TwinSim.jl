const CPU_BACKENDS = ["cpu" => CPUBackend(),
                      "cpu-threaded" => CPUBackend(threaded = true),
                      "ka-cpu" => KernelBackend()]

@testset "backend metadata" begin
    @test backend_name(CPUBackend()) === :cpu
    @test backend_name(CPUBackend(threaded = true)) === :cpu_threaded
    @test backend_name(KernelBackend()) === :ka_cpu
    @test !is_gpu(CPUBackend())
    @test !is_gpu(KernelBackend())
    @test "CPUBackend()" in available_backends()
end

@testset "unavailable GPU backends fail with a usable message" begin
    for (helper, pkg) in ((CUDADevice, "CUDA"), (MetalDevice, "Metal"), (ROCmDevice, "AMDGPU"))
        Base.identify_package(pkg) === nothing || continue   # actually loadable here
        err = try
            helper()
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin(pkg, sprint(showerror, err))
    end
end

@testset "backends agree with the reference implementation" begin
    function evolve(backend, bc; n = 48, steps = 300)
        model = Heat2D(nx = n, ny = n ÷ 2 + 1; boundary = bc)
        initialize_peak!(model.field, 100.0f0)
        initialize_peak!(model.field, 40.0f0; x = 5, y = 7)   # break the symmetry
        metrics = run!(model; backend = backend, steps = steps)
        return Array(state(model)), metrics
    end

    for bc in (Neumann(), Periodic(), Dirichlet(1.5f0))
        reference, ref_metrics = evolve(CPUBackend(), bc)
        @testset "$(nameof(typeof(bc)))" begin
            for (name, backend) in CPU_BACKENDS[2:end]
                result, metrics = evolve(backend, bc)
                # Same arithmetic in the same order: these must match exactly.
                @test result == reference
                @test metrics.steps == ref_metrics.steps
            end
            for (name, backend) in GPU_BACKENDS
                result, metrics = evolve(backend, bc)
                # GPUs may contract a multiply-add differently, so compare within
                # floating-point tolerance rather than bit for bit.
                @test result ≈ reference rtol = 1e-4 atol = 1e-5
                @test metrics.transferred_bytes > 0     # a real transfer happened
            end
        end
    end
end

@testset "RawCUDABackend" begin
    @test backend_name(RawCUDABackend()) === :cuda_raw
    @test is_gpu(RawCUDABackend())
    @test RawCUDABackend().threads == (16, 16)
    @test RawCUDABackend(threads = (32, 8)).threads == (32, 8)
    @test_throws ArgumentError RawCUDABackend(threads = (0, 16))
    @test_throws ArgumentError RawCUDABackend(threads = (64, 64))   # 4096 > 1024

    if Base.identify_package("CUDA") === nothing
        # Without CUDA loaded it must fail with a message that says so.
        err = try
            run!(Heat2D(nx = 8); backend = RawCUDABackend(), steps = 1)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("CUDA", sprint(showerror, err))
    else
        # The hand-written launch must agree with the portable kernel exactly —
        # they share `heat_update`, so only the launch differs.
        pattern = zeros(Float32, 40, 40)
        pattern[10, 10] = 1.5f0

        for kwargs in ((;),
                       (; boundary = Dirichlet(2.0f0)),
                       (; boundary = Periodic()),
                       (; boundary = Dirichlet(t -> 3.0f0 * sin(t))),
                       (; source = PatternSource(copy(pattern), 0.4f0)),
                       (; source = ProportionalSource(8.0f0, 1.5f0)))
            reference = Heat2D(nx = 40, ny = 40; kwargs...)
            initialize_peak!(reference.field, 60.0f0)
            run!(reference; backend = CPUBackend(), steps = 150)

            for threads in ((16, 16), (32, 8), (8, 32))
                model = Heat2D(nx = 40, ny = 40; kwargs...)
                initialize_peak!(model.field, 60.0f0)
                metrics = run!(model; backend = RawCUDABackend(threads = threads), steps = 150)
                @test Array(state(model)) ≈ Array(state(reference)) rtol = 1e-4 atol = 1e-5
                @test metrics.backend === :cuda_raw
            end
        end

        # Grid not a multiple of the block size: the bounds guard is doing work.
        odd_reference = Heat2D(nx = 37, ny = 23)
        initialize_peak!(odd_reference.field, 20.0f0)
        run!(odd_reference; backend = CPUBackend(), steps = 100)
        odd = Heat2D(nx = 37, ny = 23)
        initialize_peak!(odd.field, 20.0f0)
        run!(odd; backend = RawCUDABackend(), steps = 100)
        @test Array(state(odd)) ≈ Array(state(odd_reference)) rtol = 1e-4 atol = 1e-5

        # It gets host/device movement and residency from the shared machinery.
        resident = to_backend(Heat2D(nx = 32, ny = 32), RawCUDABackend())
        @test !(state(resident) isa Array)
        @test run!(resident; backend = RawCUDABackend(), steps = 20).transferred_bytes == 0
    end
end

@testset "threading does not change the result" begin
    # Guards against a race in the column decomposition: with a shared `next`
    # buffer and a real double buffer swap, threads never read what another
    # thread wrote in the same step.
    model_serial = Heat2D(nx = 101, ny = 97)
    initialize_gaussian!(model_serial.field; sigma = 12.0f0)
    model_threaded = Heat2D(nx = 101, ny = 97)
    initialize_gaussian!(model_threaded.field; sigma = 12.0f0)

    run!(model_serial; backend = CPUBackend(), steps = 500)
    run!(model_threaded; backend = CPUBackend(threaded = true), steps = 500)
    @test Array(state(model_serial)) == Array(state(model_threaded))
end

@testset "device residency across runs" begin
    for (name, backend) in GPU_BACKENDS
        @testset "$name" begin
            model = Heat2D(nx = 32, ny = 32)
            initialize_peak!(model.field, 100.0f0)

            resident = to_backend(model, backend)
            @test !(state(resident) isa Array)

            # A model already living on the device costs no transfer.
            metrics = run!(resident; backend = backend, steps = 50)
            @test metrics.transferred_bytes == 0

            # Chained runs continue from where the previous one stopped, so two
            # runs of 50 steps must equal one run of 100.
            second = run!(resident; backend = backend, steps = 50)
            @test second.transferred_bytes == 0

            reference = Heat2D(nx = 32, ny = 32)
            initialize_peak!(reference.field, 100.0f0)
            run!(reference; backend = CPUBackend(), steps = 100)
            @test Array(state(resident)) ≈ Array(state(reference)) rtol = 1e-4 atol = 1e-5

            # Single-cell accessors must work on device memory too. GPU array
            # libraries reject scalar indexing, so these have to go through an
            # explicit one-element copy rather than `data[i, j]`.
            @test center_value(resident) ≈ center_value(reference) rtol = 1e-4
            @test sum_state(resident) ≈ sum_state(reference) rtol = 1e-4

            fresh = to_backend(Heat2D(nx = 16, ny = 16), backend)
            initialize_peak!(fresh.field, 42.0f0)
            @test center_value(fresh) == 42.0f0
            @test sum_state(fresh) ≈ 42.0f0
        end
    end
end

@testset "metrics are self-consistent" begin
    model = Heat2D(nx = 64, ny = 32)
    initialize_peak!(model.field, 100.0f0)
    metrics = run!(model; backend = CPUBackend(), steps = 100)

    @test metrics.backend === :cpu
    @test metrics.precision === Float32
    @test metrics.cells == 64 * 32
    @test metrics.steps == 100
    @test metrics.stopped_by === :steps
    @test cell_updates(metrics) == 100 * 64 * 32
    @test metrics.compute_seconds > 0
    @test metrics.elapsed_seconds >= metrics.compute_seconds
    @test mlups(metrics) > 0
    @test bandwidth_gbs(metrics) > 0
    @test arithmetic_intensity(metrics) ≈ 10 / 8      # 10 FLOP per 8 bytes in Float32
    @test isfinite(gflops(metrics))
    # Float64 halves the arithmetic intensity: same FLOPs, twice the bytes.
    wide = Heat2D(nx = 64, ny = 32, initial = 0.0)
    @test arithmetic_intensity(run!(wide; steps = 10)) ≈ 10 / 16

    @test occursin("MLUP/s", sprint(show, MIME"text/plain"(), metrics))
    @test occursin("RunMetrics", sprint(show, metrics))
end
