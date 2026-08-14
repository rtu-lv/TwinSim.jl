using Test

# CSV export has no dependencies, so it is tested everywhere. Plotting is tested
# only when a Makie backend is present in the environment.
# invokelatest: the extension loaded in a newer world age than this file.
const PLOTTING = Base.invokelatest(plotting_available)

if PLOTTING
    @info "Makie backend loaded; figure tests will run"
else
    @info "No Makie backend; figure tests skipped (CSV export still tested)"
end

@testset "write_csv: MetricRecorder" begin
    recorder = MetricRecorder(total = m -> sum(state(m)), centre = m -> center_value(m))
    model = Heat2D(nx = 16, ny = 16)
    initialize_peak!(model.field, 100.0f0)
    run!(Simulation(model; stop = Steps(200)); callback = recorder, callback_every = 50)

    path = joinpath(mktempdir(), "metrics.csv")
    @test write_csv(path, recorder) == path

    lines = readlines(path)
    @test length(lines) == 5                       # header + 4 samples
    @test lines[1] == "step,simulated_time,total,centre"
    @test startswith(lines[2], "50,")
    @test length(split(lines[2], ',')) == 4
end

@testset "write_csv: ObservationSeries" begin
    observed = synthetic_series(samples = 30, seed = 3, anomalies = [LevelShift(20, 5.0)])
    path = joinpath(mktempdir(), "observations.csv")
    write_csv(path, observed)

    lines = readlines(path)
    @test length(lines) == 31
    @test lines[1] == "index,time,value,clean,anomalous"
    # The ground truth travels with the data, which is the point.
    @test endswith(lines[2], ",false")
    @test endswith(lines[end], ",true")
end

@testset "write_csv: TwinLog" begin
    observed = synthetic_series(samples = 20, seed = 4)
    model = Heat2D(nx = 16, ny = 16, dt = 0.1f0)
    loop = TwinLoop(validate = (m, o, t) -> (; innovation = o - Float64(center_value(m)),
                                               ok = true),
                    decide = (m, c, t) -> c.innovation > 0 ? :high : :low,
                    steps_per_window = 10)
    log = twin_run!(loop, model, observed)

    path = joinpath(mktempdir(), "twin.csv")
    write_csv(path, log)
    lines = readlines(path)
    @test length(lines) == 21
    # A NamedTuple check expands into one column per field.
    @test lines[1] == "window,time,observation,innovation,ok,decision,steps,compute_seconds"
    @test length(split(lines[2], ',')) == 8

    # A non-NamedTuple check collapses into a single quoted column.
    plain = twin_run!(TwinLoop(validate = (m, o, t) -> "state ok, nothing to report",
                               steps_per_window = 5),
                      Heat2D(nx = 8, ny = 8), observed)
    other = joinpath(mktempdir(), "plain.csv")
    write_csv(other, plain)
    header = readlines(other)[1]
    @test occursin("checks", header)
    # The comma inside the check string must be quoted, not left to become a
    # column break. (A naive `split(line, ',')` would count 8 fields here — which
    # is exactly why the quoting matters.)
    @test occursin("\"state ok, nothing to report\"", readlines(other)[2])
end

@testset "write_csv: sweep results" begin
    results = parameter_sweep([0.1f0, 0.2f0]; stop = Steps(50)) do alpha
        model = Heat2D(nx = 16, ny = 16, alpha = alpha)
        initialize_peak!(model.field, 100.0f0)
        model
    end

    path = joinpath(mktempdir(), "sweep.csv")
    write_csv(path, results)
    lines = readlines(path)
    @test length(lines) == 3
    @test occursin("scenario", lines[1])
    @test occursin("mlups", lines[1])

    @test_throws ArgumentError write_csv(path, NamedTuple[])
    @test_throws ArgumentError write_csv(path, [(; a = 1)])
end

@testset "plotting fallback explains itself" begin
    if !PLOTTING
        for call in (() -> plot_field(Heat2D(nx = 8)),
                     () -> plot_series(MetricRecorder(a = sum_state)),
                     () -> plot_detection([1.0], [false]),
                     () -> plot_sweep([]),
                     () -> animate_field(Heat2D(nx = 8), "x.mp4"),
                     () -> frame_callback("f_%03d.png"))
            err = try
                call()
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            # It must name the package to install and the dependency-free path.
            message = sprint(showerror, err)
            @test occursin("CairoMakie", message)
            @test occursin("write_csv", message)
        end
    end
end

if PLOTTING
    @testset "figures render to files" begin
        dir = mktempdir()

        @testset "plot_field" begin
            model = Heat2D(nx = 32, ny = 32)
            initialize_peak!(model.field, 100.0f0)
            run!(model; steps = 100)

            path = joinpath(dir, "field.png")
            save(path, plot_field(model; title = "after 100 steps"))
            @test isfile(path) && filesize(path) > 1000

            # A uniform field has a zero-width colour range, which Makie rejects
            # unless it is widened.
            flat = Heat2D(nx = 16, ny = 16, initial = 3.0f0)
            save(joinpath(dir, "flat.png"), plot_field(flat))
            @test isfile(joinpath(dir, "flat.png"))
        end

        @testset "plot_field on a GPU-resident model" begin
            for (name, backend) in GPU_BACKENDS
                model = to_backend(Heat2D(nx = 32, ny = 32), backend)
                initialize_peak!(model.field, 50.0f0)
                run!(model; backend = backend, steps = 50)
                path = joinpath(dir, "gpu_$name.png")
                save(path, plot_field(model))
                @test isfile(path) && filesize(path) > 1000
            end
        end

        @testset "plot_series" begin
            recorder = MetricRecorder(total = m -> sum(state(m)),
                                      centre = m -> center_value(m))
            model = Heat2D(nx = 24, ny = 24)
            initialize_peak!(model.field, 100.0f0)
            run!(Simulation(model; stop = Steps(300)); callback = recorder, callback_every = 25)

            path = joinpath(dir, "series.png")
            save(path, plot_series(recorder))
            @test isfile(path) && filesize(path) > 1000

            save(joinpath(dir, "one.png"), plot_series(recorder; metrics = [:centre]))
            @test isfile(joinpath(dir, "one.png"))

            @test_throws ArgumentError plot_series(recorder; metrics = [:nonexistent])
            @test_throws ArgumentError plot_series(MetricRecorder(a = sum_state))
        end

        @testset "plot_detection" begin
            observed = synthetic_series(samples = 100, seed = 9,
                                        anomalies = [LevelShift(60, 5.0)])
            residuals = observed.values .- observed.clean

            path = joinpath(dir, "detection.png")
            save(path, plot_detection(observed.times, residuals, observed.anomalous;
                                      thresholds = [1.0, 2.0, 4.0]))
            @test isfile(path) && filesize(path) > 1000

            # Times default to sample indices.
            save(joinpath(dir, "detection2.png"), plot_detection(residuals, observed.anomalous))
            @test isfile(joinpath(dir, "detection2.png"))

            @test_throws DimensionMismatch plot_detection([1.0, 2.0], [1.0], [true])
        end

        @testset "plot_sweep" begin
            results = parameter_sweep([0.05f0, 0.1f0, 0.15f0]; stop = Steps(50)) do alpha
                model = Heat2D(nx = 16, ny = 16, alpha = alpha)
                initialize_peak!(model.field, 100.0f0)
                model
            end
            path = joinpath(dir, "sweep.png")
            save(path, plot_sweep(results; y = :centre))
            @test isfile(path) && filesize(path) > 1000
            @test_throws ArgumentError plot_sweep([])
        end

        @testset "animate_field" begin
            model = Heat2D(nx = 24, ny = 24)
            initialize_peak!(model.field, 100.0f0)
            path = joinpath(dir, "heat.mp4")
            @test animate_field(model, path; frames = 8, steps_per_frame = 5) == path
            @test isfile(path) && filesize(path) > 1000
            # The run really advanced: 8 frames x 5 steps.
            @test simulated_time(model) ≈ 40 * 0.1 rtol = 1e-4

            @test_throws ArgumentError animate_field(Heat2D(nx = 8), path; frames = 0)
        end

        @testset "frame_callback streams frames" begin
            model = Heat2D(nx = 16, ny = 16)
            initialize_peak!(model.field, 100.0f0)
            frames = mktempdir()
            run!(Simulation(model; stop = Steps(60));
                 callback = frame_callback(joinpath(frames, "frame_%04d.png");
                                           colorrange = (0.0, 100.0)),
                 callback_every = 20)
            written = sort(filter(endswith(".png"), readdir(frames)))
            @test written == ["frame_0020.png", "frame_0040.png", "frame_0060.png"]
        end
    end
end
