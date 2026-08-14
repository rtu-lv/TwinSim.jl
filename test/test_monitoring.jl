@testset "synthetic_series" begin
    series = synthetic_series(samples = 120, seed = 3)
    @test length(series) == 120
    @test length(series.values) == 120
    @test !any(series.anomalous)                  # no faults requested
    @test series.values != series.clean           # but noise was applied
    @test series(0.0) ≈ series.values[1]
    @test series(119.0) ≈ series.values[end]
    @test series(0.5) ≈ (series.values[1] + series.values[2]) / 2   # interpolates

    # Reproducible from the seed alone.
    @test synthetic_series(samples = 50, seed = 7).values ==
          synthetic_series(samples = 50, seed = 7).values
    @test synthetic_series(samples = 50, seed = 7).values !=
          synthetic_series(samples = 50, seed = 8).values

    # Noise-free reproduces the clean signal exactly.
    quiet = synthetic_series(samples = 30, noise = 0.0, seed = 1)
    @test quiet.values == quiet.clean

    @test_throws ArgumentError synthetic_series(samples = 1)
    @test_throws ArgumentError synthetic_series(step = 0)
    @test_throws ArgumentError synthetic_series(noise = -1)
    @test_throws ArgumentError synthetic_series(period = 0)
    @test_throws ArgumentError synthetic_series(samples = 20, anomalies = [Spike(50, 1.0)])
end

@testset "anomalies mark the right samples" begin
    n = 100

    @testset "Spike" begin
        s = synthetic_series(samples = n, noise = 0.0, anomalies = [Spike(40, 9.0)], seed = 1)
        @test count(s.anomalous) == 1
        @test s.anomalous[40]
        @test s.values[40] ≈ s.clean[40] + 9.0
        @test s.values[41] ≈ s.clean[41]           # and nothing else moved
    end

    @testset "LevelShift persists" begin
        s = synthetic_series(samples = n, noise = 0.0, anomalies = [LevelShift(60, 4.0)], seed = 1)
        @test count(s.anomalous) == n - 59
        @test !s.anomalous[59]
        @test s.values[59] ≈ s.clean[59]
        @test s.values[60] ≈ s.clean[60] + 4.0
        @test s.values[end] ≈ s.clean[end] + 4.0   # never recovers
    end

    @testset "Drift ramps" begin
        s = synthetic_series(samples = n, noise = 0.0, anomalies = [Drift(50, 0.2)], seed = 1)
        @test s.values[50] ≈ s.clean[50] + 0.2
        @test s.values[60] ≈ s.clean[60] + 0.2 * 11
        @test s.values[end] ≈ s.clean[end] + 0.2 * (n - 49)
    end

    @testset "Stuck freezes" begin
        s = synthetic_series(samples = n, noise = 0.0, anomalies = [Stuck(30, 10)], seed = 1)
        @test count(s.anomalous) == 10
        @test all(≈(s.values[30]), s.values[30:39])
        @test s.values[40] ≈ s.clean[40]           # released afterwards
    end

    @testset "several at once" begin
        s = synthetic_series(samples = n, noise = 0.0, seed = 1,
                             anomalies = [Spike(20, 5.0), LevelShift(70, 3.0)])
        @test s.anomalous[20]
        @test !s.anomalous[21]
        @test all(s.anomalous[70:end])
    end
end

@testset "detection_report" begin
    truth = [false, false, true, true, false, false]

    perfect = detection_report([false, false, true, true, false, false], truth)
    @test perfect.detected
    @test perfect.delay == 0
    @test perfect.true_positives == 2
    @test perfect.false_positives == 0
    @test perfect.recall ≈ 1.0
    @test perfect.precision ≈ 1.0
    @test perfect.false_positive_rate ≈ 0.0

    late = detection_report([false, false, false, true, false, false], truth)
    @test late.detected
    @test late.delay == 1
    @test late.recall ≈ 0.5

    missed = detection_report([false, false, false, false, false, false], truth)
    @test !missed.detected
    @test missed.delay === missing
    @test missed.recall ≈ 0.0
    @test isnan(missed.precision)              # no flags raised at all

    # A flag *before* the anomaly is a false positive, not an early detection.
    early = detection_report([true, false, false, false, false, false], truth)
    @test !early.detected
    @test early.delay === missing
    @test early.false_positives == 1

    noisy = detection_report([true, true, true, true, true, true], truth)
    @test noisy.delay == 0
    @test noisy.false_positives == 4
    @test noisy.false_positive_rate ≈ 1.0
    @test noisy.recall ≈ 1.0
    @test noisy.precision ≈ 2 / 6

    @test_throws DimensionMismatch detection_report([true], truth)

    # No anomaly present: recall is undefined rather than zero.
    quiet = detection_report([false, false], [false, false])
    @test !quiet.detected
    @test isnan(quiet.recall)
    @test quiet.false_positive_rate ≈ 0.0
end

@testset "flagging by threshold" begin
    residuals = [0.1, -0.4, 2.5, -3.0, 0.2]
    @test flag_exceedances(residuals, 1.0) == [false, false, true, true, false]
    @test flag_exceedances(residuals, 5.0) == falses(5)
    @test flag_exceedances(residuals, 0.0) == trues(5)

    truth = [false, false, true, true, false]
    report = detection_report(residuals, truth, 1.0)
    @test report.threshold == 1.0
    @test report.delay == 0
    @test report.false_positives == 0

    sweep = threshold_sweep(residuals, truth, [0.0, 1.0, 5.0])
    @test length(sweep) == 3
    @test [r.threshold for r in sweep] == [0.0, 1.0, 5.0]
    # Raising the threshold can only reduce the flags raised.
    @test issorted([r.true_positives + r.false_positives for r in sweep]; rev = true)

    @test occursin("false positive rate", sprint(show, MIME"text/plain"(), report))
    @test occursin("DetectionReport", sprint(show, report))
end

@testset "detection on a synthetic series" begin
    # A perfect model: the residual is measurement noise plus the fault.
    observed = synthetic_series(samples = 120, noise = 0.4, seed = 42,
                                anomalies = [LevelShift(70, 6.0)])
    residuals = observed.values .- observed.clean

    tight = detection_report(residuals, observed.anomalous, 1.0)
    loose = detection_report(residuals, observed.anomalous, 3.0)

    @test tight.detected && loose.detected
    @test tight.delay == 0
    @test tight.false_positives >= loose.false_positives   # the trade-off, measured
    @test loose.false_positives == 0

    # Stuck is invisible to a check on the *values* — a frozen sensor reads
    # perfectly plausibly — but a residual against a model that expects movement
    # does see it. Which detector you use decides whether the fault exists.
    stuck = synthetic_series(samples = 120, noise = 0.05, seed = 5,
                             anomalies = [Stuck(60, 15)])

    # A range check on the values alone: the frozen samples are unremarkable.
    span = extrema(stuck.clean)
    in_range = [span[1] - 1 <= v <= span[2] + 1 for v in stuck.values]
    @test all(in_range[60:74])

    # A residual against the model does catch it, once the true signal has moved.
    residuals = stuck.values .- stuck.clean
    @test detection_report(residuals, stuck.anomalous, 3.0).detected
end

@testset "MetricRecorder" begin
    recorder = MetricRecorder(total = m -> sum(state(m)),
                              centre = m -> center_value(m))
    @test isempty(recorder)
    @test Set(keys(recorder)) == Set((:total, :centre))

    model = Heat2D(nx = 32, ny = 32)
    initialize_peak!(model.field, 100.0f0)
    run!(Simulation(model; stop = Steps(500)); callback = recorder, callback_every = 100)

    @test length(recorder) == 5
    @test recorder.steps == [100, 200, 300, 400, 500]
    @test length(recorder[:total]) == 5
    @test all(≈(100.0; rtol = 1e-4), recorder[:total])     # Neumann conserves
    @test issorted(recorder[:centre]; rev = true)          # the peak decays
    @test recorder.times[end] ≈ 50.0 rtol = 1e-4

    empty!(recorder)
    @test isempty(recorder)
    @test isempty(recorder[:total])

    @test_throws ArgumentError MetricRecorder()
    @test occursin("MetricRecorder", sprint(show, recorder))
end

@testset "MetricRecorder works on every backend" begin
    reference = nothing
    for backend in vcat([CPUBackend(), CPUBackend(threaded = true), KernelBackend()],
                        last.(GPU_BACKENDS))
        recorder = MetricRecorder(total = m -> sum(state(m)))
        model = Heat2D(nx = 32, ny = 32)
        initialize_peak!(model.field, 100.0f0)
        run!(Simulation(model; backend = backend, stop = Steps(200));
             callback = recorder, callback_every = 50)
        @test length(recorder) == 4
        if reference === nothing
            reference = recorder[:total]
        else
            @test recorder[:total] ≈ reference rtol = 1e-4
        end
    end
end

@testset "TwinLoop stages and ordering" begin
    @test_throws ArgumentError TwinLoop(check = :sometimes)
    @test_throws ArgumentError TwinLoop(steps_per_window = 0)
    @test TwinLoop().check === :forecast          # the safe default

    # A loop with every stage left at its default still runs.
    model = Heat2D(nx = 16, ny = 16)
    log = twin_run!(TwinLoop(steps_per_window = 5), model, collect(1.0:10.0))
    @test length(log) == 10
    @test all(isnothing, log.decisions)
    @test sum(m -> m.steps, log.metrics) == 50

    # The stages run, in order, once per observation.
    trace = Symbol[]
    loop = TwinLoop(
        assimilate = (m, o, t) -> (push!(trace, :assimilate); nothing),
        validate = (m, o, t) -> (push!(trace, :validate); (; value = o)),
        decide = (m, c, t) -> (push!(trace, :decide); c.value > 1.5 ? :high : :low),
        steps_per_window = 2,
    )
    log = twin_run!(loop, Heat2D(nx = 16, ny = 16), [1.0, 2.0])
    @test trace == [:validate, :assimilate, :decide, :validate, :assimilate, :decide]
    @test log.decisions == [:low, :high]
    @test log.observations == [1.0, 2.0]

    # :analysis assimilates first instead.
    empty!(trace)
    analysis = TwinLoop(
        assimilate = (m, o, t) -> (push!(trace, :assimilate); nothing),
        validate = (m, o, t) -> (push!(trace, :validate); nothing),
        steps_per_window = 2, check = :analysis,
    )
    twin_run!(analysis, Heat2D(nx = 16, ny = 16), [1.0])
    @test trace == [:assimilate, :validate]
end

@testset "twin_run! observation timing" begin
    model = Heat2D(nx = 16, ny = 16, dt = 0.1f0)

    # Default cadence: one observation per window of simulated time.
    log = twin_run!(TwinLoop(steps_per_window = 10), model, ones(4))
    @test log.times ≈ [0.0, 1.0, 2.0, 3.0]

    # Explicit times are used verbatim.
    log = twin_run!(TwinLoop(steps_per_window = 10), Heat2D(nx = 16, ny = 16),
                    ones(3); times = [5.0, 10.0, 20.0])
    @test log.times == [5.0, 10.0, 20.0]

    # An ObservationSeries brings its own.
    observed = synthetic_series(samples = 6, step = 2.0, seed = 1)
    log = twin_run!(TwinLoop(steps_per_window = 3), Heat2D(nx = 16, ny = 16), observed)
    @test log.times == observed.times
    @test log.observations == observed.values

    @test_throws DimensionMismatch twin_run!(TwinLoop(), Heat2D(nx = 8), ones(3);
                                             times = [1.0, 2.0])
end

@testset "TwinLog reports cost" begin
    model = Heat2D(nx = 24, ny = 24, dt = 0.1f0)
    log = twin_run!(TwinLoop(steps_per_window = 20), model, ones(10))
    @test length(log) == 10
    @test simulated_time(log) ≈ 10 * 20 * 0.1 rtol = 1e-4
    @test compute_seconds(log) > 0
    @test realtime_ratio(log) > 1
    @test occursin("TwinLog", sprint(show, MIME"text/plain"(), log))
end

@testset "a twin detects a fault it is not tracking" begin
    # End to end: plant develops a fault, twin does not know, innovation reveals it.
    observed = synthetic_series(samples = 100, noise = 0.2, seed = 11,
                                anomalies = [LevelShift(60, 8.0)])

    # steps_per_window * dt must equal the observation spacing, or the boundary
    # drive and the observations are sampled at different instants.
    twin = Heat2D(nx = 24, ny = 24, initial = 0.0f0, dt = 0.1f0;
                  boundary = Dirichlet(TimeSeries(observed.times, observed.clean)))
    loop = TwinLoop(
        validate = (m, o, t) -> (; innovation = o - Float64(m.field[1, 12])),
        decide = (m, c, t) -> abs(c.innovation) > 3.0 ? :alarm : :ok,
        steps_per_window = 10,     # 10 * 0.1 = 1.0, matching the 1.0 sample step
    )
    log = twin_run!(loop, twin, observed)

    innovation = [c.innovation for c in log.checks]
    report = detection_report(innovation, observed.anomalous, 3.0)
    @test report.detected
    @test report.delay <= 2
    @test report.false_positives == 0
    @test count(==(:alarm), log.decisions) > 30
end
