# Lab 03: From Simulation to Digital Twin

Labs 01 and 02 built a simulation: give it an initial state, run it, get a
result. A twin is different in three ways, and this lab adds them one at a time.

1. It **ingests measurements** from the system it mirrors.
2. It runs **against the wall clock**, not as fast as possible.
3. It **persists**, so it can be restarted without losing where it was.

```bash
julia --project=. examples/digital_twin.jl
```

## 1. Assimilation

The plant is another simulation with heaters the twin knows nothing about. The
twin sees only point measurements:

```julia
sensors = [Sensor(i, j, plant.field[i, j]) for (i, j) in probes]
nudge!(twin, sensors; gain = 0.7, radius = 4)
```

`nudge!` pulls the state towards the measurement:

```
u[i,j] += gain * w(distance) * (measured - u[i,j])
```

`gain` trades trust in the model against trust in the instrument. `radius` is the
**localisation radius**: with `radius = 0` a sensor corrects exactly one cell,
and with `radius > 0` the correction is spread over the neighbourhood with
Gaussian weight.

Measured RMS error against the plant after 40 assimilation windows on a 64x64
grid (4096 cells):

| sensors | radius | open-loop RMS | twin RMS | improvement |
|--------:|-------:|--------------:|---------:|------------:|
| 16 | 0 | 16.846 | 16.162 | 1.0x |
| 16 | 8 | 16.846 | 10.483 | 1.6x |
| 16 | 16 | 16.846 | 12.151 | 1.4x |
| 64 | 0 | 16.846 | 14.202 | 1.2x |
| 64 | 4 | 16.846 | 3.380 | **5.0x** |
| 64 | 8 | 16.846 | 7.603 | 2.2x |
| 256 | 0 | 16.846 | 8.663 | 1.9x |
| 256 | 2 | 16.846 | 1.020 | **16.5x** |
| 256 | 4 | 16.846 | 2.979 | 5.7x |

The lesson is in the `radius = 0` rows. Sixteen times more sensors, used as point
corrections, buys a factor of 1.9. The same 64 sensors with a sensible influence
radius buy a factor of 5.0. **The localisation radius matters more than the
sensor count.**

The reason is physical. A temperature reading is evidence about a region, not a
point, because diffusion correlates neighbouring cells. Nudging one cell out of
4096 injects a correction that diffusion spreads more slowly than the error
grows. Choosing the radius is choosing how far you believe that correlation
reaches — which is exactly the information a Kalman filter carries explicitly in
its covariance matrix, and which the localisation step of an ensemble Kalman
filter truncates for the same reason it is truncated here.

Notice also that the best radius tracks the sensor spacing: stride 8 wants
radius 4, stride 4 wants radius 2. Too large over-smooths and the improvement
falls again.

Questions:

1. Why does `radius = 16` do *worse* than `radius = 8` with 16 sensors?
2. The example's heaters are smooth Gaussian sources. Replace them with
   single-cell sources and re-run. Why does assimilation get so much worse, and
   what does that say about what sparse sensors can and cannot reconstruct?
3. `gain = 1.0, radius = 0` is direct insertion — the twin simply adopts the
   measurement. When is that the right choice, and when is it the worst one?
4. Sketch what would have to change for `nudge!` to become optimal
   interpolation. What extra information would you need?

## 2. Running against the clock

A batch simulation finishes as fast as it can. A twin must not: it has to stay
synchronised with the thing it mirrors.

```julia
metrics = run!(Simulation(twin; stop = UntilTime(3.0)); realtime_factor = 1.0)
```

Measured output:

```
  30 steps, simulated time 3.00, wall clock 3.00 s
  of which 0.0002 s was compute — the rest was the twin waiting for the world
```

The ratio of compute time to wall time is the twin's **headroom**: 0.0002 / 3.0
here, so this model could run about 15 000x faster than real time. That headroom
is the budget available for a finer grid, a better assimilation scheme, or
running an ensemble of twins for uncertainty estimation.

Questions:

5. At what grid size does `realtime_factor = 1.0` stop being achievable on your
   machine? Use Lab 02's table to predict it before measuring.
6. What should a twin do when it *cannot* keep up — drop steps, coarsen the
   model, or fall behind? What does each choice cost?

## 3. Persistence

```julia
save_state("twin_00030.vts", twin; step = 30, simulated_time = 3.0)
restored = load_state("twin_00030.vts")
resumed = Heat2D(restored.field; boundary = restored.boundary, alpha = 0.15f0, dt = 0.1f0)
```

Checkpoints can be written from a callback during a run:

```julia
run!(sim; callback = checkpoint_callback("out/twin_%05d.vts"), callback_every = 100)
```

The format is a documented 50-byte header plus raw column-major data (see the
comment block in `src/twin.jl`) rather than a Julia serialisation. That choice is
deliberate: it survives Julia version changes, and it can be read from C++ or
Python in a few lines.

Note that `load_state` deliberately does *not* store the model parameters. A
restart usually wants to change them, and silently reusing the old ones hides
that decision.

Questions:

7. The checkpoint stores the boundary condition but not `alpha` or `dt`. Argue
   for and against that split.
8. A callback that checkpoints every step would dominate the run time. Measure
   it: at what `callback_every` does checkpointing cost less than 5%?
9. On a GPU backend, `save_state` copies the state to the host first. Where does
   that time show up in `RunMetrics`, and why is it not counted as compute?

## Putting it together

The final exercise for the course: take the `digital_twin.jl` example, move the
twin onto a GPU backend with `to_backend`, and keep assimilation working. You
will have to decide where the host/device boundary sits, because `nudge!` needs
host memory while the stepping wants to stay on the device.

That decision — how often the twin comes back to the host, and what it costs — is
the central engineering trade-off of a GPU-accelerated digital twin, and it is
exactly the number `RunMetrics.transfer_seconds` was added to expose.
