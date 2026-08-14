# Validation against the closed-form solution of the heat equation, rather than
# against the package's own previous output. These are the tests that would
# catch a wrong coefficient, a swapped axis or a mis-scaled time step — none of
# which change any of the smoke-test properties.

"""
    moments(field; dx, dy) -> (mass, mean_x, mean_y, var_x, var_y)

Zeroth, first and second spatial moments of the field, treating it as a
distribution.
"""
function moments(field; dx = 1.0, dy = 1.0)
    u = Array(state(field))
    nx, ny = size(u)
    xs = [(i - (nx + 1) / 2) * dx for i in 1:nx]
    ys = [(j - (ny + 1) / 2) * dy for j in 1:ny]

    mass = sum(u)
    mean_x = sum(u[i, j] * xs[i] for i in 1:nx, j in 1:ny) / mass
    mean_y = sum(u[i, j] * ys[j] for i in 1:nx, j in 1:ny) / mass
    var_x = sum(u[i, j] * (xs[i] - mean_x)^2 for i in 1:nx, j in 1:ny) / mass
    var_y = sum(u[i, j] * (ys[j] - mean_y)^2 for i in 1:nx, j in 1:ny) / mass
    return (; mass, mean_x, mean_y, var_x, var_y)
end

@testset "variance grows at exactly 2*alpha*t" begin
    # For the five-point stencil the second moment satisfies
    #     sum_i x_i^2 (u_{i-1} - 2u_i + u_{i+1}) = 2*dx^2 * sum_i u_i,
    # so the discrete variance grows by exactly 2*alpha*dt per step with no
    # discretisation error at all. That makes this an unusually sharp test: it
    # pins down alpha, dt and both grid spacings simultaneously.
    #
    # dx != dy on purpose. The growth rate is independent of the spacing, so a
    # run with dx == dy cannot tell cx and cy apart; with dx != dy, swapping them
    # changes the rate by (dx/dy)^2 and the test fails.
    alpha, dt, dx, dy = 0.1, 0.2, 1.0, 2.0
    steps = 300

    model = Heat2D(nx = 129, ny = 129, initial = 0.0;
                   alpha = alpha, dt = dt, dx = dx, dy = dy, boundary = Neumann())
    initialize_gaussian!(model.field; amplitude = 1.0, sigma = 6.0, dx = dx, dy = dy)

    before = moments(model.field; dx, dy)
    run!(model; steps = steps)
    after = moments(model.field; dx, dy)

    elapsed = steps * dt
    @test after.mass ≈ before.mass rtol = 1e-12          # Neumann conserves
    @test after.var_x ≈ before.var_x + 2 * alpha * elapsed rtol = 1e-9
    @test after.var_y ≈ before.var_y + 2 * alpha * elapsed rtol = 1e-9
    # The bump must not drift.
    @test after.mean_x ≈ before.mean_x atol = 1e-9
    @test after.mean_y ≈ before.mean_y atol = 1e-9
end

@testset "solution matches the analytical Gaussian profile" begin
    # A Gaussian stays Gaussian: u(r, t) = A * s0^2/s(t)^2 * exp(-r^2/(2 s(t)^2))
    # with s(t)^2 = s0^2 + 2*alpha*t. This checks the shape, which the moment
    # test above cannot see.
    alpha, dt, dx = 0.1, 0.05, 0.25
    n, steps = 129, 400
    sigma0 = 1.0

    model = Heat2D(nx = n, ny = n, initial = 0.0;
                   alpha = alpha, dt = dt, dx = dx, dy = dx, boundary = Neumann())
    initialize_gaussian!(model.field; amplitude = 1.0, sigma = sigma0, dx = dx, dy = dx)
    run!(model; steps = steps)

    elapsed = steps * dt
    sigma_sq = sigma0^2 + 2 * alpha * elapsed
    centre = (n + 1) / 2
    u = Array(state(model))

    worst = 0.0
    for j in 1:n, i in 1:n
        rx = (i - centre) * dx
        ry = (j - centre) * dx
        exact = (sigma0^2 / sigma_sq) * exp(-(rx^2 + ry^2) / (2 * sigma_sq))
        worst = max(worst, abs(u[i, j] - exact))
    end
    @test worst < 5e-4
end

@testset "second-order spatial convergence" begin
    # Refine dx and dt together with dt ~ dx^2 so the CFL number stays fixed.
    # FTCS is first order in time and second in space, so with that coupling the
    # total error should fall by ~4x per refinement.
    alpha, sigma0, span, endtime = 0.1, 1.0, 16.0, 1.0

    function profile_error(n)
        dx = span / n
        dt = dx^2                       # CFL = alpha*dt*2/dx^2 = 0.2, constant
        steps = round(Int, endtime / dt)
        model = Heat2D(nx = n + 1, ny = n + 1, initial = 0.0;
                       alpha = alpha, dt = dt, dx = dx, dy = dx, boundary = Neumann())
        initialize_gaussian!(model.field; amplitude = 1.0, sigma = sigma0, dx = dx, dy = dx)
        run!(model; steps = steps)

        elapsed = steps * dt
        sigma_sq = sigma0^2 + 2 * alpha * elapsed
        centre = (n + 2) / 2
        u = Array(state(model))
        worst = 0.0
        for j in 1:(n + 1), i in 1:(n + 1)
            rx = (i - centre) * dx
            ry = (j - centre) * dx
            exact = (sigma0^2 / sigma_sq) * exp(-(rx^2 + ry^2) / (2 * sigma_sq))
            worst = max(worst, abs(u[i, j] - exact))
        end
        return worst
    end

    coarse = profile_error(32)
    medium = profile_error(64)
    fine = profile_error(128)

    @test medium < coarse
    @test fine < medium
    # Observed ratios are close to 4; 3.0 leaves room for the first-order time
    # term without letting a first-order scheme pass.
    @test coarse / medium > 3.0
    @test medium / fine > 3.0
end
