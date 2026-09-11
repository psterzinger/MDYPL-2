## Philipp Sterzinger 03.09.2026: Code is provided as is, no finished package
## and no guarantees given
##
## Limiting first-order conditions for mDYPL with an intercept: the oracle
## system, where the true and prior signals are known through theta0, thetaP and
## Gamma, and the SLOE system, where the true signal is replaced by plug-ins
## from one fitted data set. `solve_oracle` and `solve_sloe` solve them.

struct OracleProblem
    kappa::Float64
    alpha::Float64
    theta0::Float64
    thetaP::Float64
    Gamma::Matrix{Float64}
    basis::Matrix{Float64}
    roots::Vector{Float64}
    rank::Int
end

"""
    OracleProblem(; kappa, alpha, theta0, thetaP, Gamma)

Known-parameter system. The signal integral runs over the numerical range of
`Gamma` only, so rank-deficient Gram matrices (a null prior, or a null signal)
need no special handling by the caller.
"""
function OracleProblem(; kappa, alpha, theta0, thetaP, Gamma)
    0.0 < kappa < 1.0 || throw(ArgumentError("kappa must lie in (0,1)"))
    0.0 < alpha <= 1.0 || throw(ArgumentError("alpha must lie in (0,1]"))
    G = (Matrix{Float64}(Gamma) + Matrix{Float64}(Gamma)') / 2

    if alpha == 1.0
        gamma2 = max(G[1, 1], 0.0)
        return gamma2 > sqrt(eps()) ?
            OracleProblem(kappa, alpha, theta0, thetaP, G,
                          reshape([1.0, 0.0], 2, 1), [sqrt(gamma2)], 1) :
            OracleProblem(kappa, alpha, theta0, thetaP, G,
                          zeros(2, 0), Float64[], 0)
    end

    vals, vecs = eigen(Symmetric(G))
    minimum(vals) >= -100 * eps() * max(maximum(abs, vals), eps()) ||
        throw(ArgumentError("Gamma is not positive semidefinite"))
    active = findall(v -> v > sqrt(eps()) * max(maximum(abs, vals), eps()), vals)
    basis = Matrix{Float64}(vecs[:, active])
    for column in axes(basis, 2)
        basis[argmax(abs.(view(basis, :, column))), column] < 0 &&
            (basis[:, column] .*= -1.0)
    end
    OracleProblem(kappa, alpha, theta0, thetaP, G, basis,
                  sqrt.(max.(vals[active], 0.0)), length(active))
end

oracle_dimension(problem::OracleProblem) = problem.rank + 3

struct SLOEProblem
    kappa::Float64
    alpha::Float64
    thetaP::Float64
    delta2::Float64
    theta_fit::Float64
    nu2::Float64
    chiP::Float64
    delta::Float64
    tau::Float64
end

"""
    SLOEProblem(; kappa, alpha, thetaP, delta2, theta_fit, nu2, chiP)

Plug-in system driven by one fitted data set. When `delta2 == 0` positive
semidefiniteness requires `chiP == 0` and the reduced rank-one system is used.
"""
function SLOEProblem(; kappa, alpha, thetaP, delta2, theta_fit, nu2, chiP)
    0.0 < kappa < 1.0 || throw(ArgumentError("kappa must lie in (0,1)"))
    0.0 < alpha <= 1.0 || throw(ArgumentError("alpha must lie in (0,1]"))
    nu2 > 0.0 || throw(ArgumentError("nu2 must be strictly positive"))
    delta2 >= 0.0 || throw(ArgumentError("delta2 must be nonnegative"))
    if delta2 == 0.0
        abs(chiP) <= 100 * eps() * max(1.0, abs(chiP)) ||
            throw(ArgumentError("chiP must be zero when delta2 == 0"))
        return SLOEProblem(kappa, alpha, thetaP, delta2, theta_fit, nu2, chiP,
                           0.0, sqrt(nu2))
    end
    tau2 = nu2 - chiP^2 / delta2
    alpha == 1.0 && return SLOEProblem(kappa, alpha, thetaP, delta2, theta_fit,
                                       nu2, chiP, sqrt(delta2),
                                       tau2 > 0.0 ? sqrt(tau2) : 0.0)
    tau2 > 100 * eps() * max(1.0, nu2, chiP^2 / delta2) ||
        throw(ArgumentError("the fixed covariance of (Q2,T) is not positive definite"))
    SLOEProblem(kappa, alpha, thetaP, delta2, theta_fit, nu2, chiP,
                sqrt(delta2), sqrt(tau2))
end

sloe_rank_one(problem::SLOEProblem) = problem.alpha == 1.0 || problem.delta2 == 0.0
sloe_dimension(problem::SLOEProblem) = sloe_rank_one(problem) ? 4 : 5

"""
Quadrature rule for the signal coordinate. Rank-one problems use the panelled
rule split at the logistic transitions of the true and prior linear
predictors, which is where the integrand changes fastest.
"""
function oracle_signal_rule(problem::OracleProblem, order)
    problem.rank == 1 || return gauss_hermite_rule(order)
    offsets = [-8.0, -4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    transitions = problem.alpha == 1.0 ?
        ((problem.theta0, problem.basis[1, 1] * problem.roots[1]),) :
        ((problem.theta0, problem.basis[1, 1] * problem.roots[1]),
         (problem.thetaP, problem.basis[2, 1] * problem.roots[1]))
    points = Float64[]
    for (intercept, coefficient) in transitions
        coefficient == 0.0 && continue
        append!(points, -intercept / coefficient .+ offsets ./ abs(coefficient))
    end
    split_normal_rule(order, points)
end

function scale_moment_rows!(F, J, kappa, sigma, dsigma, isigma, lambda, dlambda, ilambda)
    n = length(F)
    EW, ER2 = F[n - 1], F[n]
    factor = lambda^2 / (kappa * sigma^2)
    for j in 1:n
        dl = j == ilambda ? dlambda : 0.0
        ds = j == isigma ? dsigma : 0.0
        J[n - 1, j] = (dl * EW + lambda * J[n - 1, j]) / kappa
        J[n, j] = factor * (J[n, j] + 2.0 * ER2 * (dl / lambda - ds / sigma))
    end
    F[n - 1] = lambda * EW / kappa - 1.0
    F[n] = factor * ER2 - 1.0
    return nothing
end

"""
    oracle_focs!(F, J, x, problem, signal, noise)

Fill the oracle residual `F` and its analytic Jacobian `J` at the solver
coordinates `x = [theta; a; log(sigma); log(lambda)]`.
"""
function oracle_focs!(F, J, x, problem::OracleProblem, signal::QuadRule,
                      noise::QuadRule)
    r = problem.rank
    n = oracle_dimension(problem)
    fill!(F, 0.0); fill!(J, 0.0)
    theta = x[1]
    sigma, dsigma = log_scale(x[n - 1])
    lambda, dlambda = log_scale(x[n])

    signal_nodes = r == 0 ? [(0.0, 0.0, 1.0)] :
                   r == 1 ? [(z, 0.0, w) for (z, w) in zip(signal.nodes, signal.weights)] :
                   [(z1, z2, w1 * w2)
                    for (z1, w1) in zip(signal.nodes, signal.weights),
                        (z2, w2) in zip(signal.nodes, signal.weights)]

    for (z1, z2, signal_weight) in signal_nodes
        Q1 = Q2 = aligned = 0.0
        if r >= 1
            Q1 += problem.basis[1, 1] * problem.roots[1] * z1
            Q2 += problem.basis[2, 1] * problem.roots[1] * z1
            aligned += x[2] * z1
        end
        if r == 2
            Q1 += problem.basis[1, 2] * problem.roots[2] * z2
            Q2 += problem.basis[2, 2] * problem.roots[2] * z2
            aligned += x[3] * z2
        end

        q0 = logistic(problem.theta0 + Q1)
        if problem.alpha == 1.0
            y0, y1 = 0.0, 1.0
        else
            qP = logistic(problem.thetaP + Q2)
            y0 = (1.0 - problem.alpha) * qP
            y1 = problem.alpha + y0
        end

        for (g, noise_weight) in zip(noise.nodes, noise.weights)
            weight = signal_weight * noise_weight
            xi = theta + aligned + sigma * g
            s0 = prox_state(xi, y0, lambda)
            s1 = prox_state(xi, y1, lambda)

            Rbar = q0 * s1.R + (1.0 - q0) * s0.R
            R2bar = q0 * s1.R^2 + (1.0 - q0) * s0.R^2
            Wbar = q0 * s1.W + (1.0 - q0) * s0.W
            F[1] += weight * Rbar
            r >= 1 && (F[2] += weight * z1 * Rbar)
            r == 2 && (F[3] += weight * z2 * Rbar)
            F[n - 1] += weight * Wbar
            F[n] += weight * R2bar

            for j in 1:n
                dxi = j == 1 ? 1.0 :
                      (r >= 1 && j == 2) ? z1 :
                      (r == 2 && j == 3) ? z2 :
                      j == n - 1 ? dsigma * g : 0.0
                dl = j == n ? dlambda : 0.0
                de0 = (dxi - dl * s0.R) / s0.D
                de1 = (dxi - dl * s1.R) / s1.D
                dR0, dR1 = s0.h * de0, s1.h * de1
                dW0 = s0.k * de0 / s0.D^2 - s0.W^2 * dl
                dW1 = s1.k * de1 / s1.D^2 - s1.W^2 * dl
                dRbar = q0 * dR1 + (1.0 - q0) * dR0
                dR2bar = 2.0 * (q0 * s1.R * dR1 + (1.0 - q0) * s0.R * dR0)
                dWbar = q0 * dW1 + (1.0 - q0) * dW0
                J[1, j] += weight * dRbar
                r >= 1 && (J[2, j] += weight * z1 * dRbar)
                r == 2 && (J[3, j] += weight * z2 * dRbar)
                J[n - 1, j] += weight * dWbar
                J[n, j] += weight * dR2bar
            end
        end
    end

    scale_moment_rows!(F, J, problem.kappa, sigma, dsigma, n - 1, lambda, dlambda, n)
end

"""
    sloe_focs!(F, J, x, problem, rule)

Fill the SLOE residual and Jacobian at `x = [theta0, a, s_r, psi, s_lambda]`,
where `Q2 = delta Z2`, `T = (chiP/delta) Z2 + tau ZT` and
`Q1 = a Z2 + r (rho ZT + s Z1)`. This parameterisation enforces positive
definiteness and gives `sigma = tau s`. The rank-one system drops `a` and the
`Z2` integral.
"""
function sloe_focs!(F, J, x, problem::SLOEProblem, rule::QuadRule)
    sloe_rank_one(problem) && return sloe_focs_rank_one!(F, J, x, problem, rule)
    fill!(F, 0.0); fill!(J, 0.0)
    theta0, a = x[1], x[2]
    r, dr = log_scale(x[3])
    rho, s, drho, ds = partial_correlation(x[4])
    lambda, dlambda = log_scale(x[5])
    sigma, dsigma = problem.tau * s, problem.tau * ds
    nodes, weights = rule.nodes, rule.weights

    for (z2, w2) in zip(nodes, weights)
        qP = logistic(problem.thetaP + problem.delta * z2)
        y0 = (1.0 - problem.alpha) * qP
        y1 = problem.alpha + y0
        for (zt, wt) in zip(nodes, weights)
            xi = problem.theta_fit + (problem.chiP / problem.delta) * z2 +
                 problem.tau * zt
            s0 = prox_state(xi, y0, lambda)
            s1 = prox_state(xi, y1, lambda)

            de0l = -dlambda * s0.R / s0.D
            de1l = -dlambda * s1.R / s1.D
            dR0l, dR1l = s0.h * de0l, s1.h * de1l
            dW0l = s0.k * de0l / s0.D^2 - s0.W^2 * dlambda
            dW1l = s1.k * de1l / s1.D^2 - s1.W^2 * dlambda
            deltaR, deltaR2, deltaW = s1.R - s0.R, s1.R^2 - s0.R^2, s1.W - s0.W

            for (z1, w1) in zip(nodes, weights)
                weight = w2 * wt * w1
                U = rho * zt + s * z1
                dU = drho * zt + ds * z1
                q0, h0, _ = logistic_derivatives(theta0 + a * z2 + r * U)
                Rbar = q0 * s1.R + (1.0 - q0) * s0.R
                R2bar = q0 * s1.R^2 + (1.0 - q0) * s0.R^2
                Wbar = q0 * s1.W + (1.0 - q0) * s0.W
                F[1] += weight * Rbar
                F[2] += weight * z2 * Rbar
                F[3] += weight * U * Rbar
                F[4] += weight * Wbar
                F[5] += weight * R2bar

                dlinear = (1.0, z2, dr * U, r * dU)
                for j in 1:5
                    if j <= 4
                        dq0 = h0 * dlinear[j]
                        dRbar, dR2bar, dWbar = dq0 * deltaR, dq0 * deltaR2, dq0 * deltaW
                    else
                        dRbar = q0 * dR1l + (1.0 - q0) * dR0l
                        dR2bar = 2.0 * (q0 * s1.R * dR1l + (1.0 - q0) * s0.R * dR0l)
                        dWbar = q0 * dW1l + (1.0 - q0) * dW0l
                    end
                    J[1, j] += weight * dRbar
                    J[2, j] += weight * z2 * dRbar
                    J[3, j] += weight * (U * dRbar + (j == 4 ? dU * Rbar : 0.0))
                    J[4, j] += weight * dWbar
                    J[5, j] += weight * dR2bar
                end
            end
        end
    end

    scale_moment_rows!(F, J, problem.kappa, sigma, dsigma, 4, lambda, dlambda, 5)
end

function sloe_focs_rank_one!(F, J, x, problem::SLOEProblem, rule::QuadRule)
    fill!(F, 0.0); fill!(J, 0.0)
    theta0 = x[1]
    gamma, dgamma = log_scale(x[2])
    rho, s, drho, ds = partial_correlation(x[3])
    lambda, dlambda = log_scale(x[4])
    sqrt_nu = sqrt(problem.nu2)
    sigma, dsigma = sqrt_nu * s, sqrt_nu * ds
    qP = logistic(problem.thetaP)
    y0 = (1.0 - problem.alpha) * qP
    y1 = problem.alpha + y0

    for (z1, w1) in zip(rule.nodes, rule.weights),
        (zt, wt) in zip(rule.nodes, rule.weights)

        weight = w1 * wt
        U = rho * z1 + s * zt
        dU = drho * z1 + ds * zt
        xi = problem.theta_fit + sqrt_nu * U
        s0 = prox_state(xi, y0, lambda)
        s1 = prox_state(xi, y1, lambda)

        q0, h0, _ = logistic_derivatives(theta0 + gamma * z1)
        Rbar = q0 * s1.R + (1.0 - q0) * s0.R
        R2bar = q0 * s1.R^2 + (1.0 - q0) * s0.R^2
        Wbar = q0 * s1.W + (1.0 - q0) * s0.W
        F[1] += weight * Rbar
        F[2] += weight * z1 * Rbar
        F[3] += weight * Wbar
        F[4] += weight * R2bar

        deltaR, deltaR2, deltaW = s1.R - s0.R, s1.R^2 - s0.R^2, s1.W - s0.W
        for j in 1:4
            dq0 = h0 * (j == 1 ? 1.0 : j == 2 ? dgamma * z1 : 0.0)
            dxi = j == 3 ? sqrt_nu * dU : 0.0
            dl = j == 4 ? dlambda : 0.0
            de0 = (dxi - dl * s0.R) / s0.D
            de1 = (dxi - dl * s1.R) / s1.D
            dR0, dR1 = s0.h * de0, s1.h * de1
            dW0 = s0.k * de0 / s0.D^2 - s0.W^2 * dl
            dW1 = s1.k * de1 / s1.D^2 - s1.W^2 * dl
            dRbar = dq0 * deltaR + q0 * dR1 + (1.0 - q0) * dR0
            dR2bar = dq0 * deltaR2 + 2.0 * (q0 * s1.R * dR1 + (1.0 - q0) * s0.R * dR0)
            dWbar = dq0 * deltaW + q0 * dW1 + (1.0 - q0) * dW0
            J[1, j] += weight * dRbar
            J[2, j] += weight * z1 * dRbar
            J[3, j] += weight * dWbar
            J[4, j] += weight * dR2bar
        end
    end

    scale_moment_rows!(F, J, problem.kappa, sigma, dsigma, 3, lambda, dlambda, 4)
end

"""Start the oracle solver from the convex combination `alpha beta0 +
(1 - alpha) betaP` of the two signals, with moderate sigma and lambda."""
function default_oracle_start(problem::OracleProblem)
    theta = problem.alpha * problem.theta0 + (1.0 - problem.alpha) * problem.thetaP
    a = problem.rank == 0 ? Float64[] :
        problem.roots .* (problem.basis' * [problem.alpha, 1.0 - problem.alpha])
    sigma = max(sqrt(problem.kappa), 0.25)
    lambda = 4.0 * problem.kappa / (1.0 - problem.kappa)
    vcat(theta, a, inverse_log_scale(sigma), inverse_log_scale(lambda))
end

"""
    default_sloe_starts(problem; nstarts = 7)

Deterministic grid of SLOE starting values. The SLOE system can have several
roots, so the solver is run from a spread of signal/correlation candidates.
"""
function default_sloe_starts(problem::SLOEProblem; nstarts = 7)
    rho_grid = [0.55, 0.0, -0.55, 0.85, -0.85, 0.30, -0.30, 0.95, -0.95]
    scale_grid = [1.0, 1.0, 1.0, 0.75, 1.25, 1.0, 1.0, 0.5, 2.0]
    a_grid = [0.0, 0.0, 0.0, 0.5, -0.5, 1.0, -1.0, 0.0, 0.0] .* problem.delta
    nstarts <= length(rho_grid) ||
        throw(ArgumentError("nstarts may not exceed $(length(rho_grid))"))
    lambda = inverse_log_scale(4.0 * problem.kappa / (1.0 - problem.kappa))
    scale = sqrt(max(0.1, problem.nu2, problem.delta2))
    [sloe_rank_one(problem) ?
     [problem.theta_fit, inverse_log_scale(scale * scale_grid[i]),
      inverse_partial_correlation(rho_grid[i]), lambda] :
     [problem.theta_fit, a_grid[i], inverse_log_scale(scale * scale_grid[i]),
      inverse_partial_correlation(rho_grid[i]), lambda]
     for i in 1:nstarts]
end

function focs_nonlinear_function(focs!, n)
    F, J, xlast = zeros(n), zeros(n, n), fill(NaN, n)
    evaluate!(x) = x == xlast || (copyto!(xlast, x); focs!(F, J, x))
    NonlinearSolve.NonlinearFunction{true}(
        (out, x, _) -> (evaluate!(x); copyto!(out, F));
        jac = (out, x, _) -> (evaluate!(x); copyto!(out, J)),
        jac_prototype = zeros(n, n))
end

function solve_focs(focs!, start, n; least_squares = false, certify!,
                    algorithm, abstol, reltol, maxiters, certification_tolerance)
    problem = (least_squares ? NonlinearSolve.NonlinearLeastSquaresProblem :
                               NonlinearSolve.NonlinearProblem)(
        focs_nonlinear_function(focs!, n), copy(start), nothing)
    solution = NonlinearSolve.solve(problem, algorithm; abstol = abstol,
                                    reltol = reltol, maxiters = maxiters)
    x = Vector{Float64}(solution.u)
    all(isfinite, x) || return x, false, Inf
    F, J = zeros(n), zeros(n, n)
    certify!(F, J, x)
    residual = norm(F, Inf)
    (x, SciMLBase.successful_retcode(solution) && residual <= certification_tolerance,
     residual)
end

function oracle_solution(problem::OracleProblem, x, converged, residual)
    r = problem.rank
    a = r == 0 ? Float64[] : x[2:(r + 1)]
    if problem.alpha == 1.0
        v = r == 0 ? zeros(2) : [a[1] / problem.roots[1], 0.0]
        u = r == 0 ? zeros(2) : symmetric_sqrt(problem.Gamma) * v
        chi = r == 0 ? zeros(2) : problem.Gamma * v
    else
        u = r == 0 ? zeros(2) : problem.basis * a
        v = r == 0 ? zeros(2) : problem.basis * (a ./ problem.roots)
        chi = r == 0 ? zeros(2) : problem.basis * (problem.roots .* a)
    end
    (; problem, theta = x[1], u, v, chi,
       sigma = exp(x[end - 1]), lambda = exp(x[end]),
       internal = x, converged, residual)
end

function symmetric_sqrt(G)
    vals, vecs = eigen(Symmetric(G))
    vecs * Diagonal(sqrt.(max.(vals, 0.0))) * vecs'
end

"""
    solve_oracle(problem; order = 36, certification_order = 48)

Solve the oracle FOCs and return a NamedTuple with the limiting `theta`, `u`,
`v`, `chi`, `sigma` and `lambda`, plus `converged` and the certified
`residual`.
"""
function solve_oracle(problem::OracleProblem;
                      start = default_oracle_start(problem),
                      order = 36,
                      certification_order = 48,
                      algorithm = NonlinearSolve.TrustRegion(),
                      abstol = 1e-10,
                      reltol = 1e-10,
                      maxiters = 150,
                      certification_tolerance = 1e-8)
    n = oracle_dimension(problem)
    signal, noise = oracle_signal_rule(problem, order), gauss_hermite_rule(order)
    cert_signal = oracle_signal_rule(problem, certification_order)
    cert_noise = gauss_hermite_rule(certification_order)
    x, converged, residual = solve_focs(
        (F, J, x) -> oracle_focs!(F, J, x, problem, signal, noise), start, n;
        certify! = (F, J, x) -> oracle_focs!(F, J, x, problem, cert_signal, cert_noise),
        algorithm = algorithm, abstol = abstol, reltol = reltol,
        maxiters = maxiters, certification_tolerance = certification_tolerance)
    oracle_solution(problem, x, converged, residual)
end

function sloe_solution(problem::SLOEProblem, x, converged, residual)
    if sloe_rank_one(problem)
        theta0 = x[1]
        gamma, _ = log_scale(x[2])
        rho, s, _, _ = partial_correlation(x[3])
        lambda, _ = log_scale(x[4])
        sqrt_nu = sqrt(problem.nu2)
        gamma2 = gamma^2
        chi0 = gamma * sqrt_nu * rho
        v = [chi0 / gamma2, 0.0]
        identified = abs(problem.chiP) <= sqrt(eps()) * max(1.0, abs(v[1]), abs(problem.chiP))
        varphi = identified ? 0.0 :
                 (problem.alpha == 1.0 && abs(v[1]) > sqrt(eps()) ?
                  problem.chiP / v[1] : NaN)
        identified = identified || isfinite(varphi)
        sigma = sqrt_nu * s
    else
        theta0, a = x[1], x[2]
        r, _ = log_scale(x[3])
        rho, s, _, _ = partial_correlation(x[4])
        lambda, _ = log_scale(x[5])
        gamma2 = a^2 + r^2
        varphi = problem.delta * a
        chi0 = (problem.chiP / problem.delta) * a + problem.tau * r * rho
        sigma = problem.tau * s
        v1 = problem.tau * rho / r
        v = [v1, problem.chiP / problem.delta2 - (a / problem.delta) * v1]
        identified = true
    end
    Gamma = [gamma2 varphi; varphi problem.delta2]
    (; problem, theta0, gamma2, varphi, lambda, chi0, sigma, v,
       u = any(!isfinite, Gamma) ? fill(NaN, 2) : symmetric_sqrt(Gamma) * v,
       Gamma, internal = x, converged, identified, residual)
end

"""
    solve_sloe(problem; nstarts = 7, order = 64, certification_order = 96)

Solve the SLOE FOCs from a grid of starts and return the accepted root with the
smallest certified residual. If no start is accepted, the best candidate is
returned with `converged = false`.
"""
function solve_sloe(problem::SLOEProblem;
                    starts = default_sloe_starts(problem; nstarts = 7),
                    order = 64,
                    certification_order = 96,
                    algorithm = NonlinearSolve.LevenbergMarquardt(),
                    abstol = 1e-10,
                    reltol = 1e-10,
                    maxiters = 200,
                    certification_tolerance = 1e-8)
    n = sloe_dimension(problem)
    rule, cert_rule = gauss_hermite_rule(order), gauss_hermite_rule(certification_order)
    candidates = []
    for start in starts
        candidate = try
            x, converged, residual = solve_focs(
                (F, J, x) -> sloe_focs!(F, J, x, problem, rule), start, n;
                least_squares = true,
                certify! = (F, J, x) -> sloe_focs!(F, J, x, problem, cert_rule),
                algorithm = algorithm, abstol = abstol, reltol = reltol,
                maxiters = maxiters, certification_tolerance = certification_tolerance)
            sloe_solution(problem, x, converged, residual)
        catch err
            err isa InterruptException && rethrow()
            continue
        end
        push!(candidates, candidate)
        candidate.converged && candidate.identified && break
    end
    isempty(candidates) && error("every SLOE start failed")
    accepted = filter(c -> c.converged && c.identified, candidates)
    isempty(accepted) ? argmin(c -> c.residual, candidates) :
                        argmin(c -> c.residual, accepted)
end
