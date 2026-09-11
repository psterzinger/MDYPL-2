## Philipp Sterzinger 03.09.2026: Code is provided as is, no finished package
## and no guarantees given
##
## Choosing the mDYPL shrinkage parameter alpha. `optimal_alpha` solves the
## limiting FOCs on a grid of alphas, reads one scalar criterion off each
## solution, and refines the bracket around the grid minimum by golden section.

with_alpha(problem::OracleProblem, alpha) =
    OracleProblem(kappa = problem.kappa, alpha = alpha, theta0 = problem.theta0,
                  thetaP = problem.thetaP, Gamma = problem.Gamma)

with_alpha(problem::SLOEProblem, alpha) =
    SLOEProblem(kappa = problem.kappa, alpha = alpha, thetaP = problem.thetaP,
                delta2 = problem.delta2, theta_fit = problem.theta_fit,
                nu2 = problem.nu2, chiP = problem.chiP)

"""Limiting MSE of the rescaled slope, `sigma^2 / v1^2`."""
adjusted_mse(solution) =
    abs(solution.v[1]) > 1e-10 ? solution.sigma^2 / solution.v[1]^2 : Inf

"""Limiting MSE of the unadjusted slope, `sigma^2 + (v - e1)' Gamma (v - e1)`."""
function unadjusted_mse(solution)
    displacement = solution.v .- [1.0, 0.0]
    value = solution.sigma^2 + dot(displacement, solution.problem.Gamma * displacement)
    value >= 0.0 ? value : Inf
end

"""
    prediction_r2(solution)

Squared correlation between the true logit `Q1` and the unadjusted limiting
logit, `chi1^2 / (gamma^2 nu^2)` with `nu^2 = |u|^2 + sigma^2`.
"""
function prediction_r2(solution)
    gamma2 = solution.problem.Gamma[1, 1]
    nu2 = dot(solution.u, solution.u) + solution.sigma^2
    (gamma2 > 0.0 && nu2 > 0.0 && !isempty(solution.chi)) || return NaN
    min(solution.chi[1]^2 / (gamma2 * nu2), 1.0)
end

function alpha_criterion(solution, criterion)
    solution.converged || return Inf
    criterion == :adjusted_mse && return adjusted_mse(solution)
    criterion == :unadjusted_mse && return unadjusted_mse(solution)
    criterion == :prediction && return (r2 = prediction_r2(solution);
                                        isfinite(r2) && r2 > 0.0 ? inv(r2) : Inf)
    throw(ArgumentError("criterion must be :adjusted_mse, :unadjusted_mse or :prediction"))
end

const ALPHA_GRID = vcat(0.05:0.05:0.95, [0.975, 0.99])

"""
    optimal_alpha(problem; criterion = :adjusted_mse, alpha_grid = ALPHA_GRID)

Minimise the alpha criterion over `alpha_grid` and refine the bracket around
the grid minimum by golden section to within `tolerance`. Returns a NamedTuple
with the chosen `alpha`, the `criterion` value there, the corresponding
solution of the limiting FOCs, and the whole `profile`. The search runs at the
cheap quadrature order `search_order`, warm starting each alpha from the
previous solution; only the winner is re-solved at the full order.
"""
function optimal_alpha(problem; criterion = :adjusted_mse, alpha_grid = ALPHA_GRID,
                       tolerance = 1e-4, search_order = 12, search_tolerance = 1e-3,
                       kwargs...)
    warm = default_oracle_start(with_alpha(problem, first(alpha_grid)))
    function search(alpha)
        candidate = with_alpha(problem, alpha)
        start = length(warm) == oracle_dimension(candidate) ? warm :
                default_oracle_start(candidate)
        solution = solve_oracle(candidate; order = search_order, start = start,
                                certification_order = search_order + 12,
                                certification_tolerance = search_tolerance, kwargs...)
        solution.converged && (warm = solution.internal)
        solution
    end
    value_at(alpha) = alpha_criterion(search(alpha), criterion)

    profile = [(alpha = alpha, value = value_at(alpha)) for alpha in alpha_grid]
    values = [entry.value for entry in profile]
    all(isinf, values) &&
        error("no alpha on the grid gave an accepted solution of the limiting FOCs")

    best = argmin(values)
    lo = alpha_grid[max(best - 1, 1)]
    hi = alpha_grid[min(best + 1, length(alpha_grid))]
    invphi = (sqrt(5.0) - 1.0) / 2.0
    c, d = hi - invphi * (hi - lo), lo + invphi * (hi - lo)
    fc, fd = value_at(c), value_at(d)
    while (hi - lo) > tolerance
        if fc <= fd
            hi, d, fd = d, c, fc
            c = hi - invphi * (hi - lo)
            fc = value_at(c)
        else
            lo, c, fc = c, d, fd
            d = lo + invphi * (hi - lo)
            fd = value_at(d)
        end
    end

    alpha = fc <= fd ? c : d
    solution = solve_oracle(with_alpha(problem, alpha); start = warm, kwargs...)
    (; alpha, criterion, value = alpha_criterion(solution, criterion),
       solution, profile)
end