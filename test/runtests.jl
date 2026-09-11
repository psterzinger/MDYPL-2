## Philipp Sterzinger 04.09.2026: Code is provided as is, no finished package
## and no guarantees given
##
## Unit tests for the methods in code/methods/MDYPL.jl: the finite-sample fit,
## the oracle first-order conditions and the SLOE system are checked against the
## reference implementations in R's brglm2 through RCall, and the Jacobians and
## the proximal map are checked against numerical derivatives and their defining
## equations.
##
## Run from the supplementary directory with
##
##     julia test/runtests.jl

using Test, LinearAlgebra, Random, Statistics, StableRNGs, Optim, RCall

supp_path = abspath(joinpath(@__DIR__, ".."))
include(joinpath(supp_path, "code", "methods", "MDYPL.jl"))
using .MDYPL

R"""
suppressPackageStartupMessages(library(brglm2))

quadrature <- statmod::gauss.quad(200, kind = "hermite")

mdypl_object <- function(Z, y, alpha, intercept) {
    glm(if (intercept) y ~ Z else y ~ Z - 1, family = binomial(), method = "mdyplFit",
        control = mdyplControl(alpha = alpha, epsilon = 1e-12, maxit = 500))
}

mdypl_coefficients <- function(Z, y, alpha, intercept) {
    unname(coef(mdypl_object(Z, y, alpha, intercept)))
}

mdypl_sloe <- function(Z, y, alpha, intercept) {
    sloe(mdypl_object(Z, y, alpha, intercept))
}

state_evolution <- function(kappa, ss, alpha, intercept, corrupted, start) {
    solution <- solve_se(kappa, ss, alpha, intercept, start = start, corrupted = corrupted,
                         gh = quadrature, init_iter = 0)
    c(as.numeric(solution), max(abs(attr(solution, "funcs"))))
}
"""

function r_mdypl_coefficients(Z, y, alpha, intercept)
    @rput Z y alpha intercept
    rcopy(R"mdypl_coefficients(Z, y, alpha, intercept)")
end

function r_mdypl_sloe(Z, y, alpha, intercept)
    @rput Z y alpha intercept
    rcopy(R"mdypl_sloe(Z, y, alpha, intercept)")
end

function r_state_evolution(kappa, ss, alpha, intercept, corrupted, start)
    @rput kappa ss alpha corrupted start
    if isnothing(intercept)
        R"intercept <- NULL"
    else
        @rput intercept
    end
    values = rcopy(R"state_evolution(kappa, ss, alpha, intercept, corrupted, start)")
    isnothing(intercept) ?
        (mu = values[1], b = values[2], sigma = values[3], residual = values[4]) :
        (mu = values[1], b = values[2], sigma = values[3], intercept = values[4],
         residual = values[5])
end

recovered_signal(kappa, ss, reference) =
    sqrt(max(ss^2 - kappa * reference.sigma^2, 0.0)) / reference.mu

function simulated_fit(seed, n, p, gamma, theta0)
    rng = StableRNG(seed)
    Z = randn(rng, n, p) ./ sqrt(p)
    beta = sqrt(p) * gamma .* normalize(randn(rng, p))
    y = Float64.(rand(rng, n) .< MDYPL.logistic.(theta0 .+ Z * beta))
    (Z = Z, y = y)
end

fit_settings = (simulated_fit(20260904, 1000, 200, 1.5, -0.5),
                simulated_fit(20260905, 600, 60, 2.0, 0.7))

limiting_kappa = 0.2
limiting_gamma2 = 5.0
limiting_alphas = (1.0, 1 / (1 + limiting_kappa))
limiting_intercepts = (-0.5, 0.0)

function limiting_oracle(alpha, theta0)
    Gamma = [limiting_gamma2 0.0; 0.0 0.0]
    solution = solve_oracle(OracleProblem(kappa = limiting_kappa, alpha = alpha,
                                          theta0 = theta0, thetaP = 0.0, Gamma = Gamma))
    (solution = solution, nu2 = dot(solution.v, Gamma * solution.v) + solution.sigma^2)
end

oracle_start(solution, intercept) =
    vcat(solution.v[1], solution.lambda, solution.sigma / sqrt(limiting_kappa),
         isnothing(intercept) ? Float64[] : intercept)

function sloe_start(oracle, theta0)
    s = oracle.solution.sigma / sqrt(oracle.nu2)
    [theta0, log(sqrt(limiting_gamma2)),
     MDYPL.inverse_partial_correlation(sqrt(max(1.0 - s^2, 0.0))),
     log(oracle.solution.lambda)]
end

function numerical_jacobian(focs!, x, n)
    F, J = zeros(n), zeros(n, n)
    numerical = zeros(n, n)
    for j in 1:n
        h = 1e-6 * max(1.0, abs(x[j]))
        forward, backward = copy(x), copy(x)
        forward[j] += h
        backward[j] -= h
        Fp, Fm = zeros(n), zeros(n)
        focs!(Fp, J, forward)
        focs!(Fm, J, backward)
        numerical[:, j] = (Fp .- Fm) ./ (2h)
    end
    focs!(F, J, x)
    (analytic = copy(J), numerical = numerical)
end

@testset "mDYPL fit against brglm2" begin
    for (index, setting) in pairs(fit_settings), intercept in (true, false),
        alpha in (1.0, 0.9, 0.75, 0.5, 0.25, 0.1)

        @testset "setting $index, $(intercept ? "with" : "without") intercept, alpha = $alpha" begin
            X = intercept ? hcat(ones(length(setting.y)), setting.Z) : setting.Z
            fit = fit_mdypl(setting.y, X, alpha)
            @test Optim.converged(fit)
            @test Optim.minimizer(fit) ≈
                  r_mdypl_coefficients(setting.Z, setting.y, alpha, intercept) atol = 1e-6
        end
    end
end

@testset "oracle first-order conditions against brglm2" begin
    for alpha in limiting_alphas, theta0 in limiting_intercepts
        oracle = limiting_oracle(alpha, theta0)
        solution = oracle.solution
        intercept = iszero(theta0) ? nothing : theta0
        reference = r_state_evolution(limiting_kappa, sqrt(limiting_gamma2), alpha,
                                      intercept, false,
                                      oracle_start(solution, isnothing(intercept) ?
                                                   nothing : solution.theta))

        @testset "alpha = $alpha, $(isnothing(intercept) ? "without" : "with") intercept" begin
            @test solution.converged
            @test solution.residual < 1e-8
            @test reference.residual < 1e-8
            @test solution.v[1] ≈ reference.mu atol = 1e-6
            @test solution.lambda ≈ reference.b atol = 1e-6
            @test solution.sigma ≈ sqrt(limiting_kappa) * reference.sigma atol = 1e-6
            @test solution.theta ≈
                  (isnothing(intercept) ? 0.0 : reference.intercept) atol = 1e-6
        end
    end
end

@testset "SLOE against brglm2" begin
    @testset "corrupted signal strength" begin
        for (index, setting) in pairs(fit_settings), intercept in (true, false),
            alpha in (0.9, 0.5, 0.25)

            X = intercept ? hcat(ones(length(setting.y)), setting.Z) : setting.Z
            n = length(setting.y)
            beta_hat = Optim.minimizer(fit_mdypl(setting.y, X, alpha))
            ytilde = mdypl_pseudo_response(setting.y, X, alpha)
            nu2 = estimate_nu2_sloe(ytilde, X, beta_hat)
            reference = r_mdypl_sloe(setting.Z, setting.y, alpha, intercept)

            @testset "setting $index, $(intercept ? "with" : "without") intercept, alpha = $alpha" begin
                @test isfinite(nu2)
                @test sqrt(nu2 * n / (n - 1)) ≈ reference atol = 1e-8
            end
        end
    end

    @testset "SLOE system" begin
        for alpha in limiting_alphas, theta0 in limiting_intercepts
            oracle = limiting_oracle(alpha, theta0)
            nu = sqrt(oracle.nu2)
            intercept = iszero(theta0) ? nothing : oracle.solution.theta
            solution = solve_sloe(SLOEProblem(kappa = limiting_kappa, alpha = alpha,
                                              thetaP = 0.0, delta2 = 0.0,
                                              theta_fit = oracle.solution.theta,
                                              nu2 = oracle.nu2, chiP = 0.0);
                                  starts = [sloe_start(oracle, theta0)])
            reference = r_state_evolution(limiting_kappa, nu, alpha, intercept, true,
                                          oracle_start(oracle.solution,
                                                       isnothing(intercept) ? nothing :
                                                       theta0))

            @testset "alpha = $alpha, $(isnothing(intercept) ? "without" : "with") intercept" begin
                @test solution.converged
                @test solution.identified
                @test solution.residual < 1e-8
                @test reference.residual < 1e-8
                @test sqrt(solution.gamma2) ≈ sqrt(limiting_gamma2) atol = 1e-6
                @test solution.theta0 ≈ theta0 atol = 1e-6
                @test recovered_signal(limiting_kappa, nu, reference) ≈
                      sqrt(limiting_gamma2) atol = 1e-6
                isnothing(intercept) || @test reference.intercept ≈ theta0 atol = 1e-6
            end
        end
    end

    @testset "SLOE recovers the oracle" begin
        Gamma = [2.25 1.2; 1.2 2.25]
        oracle = solve_oracle(OracleProblem(kappa = 0.2, alpha = 0.5, theta0 = -1.0,
                                            thetaP = -0.5, Gamma = Gamma))
        nu2 = dot(oracle.v, Gamma * oracle.v) + oracle.sigma^2
        sloe = solve_sloe(SLOEProblem(kappa = 0.2, alpha = 0.5, thetaP = -0.5,
                                      delta2 = Gamma[2, 2], theta_fit = oracle.theta,
                                      nu2 = nu2, chiP = oracle.chi[2]))
        @test sloe.converged
        @test sloe.identified
        @test sloe.theta0 ≈ -1.0 atol = 1e-3
        @test sloe.gamma2 ≈ Gamma[1, 1] atol = 1e-2
        @test sloe.varphi ≈ Gamma[1, 2] atol = 1e-2
    end
end

@testset "internal derivative and proximal map checks" begin
    @testset "logistic link and proximal map" begin
        for x in (-30.0, -1.0, 0.0, 1.0, 30.0)
            @test MDYPL.logistic(x) ≈ 1 / (1 + exp(-x)) atol = 1e-12
            @test MDYPL.softplus(x) ≈ log1p(exp(x)) atol = 1e-9
        end

        for x in (-4.0, 0.0, 2.5), y in (0.0, 0.3, 1.0), lambda in (0.0, 0.5, 3.0)
            eta = prox_logistic(x, y, lambda)
            @test eta + lambda * MDYPL.logistic(eta) ≈ x + lambda * y atol = 1e-9
            @test x + lambda * (y - 1) - 1e-12 <= eta <= x + lambda * y + 1e-12
        end
    end

    @testset "oracle Jacobian" begin
        noise = gauss_hermite_rule(12)
        for Gamma in ([2.25 1.2; 1.2 2.25], [2.25 0.0; 0.0 0.0])
            problem = OracleProblem(kappa = 0.2, alpha = 0.5, theta0 = -1.0,
                                    thetaP = -0.5, Gamma = Gamma)
            signal = MDYPL.oracle_signal_rule(problem, 12)
            focs!(F, J, x) = oracle_focs!(F, J, x, problem, signal, noise)
            check = numerical_jacobian(focs!, default_oracle_start(problem),
                                       oracle_dimension(problem))
            @test check.analytic ≈ check.numerical rtol = 1e-4
        end
    end

    @testset "SLOE Jacobian" begin
        rule = gauss_hermite_rule(10)
        for problem in (SLOEProblem(kappa = 0.2, alpha = 0.5, thetaP = -0.5, delta2 = 2.25,
                                    theta_fit = -0.8, nu2 = 1.4, chiP = 0.6),
                        SLOEProblem(kappa = 0.2, alpha = 1.0, thetaP = -0.5, delta2 = 2.25,
                                    theta_fit = -0.8, nu2 = 1.4, chiP = 0.6),
                        SLOEProblem(kappa = 0.2, alpha = 0.5, thetaP = 0.0, delta2 = 0.0,
                                    theta_fit = -0.8, nu2 = 1.4, chiP = 0.0))
            focs!(F, J, x) = sloe_focs!(F, J, x, problem, rule)
            check = numerical_jacobian(focs!, first(default_sloe_starts(problem)),
                                       sloe_dimension(problem))
            @test check.analytic ≈ check.numerical rtol = 1e-4
        end
    end
end
