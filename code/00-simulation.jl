## Philipp Sterzinger 03.09.2026: Code is provided as is, no finished package
## and no guarantees given
##
## Monte Carlo simulation for the five settings of the manuscript. `replicate`
## runs one replication through every setting and tuning stream; each stream is
## written to its own Arrow file in `results/`.

using Distributed, LinearAlgebra, Printf

num_workers = 10
BLAS.set_num_threads(1)
addprocs(num_workers; topology = :master_worker, enable_threaded_blas = false)

@everywhere begin
    using LinearAlgebra, Random, Statistics, StableRNGs, Optim
    BLAS.set_num_threads(1)
    supp_path = abspath(joinpath(@__DIR__, ".."))
    results_path = joinpath(supp_path, "results")
end
@everywhere include(joinpath(supp_path, "code", "methods", "MDYPL.jl"))
@everywhere using .MDYPL

using DataFrames, Arrow

@everywhere begin
    n_values = [125, 250, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
    kappa = 0.2
    reps = 10000
    gamma = 1.5
    pi0 = 0.2
    null_block = 20
    plr_block = 10
    unknowns_stride = 10

    base_seed = 0x6a09e667f3bcc909
    signal_seed = 0xbb67ae8584caa73b

    settings = (
        A = (label = "aligned and calibrated",           delta = gamma, corr =  0.8, piP = 0.2),
        O = (label = "orthogonal and calibrated",        delta = gamma, corr =  0.0, piP = 0.2),
        I = (label = "aligned with incorrect intercept", delta = gamma, corr =  0.8, piP = 0.8),
        M = (label = "misleading",                       delta = gamma, corr = -0.8, piP = 0.8),
        N = (label = "null",                             delta = 0.0,   corr =  0.0, piP = 0.5),
    )
    tunings = (adjusted = :adjusted_mse, unadjusted = :unadjusted_mse,
               prediction = :prediction)

    gamma_sweep = [0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 50.0]
end

"""
    nu2_bound(kappa, alpha, thetaP, delta, corr, prevalence; slack = 1.5)

Largest limiting `nu2 = v' Gamma v + sigma^2` the forward map produces at this
setting's specs, swept over the signal strength. A SLOE plug-in above it could
not have come from any true parameter value.
"""
function nu2_bound(kappa, alpha, thetaP, delta, corr, prevalence; slack = 1.5)
    bound = 0.0
    for gamma in gamma_sweep
        varphi = corr * gamma * delta
        Gamma = [gamma^2 varphi; varphi delta^2]
        solution = solve_oracle(
            OracleProblem(kappa = kappa, alpha = alpha, thetaP = thetaP,
                          theta0 = calibrated_intercept(gamma, prevalence),
                          Gamma = Gamma);
            order = 12, certification_order = 18,
            certification_tolerance = 2e-2, maxiters = 80)
        solution.converged || continue
        bound = max(bound, dot(solution.v, Gamma * solution.v) + solution.sigma^2)
    end
    slack * bound
end

@everywhere function sloe_estimate(beta_hat, theta_hat, nu2, population, kappa_n)
    failed = (theta0 = NaN, gamma = NaN, varphi = NaN, successful = false)
    isfinite(nu2) || return failed
    p = length(beta_hat)
    delta2 = dot(population.betaP, population.betaP) / p
    chiP = dot(population.betaP, beta_hat) / p
    sloe_covariance_status(delta2, nu2, chiP) || return failed
    nu2 <= population.nu2_max || return failed
    solution = try
        solve_sloe(sloe_problem_from_fit(theta_hat, beta_hat, population.betaP, kappa_n;
                                         alpha = population.alpha,
                                         thetaP = population.thetaP, nu2 = nu2))
    catch
        return failed
    end
    (theta0 = solution.theta0, gamma = sqrt(max(solution.gamma2, 0.0)),
     varphi = solution.varphi, successful = solution.converged && solution.identified)
end

@everywhere function unknown_row(rep, setup, method, estimate, population)
    state = population.state
    row = (n = setup.n, replicate = rep, setting = String(population.code),
           method = method,
           theta0_true = setup.theta0, gamma_true = gamma,
           varphi_true = population.Gamma[1, 2],
           true_theta_star = state.theta, true_v1_star = state.v[1],
           true_v2_star = state.v[2], true_sigma_star = state.sigma,
           true_lambda_star = state.lambda,
           theta0_hat = estimate.theta0, gamma_hat = estimate.gamma,
           varphi_hat = estimate.varphi, unknown_success = estimate.successful)
    absent = (foc_theta_star = NaN, foc_v1_star = NaN, foc_v2_star = NaN,
              foc_sigma_star = NaN, foc_lambda_star = NaN, foc_success = false)

    estimate.successful || return merge(row, absent)
    solution = try
        solve_estimated_focs((theta0 = estimate.theta0,
                              Gamma = [estimate.gamma^2 estimate.varphi;
                                       estimate.varphi population.delta^2]);
                             kappa = setup.p / setup.n, alpha = population.alpha,
                             thetaP = population.thetaP,
                             order = 16, certification_order = 28,
                             certification_tolerance = 1e-4)
    catch
        return merge(row, absent)
    end
    merge(row, (foc_theta_star = solution.theta, foc_v1_star = solution.v[1],
                foc_v2_star = solution.v[2], foc_sigma_star = solution.sigma,
                foc_lambda_star = solution.lambda, foc_success = solution.converged))
end

@everywhere replication_seed(n, rep) = base_seed + UInt64(n) + UInt64(rep)

@everywhere function replicate(rep, setup)
    n, p = setup.n, setup.p
    rng = StableRNG(replication_seed(n, rep))
    H = randn(rng, n, p)
    A = hcat(ones(n), H)
    A_restricted = hcat(ones(n), H[:, (plr_block + 1):end])
    y = Float64.(rand(rng, n) .< logistic.(setup.theta0 .+ H * (setup.beta0 ./ sqrt(p))))

    rows = NamedTuple[]
    unknowns = NamedTuple[]
    for population in setup.populations
        centre = population.plr_centre ./ sqrt(p)
        offset = H[:, 1:plr_block] * centre

        start = vcat(population.thetaP, population.betaP ./ sqrt(p))
        start[2:(plr_block + 1)] .-= centre

        ytilde = mdypl_pseudo_response(y, A, population.alpha;
                                       betaP = start, offset = offset)
        full = fit_logistic(ytilde, A; offset = offset, beta_init = copy(start))
        coefficients = Optim.minimizer(full)
        theta_hat = coefficients[1]
        beta_hat = sqrt(p) .* coefficients[2:end]
        beta_hat[1:plr_block] .+= population.plr_centre

        plr = NaN
        nu2 = NaN
        if population.tuning == :adjusted
            restricted = fit_logistic(ytilde, A_restricted; offset = offset,
                beta_init = vcat(coefficients[1], coefficients[(plr_block + 2):end]))
            raw = 2.0 * (Optim.minimum(restricted) - Optim.minimum(full))
            plr = raw >= 0.0 ?
                  population.state.lambda * raw / population.state.sigma^2 : NaN
        elseif population.tuning == :unadjusted && rep % unknowns_stride == 0
            nu2 = estimate_nu2_sloe(ytilde, A, coefficients; offset = offset)
            push!(unknowns,
                  unknown_row(rep, setup, "O",
                              estimate_unknowns(y, H, population.betaP;
                                                design_scale = 1 / sqrt(p)),
                              population),
                  unknown_row(rep, setup, "SLOE",
                              sloe_estimate(beta_hat, theta_hat, nu2, population, p / n),
                              population))
        end

        push!(rows, (setting = String(population.code),
                     tuning = String(population.tuning),
                     replicate = rep,
                     seed = replication_seed(n, rep),
                     converged = Optim.converged(full),
                     theta_hat = theta_hat,
                     plr = plr,
                     sloe_nu2 = nu2,
                     beta = beta_hat))
    end
    rows, unknowns
end

mkpath(results_path)

theta0 = calibrated_intercept(gamma, pi0)

oracle = Dict{Tuple{Symbol,Symbol},Any}()
for (code, setting) in pairs(settings)
    thetaP = calibrated_intercept(setting.delta, setting.piP)
    varphi = setting.corr * gamma * setting.delta
    problem = OracleProblem(kappa = kappa, alpha = 0.5, theta0 = theta0,
                            thetaP = thetaP,
                            Gamma = [gamma^2 varphi; varphi setting.delta^2])
    for (tuning, criterion) in pairs(tunings)
        result = optimal_alpha(problem; criterion = criterion)
        state = result.solution
        state.converged ||
            error("the limiting FOCs did not converge for setting $code, tuning $tuning")
        bound = tuning == :unadjusted ?
                nu2_bound(kappa, result.alpha, thetaP, setting.delta,
                          setting.corr, pi0) : Inf
        @printf("%s %-10s alpha = %.6f, criterion = %.6f, residual = %.2e\n",
                code, tuning, result.alpha, result.value, state.residual)
        oracle[(code, tuning)] = (
            code = code, label = setting.label, tuning = tuning,
            delta = setting.delta, thetaP = thetaP, Gamma = problem.Gamma,
            alpha = result.alpha, state = state, nu2_max = bound)
    end
end

for n in n_values
    p = round(Int, kappa * n)
    p / n == kappa ||
        error("kappa * n must be an integer, so that every sample size shares one oracle solve")

    rng = StableRNG(signal_seed)
    u1 = zeros(p)
    u1[(null_block + 1):end] .= normalize(randn(rng, p - null_block))
    g = randn(rng, p)
    u2 = normalize(g .- dot(g, u1) .* u1)
    beta0 = sqrt(p) * gamma .* u1
    @printf("n = %d, p = %d, kappa = %.4f, theta0 = %.8f\n", n, p, p / n, theta0)

    populations = Dict{Tuple{Symbol,Symbol},Any}()
    for (code, setting) in pairs(settings)
        betaP = setting.delta == 0.0 ? zeros(p) :
                sqrt(p) * setting.delta .*
                (setting.corr .* u1 .+ sqrt(1.0 - setting.corr^2) .* u2)
        for tuning in keys(tunings)
            q = oracle[(code, tuning)]
            populations[(code, tuning)] = merge(q,
                (betaP = betaP,
                 plr_centre = q.state.v[1] .* beta0[1:plr_block] .+
                              q.state.v[2] .* betaP[1:plr_block]))
        end
    end

    setup = (; n, p, theta0, beta0,
             populations = [populations[(code, tuning)]
                            for code in keys(settings) for tuning in keys(tunings)])

    @printf("  running %d replications on %d workers\n", reps, num_workers)
    results = pmap(rep -> replicate(rep, setup), 1:reps)

    for population in setup.populations
        rows = [row for (fits, _) in results for row in fits
                if row.setting == String(population.code) &&
                   row.tuning == String(population.tuning)]
        state = population.state
        df = DataFrame(rows)
        betas = df.beta
        select!(df, Not(:beta))
        insertcols!(df, :setting_label => population.label, :n => n, :p => p,
                    :theta0 => theta0, :alpha_opt => population.alpha,
                    :theta_star => state.theta, :v1_star => state.v[1],
                    :v2_star => state.v[2], :sigma_star => state.sigma,
                    :lambda_star => state.lambda)
        for j in 1:p
            df[!, Symbol(@sprintf("beta_%03d", j))] = [b[j] for b in betas]
        end
        stem = population.tuning == :adjusted ? "setting_$(population.code)" :
               "setting_$(population.code)_$(population.tuning)"
        Arrow.write(joinpath(results_path, @sprintf("%s_n%04d.arrow", stem, n)), df)
    end

    settings_table = map(collect(keys(settings))) do code
        adjusted = populations[(code, :adjusted)]
        row = (setting = String(code), label = settings[code].label, n = n, p = p,
               kappa = p / n, plr_block = plr_block, gamma = gamma, gamma2 = gamma^2,
               delta = settings[code].delta, delta2 = settings[code].delta^2,
               varphi = adjusted.Gamma[1, 2], prior_correlation = settings[code].corr,
               pi0 = pi0, piP = settings[code].piP, theta0 = theta0,
               thetaP = adjusted.thetaP, Gamma11 = adjusted.Gamma[1, 1],
               Gamma12 = adjusted.Gamma[1, 2], Gamma22 = adjusted.Gamma[2, 2])
        for (tuning, prefix) in ((:adjusted, ""), (:unadjusted, "unadjusted_"),
                                 (:prediction, "prediction_"))
            q, s = populations[(code, tuning)], populations[(code, tuning)].state
            fields = Symbol.(prefix .* ["alpha_opt", "theta_star", "v1_star", "v2_star",
                                        "chi1_star", "chi2_star", "sigma_star",
                                        "lambda_star"])
            row = merge(row, NamedTuple{Tuple(fields)}(
                (q.alpha, s.theta, s.v[1], s.v[2], s.chi[1], s.chi[2], s.sigma, s.lambda)))
        end
        row
    end
    Arrow.write(joinpath(results_path, @sprintf("settings_n%04d.arrow", n)),
                DataFrame(settings_table))

    vectors = vcat([("beta0", "", beta0), ("u1", "", u1), ("u2", "", u2)],
                   [(name, String(code), getfield(populations[(code, :adjusted)], field))
                    for (name, field) in (("betaP", :betaP), ("plr_centre", :plr_centre))
                    for code in keys(settings)])
    Arrow.write(joinpath(results_path, @sprintf("signals_n%04d.arrow", n)),
                DataFrame([(signal = signal, setting = setting, coordinate = j,
                            value = values[j])
                           for (signal, setting, values) in vectors
                           for j in eachindex(values)]))

    Arrow.write(joinpath(results_path, @sprintf("unknowns_n%04d.arrow", n)),
                DataFrame(reduce(vcat, (rows for (_, rows) in results);
                                 init = NamedTuple[])))

    @printf("  done: %d unconverged fits\n",
            count(!row.converged for (fits, _) in results for row in fits))
end
