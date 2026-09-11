## Philipp Sterzinger 03.09.2026: Code is provided as is, no finished package
## and no guarantees given
##
## Finite-sample mDYPL fit: replace the response by the pseudo-response
## alpha * y + (1 - alpha) * logistic(thetaP + X * betaP) and maximise the
## ordinary logistic likelihood. `fit_mdypl` does both steps.

function add_offset!(eta, offset)
    isnothing(offset) || (eta .+= offset)
    return eta
end

function logistic_nll(beta, ytilde, X, eta, buff, offset)
    mul!(eta, X, beta)
    add_offset!(eta, offset)
    buff .= softplus.(eta) .- ytilde .* eta
    sum(buff)
end

function logistic_nll_grad!(g, beta, ytilde, X, eta, buff, offset)
    mul!(eta, X, beta)
    add_offset!(eta, offset)
    buff .= logistic.(eta) .- ytilde
    mul!(g, X', buff)
end

"""
    mdypl_pseudo_response(y, X, alpha; thetaP = 0.0, betaP = missing, offset = nothing)

Form `alpha * y + (1 - alpha) * logistic(thetaP + offset + X * betaP)`. At
`alpha == 1` the prior is inert and `y` is returned unchanged.
"""
function mdypl_pseudo_response(y, X, alpha; thetaP = 0.0, betaP = missing,
                               offset = nothing)
    alpha == 1.0 && return Float64.(y)
    etaP = ismissing(betaP) ? zeros(length(y)) : X * betaP
    etaP .+= thetaP
    add_offset!(etaP, offset)
    alpha .* y .+ (1.0 - alpha) .* logistic.(etaP)
end

"""
    fit_logistic(ytilde, X; offset = nothing, beta_init = missing, kwargs...)

Fit a logistic regression with the fractional response `ytilde` and return the
`Optim.optimize` result. `X` is the full design, so include an intercept column
if one is wanted. `kwargs...` are passed on to `Optim.optimize`.
"""
function fit_logistic(ytilde, X; offset = nothing, beta_init = missing,
                      method = Optim.LBFGS(), kwargs...)
    n, p = size(X)
    eta = Vector{Float64}(undef, n)
    buff = similar(eta)

    f(beta) = logistic_nll(beta, ytilde, X, eta, buff, offset)
    g!(g, beta) = logistic_nll_grad!(g, beta, ytilde, X, eta, buff, offset)

    ismissing(beta_init) && (beta_init = zeros(p))
    Optim.optimize(f, g!, beta_init, method; kwargs...)
end

"""
    fit_mdypl(y, X, alpha; thetaP = 0.0, betaP = missing, kwargs...)

Form the mDYPL pseudo-response and fit it. Returns the `Optim.optimize` result,
so the estimate is `Optim.minimizer(fit)` and the fitted negative
log-likelihood is `Optim.minimum(fit)`.
"""
function fit_mdypl(y, X, alpha; thetaP = 0.0, betaP = missing, offset = nothing,
                   kwargs...)
    ytilde = mdypl_pseudo_response(y, X, alpha; thetaP = thetaP, betaP = betaP,
                                   offset = offset)
    fit_logistic(ytilde, X; offset = offset, kwargs...)
end

"""
    estimate_nu2_sloe(ytilde, X, beta_hat; offset = nothing)

SLOE estimate of the fitted-score variance `nu^2`, from the centred second
moment of the approximate leave-one-out linear predictors. `X` is the full
design, including any intercept column.
"""
function estimate_nu2_sloe(ytilde, X, beta_hat; offset = nothing)
    n = size(X, 1)
    eta = X * beta_hat
    add_offset!(eta, offset)
    mu = logistic.(eta)
    w = mu .* (1.0 .- mu)

    R = qr(sqrt.(w) .* X).R
    h = vec(sum(abs2, (sqrt.(w) .* X) / UpperTriangular(R); dims = 2))
    maximum(h) < 1.0 || return NaN

    scores = eta .- (h ./ (1.0 .- h)) .* ((ytilde .- mu) ./ w)
    scores .-= mean(scores)
    dot(scores, scores) / n
end

"""
    sloe_problem_from_fit(theta_hat, beta_hat, betaP, kappa; alpha, thetaP, nu2)

Assemble the SLOE plug-in system from a fitted intercept and slope, the known
prior slope and an estimate of `nu^2`.
"""
function sloe_problem_from_fit(theta_hat, beta_hat, betaP, kappa; alpha, thetaP, nu2)
    p = length(beta_hat)
    SLOEProblem(kappa = kappa, alpha = alpha, thetaP = thetaP,
                delta2 = dot(betaP, betaP) / p, theta_fit = theta_hat,
                nu2 = nu2, chiP = dot(betaP, beta_hat) / p)
end