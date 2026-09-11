## Philipp Sterzinger 03.09.2026: Code is provided as is, no finished package
## and no guarantees given
##
## Estimating the unknown true response parameters from the response moments.
## `estimate_unknowns` inverts the prevalence and the diagonal-deleted second
## moment of X'y for theta0 and gamma, and the prior/response covariance for
## varphi.

logit(p) = log(p) - log1p(-p)

function channel_moments(theta, gamma, rule::QuadRule)
    prevalence = c = 0.0
    for (z, w) in zip(rule.nodes, rule.weights)
        mu = logistic(theta + gamma * z)
        prevalence += w * mu
        c += w * mu * (1.0 - mu)
    end
    (prevalence = prevalence, c = c, zeta = gamma * c)
end

function channel_intercept(prevalence, gamma, rule::QuadRule; tolerance = 1e-12)
    gamma == 0.0 && return logit(prevalence)
    centre = logit(prevalence)
    radius = max(12.0 + 8.0 * gamma, abs(centre) + 8.0 + 4.0 * gamma)
    lo, hi = centre - radius, centre + radius
    for _ in 1:200
        mid = (lo + hi) / 2
        f = channel_moments(mid, gamma, rule).prevalence - prevalence
        (abs(f) <= tolerance || hi - lo <= tolerance * (1.0 + abs(mid))) && return mid
        f > 0.0 ? (hi = mid) : (lo = mid)
    end
    (lo + hi) / 2
end

function invert_channel(prevalence, zeta2, rule::QuadRule;
                        gamma_upper = 200.0, tolerance = 1e-10)
    target = sqrt(max(zeta2, 0.0))
    at(gamma) = (theta = channel_intercept(prevalence, gamma, rule);
                 merge((theta = theta, gamma = gamma),
                       channel_moments(theta, gamma, rule)))
    target <= tolerance && return merge(at(0.0), (successful = true,))

    lo, hi = 0.0, 1.0
    while at(hi).zeta < target && hi < gamma_upper
        hi = min(2.0 * hi, gamma_upper)
    end
    at(hi).zeta < target && return merge(at(hi), (successful = false,))

    best = at(hi)
    for _ in 1:140
        mid = (lo + hi) / 2
        best = at(mid)
        (abs(best.zeta - target) <= tolerance * max(1.0, target) ||
         hi - lo <= tolerance * (1.0 + mid)) && break
        best.zeta > target ? (hi = mid) : (lo = mid)
    end
    merge(best, (successful = true,))
end

"""
    estimate_unknowns(y, X, betaP; design_scale = 1.0)

Estimate `(theta0, gamma, varphi)` from the response moments, and return them
with the estimated `Gamma` and a `successful` flag. The design is interpreted as
`design_scale * X`, which lets simulations pass an unscaled Gaussian matrix `H`
with `design_scale = 1/sqrt(p)` instead of allocating `H / sqrt(p)`.
"""
function estimate_unknowns(y, X, betaP; design_scale = 1.0, quadrature_order = 80,
                           gamma_upper = 200.0, tolerance = 1e-10)
    n, p = size(X)
    Xty = design_scale .* (X' * y)
    row_norm2 = design_scale^2 .* vec(sum(abs2, X; dims = 2))
    zeta2 = p * (dot(Xty, Xty) - dot(y, row_norm2)) / (n * (n - 1))

    delta = sqrt(dot(betaP, betaP) / p)
    prior_covariance = 0.0
    if delta > 0.0
        prior_score = design_scale .* (X * betaP)
        prior_covariance = (dot(prior_score, y) - sum(prior_score) * sum(y) / n) / (n - 1)
    end

    rule = gauss_hermite_rule(quadrature_order)
    channel = invert_channel(clamp(mean(y), 1e-10, 1 - 1e-10), zeta2, rule;
                             gamma_upper = gamma_upper, tolerance = tolerance)
    gamma, c = channel.gamma, channel.c
    if !(channel.successful && isfinite(c) && c > sqrt(eps()))
        return (; theta0 = channel.theta, gamma = NaN, varphi = NaN, delta,
                  Gamma = fill(NaN, 2, 2), c, successful = false)
    end

    bound = gamma * delta
    varphi = delta == 0.0 ? 0.0 : clamp(prior_covariance / c, -bound, bound)
    (; theta0 = channel.theta, gamma, varphi, delta,
       Gamma = [gamma^2 varphi; varphi delta^2], c, successful = true)
end

"""
    solve_estimated_focs(estimate; kappa, alpha, thetaP, kwargs...)

Solve the limiting FOCs with the unknown true parameters replaced by `estimate`.
"""
solve_estimated_focs(estimate; kappa, alpha, thetaP, kwargs...) =
    solve_oracle(OracleProblem(kappa = kappa, alpha = alpha, theta0 = estimate.theta0,
                               thetaP = thetaP, Gamma = estimate.Gamma); kwargs...)

"""Intercept giving the requested marginal prevalence at signal strength gamma."""
calibrated_intercept(gamma, prevalence; quadrature_order = 32) =
    channel_intercept(prevalence, gamma, gauss_hermite_rule(quadrature_order))

"""
    sloe_covariance_status(delta2, nu2, chiP)

Whether `[delta2 chiP; chiP nu2]` is a valid covariance matrix, which is what
rules out SLOE plug-ins that no true parameter value could have produced.
"""
function sloe_covariance_status(delta2, nu2, chiP)
    all(isfinite, (delta2, nu2, chiP)) || return false
    nu2 > 0.0 || return false
    delta2 == 0.0 && return abs(chiP) <= 100 * eps() * max(1.0, abs(chiP))
    nu2 - chiP^2 / delta2 > 100 * eps() * max(1.0, nu2, chiP^2 / delta2)
end
