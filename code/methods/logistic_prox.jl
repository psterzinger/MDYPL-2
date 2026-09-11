## Philipp Sterzinger 03.09.2026: Code is provided as is, no finished package
## and no guarantees given
##
## Logistic link, the proximal map of the logistic loss, the transforms keeping
## the FOC solvers in their feasible set, and the Gaussian quadrature rules.
## `prox_state` returns everything the limiting moment equations need.

logistic(x) = x >= 0 ? inv(1.0 + exp(-x)) : (z = exp(x); z / (1.0 + z))
softplus(x) = x >= 0 ? x + log1p(exp(-x)) : log1p(exp(x))

function logistic_derivatives(x)
    mu = logistic(x)
    h = mu * (1.0 - mu)
    (mu, h, h * (1.0 - 2.0 * mu))
end

log_scale(x) = (v = exp(x); (v, v))
inverse_log_scale(y) = log(y)

function partial_correlation(psi)
    s = inv(hypot(1.0, psi))
    rho = psi * s
    (rho, s, s^3, -rho * s^2)
end

inverse_partial_correlation(rho) = rho / sqrt((1.0 - rho) * (1.0 + rho))

struct ProxState
    eta::Float64
    h::Float64
    k::Float64
    D::Float64
    R::Float64
    W::Float64
end

function prox_state(x, y, lambda; tol = 16 * eps(), maxiter = 80)
    if lambda == 0.0
        mu, h, k = logistic_derivatives(x)
        return ProxState(x, h, k, 1.0, mu - y, h)
    end

    lo = x + lambda * (y - 1.0)
    hi = x + lambda * y
    eta = clamp(x, lo, hi)
    atol = tol * max(1.0, abs(x), lambda)

    mu, h, k = logistic_derivatives(eta)
    f = eta + lambda * mu - x - lambda * y
    f > 0.0 ? (hi = eta) : (lo = eta)

    iter = 0
    while abs(f) > atol && iter < maxiter
        iter += 1
        candidate = eta - f / (1.0 + lambda * h)
        guard = (hi - lo) / 10.0
        (lo + guard < candidate < hi - guard) && isfinite(candidate) ||
            (candidate = (lo + hi) / 2.0)
        eta = candidate
        mu, h, k = logistic_derivatives(eta)
        f = eta + lambda * mu - x - lambda * y
        f > 0.0 ? (hi = eta) : (lo = eta)
    end

    D = 1.0 + lambda * h
    ProxState(eta, h, k, D, mu - y, h / D)
end

"""
    prox_logistic(x, y, lambda)

Compute `prox_{lambda * rho}(x + lambda * y)` for `rho(t) = log(1 + exp(t))`.
"""
prox_logistic(x, y, lambda; kwargs...) = prox_state(x, y, lambda; kwargs...).eta

struct QuadRule
    nodes::Vector{Float64}
    weights::Vector{Float64}
end

"""Gauss-Hermite rule normalised to integrate against the standard normal."""
function gauss_hermite_rule(order)
    nodes, weights = gausshermite(order; normalize = true)
    QuadRule(nodes, weights ./ sum(weights))
end

dnorm(x) = exp(-x^2 / 2) / sqrt(2 * pi)

"""
    split_normal_rule(order, splitpoints; bound = 9)

Panelled Gauss-Legendre rule for the standard normal on `[-bound, bound]`, with
panel boundaries at zero and at every split point inside the range.
"""
function split_normal_rule(order, splitpoints; bound = 9.0)
    breaks = sort!(unique!(vcat([-bound, 0.0, bound],
                                filter(z -> isfinite(z) && -bound < z < bound,
                                       splitpoints))))
    base_nodes, base_weights = gausslegendre(order)
    nodes = Float64[]
    weights = Float64[]
    for panel in 1:(length(breaks) - 1)
        mid = (breaks[panel] + breaks[panel + 1]) / 2
        half = (breaks[panel + 1] - breaks[panel]) / 2
        for j in eachindex(base_nodes)
            z = mid + half * base_nodes[j]
            push!(nodes, z)
            push!(weights, half * base_weights[j] * dnorm(z))
        end
    end
    QuadRule(nodes, weights ./ sum(weights))
end
