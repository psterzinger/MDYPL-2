## Philipp Sterzinger 03.09.2026: Code is provided as is, no finished package
## and no guarantees given
##
## The six single-panel figures of the introduction, all for the misleading-prior
## setting: the fitted slope and intercept against their limits and against the
## truth, and the two normal QQ plots.

using Arrow, CairoMakie, DataFrames, Distributions, LaTeXStrings, LinearAlgebra,
      Random, StableRNGs, Statistics

supp_path = abspath(joinpath(@__DIR__, ".."))
results_path = joinpath(supp_path, "results")
figures_path = joinpath(supp_path, "figures")

include(joinpath(supp_path, "code", "methods", "PublicationPlots.jl"))
using .PublicationPlots
include(joinpath(supp_path, "code", "methods", "simulation_output.jl"))
using .SimulationOutput

setting = "M"
reference_n = 4000
qq_coordinate = 1
qq_replications = 500
interval = (0.10, 0.90)

purple = RGBf(0.47, 0.24, 0.56)
teal = RGBf(0.08, 0.66, 0.64)
blue = RGBf(0.20, 0.38, 0.55)
gold = RGBf(0.72, 0.54, 0.07)
truth_grey = RGBf(0.44, 0.44, 0.44)

logistic(x) = x >= 0 ? inv(1.0 + exp(-x)) : (z = exp(x); z / (1.0 + z))

function styled_axis(fig; kwargs...)
    axis = Axis(fig[1, 1]; kwargs...)
    axis.xticklabelsize = 22
    axis.yticklabelsize = 22
    axis.xlabelsize = 30
    axis.ylabelsize = 30
    axis.xgridcolor = RGBAf(0.75, 0.75, 0.78, 0.35)
    axis.ygridcolor = RGBAf(0.75, 0.75, 0.78, 0.35)
    axis.topspinevisible = false
    axis.rightspinevisible = false
    axis
end

function padded_limits(values)
    lo, hi = extrema(values)
    pad = hi == lo ? 1.0 : 0.05 * (hi - lo)
    (lo - pad, hi + pad)
end

function scatter_figure(x, y, xlabel, ylabel; through_origin = false)
    lo, hi = padded_limits(vcat(x, y))
    fig = Figure(size = (500, 420), fontsize = 15)
    axis = styled_axis(fig; xlabel, ylabel, aspect = DataAspect())
    scatter!(axis, x, y; color = (purple, 0.72), markersize = 8)
    lines!(axis, [lo, hi], [lo, hi]; color = (:black, 0.75), linestyle = :dash)
    fit_x = collect(extrema(x))
    if through_origin
        lines!(axis, fit_x, (dot(x, y) / dot(x, x)) .* fit_x;
               color = :red, linestyle = :dash, linewidth = 2)
    else
        slope = cov(x, y) / var(x)
        lines!(axis, fit_x, (mean(y) - slope * mean(x)) .+ slope .* fit_x;
               color = :red, linestyle = :dash, linewidth = 2)
    end
    xlims!(axis, lo, hi)
    ylims!(axis, lo, hi)
    fig
end

function qq_figure(values, xlabel, ylabel; ylabelsize = 30)
    m = length(values)
    theoretical = quantile.(Normal(), ((1:m) .- 0.5) ./ m)
    empirical = sort(values)
    lo, hi = padded_limits(vcat(theoretical, empirical))
    fig = Figure(size = (500, 420), fontsize = 15)
    axis = styled_axis(fig; xlabel, ylabel, aspect = DataAspect())
    axis.ylabelsize = ylabelsize
    scatter!(axis, theoretical, empirical; color = (teal, 0.78), markersize = 8)
    lines!(axis, [lo, hi], [lo, hi]; color = (:black, 0.75), linestyle = :dash)
    xlims!(axis, lo, hi)
    ylims!(axis, lo, hi)
    fig
end

reference = read_setting(results_path, setting, reference_n)
betas = beta_matrix(reference)
rows = converged_rows(reference, betas)

signals = read_signals(results_path, reference_n)
beta0 = signal_vector(signals, "beta0")
betaP = signal_vector(signals, "betaP"; setting = setting)

theta = reference.theta_hat[rows]
representative = rows[argmin(abs.(theta .- median(theta)))]
beta_hat = vec(view(betas, representative, :))

beta_star = reference.v1_star[1] .* beta0 .+ reference.v2_star[1] .* betaP
standardised = (beta_hat .- beta_star) ./ reference.sigma_star[1]

function classical_z(seed, n, p, theta_hat, beta_hat, k)
    H = randn(StableRNG(seed), n, p)
    mu = logistic.(theta_hat .+ H * (beta_hat ./ sqrt(p)))
    A = hcat(ones(n), H) .* sqrt.(mu .* (1.0 .- mu))
    e = zeros(p + 1)
    e[k + 1] = 1.0
    variance = (Symmetric(A' * A) \ e)[k + 1]
    variance > 0 ? beta_hat[k] / sqrt(p * variance) : NaN
end

function classical_z_values(k)
    n, p = reference.n[1], reference.p[1]
    values = Float64[]
    for i in first(rows, qq_replications)
        z = try
            classical_z(reference.seed[i], n, p, reference.theta_hat[i],
                        vec(view(betas, i, :)), k)
        catch
            NaN
        end
        isfinite(z) && push!(values, z)
    end
    isempty(values) && error("no finite classical z statistics were produced")
    values
end

function intercept_summary(n)
    df = read_setting(results_path, setting, n)
    theta = df.theta_hat[converged_rows(df, beta_matrix(df))]
    (n = n, mean = mean(theta), lower = quantile(theta, interval[1]),
     upper = quantile(theta, interval[2]), theta_star = df.theta_star[1],
     theta0 = df.theta0[1])
end

summaries = intercept_summary.(sample_sizes(results_path))

reference_markersize = 8.0

function reference_marker!(axis, value, colour)
    position = lift(axis.blockscene, axis.scene.viewport,
                    axis.finallimits) do area, limits
        lo, hi = minimum(limits)[2], maximum(limits)[2]
        relative = (value - lo) / (hi - lo)
        [Point2f(left(area) - 0.5 * reference_markersize,
                 bottom(area) + relative * height(area))]
    end
    marker = scatter!(axis.blockscene, position; marker = :rect,
                      markersize = reference_markersize, color = colour,
                      strokecolor = colour, strokewidth = 0.5)
    translate!(marker, 0, 0, 30)
    return nothing
end

function intercept_figure(summaries, reference_value, colour, linestyle;
                          marker = false)
    n = [item.n for item in summaries]
    means = [item.mean for item in summaries]
    fig = Figure(size = (500, 420), fontsize = 15)
    axis = styled_axis(fig; xlabel = L"n", ylabel = L"\hat{\theta}^{\mathrm{DY}}")
    length(n) > 1 && band!(axis, n, [item.lower for item in summaries],
                           [item.upper for item in summaries]; color = (blue, 0.18))
    hlines!(axis, [reference_value]; color = colour, linestyle = linestyle,
            linewidth = 2.6)
    lines!(axis, n, means; color = blue, linewidth = 3)
    scatter!(axis, n, means; color = blue, markersize = 8)
    length(n) == 1 && xlims!(axis, n[1] - max(n[1] ÷ 10, 1), n[1] + max(n[1] ÷ 10, 1))
    marker && reference_marker!(axis, reference_value, colour)
    fig
end

mkpath(figures_path)
enable_widehat!()

save(joinpath(figures_path, "01-beta.pdf"),
     scatter_figure(beta_star, beta_hat, L"v_1^*\beta_{0,j}+v_2^*\beta_{P,j}",
                    L"\hat{\beta}^{\mathrm{DY}}_j"))
save(joinpath(figures_path, "01-beta-truth.pdf"),
     scatter_figure(beta0, beta_hat, L"\beta_{0,j}", L"\hat{\beta}^{\mathrm{DY}}_j";
                    through_origin = true))
save(joinpath(figures_path, "01-qq.pdf"),
     qq_figure(standardised, L"\mathrm{N}(0,1)",
               L"(\hat{\beta}^{\mathrm{DY}}_j-\beta_j^*)/\sigma^*"))
save(joinpath(figures_path, "01-qq-classical.pdf"),
     qq_figure(classical_z_values(qq_coordinate), L"\mathrm{N}(0,1)",
               L"\hat{\beta}^{\mathrm{DY}}_j/[(X^\top \widehat{W}X)^{-1}]_{jj}^{1/2}";
               ylabelsize = 30))
save(joinpath(figures_path, "01-intercept.pdf"),
     intercept_figure(summaries, mean(item.theta_star for item in summaries),
                      truth_grey, :solid; marker = true))
save(joinpath(figures_path, "01-intercept-truth.pdf"),
     intercept_figure(summaries, mean(item.theta0 for item in summaries), gold, :dot))

println("wrote the introduction figures to ", figures_path)
