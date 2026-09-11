## Philipp Sterzinger 03.09.2026: Code is provided as is, no finished package
## and no guarantees given
##
## Misclassification error rate of the mDYPL linear score, adjusted against
## unadjusted. `gaussian_mer` evaluates the error rate of a linear score by
## Gaussian quadrature against the true logit.

using Arrow, CairoMakie, DataFrames, Distributions, FastGaussQuadrature,
      LaTeXStrings, LinearAlgebra, Statistics

supp_path = abspath(joinpath(@__DIR__, ".."))
results_path = joinpath(supp_path, "results")
figures_path = joinpath(supp_path, "figures")

include(joinpath(supp_path, "code", "methods", "PublicationPlots.jl"))
using .PublicationPlots
include(joinpath(supp_path, "code", "methods", "simulation_output.jl"))
using .SimulationOutput

setting_codes = ["A", "O", "I", "M", "N"]
setting_titles = Dict(
    "A" => L"\mathrm{A}\,\cdot\,\mathrm{Aligned}",
    "O" => L"\mathrm{O}\,\cdot\,\mathrm{Orthogonal}",
    "I" => L"\mathrm{I}\,\cdot\,\mathrm{Intercept\ mismatch}",
    "M" => L"\mathrm{M}\,\cdot\,\mathrm{Misleading}",
    "N" => L"\mathrm{N}\,\cdot\,\mathrm{Null\ prior}")

interval = (0.10, 0.90)
quadrature_order = 64

ink = RGBf(0.15, 0.15, 0.16)
grid_colour = RGBf(0.84, 0.85, 0.88)
paper = RGBf(1, 1, 1)
purple = RGBf(0.30, 0.00, 0.34)
teal = RGBf(0.00, 0.61, 0.58)
mid_grey = RGBf(0.45, 0.45, 0.48)

logistic(x) = x >= 0 ? inv(1.0 + exp(-x)) : (z = exp(x); z / (1.0 + z))

let (nodes, weights) = gausshermite(quadrature_order)
    global quad_nodes = sqrt(2) .* nodes
    global quad_weights = weights ./ sqrt(pi)
end

function gaussian_mer(theta0, gamma2, mean, variance, covariance, threshold,
                      orientation)
    if orientation == 0
        q = gamma2 <= 0 ? logistic(theta0) :
            sum(quad_weights .* logistic.(theta0 .+ sqrt(gamma2) .* quad_nodes))
        return min(q, 1 - q)
    end
    conditional_sd = sqrt(max(variance - covariance^2 / gamma2, 0.0))
    risk = 0.0
    for i in eachindex(quad_nodes)
        true_logit = theta0 + sqrt(gamma2) * quad_nodes[i]
        q = logistic(true_logit)
        conditional_mean = mean + covariance / gamma2 * (true_logit - theta0)
        below = conditional_sd > 1e-12 ?
                cdf(Normal(), (threshold - conditional_mean) / conditional_sd) :
                (conditional_mean <= threshold ? 1.0 : 0.0)
        risk += quad_weights[i] * (orientation > 0 ? q * below + (1 - q) * (1 - below) :
                                                     q * (1 - below) + (1 - q) * below)
    end
    risk
end

function score_law(theta0, gamma2, mean, variance, covariance)
    (variance > 0 && gamma2 > 0 && all(isfinite, (mean, variance, covariance))) ||
        return (threshold = NaN, orientation = 0, risk = NaN)
    if abs(covariance) <= 1e-10 * max(sqrt(gamma2 * variance), 1.0)
        return (threshold = NaN, orientation = 0,
                risk = gaussian_mer(theta0, gamma2, mean, variance, covariance, 0.0, 0))
    end
    orientation = covariance > 0 ? 1 : -1
    threshold = mean - variance / covariance * theta0
    (threshold = threshold, orientation = orientation,
     risk = gaussian_mer(theta0, gamma2, mean, variance, covariance, threshold,
                         orientation))
end

function mer_rows(n, code, score)
    tuning = score == "Adjusted" ? :adjusted : :prediction
    df = read_setting(results_path, code, n; tuning = tuning)
    settings = setting_row(read_settings(results_path, n), code)
    signals = read_signals(results_path, n)
    beta0 = signal_vector(signals, "beta0")
    betaP = signal_vector(signals, "betaP"; setting = code)
    p = length(beta0)

    theta0, gamma2, varphi = settings.theta0, settings.Gamma11, settings.Gamma12
    v1, v2, sigma = df.v1_star[1], df.v2_star[1], df.sigma_star[1]

    law = if score == "Adjusted"
        score_law(theta0, gamma2, theta0, gamma2 + sigma^2 / v1^2, gamma2)
    else
        v = [v1, v2]
        Gamma = [gamma2 varphi; varphi settings.Gamma22]
        score_law(theta0, gamma2, df.theta_star[1], dot(v, Gamma * v) + sigma^2,
                  gamma2 * v1 + varphi * v2)
    end

    betas = beta_matrix(df)
    sample_gamma2 = dot(beta0, beta0) / p
    rows = NamedTuple[]
    for i in converged_rows(df, betas)
        beta_hat = vec(view(betas, i, :))
        mer = if score == "Adjusted"
            variance = max((dot(beta_hat, beta_hat) - 2 * v2 * dot(betaP, beta_hat) +
                            v2^2 * dot(betaP, betaP)) / (v1^2 * p), eps())
            covariance = (dot(beta0, beta_hat) - v2 * dot(beta0, betaP)) / (v1 * p)
            gaussian_mer(theta0, sample_gamma2, theta0, variance, covariance,
                         law.threshold, law.orientation)
        else
            gaussian_mer(theta0, sample_gamma2, df.theta_hat[i],
                         dot(beta_hat, beta_hat) / p, dot(beta0, beta_hat) / p,
                         law.threshold, law.orientation)
        end
        push!(rows, (setting = code, n = n, score = score, mer = mer,
                     theory = law.risk))
    end
    rows
end

function mer_summary()
    raw = DataFrame(reduce(vcat, [mer_rows(n, code, score)
                                  for n in sample_sizes(results_path)
                                  for code in setting_codes
                                  for score in ("Adjusted", "Unadjusted")]))
    combine(groupby(raw, [:setting, :n, :score]),
            :mer => median => :median,
            :mer => (v -> quantile(v, interval[1])) => :lower,
            :mer => (v -> quantile(v, interval[2])) => :upper,
            :theory => first => :theory)
end

function mer_figure(summary)
    fig = Figure(size = tile_figure_size(length(setting_codes), 1))
    lo, hi = extrema(vcat(summary.lower, summary.upper, summary.theory))
    span = max(hi - lo, 0.02)
    ylimits = (max(0.0, lo - 0.10 * span), min(1.0, hi + 0.10 * span))
    handles = Dict{String,Any}()
    theory_handle = nothing

    for (column, code) in enumerate(setting_codes)
        axis = Axis(fig[1, column]; title = setting_titles[code],
                    titlesize = COLUMN_HEADER_SIZE,
                    xlabel = column == cld(length(setting_codes), 2) ? L"n" : "",
                    xlabelsize = AXIS_LABEL_SIZE,
                    ylabel = column == 1 ? L"\mathrm{MER}" : "",
                    ylabelsize = COLUMN_HEADER_SIZE,
                    width = TILE_WIDTH, height = TILE_HEIGHT,
                    yticklabelsvisible = column == 1, yticksvisible = column == 1)
        for (score, colour) in (("Adjusted", teal), ("Unadjusted", purple))
            panel = sort(summary[(summary.setting .== code) .&
                                 (summary.score .== score), :], :n)
            handles[score] = lines!(axis, panel.n, panel.median; color = colour,
                                    linewidth = 2.4)
            scatter!(axis, panel.n, panel.median; color = colour, markersize = 6)
            band!(axis, panel.n, panel.lower, panel.upper; color = (colour, 0.12))
            theory = hlines!(axis, [mean(panel.theory)]; color = mid_grey,
                             linestyle = :dash, linewidth = 1.7)
            theory_handle === nothing && (theory_handle = theory)
        end
        ylims!(axis, ylimits...)
    end

    Legend(fig[2, 1:length(setting_codes)],
           [handles["Adjusted"], handles["Unadjusted"], theory_handle],
           ["Adjusted", "Unadjusted", "Theoretical limit"];
           orientation = :horizontal, tellwidth = false, tellheight = true)
    rowgap!(fig.layout, TILE_ROW_GAP)
    colgap!(fig.layout, TILE_COLUMN_GAP)
    resize_to_layout!(fig)
    fig
end

set_theme!(Theme(fontsize = BASE_FONT_SIZE,
                 Figure = (backgroundcolor = paper, figure_padding = (10, 14, 8, 8)),
                 Axis = (backgroundcolor = paper,
                         xlabelsize = AXIS_LABEL_SIZE, ylabelsize = AXIS_LABEL_SIZE,
                         xticklabelsize = TICK_LABEL_SIZE,
                         yticklabelsize = TICK_LABEL_SIZE,
                         xgridcolor = (grid_colour, 0.7),
                         ygridcolor = (grid_colour, 0.7),
                         topspinevisible = false, rightspinevisible = false,
                         leftspinecolor = ink, bottomspinecolor = ink),
                 Legend = (framevisible = false, backgroundcolor = :transparent,
                           labelsize = LEGEND_SIZE)))

mkpath(figures_path)
save(joinpath(figures_path, "04-mer.pdf"), mer_figure(mer_summary()))
println("wrote the prediction figure to ", figures_path)
