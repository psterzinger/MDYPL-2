## Philipp Sterzinger 03.09.2026: Code is provided as is, no finished package
## and no guarantees given
##
## Convergence of the mDYPL estimator to its limit as the sample size grows.
## `metric_rows` forms the per-replication metrics and the limits the FOCs
## predict for them.

using Arrow, CairoMakie, DataFrames, LaTeXStrings, LinearAlgebra, Statistics

supp_path = abspath(joinpath(@__DIR__, ".."))
results_path = joinpath(supp_path, "results")
figures_path = joinpath(supp_path, "figures")

include(joinpath(supp_path, "code", "methods", "PublicationPlots.jl"))
using .PublicationPlots
include(joinpath(supp_path, "code", "methods", "simulation_output.jl"))
using .SimulationOutput

enable_check_accent!()

setting_codes = ["A", "O", "I", "M", "N"]
setting_labels = Dict("A" => "Aligned", "O" => "Orthogonal",
                      "I" => "Intercept mismatch", "M" => "Misleading",
                      "N" => "Null prior")
setting_titles = Dict(
    "A" => L"\mathrm{A}\,\cdot\,\mathrm{Aligned}",
    "O" => L"\mathrm{O}\,\cdot\,\mathrm{Orthogonal}",
    "I" => L"\mathrm{I}\,\cdot\,\mathrm{Intercept\ mismatch}",
    "M" => L"\mathrm{M}\,\cdot\,\mathrm{Misleading}",
    "N" => L"\mathrm{N}\,\cdot\,\mathrm{Null\ prior}")

interval = (0.10, 0.90)

metrics = [
    (:projection,   L"p^{-1}\beta_0^\top(\hat{\beta}^{\mathrm{DY}}-\beta^*)"),
    (:target_mse,   L"p^{-1}\Vert\hat{\beta}^{\mathrm{DY}}-\beta^*\Vert_2^2"),
    (:raw_mse,      L"p^{-1}\Vert\hat{\beta}^{\mathrm{DY}}-\beta_0\Vert_2^2"),
    (:adjusted_mse, L"p^{-1}\Vert\check{\beta}-\beta_0\Vert_2^2"),
    (:theta,        L"\hat{\theta}_{\mathrm{DY}}"),
]

blue = RGBf(0.08, 0.35, 0.64)
grid_colour = RGBf(0.87, 0.88, 0.90)
mid_grey = RGBf(0.45, 0.45, 0.48)
gold = RGBf(0.72, 0.54, 0.07)
ink = RGBf(0.15, 0.15, 0.16)
paper = RGBf(1, 1, 1)
purple = RGBf(0.30, 0.00, 0.34)
teal = RGBf(0.00, 0.61, 0.58)

function metric_rows(n, code)
    signals = read_signals(results_path, n)
    beta0 = signal_vector(signals, "beta0")
    betaP = signal_vector(signals, "betaP"; setting = code)
    p = length(beta0)
    rows = NamedTuple[]

    adjusted = read_setting(results_path, code, n)
    betas = beta_matrix(adjusted)
    v1, v2 = adjusted.v1_star[1], adjusted.v2_star[1]
    sigma, theta_star, theta0 = adjusted.sigma_star[1], adjusted.theta_star[1],
                                adjusted.theta0[1]
    beta_star = v1 .* beta0 .+ v2 .* betaP
    for i in converged_rows(adjusted, betas)
        beta = vec(view(betas, i, :))
        residual = beta .- beta_star
        rescaled = (beta .- v2 .* betaP) ./ v1
        for (metric, value, target) in (
                (:projection, dot(beta0, residual) / p, 0.0),
                (:target_mse, sum(abs2, residual) / p, sigma^2),
                (:adjusted_mse, sum(abs2, rescaled .- beta0) / p, sigma^2 / v1^2),
                (:theta, adjusted.theta_hat[i], theta_star))
            push!(rows, (; setting = code, n, metric, value, target, theta0))
        end
    end

    unadjusted = read_setting(results_path, code, n; tuning = :unadjusted)
    betas = beta_matrix(unadjusted)
    u1, u2 = unadjusted.v1_star[1], unadjusted.v2_star[1]
    target = sum(abs2, u1 .* beta0 .+ u2 .* betaP .- beta0) / p +
             unadjusted.sigma_star[1]^2
    for i in converged_rows(unadjusted, betas)
        push!(rows, (setting = code, n = n, metric = :raw_mse,
                     value = sum(abs2, vec(view(betas, i, :)) .- beta0) / p,
                     target = target, theta0 = theta0))
    end
    rows
end

function summarise()
    raw = DataFrame(reduce(vcat, [metric_rows(n, code)
                                  for n in sample_sizes(results_path)
                                  for code in setting_codes]))
    combine(groupby(raw, [:setting, :n, :metric]),
            :value => mean => :mean,
            :value => (v -> quantile(v, interval[1])) => :lower,
            :value => (v -> quantile(v, interval[2])) => :upper,
            :target => first => :target,
            :theta0 => first => :theta0)
end

function convergence_panel!(slot, summary, code, metric; xlabel = "")
    rows = sort(summary[(summary.setting .== code) .& (summary.metric .== metric), :], :n)
    axis = Axis(slot; xlabel, xlabelsize = AXIS_LABEL_SIZE,
                width = TILE_WIDTH, height = TILE_HEIGHT)
    band!(axis, rows.n, rows.lower, rows.upper; color = (blue, 0.15))
    lines!(axis, rows.n, rows.mean; color = blue, linewidth = 2.2)
    scatter!(axis, rows.n, rows.mean; color = blue, markersize = 4.5)
    hlines!(axis, [mean(rows.target)]; color = mid_grey, linestyle = :dash,
            linewidth = 1.5)
    metric == :theta && hlines!(axis, [mean(rows.theta0)]; color = gold,
                                linestyle = :dot, linewidth = 1.8)
    axis
end

function combined_figure(summary)
    nrows, ncols = length(metrics), length(setting_codes)
    fig = Figure(size = tile_figure_size(ncols, nrows; width_extra = 120,
                                         height_extra = 100))
    for (column, code) in enumerate(setting_codes)
        Label(fig[1, column], "$code · $(setting_labels[code])";
              fontsize = COLUMN_HEADER_SIZE, font = :bold, tellwidth = false)
    end

    axes = Matrix{Axis}(undef, nrows, ncols)
    for (row, (metric, label)) in enumerate(metrics)
        Label(fig[row + 1, 0], label; fontsize = ROW_HEADER_SIZE, rotation = pi / 2,
              tellwidth = true, padding = (0, 12, 0, 0))
        for (column, code) in enumerate(setting_codes)
            axes[row, column] = convergence_panel!(fig[row + 1, column], summary,
                                                   code, metric;
                                                   xlabel = row == nrows ? L"n" : "")
            column == 1 || hideydecorations!(axes[row, column]; grid = false)
        end
        linkyaxes!(axes[row, :]...)
    end

    Legend(fig[nrows + 2, 1:ncols],
           [LineElement(color = blue, linewidth = 2.2),
            LineElement(color = mid_grey, linestyle = :dash, linewidth = 1.5)],
           ["Empirical", "Theoretical"];
           orientation = :horizontal, labelsize = LEGEND_SIZE, framevisible = false,
           backgroundcolor = :transparent, tellwidth = false, tellheight = true)
    colgap!(fig.layout, TILE_COLUMN_GAP)
    rowgap!(fig.layout, TILE_ROW_GAP)
    resize_to_layout!(fig)
    fig
end

function slope_mse_figure(summary)
    fig = Figure(size = tile_figure_size(length(setting_codes), 1),
                 backgroundcolor = paper, figure_padding = (10, 14, 8, 8))
    rows = summary[in.(summary.metric, Ref((:raw_mse, :adjusted_mse))), :]
    lo, hi = extrema(vcat(rows.lower, rows.upper, rows.target))
    span = max(hi - lo, 0.1)
    ylimits = (max(0.0, lo - 0.10 * span), hi + 0.10 * span)

    series = [(label = "Adjusted", metric = :adjusted_mse, color = teal),
              (label = "Unadjusted", metric = :raw_mse, color = purple)]
    handles = Dict{String,Any}()
    theory_handle = nothing

    for (column, code) in enumerate(setting_codes)
        axis = Axis(fig[1, column];
                    title = setting_titles[code], titlesize = COLUMN_HEADER_SIZE,
                    xlabel = column == cld(length(setting_codes), 2) ? L"n" : "",
                    ylabel = column == 1 ? L"\mathrm{Slope\ MSE}" : "",
                    ylabelsize = COLUMN_HEADER_SIZE, xlabelsize = AXIS_LABEL_SIZE,
                    xticklabelsize = TICK_LABEL_SIZE, yticklabelsize = TICK_LABEL_SIZE,
                    width = TILE_WIDTH, height = TILE_HEIGHT,
                    yticklabelsvisible = column == 1, yticksvisible = column == 1,
                    backgroundcolor = paper, xgridcolor = (grid_colour, 0.7),
                    ygridcolor = (grid_colour, 0.7), topspinevisible = false,
                    rightspinevisible = false, leftspinecolor = ink,
                    bottomspinecolor = ink)
        for item in series
            panel = sort(summary[(summary.setting .== code) .&
                                 (summary.metric .== item.metric), :], :n)
            handles[item.label] = lines!(axis, panel.n, panel.mean;
                                         color = item.color, linewidth = 2.4)
            scatter!(axis, panel.n, panel.mean; color = item.color, markersize = 6)
            band!(axis, panel.n, panel.lower, panel.upper; color = (item.color, 0.12))
            theory = hlines!(axis, [mean(panel.target)]; color = mid_grey,
                             linestyle = :dash, linewidth = 1.7)
            theory_handle === nothing && (theory_handle = theory)
        end
        ylims!(axis, ylimits...)
    end

    Legend(fig[2, 1:length(setting_codes)],
           [handles["Adjusted"], handles["Unadjusted"], theory_handle],
           ["Adjusted", "Unadjusted", "Theoretical limit"];
           orientation = :horizontal, labelsize = LEGEND_SIZE, framevisible = false,
           backgroundcolor = :transparent, tellwidth = false, tellheight = true)
    rowgap!(fig.layout, TILE_ROW_GAP)
    colgap!(fig.layout, TILE_COLUMN_GAP)
    resize_to_layout!(fig)
    fig
end

set_theme!(Theme(fontsize = BASE_FONT_SIZE, Figure = (backgroundcolor = :white,),
                 Axis = (backgroundcolor = RGBf(0.985, 0.986, 0.988),
                         xlabelsize = AXIS_LABEL_SIZE, ylabelsize = AXIS_LABEL_SIZE,
                         xticklabelsize = TICK_LABEL_SIZE,
                         yticklabelsize = TICK_LABEL_SIZE,
                         xgridcolor = (grid_colour, 0.7),
                         ygridcolor = (grid_colour, 0.7),
                         topspinevisible = false, rightspinevisible = false)))

mkpath(figures_path)
summary = summarise()
save(joinpath(figures_path, "S-convergence-all-settings.pdf"), combined_figure(summary))
save(joinpath(figures_path, "03-slope-mse.pdf"), slope_mse_figure(summary))
println("wrote the convergence figures to ", figures_path)
