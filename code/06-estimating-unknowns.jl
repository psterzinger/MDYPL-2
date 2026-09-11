## Philipp Sterzinger 03.09.2026: Code is provided as is, no finished package
## and no guarantees given
##
## The response-moment estimator of the unknown true parameters against SLOE,
## and the state parameters the limiting FOCs give from each estimate.

using CairoMakie, Colors, DataFrames, LaTeXStrings, Printf, Statistics

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
methods = ["O", "SLOE"]
method_labels = Dict("O" => "Response moments", "SLOE" => "SLOE")

main_setting = "A"
main_method = "O"
static_n = last(sample_sizes(results_path))
central_mass = 0.99
interval = (0.10, 0.90)

colour = Dict("O" => colorant"#4B0055", "SLOE" => colorant"#009B95")
main_colour = colorant"#0072B2"
mean_colour = colorant"#D62728"
truth_colour = colorant"#707070"
grid_colour = RGBAf(0.64, 0.68, 0.72, 0.22)
panel_background = colorant"#FCFCFC"
text_colour = colorant"#252525"
truth_linewidth = 2.6
APPENDIX_ROW_HEADER_SIZE = 32
truth_markersize = 8.0

unknown_metrics = [(:theta0_hat, :theta0_true, L"\hat{\theta}_0", :unknown_success),
                   (:gamma_hat, :gamma_true, L"\hat{\gamma}", :unknown_success),
                   (:varphi_hat, :varphi_true, L"\hat{\varphi}", :unknown_success)]
state_metrics = [(:foc_theta_star, :true_theta_star, L"\hat{\theta}^{*}", :foc_success),
                 (:foc_v1_star, :true_v1_star, L"\hat{v}_1^{*}", :foc_success),
                 (:foc_v2_star, :true_v2_star, L"\hat{v}_2^{*}", :foc_success),
                 (:foc_sigma_star, :true_sigma_star, L"\hat{\sigma}^{*}", :foc_success),
                 (:foc_lambda_star, :true_lambda_star, L"\hat{\lambda}^{*}", :foc_success)]
all_metrics = vcat(unknown_metrics, state_metrics)

estimates = reduce(vcat, [read_unknowns(results_path, n)
                          for n in sample_sizes(results_path)])

function metric_values(setting, method, n, metric)
    (value, truth, _, success) = metric
    rows = estimates[(estimates.setting .== setting) .& (estimates.method .== method) .&
                     (estimates.n .== n) .& estimates[!, success], :]
    isempty(rows) && return (values = Float64[], truth = NaN)
    values = Float64.(rows[!, value])
    (values = filter(isfinite, values), truth = Float64(first(rows[!, truth])))
end

no_data!(axis) = text!(axis, 0.5, 0.5; text = "no data", align = (:center, :center),
                       space = :relative, fontsize = EMPTY_PANEL_SIZE)

function two_digit_ticks(lo, hi; target = 3)
    hi > lo || return [lo]
    raw = (hi - lo) / target
    base = 10.0^max(-2.0, floor(log10(raw)))
    step = base
    for candidate in (base, 2base, 5base, 10base)
        step = candidate
        candidate >= raw && break
    end
    step = max(step, 0.01)
    collect(ceil(lo / step) * step:step:hi)
end

function truth_marker!(axis, target; direction = :x)
    isfinite(target) || return nothing
    position = lift(axis.blockscene, axis.scene.viewport,
                    axis.finallimits) do area, limits
        index = direction === :x ? 1 : 2
        lo = limits.origin[index]
        span = limits.widths[index]
        span > 0 || return Point2f[]
        relative = (target - lo) / span
        0 <= relative <= 1 || return Point2f[]
        offset = 0.5 * truth_markersize
        direction === :x ?
            [Point2f(left(area) + relative * widths(area)[1], bottom(area) - offset)] :
            [Point2f(left(area) - offset, bottom(area) + relative * widths(area)[2])]
    end
    marker = scatter!(axis.blockscene, position; marker = :rect,
                      markersize = truth_markersize, color = truth_colour,
                      strokecolor = truth_colour, strokewidth = 0.5)
    translate!(marker, 0, 0, 30)
    return nothing
end

function convergence_panel!(axis, setting, metric)
    sizes = sample_sizes(results_path)
    drawn = false
    for method in methods
        summary = [(n, metric_values(setting, method, n, metric)) for n in sizes]
        keep = [(n, item) for (n, item) in summary if !isempty(item.values)]
        isempty(keep) && continue
        drawn = true
        ns = [n for (n, _) in keep]
        means = [mean(item.values) for (_, item) in keep]
        lower = [quantile(item.values, interval[1]) for (_, item) in keep]
        upper = [quantile(item.values, interval[2]) for (_, item) in keep]
        band!(axis, ns, lower, upper; color = (colour[method], 0.14))
        lines!(axis, ns, means; color = colour[method], linewidth = 2.2)
        scatter!(axis, ns, means; color = colour[method], markersize = 6)
    end
    truth = metric_values(setting, first(methods), last(sizes), metric).truth
    if isfinite(truth)
        hlines!(axis, [truth]; color = truth_colour, linestyle = :dash,
                linewidth = 1.8)
        truth_marker!(axis, truth; direction = :y)
    end
    drawn || no_data!(axis)
    axis
end

function histogram_panel!(axis, setting, metric)
    pooled = Float64[]
    for method in methods
        append!(pooled, metric_values(setting, method, static_n, metric).values)
    end
    if isempty(pooled)
        no_data!(axis)
        return axis
    end
    tail = (1 - central_mass) / 2
    lo, hi = quantile(pooled, tail), quantile(pooled, 1 - tail)
    if hi <= lo
        pad = max(abs(lo), 1.0) * 0.05
        lo, hi = lo - pad, lo + pad
    end
    edges = collect(range(lo, hi; length = 26))
    for method in methods
        values = metric_values(setting, method, static_n, metric).values
        isempty(values) && continue
        hist!(axis, clamp.(values, lo, hi); bins = edges, normalization = :pdf,
              color = (colour[method], 0.42), strokecolor = (colour[method], 0.85),
              strokewidth = 0.6)
    end
    truth = metric_values(setting, first(methods), static_n, metric).truth
    isfinite(truth) && vlines!(axis, [truth]; color = truth_colour,
                               linestyle = :dash, linewidth = truth_linewidth)
    axis.xticks = two_digit_ticks(lo, hi)
    axis.xtickformat = values -> [@sprintf("%.2f", value) for value in values]
    xlims!(axis, lo, hi)
    truth_marker!(axis, truth)
    axis
end

function metric_figure(panel!, metrics, xlabel)
    nrows, ncols = length(metrics), length(setting_codes)
    fig = Figure(size = tile_figure_size(ncols, nrows; width_extra = 130,
                                         height_extra = 120))
    for (column, code) in enumerate(setting_codes)
        Label(fig[1, column], "$code · $(setting_labels[code])";
              fontsize = COLUMN_HEADER_SIZE, font = :bold, tellwidth = false)
    end
    for (row, metric) in enumerate(metrics)
        Label(fig[row + 1, 0], metric[3]; fontsize = APPENDIX_ROW_HEADER_SIZE,
              rotation = 0, tellwidth = true, padding = (0, 12, 0, 0))
        for (column, code) in enumerate(setting_codes)
            axis = Axis(fig[row + 1, column];
                        xlabel = row == nrows ? xlabel : "",
                        xlabelsize = AXIS_LABEL_SIZE,
                        width = TILE_WIDTH, height = TILE_HEIGHT,
                        backgroundcolor = panel_background,
                        xgridcolor = grid_colour, ygridcolor = grid_colour,
                        xticklabelsize = TICK_LABEL_SIZE,
                        yticklabelsize = TICK_LABEL_SIZE,
                        topspinevisible = false, rightspinevisible = false)
            panel!(axis, code, metric)
            column == 1 || hideydecorations!(axis; grid = false)
        end
    end

    Legend(fig[nrows + 2, 1:ncols],
           vcat([PolyElement(color = (colour[method], 0.42),
                             strokecolor = colour[method], strokewidth = 1)
                 for method in methods],
                [[LineElement(color = truth_colour, linestyle = :dash, linewidth = 2),
                  MarkerElement(color = truth_colour, marker = :rect,
                                markersize = truth_markersize,
                                strokecolor = truth_colour, strokewidth = 0.5)]]),
           vcat([method_labels[method] for method in methods], ["Truth"]);
           orientation = :horizontal, labelsize = LEGEND_SIZE, framevisible = false,
           backgroundcolor = :transparent, tellwidth = false, tellheight = true)
    colgap!(fig.layout, TILE_COLUMN_GAP)
    rowgap!(fig.layout, TILE_ROW_GAP)
    resize_to_layout!(fig)
    fig
end

function main_convergence_figure()
    sizes = sample_sizes(results_path)
    fig = Figure(size = tile_figure_size(length(unknown_metrics), 1;
                                         width_extra = 150, height_extra = 70))
    for (column, metric) in enumerate(unknown_metrics)
        axis = Axis(fig[1, column]; xlabel = L"n", title = metric[3],
                    titlesize = COLUMN_HEADER_SIZE, titlegap = 8,
                    xlabelsize = AXIS_LABEL_SIZE,
                    width = TILE_WIDTH, height = TILE_HEIGHT,
                    backgroundcolor = panel_background,
                    xgridcolor = grid_colour, ygridcolor = grid_colour,
                    xticklabelsize = TICK_LABEL_SIZE,
                    yticklabelsize = TICK_LABEL_SIZE,
                    topspinevisible = false, rightspinevisible = false)
        summary = [(n, metric_values(main_setting, main_method, n, metric))
                   for n in sizes]
        keep = [(n, item) for (n, item) in summary if !isempty(item.values)]
        ns = [n for (n, _) in keep]
        band!(axis, ns, [quantile(item.values, interval[1]) for (_, item) in keep],
              [quantile(item.values, interval[2]) for (_, item) in keep];
              color = (main_colour, 0.18))
        means = [mean(item.values) for (_, item) in keep]
        lines!(axis, ns, means; color = main_colour, linewidth = 2.4)
        scatter!(axis, ns, means; color = main_colour, markersize = 6)
        truth = metric_values(main_setting, main_method, static_n, metric).truth
        isfinite(truth) && hlines!(axis, [truth]; color = truth_colour,
                                   linestyle = :dash, linewidth = 1.8)
    end
    colgap!(fig.layout, TILE_COLUMN_GAP)
    resize_to_layout!(fig)
    fig
end

function main_histogram_figure()
    fig = Figure(size = tile_figure_size(length(state_metrics), 1;
                                         width_extra = 150, height_extra = 90))
    for (column, metric) in enumerate(state_metrics)
        axis = Axis(fig[1, column]; title = metric[3],
                    titlesize = COLUMN_HEADER_SIZE, titlegap = 8,
                    xticks = LinearTicks(3),
                    ylabel = column == 1 ? "Density" : "",
                    ylabelsize = AXIS_LABEL_SIZE,
                    width = TILE_WIDTH, height = TILE_HEIGHT,
                    backgroundcolor = panel_background,
                    xgridcolor = grid_colour, ygridcolor = grid_colour,
                    xticklabelsize = TICK_LABEL_SIZE,
                    yticklabelsize = TICK_LABEL_SIZE,
                    yticklabelsvisible = column == 1, yticksvisible = column == 1,
                    topspinevisible = false, rightspinevisible = false)
        item = metric_values(main_setting, main_method, static_n, metric)
        if isempty(item.values)
            no_data!(axis)
            continue
        end
        tail = (1 - central_mass) / 2
        lo, hi = quantile(item.values, tail), quantile(item.values, 1 - tail)
        if hi <= lo
            pad = max(abs(lo), 1.0) * 0.05
            lo, hi = lo - pad, lo + pad
        end
        hist!(axis, clamp.(item.values, lo, hi);
              bins = collect(range(lo, hi; length = 26)), normalization = :pdf,
              color = (main_colour, 0.42), strokecolor = (main_colour, 0.85),
              strokewidth = 0.6)
        vlines!(axis, [mean(item.values)]; color = mean_colour, linestyle = :dash,
                linewidth = 2.0)
        isfinite(item.truth) && vlines!(axis, [item.truth]; color = truth_colour,
                                        linestyle = :solid, linewidth = truth_linewidth)
        xlims!(axis, lo, hi)
        truth_marker!(axis, item.truth)
    end
    Legend(fig[2, 1:length(state_metrics)],
           [LineElement(color = mean_colour, linestyle = :dash, linewidth = 2.0),
            [LineElement(color = truth_colour, linestyle = :solid,
                         linewidth = truth_linewidth),
             MarkerElement(color = truth_colour, marker = :rect,
                           markersize = truth_markersize,
                           strokecolor = truth_colour, strokewidth = 0.5)]],
           ["Monte Carlo mean", "Truth"];
           orientation = :horizontal, labelsize = LEGEND_SIZE, framevisible = false,
           backgroundcolor = :transparent, tellwidth = false, tellheight = true)
    colgap!(fig.layout, TILE_COLUMN_GAP)
    rowgap!(fig.layout, TILE_ROW_GAP)
    resize_to_layout!(fig)
    fig
end

set_theme!(Theme(fontsize = BASE_FONT_SIZE, textcolor = text_colour,
                 Figure = (backgroundcolor = :white,)))

mkpath(figures_path)
for (stem, figure) in (
        ("07-response-moment-convergence", main_convergence_figure()),
        ("07-state-parameter-histograms-n$(static_n)", main_histogram_figure()),
        ("S-unknowns-all-parameters-convergence",
         metric_figure(convergence_panel!, all_metrics, L"n")),
        ("S-unknowns-all-parameters-histograms-n$(static_n)",
         metric_figure(histogram_panel!, all_metrics, "")))
    save(joinpath(figures_path, "$stem.pdf"), figure)
    println("wrote ", joinpath(figures_path, "$stem.pdf"))
end
