## Philipp Sterzinger 03.09.2026: Code is provided as is, no finished package
## and no guarantees given
##
## QQ plots and p-value histograms of the adjusted z and the rescaled PLR
## statistic, one column per setting, against their limiting laws.

using CairoMakie, Colors, DataFrames, Distributions, LaTeXStrings, Statistics

supp_path = abspath(joinpath(@__DIR__, ".."))
results_path = joinpath(supp_path, "results")
figures_path = joinpath(supp_path, "figures")

include(joinpath(supp_path, "code", "methods", "PublicationPlots.jl"))
using .PublicationPlots
include(joinpath(supp_path, "code", "methods", "simulation_output.jl"))
using .SimulationOutput

enable_check_accent!()

setting_codes = ["A", "O", "I", "M", "N"]
setting_titles = Dict(
    "A" => L"\mathrm{A}\,\cdot\,\mathrm{Aligned}",
    "O" => L"\mathrm{O}\,\cdot\,\mathrm{Orthogonal}",
    "I" => L"\mathrm{I}\,\cdot\,\mathrm{Intercept\ mismatch}",
    "M" => L"\mathrm{M}\,\cdot\,\mathrm{Misleading}",
    "N" => L"\mathrm{N}\,\cdot\,\mathrm{Null\ prior}")

reference_n = last(sample_sizes(results_path))
pvalue_bins = 20

turquoise = colorant"#009B95"
ink = colorant"#27272A"
grid_colour = colorant"#D9DCE1"
paper = colorant"#FFFFFF"

statistics = Dict(code => inference_statistics(results_path, code, reference_n)
                  for code in setting_codes)
plr_df = statistics[first(setting_codes)].plr_df

common = reduce(intersect, [Set(item.replicate) for item in values(statistics)])

function selected(code, statistic)
    item = statistics[code]
    keep = in.(item.replicate, Ref(common))
    (values = getproperty(item, statistic)[keep],
     pvalues = getproperty(item, Symbol(statistic, :_pvalue))[keep])
end

function pvalue_density(pvalues)
    edges = range(0.0, 1.0; length = pvalue_bins + 1)
    counts = zeros(pvalue_bins)
    for value in pvalues
        p = clamp(value, 0.0, 1.0)
        counts[p == 1.0 ? pvalue_bins : floor(Int, p * pvalue_bins) + 1] += 1.0
    end
    width = inv(pvalue_bins)
    ((edges[1:(end - 1)] .+ edges[2:end]) ./ 2, counts ./ (length(pvalues) * width),
     width)
end

function inference_figure(statistic)
    normal_reference = statistic == :z
    reference = normal_reference ? Normal() : Chisq(plr_df)
    data = Dict(code => let item = selected(code, statistic)
                    m = length(item.values)
                    merge(item, (theoretical = quantile.(reference,
                                                         ((1:m) .- 0.5) ./ m),
                                 empirical = sort(item.values)))
                end for code in setting_codes)

    lo = minimum(min(minimum(d.theoretical), minimum(d.empirical)) for d in values(data))
    hi = maximum(max(maximum(d.theoretical), maximum(d.empirical)) for d in values(data))
    limits = normal_reference ?
             (-1.06 * max(abs(lo), abs(hi), 1.0), 1.06 * max(abs(lo), abs(hi), 1.0)) :
             (0.0, 1.06 * max(hi, 1.0))
    density_upper = 1.12 * maximum(maximum(pvalue_density(d.pvalues)[2])
                                   for d in values(data))

    x_label = normal_reference ? L"\mathrm{N}(0,1)" :
              latexstring("\\chi_{$plr_df}^2")
    y_label = normal_reference ? L"\check{\beta}_1/(\sigma^*/v_1^*)" :
              L"2\lambda^*\Lambda_I/(\sigma^*)^2"

    fig = Figure(size = tile_figure_size(length(setting_codes), 2;
                                         width_extra = 110, height_extra = 105))
    middle = cld(length(setting_codes), 2)

    for (column, code) in enumerate(setting_codes)
        item = data[code]
        qq = Axis(fig[1, column]; title = setting_titles[code],
                  xlabel = column == middle ? x_label : "",
                  ylabel = column == 1 ? y_label : "",
                  aspect = AxisAspect(1), width = TILE_WIDTH, height = TILE_HEIGHT,
                  yticklabelsvisible = column == 1, yticksvisible = column == 1)
        lines!(qq, [limits[1], limits[2]], [limits[1], limits[2]];
               color = ink, linewidth = 1.6, linestyle = :dash)
        scatter!(qq, item.theoretical, item.empirical; color = (turquoise, 0.64),
                 strokecolor = (turquoise, 0.24), strokewidth = 0.25, markersize = 4.8)
        limits!(qq, limits..., limits...)

        centres, density, width = pvalue_density(item.pvalues)
        hist = Axis(fig[2, column]; xlabel = column == middle ? L"p\mathrm{-value}" : "",
                    ylabel = column == 1 ? "Density" : "",
                    xticks = ([0.0, 0.25, 0.50, 0.75, 1.0],
                              ["0", ".25", ".50", ".75", "1"]),
                    aspect = AxisAspect(1), width = TILE_WIDTH, height = TILE_HEIGHT,
                    yticklabelsvisible = column == 1, yticksvisible = column == 1)
        barplot!(hist, centres, density; width = 0.92 * width,
                 color = (turquoise, 0.46), strokecolor = (turquoise, 0.82),
                 strokewidth = 0.65)
        hlines!(hist, [1.0]; color = ink, linewidth = 1.5, linestyle = :dash)
        limits!(hist, 0.0, 1.0, 0.0, density_upper)
    end

    colgap!(fig.layout, TILE_COLUMN_GAP)
    rowgap!(fig.layout, TILE_ROW_GAP)
    for column in 1:length(setting_codes)
        colsize!(fig.layout, column, Fixed(TILE_WIDTH))
    end
    rowsize!(fig.layout, 1, Fixed(TILE_HEIGHT))
    rowsize!(fig.layout, 2, Fixed(TILE_HEIGHT))
    resize_to_layout!(fig)
    fig
end

set_theme!(Theme(fontsize = BASE_FONT_SIZE,
                 Figure = (backgroundcolor = paper,),
                 Axis = (backgroundcolor = paper,
                         xlabelsize = AXIS_LABEL_SIZE, ylabelsize = AXIS_LABEL_SIZE,
                         xticklabelsize = TICK_LABEL_SIZE,
                         yticklabelsize = TICK_LABEL_SIZE,
                         titlesize = COLUMN_HEADER_SIZE,
                         xgridcolor = (grid_colour, 0.7),
                         ygridcolor = (grid_colour, 0.7),
                         topspinevisible = false, rightspinevisible = false,
                         leftspinecolor = ink, bottomspinecolor = ink),
                 Legend = (framevisible = false, backgroundcolor = :transparent,
                           labelsize = LEGEND_SIZE)))

mkpath(figures_path)
save(joinpath(figures_path, "05-adjusted-z.pdf"), inference_figure(:z))
save(joinpath(figures_path, "S-plr.pdf"), inference_figure(:plr))
println("wrote the inference figures to ", figures_path, " using n = ", reference_n)
