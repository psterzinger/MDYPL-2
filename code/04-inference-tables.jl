## Philipp Sterzinger 03.09.2026: Code is provided as is, no finished package
## and no guarantees given
##
## LaTeX quantile and coverage tables for the adjusted z and the rescaled PLR
## statistic. `quantile_table` and `coverage_table` write one table each.

using Distributions, Printf, Statistics

supp_path = abspath(joinpath(@__DIR__, ".."))
results_path = joinpath(supp_path, "results")

include(joinpath(supp_path, "code", "methods", "simulation_output.jl"))
using .SimulationOutput

setting_codes = ["A", "O", "I", "M", "N"]
setting_names = Dict("A" => "Aligned", "O" => "Orthogonal", "I" => "Wrong intercept",
                     "M" => "Misleading", "N" => "Null prior")
coverage_levels = [0.90, 0.95, 0.99]
quantile_probabilities = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

reference_n = last(sample_sizes(results_path))

statistics = Dict(code => inference_statistics(results_path, code, reference_n)
                  for code in setting_codes)
plr_df = statistics[first(setting_codes)].plr_df

common = reduce(intersect, [Set(item.replicate) for item in values(statistics)])
replications = length(common)

function selected(code, statistic)
    item = statistics[code]
    keep = in.(item.replicate, Ref(common))
    (values = getproperty(item, statistic)[keep],
     pvalues = getproperty(item, Symbol(statistic, :_pvalue))[keep])
end

row(cells...) = string(join(cells, " & "), " \\\\")
thousands(k) = replace(string(k), r"(?<=\d)(?=(\d{3})+$)" => "{,}")
fixed(values) = [@sprintf("%.3f", value) for value in values]
setting_row_label(code) = "\$\\mathrm{$code}\$ \$\\cdot\$ $(setting_names[code])"

statistic_tex(statistic) =
    statistic == :z ? "\\bb Z_{1}^{\\mathrm{adj}}(0)" :
                      "2\\lambda^*\\Lambda_I/(\\sigma^*)^2"

function quantile_table(io, statistic)
    reference = statistic == :z ? Normal() : Chisq(plr_df)
    reference_tex = statistic == :z ? "\\mathrm{N}(0,1)" : "\\chi_{$plr_df}^2"
    caption = statistic == :z ?
        "Empirical quantiles of the oracle-adjusted statistic " *
        "\$\\bb Z_{1}^{\\mathrm{adj}}(0)\$ for the first true-null coordinate, " *
        "compared with the corresponding \$\\mathrm N(0,1)\$ quantiles." :
        "Empirical quantiles of the candidate oracle-rescaled penalised " *
        "likelihood-ratio statistic \$2\\lambda^*\\Lambda_I/(\\sigma^*)^2\$, for " *
        "\$I=\\{1,\\ldots,$plr_df\\}\$, compared with the corresponding " *
        "\$\\chi_{$plr_df}^2\$ quantiles."

    println(io, raw"\begin{table}[H]")
    println(io, raw"\centering")
    println(io, raw"\begin{tabular}{@{}lrrrrrrrrr@{}}")
    println(io, raw"\toprule")
    println(io, row("Setting", ["$(round(Int, 100 * q))\\%"
                                for q in quantile_probabilities]...))
    println(io, raw"\midrule")
    println(io, row("\$$reference_tex\$ reference",
                    fixed(quantile.(reference, quantile_probabilities))...))
    for code in setting_codes
        item = selected(code, statistic)
        println(io, row(setting_row_label(code),
                        fixed(quantile(item.values, quantile_probabilities))...))
    end
    println(io, raw"\bottomrule")
    println(io, raw"\end{tabular}")
    println(io, "\\caption{$caption\nResults are based on \$$(thousands(replications))\$ ",
            "replications with \$n=$reference_n\$ and \$p=$(round(Int, 0.2reference_n))\$.}")
    println(io, "\\label{tab:inference-",
            statistic == :z ? "adjusted-z" : "plr", "-quantiles}")
    println(io, raw"\end{table}")
end

function coverage_table(io, statistic)
    caption = statistic == :z ?
        "Empirical coverage of nominal \$90\\%\$, \$95\\%\$, and \$99\\%\$ " *
        "two-sided confidence intervals for the null coordinate " *
        "\$\\beta_{0,1}=0\$, based on the oracle-adjusted statistic " *
        "\$\\bb Z_{1}^{\\mathrm{adj}}(0)\$, along with Monte Carlo standard errors." :
        "Empirical coverage of nominal \$90\\%\$, \$95\\%\$, and \$99\\%\$ " *
        "confidence regions of the oracle-rescaled PLR statistic " *
        "\$2\\lambda^*\\Lambda_I/(\\sigma^*)^2\$ for the null block " *
        "\$I=\\{1,\\ldots,$plr_df\\}\$, against the \$\\chi_{$plr_df}^2\$ " *
        "distribution, along with Monte Carlo standard errors."

    coverage = Dict(code => let item = selected(code, statistic)
                        c = [mean(item.pvalues .>= 1 - level) for level in coverage_levels]
                        (c = c, se = sqrt.(c .* (1 .- c) ./ length(item.pvalues)))
                    end for code in setting_codes)

    println(io, raw"\begin{table}[H]")
    println(io, raw"\centering")
    println(io, raw"\begin{tabular}{@{}lrrrrr@{}}")
    println(io, raw"\toprule")
    println(io, row("Nominal coverage",
                    "\\multicolumn{$(length(setting_codes))}{c}{Setting}"))
    println(io, "\\cmidrule(l){2-$(length(setting_codes) + 1)}")
    println(io, row("", ["`\$\\mathrm{$code}\$'" for code in setting_codes]...))
    println(io, raw"\midrule")
    for (i, level) in enumerate(coverage_levels)
        println(io, row("$(round(Int, 100 * level))\\%",
                        fixed([coverage[code].c[i] for code in setting_codes])...),
                raw"[-2pt]")
        println(io, row("", ["($value)" for value in
                             fixed([coverage[code].se[i] for code in setting_codes])]...))
    end
    println(io, raw"\bottomrule")
    println(io, raw"\end{tabular}")
    println(io, "\\caption{$caption\nResults are based on \$$(thousands(replications))\$ ",
            "replications with \$n=$reference_n\$ and \$p=$(round(Int, 0.2reference_n))\$.}")
    println(io, "\\label{tab:inference-",
            statistic == :z ? "adjusted-z" : "plr", "-coverage}")
    println(io, raw"\end{table}")
end

for (file, table, statistic) in (("05-plr-coverage.tex", coverage_table, :plr),
                                 ("S-adjusted-z-coverage.tex", coverage_table, :z),
                                 ("S-plr-quantiles.tex", quantile_table, :plr),
                                 ("S-adjusted-z-quantiles.tex", quantile_table, :z))
    path = joinpath(results_path, file)
    open(io -> table(io, statistic), path, "w")
    println("wrote ", path)
end
