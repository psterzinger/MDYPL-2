## Philipp Sterzinger 03.09.2026: Code is provided as is, no finished package
## and no guarantees given
##
## Readers for the Arrow files that 00-simulation.jl writes into `results/`.
## `read_setting` gives one setting's replications, `inference_statistics` the
## two pivotal statistics of the inference section.

module SimulationOutput

using Arrow, DataFrames, Distributions

export sample_sizes, read_setting, read_settings, read_signals, read_unknowns,
       beta_matrix, signal_vector, setting_row, converged_rows,
       inference_statistics

"""Sample sizes the simulation has been run at, in increasing order."""
sample_sizes(results_path) =
    sort([parse(Int, match(r"^settings_n(\d+)\.arrow$", file)[1])
          for file in readdir(results_path)
          if occursin(r"^settings_n\d+\.arrow$", file)])

read_arrow(results_path, stem, n) =
    DataFrame(Arrow.Table(joinpath(results_path,
                                   "$(stem)_n$(lpad(n, 4, '0')).arrow")))

"""
    read_setting(results_path, code, n; tuning = :adjusted)

One setting's replications. `tuning` is `:adjusted`, `:unadjusted` or
`:prediction`.
"""
read_setting(results_path, code, n; tuning = :adjusted) =
    read_arrow(results_path,
               tuning == :adjusted ? "setting_$(code)" : "setting_$(code)_$(tuning)",
               n)

read_settings(results_path, n) = read_arrow(results_path, "settings", n)
read_signals(results_path, n) = read_arrow(results_path, "signals", n)
read_unknowns(results_path, n) = read_arrow(results_path, "unknowns", n)

"""Fitted slopes as a replications-by-p matrix, from the beta_001, ... columns."""
beta_matrix(df) =
    Matrix{Float64}(df[:, sort(filter(name -> occursin(r"^beta_\d+$", name), names(df)))])

"""One of the fixed signal vectors, as stored by the simulation."""
signal_vector(signals, name; setting = "") =
    Float64.(sort(signals[(signals.signal .== name) .&
                          (signals.setting .== setting), :], :coordinate).value)

"""The row of the settings table for one setting code."""
setting_row(settings, code) = only(eachrow(settings[settings.setting .== code, :]))

"""Indices of the replications whose fit converged and is finite."""
converged_rows(df, betas) =
    [i for i in 1:nrow(df) if df.converged[i] && isfinite(df.theta_hat[i]) &&
                              all(isfinite, view(betas, i, :))]

"""
    inference_statistics(results_path, code, n)

The two pivotal statistics of the inference section, for one setting.

  * the adjusted z statistic `(beta_hat_1 - beta_1^*) / sigma^*`, which is
    standard normal in the limit, where `beta_1^* = v1 beta0_1 + v2 betaP_1`
  * the rescaled PLR statistic, which is chi-squared on `plr_block` degrees of
    freedom in the limit

Returns `(; z, z_pvalue, plr, plr_pvalue, replicate, label, plr_df)`, restricted
to the replications where both statistics are finite.
"""
function inference_statistics(results_path, code, n; test_coordinate = 1)
    df = read_setting(results_path, code, n)
    signals = read_signals(results_path, n)
    plr_df = setting_row(read_settings(results_path, n), code).plr_block

    beta_star = df.v1_star[1] * signal_vector(signals, "beta0")[test_coordinate] +
                df.v2_star[1] * signal_vector(signals, "betaP";
                                              setting = code)[test_coordinate]
    z = (df[!, Symbol("beta_", lpad(test_coordinate, 3, '0'))] .- beta_star) ./
        df.sigma_star[1]
    plr = Float64.(df.plr)

    keep = df.converged .& isfinite.(z) .& isfinite.(plr)
    (z = z[keep],
     z_pvalue = 2.0 .* ccdf.(Normal(), abs.(z[keep])),
     plr = plr[keep],
     plr_pvalue = ccdf.(Chisq(plr_df), plr[keep]),
     replicate = df.replicate[keep],
     label = df.setting_label[1],
     plr_df = plr_df)
end

end
