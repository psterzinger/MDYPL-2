# Supplementary material for “Proportional-limit asymptotics for Diaconis–Ylvisaker-penalised logistic regression with fitted intercept”
Philipp Sterzinger
September 7, 2026

This directory accompanies

> Sterzinger P (2026). *Proportional-limit asymptotics for
> Diaconis–Ylvisaker-penalised logistic regression with fitted
> intercept*. [arXiv preprint](https://arxiv.org/pdf/2609.09831)

<!-- The arXiv link will be added when the preprint identifier is available.
A compiled copy of the supplementary material is provided as
[`hdl-supplementary.pdf`](hdl-supplementary.pdf). -->

# Directory structure

| Path | Contents |
|----|----|
| [`code/`](code/) | Julia scripts for the simulations, figures, and tables |
| [`code/methods/`](code/methods/) | estimator, limiting-equation, tuning, plotting, and input/output helpers |
| [`results/`](results/) | Arrow simulation output and generated LaTeX tables |
| [`figures/`](figures/) | generated PDF figures |
| [`test/`](test/) | regression tests for the numerical methods |
| [`hdl-supplementary.pdf`](hdl-supplementary.pdf) | compiled supplementary material |

The estimator is exposed through [`MDYPL.jl`](code/methods/MDYPL.jl),
which includes the following files.

| File | Contents |
|----|----|
| [`logistic_prox.jl`](code/methods/logistic_prox.jl) | logistic link, proximal map of the logistic loss, and Gaussian quadrature |
| [`limiting_focs.jl`](code/methods/limiting_focs.jl) | limiting first-order conditions and their solvers |
| [`alpha_tuning.jl`](code/methods/alpha_tuning.jl) | shrinkage-parameter selection |
| [`logistic_fit.jl`](code/methods/logistic_fit.jl) | finite-sample mDYPL fitting |
| [`estimating_unknowns.jl`](code/methods/estimating_unknowns.jl) | response-moment estimation of unknown population parameters |

[`PublicationPlots.jl`](code/methods/PublicationPlots.jl) defines the
labels, geometry, colours, and typography shared by the figures.
[`simulation_output.jl`](code/methods/simulation_output.jl) reads and
validates the Arrow files written by the simulation.

# Software environment

The results were reproduced with Julia 1.11.7 and the package versions
below.

| Package             |  Version | Used by                          |
|---------------------|---------:|----------------------------------|
| Arrow               |    2.8.1 | simulation and post-processing   |
| CairoMakie          |   0.12.9 | figures                          |
| Colors              |  0.12.11 | figures                          |
| DataFrames          |    1.8.0 | simulation and post-processing   |
| Distributions       | 0.25.122 | numerical methods and tests      |
| FastGaussQuadrature |    1.0.2 | numerical methods                |
| LaTeXStrings        |    1.4.0 | figures                          |
| MathTeXEngine       |    0.6.6 | figures                          |
| NonlinearSolve      |   4.12.0 | limiting equations               |
| Optim               |   1.13.2 | finite-sample fitting and tuning |
| SciMLBase           |  2.120.0 | nonlinear-solver interface       |
| StableRNGs          |    1.0.4 | Monte Carlo simulation           |
| RCall               |   0.14.9 | regression tests only            |

The regression tests also call R 4.6.0 with `brglm2` 1.1.0 and `statmod`
1.5.1. 
<!-- These are the versions used to verify this bundle, rather than -->
<!-- declared minimum versions. -->

<!-- This directory does not contain a Julia `Project.toml` or
`Manifest.toml`. Consequently, the packages must be available in the
active Julia environment, and their versions are not pinned
automatically. -->

# Reproducing the results

Run the commands below from this directory. The scripts derive paths
from their own locations, so they can also be invoked with absolute
paths from another working directory.

## 1. Run the simulation

``` bash
julia code/00-simulation.jl
```

[`00-simulation.jl`](code/00-simulation.jl) must finish before any of
the post-processing scripts are run. It writes the following
intermediate files to [`results/`](results/), for each requested sample
size:

- `setting_<code>_n<nnnn>.arrow` for the adjusted-MSE tuning stream;
- `setting_<code>_unadjusted_n<nnnn>.arrow` for the unadjusted-MSE
  stream;
- `setting_<code>_prediction_n<nnnn>.arrow` for the prediction stream;
- `settings_n<nnnn>.arrow`, `signals_n<nnnn>.arrow`, and
  `unknowns_n<nnnn>.arrow` for shared settings, signal summaries, and
  estimates of unknown population parameters.
<!-- 
The publication configuration uses ten sample sizes from 125 to 4000,
10,000 Monte Carlo replications, five prior settings, and three tuning
streams. It creates 180 Arrow files; the current full result set
occupies about 4.3 GB. The run is computationally expensive.

The script starts 10 Julia worker processes and fixes BLAS to one thread
per process. Change `num_workers` near the top of the script to suit the
available memory and CPU resources. Julia’s [parallel-computing
manual](https://docs.julialang.org/en/v1/manual/parallel-computing/)
describes the `Distributed` execution model used here. -->

## 2. Generate figures and tables

After the simulation has completed, run:

``` bash
julia code/01-intro-plots.jl
julia code/03-convergence-plots.jl
julia code/03-prediction-plots.jl
julia code/04-inference-plots.jl
julia code/04-inference-tables.jl
julia code/06-estimating-unknowns.jl
```

<!-- The six post-processing scripts depend on the simulation output but not
on one another. The table records their generated artefacts by role
rather than by figure or table number, because numbering can change
between manuscript versions and between the combined and standalone
supplements. -->

The table below records their outputs. 

| Script | Generated artefact | Role |
|----|----|----|
| [`01-intro-plots.jl`](code/01-intro-plots.jl) | [`01-beta-truth.pdf`](figures/01-beta-truth.pdf), [`01-qq-classical.pdf`](figures/01-qq-classical.pdf), [`01-intercept-truth.pdf`](figures/01-intercept-truth.pdf) | main-text truth-based comparisons |
|  | [`01-beta.pdf`](figures/01-beta.pdf), [`01-qq.pdf`](figures/01-qq.pdf), [`01-intercept.pdf`](figures/01-intercept.pdf) | main-text plug-in comparisons |
| [`03-convergence-plots.jl`](code/03-convergence-plots.jl) | [`03-slope-mse.pdf`](figures/03-slope-mse.pdf) | main-text slope MSE |
|  | [`S-convergence-all-settings.pdf`](figures/S-convergence-all-settings.pdf) | supplementary convergence results |
| [`03-prediction-plots.jl`](code/03-prediction-plots.jl) | [`04-mer.pdf`](figures/04-mer.pdf) | main-text prediction error |
| [`04-inference-plots.jl`](code/04-inference-plots.jl) | [`05-adjusted-z.pdf`](figures/05-adjusted-z.pdf) | main-text adjusted statistics |
|  | [`S-plr.pdf`](figures/S-plr.pdf) | supplementary penalised likelihood-ratio results |
| [`04-inference-tables.jl`](code/04-inference-tables.jl) | [`05-plr-coverage.tex`](results/05-plr-coverage.tex) | main-text penalised likelihood-ratio coverage |
|  | [`S-adjusted-z-coverage.tex`](results/S-adjusted-z-coverage.tex) | supplementary adjusted-statistic coverage |
|  | [`S-plr-quantiles.tex`](results/S-plr-quantiles.tex), [`S-adjusted-z-quantiles.tex`](results/S-adjusted-z-quantiles.tex) | supplementary quantiles |
| [`06-estimating-unknowns.jl`](code/06-estimating-unknowns.jl) | [`07-response-moment-convergence.pdf`](figures/07-response-moment-convergence.pdf), [`07-state-parameter-histograms-n4000.pdf`](figures/07-state-parameter-histograms-n4000.pdf) | main-text estimation of unknown parameters |
|  | [`S-unknowns-all-parameters-convergence.pdf`](figures/S-unknowns-all-parameters-convergence.pdf), [`S-unknowns-all-parameters-histograms-n4000.pdf`](figures/S-unknowns-all-parameters-histograms-n4000.pdf) | supplementary estimation results |
<!-- 
## Simulation design

The main constants near the top of
[`00-simulation.jl`](code/00-simulation.jl) are

``` julia
n_values = [125, 250, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
kappa = 0.2
reps = 10000
gamma = 1.5
pi0 = 0.2
null_block = 20
plr_block = 10
unknowns_stride = 10
```

The five settings differ in the prior signal strength `delta`, its
correlation `corr` with the true signal, and the prevalence `piP`
implied by the prior.

| Setting | Description                      | `delta` | `corr` | `piP` |
|---------|----------------------------------|--------:|-------:|------:|
| A       | aligned and calibrated           | `gamma` |    0.8 |   0.2 |
| O       | orthogonal and calibrated        | `gamma` |    0.0 |   0.2 |
| I       | aligned with incorrect intercept | `gamma` |    0.8 |   0.8 |
| M       | misleading                       | `gamma` |   −0.8 |   0.8 |
| N       | null                             |       0 |      — |   0.5 |

For each setting, the population-optimal shrinkage parameter is found
under three criteria and the estimator is fitted on the same Monte Carlo
data:

- `adjusted` minimises the limiting MSE of the rescaled slope and
  supplies the estimation, penalised likelihood-ratio, and inference
  results;
- `unadjusted` minimises the limiting MSE of the unadjusted slope and
  supplies the SLOE and response-moment calculations;
- `prediction` maximises the squared correlation between the true and
  fitted limiting logits and supplies the prediction results.

For a smoke run, reduce `n_values` and `reps` in `00-simulation.jl`. If
the reduced grid omits 4000, also change `reference_n = 4000` in
[`01-intro-plots.jl`](code/01-intro-plots.jl). If it supplies fewer than
500 usable fits, reduce that script’s `qq_replications = 500` as well. -->

# Tests

Run the regression suite with

``` bash
julia test/runtests.jl
```

<!-- The suite contains 204 tests. All 204 passed in the software environment
recorded above. -->

<!-- # Maintaining this README

[`README.qmd`](README.qmd) is the source document. Render the
GitHub-facing [`README.md`](README.md) with

``` bash
quarto render README.qmd
``` -->
