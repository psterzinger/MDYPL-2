## Philipp Sterzinger 03.09.2026: Code is provided as is, no finished package
## and no guarantees given
##
## Diaconis-Ylvisaker prior penalized likelihood (mDYPL) for logistic regression
## with an intercept and a known prior slope, in the regime p/n -> kappa in
## (0,1). `solve_oracle` and `solve_sloe` solve the limiting first-order
## conditions; `fit_mdypl` is the finite-sample fit.

module MDYPL

using LinearAlgebra
using Statistics
using FastGaussQuadrature: gausshermite, gausslegendre
import NonlinearSolve
import Optim
import SciMLBase

export logistic, softplus, prox_logistic, gauss_hermite_rule

export OracleProblem, SLOEProblem, solve_oracle, solve_sloe
export oracle_focs!, sloe_focs!, oracle_dimension, sloe_dimension
export default_oracle_start, default_sloe_starts

export optimal_alpha, with_alpha
export adjusted_mse, unadjusted_mse, prediction_r2, alpha_criterion, ALPHA_GRID

export mdypl_pseudo_response, fit_mdypl, fit_logistic
export estimate_nu2_sloe, sloe_problem_from_fit

export estimate_unknowns, solve_estimated_focs
export sloe_covariance_status, calibrated_intercept

include("logistic_prox.jl")
include("limiting_focs.jl")
include("alpha_tuning.jl")
include("logistic_fit.jl")
include("estimating_unknowns.jl")

end
