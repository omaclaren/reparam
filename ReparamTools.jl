module ReparamTools

using Distributions
using LinearAlgebra
using ForwardDiff
using NLopt
using Plots
using LaTeXStrings

# Include component files
include("utils.jl")
include("parameterizations.jl")
include("core.jl")
include("visualization.jl")

# Export commonly used functions
export construct_lnlike_xy, construct_lnlike_XY, construct_distrib_XY, construct_ϕ_XY
export profile_target, get_1D_profiles_from_2D
export construct_ellipse_lnlike_approx, construct_upper_lower_profile_wise_CIs_for_mean
export plot_1D_profile, plot_1D_profile_comparison
export plot_2D_contour, plot_1D_profile_comparison, plot_2D_contour_comparison
export plot_profile_wise_CI_for_mean
export scale_and_round, reparam

end