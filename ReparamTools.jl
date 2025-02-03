module ReparamTools

using Distributions
using LinearAlgebra
using SparseArrays
using ForwardDiff
using NLopt
using Plots
using LaTeXStrings
using Measures

# Include component files
include("utils.jl")
include("parameterizations.jl")
include("core.jl")
include("visualization.jl")

# Export commonly used functions
export 
    # Core functionality
    construct_lnlike_xy,
    construct_lnlike_XY,
    construct_distrib_XY,
    construct_ϕ_XY,
    profile_target,
    get_1D_profiles_from_2D,
    construct_ellipse_lnlike_approx,
    construct_upper_lower_profile_wise_CIs_for_mean,
    compute_ϕ_Jacobian,

    # Visualization functions
    plot_1D_profile,
    plot_1D_profile_comparison,
    plot_2D_contour,
    plot_2D_contour_comparison,
    plot_profile_wise_CI_for_mean,

    # Utility functions
    scale_and_round,
    reparam,
    generate_initial_guesses,
    construct_observation_matrix

end # module