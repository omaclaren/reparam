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
include("workflows.jl")

# Export commonly used functions
export construct_lnlike_xy, construct_lnlike_XY, construct_distrib_XY, construct_ϕ_XY
export profile_target
export plot_1D_profile, plot_2D_contour
export execute_model_analysis_workflow_up_to_2D_profiles 
export scale_and_round, reparam

end