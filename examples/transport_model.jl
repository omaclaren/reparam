# Include ReparamTools.jl code if not already loaded
if !@isdefined(ReparamTools)
    include("../ReparamTools.jl")
    println("✓ ReparamTools module included")
else
    println("✓ ReparamTools module already included")
end

# Load required packages 
using .ReparamTools
using Plots
using Distributions
using LinearAlgebra
using Random

# Set random seed for reproducibility
Random.seed!(1)

# --------------------------------------------------------
# Model Definition
# --------------------------------------------------------
function solve_model(θ, x, L)
    """
    transport model solution: maps parameters θ = [T₁, T₂, R]
    to solution values on the specified spatial grid. Basis for 
    the ϕ mapping function.
    
    Parameters:
    - θ: Vector θ = [T₁, T₂, R] of transport and source parameters
    - x: Spatial grid points
    - L: Domain length
    
    Returns:
    - Vector of solution values on the spatial grid
    """
    y = Vector{eltype(θ)}(undef, length(x))
    mid_index = Int((length(x)-1)/2)
    
    # Compute solution parameters
    β = L^2*θ[3]/8*(1/θ[2]-1/θ[1])
    α = θ[3]*L/(2*θ[2])-β/L
    
    # Solution in each region
    H_1(x) = -θ[3]/(2*θ[1])*x^2 + α*x
    H_2(x) = -θ[3]/(2*θ[2])*x^2 + α*x + β
    
    # Evaluate solution in each region
    for i in 1:mid_index 
        y[i] = H_1(x[i])
    end 
    for i in mid_index:length(x) 
        y[i] = H_2(x[i])
    end 
    return y
end

# Creates a ϕ_mapping function with fixed grid parameters
function create_ϕ_mapping(x, L)
    """
    Creates a mapping function from parameters to solution values
    with fixed grid parameters.
    
    Parameters:
    - x: Spatial grid points
    - L: Domain length
    
    Returns:
    - Function mapping θ to solution values
    """
    return θ -> solve_model(θ, x, L)
end

# --------------------------------------------------------
# Setup 
# --------------------------------------------------------

# Fine grid setup
L = 100
x = LinRange(0, L, 201)
indices_fine = 1:length(x)

# Observation grid and matrix
obs_step_size = 10
indices_obs = 0+obs_step_size:obs_step_size:length(x)-obs_step_size
obs_matrix = construct_observation_matrix(indices_obs, indices_fine)
# Option 1. Use matrix multiplication to get the observation points
# x_obs = obs_matrix * x
# Option 2. Use the observation indices directly
x_obs = x[indices_obs]

# --------------------------------------------------------
# Original Parameterization Analysis and Data Generation
# Define xy = θ = [T₁, T₂, R]
# --------------------------------------------------------

# Define ϕ mapping function for fine grid
ϕ_func_xy = create_ϕ_mapping(x, L)

# Distribution in original parameterization on observation grid
σ = 0.2
# Option 1. Use matrix multiplication to get the observation points
# distrib_xy = xy -> MvLogNormal(log.(abs.(obs_matrix*ϕ_func_xy(xy))), σ^2*I(length(x_obs)))
# Option 2. Use the observation indices directly
distrib_xy = xy -> MvLogNormal(log.(abs.(ϕ_func_xy(xy)[indices_obs])), σ^2*I(length(x_obs)))

# Distribution in original parameterization on fine grid
distrib_fine_xy = xy -> MvLogNormal(log.(abs.(ϕ_func_xy(xy))), σ^2*I(length(x)))

# Variables and bounds
varnames = Dict("ψ1" => "T_1", "ψ2" => "T_2", "ψ3" => "R")
varnames["ψ1_save"] = "T_1"
varnames["ψ2_save"] = "T_2"
varnames["ψ3_save"] = "R"

# Parameter bounds
T_1_min, T_1_max = 0.1, 5.0
T_2_min, T_2_max = 0.1, 5.0
R_min, R_max = 0.1, 5.0

xy_lower_bounds = [T_1_min, T_2_min, R_min]
xy_upper_bounds = [T_1_max, T_2_max, R_max]

# Initial guess for optimisation
xy_initial = 0.5 * (xy_lower_bounds + xy_upper_bounds)

# True parameter
T_1_true, T_2_true, R_true = 3.0, 1.0, 1.0
xy_true = [T_1_true, T_2_true, R_true]

# Generate data
Nrep = 1
data = rand(distrib_xy(xy_true), Nrep)

# Visualize data
scatter(x_obs, data)

# Construct likelihood
lnlike_xy = construct_lnlike_xy(distrib_xy, data; dist_type=:multi)
model_name = "transport_xy"
print(model_name*"\n")

grid_steps = [500]
dim_all = length(xy_initial)
indices_all = 1:dim_all

# Point estimation (MLE)
point_estimation_method = :LN_BOBYQA
target_indices = []  # empty for MLE
n_guesses = 3
# Generate multiple initial guesses
nuisance_guesses = generate_initial_guesses(xy_lower_bounds, xy_upper_bounds, n_guesses)

xy_MLE, lnlike_xy_MLE = profile_target(lnlike_xy, target_indices,
    xy_lower_bounds, xy_upper_bounds, 
    xy_initial; grid_steps=grid_steps, ω_initial_extras=nuisance_guesses,
    method=point_estimation_method)

# Quadratic approximation at MLE
lnlike_xy_ellipse, H_xy_ellipse = construct_ellipse_lnlike_approx(lnlike_xy, xy_MLE)

# Eigenanalysis of Hessian at MLE
evals, evecs = eigen(H_xy_ellipse; sortby = x -> -real(x))
println("Eigenvectors and eigenvalues for "*model_name)
println("Eigenvalues: ", evals)
println("Eigenvectors: ", evecs)

# Jacobian and svd analysis of φ mapping at MLE
# J_ϕ_xy = ForwardDiff.jacobian(ϕ_func_xy, xy_MLE)
J_ϕ_xy, U_xy, S_xy, Vt_xy = compute_ϕ_Jacobian(ϕ_func_xy, xy_MLE; method_type=:auto, compute_svd=true)
# print singular values and left/right singular vectors
println("Singular values for "*model_name)
println(S_xy)
# println("Left singular vectors for "*model_name)
# println(U_xy)
println("Right singular vectors for "*model_name)
println(Vt_xy)

# Calculate prediction at MLE for reference distribution on fine grid
pred_mean_MLE = mean(distrib_fine_xy(xy_MLE))
true_mean = mean(distrib_fine_xy(xy_true))

# 1D Profiles
profile_method = :LN_BOBYQA
for i in 1:dim_all
    target_index = i
    nuisance_indices = setdiff(indices_all, target_index)
    nuisance_guess = xy_MLE[nuisance_indices]
    # nuisance_guess = xy_initial[nuisance_indices] # can use xy_initial as well

    # generate multiple initial guesses
    n_guesses_profiling = 3
    nuisance_guess_extras = generate_initial_guesses(xy_lower_bounds[nuisance_indices], xy_upper_bounds[nuisance_indices], n_guesses_profiling)

    # Print variable name
    print("Variable: ", varnames["ψ"*string(i)], "\n")

    # Profile full likelihood
    ψω_values, lnlike_ψ_values = profile_target(lnlike_xy, 
        target_index,
        xy_lower_bounds, 
        xy_upper_bounds,
        nuisance_guess; 
        grid_steps=grid_steps,
        ω_initial_extras=nuisance_guess_extras,
        method=profile_method)

    # Profile quadratic approximation
    ψω_ellipse_values, lnlike_ψ_ellipse_values = profile_target(lnlike_xy_ellipse,
        target_index,
        xy_lower_bounds,
        xy_upper_bounds,
        nuisance_guess;
        grid_steps=grid_steps,
        ω_initial_extras=nuisance_guess_extras,
        method=profile_method)

    # Extract profiled parameter values
    ψ_values = [ψω[target_index] for ψω in ψω_values]
    ψ_ellipse_values = [ψω[target_index] for ψω in ψω_ellipse_values]

    # Plot profiles
    plot_1D_profile(model_name, ψ_values, lnlike_ψ_values,
        varnames["ψ"*string(i)];
        varname_save=varnames["ψ"*string(i)*"_save"],
        ψ_true=xy_true[i])

    plot_1D_profile_comparison(model_name, model_name*"_ellipse",
        ψ_values, ψ_ellipse_values,
        lnlike_ψ_values, lnlike_ψ_ellipse_values,
        varnames["ψ"*string(i)];
        varname_save=varnames["ψ"*string(i)*"_save"],
        ψ_true=xy_true[i])

    # Prediction CIs. Use fine grid distribution for prediction
    lower_ψ, upper_ψ, _ = construct_upper_lower_profile_wise_CIs_for_mean(
        distrib_fine_xy, ψω_values, lnlike_ψ_values; l_level=95, df=1)

    plot_profile_wise_CI_for_mean(
        x, lower_ψ, upper_ψ, pred_mean_MLE,
        model_name, "x", "x", data_indep=x_obs,
        data_dep=data, true_mean=true_mean,
        target=varnames["ψ"*string(i)])
end

# 2D Profiles
param_pairs = [(i,j) for i in 1:dim_all for j in (i+1):dim_all]
profile_method = :LN_BOBYQA

for (i,j) in param_pairs
    target_indices_ij = [i,j]
    nuisance_indices = setdiff(indices_all, target_indices_ij)
    nuisance_guess = xy_MLE[nuisance_indices]
    # nuisance_guess = xy_initial[nuisance_indices] # can use xy_initial as well

    # generate multiple initial guesses
    if length(nuisance_indices) > 0
        n_guesses_profiling = 3
        nuisance_guess_extras = generate_initial_guesses(xy_lower_bounds[nuisance_indices], xy_upper_bounds[nuisance_indices], n_guesses_profiling)
    else
        nuisance_guess_extras = nothing
    end

    # extract true values for plotting
    ψ_true_pair = xy_true[target_indices_ij]

    # Print variable names for this pair
    print("Pair: ", varnames["ψ"*string(i)], ", ", varnames["ψ"*string(j)], "\n")

    # Create a copy of varnames for this iteration
    current_varnames = deepcopy(varnames)

    # Update varnames for this pair
    current_varnames["ψ1"] = varnames["ψ"*string(i)]
    current_varnames["ψ2"] = varnames["ψ"*string(j)]
    current_varnames["ψ1_save"] = varnames["ψ"*string(i)*"_save"]
    current_varnames["ψ2_save"] = varnames["ψ"*string(j)*"_save"]

    # Profile full likelihood
    ψω_values, lnlike_ψ_values = profile_target(lnlike_xy, target_indices_ij,
        xy_lower_bounds, xy_upper_bounds,
        nuisance_guess; grid_steps=grid_steps,
        ω_initial_extras=nuisance_guess_extras,
        method=profile_method)

    # Profile quadratic approximation
    ψω_ellipse_values, lnlike_ψ_ellipse_values = profile_target(lnlike_xy_ellipse,
        target_indices_ij,
        xy_lower_bounds,
        xy_upper_bounds,
        nuisance_guess;
        grid_steps=grid_steps,
        ω_initial_extras=nuisance_guess_extras,
        method=profile_method)

    # Extract profiled parameter values
    ψ_values = [ψω[target_indices_ij] for ψω in ψω_values]
    ψ_ellipse_values = [ψω[target_indices_ij] for ψω in ψω_ellipse_values]

    # Plot contours
    plot_2D_contour(model_name, ψ_values, lnlike_ψ_values,
        current_varnames; ψ_true=ψ_true_pair)

    # Plot comparison with quadratic approximation
    plot_2D_contour_comparison(model_name, model_name*"_ellipse",
        ψ_values, ψ_ellipse_values,
        lnlike_ψ_values, lnlike_ψ_ellipse_values,
        current_varnames; ψ_true=ψ_true_pair)

    # Get and plot 1D profiles from 2D grid
    ψ1_values, ψ2_values, like_ψ1_values, like_ψ2_values = get_1D_profiles_from_2D(ψ_values, lnlike_ψ_values)

    plot_1D_profile(model_name, ψ1_values, log.(like_ψ1_values),
        current_varnames["ψ1"];
        varname_save=current_varnames["ψ1_save"]*"_from_2D",
        ψ_true=ψ_true_pair[1])

    plot_1D_profile(model_name, ψ2_values, log.(like_ψ2_values),
        current_varnames["ψ2"];
        varname_save=current_varnames["ψ2_save"]*"_from_2D",
        ψ_true=ψ_true_pair[2])

    # Calculate 2D prediction CIs using distribution on fine grid
    lower_ψ1ψ2, upper_ψ1ψ2, _ = construct_upper_lower_profile_wise_CIs_for_mean(
        distrib_fine_xy, ψω_values, lnlike_ψ_values; l_level=95, df=2)

    plot_profile_wise_CI_for_mean(
        x, lower_ψ1ψ2, upper_ψ1ψ2, pred_mean_MLE,
        model_name, "x", "x",
        data_indep=x_obs, data_dep=data,
        true_mean=true_mean,
        target=current_varnames["ψ1"]*", "*current_varnames["ψ2"])
end

# --------------------------------------------------------
# Log Parameterization Analysis
# --------------------------------------------------------
model_name = "transport_log"
print(model_name*"\n")

# Coordinate transformation
xytoXY_log(xy) = log.(xy)
XYtoxy_log(XY) = exp.(XY)

# Transform bounds and parameters
XY_log_lower_bounds = log.(xy_lower_bounds)
XY_log_upper_bounds = log.(xy_upper_bounds)
XY_log_initial = xytoXY_log(xy_initial)
XY_log_true = xytoXY_log(xy_true)

# Transform likelihood, distributions (obs and fine) and ϕ mapping
lnlike_XY_log = construct_lnlike_XY(lnlike_xy, XYtoxy_log)
distrib_XY_log = construct_distrib_XY(distrib_xy, XYtoxy_log)
distrib_fine_XY_log = construct_distrib_XY(distrib_fine_xy, XYtoxy_log)
ϕ_func_XY_log = construct_ϕ_XY(ϕ_func_xy, XYtoxy_log)

# Update variable names for log coordinates
varnames["ψ1"] = "\\ln\\ T_1"
varnames["ψ2"] = "\\ln\\ T_2"
varnames["ψ3"] = "\\ln\\ R"
varnames["ψ1_save"] = "ln_T_1"
varnames["ψ2_save"] = "ln_T_2"
varnames["ψ3_save"] = "ln_R"

# Point estimation in log coordinates
point_estimation_method = :LN_BOBYQA
target_indices = []  # empty for MLE
# Generate multiple initial guesses
n_guesses = 3
nuisance_guesses = generate_initial_guesses(XY_log_lower_bounds, XY_log_upper_bounds, n_guesses)

XY_log_MLE, lnlike_XY_log_MLE = profile_target(lnlike_XY_log, target_indices,
    XY_log_lower_bounds, XY_log_upper_bounds, 
    XY_log_initial; grid_steps=grid_steps,
    ω_initial_extras=nuisance_guesses,
    method=point_estimation_method)

# Quadratic approximation at MLE
lnlike_XY_log_ellipse, H_XY_log_ellipse = construct_ellipse_lnlike_approx(lnlike_XY_log, XY_log_MLE)

# Eigenanalysis in log coordinates
evals_log, evecs_log = eigen(H_XY_log_ellipse; sortby = x -> -real(x))
println("Eigenvectors and eigenvalues for "*model_name)
println("Eigenvalues: ", evals_log)
println("Eigenvectors: ", evecs_log)

# Jacobian and svd analysis of φ mapping at MLE
# J_ϕ_XY_log = ForwardDiff.jacobian(ϕ_func_XY_log, XY_log_MLE)
J_ϕ_XY_log, U_XY_log, S_XY_log, Vt_XY_log = compute_ϕ_Jacobian(ϕ_func_XY_log, XY_log_MLE; method_type=:auto, compute_svd=true)
# print singular values and left/right singular vectors
println("Singular values for "*model_name)
println(S_XY_log)
# println("Left singular vectors for "*model_name)
# println(U_XY_log)
println("Right singular vectors for "*model_name)
println(Vt_XY_log)

# Calculate prediction at MLE for reference using distribution on fine grid
pred_mean_MLE_log = mean(distrib_fine_XY_log(XY_log_MLE))
true_mean_log = mean(distrib_fine_XY_log(XY_log_true))

# 1D Profiles in log coordinates
profile_method = :LN_BOBYQA
for i in 1:dim_all
    target_index = i
    nuisance_indices = setdiff(indices_all, target_index)
    nuisance_guess = XY_log_MLE[nuisance_indices]
    # nuisance_guess = XY_log_initial[nuisance_indices] # can use XY_log_initial as well

    # generate multiple initial guesses
    n_guesses_profiling = 3
    nuisance_guess_extras = generate_initial_guesses(XY_log_lower_bounds[nuisance_indices], XY_log_upper_bounds[nuisance_indices], n_guesses_profiling)

    # Profile full likelihood
    ψω_values, lnlike_ψ_values = profile_target(lnlike_XY_log, target_index,
        XY_log_lower_bounds, XY_log_upper_bounds,
        nuisance_guess; grid_steps=grid_steps,
        ω_initial_extras=nuisance_guess_extras,
        method=profile_method)

    # Profile quadratic approximation
    ψω_ellipse_values, lnlike_ψ_ellipse_values = profile_target(lnlike_XY_log_ellipse,
        target_index,
        XY_log_lower_bounds,
        XY_log_upper_bounds,
        nuisance_guess;
        grid_steps=grid_steps,
        ω_initial_extras=nuisance_guess_extras,
        method=profile_method)

    # Extract profiled parameter values
    ψ_values = [ψω[target_index] for ψω in ψω_values]
    ψ_ellipse_values = [ψω[target_index] for ψω in ψω_ellipse_values]

    # Plot profiles
    plot_1D_profile(model_name, ψ_values, lnlike_ψ_values,
        varnames["ψ"*string(i)];
        varname_save=varnames["ψ"*string(i)*"_save"],
        ψ_true=XY_log_true[i])

    plot_1D_profile_comparison(model_name, model_name*"_ellipse",
        ψ_values, ψ_ellipse_values,
        lnlike_ψ_values, lnlike_ψ_ellipse_values,
        varnames["ψ"*string(i)];
        varname_save=varnames["ψ"*string(i)*"_save"],
        ψ_true=XY_log_true[i])

    # Prediction CIs using distribution on fine grid
    lower_ψ, upper_ψ, _ = construct_upper_lower_profile_wise_CIs_for_mean(
        distrib_fine_XY_log, ψω_values, lnlike_ψ_values; l_level=95, df=1)

    plot_profile_wise_CI_for_mean(
        x, lower_ψ, upper_ψ, pred_mean_MLE_log,
        model_name, "x", "x", data_indep=x_obs,
        data_dep=data, true_mean=true_mean_log,
        target=varnames["ψ"*string(i)])
end

# 2D Profiles in log coordinates
profile_method = :LN_BOBYQA
param_pairs = [(i,j) for i in 1:dim_all for j in (i+1):dim_all]

for (i,j) in param_pairs
    target_indices_ij = [i,j]
    nuisance_indices = setdiff(indices_all, target_indices_ij)
    nuisance_guess = XY_log_MLE[nuisance_indices]
    # nuisance_guess = XY_log_initial[nuisance_indices] # can use XY_log_initial as well

    # generate multiple initial guesses
    if length(nuisance_indices) > 0
        n_guesses_profiling = 3
        nuisance_guess_extras = generate_initial_guesses(XY_log_lower_bounds[nuisance_indices], XY_log_upper_bounds[nuisance_indices], n_guesses_profiling)
    else
        nuisance_guess_extras = nothing
    end

    ψ_true_pair = XY_log_true[target_indices_ij]

    # Create a copy of varnames for this iteration
    current_varnames = deepcopy(varnames)

    # Update varnames for this pair
    current_varnames["ψ1"] = varnames["ψ"*string(i)]
    current_varnames["ψ2"] = varnames["ψ"*string(j)]
    current_varnames["ψ1_save"] = varnames["ψ"*string(i)*"_save"]
    current_varnames["ψ2_save"] = varnames["ψ"*string(j)*"_save"]

    # Profile full likelihood
    ψω_values, lnlike_ψ_values = profile_target(lnlike_XY_log, target_indices_ij,
        XY_log_lower_bounds, XY_log_upper_bounds,
        nuisance_guess; grid_steps=grid_steps,
        ω_initial_extras=nuisance_guess_extras,
        method=profile_method)

    # Profile quadratic approximation
    ψω_ellipse_values, lnlike_ψ_ellipse_values = profile_target(lnlike_XY_log_ellipse,
        target_indices_ij,
        XY_log_lower_bounds,
        XY_log_upper_bounds,
        nuisance_guess;
        grid_steps=grid_steps,
        ω_initial_extras=nuisance_guess_extras,
        method=profile_method)

    # Extract profiled parameter values
    ψ_values = [ψω[target_indices_ij] for ψω in ψω_values]
    ψ_ellipse_values = [ψω[target_indices_ij] for ψω in ψω_ellipse_values]

    # Plot contours
    plot_2D_contour(model_name, ψ_values, lnlike_ψ_values,
        current_varnames; ψ_true=ψ_true_pair)

    # Plot comparison with quadratic approximation
    plot_2D_contour_comparison(model_name, model_name*"_ellipse",
        ψ_values, ψ_ellipse_values,
        lnlike_ψ_values, lnlike_ψ_ellipse_values,
        current_varnames; ψ_true=ψ_true_pair)

    # Get and plot 1D profiles from 2D grid
    ψ1_values, ψ2_values, like_ψ1_values, like_ψ2_values = get_1D_profiles_from_2D(ψ_values, lnlike_ψ_values)

    plot_1D_profile(model_name, ψ1_values, log.(like_ψ1_values),
        current_varnames["ψ1"];
        varname_save=current_varnames["ψ1_save"]*"_from_2D",
        ψ_true=ψ_true_pair[1])

    plot_1D_profile(model_name, ψ2_values, log.(like_ψ2_values),
        current_varnames["ψ2"];
        varname_save=current_varnames["ψ2_save"]*"_from_2D",
        ψ_true=ψ_true_pair[2])

    # 2D prediction CIs using distribution on fine grid
    lower_ψ1ψ2, upper_ψ1ψ2, _ = construct_upper_lower_profile_wise_CIs_for_mean(
        distrib_fine_XY_log, ψω_values, lnlike_ψ_values; l_level=95, df=2)

    plot_profile_wise_CI_for_mean(
        x, lower_ψ1ψ2, upper_ψ1ψ2, pred_mean_MLE_log,
        model_name, "x", "x",
        data_indep=x_obs, data_dep=data,
        true_mean=true_mean_log,
        target=current_varnames["ψ1"]*", "*current_varnames["ψ2"])
end

# --------------------------------------------------------
# Sloppihood-Informed Parameterization Analysis
# --------------------------------------------------------
model_name = "transport_iir"
print(model_name*"\n")

# Scale and round eigenvectors for iir transformation
# Option 1. based on the eigenvalues and eigenvectors from the log parameterization
# Option 2. based on the right singular vectors from the log parameterization
use_singular_vectors = true
if use_singular_vectors
    evecs_scaled = scale_and_round(Vt_XY_log; round_within=0.5, column_scales=[1,1,1])
else
    evecs_scaled = scale_and_round(evecs_log; round_within=0.5, column_scales=[1,1,1])
end
println("Transformations:")
display(evecs_scaled)
display(inv(evecs_scaled))

println("Original right singular vectors:")
display(Vt_XY_log)

# Construct transformation
xytoXY_iir, XYtoxy_iir = reparam(evecs_scaled)

# Transform likelihood, distributions (obs and fine) and ϕ mapping
lnlike_XY_iir = construct_lnlike_XY(lnlike_xy, XYtoxy_iir)
distrib_XY_iir = construct_distrib_XY(distrib_xy, XYtoxy_iir)
distrib_fine_XY_iir = construct_distrib_XY(distrib_fine_xy, XYtoxy_iir)
ϕ_func_XY_iir = construct_ϕ_XY(ϕ_func_xy, XYtoxy_iir)

# Set bounds for iir coordinates (manual due to non-monotonic transform)
XY_iir_lower_bounds = [0.5, 0.0001, 0.00001]
XY_iir_upper_bounds = [1.5, 10, 1000]
XY_iir_initial = [1.0, 1.0, 10.0]
XY_iir_true = xytoXY_iir(xy_true)

# Update variable names for iir coordinates
if use_singular_vectors
    varnames["ψ1"] = "\\frac{T_2}{R}"
    varnames["ψ2"] = "\\frac{T_1}{\\sqrt{T_2R}}"
    varnames["ψ3"] = "T_1 T_2 R"
    varnames["ψ1_save"] = "T_2_over_R"
    varnames["ψ2_save"] = "T_1_over_sqrt_T_2_R"
    varnames["ψ3_save"] = "T_1_T_2_R"
else
    varnames["ψ1"] = "\\frac{T_2}{R}"
    varnames["ψ2"] = "\\frac{T_1}{\\sqrt{T_2R}}"
    varnames["ψ3"] = "T_1 T_2 R"
    varnames["ψ1_save"] = "T_2_over_R"*"_approx"
    varnames["ψ2_save"] = "T_1_over_sqrt_T_2_R"*"_approx"
    varnames["ψ3_save"] = "T_1_T_2_R"*"_approx"
end

# Point estimation in iir coordinates
target_indices = []  # empty for MLE
n_guesses = 3
# Generate multiple initial guesses
nuisance_guesses = generate_initial_guesses(XY_iir_lower_bounds, XY_iir_upper_bounds, n_guesses)

XY_iir_MLE, lnlike_XY_iir_MLE = profile_target(lnlike_XY_iir, target_indices,
    XY_iir_lower_bounds, XY_iir_upper_bounds, 
    XY_iir_initial; grid_steps=grid_steps,
    ω_initial_extras=nuisance_guesses,
    method=point_estimation_method)

# Quadratic approximation at MLE
lnlike_XY_iir_ellipse, H_XY_iir_ellipse = construct_ellipse_lnlike_approx(lnlike_XY_iir, XY_iir_MLE)

# Eigenanalysis in iir coordinates
evals_iir, evecs_iir = eigen(H_XY_iir_ellipse; sortby = x -> -real(x))
println("Eigenvectors and eigenvalues for "*model_name)
println("Eigenvalues: ", evals_iir)
println("Eigenvectors: ", evecs_iir)

# Jacobian and svd analysis of φ mapping at MLE
# J_ϕ_XY_iir = ForwardDiff.jacobian(ϕ_func_XY_iir, XY_iir_MLE)
J_ϕ_XY_iir, U_XY_iir, S_XY_iir, Vt_XY_iir = compute_ϕ_Jacobian(ϕ_func_XY_iir, XY_iir_MLE; method_type=:auto, compute_svd=true)
# print singular values and left/right singular vectors
println("Singular values for "*model_name)
println(S_XY_iir)
# println("Left singular vectors for "*model_name)
# println(U_XY_iir)
println("Right singular vectors for "*model_name)
println(Vt_XY_iir)

# Calculate prediction at MLE for reference using distribution on fine grid
pred_mean_MLE_iir = mean(distrib_fine_XY_iir(XY_iir_MLE))
true_mean_iir = mean(distrib_fine_XY_iir(XY_iir_true))

# 1D Profiles in iir coordinates
profile_method = :LN_BOBYQA
for i in 1:dim_all
    target_index = i
    nuisance_indices = setdiff(indices_all, target_index)
    nuisance_guess = XY_iir_MLE[nuisance_indices]
    # nuisance_guess = XY_iir_initial[nuisance_indices] # can use XY_iir_initial as well

    # generate multiple initial guesses
    n_guesses_profiling = 3
    nuisance_guess_extras = generate_initial_guesses(XY_iir_lower_bounds[nuisance_indices], XY_iir_upper_bounds[nuisance_indices], n_guesses_profiling)

    # Profile full likelihood
    ψω_values, lnlike_ψ_values = profile_target(lnlike_XY_iir, target_index,
        XY_iir_lower_bounds, XY_iir_upper_bounds,
        nuisance_guess; grid_steps=grid_steps,
        ω_initial_extras=nuisance_guess_extras,
        method=profile_method)

    # Profile quadratic approximation
    ψω_ellipse_values, lnlike_ψ_ellipse_values = profile_target(lnlike_XY_iir_ellipse,
        target_index,
        XY_iir_lower_bounds,
        XY_iir_upper_bounds,
        nuisance_guess;
        grid_steps=grid_steps,
        ω_initial_extras=nuisance_guess_extras,
        method=profile_method)

    # Extract profiled parameter values
    ψ_values = [ψω[target_index] for ψω in ψω_values]
    ψ_ellipse_values = [ψω[target_index] for ψω in ψω_ellipse_values]

    # Plot profiles
    plot_1D_profile(model_name, ψ_values, lnlike_ψ_values,
        varnames["ψ"*string(i)];
        varname_save=varnames["ψ"*string(i)*"_save"],
        ψ_true=XY_iir_true[i])

    plot_1D_profile_comparison(model_name, model_name*"_ellipse",
        ψ_values, ψ_ellipse_values,
        lnlike_ψ_values, lnlike_ψ_ellipse_values,
        varnames["ψ"*string(i)];
        varname_save=varnames["ψ"*string(i)*"_save"],
        ψ_true=XY_iir_true[i])

    # Prediction CIs using distribution on fine grid
    lower_ψ, upper_ψ, _ = construct_upper_lower_profile_wise_CIs_for_mean(
        distrib_fine_XY_iir, ψω_values, lnlike_ψ_values; l_level=95, df=1)

    plot_profile_wise_CI_for_mean(
        x, lower_ψ, upper_ψ, pred_mean_MLE_iir,
        model_name, "x", "x", data_indep=x_obs,
        data_dep=data, true_mean=true_mean_iir,
        target=varnames["ψ"*string(i)])
end

# 2D Profiles in iir coordinates
profile_method = :LN_BOBYQA
param_pairs = [(i,j) for i in 1:dim_all for j in (i+1):dim_all]

for (i,j) in param_pairs
    target_indices_ij = [i,j]
    nuisance_indices = setdiff(indices_all, target_indices_ij)
    nuisance_guess = XY_iir_MLE[nuisance_indices]
    # nuisance_guess = XY_iir_initial[nuisance_indices] # can use XY_iir_initial as well

    # generate multiple initial guesses
    if length(nuisance_indices) > 0
        n_guesses_profiling = 3
        nuisance_guess_extras = generate_initial_guesses(XY_iir_lower_bounds[nuisance_indices], XY_iir_upper_bounds[nuisance_indices], n_guesses_profiling)
    else
        nuisance_guess_extras = nothing
    end

    ψ_true_pair = XY_iir_true[target_indices_ij]

    # Create a copy of varnames for this iteration
    current_varnames = deepcopy(varnames)

    # Update varnames for this pair
    current_varnames["ψ1"] = varnames["ψ"*string(i)]
    current_varnames["ψ2"] = varnames["ψ"*string(j)]
    current_varnames["ψ1_save"] = varnames["ψ"*string(i)*"_save"]
    current_varnames["ψ2_save"] = varnames["ψ"*string(j)*"_save"]

    # Profile full likelihood
    ψω_values, lnlike_ψ_values = profile_target(lnlike_XY_iir, target_indices_ij,
        XY_iir_lower_bounds, XY_iir_upper_bounds,
        nuisance_guess; grid_steps=grid_steps,
        ω_initial_extras=nuisance_guess_extras,
        method=profile_method)

    # Profile quadratic approximation
    ψω_ellipse_values, lnlike_ψ_ellipse_values = profile_target(lnlike_XY_iir_ellipse,
        target_indices_ij,
        XY_iir_lower_bounds,
        XY_iir_upper_bounds,
        nuisance_guess;
        grid_steps=grid_steps,
        ω_initial_extras=nuisance_guess_extras,
        method=profile_method)

    # Extract profiled parameter values
    ψ_values = [ψω[target_indices_ij] for ψω in ψω_values]
    ψ_ellipse_values = [ψω[target_indices_ij] for ψω in ψω_ellipse_values]

    # Plot contours
    plot_2D_contour(model_name, ψ_values, lnlike_ψ_values,
        current_varnames; ψ_true=ψ_true_pair)

    # Plot comparison with quadratic approximation
    plot_2D_contour_comparison(model_name, model_name*"_ellipse",
        ψ_values, ψ_ellipse_values,
        lnlike_ψ_values, lnlike_ψ_ellipse_values,
        current_varnames; ψ_true=ψ_true_pair)

    # Get and plot 1D profiles from 2D grid
    ψ1_values, ψ2_values, like_ψ1_values, like_ψ2_values = get_1D_profiles_from_2D(ψ_values, lnlike_ψ_values)

    plot_1D_profile(model_name, ψ1_values, log.(like_ψ1_values),
        current_varnames["ψ1"];
        varname_save=current_varnames["ψ1_save"]*"_from_2D",
        ψ_true=ψ_true_pair[1])

    plot_1D_profile(model_name, ψ2_values, log.(like_ψ2_values),
        current_varnames["ψ2"];
        varname_save=current_varnames["ψ2_save"]*"_from_2D",
        ψ_true=ψ_true_pair[2])

    # 2D prediction CIs using distribution on fine grid
    lower_ψ1ψ2, upper_ψ1ψ2, _ = construct_upper_lower_profile_wise_CIs_for_mean(
        distrib_fine_XY_iir, ψω_values, lnlike_ψ_values; l_level=95, df=2)

    plot_profile_wise_CI_for_mean(
        x, lower_ψ1ψ2, upper_ψ1ψ2, pred_mean_MLE_iir,
        model_name, "x", "x",
        data_indep=x_obs, data_dep=data,
        true_mean=true_mean_iir,
        target=current_varnames["ψ1"]*", "*current_varnames["ψ2"])
end