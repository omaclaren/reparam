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
using DifferentialEquations

# Set random seed for reproducibility
Random.seed!(321)

# --------------------------------------------------------
# Model Definition
# --------------------------------------------------------

# Define the Monod model ODE 
function DE!(dC, C, θ, t)
    """
    Monod model ODE definition.
    
    Parameters:
    - dC: Rate of change vector (modified in-place)
    - C: Current state vector
    - θ: Parameter vector 
    - t: Current time
    """
    dC[1] = -θ[1]*C[1]/(θ[2] + C[1])
end

# ODE model solver  
function solve_ode(t_save, θ; solver=Rodas4())
    """
    Monod model solution: maps parameters θ to solution values 
    on the specified time grid.
    
    Parameters:
    - θ: Parameter vector
    - t: Time grid points
    - solver: ODE solver (default: Rodas4())
    
    Returns:
    - Vector of solution values on the time grid
    """
    tspan = (0.0, maximum(t_save))
    prob = ODEProblem(DE!, [1.0], tspan, θ)
    sol = solve(prob, solver, saveat=t_save, abstol=1e-12, reltol=1e-9)
    return sol[1, :]
end

# Creates a ϕ mapping function with fixed grid parameters
function create_ϕ_mapping(t)
    """
    Create a ϕ mapping function from model parameters to solution values
    with fixed grid parameters.
    
    Parameters:
    - t: Time grid points
    
    Returns:
    - ϕ mapping function from θ to solution values
    """
    return θ -> solve_ode(t, θ)
end

# --------------------------------------------------------
# Setup and Data Generation
# --------------------------------------------------------

# Fine grid setup
T = 20
NT = 201
t = LinRange(0, T, NT)
indices_fine = 1:NT

# Observation grid setup
NT_obs = 11
indices_obs = 1:Int((NT-1)/(NT_obs-1)):NT
obs_matrix = construct_observation_matrix(indices_obs, indices_fine)
t_obs = t[indices_obs]

# Initial condition and observation parameters
C0 = 0.5
σ = 0.05

# --------------------------------------------------------
# --- Analysis in original parameterisation and data generation ---
# --------------------------------------------------------
# Define ϕ mapping in original coordinates on fine grid
ϕ_func_xy = create_ϕ_mapping(t)

# Parameter -> data distribution (forward) mapping on fine grid
solver = Rodas4()
distrib_fine_xy = xy -> MvNormal(solve_ode(t, xy; solver=solver), σ^2*I(NT))

# Parameter -> data distribution (forward) mapping on observation grid
distrib_xy = xy -> MvNormal(solve_ode(t_obs, xy; solver=solver), σ^2*I(NT_obs))

# Variable names
varnames = Dict("ψ1" => "k_1", "ψ2" => "k_2")
varnames["ψ1_save"] = "k_1"
varnames["ψ2_save"] = "k_2"

# Parameter bounds
k1_min, k1_max = 0.1, 10.0
k2_min, k2_max = 0.1, 50.0
xy_lower_bounds = [k1_min, k2_min]
xy_upper_bounds = [k1_max, k2_max]

# Initial guess for optimization
xy_initial = 0.5 * (xy_lower_bounds + xy_upper_bounds)

# True parameter
k1_true, k2_true = 1.0, 5.0
xy_true = [k1_true, k2_true]

# Generate data
Nrep = 1
data = rand(distrib_xy(xy_true), Nrep)

# Visualize data and true solution
scatter(t_obs, data, label="Data")
plot!(t, solve_ode(t, xy_true), label="True Solution", xlabel="Time", ylabel="Concentration", legend=:topleft)

# Construct log-likelihood in original parameterization given (iid) data
lnlike_xy = construct_lnlike_xy(distrib_xy, data; dist_type=:multi)
model_name = "monod_model_xy"
println(model_name)

# Grid sizes for profiling
grid_steps = [500]
dim_all = length(xy_initial)
indices_all = 1:dim_all

# Point estimation (MLE)
point_estimation_method = :LN_BOBYQA
target_indices = [] # Empty target indices for MLE
n_guesses = 3

# Generate multiple initial guesses
nuisance_guesses = generate_initial_guesses(xy_lower_bounds, xy_upper_bounds, n_guesses)

xy_MLE, lnlike_xy_MLE = profile_target(lnlike_xy, target_indices,
    xy_lower_bounds, xy_upper_bounds, 
    xy_initial; grid_steps=grid_steps, ω_initial_extras=nuisance_guesses,
    method=point_estimation_method)

# Quadratic approximation at MLE
lnlike_xy_ellipse, H_xy_ellipse = construct_ellipse_lnlike_approx(lnlike_xy, xy_MLE)

# Eigenanalysis
evals, evecs = eigen(H_xy_ellipse; sortby = x -> -real(x))
println("Eigenvectors and eigenvalues for "*model_name)
println("Eigenvalues: ", evals)
println("Eigenvectors: ", evecs)

# Determine svd of phi mapping in xy coordinates
J_ϕ_xy, U_xy, S_xy, Vt_xy = compute_ϕ_Jacobian(ϕ_func_xy, xy_MLE; method_type=:auto, compute_svd=true)
println("\nSVD analysis in original coordinates:")
println("Singular values: ", S_xy)
println("Right singular vectors (V): ")
display(Vt_xy)

# Calculate prediction at MLE for reference
pred_mean_MLE = mean(distrib_fine_xy(xy_MLE))
true_mean = mean(distrib_fine_xy(xy_true))

# 1D Profiles
profile_method = :LN_BOBYQA
for i in 1:dim_all
    target_index = i
    nuisance_indices = setdiff(indices_all, target_index)
    nuisance_guess = xy_MLE[nuisance_indices]

    print("Variable: ", varnames["ψ"*string(i)], "\n")

    # Generate multiple initial guesses for nuisance parameters
    n_guesses_profiling = 3
    nuisance_guesses = generate_initial_guesses(xy_lower_bounds[nuisance_indices],
        xy_upper_bounds[nuisance_indices], n_guesses_profiling)

    # Profile full likelihood
    ψω_values, lnlike_ψ_values = profile_target(lnlike_xy, 
        target_index,
        xy_lower_bounds, 
        xy_upper_bounds,
        nuisance_guess; 
        grid_steps=grid_steps,
        ω_initial_extras=nuisance_guesses,
        method=profile_method)

    # Profile quadratic approximation
    ψω_ellipse_values, lnlike_ψ_ellipse_values = profile_target(lnlike_xy_ellipse,
        target_index,
        xy_lower_bounds, 
        xy_upper_bounds,
        nuisance_guess; 
        grid_steps=grid_steps,
        ω_initial_extras=nuisance_guesses,
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

    # Prediction CIs. Use fine grid for prediction
    lower_ψ, upper_ψ, _ = construct_upper_lower_profile_wise_CIs_for_mean(
        distrib_fine_xy, ψω_values, lnlike_ψ_values; l_level=95, df=1)

    plot_profile_wise_CI_for_mean(
        t, lower_ψ, upper_ψ, pred_mean_MLE,
        model_name, "t", "t",
        data_indep=t_obs, data_dep=data, 
        true_mean=true_mean,
        target=varnames["ψ"*string(i)])
end

# 2D Profiles
# Technically no profiles as we are in 2D but write generically for future extension

param_pairs = [(i, j) for i in 1:dim_all for j in i+1:dim_all]
profile_method = :LN_BOBYQA

for (i, j) in param_pairs
    target_indices_ij = [i,j]
    nuisance_indices = setdiff(indices_all, target_indices_ij)
    nuisance_guess = xy_MLE[nuisance_indices]
    ψ_true_pair = xy_true[target_indices_ij]

    # Generate multiple initial guesses for nuisance parameters if needed
    if length(nuisance_indices) > 0
        n_guesses_profiling = 3
        nuisance_guesses = generate_initial_guesses(xy_lower_bounds[nuisance_indices],
            xy_upper_bounds[nuisance_indices], n_guesses_profiling)
    else
        nuisance_guesses = nothing
    end

    print("Variables: ", varnames["ψ"*string(i)], ", ", varnames["ψ"*string(j)], "\n")

    # Create a copy of varnames for this iteration
    current_varnames = deepcopy(varnames)
    current_varnames["ψ1"] = varnames["ψ"*string(i)]
    current_varnames["ψ2"] = varnames["ψ"*string(j)]
    current_varnames["ψ1_save"] = varnames["ψ"*string(i)*"_save"]
    current_varnames["ψ2_save"] = varnames["ψ"*string(j)*"_save"]
    
    # Profile full likelihood
    ψω_values, lnlike_ψ_values = profile_target(lnlike_xy, target_indices_ij,
        xy_lower_bounds, xy_upper_bounds,
        nuisance_guess; grid_steps=grid_steps,
        ω_initial_extras=nuisance_guesses,
        method=profile_method)

    # Profile quadratic approximation
    ψω_ellipse_values, lnlike_ψ_ellipse_values = profile_target(lnlike_xy_ellipse,
        target_indices_ij,
        xy_lower_bounds, xy_upper_bounds,
        nuisance_guess; grid_steps=grid_steps,
        ω_initial_extras=nuisance_guesses,
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
    ψ1_values, ψ2_values, like_ψ1_values, like_ψ2_values = get_1D_profiles_from_2D(
        ψ_values, lnlike_ψ_values)

    plot_1D_profile(model_name, ψ1_values, log.(like_ψ1_values),
        current_varnames["ψ1"];
        varname_save=current_varnames["ψ1_save"]*"_from_2D",
        ψ_true=ψ_true_pair[1])

    plot_1D_profile(model_name, ψ2_values, log.(like_ψ2_values),
        current_varnames["ψ2"];
        varname_save=current_varnames["ψ2_save"]*"_from_2D",
        ψ_true=ψ_true_pair[2])

    # 2D prediction CIs using fine grid for predictions
    lower_ψ1ψ2, upper_ψ1ψ2, _ = construct_upper_lower_profile_wise_CIs_for_mean(
        distrib_fine_xy, ψω_values, lnlike_ψ_values; l_level=95, df=2)

    plot_profile_wise_CI_for_mean(
        t, lower_ψ1ψ2, upper_ψ1ψ2, pred_mean_MLE,
        model_name, "t", "t",
        data_indep=t_obs, data_dep=data,
        true_mean=true_mean,
        target=current_varnames["ψ1"]*"_"*current_varnames["ψ2"])

end

# --------------------------------------------------------
# Log Parameterization Analysis
# --------------------------------------------------------
model_name = "monod_model_log"
println(model_name)

# Coordinate transformation
xytoXY_log(xy) = log.(xy)
XYtoxy_log(XY) = exp.(XY)

# Transform bounds and parameters
XY_log_lower_bounds = log.(xy_lower_bounds)
XY_log_upper_bounds = log.(xy_upper_bounds)
XY_log_initial = xytoXY_log(xy_initial)
XY_log_true = xytoXY_log(xy_true)

# Transform likelihood, distribution, and phi mapping
lnlike_XY_log = construct_lnlike_XY(lnlike_xy, XYtoxy_log)
distrib_fine_XY_log = construct_distrib_XY(distrib_fine_xy, XYtoxy_log)
ϕ_func_XY_log = construct_ϕ_XY(ϕ_func_xy, XYtoxy_log)

# Update variable names for log coordinates
varnames["ψ1"] = "\\ln\\ k_1"
varnames["ψ2"] = "\\ln\\ k_2"
varnames["ψ1_save"] = "ln_k_1"
varnames["ψ2_save"] = "ln_k_2"

# Point estimation in log coordinates
target_indices = []  # empty for MLE
n_guesses = 3
nuisance_guesses = generate_initial_guesses(XY_log_lower_bounds, XY_log_upper_bounds, n_guesses)

XY_log_MLE, lnlike_XY_log_MLE = profile_target(lnlike_XY_log, target_indices,
    XY_log_lower_bounds, XY_log_upper_bounds, 
    XY_log_initial; grid_steps=grid_steps, ω_initial_extras=nuisance_guesses,
    method=point_estimation_method)

# Quadratic approximation at MLE
lnlike_XY_log_ellipse, H_XY_log_ellipse = construct_ellipse_lnlike_approx(lnlike_XY_log, XY_log_MLE)

# Eigenanalysis in log coordinates
evals_log, evecs_log = eigen(H_XY_log_ellipse; sortby = x -> -real(x))
println("Eigenvectors and eigenvalues for "*model_name)
println("Eigenvalues: ", evals_log)
println("Eigenvectors: ", evecs_log)

# Determine svd of phi mapping in log coordinates
J_ϕ_XY_log, U_XY_log, S_XY_log, Vt_XY_log = compute_ϕ_Jacobian(ϕ_func_XY_log, XY_log_MLE; method_type=:auto, compute_svd=true)
println("\nSVD analysis in log coordinates:")
println("Singular values: ", S_XY_log)
println("Right singular vectors (V): ")
display(Vt_XY_log)

# Calculate prediction at MLE for reference
pred_mean_MLE_log = mean(distrib_fine_XY_log(XY_log_MLE))
true_mean_log = mean(distrib_fine_XY_log(XY_log_true))

# 1D Profiles in log coordinates
profile_method = :LN_BOBYQA
for i in 1:dim_all
    target_index = i
    nuisance_indices = setdiff(indices_all, target_index)
    nuisance_guess = XY_log_MLE[nuisance_indices]

    print("Variable: ", varnames["ψ"*string(i)], "\n")

    # Generate multiple initial guesses for nuisance parameters
    n_guesses_profiling = 3
    nuisance_guesses = generate_initial_guesses(XY_log_lower_bounds[nuisance_indices],
        XY_log_upper_bounds[nuisance_indices], n_guesses_profiling)

    # Profile full likelihood
    ψω_values, lnlike_ψ_values = profile_target(lnlike_XY_log, target_index,
        XY_log_lower_bounds, XY_log_upper_bounds,
        nuisance_guess; grid_steps=grid_steps,
        ω_initial_extras=nuisance_guesses,
        method=profile_method)

    # Profile quadratic approximation
    ψω_ellipse_values, lnlike_ψ_ellipse_values = profile_target(lnlike_XY_log_ellipse,
        target_index,
        XY_log_lower_bounds, XY_log_upper_bounds,
        nuisance_guess; grid_steps=grid_steps,
        ω_initial_extras=nuisance_guesses,
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

    # Prediction CIs using fine grid distribution
    lower_ψ, upper_ψ, _ = construct_upper_lower_profile_wise_CIs_for_mean(
        distrib_fine_XY_log, ψω_values, lnlike_ψ_values; l_level=95, df=1)

    plot_profile_wise_CI_for_mean(
        t, lower_ψ, upper_ψ, pred_mean_MLE_log,
        model_name, "t", "t",
        data_indep=t_obs, data_dep=data,
        true_mean=true_mean_log,
        target=varnames["ψ"*string(i)])
end

# 2D Profiles in log coordinates
for (i,j) in param_pairs
    target_indices_ij = [i,j]
    nuisance_indices = setdiff(indices_all, target_indices_ij)
    nuisance_guess = XY_log_MLE[nuisance_indices]
    ψ_true_pair = XY_log_true[target_indices_ij]

    # Generate multiple initial guesses if needed
    if length(nuisance_indices) > 0
        n_guesses_profiling = 3
        nuisance_guesses = generate_initial_guesses(XY_log_lower_bounds[nuisance_indices],
            XY_log_upper_bounds[nuisance_indices], n_guesses_profiling)
    else
        nuisance_guesses = nothing
    end

    # Create a copy of varnames for this iteration
    current_varnames = deepcopy(varnames)
    current_varnames["ψ1"] = varnames["ψ"*string(i)]
    current_varnames["ψ2"] = varnames["ψ"*string(j)]
    current_varnames["ψ1_save"] = varnames["ψ"*string(i)*"_save"]
    current_varnames["ψ2_save"] = varnames["ψ"*string(j)*"_save"]

    # Profile full likelihood
    ψω_values, lnlike_ψ_values = profile_target(lnlike_XY_log, target_indices_ij,
        XY_log_lower_bounds, XY_log_upper_bounds,
        nuisance_guess; grid_steps=grid_steps,
        ω_initial_extras=nuisance_guesses,
        method=profile_method)

    # Profile quadratic approximation
    ψω_ellipse_values, lnlike_ψ_ellipse_values = profile_target(lnlike_XY_log_ellipse,
        target_indices_ij,
        XY_log_lower_bounds, XY_log_upper_bounds,
        nuisance_guess; grid_steps=grid_steps,
        ω_initial_extras=nuisance_guesses,
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
    ψ1_values, ψ2_values, like_ψ1_values, like_ψ2_values = get_1D_profiles_from_2D(
        ψ_values, lnlike_ψ_values)

    plot_1D_profile(model_name, ψ1_values, log.(like_ψ1_values),
        current_varnames["ψ1"];
        varname_save=current_varnames["ψ1_save"]*"_from_2D",
        ψ_true=ψ_true_pair[1])

    plot_1D_profile(model_name, ψ2_values, log.(like_ψ2_values),
        current_varnames["ψ2"];
        varname_save=current_varnames["ψ2_save"]*"_from_2D",
        ψ_true=ψ_true_pair[2])

    # 2D prediction CIs using fine grid distribution
    lower_ψ1ψ2, upper_ψ1ψ2, _ = construct_upper_lower_profile_wise_CIs_for_mean(
        distrib_fine_XY_log, ψω_values, lnlike_ψ_values; l_level=95, df=2)

    plot_profile_wise_CI_for_mean(
        t, lower_ψ1ψ2, upper_ψ1ψ2, pred_mean_MLE_log,
        model_name, "t", "t",
        data_indep=t_obs, data_dep=data,
        true_mean=true_mean_log,
        target=current_varnames["ψ1"]*"_"*current_varnames["ψ2"])
end

# --------------------------------------------------------
# Sloppihood-Informed Parameterization Analysis
# --------------------------------------------------------
model_name = "monod_model_iir"
println(model_name)

# Scale and round eigenvectors for iir transformation
# Option 1: based on eigenvectors from Fisher Information
# Option 2: based on the right singular vectors from the phi mapping
use_singular_vectors = true
if use_singular_vectors
    evecs_scaled = scale_and_round(Vt_XY_log; column_scales=[1,1])
else
    evecs_scaled = scale_and_round(evecs_log; column_scales=[1,1])
end

println("Transformations:")
display(evecs_scaled)
display(inv(evecs_scaled))

println("Original right singular vectors:")
display(Vt_XY_log)

# Construct transformation
xytoXY_iir, XYtoxy_iir = reparam(evecs_scaled)

# Transform likelihood, distribution, and phi mapping
lnlike_XY_iir = construct_lnlike_XY(lnlike_xy, XYtoxy_iir)
distrib_fine_XY_iir = construct_distrib_XY(distrib_fine_xy, XYtoxy_iir)
ϕ_func_XY_iir = construct_ϕ_XY(ϕ_func_xy, XYtoxy_iir)

# Set bounds for iir coordinates based on transformation of original bounds
# Note: These bounds might need manual adjustment
XY_iir_lower_bounds = [0.05, 0.05]  # k2/k1, k1*k2
XY_iir_upper_bounds = [10.0, 100]  # k2/k1, k1*k2

# Initial guess for iir coordinates (manual coz non-monotonic/complex transform)
# XY_iir_initial = xytoXY_iir(xy_initial)
XY_iir_initial = [1.0, 10.0]

# Check if initial guess is inside bounds
all_inside = true
for i in 1:length(XY_iir_initial)
    if XY_iir_initial[i] < XY_iir_lower_bounds[i] || XY_iir_initial[i] > XY_iir_upper_bounds[i]
        all_inside = false
        println("Warning: Initial guess component $i is outside bounds")
    end
end
if !all_inside
    println(XY_iir_initial)
    error("Initial guess must be inside bounds")
end

# Transform true value to iir coordinates
XY_iir_true = xytoXY_iir(xy_true)

# Update variable names for iir coordinates
varnames["ψ1"] = "\\frac{k_2}{k_1}"
varnames["ψ2"] = "k_1k_2"
varnames["ψ1_save"] = "k_2_over_k_1"
varnames["ψ2_save"] = "k_1k_2"

# Point estimation in iir coordinates
target_indices = []  # empty for MLE
n_guesses = 3
nuisance_guesses = generate_initial_guesses(XY_iir_lower_bounds, XY_iir_upper_bounds, n_guesses)

XY_iir_MLE, lnlike_XY_iir_MLE = profile_target(lnlike_XY_iir, target_indices,
    XY_iir_lower_bounds, XY_iir_upper_bounds, 
    XY_iir_initial; grid_steps=grid_steps, ω_initial_extras=nuisance_guesses,
    method=point_estimation_method)

# Quadratic approximation at MLE
lnlike_XY_iir_ellipse, H_XY_iir_ellipse = construct_ellipse_lnlike_approx(lnlike_XY_iir, XY_iir_MLE)

# Eigenanalysis in iir coordinates
evals_iir, evecs_iir = eigen(H_XY_iir_ellipse; sortby = x -> -real(x))
println("Eigenvectors and eigenvalues for "*model_name)
println("Eigenvalues: ", evals_iir)
println("Eigenvectors: ", evecs_iir)

# Determine svd of phi mapping in iir coordinates
J_ϕ_XY_iir, U_XY_iir, S_XY_iir, Vt_XY_iir = compute_ϕ_Jacobian(ϕ_func_XY_iir, XY_iir_MLE; method_type=:auto, compute_svd=true)
println("\nSVD analysis in iir coordinates:")
println("Singular values: ", S_XY_iir)
println("Right singular vectors (V): ")
display(Vt_XY_iir)

# Calculate prediction at MLE for reference
pred_mean_MLE_iir = mean(distrib_fine_XY_iir(XY_iir_MLE))
true_mean_iir = mean(distrib_fine_XY_iir(XY_iir_true))

# 1D Profiles
profile_method = :LN_BOBYQA
for i in 1:dim_all
    target_index = i
    nuisance_indices = setdiff(indices_all, target_index)
    nuisance_guess = XY_iir_MLE[nuisance_indices]

    print("Variable: ", varnames["ψ"*string(i)], "\n")

    # Generate multiple initial guesses for nuisance parameters
    n_guesses_profiling = 3
    nuisance_guesses = generate_initial_guesses(XY_iir_lower_bounds[nuisance_indices],
        XY_iir_upper_bounds[nuisance_indices], n_guesses_profiling)

    # Profile full likelihood
    ψω_values, lnlike_ψ_values = profile_target(lnlike_XY_iir, target_index,
        XY_iir_lower_bounds, XY_iir_upper_bounds,
        nuisance_guess; grid_steps=grid_steps,
        ω_initial_extras=nuisance_guesses,
        method=profile_method)

    # Profile quadratic approximation
    ψω_ellipse_values, lnlike_ψ_ellipse_values = profile_target(lnlike_XY_iir_ellipse,
        target_index,
        XY_iir_lower_bounds, XY_iir_upper_bounds,
        nuisance_guess; grid_steps=grid_steps,
        ω_initial_extras=nuisance_guesses,
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

    # Prediction CIs using fine grid distribution
    lower_ψ, upper_ψ, _ = construct_upper_lower_profile_wise_CIs_for_mean(
        distrib_fine_XY_iir, ψω_values, lnlike_ψ_values; l_level=95, df=1)

    plot_profile_wise_CI_for_mean(
        t, lower_ψ, upper_ψ, pred_mean_MLE_iir,
        model_name, "t", "t",
        data_indep=t_obs, data_dep=data,
        true_mean=true_mean_iir,
        target=varnames["ψ"*string(i)])
end

# 2D Profiles
for (i,j) in param_pairs
    target_indices_ij = [i,j]
    nuisance_indices = setdiff(indices_all, target_indices_ij)
    nuisance_guess = XY_iir_MLE[nuisance_indices]
    ψ_true_pair = XY_iir_true[target_indices_ij]

    # Generate multiple initial guesses if needed
    if length(nuisance_indices) > 0
        n_guesses_profiling = 3
        nuisance_guesses = generate_initial_guesses(XY_iir_lower_bounds[nuisance_indices],
            XY_iir_upper_bounds[nuisance_indices], n_guesses_profiling)
    else
        nuisance_guesses = nothing
    end

    # Create a copy of varnames for this iteration
    current_varnames = deepcopy(varnames)
    current_varnames["ψ1"] = varnames["ψ"*string(i)]
    current_varnames["ψ2"] = varnames["ψ"*string(j)]
    current_varnames["ψ1_save"] = varnames["ψ"*string(i)*"_save"]
    current_varnames["ψ2_save"] = varnames["ψ"*string(j)*"_save"]

    # Profile full likelihood
    ψω_values, lnlike_ψ_values = profile_target(lnlike_XY_iir, target_indices_ij,
        XY_iir_lower_bounds, XY_iir_upper_bounds,
        nuisance_guess; grid_steps=grid_steps,
        ω_initial_extras=nuisance_guesses,
        method=profile_method)

    # Profile quadratic approximation
    ψω_ellipse_values, lnlike_ψ_ellipse_values = profile_target(lnlike_XY_iir_ellipse,
        target_indices_ij,
        XY_iir_lower_bounds, XY_iir_upper_bounds,
        nuisance_guess; grid_steps=grid_steps,
        ω_initial_extras=nuisance_guesses,
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
    ψ1_values, ψ2_values, like_ψ1_values, like_ψ2_values = get_1D_profiles_from_2D(
        ψ_values, lnlike_ψ_values)

    plot_1D_profile(model_name, ψ1_values, log.(like_ψ1_values),
        current_varnames["ψ1"];
        varname_save=current_varnames["ψ1_save"]*"_from_2D",
        ψ_true=ψ_true_pair[1])

    plot_1D_profile(model_name, ψ2_values, log.(like_ψ2_values),
        current_varnames["ψ2"];
        varname_save=current_varnames["ψ2_save"]*"_from_2D",
        ψ_true=ψ_true_pair[2])

    # 2D prediction CIs using fine grid distribution
    lower_ψ1ψ2, upper_ψ1ψ2, _ = construct_upper_lower_profile_wise_CIs_for_mean(
        distrib_fine_XY_iir, ψω_values, lnlike_ψ_values; l_level=95, df=2)

    plot_profile_wise_CI_for_mean(
        t, lower_ψ1ψ2, upper_ψ1ψ2, pred_mean_MLE_iir,
        model_name, "t", "t",
        data_indep=t_obs, data_dep=data,
        true_mean=true_mean_iir,
        target=current_varnames["ψ1"]*"_"*current_varnames["ψ2"])
end

