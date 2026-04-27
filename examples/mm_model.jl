# Run with:
#   julia --project=. "examples/mm_model.jl"
#
# This example fits the Michaelis-Menten/Monod model, computes the invariant
# split in log coordinates, and builds an interpretable reparameterisation.
#
# Set `limit = true` below for the exact-limit IIR case, or `limit = false`
# for the non-limit practical-identifiability case.

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

# Example-local helpers for the practical directional probe.
include("_directional_practical_probe_helpers.jl")

# Set random seed for reproducibility
Random.seed!(4321)

# --------------------------------------------------------
# Model Definition
# --------------------------------------------------------

# Define the Michaelis-Menten model ODE 
function DE!(dS, S, θ, t)
    """
    Michaelis-Menten model ODE definition.
    
    Parameters:
    - dS: Rate of change vector (modified in-place)
    - S: Current state vector
    - θ: Parameter vector 
    - t: Current time
    """

    dS[1] = -θ[1]*S[1]/(θ[2] + S[1])
end

# Define the Michaelis-Menten model ODE: limit form
function DE_limit!(dS, S, θ, t)
    """
    Michaelis-Menten model ODE definition for limit.
    
    Parameters:
    - dS: Rate of change vector (modified in-place)
    - S: Current state vector
    - θ: Parameter vector 
    - t: Current time
    """
    dS[1] = -θ[1]*S[1]/(θ[2])
end

# ODE model solver  
function solve_ode(t_save, θ, S0; solver=Rodas4(), limit=false)
    """
    Michaelis-Menten model solution: maps parameters θ to solution values 
    on the specified time grid.
    
    Parameters:
    - t_save: Time grid points
    - θ: Parameter vector
    - S0: Initial condition
    - solver: ODE solver (default: Rodas4())
    - limit: Whether to use the limit form of the model (default: false)
    
    Returns:
    - Vector of solution values on the time grid
    """
    tspan = (0.0, maximum(t_save))
    if limit
        ODE = DE_limit!
    else
        ODE = DE!
    end
    prob = ODEProblem(ODE, [S0], tspan, θ)
    sol = solve(prob, solver, saveat=t_save, abstol=1e-12, reltol=1e-9)
    return sol[1, :]
end

# Creates a ϕ mapping function with fixed grid parameters
function create_ϕ_mapping(t, S0; limit=false)
    """
    Create a ϕ mapping function from model parameters to solution values
    with fixed grid parameters.
    
    Parameters:
    - t: Time grid points
    - S0: Initial condition
    - limit: Whether to use the limit form of the model (default: false)
    
    Returns:
    - ϕ mapping function from θ to solution values
    """
    return θ -> solve_ode(t, θ, S0; limit=limit)
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
S0 = 1.0
σ = 0.05

# --------------------------------------------------------
# --- Analysis in original parameterisation and data generation ---
# --------------------------------------------------------
# Choose whether to use the limit form of the model.
# Set to false here to inspect the non-limit practical-identifiability case.
limit = false

if limit
    model_name = "mm_model_xy_limit"
else
    model_name = "mm_model_xy"
end
println(model_name)

# Define ϕ mapping in original coordinates on fine grid
ϕ_func_xy = create_ϕ_mapping(t, S0; limit=limit)

# Parameter -> data distribution (forward) mapping on fine grid
solver = Rodas4()
distrib_fine_xy = xy -> MvNormal(solve_ode(t, xy, S0; solver=solver, limit=limit), σ^2*I(NT))

# Parameter -> data distribution (forward) mapping on observation grid
distrib_xy = xy -> MvNormal(solve_ode(t_obs, xy, S0; solver=solver, limit=limit), σ^2*I(NT_obs))

# Variable names
varnames = Dict("ψ1" => "ν", "ψ2" => "K")
varnames["ψ1_save"] = "nu"
varnames["ψ2_save"] = "K" 

# Parameter bounds
ν_min, ν_max = 0.1, 10.0
K_min, K_max = 0.1, 50.0
xy_lower_bounds = [ν_min, K_min]
xy_upper_bounds = [ν_max, K_max]

# Initial guess for optimization
xy_initial = 0.5 * (xy_lower_bounds + xy_upper_bounds)

# True parameter
ν_true, K_true = 1.0, 5.0
xy_true = [ν_true, K_true]

# Generate new data
# Nrep = 1
# data = rand(distrib_xy(xy_true), Nrep)
# Use saved realisation for reproducibility
data = [1.04, 0.66, 0.50, 0.36, 0.28, 0.18, 0.01, 0.08, 0.02, 0.07, 0.05]

# Visualize data and true solution
scatter(t_obs, data, label="Data")
plot!(t, solve_ode(t, xy_true, S0), label="True Solution", xlabel="Time", ylabel="Concentration", legend=:topleft)

# Construct log-likelihood in original parameterization given (iid) data
lnlike_xy = construct_lnlike_xy(distrib_xy, data; dist_type=:multi)

# Grid sizes for profiling
grid_steps = [500]
dim_all = length(xy_initial)
indices_all = 1:dim_all

# Point estimation (MLE)
point_estimation_method = :LN_BOBYQA
target_indices = [] # Empty target indices for MLE
n_guesses = 3

# Generate multiple initial guesses
xy_mle_initial_guesses = generate_initial_guesses(xy_lower_bounds, xy_upper_bounds, n_guesses)

xy_MLE, lnlike_xy_MLE = profile_target(lnlike_xy, target_indices,
    xy_lower_bounds, xy_upper_bounds, 
    xy_initial; grid_steps=grid_steps, ω_initial_extras=xy_mle_initial_guesses,
    method=point_estimation_method)

# Quadratic approximation at MLE
lnlike_xy_ellipse, H_xy_ellipse = construct_ellipse_lnlike_approx(lnlike_xy, xy_MLE)

# Eigenanalysis
evals, evecs = eigen(H_xy_ellipse; sortby = x -> -real(x))
println("Eigenvectors and eigenvalues for "*model_name)
println("Eigenvalues: ", evals)
println("Eigenvectors: ", evecs)

# Determine svd of phi mapping in xy coordinates
J_ϕ_xy, U_xy, S_xy, V_xy = compute_ϕ_Jacobian(ϕ_func_xy, xy_MLE; method_type=:auto, compute_svd=true)
println("\nSVD analysis in original coordinates:")
println("Singular values: ", S_xy)
println("Right singular vectors (V): ")
display(V_xy)

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
    xy_profile_initial_guesses = generate_initial_guesses(xy_lower_bounds[nuisance_indices],
        xy_upper_bounds[nuisance_indices], n_guesses_profiling)

    # Profile full likelihood
    ψω_values, lnlike_ψ_values = profile_target(lnlike_xy, 
        target_index,
        xy_lower_bounds, 
        xy_upper_bounds,
        nuisance_guess; 
        grid_steps=grid_steps,
        ω_initial_extras=xy_profile_initial_guesses,
        method=profile_method)

    # Profile quadratic approximation
    ψω_ellipse_values, lnlike_ψ_ellipse_values = profile_target(lnlike_xy_ellipse,
        target_index,
        xy_lower_bounds, 
        xy_upper_bounds,
        nuisance_guess; 
        grid_steps=grid_steps,
        ω_initial_extras=xy_profile_initial_guesses,
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
        distrib_fine_xy, ψω_values, lnlike_ψ_values; l_level=95, df=2)

    plot_profile_wise_CI_for_mean(
        t, lower_ψ, upper_ψ, pred_mean_MLE,
        model_name, "S", "t", "t",
        data_indep=t_obs, data_dep=data, 
        true_mean=true_mean,
        target=varnames["ψ"*string(i)],
        target_save=varnames["ψ"*string(i)*"_save"])
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
        xy_pair_initial_guesses = generate_initial_guesses(xy_lower_bounds[nuisance_indices],
            xy_upper_bounds[nuisance_indices], n_guesses_profiling)
    else
        xy_pair_initial_guesses = nothing
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
        ω_initial_extras=xy_pair_initial_guesses,
        method=profile_method)

    # Profile quadratic approximation
    ψω_ellipse_values, lnlike_ψ_ellipse_values = profile_target(lnlike_xy_ellipse,
        target_indices_ij,
        xy_lower_bounds, xy_upper_bounds,
        nuisance_guess; grid_steps=grid_steps,
        ω_initial_extras=xy_pair_initial_guesses,
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
        model_name, "S", "t", "t",
        data_indep=t_obs, data_dep=data,
        true_mean=true_mean,
        target=current_varnames["ψ1"]*", "*current_varnames["ψ2"],
        target_save=current_varnames["ψ1_save"]*"_"*current_varnames["ψ2_save"])

end

# --------------------------------------------------------
# Log Parameterization Analysis
# --------------------------------------------------------
if limit
    model_name = "mm_model_log_limit"
else
    model_name = "mm_model_log"
end
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
varnames["ψ1"] = "\\ln\\ \\nu"
varnames["ψ2"] = "\\ln\\ K"
varnames["ψ1_save"] = "ln_nu"
varnames["ψ2_save"] = "ln_K"

# Point estimation in log coordinates
target_indices = []  # empty for MLE
n_guesses = 3
XY_log_mle_initial_guesses = generate_initial_guesses(XY_log_lower_bounds, XY_log_upper_bounds, n_guesses)

XY_log_MLE, lnlike_XY_log_MLE = profile_target(lnlike_XY_log, target_indices,
    XY_log_lower_bounds, XY_log_upper_bounds, 
    XY_log_initial; grid_steps=grid_steps, ω_initial_extras=XY_log_mle_initial_guesses,
    method=point_estimation_method)

# Quadratic approximation at MLE
lnlike_XY_log_ellipse, H_XY_log_ellipse = construct_ellipse_lnlike_approx(lnlike_XY_log, XY_log_MLE)

# Eigenanalysis in log coordinates
evals_log, evecs_log = eigen(H_XY_log_ellipse; sortby = x -> -real(x))
println("Eigenvectors and eigenvalues for "*model_name)
println("Eigenvalues: ", evals_log)
println("Eigenvectors: ", evecs_log)

# Determine svd of phi mapping in log coordinates
J_ϕ_XY_log, U_XY_log, S_XY_log, V_XY_log = compute_ϕ_Jacobian(ϕ_func_XY_log, XY_log_MLE; method_type=:auto, compute_svd=true)
println("\nSVD analysis in log coordinates:")
println("Singular values: ", S_XY_log)
println("Right singular vectors (V): ")
display(V_XY_log)

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
    XY_log_profile_initial_guesses = generate_initial_guesses(XY_log_lower_bounds[nuisance_indices],
        XY_log_upper_bounds[nuisance_indices], n_guesses_profiling)

    # Profile full likelihood
    ψω_values, lnlike_ψ_values = profile_target(lnlike_XY_log, target_index,
        XY_log_lower_bounds, XY_log_upper_bounds,
        nuisance_guess; grid_steps=grid_steps,
        ω_initial_extras=XY_log_profile_initial_guesses,
        method=profile_method)

    # Profile quadratic approximation
    ψω_ellipse_values, lnlike_ψ_ellipse_values = profile_target(lnlike_XY_log_ellipse,
        target_index,
        XY_log_lower_bounds, XY_log_upper_bounds,
        nuisance_guess; grid_steps=grid_steps,
        ω_initial_extras=XY_log_profile_initial_guesses,
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
        distrib_fine_XY_log, ψω_values, lnlike_ψ_values; l_level=95, df=2)

    plot_profile_wise_CI_for_mean(
        t, lower_ψ, upper_ψ, pred_mean_MLE_log,
        model_name, "S", "t", "t",
        data_indep=t_obs, data_dep=data,
        true_mean=true_mean_log,
        target=varnames["ψ"*string(i)],
        target_save=varnames["ψ"*string(i)*"_save"])
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
        XY_log_pair_initial_guesses = generate_initial_guesses(XY_log_lower_bounds[nuisance_indices],
            XY_log_upper_bounds[nuisance_indices], n_guesses_profiling)
    else
        XY_log_pair_initial_guesses = nothing
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
        ω_initial_extras=XY_log_pair_initial_guesses,
        method=profile_method)

    # Profile quadratic approximation
    ψω_ellipse_values, lnlike_ψ_ellipse_values = profile_target(lnlike_XY_log_ellipse,
        target_indices_ij,
        XY_log_lower_bounds, XY_log_upper_bounds,
        nuisance_guess; grid_steps=grid_steps,
        ω_initial_extras=XY_log_pair_initial_guesses,
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
        model_name, "S", "t", "t",
        data_indep=t_obs, data_dep=data,
        true_mean=true_mean_log,
        target=current_varnames["ψ1"]*", "*current_varnames["ψ2"],
        target_save=current_varnames["ψ1_save"]*"_"*current_varnames["ψ2_save"])
end

# --------------------------------------------------------
# Invariant Subspace Analysis in Log Coordinates
# --------------------------------------------------------
println("\n" * "="^60)
println("Invariant Subspace Analysis in Log Coordinates")
println("="^60)

S_inv, N_inv, N_perp_inv, rank_inv = find_invariant_subspace(
    ϕ_func_XY_log, XY_log_MLE; verbose=true)

println("\nJacobian Analysis:")
println("  Singular values: ", S_inv)
println("  Numerical rank: ", rank_inv)
println("  Dimension of invariant null space (N): ", size(N_inv, 2))
println("  Dimension of identifiable space (N_perp): ", size(N_perp_inv, 2))

null_space_dim = length(XY_log_MLE) - rank_inv
if size(N_inv, 2) == null_space_dim && null_space_dim > 0
    println("\nFull null space is invariant")
    println("  Type: Minimal image reparameterization")
    reparam_type = "minimal_image"
elseif size(N_inv, 2) > 0 && size(N_inv, 2) < null_space_dim
    println("\nPartial null space is invariant (dimension ", size(N_inv, 2), " of ", null_space_dim, ")")
    println("  Type: Image (not minimal) reparameterization")
    reparam_type = "image"
elseif size(N_inv, 2) == 0 && null_space_dim > 0
    println("\nNo null space is invariant")
    println("  Type: Image (not minimal) reparameterization")
    reparam_type = "image"
else
    println("\nNo null space detected")
    println("  Type: Appears structurally identifiable")
    reparam_type = "identifiable"
end

println("\nIdentifiable space basis N_perp (columns):")
display(N_perp_inv)
if size(N_inv, 2) > 0
    println("\nInvariant null space basis N (columns):")
    display(N_inv)
end

σ_rel = S_inv[1] > 0 ? S_inv ./ S_inv[1] : zeros(length(S_inv))
println("\nRelative singular values (σᵢ/σ₁): ", round.(σ_rel, digits=4))

# Shared monomial basis view for the final interpretable coordinates.
# Use informed selection on the identified side and simplicity-only selection on the null side.
param_names_iir = ["ν", "K"]
residual_cap_iir = 1e-2
identified_basis_result = informed_monomial_basis_search(
    N_perp_inv, J_ϕ_XY_log' * J_ϕ_XY_log, S_inv[1]^2, param_names_iir;
    s_max=2, c_max=1, residual_cap=residual_cap_iir)
null_basis_result = simple_monomial_basis_search(
    N_inv, param_names_iir; s_max=2, c_max=1, residual_cap=residual_cap_iir, retry_support=true)

if !identified_basis_result.basis_ok
    error("Stepwise informed simple basis search failed on the identified side N_perp")
end
if !null_basis_result.basis_ok
    error("Singleton-first sparse basis search failed on the invariant null side N")
end

identified_basis_columns = monomial_basis_matrix(identified_basis_result.selected, length(param_names_iir))
identified_basis_labels = basis_labels(identified_basis_result.selected)

# For this example, use the reciprocal sign convention K/ν rather than ν/K.
# This is the same 1D identified subspace, but it produces the natural plotting orientation.
if size(identified_basis_columns, 2) >= 1
    identified_basis_columns[:, 1] .*= -1
    identified_basis_labels[1] = "K/(ν)"
end

null_basis_columns = monomial_basis_matrix(null_basis_result.selected, length(param_names_iir))
null_basis_labels = basis_labels(null_basis_result.selected)
final_basis_columns = hcat(identified_basis_columns, null_basis_columns)
final_basis_labels = vcat(identified_basis_labels, null_basis_labels)
final_log_A = final_basis_columns'

if limit
    println("\nLimit case: expect one exact invariant log direction corresponding to νK.")
else
    println("\nNon-limit case: no exact invariant null space is expected; use practical diagnostics.")
end

if !limit && rank_inv == length(XY_log_MLE) && length(S_inv) > 1
    println("\nDirectional Practical Check under two weak-coordinate choices:")

    svd_log_probe = svd(J_ϕ_XY_log)
    v_weak_raw = svd_log_probe.V[:, end]
    raw_weak_coord_row = svd_log_probe.V[:, end]
    raw_probe = directional_practical_probe(
        ϕ_func_XY_log, XY_log_MLE, v_weak_raw,
        XY_log_lower_bounds, XY_log_upper_bounds)

    weak_coord_index = size(N_perp_inv, 2)
    monomial_weak_coord_row = final_log_A[weak_coord_index, :]
    monomial_weak_direction = inv(final_log_A)[:, weak_coord_index]
    monomial_probe = directional_practical_probe(
        ϕ_func_XY_log, XY_log_MLE, monomial_weak_direction,
        XY_log_lower_bounds, XY_log_upper_bounds)

    print_mm_directional_practical_probe(
        "Exact local weak coordinate (SVD basis)", raw_probe, XYtoxy_log, XY_log_MLE;
        coord_row=raw_weak_coord_row)

    print_mm_directional_practical_probe(
        "Simple monomial weak coordinate (interpretable basis)", monomial_probe, XYtoxy_log, XY_log_MLE;
        coord_row=monomial_weak_coord_row)

    println("\n  Summary for the non-limit case:")
    println("    The model is full rank here, so νK is not an invariant combination.")
    println("    But νK remains a practically weak direction near the MLE.")
    println("    The asymmetry in ε(-)/ε(+) shows this weakness is one-sided rather than exactly invariant.")
end

# --------------------------------------------------------
# Limit-case IIR / Non-limit Interpretable Reparameterization
# --------------------------------------------------------
if limit
    model_name = "mm_model_iir_limit"
    println("\n" * "="^60)
    println("Limit-case IIR Parameterization: ", model_name)
    println("="^60)
else
    model_name = "mm_model_iir"
    println("\n" * "="^60)
    println("Non-limit Interpretable Reparameterization: ", model_name)
    println("="^60)
end

println("\nSelected simple monomial basis labels:")
for (i, label) in enumerate(final_basis_labels)
    println("  ψ_", i, " = ", label)
end
println("\nSelected simple monomial transformation matrix (rows are log-parameter combinations):")
display(final_log_A)
println("\nInverse transformation matrix:")
display(inv(final_log_A))

if limit
    println("\nLimit case interpretation:")
    println("  First coordinate is the identifiable combination K/ν.")
    println("  Second coordinate is the invariant combination νK.")
elseif reparam_type == "identifiable"
    println("\nNon-limit interpretation:")
    println("  No exact invariant null space is present.")
    println("  The informed simple monomial basis is used as a local interpretable basis.")
    println("  K/ν is the stronger local combination; νK is weaker but non-invariant.")
else
    println("\nNon-limit interpretation:")
    println("  A mixed image reparameterization was detected; keep both coordinates.")
end

# Define coordinate transformation using the selected monomial basis.
# reparam expects columns = parameter combinations, so pass final_basis_columns.
xytoXY_iir, XYtoxy_iir = reparam(final_basis_columns)

# Transform likelihood, distribution, and phi mapping
lnlike_XY_iir = construct_lnlike_XY(lnlike_xy, XYtoxy_iir)
distrib_fine_XY_iir = construct_distrib_XY(distrib_fine_xy, XYtoxy_iir)
ϕ_func_XY_iir = construct_ϕ_XY(ϕ_func_xy, XYtoxy_iir)

# Set bounds for interpretable coordinates
XY_iir_lower_bounds = [0.05, 0.05]  # K/ν, ν*K
XY_iir_upper_bounds = [10.0, 100]   # K/ν, ν*K

# Initial guess for interpretable coordinates
XY_iir_initial = [1.0, 10.0]
inside_mask = (XY_iir_lower_bounds .<= XY_iir_initial) .& (XY_iir_initial .<= XY_iir_upper_bounds)
if !all(inside_mask)
    for i in eachindex(XY_iir_initial)
        if !inside_mask[i]
            println("Warning: Initial guess component $i is outside bounds")
        end
    end
    println(XY_iir_initial)
    error("Initial guess must be inside bounds")
end

# Transform true value to interpretable coordinates
XY_iir_true = xytoXY_iir(xy_true)

# Update variable names for interpretable coordinates
varnames["ψ1"] = "\\frac{K}{\\nu}"
varnames["ψ2"] = "\\nu K"
varnames["ψ1_save"] = "K_over_nu"
varnames["ψ2_save"] = "nu_K"

# Point estimation in iir coordinates
target_indices = []  # empty for MLE
n_guesses = 3
XY_iir_mle_initial_guesses = generate_initial_guesses(XY_iir_lower_bounds, XY_iir_upper_bounds, n_guesses)

XY_iir_MLE, lnlike_XY_iir_MLE = profile_target(lnlike_XY_iir, target_indices,
    XY_iir_lower_bounds, XY_iir_upper_bounds, 
    XY_iir_initial; grid_steps=grid_steps, ω_initial_extras=XY_iir_mle_initial_guesses,
    method=point_estimation_method)

# Quadratic approximation at MLE
lnlike_XY_iir_ellipse, H_XY_iir_ellipse = construct_ellipse_lnlike_approx(lnlike_XY_iir, XY_iir_MLE)

# Eigenanalysis in iir coordinates
evals_iir, evecs_iir = eigen(H_XY_iir_ellipse; sortby = x -> -real(x))
println("Eigenvectors and eigenvalues for "*model_name)
println("Eigenvalues: ", evals_iir)
println("Eigenvectors: ", evecs_iir)

# Determine svd of phi mapping in iir coordinates
J_ϕ_XY_iir, U_XY_iir, S_XY_iir, V_XY_iir = compute_ϕ_Jacobian(ϕ_func_XY_iir, XY_iir_MLE; method_type=:auto, compute_svd=true)
println("\nSVD analysis in iir coordinates:")
println("Singular values: ", S_XY_iir)
println("Right singular vectors (V): ")
display(V_XY_iir)

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
    XY_iir_profile_initial_guesses = generate_initial_guesses(XY_iir_lower_bounds[nuisance_indices],
        XY_iir_upper_bounds[nuisance_indices], n_guesses_profiling)

    # Profile full likelihood
    ψω_values, lnlike_ψ_values = profile_target(lnlike_XY_iir, target_index,
        XY_iir_lower_bounds, XY_iir_upper_bounds,
        nuisance_guess; grid_steps=grid_steps,
        ω_initial_extras=XY_iir_profile_initial_guesses,
        method=profile_method)

    # Profile quadratic approximation
    ψω_ellipse_values, lnlike_ψ_ellipse_values = profile_target(lnlike_XY_iir_ellipse,
        target_index,
        XY_iir_lower_bounds, XY_iir_upper_bounds,
        nuisance_guess; grid_steps=grid_steps,
        ω_initial_extras=XY_iir_profile_initial_guesses,
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
        distrib_fine_XY_iir, ψω_values, lnlike_ψ_values; l_level=95, df=2)

    plot_profile_wise_CI_for_mean(
        t, lower_ψ, upper_ψ, pred_mean_MLE_iir,
        model_name, "S", "t", "t",
        data_indep=t_obs, data_dep=data,
        true_mean=true_mean_iir,
        target=varnames["ψ"*string(i)],
        target_save=varnames["ψ"*string(i)*"_save"])
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
        XY_iir_pair_initial_guesses = generate_initial_guesses(XY_iir_lower_bounds[nuisance_indices],
            XY_iir_upper_bounds[nuisance_indices], n_guesses_profiling)
    else
        XY_iir_pair_initial_guesses = nothing
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
        ω_initial_extras=XY_iir_pair_initial_guesses,
        method=profile_method)

    # Profile quadratic approximation
    ψω_ellipse_values, lnlike_ψ_ellipse_values = profile_target(lnlike_XY_iir_ellipse,
        target_indices_ij,
        XY_iir_lower_bounds, XY_iir_upper_bounds,
        nuisance_guess; grid_steps=grid_steps,
        ω_initial_extras=XY_iir_pair_initial_guesses,
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
        model_name, "S", "t", "t",
        data_indep=t_obs, data_dep=data,
        true_mean=true_mean_iir,
        target=current_varnames["ψ1"]*", "*current_varnames["ψ2"],
        target_save=current_varnames["ψ1_save"]*"_"*current_varnames["ψ2_save"])
end

