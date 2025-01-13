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
Random.seed!(1)

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
σ = 0.1

# --------------------------------------------------------
# --- Analysis in original parameterisation and data generation ---
# --------------------------------------------------------
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

