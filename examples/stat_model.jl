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
Random.seed!(12)

# --------------------------------------------------------
# Model Definition
# Define model in xy = θ = [n, p] parameterization
# --------------------------------------------------------
# boolean for whether to use Poisson limit
poisson_limit = true
# Parameter -> data parameter mapping 
if poisson_limit
    ϕ_xy = xy -> [xy[1]*xy[2], xy[1]*xy[2]] # Maps (n,p) to (np,np)
else
    ϕ_xy = xy -> [xy[1]*xy[2], xy[1]*xy[2]*(1-xy[2])]  # Maps (n,p) to (np,np*(1-p))
end

# --------------------------------------------------------
# Setup and Data Generation
# --------------------------------------------------------

# Parameter -> distribution mapping. Use ϕ_xy explicitly 
distrib_xy = xy -> Normal(ϕ_xy(xy)[1], sqrt(ϕ_xy(xy)[2]))

# Variables and bounds
varnames = Dict("ψ1" => "n", "ψ2" => "p")
varnames["ψ1_save"] = "n"
varnames["ψ2_save"] = "p"

# Parameter bounds
n_min, n_max = 0.1, 500.0
p_min, p_max = 0.0001, 1.0

xy_lower_bounds = [n_min, p_min]
xy_upper_bounds = [n_max, p_max]

# Initial guess for optimisation
xy_initial = [50.0, 0.3]

# True parameter values
n_true, p_true = 100.0, 0.2
xy_true = [n_true, p_true]

# Generate/load data
# N_samples = 10
# data = rand(distrib_xy(xy_true),N_samples)
data = [21.9, 22.3, 12.8, 16.4, 16.4, 20.3, 16.2, 20.0, 19.7, 24.4]

# --------------------------------------------------------
# Original Parameterization Analysis
# --------------------------------------------------------
# Construct likelihood
lnlike_xy = construct_lnlike_xy(distrib_xy, data)
if poisson_limit
    model_name = "stat_model_xy_poisson"
else
    model_name = "stat_model_xy"
end
grid_steps = [500]
dim_all = length(xy_initial)
indices_all = 1:dim_all

# Point estimation (MLE)
target_indices = []  # empty for MLE
xy_MLE, lnlike_xy_MLE = profile_target(lnlike_xy, target_indices,
    xy_lower_bounds, xy_upper_bounds, 
    xy_initial; grid_steps=grid_steps)

# Quadratic approximation at MLE
lnlike_xy_ellipse, H_xy_ellipse = construct_ellipse_lnlike_approx(lnlike_xy, xy_MLE)

# Eigenanalysis of Fisher Information
evals, evecs = eigen(H_xy_ellipse; sortby = x -> -real(x))
println("Eigenvectors and eigenvalues for "*model_name)
println("Eigenvalues: ", evals)
println("Eigenvectors: ", evecs)

# Calculate prediction at MLE for reference
pred_mean_MLE = mean(distrib_xy(xy_MLE))
true_mean = mean(distrib_xy(xy_true))

# Determine svd of phi mapping in xy coordinates
J_ϕ_xy = compute_ϕ_Jacobian(ϕ_xy, xy_MLE)
U_xy, S_xy, Vt_xy = svd(J_ϕ_xy)
println("\nSVD analysis in original coordinates:")
println("Singular values: ", S_xy)
println("Right singular vectors (V): ")
display(Vt_xy)

# 1D Profiles
for i in 1:dim_all
    target_index = i
    nuisance_indices = setdiff(indices_all, target_index)
    nuisance_guess = xy_MLE[nuisance_indices]

    print("Variable: ", varnames["ψ"*string(i)], "\n")

    # Profile full likelihood
    ψω_values, lnlike_ψ_values = profile_target(lnlike_xy, target_index,
        xy_lower_bounds, xy_upper_bounds,
        nuisance_guess; grid_steps=grid_steps)

    # Profile quadratic approximation
    ψω_ellipse_values, lnlike_ψ_ellipse_values = profile_target(lnlike_xy_ellipse,
        target_index,
        xy_lower_bounds, xy_upper_bounds,
        nuisance_guess; grid_steps=grid_steps)

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
end

# 2D Profiles
param_pairs = [(i,j) for i in 1:dim_all for j in (i+1):dim_all]

for (i,j) in param_pairs
    target_indices_ij = [i,j]
    nuisance_indices = setdiff(indices_all, target_indices_ij)
    nuisance_guess = xy_MLE[nuisance_indices]
    ψ_true_pair = xy_true[target_indices_ij]

    # Create a copy of varnames for this iteration
    current_varnames = deepcopy(varnames)
    current_varnames["ψ1"] = varnames["ψ"*string(i)]
    current_varnames["ψ2"] = varnames["ψ"*string(j)]
    current_varnames["ψ1_save"] = varnames["ψ"*string(i)*"_save"]
    current_varnames["ψ2_save"] = varnames["ψ"*string(j)*"_save"]

    # Profile full likelihood
    ψω_values, lnlike_ψ_values = profile_target(lnlike_xy, target_indices_ij,
        xy_lower_bounds, xy_upper_bounds,
        nuisance_guess; grid_steps=grid_steps)

    # Profile quadratic approximation
    ψω_ellipse_values, lnlike_ψ_ellipse_values = profile_target(lnlike_xy_ellipse,
        target_indices_ij,
        xy_lower_bounds, xy_upper_bounds,
        nuisance_guess; grid_steps=grid_steps)

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
end

# --------------------------------------------------------
# Log Parameterization Analysis
# --------------------------------------------------------

if poisson_limit
    model_name = "stat_model_log_poisson"
else
    model_name = "stat_model_log"
end

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
distrib_XY_log = construct_distrib_XY(distrib_xy, XYtoxy_log)
ϕ_XY_log = construct_ϕ_XY(ϕ_xy, XYtoxy_log)

# Update variable names for log coordinates
varnames["ψ1"] = "\\ln\\ n"
varnames["ψ2"] = "\\ln\\ p"
varnames["ψ1_save"] = "ln_n"
varnames["ψ2_save"] = "ln_p"

# Point estimation in log coordinates
target_indices = []  # empty for MLE
XY_log_MLE, lnlike_XY_log_MLE = profile_target(lnlike_XY_log, target_indices,
    XY_log_lower_bounds, XY_log_upper_bounds, 
    XY_log_initial; grid_steps=grid_steps)

# Quadratic approximation at MLE
lnlike_XY_log_ellipse, H_XY_log_ellipse = construct_ellipse_lnlike_approx(lnlike_XY_log, XY_log_MLE)

# Eigenanalysis in log coordinates
evals_log, evecs_log = eigen(H_XY_log_ellipse; sortby = x -> -real(x))
println("Eigenvectors and eigenvalues for "*model_name)
for (i, eveci) in enumerate(eachcol(evecs_log))
    println("value: ", evals_log[i])
    println("vector: ", evecs_log[:,i])
end

# Determine svd of phi mapping in log coordinates
J_ϕ_XY_log, U_XY_log, S_XY_log, Vt_XY_log = compute_ϕ_Jacobian(ϕ_XY_log, XY_log_MLE, compute_svd=true)
println("\nSVD analysis in log coordinates:")
println("Singular values: ", S_XY_log)
println("Right singular vectors (V): ")
display(Vt_XY_log)

# Compare eigenvectors from Fisher Information with singular vectors
println("\nComparison of eigenvectors (1) and singular vectors (2):")
display(evecs_log)
display(Vt_XY_log)

for i in 1:dim_all
    target_index = i
    nuisance_indices = setdiff(indices_all, target_index)
    nuisance_guess = XY_log_MLE[nuisance_indices]

    print("Variable: ", varnames["ψ"*string(i)], "\n")

    # Profile full likelihood
    ψω_values, lnlike_ψ_values = profile_target(lnlike_XY_log, target_index,
        XY_log_lower_bounds, XY_log_upper_bounds,
        nuisance_guess; grid_steps=grid_steps)

    # Profile quadratic approximation
    ψω_ellipse_values, lnlike_ψ_ellipse_values = profile_target(lnlike_XY_log_ellipse,
        target_index,
        XY_log_lower_bounds, XY_log_upper_bounds,
        nuisance_guess; grid_steps=grid_steps)

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
end

# 2D Profiles
param_pairs = [(i,j) for i in 1:dim_all for j in (i+1):dim_all]

for (i,j) in param_pairs
    target_indices_ij = [i,j]
    nuisance_indices = setdiff(indices_all, target_indices_ij)
    nuisance_guess = XY_log_MLE[nuisance_indices]
    ψ_true_pair = XY_log_true[target_indices_ij]

    # Create a copy of varnames for this iteration
    current_varnames = deepcopy(varnames)
    current_varnames["ψ1"] = varnames["ψ"*string(i)]
    current_varnames["ψ2"] = varnames["ψ"*string(j)]
    current_varnames["ψ1_save"] = varnames["ψ"*string(i)*"_save"]
    current_varnames["ψ2_save"] = varnames["ψ"*string(j)*"_save"]

    # Profile full likelihood
    ψω_values, lnlike_ψ_values = profile_target(lnlike_XY_log, target_indices_ij,
        XY_log_lower_bounds, XY_log_upper_bounds,
        nuisance_guess; grid_steps=grid_steps)

    # Profile quadratic approximation
    ψω_ellipse_values, lnlike_ψ_ellipse_values = profile_target(lnlike_XY_log_ellipse,
        target_indices_ij,
        XY_log_lower_bounds, XY_log_upper_bounds,
        nuisance_guess; grid_steps=grid_steps)

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
end

# --------------------------------------------------------
# Sloppihood-Informed Parameterization Analysis
# --------------------------------------------------------
if poisson_limit
    model_name = "stat_model_iir_poisson"
else
    model_name = "stat_model_iir"
end
println(model_name)

# Scale and round eigenvectors for iir transformation
# Option 1: based on eigenvectors from Fisher Information
# Option 2: based on the right singular vectors from the phi mapping
use_singular_vectors = true
if use_singular_vectors
    evecs_scaled = scale_and_round(Vt_XY_log; column_scales=[1,1]) 
else
    evals_scaled = scale_and_round(evecs_log; column_scales=[1,1])
end
# evecs_scaled = scale_and_round(evecs_log; column_scales=[1,1])
println("Transformations:")
display(evecs_scaled)
display(inv(evecs_scaled))

println("Original right singular vectors:")
display(Vt_XY_log)

# Construct transformation
xytoXY_iir, XYtoxy_iir = reparam(evecs_scaled)

# Transform likelihood, distribution, and phi mapping
lnlike_XY_iir = construct_lnlike_XY(lnlike_xy, XYtoxy_iir)
distrib_XY_iir = construct_distrib_XY(distrib_xy, XYtoxy_iir)
ϕ_XY_iir = construct_ϕ_XY(ϕ_xy, XYtoxy_iir)

# Set bounds for iir coordinates
XY_iir_lower_bounds = [13.0, 25.0]
XY_iir_upper_bounds = [25.0, 1000.0]
XY_iir_initial = [mean([XY_iir_lower_bounds[1], XY_iir_upper_bounds[1]]),
                  mean([XY_iir_lower_bounds[2], XY_iir_upper_bounds[2]])]

# transform true value
XY_iir_true = xytoXY_iir(xy_true)

# Update variable names for iir coordinates
varnames["ψ1"] = "np"
varnames["ψ2"] = "\\frac{n}{p}"
varnames["ψ1_save"] = "np"
varnames["ψ2_save"] = "n_over_p"

# Point estimation in iir coordinates
target_indices = []  # empty for MLE
XY_iir_MLE, lnlike_XY_iir_MLE = profile_target(lnlike_XY_iir, target_indices,
    XY_iir_lower_bounds, XY_iir_upper_bounds, 
    XY_iir_initial; grid_steps=grid_steps)

# Quadratic approximation at MLE
lnlike_XY_iir_ellipse, H_XY_iir_ellipse = construct_ellipse_lnlike_approx(lnlike_XY_iir, XY_iir_MLE)

# Eigenanalysis in iir coordinates
evals_iir, evecs_iir = eigen(H_XY_iir_ellipse; sortby = x -> -real(x))
println("Eigenvectors and eigenvalues for "*model_name)
println("Eigenvalues: ", evals_iir)
println("Eigenvectors: ", evecs_iir)

# Determine svd of phi mapping in iir coordinates
J_ϕ_XY_iir, U_XY_iir, S_XY_iir, Vt_XY_iir = compute_ϕ_Jacobian(ϕ_XY_iir, XY_iir_MLE, compute_svd=true)

# Compare eigenvectors from Fisher Information with singular vectors
println("\nComparison of eigenvectors (1) and singular vectors (2):")
display(evecs_iir)
display(Vt_XY_iir')

# 1D Profiles
for i in 1:dim_all
    target_index = i
    nuisance_indices = setdiff(indices_all, target_index)
    nuisance_guess = XY_iir_MLE[nuisance_indices]

    print("Variable: ", varnames["ψ"*string(i)], "\n")

    # Profile full likelihood
    ψω_values, lnlike_ψ_values = profile_target(lnlike_XY_iir, target_index,
        XY_iir_lower_bounds, XY_iir_upper_bounds,
        nuisance_guess; grid_steps=grid_steps)

    # Profile quadratic approximation
    ψω_ellipse_values, lnlike_ψ_ellipse_values = profile_target(lnlike_XY_iir_ellipse,
        target_index,
        XY_iir_lower_bounds, XY_iir_upper_bounds,
        nuisance_guess; grid_steps=grid_steps)

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
end

# 2D Profiles
param_pairs = [(i,j) for i in 1:dim_all for j in (i+1):dim_all]

for (i,j) in param_pairs
    target_indices_ij = [i,j]
    nuisance_indices = setdiff(indices_all, target_indices_ij)
    nuisance_guess = XY_iir_MLE[nuisance_indices]
    ψ_true_pair = XY_iir_true[target_indices_ij]

    # Create a copy of varnames for this iteration
    current_varnames = deepcopy(varnames)
    current_varnames["ψ1"] = varnames["ψ"*string(i)]
    current_varnames["ψ2"] = varnames["ψ"*string(j)]
    current_varnames["ψ1_save"] = varnames["ψ"*string(i)*"_save"]
    current_varnames["ψ2_save"] = varnames["ψ"*string(j)*"_save"]

    # Profile full likelihood
    ψω_values, lnlike_ψ_values = profile_target(lnlike_XY_iir, target_indices_ij,
        XY_iir_lower_bounds, XY_iir_upper_bounds,
        nuisance_guess; grid_steps=grid_steps)

    # Profile quadratic approximation
    ψω_ellipse_values, lnlike_ψ_ellipse_values = profile_target(lnlike_XY_iir_ellipse,
        target_indices_ij,
        XY_iir_lower_bounds, XY_iir_upper_bounds,
        nuisance_guess; grid_steps=grid_steps)

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
end