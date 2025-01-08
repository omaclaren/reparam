include("../ReparamTools.jl")
using .ReparamTools
using Plots
using Distributions
using LinearAlgebra
using Random

# Set random seed for reproducibility
Random.seed!(12)

# --------------------------------------------------------
# Model Definition
# --------------------------------------------------------
# Parameter -> data parameter mapping
ϕ_xy = xy -> [xy[1]*xy[2], xy[2]]  # n and p, forward mapping

# Parameter -> distribution mapping. Could use Φ_xy
distrib_xy = xy -> Normal(xy[1]*xy[2], sqrt(xy[1]*xy[2]*(1-xy[2])))

# --------------------------------------------------------
# Setup and Data Generation
# --------------------------------------------------------
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
xy_initial = [50.0, 0.3]  # n and p starting guesses

# True parameter values
n_true, p_true = 100.0, 0.2
xy_true = [n_true, p_true]

# Generate/use data
# Option 1: Generate synthetic data
# N_samples = 50
# data = rand(distrib_xy(xy_true), N_samples)

# Option 2: Use previously generated data for reproducibility
data = [21.9, 22.3, 12.8, 16.4, 16.4, 20.3, 16.2, 20.0, 19.7, 24.4]

# --------------------------------------------------------
# Original Parameterization Analysis
# --------------------------------------------------------
# Construct likelihood
lnlike_xy = construct_lnlike_xy(distrib_xy, data)
model_name = "stat_model_xy"
grid_steps = [500]
dim_all = length(xy_initial)
indices_all = 1:dim_all

# Point estimation (MLE)
target_indices = []  # empty for MLE
θ_MLE, lnlike_θ_MLE = profile_target(lnlike_xy, target_indices,
    xy_lower_bounds, xy_upper_bounds, 
    xy_initial; grid_steps=grid_steps)

# Quadratic approximation at MLE
lnlike_θ_ellipse, H_θ_ellipse = construct_ellipse_lnlike_approx(lnlike_xy, θ_MLE)

# Eigenanalysis
evals, evecs = eigen(H_θ_ellipse; sortby = x -> -real(x))
println("Eigenvectors and eigenvalues for "*model_name)
for (i, eveci) in enumerate(eachcol(evecs))
    println("value: ", evals[i])
    println("vector: ", evecs[:,i])
end

# Calculate prediction at MLE for reference
pred_mean_MLE = mean(distrib_xy(θ_MLE))
true_mean = mean(distrib_xy(xy_true))

# 1D Profiles
for i in 1:dim_all
    target_index = i
    nuisance_indices = setdiff(indices_all, target_index)
    nuisance_guess = θ_MLE[nuisance_indices]

    print("Variable: ", varnames["ψ"*string(i)], "\n")

    # Profile full likelihood
    ψω_values, lnlike_ψ_values = profile_target(lnlike_xy, target_index,
        xy_lower_bounds, xy_upper_bounds,
        nuisance_guess; grid_steps=grid_steps)

    # Profile quadratic approximation
    ψω_ellipse_values, lnlike_ψ_ellipse_values = profile_target(lnlike_θ_ellipse,
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
    nuisance_guess = θ_MLE[nuisance_indices]
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
    ψω_ellipse_values, lnlike_ψ_ellipse_values = profile_target(lnlike_θ_ellipse,
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
model_name = "stat_model_log"

# Coordinate transformation
xytoXY_log(xy) = log.(xy)
XYtoxy_log(XY) = exp.(XY)

# Transform bounds and parameters
XY_log_lower_bounds = log.(xy_lower_bounds)
XY_log_upper_bounds = log.(xy_upper_bounds)
XY_log_initial = xytoXY_log(xy_initial)
XY_log_true = xytoXY_log(xy_true)

# Transform likelihood
lnlike_XY_log = construct_lnlike_XY(lnlike_xy, XYtoxy_log)

# Update variable names for log coordinates
varnames["ψ1"] = "\\ln\\ n"
varnames["ψ2"] = "\\ln\\ p"
varnames["ψ1_save"] = "ln_n"
varnames["ψ2_save"] = "ln_p"

# Point estimation in log coordinates
target_indices = []  # empty for MLE
θ_log_MLE, lnlike_θ_log_MLE = profile_target(lnlike_XY_log, target_indices,
    XY_log_lower_bounds, XY_log_upper_bounds, 
    XY_log_initial; grid_steps=grid_steps)

# Quadratic approximation at MLE
lnlike_θ_log_ellipse, H_θ_log_ellipse = construct_ellipse_lnlike_approx(lnlike_XY_log, θ_log_MLE)

# Eigenanalysis in log coordinates
evals_log, evecs_log = eigen(H_θ_log_ellipse; sortby = x -> -real(x))
println("Eigenvectors and eigenvalues for "*model_name)
for (i, eveci) in enumerate(eachcol(evecs_log))
    println("value: ", evals_log[i])
    println("vector: ", evecs_log[:,i])
end

# 1D Profiles in log coordinates
for i in 1:dim_all
    target_index = i
    nuisance_indices = setdiff(indices_all, target_index)
    nuisance_guess = θ_log_MLE[nuisance_indices]

    print("Variable: ", varnames["ψ"*string(i)], "\n")

    # Profile full likelihood
    ψω_values, lnlike_ψ_values = profile_target(lnlike_XY_log, target_index,
        XY_log_lower_bounds, XY_log_upper_bounds,
        nuisance_guess; grid_steps=grid_steps)

    # Profile quadratic approximation
    ψω_ellipse_values, lnlike_ψ_ellipse_values = profile_target(lnlike_θ_log_ellipse,
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

# 2D Profiles in log coordinates
for (i,j) in param_pairs
    target_indices_ij = [i,j]
    nuisance_indices = setdiff(indices_all, target_indices_ij)
    nuisance_guess = θ_log_MLE[nuisance_indices]
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
    ψω_ellipse_values, lnlike_ψ_ellipse_values = profile_target(lnlike_θ_log_ellipse,
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
model_name = "stat_model_sip"

# Scale and round eigenvectors for SIP transformation
evecs_scaled = scale_and_round(evecs_log; column_scales=[1,1])
println("Transformations:")
display(evecs_scaled)
display(inv(evecs_scaled))

# Construct transformation
xytoXY_sip, XYtoxy_sip = reparam(evecs_scaled)

# Transform likelihood
lnlike_XY_sip = construct_lnlike_XY(lnlike_xy, XYtoxy_sip)

# Set bounds for SIP coordinates
XY_sip_lower_bounds = [13.0, 25.0]
XY_sip_upper_bounds = [25.0, 1000.0]
XY_sip_initial = [mean([XY_sip_lower_bounds[1], XY_sip_upper_bounds[1]]),
                  mean([XY_sip_lower_bounds[2], XY_sip_upper_bounds[2]])]
XY_sip_true = xytoXY_sip(xy_true)

# Update variable names for SIP coordinates. 
# TODO whether to use finite data evecs or infinite data evecs
varnames["ψ1"] = "np"
varnames["ψ2"] = "\\frac{n}{p}"
varnames["ψ1_save"] = "np"
varnames["ψ2_save"] = "n_over_p"

# Point estimation in SIP coordinates
target_indices = []  # empty for MLE
θ_sip_MLE, lnlike_θ_sip_MLE = profile_target(lnlike_XY_sip, target_indices,
    XY_sip_lower_bounds, XY_sip_upper_bounds, 
    XY_sip_initial; grid_steps=grid_steps)

# Quadratic approximation at MLE
lnlike_θ_sip_ellipse, H_θ_sip_ellipse = construct_ellipse_lnlike_approx(lnlike_XY_sip, θ_sip_MLE)

# Eigenanalysis in SIP coordinates
evals_sip, evecs_sip = eigen(H_θ_sip_ellipse; sortby = x -> -real(x))
println("Eigenvectors and eigenvalues for "*model_name)
for (i, eveci) in enumerate(eachcol(evecs_sip))
    println("value: ", evals_sip[i])
    println("vector: ", evecs_sip[:,i])
end

# 1D Profiles in SIP coordinates
for i in 1:dim_all
    target_index = i
    nuisance_indices = setdiff(indices_all, target_index)
    nuisance_guess = θ_sip_MLE[nuisance_indices]

    print("Variable: ", varnames["ψ"*string(i)], "\n")

    # Profile full likelihood
    ψω_values, lnlike_ψ_values = profile_target(lnlike_XY_sip, target_index,
        XY_sip_lower_bounds, XY_sip_upper_bounds,
        nuisance_guess; grid_steps=grid_steps)

    # Profile quadratic approximation
    ψω_ellipse_values, lnlike_ψ_ellipse_values = profile_target(lnlike_θ_sip_ellipse,
        target_index,
        XY_sip_lower_bounds, XY_sip_upper_bounds,
        nuisance_guess; grid_steps=grid_steps)

    # Extract profiled parameter values
    ψ_values = [ψω[target_index] for ψω in ψω_values]
    ψ_ellipse_values = [ψω[target_index] for ψω in ψω_ellipse_values]

    # Plot profiles
    plot_1D_profile(model_name, ψ_values, lnlike_ψ_values,
        varnames["ψ"*string(i)];
        varname_save=varnames["ψ"*string(i)*"_save"],
        ψ_true=XY_sip_true[i])

    plot_1D_profile_comparison(model_name, model_name*"_ellipse",
        ψ_values, ψ_ellipse_values,
        lnlike_ψ_values, lnlike_ψ_ellipse_values,
        varnames["ψ"*string(i)];
        varname_save=varnames["ψ"*string(i)*"_save"],
        ψ_true=XY_sip_true[i])
end

# 2D Profiles in SIP coordinates
for (i,j) in param_pairs
    target_indices_ij = [i,j]
    nuisance_indices = setdiff(indices_all, target_indices_ij)
    nuisance_guess = θ_sip_MLE[nuisance_indices]
    ψ_true_pair = XY_sip_true[target_indices_ij]

    # Create a copy of varnames for this iteration
    current_varnames = deepcopy(varnames)
    current_varnames["ψ1"] = varnames["ψ"*string(i)]
    current_varnames["ψ2"] = varnames["ψ"*string(j)]
    current_varnames["ψ1_save"] = varnames["ψ"*string(i)*"_save"]
    current_varnames["ψ2_save"] = varnames["ψ"*string(j)*"_save"]

    # Profile full likelihood
    ψω_values, lnlike_ψ_values = profile_target(lnlike_XY_sip, target_indices_ij,
        XY_sip_lower_bounds, XY_sip_upper_bounds,
        nuisance_guess; grid_steps=grid_steps)

    # Profile quadratic approximation
    ψω_ellipse_values, lnlike_ψ_ellipse_values = profile_target(lnlike_θ_sip_ellipse,
        target_indices_ij,
        XY_sip_lower_bounds, XY_sip_upper_bounds,
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