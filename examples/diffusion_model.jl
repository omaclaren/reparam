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
function solve_model(x, L, xy)
    y = Vector{eltype(xy)}(undef, length(x))
    mid_index = Int((length(x)-1)/2)
    β = L^2*xy[3]/8*(1/xy[2]-1/xy[1])
    α = xy[3]*L/(2*xy[2])-β/L
    Φ_1(x) = -xy[3]/(2*xy[1])*x^2 + α*x
    Φ_2(x) = -xy[3]/(2*xy[2])*x^2 + α*x + β
    for i in 1:mid_index # x=0 to x=50
        y[i] = Φ_1(x[i])
    end 
    for i in mid_index:length(x) # x=50 to x=100
        y[i] = Φ_2(x[i])
    end 
    return y
end

# --------------------------------------------------------
# Setup and Data Generation
# --------------------------------------------------------

# Grid setup
L = 100
x = LinRange(0, L, 201)
x_data = x[2:end-1]

# Distribution in original parameterization
σ = 0.2
distrib_xy = xy -> MvLogNormal(log.(abs.(solve_model(x, L, xy))[2:end-1]), σ^2*I(length(x_data)))

# Variables and bounds
varnames = Dict("ψ1" => "D_1", "ψ2" => "D_2", "ψ3" => "R")
varnames["ψ1_save"] = "D_1"
varnames["ψ2_save"] = "D_2"
varnames["ψ3_save"] = "R"

# Parameter bounds
D_1_min, D_1_max = 0.1, 5.0
D_2_min, D_2_max = 0.1, 5.0
R_min, R_max = 0.1, 5.0

xy_lower_bounds = [D_1_min, D_2_min, R_min]
xy_upper_bounds = [D_1_max, D_2_max, R_max]

# Initial guess for optimisation
xy_initial = 0.5 * (xy_lower_bounds + xy_upper_bounds)

# True parameter
D_1_true, D_2_true, R_true = 3.0, 1.0, 1.0
xy_true = [D_1_true, D_2_true, R_true]

# Generate data
Nrep = 1
data = rand(distrib_xy(xy_true), Nrep)

# Visualize data
scatter(x, [0; data; 0])

# --------------------------------------------------------
# Original Parameterization Analysis
# --------------------------------------------------------
# Construct likelihood
lnlike_xy = construct_lnlike_xy(distrib_xy, data; dist_type=:multi)
model_name = "diffusion_xy"
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
    # nuisance_guess = xy_initial[nuisance_indices] # can use xy_initial as well

    # Print variable name
    print("Variable: ", varnames["ψ"*string(i)], "\n")

    # Profile full likelihood
    ψω_values, lnlike_ψ_values = profile_target(lnlike_xy, 
        target_index,
        xy_lower_bounds, 
        xy_upper_bounds,
        nuisance_guess; 
        grid_steps=grid_steps)

    # Profile quadratic approximation
    ψω_ellipse_values, lnlike_ψ_ellipse_values = profile_target(lnlike_θ_ellipse,
        target_index,
        xy_lower_bounds,
        xy_upper_bounds,
        nuisance_guess;
        grid_steps=grid_steps)

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

    # Prediction CIs
    lower_ψ, upper_ψ, _ = construct_upper_lower_profile_wise_CIs_for_mean(
        distrib_xy, ψω_values, lnlike_ψ_values; l_level=95, df=1)

    plot_profile_wise_CI_for_mean(
        x_data, lower_ψ, upper_ψ, pred_mean_MLE,
        model_name, "x", "x",
        data=data, true_mean=true_mean,
        target=varnames["ψ"*string(i)])
end

# 2D Profiles
param_pairs = [(i,j) for i in 1:dim_all for j in (i+1):dim_all]

for (i,j) in param_pairs
    target_indices_ij = [i,j]
    nuisance_indices = setdiff(indices_all, target_indices_ij)
    nuisance_guess = θ_MLE[nuisance_indices]
    # nuisance_guess = xy_initial[nuisance_indices] # can use xy_initial as well
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
        nuisance_guess; grid_steps=grid_steps)

    # Profile quadratic approximation
    ψω_ellipse_values, lnlike_ψ_ellipse_values = profile_target(lnlike_θ_ellipse,
        target_indices_ij,
        xy_lower_bounds,
        xy_upper_bounds,
        nuisance_guess;
        grid_steps=grid_steps)

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

    # 2D prediction CIs
    lower_ψ1ψ2, upper_ψ1ψ2, _ = construct_upper_lower_profile_wise_CIs_for_mean(
        distrib_xy, ψω_values, lnlike_ψ_values; l_level=95, df=2)

    plot_profile_wise_CI_for_mean(
        x_data, lower_ψ1ψ2, upper_ψ1ψ2, pred_mean_MLE,
        model_name, "x", "x",
        data=data, true_mean=true_mean,
        target=current_varnames["ψ1"]*"_"*current_varnames["ψ2"])
end

# --------------------------------------------------------
# Log Parameterization Analysis
# --------------------------------------------------------
model_name = "diffusion_log"

# Coordinate transformation
xytoXY_log(xy) = log.(xy)
XYtoxy_log(XY) = exp.(XY)

# Transform bounds and parameters
XY_log_lower_bounds = log.(xy_lower_bounds)
XY_log_upper_bounds = log.(xy_upper_bounds)
XY_log_initial = xytoXY_log(xy_initial)
XY_log_true = xytoXY_log(xy_true)

# Transform likelihood and distribution
lnlike_XY_log = construct_lnlike_XY(lnlike_xy, XYtoxy_log)
distrib_XY_log = construct_distrib_XY(distrib_xy, XYtoxy_log)

# Update variable names for log coordinates
varnames["ψ1"] = "\\ln\\ D_1"
varnames["ψ2"] = "\\ln\\ D_2"
varnames["ψ3"] = "\\ln\\ R"
varnames["ψ1_save"] = "ln_D_1"
varnames["ψ2_save"] = "ln_D_2"
varnames["ψ3_save"] = "ln_R"

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

# Calculate prediction at MLE for reference
pred_mean_MLE_log = mean(distrib_XY_log(θ_log_MLE))
true_mean_log = mean(distrib_XY_log(XY_log_true))

# 1D Profiles in log coordinates
for i in 1:dim_all
    target_index = i
    nuisance_indices = setdiff(indices_all, target_index)
    nuisance_guess = θ_log_MLE[nuisance_indices]
    # nuisance_guess = XY_log_initial[nuisance_indices] # can use XY_log_initial as well

    # Profile full likelihood
    ψω_values, lnlike_ψ_values = profile_target(lnlike_XY_log, target_index,
        XY_log_lower_bounds, XY_log_upper_bounds,
        nuisance_guess; grid_steps=grid_steps)

    # Profile quadratic approximation
    ψω_ellipse_values, lnlike_ψ_ellipse_values = profile_target(lnlike_θ_log_ellipse,
        target_index,
        XY_log_lower_bounds,
        XY_log_upper_bounds,
        nuisance_guess;
        grid_steps=grid_steps)

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

    # Prediction CIs
    lower_ψ, upper_ψ, _ = construct_upper_lower_profile_wise_CIs_for_mean(
        distrib_XY_log, ψω_values, lnlike_ψ_values; l_level=95, df=1)

    plot_profile_wise_CI_for_mean(
        x_data, lower_ψ, upper_ψ, pred_mean_MLE_log,
        model_name, "x", "x",
        data=data, true_mean=true_mean_log,
        target=varnames["ψ"*string(i)])
end

# 2D Profiles in log coordinates
param_pairs = [(i,j) for i in 1:dim_all for j in (i+1):dim_all]

for (i,j) in param_pairs
    target_indices_ij = [i,j]
    nuisance_indices = setdiff(indices_all, target_indices_ij)
    nuisance_guess = θ_log_MLE[nuisance_indices]
    # nuisance_guess = XY_log_initial[nuisance_indices] # can use XY_log_initial as well
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
        nuisance_guess; grid_steps=grid_steps)

    # Profile quadratic approximation
    ψω_ellipse_values, lnlike_ψ_ellipse_values = profile_target(lnlike_θ_log_ellipse,
        target_indices_ij,
        XY_log_lower_bounds,
        XY_log_upper_bounds,
        nuisance_guess;
        grid_steps=grid_steps)

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

    # 2D prediction CIs
    lower_ψ1ψ2, upper_ψ1ψ2, _ = construct_upper_lower_profile_wise_CIs_for_mean(
        distrib_XY_log, ψω_values, lnlike_ψ_values; l_level=95, df=2)

    plot_profile_wise_CI_for_mean(
        x_data, lower_ψ1ψ2, upper_ψ1ψ2, pred_mean_MLE_log,
        model_name, "x", "x",
        data=data, true_mean=true_mean_log,
        target=current_varnames["ψ1"]*"_"*current_varnames["ψ2"])
end

# --------------------------------------------------------
# Sloppihood-Informed Parameterization Analysis
# --------------------------------------------------------
model_name = "diffusion_sip"

# Scale and round eigenvectors for SIP transformation
evecs_scaled = scale_and_round(evecs_log; round_within=0.5, column_scales=[1,1,1])
println("Transformations:")
display(evecs_scaled)
display(inv(evecs_scaled))

# Construct transformation
xytoXY_sip, XYtoxy_sip = reparam(evecs_scaled)

# Transform likelihood and distribution
lnlike_XY_sip = construct_lnlike_XY(lnlike_xy, XYtoxy_sip)
distrib_XY_sip = construct_distrib_XY(distrib_xy, XYtoxy_sip)

# Set bounds for SIP coordinates (manual due to non-monotonic transform)
XY_sip_lower_bounds = [0.5, 0.0001, 0.00001]
XY_sip_upper_bounds = [1.5, 10, 1000]
XY_sip_initial = [1, 1, 10]
XY_sip_true = xytoXY_sip(xy_true)

# Update variable names for SIP coordinates
varnames["ψ1"] = "\\frac{D_2}{R}"
varnames["ψ2"] = "\\frac{D_1}{\\sqrt{D_2R}}"
varnames["ψ3"] = "D_1 D_2 R"
varnames["ψ1_save"] = "D_2_over_R"
varnames["ψ2_save"] = "D_1_over_sqrt_D_2_R"
varnames["ψ3_save"] = "D_1_D_2_R"

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

# Calculate prediction at MLE for reference
pred_mean_MLE_sip = mean(distrib_XY_sip(θ_sip_MLE))
true_mean_sip = mean(distrib_XY_sip(XY_sip_true))

# 1D Profiles in SIP coordinates
for i in 1:dim_all
    target_index = i
    nuisance_indices = setdiff(indices_all, target_index)
    nuisance_guess = θ_sip_MLE[nuisance_indices]
    # nuisance_guess = XY_sip_initial[nuisance_indices] # can use XY_sip_initial as well

    # Profile full likelihood
    ψω_values, lnlike_ψ_values = profile_target(lnlike_XY_sip, target_index,
        XY_sip_lower_bounds, XY_sip_upper_bounds,
        nuisance_guess; grid_steps=grid_steps)

    # Profile quadratic approximation
    ψω_ellipse_values, lnlike_ψ_ellipse_values = profile_target(lnlike_θ_sip_ellipse,
        target_index,
        XY_sip_lower_bounds,
        XY_sip_upper_bounds,
        nuisance_guess;
        grid_steps=grid_steps)

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

    # Prediction CIs
    lower_ψ, upper_ψ, _ = construct_upper_lower_profile_wise_CIs_for_mean(
        distrib_XY_sip, ψω_values, lnlike_ψ_values; l_level=95, df=1)

    plot_profile_wise_CI_for_mean(
        x_data, lower_ψ, upper_ψ, pred_mean_MLE_sip,
        model_name, "x", "x",
        data=data, true_mean=true_mean_sip,
        target=varnames["ψ"*string(i)])
end

# 2D Profiles in SIP coordinates
for (i,j) in param_pairs
    target_indices_ij = [i,j]
    nuisance_indices = setdiff(indices_all, target_indices_ij)
    nuisance_guess = θ_sip_MLE[nuisance_indices]
    # nuisance_guess = XY_sip_initial[nuisance_indices] # can use XY_sip_initial as well
    ψ_true_pair = XY_sip_true[target_indices_ij]

    # Create a copy of varnames for this iteration
    current_varnames = deepcopy(varnames)

    # Update varnames for this pair
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
        XY_sip_lower_bounds,
        XY_sip_upper_bounds,
        nuisance_guess;
        grid_steps=grid_steps)

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

    # 2D prediction CIs
    lower_ψ1ψ2, upper_ψ1ψ2, _ = construct_upper_lower_profile_wise_CIs_for_mean(
        distrib_XY_sip, ψω_values, lnlike_ψ_values; l_level=95, df=2)

    plot_profile_wise_CI_for_mean(
        x_data, lower_ψ1ψ2, upper_ψ1ψ2, pred_mean_MLE_sip,
        model_name, "x", "x",
        data=data, true_mean=true_mean_sip,
        target=current_varnames["ψ1"]*"_"*current_varnames["ψ2"])
end