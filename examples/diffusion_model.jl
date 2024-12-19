include("../ReparamTools.jl")
using .ReparamTools
using Plots
using Distributions
using LinearAlgebra

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

# Run analysis workflow
model_name = "diffusion_xy"
grid_steps = [500]

θ_MLE, evals, evecs = execute_model_analysis_workflow_up_to_2D_profiles(
    model_name,
    varnames,
    lnlike_xy,
    xy_lower_bounds,
    xy_upper_bounds,
    xy_initial;
    grid_steps=grid_steps,
    ψ_true=xy_true,
    return_info=true
)

# --------------------------------------------------------
# Log Parameterization Analysis
# --------------------------------------------------------
model_name = "diffusion_log"

# Coordinate transformation
xytoXY_log(xy) = log.(xy)
XYtoxy_log(XY) = exp.(XY)

# Update variable names for log coordinates
varnames["ψ1"] = "\\ln\\ D_1"
varnames["ψ2"] = "\\ln\\ D_2"
varnames["ψ3"] = "\\ln\\ R"
varnames["ψ1_save"] = "ln_D_1"
varnames["ψ2_save"] = "ln_D_2"
varnames["ψ3_save"] = "ln_R"

# Transform bounds and parameters
XY_log_lower_bounds = log.(xy_lower_bounds)
XY_log_upper_bounds = log.(xy_upper_bounds)
XY_log_initial = xytoXY_log(xy_initial)
XY_log_true = xytoXY_log(xy_true)

# Transform likelihood
lnlike_XY_log = construct_lnlike_XY(lnlike_xy, XYtoxy_log)

# Run analysis in log coordinates
θ_log_MLE, evals_log, evecs_log = execute_model_analysis_workflow_up_to_2D_profiles(
    model_name,
    varnames,
    lnlike_XY_log,
    XY_log_lower_bounds,
    XY_log_upper_bounds,
    XY_log_initial;
    grid_steps=grid_steps,
    ψ_true=XY_log_true,
    return_info=true
)

# --------------------------------------------------------
# Sloppihood-Informed Parameterization Analysis
# --------------------------------------------------------
model_name = "diffusion_sip"

# Scale and round eigenvectors
evecs_scaled = scale_and_round(evecs_log; round_within=0.5, column_scales=[1,1,1])
println("Transformations:")
display(evecs_scaled)
display(inv(evecs_scaled))

# Construct transformation
xytoXY_sip, XYtoxy_sip = reparam(evecs_scaled)

# Update variable names for SIP coordinates
varnames["ψ1"] = "\\frac{D_2}{R}"
varnames["ψ2"] = "\\frac{D_1}{\\sqrt{D_2R}}"
varnames["ψ3"] = "D_1 D_2 R"
varnames["ψ1_save"] = "D_2_over_R"
varnames["ψ2_save"] = "D_1_over_sqrt_D_2_R"
varnames["ψ3_save"] = "D_1_D_2_R"

# Transform likelihood and distribution
lnlike_XY_sip = construct_lnlike_XY(lnlike_xy, XYtoxy_sip)
distrib_XY_sip = construct_distrib_XY(distrib_xy, XYtoxy_sip)

# Set bounds for SIP coordinates (manual due to non-monotonic transform)
XY_sip_lower_bounds = [0.5, 0.0001, 0.00001]
XY_sip_upper_bounds = [1.5, 10, 1000]
XY_sip_initial = [1, 1, 10]
XY_sip_true = xytoXY_sip(xy_true)

# Run analysis in SIP coordinates
execute_model_analysis_workflow_up_to_2D_profiles(
    model_name,
    varnames,
    lnlike_XY_sip,
    XY_sip_lower_bounds,
    XY_sip_upper_bounds,
    XY_sip_initial;
    grid_steps=grid_steps,
    ψ_true=XY_sip_true
)