using Plots
using Distributions
using NLopt
using ForwardDiff
using LinearAlgebra
using DifferentialEquations
using Interpolations

# ---------------------------------------------
# ---------- Load 'Sloppihood' tools ----------
# ---------------------------------------------
include("SloppihoodTools.jl")
using .SloppihoodTools

# ---------------------------------------------
# ---- User inputs in original 'theta' param ----
# ---------------------------------------------
# Define model: map from mechanstic params to data distribution parameters
function solve_model(x,L,xy)
    # y=zeros(length(x))
    # ensure y can handle dual numbers
    y = Vector{eltype(xy)}(undef, length(x))
    mid_index = Int((length(x)-1)/2)
    β=L^2*xy[3]/8*(1/xy[2]-1/xy[1])
    α=xy[3]*L/(2*xy[2])-β/L
    Φ_1(x)=-xy[3]/(2*xy[1])*x^2 + α*x
    Φ_2(x)=-xy[3]/(2*xy[2])*x^2 + α*x + β
    for i in 1:mid_index # x=0 to x=50
        y[i] = Φ_1(x[i])
    end 
    for i in mid_index:length(x) # x=50 to x=100
        y[i] = Φ_2(x[i])
    end 
    return y
end;

# parameter -> data dist (forward) mapping
L=100
x=LinRange(0,L,201)
x_data = x[2:end-1]
# distrib_xy(xy) = MvLogNormal(log.(abs.(solve_model(x_data,L,xy))),σ^2*I(length(x_data)))
distrib_xy(xy) =  MvLogNormal(log.(abs.(solve_model(x,L,xy))[2:end-1]),σ^2*I(length(x_data)))

# distrib_xy(xy) = MultivariateNormal(solve_ode(t_data,xy;solver=solver),σ^2*I(length(t_data)))

# variables and bounds
varnames = Dict("ψ1"=>"D_1", "ψ2"=>"D_2", "ψ3"=>"R")
varnames["ψ1_save"]="D_1"
varnames["ψ2_save"]="D_2"
varnames["ψ3_save"]="R"

# parameter bounds
D_1_min=0.1; D_1_max=5.0
D_2_min=0.1; D_2_max=5.0
R_min=0.1; R_max=5.0

xy_lower_bounds = [D_1_min,D_2_min,R_min]
xy_upper_bounds = [D_1_max,D_2_max,R_max]


# initial guess for optimisation
xy_initial =  0.5*(xy_lower_bounds + xy_upper_bounds) 

# true parameter
D_1_true=3.0; D_2_true=1.0; R_true=1.0;
xy_true = [D_1_true,D_2_true,R_true] 


# generate data
σ=0.2

Nrep = 1
data = rand(distrib_xy(xy_true),Nrep)



#data = [0.983615  0.695977  0.441759  0.302958  0.193174  0.130742  0.0871124  0.04112  0.0269384  0.0331469]
scatter(x,[0;data;0])

# ---- use above to construct log likelihood in original parameterisation given (iid) data
lnlike_xy = SloppihoodTools.construct_lnlike_xy(distrib_xy,data;dist_type=:multi)
# ----

# ---------------------------------------------
# --- Analysis in original parameterisation ---
# ---------------------------------------------
# grid sizes for profiling
model_name="diffusion_xy"
grid_steps = [500]
# carries out 2D analysis first, then each 1D profile, and plots
# SloppihoodTools.execute_model_analysis_workflow_2D_profile("diffusion_xy",varnames,lnlike_xy,xy_lower_bounds,xy_upper_bounds,xy_initial;grid_steps=grid_steps,ψ_true=xy_true)

dim_all = length(xy_initial)
indices_all = 1:dim_all # including higher dim nuisance if needed
# --- point optimisation in original via profiling out all parameters
target_indices = [] # for point estimation whole parameter is nuisance!
nuisance_guess = xy_initial 
# below 'profiles out' all parameters = does MLE
xy_MLE, lnlike_xy_MLE = SloppihoodTools.profile_target(lnlike_xy,target_indices,xy_lower_bounds,xy_upper_bounds,nuisance_guess;grid_steps=grid_steps)
# ---
xy_centre = xy_MLE
lnlike_xy_ellipse, H_xy_ellipse = SloppihoodTools.construct_ellipse_lnlike_approx(lnlike_xy,xy_centre)

#ForwardDiff.hessian(lnlike_xy, xy_centre)

# --- find eigenvectors (SVD of observed likelihood)
evals, evecs = eigen(H_xy_ellipse; sortby = x -> -real(x))
println("Eigenvectors and eigenvalues for "*model_name)
for (i,eveci) in enumerate(eachcol(evecs))
    evecs[:,i] = eveci 
    println("value:")
    println(evals[i])
    println("vector:")
    println(evecs[:,i])
end

SloppihoodTools.scale_and_round(evecs,round_within=0.5,column_scales=[1,1,1])

# --- 1D profiles 
# - First ψ1 component 
# based on full likelihood
target_indices = 1 # 
nuisance_indices = setdiff(indices_all,target_indices)
nuisance_guess = xy_initial[nuisance_indices]
# gets interest and nuisance values along profile in θ form
ψ1ω_values, lnlike_ψ1_values = SloppihoodTools.profile_target(lnlike_xy,target_indices,xy_lower_bounds,xy_upper_bounds,nuisance_guess;use_last_as_guess=true,method=:LN_NELDERMEAD,optmaxtime=500,grid_steps=grid_steps)
# extract interest parameter
ψ1_values = [ψ1ω[target_indices] for ψ1ω in ψ1ω_values] 
# get location of max
max_indices = argmax(lnlike_ψ1_values)
ψ1_max = ψ1_values[max_indices[1]]

# Next ψ1 component from ellipse
# gets interest and nuisance values along profile. Same arguments as before except likelihood
ψ1ω_ellipse_values, lnlike_ψ1_ellipse_values = SloppihoodTools.profile_target(lnlike_xy_ellipse,target_indices,xy_lower_bounds,xy_upper_bounds,nuisance_guess;use_last_as_guess=true,method=:LN_NELDERMEAD, optmaxtime=500, grid_steps=grid_steps)
# extract interest parameter
ψ1_ellipse_values = [ψ1ω[target_indices] for ψ1ω in ψ1ω_ellipse_values] 

# Mean prediction CI
lower_ψ1, upper_ψ1, _ = SloppihoodTools.construct_upper_lower_profile_wise_CIs_for_mean(distrib_xy, ψ1ω_values, lnlike_ψ1_values; l_level=95,df=1)
pred_mean_MLE = mean(distrib_xy(xy_MLE))
true_mean = mean(distrib_xy(xy_true))
SloppihoodTools.plot_profile_wise_CI_for_mean(x_data,lower_ψ1,upper_ψ1,pred_mean_MLE,model_name,"x","x",data=data,true_mean=true_mean,target=varnames["ψ1"])

# - Now ψ2 component 
# based on full likelihood
target_indices = 2 # 
nuisance_indices = setdiff(indices_all,target_indices)
nuisance_guess = xy_initial[nuisance_indices]
# gets interest and nuisance values along profile
ψ2ω_values, lnlike_ψ2_values = SloppihoodTools.profile_target(lnlike_xy,target_indices,xy_lower_bounds,xy_upper_bounds,nuisance_guess;use_last_as_guess=true,method=:LN_NELDERMEAD, optmaxtime=500, grid_steps=grid_steps)
# extract interest parameter
ψ2_values = [ψ2ω[target_indices] for ψ2ω in ψ2ω_values] 
# get location of max
max_indices = argmax(lnlike_ψ2_values)
ψ2_max = ψ2_values[max_indices[1]] #still [1]

# Next ψ2 component from ellipse
# gets interest and nuisance values along profile. Same arguments as before except likelihood
ψ2ω_ellipse_values, lnlike_ψ2_ellipse_values = SloppihoodTools.profile_target(lnlike_xy_ellipse,target_indices,xy_lower_bounds,xy_upper_bounds,nuisance_guess;use_last_as_guess=true,method=:LN_NELDERMEAD,optmaxtime=500,grid_steps=grid_steps)
# extract interest parameter
ψ2_ellipse_values = [ψ2ω[target_indices] for ψ2ω in ψ2ω_ellipse_values] 

# Mean prediction CI
lower_ψ2, upper_ψ2, pred_matrix_ψ2 = SloppihoodTools.construct_upper_lower_profile_wise_CIs_for_mean(distrib_xy, ψ2ω_values, lnlike_ψ2_values; l_level=95,df=1)
SloppihoodTools.plot_profile_wise_CI_for_mean(x_data,lower_ψ2,upper_ψ2,pred_mean_MLE,model_name,"x","x",data=data,true_mean=true_mean,target=varnames["ψ2"])


# - Now ψ3 component 
# based on full likelihood
target_indices = 3 # 
nuisance_indices = setdiff(indices_all,target_indices)
nuisance_guess = xy_initial[nuisance_indices]
# gets interest and nuisance values along profile. In θ order?
ψ3ω_values, lnlike_ψ3_values = SloppihoodTools.profile_target(lnlike_xy,target_indices,xy_lower_bounds,xy_upper_bounds,nuisance_guess;use_last_as_guess=true,method=:LN_NELDERMEAD,optmaxtime=500,grid_steps=grid_steps)
# extract interest parameter
ψ3_values = [ψ3ω[target_indices] for ψ3ω in ψ3ω_values] 
# get location of max
max_indices = argmax(lnlike_ψ3_values)
ψ3_max = ψ3_values[max_indices[1]] #still [1]

# Next ψ3 component from ellipse
# gets interest and nuisance values along profile. Same arguments as before except likelihood
ψ3ω_ellipse_values, lnlike_ψ3_ellipse_values = SloppihoodTools.profile_target(lnlike_xy_ellipse,target_indices,xy_lower_bounds,xy_upper_bounds,nuisance_guess;use_last_as_guess=true,method=:LN_NELDERMEAD,optmaxtime=500,grid_steps=grid_steps)
# extract interest parameter
ψ3_ellipse_values = [ψ3ω[target_indices] for ψ3ω in ψ3ω_ellipse_values] 

# Mean prediction CI
lower_ψ3, upper_ψ3, pred_matrix_ψ3 = SloppihoodTools.construct_upper_lower_profile_wise_CIs_for_mean(distrib_xy, ψ3ω_values, lnlike_ψ3_values; l_level=95,df=1)
SloppihoodTools.plot_profile_wise_CI_for_mean(x_data,lower_ψ3,upper_ψ3,pred_mean_MLE,model_name,"x","x",data=data,true_mean=true_mean,target=varnames["ψ3"])


# SloppihoodTools.plot_1D_profile(model_name, ψ1_values, lnlike_ψ1_values, varnames["ψ1"];varname_save=varnames["ψ1_save"],ψ_true=xy_true[1])
# SloppihoodTools.plot_1D_profile(model_name, ψ2_values, lnlike_ψ2_values, varnames["ψ2"];varname_save=varnames["ψ2_save"],ψ_true=xy_true[2])
# SloppihoodTools.plot_1D_profile(model_name, ψ3_values, lnlike_ψ3_values, varnames["ψ3"];varname_save=varnames["ψ3_save"],ψ_true=xy_true[3])

# ---------------
# - now pariwise
# ---------------

# combo 1
target_indices = [1,2]; 
nuisance_indices = setdiff(indices_all,target_indices)
nuisance_guess = xy_initial[nuisance_indices]
ψ1ψ2_true = xy_true[target_indices]

# - 2D profile likelihood
ψ1ψ2ω_values, lnlike_ψ1ψ2_values = SloppihoodTools.profile_target(lnlike_xy,target_indices,xy_lower_bounds,xy_upper_bounds,nuisance_guess;grid_steps=grid_steps)
# extract interest parameter vector
ψ1ψ2_values = [ψ1ψ2[target_indices] for ψ1ψ2 in ψ1ψ2ω_values]
max_indices = argmax(lnlike_ψ1ψ2_values)
ψ1ψ2_MLE_grid = ψ1ψ2_values[max_indices]

SloppihoodTools.plot_2D_contour(model_name, ψ1ψ2_values, lnlike_ψ1ψ2_values, varnames;ψ_true=ψ1ψ2_true,save_dir="./", file_extension=".svg")

# - use above for 1D profiles
ψ1_values, ψ2_values, like_ψ1_values, like_ψ2_values = SloppihoodTools.get_1D_from_2D_grid(ψ1ψ2_values, lnlike_ψ1ψ2_values)
SloppihoodTools.plot_1D_profile(model_name, ψ1_values, lnlike_ψ1_values, varnames["ψ1"];varname_save=varnames["ψ1_save"],ψ_true=ψ1ψ2_true[1])
SloppihoodTools.plot_1D_profile(model_name, ψ2_values, lnlike_ψ2_values, varnames["ψ2"];varname_save=varnames["ψ2_save"],ψ_true=ψ1ψ2_true[2])

# combo 2
target_indices = [1,3]; 
nuisance_indices = setdiff(indices_all,target_indices)
nuisance_guess = xy_initial[nuisance_indices]
ψ1ψ2_true = xy_true[target_indices]
# new variable names
varnames["ψ1"]="D_1"
varnames["ψ2"]="R"
varnames["ψ1_save"]="D_1"
varnames["ψ2_save"]="R"
# - use above for last 1D profile
ψ1_values, ψ2_values, like_ψ1_values, like_ψ2_values = SloppihoodTools.get_1D_from_2D_grid(ψ1ψ2_values, lnlike_ψ1ψ2_values)
SloppihoodTools.plot_1D_profile(model_name, ψ1_values, lnlike_ψ1_values, varnames["ψ1"];varname_save=varnames["ψ1_save"],ψ_true=ψ1ψ2_true[1])
SloppihoodTools.plot_1D_profile(model_name, ψ2_values, lnlike_ψ2_values, varnames["ψ2"];varname_save=varnames["ψ2_save"],ψ_true=ψ1ψ2_true[2])


# - 2D profile likelihood
ψ1ψ2ω_values, lnlike_ψ1ψ2_values = SloppihoodTools.profile_target(lnlike_xy,target_indices,xy_lower_bounds,xy_upper_bounds,nuisance_guess;grid_steps=grid_steps)
# extract interest parameter vector
ψ1ψ2_values = [ψ1ψ2[target_indices] for ψ1ψ2 in ψ1ψ2ω_values]
max_indices = argmax(lnlike_ψ1ψ2_values)
ψ1ψ2_MLE_grid = ψ1ψ2_values[max_indices]

SloppihoodTools.plot_2D_contour(model_name, ψ1ψ2_values, lnlike_ψ1ψ2_values, varnames;ψ_true=ψ1ψ2_true,save_dir="./", file_extension=".svg")

# - use above for 1D profiles
ψ1_values, ψ2_values, like_ψ1_values, like_ψ2_values = SloppihoodTools.get_1D_from_2D_grid(ψ1ψ2_values, lnlike_ψ1ψ2_values)
SloppihoodTools.plot_1D_profile(model_name, ψ1_values, lnlike_ψ1_values, varnames["ψ1"];varname_save=varnames["ψ1_save"],ψ_true=ψ1ψ2_true[1])
SloppihoodTools.plot_1D_profile(model_name, ψ2_values, lnlike_ψ2_values, varnames["ψ2"];varname_save=varnames["ψ2_save"],ψ_true=ψ1ψ2_true[1])

# combo 3
target_indices = [2,3]; 
nuisance_indices = setdiff(indices_all,target_indices)
nuisance_guess = xy_initial[nuisance_indices]
ψ1ψ2_true = xy_true[target_indices]
# new variable names
varnames["ψ1"]="D_2"
varnames["ψ2"]="R"
varnames["ψ1_save"]="D_2"
varnames["ψ2_save"]="R"

# - 2D profile likelihood
ψ1ψ2ω_values, lnlike_ψ1ψ2_values = SloppihoodTools.profile_target(lnlike_xy,target_indices,xy_lower_bounds,xy_upper_bounds,nuisance_guess;grid_steps=grid_steps)
# extract interest parameter vector
ψ1ψ2_values = [ψ1ψ2[target_indices] for ψ1ψ2 in ψ1ψ2ω_values]
max_indices = argmax(lnlike_ψ1ψ2_values)
ψ1ψ2_MLE_grid = ψ1ψ2_values[max_indices]

SloppihoodTools.plot_2D_contour(model_name, ψ1ψ2_values, lnlike_ψ1ψ2_values, varnames;ψ_true=ψ1ψ2_true,save_dir="./", file_extension=".svg")

# - use above for 1D profiles
ψ1_values, ψ2_values, like_ψ1_values, like_ψ2_values = SloppihoodTools.get_1D_from_2D_grid(ψ1ψ2_values, lnlike_ψ1ψ2_values)
SloppihoodTools.plot_1D_profile(model_name, ψ1_values, lnlike_ψ1_values, varnames["ψ1"];varname_save=varnames["ψ1_save"],ψ_true=ψ1ψ2_true[1])
SloppihoodTools.plot_1D_profile(model_name, ψ2_values, lnlike_ψ2_values, varnames["ψ2"];varname_save=varnames["ψ2_save"],ψ_true=ψ1ψ2_true[2])


# ---------------------------------------------
# ----- Analysis in log parameterisation ------
# ---------------------------------------------
model_name="diffusion_log"
xytoXY_log(xy) = log.(xy)
XYtoxy_log(XY) = exp.(XY)
# new variable names
varnames["ψ1"]="\\ln\\ D_1"
varnames["ψ2"]="\\ln\\ D_2"
varnames["ψ3"]="\\ln\\ R"
varnames["ψ1_save"]="ln_D_1"
varnames["ψ2_save"]="ln_D_2"
varnames["ψ3_save"]="ln_R"
# parameter bounds -- can do via xytoXY or manually
XY_log_lower_bounds = log.(xy_lower_bounds)
XY_log_upper_bounds = log.(xy_upper_bounds)
# new true value
XY_log_true = xytoXY_log(xy_true)
# initial guess for optimisation -- can do via xytoXY but also easy to do manually or based on bounds
XY_log_initial =  xytoXY_log(xy_initial) 
# new likelihood
lnlike_XY_log = SloppihoodTools.construct_lnlike_XY(lnlike_xy,XYtoxy_log)

# point est
dim_all = length(XY_log_initial)
indices_all = 1:dim_all # including higher dim nuisance if needed
# --- point optimisation in original via profiling out all parameters
target_indices = [] # for point estimation whole parameter is nuisance!
nuisance_guess = XY_log_initial 
# below 'profiles out' all parameters = does MLE
XY_log_MLE, lnlike_XY_log_MLE = SloppihoodTools.profile_target(lnlike_XY_log,target_indices,XY_log_lower_bounds,XY_log_upper_bounds,nuisance_guess;grid_steps=grid_steps)
# ---
XY_log_centre = XY_log_MLE
lnlike_XY_log_ellipse, H_XY_log_ellipse = SloppihoodTools.construct_ellipse_lnlike_approx(lnlike_XY_log,XY_log_centre)

# --- find eigenvectors (SVD of observed likelihood)
evals, evecs = eigen(H_XY_log_ellipse; sortby = x -> -real(x))
println("Eigenvectors and eigenvalues for "*model_name)
for (i,eveci) in enumerate(eachcol(evecs))
    evecs[:,i] = eveci 
    println("value:")
    println(evals[i])
    println("vector:")
    println(evecs[:,i])
end
SloppihoodTools.scale_and_round(evecs,column_scales=[1,1,1])

# --- 1D profiles 
# - First ψ1 component 
# based on full likelihood
target_indices = 1 # 
nuisance_indices = setdiff(indices_all,target_indices)
nuisance_guess = XY_log_initial[nuisance_indices]
# gets interest and nuisance values along profile in θ form
ψ1ω_values, lnlike_ψ1_values = SloppihoodTools.profile_target(lnlike_XY_log,target_indices,XY_log_lower_bounds,XY_log_upper_bounds,nuisance_guess;use_last_as_guess=true,method=:LN_NELDERMEAD,grid_steps=grid_steps)
# extract interest parameter
ψ1_values = [ψ1ω[target_indices] for ψ1ω in ψ1ω_values] 
# get location of max
max_indices = argmax(lnlike_ψ1_values)
ψ1_max = ψ1_values[max_indices[1]]

# Next ψ1 component from ellipse
# gets interest and nuisance values along profile. Same arguments as before except likelihood
ψ1ω_ellipse_values, lnlike_ψ1_ellipse_values = SloppihoodTools.profile_target(lnlike_XY_log_ellipse,target_indices,XY_log_lower_bounds,XY_log_upper_bounds,nuisance_guess;grid_steps=grid_steps,use_last_as_guess=true,method=:LN_NELDERMEAD)
# extract interest parameter
ψ1_ellipse_values = [ψ1ω[target_indices] for ψ1ω in ψ1ω_ellipse_values] 

# - Now ψ2 component 
# based on full likelihood
target_indices = 2 # 
nuisance_indices = setdiff(indices_all,target_indices)
nuisance_guess = XY_log_initial[nuisance_indices]
# gets interest and nuisance values along profile
ψ2ω_values, lnlike_ψ2_values = SloppihoodTools.profile_target(lnlike_XY_log,target_indices,XY_log_lower_bounds,XY_log_upper_bounds,nuisance_guess;use_last_as_guess=true,method=:LN_NELDERMEAD,grid_steps=grid_steps)
# extract interest parameter
ψ2_values = [ψ2ω[target_indices] for ψ2ω in ψ2ω_values] 
# get location of max
max_indices = argmax(lnlike_ψ2_values)
ψ2_max = ψ2_values[max_indices[1]] #still [1]

# Next ψ2 component from ellipse
# gets interest and nuisance values along profile. Same arguments as before except likelihood
ψ2ω_ellipse_values, lnlike_ψ2_ellipse_values = SloppihoodTools.profile_target(lnlike_XY_log_ellipse,target_indices,XY_log_lower_bounds,XY_log_upper_bounds,nuisance_guess;use_last_as_guess=true,method=:LN_NELDERMEAD,grid_steps=grid_steps)
# extract interest parameter
ψ2_ellipse_values = [ψ2ω[target_indices] for ψ2ω in ψ2ω_ellipse_values] 

# - Now ψ3 component 
# based on full likelihood
target_indices = 3 # 
nuisance_indices = setdiff(indices_all,target_indices)
nuisance_guess = XY_log_initial[nuisance_indices]
# gets interest and nuisance values along profile. In θ order?
ψ3ω_values, lnlike_ψ3_values = SloppihoodTools.profile_target(lnlike_XY_log,target_indices,XY_log_lower_bounds,XY_log_upper_bounds,nuisance_guess;use_last_as_guess=true,method=:LN_NELDERMEAD,grid_steps=grid_steps)
# extract interest parameter
ψ3_values = [ψ3ω[target_indices] for ψ3ω in ψ3ω_values] 
# get location of max
max_indices = argmax(lnlike_ψ3_values)
ψ3_max = ψ3_values[max_indices[1]] #still [1]

# Next ψ3 component from ellipse
# gets interest and nuisance values along profile. Same arguments as before except likelihood
ψ3ω_ellipse_values, lnlike_ψ3_ellipse_values = SloppihoodTools.profile_target(lnlike_XY_log_ellipse,target_indices,XY_log_lower_bounds,XY_log_upper_bounds,nuisance_guess;use_last_as_guess=true,method=:LN_NELDERMEAD,grid_steps=grid_steps)
# extract interest parameter
ψ3_ellipse_values = [ψ3ω[target_indices] for ψ3ω in ψ3ω_ellipse_values] 


SloppihoodTools.plot_1D_profile(model_name, ψ1_values, lnlike_ψ1_values, varnames["ψ1"];varname_save=varnames["ψ1_save"],ψ_true=XY_log_true[1])
SloppihoodTools.plot_1D_profile(model_name, ψ2_values, lnlike_ψ2_values, varnames["ψ2"];varname_save=varnames["ψ2_save"],ψ_true=XY_log_true[2])
SloppihoodTools.plot_1D_profile(model_name, ψ3_values, lnlike_ψ3_values, varnames["ψ3"];varname_save=varnames["ψ3_save"],ψ_true=XY_log_true[3])
# ---------------------------------------------
# carries out 2D analysis first, then each 1D profile, and plots
# θ_XY_MLE, evals, evecs = SloppihoodTools.execute_model_analysis_workflow_2D_profile("diffusion_model_log",varnames,lnlike_XY_log,XY_log_lower_bounds,XY_log_upper_bounds,XY_log_initial;grid_steps=grid_steps,ψ_true=XY_log_true, return_info=true)

# ----------------------------------------------------
# - Analysis in sloppihood-informed parameterisation -
# ----------------------------------------------------
model_name="diffusion_sip"
# overall scaling
evecs_scaled = SloppihoodTools.scale_and_round(evecs,column_scales=[1,1,1])
print("transformations:")
display(evecs_scaled)
display(inv(evecs_scaled))
# reparam
xytoXY_sip, XYtoxy_sip = SloppihoodTools.reparam(evecs_scaled)

# new variable names
varnames["ψ1"]="\\frac{D_2}{R}"
varnames["ψ2"]="\\frac{D_1}{\\sqrt{D_2R}}"
varnames["ψ3"]="D_1 D_2 R"
varnames["ψ1_save"]="D_2_over_R"
varnames["ψ2_save"]="D_1_over_sqrt_D_2_R"
varnames["ψ3_save"]="D_1_D_2_R"

# --- parameter bounds -- not monotonic, not independent. Do manual for now.
# parameter bounds
XY_sip_lower_bounds = [xy_lower_bounds[1]/xy_upper_bounds[3],xy_lower_bounds[1]/(sqrt(xy_upper_bounds[2]*xy_upper_bounds[3])),xy_lower_bounds[1]*xy_lower_bounds[2]*xy_lower_bounds[3]]
XY_sip_upper_bounds = [xy_upper_bounds[1]/xy_lower_bounds[3],xy_upper_bounds[1]/(sqrt(xy_lower_bounds[2]*xy_lower_bounds[3])),xy_upper_bounds[1]*xy_upper_bounds[2]*xy_upper_bounds[3]]

XY_sip_lower_bounds = [0.5,0.0001,0.00001]
XY_sip_upper_bounds = [1.5,10,1000]

# initial guess for optimisation
XY_sip_initial =  [1,1,10] #xytoXY_sip(xy_initial) # starting guesses

# new true value
XY_sip_true = xytoXY_sip(xy_true)
# new likelihood
lnlike_XY_sip = SloppihoodTools.construct_lnlike_XY(lnlike_xy,XYtoxy_sip)

# new distib model
distrib_XY_sip = SloppihoodTools.construct_distrib_XY(distrib_xy,XYtoxy_sip)

# point est
dim_all = length(XY_sip_initial)
indices_all = 1:dim_all # including higher dim nuisance if needed
# --- point optimisation in original via profiling out all parameters
target_indices = [] # for point estimation whole parameter is nuisance!
nuisance_guess = XY_sip_initial 
# below 'profiles out' all parameters = does MLE
XY_sip_MLE, lnlike_XY_sip_MLE = SloppihoodTools.profile_target(lnlike_XY_sip,target_indices,XY_sip_lower_bounds,XY_sip_upper_bounds,nuisance_guess;use_last_as_guess=false,grid_steps=grid_steps)
# ---
XY_sip_centre = XY_sip_MLE
lnlike_XY_sip_ellipse, H_XY_sip_ellipse = SloppihoodTools.construct_ellipse_lnlike_approx(lnlike_XY_sip,XY_sip_centre)

ForwardDiff.hessian(lnlike_XY_sip, XY_sip_centre)

# --- find eigenvectors (SVD of observed likelihood)
evals, evecs = eigen(H_XY_sip_ellipse; sortby = x -> -real(x))
println("Eigenvectors and eigenvalues for "*model_name)
for (i,eveci) in enumerate(eachcol(evecs))
    evecs[:,i] = eveci 
    println("value:")
    println(evals[i])
    println("vector:")
    println(evecs[:,i])
end
SloppihoodTools.scale_and_round(evecs,column_scales=[1,1,1])

# --- 1D profiles 
# - First ψ1 component 
# based on full likelihood
target_indices = 1 # 
nuisance_indices = setdiff(indices_all,target_indices)
nuisance_guess = XY_sip_initial[nuisance_indices]
# gets interest and nuisance values along profile in θ form
ψ1ω_values, lnlike_ψ1_values = SloppihoodTools.profile_target(lnlike_XY_sip,target_indices,XY_sip_lower_bounds,XY_sip_upper_bounds,nuisance_guess;use_last_as_guess=true,method=:LN_NELDERMEAD,grid_steps=grid_steps)
# extract interest parameter
ψ1_values = [ψ1ω[target_indices] for ψ1ω in ψ1ω_values] 
# get location of max
max_indices = argmax(lnlike_ψ1_values)
ψ1_max = ψ1_values[max_indices[1]]

# Next ψ1 component from ellipse
# gets interest and nuisance values along profile. Same arguments as before except likelihood
ψ1ω_ellipse_values, lnlike_ψ1_ellipse_values = SloppihoodTools.profile_target(lnlike_XY_sip_ellipse,target_indices,XY_sip_lower_bounds,XY_sip_upper_bounds,nuisance_guess;grid_steps=grid_steps,use_last_as_guess=true,method=:LN_NELDERMEAD)
# extract interest parameter
ψ1_ellipse_values = [ψ1ω[target_indices] for ψ1ω in ψ1ω_ellipse_values] 

# Mean prediction CI
lower_ψ1, upper_ψ1, pred_matrix_ψ1 = SloppihoodTools.construct_upper_lower_profile_wise_CIs_for_mean(distrib_XY_sip, ψ1ω_values, lnlike_ψ1_values; l_level=95,df=1)
SloppihoodTools.plot_profile_wise_CI_for_mean(x_data,lower_ψ1,upper_ψ1,pred_mean_MLE,model_name,"x","x",data=data,true_mean=true_mean,target=varnames["ψ1"])


# - Now ψ2 component 
# based on full likelihood
target_indices = 2 # 
nuisance_indices = setdiff(indices_all,target_indices)
nuisance_guess = XY_sip_initial[nuisance_indices]
# gets interest and nuisance values along profile
ψ2ω_values, lnlike_ψ2_values = SloppihoodTools.profile_target(lnlike_XY_sip,target_indices,XY_sip_lower_bounds,XY_sip_upper_bounds,nuisance_guess;grid_steps=grid_steps,use_last_as_guess=true,method=:LN_NELDERMEAD)
# extract interest parameter
ψ2_values = [ψ2ω[target_indices] for ψ2ω in ψ2ω_values] 
# get location of max
max_indices = argmax(lnlike_ψ2_values)
ψ2_max = ψ2_values[max_indices[1]] #still [1]

# Next ψ2 component from ellipse
# gets interest and nuisance values along profile. Same arguments as before except likelihood
ψ2ω_ellipse_values, lnlike_ψ2_ellipse_values = SloppihoodTools.profile_target(lnlike_XY_sip_ellipse,target_indices,XY_sip_lower_bounds,XY_sip_upper_bounds,nuisance_guess;grid_steps=grid_steps,use_last_as_guess=true,method=:LN_NELDERMEAD)
# extract interest parameter
ψ2_ellipse_values = [ψ2ω[target_indices] for ψ2ω in ψ2ω_ellipse_values] 

# Mean prediction CI
lower_ψ2, upper_ψ2, pred_matrix_ψ2 = SloppihoodTools.construct_upper_lower_profile_wise_CIs_for_mean(distrib_XY_sip, ψ2ω_values, lnlike_ψ2_values; l_level=95,df=1)
SloppihoodTools.plot_profile_wise_CI_for_mean(x_data,lower_ψ2,upper_ψ2,pred_mean_MLE,model_name,"x","x",data=data,true_mean=true_mean,target=varnames["ψ2"])


# - Now ψ3 component 
# based on full likelihood
target_indices = 3 # 
nuisance_indices = setdiff(indices_all,target_indices)
nuisance_guess = XY_sip_initial[nuisance_indices]
# gets interest and nuisance values along profile. In θ order?
ψ3ω_values, lnlike_ψ3_values = SloppihoodTools.profile_target(lnlike_XY_sip,target_indices,XY_sip_lower_bounds,XY_sip_upper_bounds,nuisance_guess;grid_steps=grid_steps,use_last_as_guess=true,method=:LN_NELDERMEAD)
# extract interest parameter
ψ3_values = [ψ3ω[target_indices] for ψ3ω in ψ3ω_values] 
# get location of max
max_indices = argmax(lnlike_ψ3_values)
ψ3_max = ψ3_values[max_indices[1]] #still [1]

# Next ψ3 component from ellipse
# gets interest and nuisance values along profile. Same arguments as before except likelihood
ψ3ω_ellipse_values, lnlike_ψ3_ellipse_values = SloppihoodTools.profile_target(lnlike_XY_sip_ellipse,target_indices,XY_sip_lower_bounds,XY_sip_upper_bounds,nuisance_guess;grid_steps=grid_steps,use_last_as_guess=true,method=:LN_NELDERMEAD)
# extract interest parameter
ψ3_ellipse_values = [ψ3ω[target_indices] for ψ3ω in ψ3ω_ellipse_values] 

# Mean prediction CI
lower_ψ3, upper_ψ3, pred_matrix_ψ3 = SloppihoodTools.construct_upper_lower_profile_wise_CIs_for_mean(distrib_XY_sip, ψ3ω_values, lnlike_ψ3_values; l_level=95,df=1)
SloppihoodTools.plot_profile_wise_CI_for_mean(x_data,lower_ψ3,upper_ψ3,pred_mean_MLE,model_name,"x","x",data=data,true_mean=true_mean,target=varnames["ψ3"])


# plot the profiles
SloppihoodTools.plot_1D_profile(model_name, ψ1_values, lnlike_ψ1_values, varnames["ψ1"];varname_save=varnames["ψ1_save"],ψ_true=XY_sip_true[1])
SloppihoodTools.plot_1D_profile(model_name, ψ2_values, lnlike_ψ2_values, varnames["ψ2"];varname_save=varnames["ψ2_save"],ψ_true=XY_sip_true[2])
SloppihoodTools.plot_1D_profile(model_name, ψ3_values, lnlike_ψ3_values, varnames["ψ3"];varname_save=varnames["ψ3_save"],ψ_true=XY_sip_true[3])


# - now pariwise.
# combo 1
target_indices = [1,2]; 
nuisance_indices = setdiff(indices_all,target_indices)
nuisance_guess = XY_sip_initial[nuisance_indices]
ψ1ψ2_true = XY_sip_true[target_indices]

# - 2D profile likelihood
ψ1ψ2ω_values, lnlike_ψ1ψ2_values = SloppihoodTools.profile_target(lnlike_XY_sip,target_indices,XY_sip_lower_bounds,XY_sip_upper_bounds,nuisance_guess;grid_steps=grid_steps)
# extract interest parameter vector
ψ1ψ2_values = [ψ1ψ2[target_indices] for ψ1ψ2 in ψ1ψ2ω_values]
max_indices = argmax(lnlike_ψ1ψ2_values)
ψ1ψ2_MLE_grid = ψ1ψ2_values[max_indices]

SloppihoodTools.plot_2D_contour(model_name, ψ1ψ2_values, lnlike_ψ1ψ2_values, varnames;ψ_true=ψ1ψ2_true,save_dir="./", file_extension=".svg")

# Mean prediction CI based on pair.
lower_ψ1ψ2, upper_ψ1ψ2, pred_matrix_ψ1ψ2 = SloppihoodTools.construct_upper_lower_profile_wise_CIs_for_mean(distrib_XY_sip, ψ1ψ2ω_values, lnlike_ψ1ψ2_values; l_level=95,df=2)
SloppihoodTools.plot_profile_wise_CI_for_mean(x_data,lower_ψ1ψ2,upper_ψ1ψ2,pred_mean_MLE,model_name,"x","x",data=data,true_mean=true_mean,target=varnames["ψ1"]*","*varnames["ψ2"])


# combo 2
target_indices = [1,3]; 
nuisance_indices = setdiff(indices_all,target_indices)
nuisance_guess = XY_sip_initial[nuisance_indices]
ψ1ψ2_true = XY_sip_true[target_indices]
# new variable names
varnames["ψ1"]="\\frac{D_2}{R}"
varnames["ψ2"]="D_1 D_2 R"
varnames["ψ1_save"]="D_2_over_R"
varnames["ψ2_save"]="D_1_D_2_R"

# - 2D profile likelihood
ψ1ψ2ω_values, lnlike_ψ1ψ2_values = SloppihoodTools.profile_target(lnlike_XY_sip,target_indices,XY_sip_lower_bounds,XY_sip_upper_bounds,nuisance_guess;grid_steps=grid_steps)
# extract interest parameter vector
ψ1ψ2_values = [ψ1ψ2[target_indices] for ψ1ψ2 in ψ1ψ2ω_values]
max_indices = argmax(lnlike_ψ1ψ2_values)
ψ1ψ2_MLE_grid = ψ1ψ2_values[max_indices]

SloppihoodTools.plot_2D_contour(model_name, ψ1ψ2_values, lnlike_ψ1ψ2_values, varnames;ψ_true=ψ1ψ2_true,save_dir="./", file_extension=".svg")

# Mean prediction CI based on pair.
lower_ψ1ψ2, upper_ψ1ψ2, pred_matrix_ψ1ψ2 = SloppihoodTools.construct_upper_lower_profile_wise_CIs_for_mean(distrib_XY_sip, ψ1ψ2ω_values, lnlike_ψ1ψ2_values; l_level=95,df=2)
SloppihoodTools.plot_profile_wise_CI_for_mean(x_data,lower_ψ1ψ2,upper_ψ1ψ2,pred_mean_MLE,model_name,"x","x",data=data,true_mean=true_mean,target=varnames["ψ1"]*","*varnames["ψ2"])


# combo 3
target_indices = [2,3]; 
nuisance_indices = setdiff(indices_all,target_indices)
nuisance_guess = XY_sip_initial[nuisance_indices]
ψ1ψ2_true = XY_sip_true[target_indices]
# new variable names

varnames["ψ1"]="\\frac{D_1}{\\sqrt{D_2R}}"
varnames["ψ2"]="D_1 D_2 R"
varnames["ψ1_save"]="D_1_over_sqrt_D_2_R"
varnames["ψ2_save"]="D_1_D_2_R"

# - 2D profile likelihood
ψ1ψ2ω_values, lnlike_ψ1ψ2_values = SloppihoodTools.profile_target(lnlike_XY_sip,target_indices,XY_sip_lower_bounds,XY_sip_upper_bounds,nuisance_guess;grid_steps=grid_steps)
# extract interest parameter vector
ψ1ψ2_values = [ψ1ψ2[target_indices] for ψ1ψ2 in ψ1ψ2ω_values]
max_indices = argmax(lnlike_ψ1ψ2_values)
ψ1ψ2_MLE_grid = ψ1ψ2_values[max_indices]

SloppihoodTools.plot_2D_contour(model_name, ψ1ψ2_values, lnlike_ψ1ψ2_values, varnames;ψ_true=ψ1ψ2_true,save_dir="./", file_extension=".svg")

# Mean prediction CI based on pair.
lower_ψ1ψ2, upper_ψ1ψ2, pred_matrix_ψ1ψ2 = SloppihoodTools.construct_upper_lower_profile_wise_CIs_for_mean(distrib_XY_sip, ψ1ψ2ω_values, lnlike_ψ1ψ2_values; l_level=95,df=2)
SloppihoodTools.plot_profile_wise_CI_for_mean(x_data,lower_ψ1ψ2,upper_ψ1ψ2,pred_mean_MLE,model_name,"x","x",data=data,true_mean=true_mean,target=varnames["ψ1"]*","*varnames["ψ2"])
