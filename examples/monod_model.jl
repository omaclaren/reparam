using Plots
using Distributions
using NLopt
using ForwardDiff
using LinearAlgebra
using DifferentialEquations

# ---------------------------------------------
# ---------- Load 'Sloppihood' tools ----------
# ---------------------------------------------
include("SloppihoodTools.jl")
using .SloppihoodTools

# ---------------------------------------------
# ---- User inputs in original 'x,y' param ----
# ---------------------------------------------
# Define ODE model
function DE!(dC,C,xy,t)
    k1,k2=xy
    dC[1]=-k1*C[1]/(k2+C[1]);
end

# ODE model solver
function solve_ode(t_save,xy;solver=Rodas4())
    tspan=(0.0,maximum(t_save));
    prob=ODEProblem(DE!,[1.0],tspan,xy);
    sol=solve(prob,solver,saveat=t_save,abstol=1e-12,reltol=1e-9);
    return sol[1,:]
end

# time measurements and initial condition
#C0=10;  T=30; #This choice is NOT sloppy
C0=0.5; T=20; #This shoice IS sloppy
NT = 10
t_data=LinRange(0,T,NT)
t_model=LinRange(0,T,10*NT)
σ = 0.01
solver=Rodas4()
# parameter -> data dist (forward) mapping
distrib_xy(xy) = MultivariateNormal(solve_ode(t_data,xy;solver=solver),σ^2*I(length(t_data)))

# variables and bounds
varnames = Dict("ψ1"=>"k_1", "ψ2"=>"k_2")
varnames["ψ1_save"]="k_1"
varnames["ψ2_save"]="k_2"
# parameter bounds
k1min=0.1; k1max=10.0
k2min=0.1; k2max=50
xy_lower_bounds = [k1min,k2min]
xy_upper_bounds = [k1max,k2max]
# initial guess for optimisation
xy_initial =  0.5*(xy_lower_bounds + xy_upper_bounds) # [1.5, 1.5]# x (i.e. n) and y (i.e. p), starting guesses

# true parameter
k1_true=1.0; k2_true=5.0;
xy_true = [k1_true,k2_true] #x,y, truth. N, p
# generate data
Nrep = 1
data = rand(distrib_xy(xy_true),Nrep)
#data = [0.983615  0.695977  0.441759  0.302958  0.193174  0.130742  0.0871124  0.04112  0.0269384  0.0331469]
scatter(t_data,data)

# ---- use above to construct log likelihood in original parameterisation given (iid) data
lnlike_xy = SloppihoodTools.construct_lnlike_xy(distrib_xy,data;dist_type=:multi)
# ----

# ---------------------------------------------
# --- Analysis in original parameterisation ---
# ---------------------------------------------
# grid sizes for profiling
grid_steps = [500]
# carries out 2D analysis first, then each 1D profile, and plots
SloppihoodTools.execute_model_analysis_workflow_2D_profile("monod_model_xy",varnames,lnlike_xy,xy_lower_bounds,xy_upper_bounds,xy_initial;grid_steps=grid_steps,ψ_true=xy_true)

# ---------------------------------------------
# ----- Analysis in log parameterisation ------
# ---------------------------------------------
xytoXY_log(xy) = log.(xy)
XYtoxy_log(XY) = exp.(XY)
# new variable names
varnames["ψ1"]="\\ln\\ k_1"
varnames["ψ2"]="\\ln\\ k_2"
varnames["ψ1_save"]="ln_k_1"
varnames["ψ2_save"]="ln_k_2"
# parameter bounds -- can do via xytoXY or manually
XY_log_lower_bounds = log.(xy_lower_bounds)
XY_log_upper_bounds = log.(xy_upper_bounds)
# new true value
XY_log_true = xytoXY_log(xy_true)
# initial guess for optimisation -- can do via xytoXY but also easy to do manually or based on bounds
XY_log_initial =  xytoXY_log(xy_initial) 
# new likelihood
lnlike_XY_log = SloppihoodTools.construct_lnlike_XY(lnlike_xy,XYtoxy_log)
# carries out 2D analysis first, then each 1D profile, and plots
θ_XY_MLE, evals, evecs = SloppihoodTools.execute_model_analysis_workflow_2D_profile("monod_model_log",varnames,lnlike_XY_log,XY_log_lower_bounds,XY_log_upper_bounds,XY_log_initial;grid_steps=grid_steps,ψ_true=XY_log_true, return_info=true)

# ----------------------------------------------------
# - Analysis in sloppihood-informed parameterisation -
# ----------------------------------------------------
# overall scaling
# evecs_scaled = round.(evecs/evecs[argmax(abs.(evecs))],digits=1)
# evecs_scaled = round.(evecs/(0.25*minimum(abs.(evecs[evecs .!= 0]))))*0.25
# # columnwise scalings, e.g. switch ratio = switch signs
# componentwise_scaling = diagm([1,-1])

evecs_scaled = SloppihoodTools.scale_and_round(evecs,column_scales=[1,-1])
print("transformations:")
display(evecs_scaled)
display(inv(evecs_scaled))
xytoXY_sip, XYtoxy_sip = SloppihoodTools.reparam(evecs_scaled)

# new variable names
varnames["ψ1"]="\\frac{k_2}{k_1}"
varnames["ψ2"]="k_1k_2"
varnames["ψ1_save"]="k_2_over_k_1"
varnames["ψ2_save"]="k_1k_2"

# --- parameter bounds -- not monotonic, not independent. Do manual for now.
# parameter bounds
X_sip_lb=0.05; X_sip_ub=25.0; 
Y_sip_lb=2.5; Y_sip_ub=7.5

XY_sip_lower_bounds = [X_sip_lb,Y_sip_lb]
XY_sip_upper_bounds = [X_sip_ub,Y_sip_ub]

# initial guess for optimisation
XY_sip_initial =  xytoXY_sip(xy_initial) # starting guesses

# correct initial guesses if needed
if XY_sip_initial[1] <= XY_sip_lower_bounds[1] || XY_sip_initial[1] >= XY_sip_upper_bounds[1]
    XY_sip_initial[1] = mean([XY_sip_lower_bounds[1],XY_sip_upper_bounds[1]])
end
if XY_sip_initial[2] <= XY_sip_lower_bounds[2] || XY_sip_initial[2] >= XY_sip_upper_bounds[2]
    XY_sip_initial[2] = mean([XY_sip_lower_bounds[2],XY_sip_upper_bounds[2]])
end

# new true value
XY_sip_true = xytoXY_sip(xy_true)
# new likelihood
lnlike_XY_sip = SloppihoodTools.construct_lnlike_XY(lnlike_xy,XYtoxy_sip)

# carries out 2D analysis first, then each 1D profile, and plots
SloppihoodTools.execute_model_analysis_workflow_2D_profile("monod_model_sip",varnames,lnlike_XY_sip,XY_sip_lower_bounds,XY_sip_upper_bounds,XY_sip_initial;grid_steps=grid_steps,ψ_true=XY_sip_true)
