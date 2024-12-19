using Plots
using Distributions
using NLopt
using ForwardDiff
using LinearAlgebra

# ---------------------------------------------
# ---------- Load 'Sloppihood' tools ----------
# ---------------------------------------------
include("SloppihoodTools.jl")
using .SloppihoodTools

# ---------------------------------------------
# ---- User inputs in original 'x,y' param ----
# ---------------------------------------------
# parameter -> data parameter mapping
ϕ_xy = xy -> [xy[1]*xy[2],xy[2]] # x (i.e. n) and y (i.e. p), forward mapping
# parameter -> data dist (forward) mapping
distrib_xy(xy) = Normal(xy[1]*xy[2],sqrt(xy[1]*xy[2]*(1-xy[2]))) # 
# variables
varnames = Dict("ψ1"=>"n", "ψ2"=>"p")
varnames["ψ1_save"] = "n"
varnames["ψ2_save"] = "p" #if you want different names for saved files e.g. use latex in other names
# initial guess for optimisation
xy_initial =  [50, 0.3]# x (i.e. n) and y (i.e. p), starting guesses
# parameter bounds
xy_lower_bounds = [0.1,0.0001]
xy_upper_bounds = [500,1.0]
# true parameter
xy_true = [100,0.2] #x,y, truth. N, p
N_samples = 50 # measurements of model
# generate data
#data = rand(distrib_xy(xy_true),N_samples)
data = [21.9,22.3,12.8,16.4,16.4,20.3,16.2,20.0,19.7,24.4]

# ---- use above to construct log likelihood in original parameterisation given (iid) data
lnlike_xy = SloppihoodTools.construct_lnlike_xy(distrib_xy,data)
# ----

# ---------------------------------------------
# --- Analysis in original parameterisation ---
# ---------------------------------------------
# grid sizes for profiling
grid_steps = [500]
# carries out 2D analysis first, then each 1D profile, and plots
θ_xy_MLE, evals_xy, evecs_xy = SloppihoodTools.execute_model_analysis_workflow_2D_profile("stat_model_xy",varnames,lnlike_xy,xy_lower_bounds,xy_upper_bounds,xy_initial;grid_steps=grid_steps,ψ_true=xy_true, return_info=true)

# --- Determine svd of phi mapping in xy coordinates
J_ϕ_xy = ForwardDiff.jacobian(ϕ_xy, θ_xy_MLE)
U_xy, S_xy, Vt_xy = svd(J_ϕ_xy)
print(Vt_xy)

# ---------------------------------------------
# ----- Analysis in log parameterisation ------
# ---------------------------------------------
grid_steps = [500]
xytoXY_log(xy) = log.(xy)
XYtoxy_log(XY) = exp.(XY)
# new variable names (for plotting etc -- use latex)
varnames["ψ1"]="\\ln\\ n"
varnames["ψ2"]="\\ln\\ p"
# names for saving 
varnames["ψ1_save"]="ln_n"
varnames["ψ2_save"]="ln_p"
# parameter bounds -- could do via xytoXY but also easy to do manually
XY_log_lower_bounds = [3,-4]
XY_log_upper_bounds = [6.5,0]
# new true value
XY_log_true = xytoXY_log(xy_true)
# initial guess for optimisation -- can do via xytoXY but also easy to do manually or based on bounds
XY_log_initial =  xytoXY_log(xy_initial) 
# new phi mapping
ϕ_XY_log = SloppihoodTools.construct_ϕ_XY(ϕ_xy, XYtoxy_log)
# new likelihood
lnlike_XY_log = SloppihoodTools.construct_lnlike_XY(lnlike_xy,XYtoxy_log)
# carries out 2D analysis first, then each 1D profile, and plots
θ_XY_log_MLE, evals_XY_log, evecs_XY_log = SloppihoodTools.execute_model_analysis_workflow_2D_profile("stat_model_log",varnames,lnlike_XY_log,XY_log_lower_bounds,XY_log_upper_bounds,XY_log_initial;grid_steps=grid_steps,ψ_true=XY_log_true, return_info=true)

# --- Determine svd of phi mapping in log coordinates
J_ϕ_XY_log = ForwardDiff.jacobian(ϕ_XY_log, θ_XY_log_MLE)
U_XY_log, S_XY_log, Vt_XY_log = svd(J_ϕ_XY_log)
display(evecs_XY_log)
display(Vt_XY_log)

# ----------------------------------------------------
# - Analysis in sloppihood-informed parameterisation -
# ----------------------------------------------------
grid_steps = [500]
# overall scale factor
# evecs_scaled = round.(evecs/evecs[argmax(abs.(evecs))],digits=1)
# evecs_scaled = round.(evecs/(0.25*evecs[argmax(abs.(evecs))]))*0.25
# evecs_scaled = round.(evecs/(0.25*minimum(abs.(evecs[evecs .!= 0]))))*0.25
# # columnwise scalings, e.g. switch ratio = switch signs
# componentwise_scaling = diagm([1,-1])
# evecs_scaled = evecs_scaled*componentwise_scaling

# evecs_scaled = SloppihoodTools.scale_and_round(evecs,column_scales=[1,-1])
# print("transformations:")
# print(evecs_scaled)
# print(inv(evecs_scaled'))

evecs_scaled = SloppihoodTools.scale_and_round(evecs_XY_log,column_scales=[1,1])
#evecs_scaled = evecs * diagm([-1,1])

# evecs_norm = evecs ./ maximum(abs.(evecs))
# evecs_scaled = evecs_norm * diagm([-1,1])

#evecs_scaled = SloppihoodTools.scale_and_round(evecs, round_within=0.1, column_scales=[-1,1])

print("transformations:")
display(evecs_scaled)
display(inv(evecs_scaled))
xytoXY_sip, XYtoxy_sip = SloppihoodTools.reparam(evecs_scaled)

# xytoXY_sip(xy) = exp.(evecs_scaled'*log.(xy))
# XYtoxy_sip(XY) = exp.(inv(evecs_scaled')*log.(XY))

#xytoXY_sip(xy) = [xy[1]*xy[2]; xy[1]/xy[2]]
#XYtoxy_sip(XY) = [(XY[1]/XY[2])^0.5; (XY[1]*XY[2])^0.5] 
#XYtoxy_sip(XY) = [(XY[1]*XY[2])^0.5; (XY[1]/XY[2])^0.5] 
# new variable names
varnames["ψ1"]="np"
#varnames["ψ2"]="p/n"
varnames["ψ2"]="\\frac{n}{p}"
# names for saving 
varnames["ψ1_save"]="np"
varnames["ψ2_save"]="n_over_p"
# initial guess for optimisation
XY_sip_initial =  xytoXY_sip(xy_initial)# x (i.e. n) and y (i.e. p), starting guesses. Might need to correct if doesn't sat. bounds.

# --- parameter bounds -- not monotonic, not independent. 
# commented below various experiments on bounds.
#XY_sip_lbs_naive = [13,1e-9]
#XY_sip_ubs_naive = [25,1.0/10]
#XY_sip_lbs_naive = [13,25]
#XY_sip_ubs_naive = [25,1000]
# to ensure 0 <= p <= 1
#XY_sip_lb_funcs = [Y-> XY_sip_lbs_naive[1], X-> XY_sip_lbs_naive[2]]
#XY_sip_ub_funcs = [Y-> XY_sip_ubs_naive[1], X-> 1/X]
#XY_sip_lb_funcs = [Y-> XY_sip_lbs_naive[1], X-> XY_sip_lbs_naive[2]]
#XY_sip_ub_funcs = [Y-> XY_sip_ubs_naive[1], X-> XY_sip_ubs_naive[2]]
# new tight bounds
# XY_sip_lower_bounds, XY_sip_upper_bounds = SloppihoodTools.construct_2D_internal_constraint_box(XY_sip_lbs_naive,XY_sip_ubs_naive,XY_sip_lb_funcs,XY_sip_ub_funcs)

XY_sip_lower_bounds = [13,25]
XY_sip_upper_bounds = [25,1000]

# Guess may be outside
if XY_sip_initial[1] <= XY_sip_lower_bounds[1] || XY_sip_initial[1] >= XY_sip_upper_bounds[1]
    XY_sip_initial[1] = mean([XY_sip_lower_bounds[1],XY_sip_upper_bounds[1]])
end
if XY_sip_initial[2] <= XY_sip_lower_bounds[2] || XY_sip_initial[2] >= XY_sip_upper_bounds[2]
    XY_sip_initial[2] = mean([XY_sip_lower_bounds[2],XY_sip_upper_bounds[2]])
end

# new true value
XY_sip_true = xytoXY_sip(xy_true)
# new ϕ mapping
ϕ_XY_sip = SloppihoodTools.construct_ϕ_XY(ϕ_xy, XYtoxy_sip)
# new likelihood
lnlike_XY_sip = SloppihoodTools.construct_lnlike_XY(lnlike_xy,XYtoxy_sip)
# carries out 2D analysis first, then each 1D profile, and plots
θ_XY_sip_MLE, evals_XY_sip, evecs_XY_sip = SloppihoodTools.execute_model_analysis_workflow_2D_profile("stat_model_sip",varnames,lnlike_XY_sip,XY_sip_lower_bounds,XY_sip_upper_bounds,XY_sip_initial;grid_steps=grid_steps,ψ_true=XY_sip_true, return_info=true)

# --- Determine svd of phi mapping in SIP coordinates
J_ϕ_XY_sip = ForwardDiff.jacobian(ϕ_XY_sip, θ_XY_sip_MLE)
U_XY_sip, S_XY_sip, Vt_XY_sip = svd(J_ϕ_XY_sip)
display(evecs_XY_sip)
display(Vt_XY_sip)