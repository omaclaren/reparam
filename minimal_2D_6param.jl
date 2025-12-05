# Minimal 2D Profile - 6 parameters
# Profile: β₁, K₁ (target)
# Optimize: β₂, β₃, K₂, K₃ (4 nuisance parameters)
# Fixed: everything else at true values

using Distributed
addprocs(4)
println("Workers: ", workers())

# Load on main
include("examples/RepressilatorModel.jl")
include("ReparamTools.jl")
using .RepressilatorModel
using .ReparamTools
using Distributions, LinearAlgebra, Random

# Load on workers
@everywhere begin
    include(joinpath(@__DIR__, "examples", "RepressilatorModel.jl"))
    include(joinpath(@__DIR__, "ReparamTools.jl"))
    using .RepressilatorModel
    using .ReparamTools
    using Distributions, LinearAlgebra
end

# === MODEL SETUP ===
Random.seed!(42)
NT, T_end = 8, 10000.0
t_obs = LinRange(0, T_end, NT)
X0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
σ = 1.0

# True parameters (18 total)
# Indices: α₀=[1,2,3], α=[4,5,6], β=[7,8,9], K=[10,11,12], k_degm=[13,14,15], k_degp=[16,17,18]
θ_true = [0.008, 0.009, 0.010,      # α₀ (1-3)
          1.0, 1.2, 1.5,             # α  (4-6)
          0.02, 0.025, 0.015,        # β  (7-9) <- β₁=7, β₂=8, β₃=9
          30.0, 28.0, 32.0,          # K  (10-12) <- K₁=10, K₂=11, K₃=12
          0.006, 0.0055, 0.0065,     # k_degm (13-15)
          0.0012, 0.0011, 0.0013]    # k_degp (16-18)

# Generate data
y_true = RepressilatorModel.predict_mRNA(θ_true, t_obs, X0)
data = y_true + σ * randn(length(y_true))

# === 6-PARAMETER MODEL ===
# Free: β₁, β₂, β₃, K₁, K₂, K₃ (indices 7,8,9,10,11,12)
# In reduced 6-param space: [β₁, β₂, β₃, K₁, K₂, K₃] = indices [1,2,3,4,5,6]
# Profile targets: β₁, K₁ = indices [1, 4]
# Nuisance: β₂, β₃, K₂, K₃ = indices [2,3,5,6]

free_indices = [7, 8, 9, 10, 11, 12]  # β₁, β₂, β₃, K₁, K₂, K₃
fixed_indices = setdiff(1:18, free_indices)
θ_fixed = θ_true[fixed_indices]

# Send to workers
@everywhere t_obs = $t_obs
@everywhere X0 = $X0
@everywhere σ = $σ
@everywhere data = $data
@everywhere fixed_indices = $fixed_indices
@everywhere free_indices = $free_indices
@everywhere θ_fixed = $θ_fixed

# Likelihood over 6 free parameters
@everywhere function lnlike_6param(θ6_log)
    θ6 = exp.(θ6_log)
    θ_full = zeros(18)
    θ_full[fixed_indices] = θ_fixed
    θ_full[free_indices] = θ6
    try
        pred = RepressilatorModel.predict_mRNA(θ_full, t_obs, X0)
        return logpdf(MvNormal(pred, σ^2 * I(length(pred))), data)
    catch
        return -Inf
    end
end

# === BOUNDS (6 params) ===
θ6_true = θ_true[free_indices]  # [β₁, β₂, β₃, K₁, K₂, K₃]
θ6_log_true = log.(θ6_true)

# Target: β₁ (idx 1), K₁ (idx 4) - profile over these
# Nuisance: β₂, β₃, K₂, K₃ (idx 2,3,5,6) - optimize over these
target = [1, 4]
nuisance_idx = [2, 3, 5, 6]

# Target bounds: ±2 (grid range)
# Nuisance bounds: ±4 (wider for optimization)
θ6_log_lower = θ6_log_true .- 4.0
θ6_log_upper = θ6_log_true .+ 4.0
θ6_log_lower[target] .= θ6_log_true[target] .- 2.0
θ6_log_upper[target] .= θ6_log_true[target] .+ 2.0

# === MLE (6 params) ===
println("\nFinding MLE over 6 parameters (β₁,β₂,β₃,K₁,K₂,K₃)...")
θ6_log_MLE, ll_MLE = ReparamTools.profile_target(lnlike_6param, Int[], θ6_log_lower, θ6_log_upper, θ6_log_true;
                                    optmaxtime=60.0, method=:LN_BOBYQA)
println("MLE log-likelihood: ", ll_MLE)
println("MLE: β₁=$(round(exp(θ6_log_MLE[1]), digits=4)), K₁=$(round(exp(θ6_log_MLE[4]), digits=1))")

# === 2D PROFILE ===
println("\n2D profile (β₁, K₁) with 4 nuisance params (β₂,β₃,K₂,K₃)...")
nuisance_guess = θ6_log_MLE[nuisance_idx]

GRID = 30  # 30×30 = 900 points
t_start = time()

θ_vals, ll_vals = ReparamTools.profile_target(lnlike_6param, target, θ6_log_lower, θ6_log_upper, nuisance_guess;
                                  grid_steps=GRID,
                                  use_distributed=false,  # Sequential for better continuation
                                  optmaxtime=30.0,
                                  method=:LN_BOBYQA)

elapsed = time() - t_start
println("Done in $(round(elapsed/60, digits=1)) minutes")
println("Finite values: $(sum(isfinite.(ll_vals)))/$(length(ll_vals))")

# Diagnostic: check ll_vals range
println("LL values (after profile_target normalization):")
println("  Min: ", minimum(ll_vals))
println("  Max: ", maximum(ll_vals))
println("  Range: ", maximum(ll_vals) - minimum(ll_vals))

# === PLOT ===
using Plots

β1_vals = unique([exp(θ[1]) for θ in θ_vals])
K1_vals = unique([exp(θ[4]) for θ in θ_vals])
ll_grid = reshape(ll_vals, length(β1_vals), length(K1_vals))'

# Negative log-likelihood (normalized)
ll_max = maximum(ll_grid[isfinite.(ll_grid)])
nll_grid = -(ll_grid .- ll_max)
nll_clipped = clamp.(nll_grid, 0, 10)

plt = contourf(β1_vals, K1_vals, nll_clipped, color=:viridis, levels=20,
               xlabel="β₁", ylabel="K₁", title="2D NLL (β₁, K₁) - 6 param model (4 nuisance)")
scatter!([exp(θ6_log_MLE[1])], [exp(θ6_log_MLE[4])], mc=:white, ms=8, label="MLE")
scatter!([θ6_true[1]], [θ6_true[4]], mc=:red, ms=10, markershape=:star, label="True")

# 95% CI contour
contour!(β1_vals, K1_vals, nll_grid, levels=[3.0], color=:white, lw=2, linestyle=:dash)

savefig(plt, "minimal_2D_6param_result.png")
println("Saved: minimal_2D_6param_result.png")

# Cleanup
rmprocs(workers())
