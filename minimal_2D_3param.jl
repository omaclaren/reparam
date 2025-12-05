# Minimal 2D Profile - 3 parameters
# Profile: β₁, K₁ (target)
# Optimize: β₂ (1 nuisance parameter)
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
θ_true = [0.008, 0.009, 0.010,      # α₀
          1.0, 1.2, 1.5,             # α
          0.02, 0.025, 0.015,        # β  <- β₁=7, β₂=8
          30.0, 28.0, 32.0,          # K  <- K₁=10
          0.006, 0.0055, 0.0065,     # k_degm
          0.0012, 0.0011, 0.0013]    # k_degp

# Generate data
y_true = RepressilatorModel.predict_mRNA(θ_true, t_obs, X0)
data = y_true + σ * randn(length(y_true))

# === 3-PARAMETER MODEL ===
# Free: β₁ (idx 7), β₂ (idx 8), K₁ (idx 10)
# In reduced space: [β₁, β₂, K₁] = indices [1, 2, 3]
# Profile targets: β₁, K₁ = indices [1, 3]
# Nuisance: β₂ = index [2]

free_indices = [7, 8, 10]  # β₁, β₂, K₁
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

# Likelihood over 3 free parameters
@everywhere function lnlike_3param(θ3_log)
    θ3 = exp.(θ3_log)
    θ_full = zeros(18)
    θ_full[fixed_indices] = θ_fixed
    θ_full[free_indices] = θ3
    try
        pred = RepressilatorModel.predict_mRNA(θ_full, t_obs, X0)
        return logpdf(MvNormal(pred, σ^2 * I(length(pred))), data)
    catch
        return -Inf
    end
end

# === BOUNDS (3 params: β₁, β₂, K₁) ===
θ3_true = θ_true[free_indices]  # [0.02, 0.025, 30.0]
θ3_log_true = log.(θ3_true)

# Target: β₁ (idx 1), K₁ (idx 3) - profile over these
# Nuisance: β₂ (idx 2) - optimize over this
target = [1, 3]
nuisance_idx = [2]

# Target bounds: ±2 (grid range)
# Nuisance bounds: ±4 (wider for optimization)
θ3_log_lower = θ3_log_true .- 4.0
θ3_log_upper = θ3_log_true .+ 4.0
θ3_log_lower[target] .= θ3_log_true[target] .- 2.0
θ3_log_upper[target] .= θ3_log_true[target] .+ 2.0

# === MLE (3 params) ===
println("\nFinding MLE over 3 parameters...")
θ3_log_MLE, ll_MLE = ReparamTools.profile_target(lnlike_3param, Int[], θ3_log_lower, θ3_log_upper, θ3_log_true;
                                    optmaxtime=60.0)
println("MLE log-likelihood: ", ll_MLE)
println("MLE: β₁=$(exp(θ3_log_MLE[1])), β₂=$(exp(θ3_log_MLE[2])), K₁=$(exp(θ3_log_MLE[3]))")

# === 2D PROFILE ===
println("\n2D profile (β₁, K₁) with 1 nuisance param (β₂)...")
nuisance_guess = θ3_log_MLE[nuisance_idx]

GRID = 20  # 20×20 = 400 points
t_start = time()

θ_vals, ll_vals = ReparamTools.profile_target(lnlike_3param, target, θ3_log_lower, θ3_log_upper, nuisance_guess;
                                  grid_steps=GRID,
                                  use_distributed=true,
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
K1_vals = unique([exp(θ[3]) for θ in θ_vals])
ll_grid = reshape(ll_vals, length(β1_vals), length(K1_vals))'

# Negative log-likelihood (normalized)
ll_max = maximum(ll_grid[isfinite.(ll_grid)])
nll_grid = -(ll_grid .- ll_max)
nll_clipped = clamp.(nll_grid, 0, 10)

plt = contourf(β1_vals, K1_vals, nll_clipped, color=:viridis, levels=20,
               xlabel="β₁", ylabel="K₁", title="2D NLL (β₁, K₁) - 3 param model (1 nuisance)")
scatter!([exp(θ3_log_MLE[1])], [exp(θ3_log_MLE[3])], mc=:white, ms=8, label="MLE")
scatter!([θ3_true[1]], [θ3_true[3]], mc=:red, ms=10, markershape=:star, label="True")

# 95% CI contour
contour!(β1_vals, K1_vals, nll_grid, levels=[3.0], color=:white, lw=2, linestyle=:dash)

savefig(plt, "minimal_2D_3param_result.png")
println("Saved: minimal_2D_3param_result.png")

# Cleanup
rmprocs(workers())
