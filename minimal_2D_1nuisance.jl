# Minimal 2D Profile with 1 Nuisance Parameter
# Profile over (β₁, K₁) while optimizing over β₂

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
NT, T_end = 4, 10000.0
t_obs = LinRange(0, T_end, NT)
X0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
σ = 10.0

# True parameters (18 total)
θ_true = [0.008, 0.009, 0.010,      # α₀ (1-3)
          1.0, 1.2, 1.5,             # α  (4-6)
          0.02, 0.025, 0.015,        # β  (7-9)
          30.0, 28.0, 32.0,          # K  (10-12)
          0.006, 0.0055, 0.0065,     # k_degm (13-15)
          0.0012, 0.0011, 0.0013]    # k_degp (16-18)

# Generate data
y_true = RepressilatorModel.predict_mRNA(θ_true, t_obs, X0)
data = y_true + σ * randn(length(y_true))

# True values
β1_true = θ_true[7]   # 0.02
K1_true = θ_true[10]  # 30.0
β2_true = θ_true[8]   # 0.025
ratio_true = K1_true / β1_true

println("True values: β₁=$β1_true, K₁=$K1_true, β₂=$β2_true")
println("True ratio K₁/β₁=$ratio_true")

# === PARAMETER SETUP ===
# 3-parameter problem: β₁, β₂, K₁ (indices 7, 8, 10 in full model)
# Target: β₁, K₁ (we'll profile over these)
# Nuisance: β₂ (optimize at each grid point)

free_indices = [7, 8, 10]  # β₁, β₂, K₁
fixed_indices = setdiff(1:18, free_indices)
θ_fixed = θ_true[fixed_indices]

@everywhere t_obs = $t_obs
@everywhere X0 = $X0
@everywhere σ = $σ
@everywhere data = $data
@everywhere fixed_indices = $fixed_indices
@everywhere θ_fixed = $θ_fixed

# Likelihood in 3-param log space: [log(β₁), log(β₂), log(K₁)]
@everywhere function lnlike_3param(θ3_log)
    # θ3_log = [log(β₁), log(β₂), log(K₁)]
    β1 = exp(θ3_log[1])
    β2 = exp(θ3_log[2])
    K1 = exp(θ3_log[3])
    
    # Reconstruct full 18-parameter vector
    θ_full = zeros(18)
    θ_full[fixed_indices] = θ_fixed
    θ_full[7] = β1
    θ_full[8] = β2
    θ_full[10] = K1
    
    try
        pred = RepressilatorModel.predict_mRNA(θ_full, t_obs, X0)
        return logpdf(MvNormal(pred, σ^2 * I(length(pred))), data)
    catch
        return -Inf
    end
end

# === BOUNDS ===
# Target params: β₁, K₁ (params 1 and 3 in our 3-param space)
# Nuisance param: β₂ (param 2 in our 3-param space)

θ3_true = [β1_true, β2_true, K1_true]
θ3_log_true = log.(θ3_true)

# Bounds for all 3 params in log space
θ3_log_lower = [log(0.004), log(0.005), log(0.5)]    # β₁, β₂, K₁ lower
θ3_log_upper = [log(0.09), log(0.12), log(200.0)]    # β₁, β₂, K₁ upper

# Target indices (in 3-param space): β₁=1, K₁=3
# Nuisance index: β₂=2
target = [1, 3]  # Profile over β₁ and K₁

# === 2D PROFILE with nuisance optimization ===
println("\n2D profile over (β₁, K₁) with β₂ as nuisance...")
println("At each grid point, optimize β₂ to maximize likelihood")

GRID = 50  # 50×50 = 2500 points (smaller grid since optimization at each point)
t_start = time()

# Initial guess for nuisance param (β₂) - use true value in log space
nuisance_guess = [θ3_log_true[2]]  # log(β₂)

θ_vals, ll_vals = ReparamTools.profile_target(lnlike_3param, target, θ3_log_lower, θ3_log_upper, nuisance_guess;
                                  grid_steps=GRID,
                                  use_distributed=true,
                                  method=:LN_BOBYQA,  # Gradient-free optimizer
                                  optmaxtime=10.0)

elapsed = time() - t_start
println("Done in $(round(elapsed, digits=1)) seconds")
println("Finite values: $(sum(isfinite.(ll_vals)))/$(length(ll_vals))")
println("LL range: $(minimum(ll_vals)) to $(maximum(ll_vals))")

# === PLOT ===
using Plots
using Distributions

# Construct grid directly from bounds (same as what profile_target uses internally)
β1_log_vals = range(θ3_log_lower[target[1]], θ3_log_upper[target[1]], length=GRID)
K1_log_vals = range(θ3_log_lower[target[2]], θ3_log_upper[target[2]], length=GRID)
β1_vals = exp.(β1_log_vals)
K1_vals = exp.(K1_log_vals)

println("Grid: $(length(β1_vals)) × $(length(K1_vals)) = $(length(β1_vals) * length(K1_vals)) points")
println("Total ll_vals: $(length(ll_vals))")

# Reshape - ll_vals should be length(β1_vals) * length(K1_vals)
ll_matrix = reshape(ll_vals, length(β1_vals), length(K1_vals))

# Convert to normalized likelihood
ll_max = maximum(ll_matrix[isfinite.(ll_matrix)])
like_matrix = exp.(ll_matrix .- ll_max)
like_for_plots = like_matrix'

# Chi-square threshold
lstar = exp(-quantile(Chisq(2), 0.95)/2)
println("95% CI threshold: $(round(lstar, digits=3))")

# Plot
plt = contourf(β1_vals, K1_vals, like_for_plots, color=:dense, levels=20, lw=0,
              xlabel="β₁", ylabel="K₁", 
              title="Profile likelihood (β₂ optimized)",
              size=(700, 600))
scatter!([β1_true], [K1_true], mc=:darkgoldenrod, msc=:match, ms=10, markershape=:star, label="True")
contour!(β1_vals, K1_vals, like_for_plots, levels=[lstar], color=:black, lw=1, legend=false)

# Reference line at true ratio
β1_line = range(minimum(β1_vals), maximum(β1_vals), 100)
plot!(β1_line, ratio_true .* β1_line, color=:gray, lw=1, ls=:dash, label="K₁/β₁ = $ratio_true")

savefig(plt, "minimal_2D_1nuisance_result.png")
println("Saved: minimal_2D_1nuisance_result.png")

# Cleanup
rmprocs(workers())
