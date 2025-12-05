# Minimal 2D Distributed Profile - Repressilator (β₁, K₁)
# Stripped down to essentials for understanding

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

# True parameters (18 total, EXACT from Eisenberg & Hayashi)
θ_true = [0.008, 0.009, 0.010,      # α₀ (basal)
          1.0, 1.2, 1.5,             # α (regulated)
          0.02, 0.025, 0.015,        # β (translation) <- β₁ is index 7
          30.0, 28.0, 32.0,          # K (inhibition)  <- K₁ is index 10
          0.006, 0.0055, 0.0065,     # k_degm
          0.0012, 0.0011, 0.0013]    # k_degp

# Generate data
y_true = RepressilatorModel.predict_mRNA(θ_true, t_obs, X0)
data = y_true + σ * randn(length(y_true))

# === REDUCED 4-PARAMETER MODEL ===
# Profile: β₁ (idx 1), K₁ (idx 3) in reduced space
# Optimize: β₂ (idx 2), K₂ (idx 4) in reduced space
# Fix: everything else at true values

# Indices in full 18-param space
free_indices = [7, 8, 10, 11]  # β₁, β₂, K₁, K₂
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

# Likelihood over 4 free parameters (no type annotations!)
@everywhere function lnlike_reduced(θ4_log)
    θ4 = exp.(θ4_log)
    # Reconstruct full 18-param vector
    θ_full = zeros(18)
    θ_full[fixed_indices] = θ_fixed
    θ_full[free_indices] = θ4
    try
        pred = RepressilatorModel.predict_mRNA(θ_full, t_obs, X0)
        return logpdf(MvNormal(pred, σ^2 * I(length(pred))), data)
    catch
        return -Inf
    end
end

# === BOUNDS (4 params: β₁, β₂, K₁, K₂) ===
θ4_true = θ_true[free_indices]
θ4_log_true = log.(θ4_true)

# In reduced space: indices 1,3 are target (β₁, K₁), indices 2,4 are nuisance (β₂, K₂)
target = [1, 3]  # β₁, K₁ in 4-param space

θ4_log_lower = θ4_log_true .- 4.0
θ4_log_upper = θ4_log_true .+ 4.0
θ4_log_lower[target] .= θ4_log_true[target] .- 2.0
θ4_log_upper[target] .= θ4_log_true[target] .+ 2.0

# === MLE (4 params) ===
println("\nFinding MLE over 4 parameters...")
θ4_log_MLE, ll_MLE = ReparamTools.profile_target(lnlike_reduced, Int[], θ4_log_lower, θ4_log_upper, θ4_log_true;
                                    optmaxtime=60.0)
println("MLE log-likelihood: ", ll_MLE)
println("MLE params: β₁=$(exp(θ4_log_MLE[1])), β₂=$(exp(θ4_log_MLE[2])), K₁=$(exp(θ4_log_MLE[3])), K₂=$(exp(θ4_log_MLE[4]))")

# === 2D PROFILE (profile β₁,K₁ over nuisance β₂,K₂) ===
println("\nRunning 2D profile (β₁, K₁) with only 2 nuisance params (β₂, K₂)...")
nuisance_indices = [2, 4]  # β₂, K₂ in 4-param space
nuisance_guess = θ4_log_MLE[nuisance_indices]

GRID = 7  # 7×7 = 49 points
t_start = time()

θ_vals, ll_vals = ReparamTools.profile_target(lnlike_reduced, target, θ4_log_lower, θ4_log_upper, nuisance_guess;
                                  grid_steps=GRID,
                                  use_distributed=true,
                                  optmaxtime=60.0)

elapsed = time() - t_start
println("Done in $(round(elapsed/60, digits=1)) minutes")
println("Finite values: $(sum(isfinite.(ll_vals)))/$(length(ll_vals))")

# === PLOT ===
using Plots

# In 4-param space: idx 1 = β₁, idx 3 = K₁
# Base.product iterates first arg (β₁) fastest, so reshape as (n_β, n_K) then transpose
β1_vals = unique([exp(θ[1]) for θ in θ_vals])
K1_vals = unique([exp(θ[3]) for θ in θ_vals])
ll_grid = reshape(ll_vals, length(β1_vals), length(K1_vals))'
ll_norm = ll_grid .- maximum(ll_grid[isfinite.(ll_grid)])
like_grid = exp.(ll_norm)

plt = contourf(β1_vals, K1_vals, like_grid, color=:viridis, levels=20,
               xlabel="β₁", ylabel="K₁", title="2D Profile (β₁, K₁) - 4 param model")
scatter!([exp(θ4_log_MLE[1])], [exp(θ4_log_MLE[3])], mc=:white, ms=8, label="MLE")
scatter!([θ4_true[1]], [θ4_true[3]], mc=:red, ms=8, markershape=:star, label="True")

savefig(plt, "minimal_2D_result.png")
println("Saved: minimal_2D_result.png")

# Cleanup
rmprocs(workers())
