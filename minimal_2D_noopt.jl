# Minimal 2D Profile - NO nuisance optimization
# Just grid evaluation of likelihood over (β₁, K₁)
# All other 16 parameters fixed at true values

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
          0.02, 0.025, 0.015,        # β  <- β₁ is index 7
          30.0, 28.0, 32.0,          # K  <- K₁ is index 10
          0.006, 0.0055, 0.0065,     # k_degm
          0.0012, 0.0011, 0.0013]    # k_degp

# Generate data
y_true = RepressilatorModel.predict_mRNA(θ_true, t_obs, X0)
data = y_true + σ * randn(length(y_true))

# === 2-PARAMETER MODEL: only β₁ and K₁ free ===
free_indices = [7, 10]  # β₁, K₁
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

# Likelihood over just 2 parameters (no nuisance optimization!)
@everywhere function lnlike_2param(θ2_log)
    θ2 = exp.(θ2_log)
    θ_full = zeros(18)
    θ_full[fixed_indices] = θ_fixed
    θ_full[free_indices] = θ2
    try
        pred = RepressilatorModel.predict_mRNA(θ_full, t_obs, X0)
        return logpdf(MvNormal(pred, σ^2 * I(length(pred))), data)
    catch
        return -Inf
    end
end

# === BOUNDS (2 params: β₁, K₁) ===
θ2_true = θ_true[free_indices]  # [0.02, 30.0]
θ2_log_true = log.(θ2_true)

# Both are target params - profile over both (no nuisance)
target = [1, 2]  # both params in 2-param space

θ2_log_lower = θ2_log_true .- 2.0
θ2_log_upper = θ2_log_true .+ 2.0

# === 2D PROFILE (no nuisance - just grid evaluation) ===
println("\n2D profile over (β₁, K₁) - no nuisance optimization")
println("This is just a grid evaluation of the likelihood")

GRID = 100  # 100×100 = 10000 points (can be dense since no optimization)
t_start = time()

# Empty nuisance guess since there are no nuisance params
nuisance_guess = Float64[]

θ_vals, ll_vals = ReparamTools.profile_target(lnlike_2param, target, θ2_log_lower, θ2_log_upper, nuisance_guess;
                                  grid_steps=GRID,
                                  use_distributed=true,
                                  optmaxtime=10.0)

elapsed = time() - t_start
println("Done in $(round(elapsed, digits=1)) seconds")
println("Finite values: $(sum(isfinite.(ll_vals)))/$(length(ll_vals))")

# === PLOT ===
using Plots

β1_vals = unique([exp(θ[1]) for θ in θ_vals])
K1_vals = unique([exp(θ[2]) for θ in θ_vals])
ll_grid = reshape(ll_vals, length(β1_vals), length(K1_vals))'

# Convert to negative log-likelihood (normalized so min = 0)
ll_max = maximum(ll_grid[isfinite.(ll_grid)])
nll_grid = -(ll_grid .- ll_max)  # NLL: min at MLE = 0, increases away

# Clip to show structure (e.g., 0 to 10 covers useful range)
nll_clipped = clamp.(nll_grid, 0, 10)

plt = contourf(β1_vals, K1_vals, nll_clipped, color=:viridis, levels=20,
               xlabel="β₁", ylabel="K₁", title="2D Negative Log-Likelihood (β₁, K₁) - clipped [0, 10]")
scatter!([θ2_true[1]], [θ2_true[2]], mc=:red, ms=10, markershape=:star, label="True")

# Add 95% CI contour (chi-sq df=2, threshold = 3.0 for NLL)
contour!(β1_vals, K1_vals, nll_grid, levels=[3.0], color=:white, lw=2, linestyle=:dash)

savefig(plt, "minimal_2D_noopt_result.png")
println("Saved: minimal_2D_noopt_result.png")

# Cleanup
rmprocs(workers())
