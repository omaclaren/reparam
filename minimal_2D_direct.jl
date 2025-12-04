# Minimal 2D Profile - Direct Evaluation in Original (β₁, K₁) Space
# Compare with minimal_2D_reparam.jl which profiles in transformed space

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
NT, T_end = 4, 10000.0   # Same as reparam version
t_obs = LinRange(0, T_end, NT)
X0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
σ = 10.0  # Noise level

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

# True values for gene 1
β1_true = θ_true[7]   # 0.02
K1_true = θ_true[10]  # 30.0
ratio_true = K1_true / β1_true  # 1500

println("True values: β₁=$β1_true, K₁=$K1_true, ratio K₁/β₁=$ratio_true")

# === LIKELIHOOD in original (β₁, K₁) space ===
free_orig_indices = [7, 10]  # β₁, K₁ in original 18-param space
fixed_indices = setdiff(1:18, free_orig_indices)
θ_fixed = θ_true[fixed_indices]

@everywhere t_obs = $t_obs
@everywhere X0 = $X0
@everywhere σ = $σ
@everywhere data = $data
@everywhere fixed_indices = $fixed_indices
@everywhere θ_fixed = $θ_fixed

# Likelihood in log(β₁), log(K₁) space
@everywhere function lnlike_θ_log(θ_log)
    # θ_log = [log(β₁), log(K₁)]
    β1 = exp(θ_log[1])
    K1 = exp(θ_log[2])
    
    # Reconstruct full 18-parameter vector
    θ_full = zeros(18)
    θ_full[fixed_indices] = θ_fixed
    θ_full[7] = β1
    θ_full[10] = K1
    
    try
        pred = RepressilatorModel.predict_mRNA(θ_full, t_obs, X0)
        return logpdf(MvNormal(pred, σ^2 * I(length(pred))), data)
    catch
        return -Inf
    end
end

# === BOUNDS in log(θ) space ===
# Match the θ-space coverage from minimal_2D_reparam.jl
# β₁ ∈ [0.004, 0.09], K₁ ∈ [0.5, 200]
θ_log_lower = [log(0.004), log(0.5)]
θ_log_upper = [log(0.09), log(200.0)]

# Both are targets (no nuisance params - just 2D grid)
target = [1, 2]

# === 2D PROFILE in θ space ===
println("\n2D profile in original space (β₁, K₁)...")
println("Direct grid evaluation - no transformation")

GRID = 100  # 100×100 = 10000 points
t_start = time()

nuisance_guess = Float64[]  # No nuisance params

θ_vals, ll_vals = ReparamTools.profile_target(lnlike_θ_log, target, θ_log_lower, θ_log_upper, nuisance_guess;
                                  grid_steps=GRID,
                                  use_distributed=true,
                                  optmaxtime=10.0)

elapsed = time() - t_start
println("Done in $(round(elapsed, digits=1)) seconds")
println("Finite values: $(sum(isfinite.(ll_vals)))/$(length(ll_vals))")
println("LL range: $(minimum(ll_vals)) to $(maximum(ll_vals))")

# === PLOT ===
using Plots
using Distributions  # for Chisq

# Extract grid values
β1_vals = unique([exp(θ[1]) for θ in θ_vals])
K1_vals = unique([exp(θ[2]) for θ in θ_vals])

# Reshape: ll_vals is ordered as (β1, K1) pairs with β1 varying fastest
# After reshape: ll_matrix[i, j] = LL at (β1_vals[i], K1_vals[j])
ll_matrix = reshape(ll_vals, length(β1_vals), length(K1_vals))

# Convert to normalized likelihood scale (matching visualization.jl)
ll_max = maximum(ll_matrix[isfinite.(ll_matrix)])
like_matrix = exp.(ll_matrix .- ll_max)

# For Plots.jl contourf(x, y, z): expects z[j, i] at (x[i], y[j]), so pass like_matrix'
like_for_plots = like_matrix'

# Chi-square threshold for 95% CI with 2 parameters
l_level = 95
df = 2
lstar = exp(-quantile(Chisq(df), l_level/100)/2)
println("95% CI threshold (likelihood): $(round(lstar, digits=3))")

# Plot: Original (β₁, K₁) space with contours
plt = contourf(β1_vals, K1_vals, like_for_plots, color=:dense, levels=20, lw=0,
              xlabel="β₁", ylabel="K₁", title="Profile likelihood in θ-space (direct)",
              size=(700, 600))
scatter!([β1_true], [K1_true], mc=:darkgoldenrod, msc=:match, ms=10, markershape=:star, label="True")
contour!(β1_vals, K1_vals, like_for_plots, levels=[lstar], color=:black, lw=1, legend=false)

# Add lines showing constant ratio (for reference)
# The CI should follow lines K₁ = ratio × β₁
β1_line = range(minimum(β1_vals), maximum(β1_vals), 100)
plot!(β1_line, ratio_true .* β1_line, color=:gray, lw=1, ls=:dash, label="K₁/β₁ = $ratio_true")

savefig(plt, "minimal_2D_direct_result.png")
println("Saved: minimal_2D_direct_result.png")

# === Print comparison info ===
println("\n=== Comparison with ψ-space approach ===")
println("Direct θ-space: Grid is rectangular in (β₁, K₁)")
println("  - CI appears as diagonal wedge through origin")
println("  - Points distributed uniformly in log(β₁) × log(K₁)")
println("\nTransformed ψ-space: Grid is rectangular in (K₁/β₁, β₁)")
println("  - CI appears as vertical band (ψ₁ bounded, ψ₂ unbounded)")
println("  - Points distributed along constant-ratio lines")
println("\nBoth approaches give same likelihood values,")
println("but ψ-space makes identifiability structure clearer.")

# Cleanup
rmprocs(workers())
