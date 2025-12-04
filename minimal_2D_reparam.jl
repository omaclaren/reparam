# Minimal 2D Profile - Reparameterized Space
# Use ReparamTools' reparam() function to profile in (ψ₁ = K₁/β₁, ψ₂ = β₁) space
# Then transform back to (β₁, K₁) for plotting

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
NT, T_end = 4, 10000.0   # Fewer observations → wider CI
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
ψ1_true = K1_true / β1_true  # 1500 (the identifiable ratio)

println("True values: β₁=$β1_true, K₁=$K1_true, ψ₁=K₁/β₁=$ψ1_true")

# === REPARAMETERIZATION using reparam() ===
# We work with 2 params: θ = [β₁, K₁] (indices 7, 10 in full model)
# Transform to: ψ = [K₁/β₁, β₁]
# 
# In log space: log(ψ) = A * log(θ) where
#   log(ψ₁) = log(K₁) - log(β₁) = -1*log(β₁) + 1*log(K₁)
#   log(ψ₂) = log(β₁)           =  1*log(β₁) + 0*log(K₁)
#
# So A = [-1  1;   (rows are the exponents)
#          1  0]
# And A_T (columns are combinations) for reparam() is:
# A_T = [-1  1;
#         1  0]

# reparam() expects columns to be the combinations
A_T = [-1.0  1.0;    # col 1: ψ₁ = K/β (exponents: β^-1 * K^1)
        1.0  0.0]    # col 2: ψ₂ = β   (exponents: β^1)

# Create transformations using ReparamTools
θ_to_ψ, ψ_to_θ = ReparamTools.reparam(A_T)

# Test the transformation
θ_2 = [β1_true, K1_true]  # 2-param subset
ψ_2 = θ_to_ψ(θ_2)
θ_back = ψ_to_θ(ψ_2)
println("\nTransformation test:")
println("  θ = [β₁, K₁] = $θ_2")
println("  ψ = [K₁/β₁, β₁] = $ψ_2")
println("  θ_back = $θ_back")
println("  Match: $(isapprox(θ_2, θ_back))")

# === LIKELIHOOD in ψ space ===
# Fixed parameters (all except β₁, K₁)
free_orig_indices = [7, 10]  # β₁, K₁ in original 18-param space
fixed_indices = setdiff(1:18, free_orig_indices)
θ_fixed = θ_true[fixed_indices]

@everywhere t_obs = $t_obs
@everywhere X0 = $X0
@everywhere σ = $σ
@everywhere data = $data
@everywhere fixed_indices = $fixed_indices
@everywhere θ_fixed = $θ_fixed

# Send transformation to workers
@everywhere A_T = $A_T
@everywhere begin
    # Recreate transformation functions on workers
    θ_to_ψ_local, ψ_to_θ_local = ReparamTools.reparam(A_T)
end

# Likelihood in ψ-log space
@everywhere function lnlike_ψ_log(ψ_log)
    # ψ_log = [log(ψ₁), log(ψ₂)] where ψ₁ = K₁/β₁, ψ₂ = β₁
    ψ = exp.(ψ_log)
    θ_2 = ψ_to_θ_local(ψ)  # [β₁, K₁]
    
    # Reconstruct full 18-parameter vector
    θ_full = zeros(18)
    θ_full[fixed_indices] = θ_fixed
    θ_full[7] = θ_2[1]   # β₁
    θ_full[10] = θ_2[2]  # K₁
    
    try
        pred = RepressilatorModel.predict_mRNA(θ_full, t_obs, X0)
        return logpdf(MvNormal(pred, σ^2 * I(length(pred))), data)
    catch
        return -Inf
    end
end

# === BOUNDS in ψ-log space ===
ψ_true = [ψ1_true, β1_true]  # [1500, 0.02]
ψ_log_true = log.(ψ_true)

# Asymmetric range: extend lower ψ₁ to capture low K₁ + high β₁ combos
# Low K₁ (~5) with high β₁ (~0.09) needs ψ₁ ~ 55
# So extend ψ₁ down to ~50: log(50) - log(1500) ≈ -3.4
ψ_log_lower = [ψ_log_true[1] - 2.5, ψ_log_true[2] - 1.5]  # ψ₁ down to ~120, ψ₂ down to ~0.0045
ψ_log_upper = [ψ_log_true[1] + 1.5, ψ_log_true[2] + 1.5]  # ψ₁ up to ~6700, ψ₂ up to ~0.09

# Both are targets (no nuisance params - just 2D grid)
target = [1, 2]

# === 2D PROFILE in ψ space ===
println("\n2D profile in reparameterized space (ψ₁=K₁/β₁, ψ₂=β₁)...")
println("No nuisance optimization needed - just grid evaluation")

GRID = 100  # 100×100 = 10000 points
t_start = time()

nuisance_guess = Float64[]  # No nuisance params

ψ_vals, ll_vals = ReparamTools.profile_target(lnlike_ψ_log, target, ψ_log_lower, ψ_log_upper, nuisance_guess;
                                  grid_steps=GRID,
                                  use_distributed=true,
                                  optmaxtime=10.0)

elapsed = time() - t_start
println("Done in $(round(elapsed, digits=1)) seconds")
println("Finite values: $(sum(isfinite.(ll_vals)))/$(length(ll_vals))")
println("LL range: $(minimum(ll_vals)) to $(maximum(ll_vals))")

# === PLOT in BOTH spaces ===
using Plots
using Distributions  # for Chisq

# Extract grid values in ψ space
ψ1_vals = unique([exp(ψ[1]) for ψ in ψ_vals])  # K₁/β₁
ψ2_vals = unique([exp(ψ[2]) for ψ in ψ_vals])  # β₁

# Reshape: ll_vals is ordered as (ψ1, ψ2) pairs with ψ1 varying fastest
# After reshape: ll_matrix[i, j] = LL at (ψ1_vals[i], ψ2_vals[j])
ll_matrix = reshape(ll_vals, length(ψ1_vals), length(ψ2_vals))

# Convert to normalized likelihood scale (matching visualization.jl)
ll_max = maximum(ll_matrix[isfinite.(ll_matrix)])
like_matrix = exp.(ll_matrix .- ll_max)  # Normalized likelihood in [0, 1]

# For Plots.jl contourf(x, y, z): expects z[j, i] at (x[i], y[j]), so pass like_matrix'
like_for_plots = like_matrix'

# Chi-square threshold for 95% CI with 2 parameters
l_level = 95
df = 2
lstar = exp(-quantile(Chisq(df), l_level/100)/2)
println("95% CI threshold (likelihood): $(round(lstar, digits=3))")

# Plot 1: Reparameterized space (ψ₁ vs ψ₂) - using :dense colormap like visualization.jl
p1 = contourf(ψ1_vals, ψ2_vals, like_for_plots, color=:dense, levels=20, lw=0,
              xlabel="ψ₁ = K₁/β₁", ylabel="ψ₂ = β₁", title="Profile likelihood in ψ-space")
scatter!([ψ1_true], [β1_true], mc=:darkgoldenrod, msc=:match, ms=10, markershape=:star, label="True")
contour!(ψ1_vals, ψ2_vals, like_for_plots, levels=[lstar], color=:black, lw=1, legend=false)

# Plot 2: Transform back to original (β₁, K₁) space
# For each grid point ψ = [ψ₁, ψ₂], compute θ = [β₁, K₁]
# like_matrix[i, j] = likelihood at (ψ1_vals[i], ψ2_vals[j])

β1_flat = Float64[]
K1_flat = Float64[]
like_flat = Float64[]

for (i, ψ1) in enumerate(ψ1_vals)
    for (j, ψ2) in enumerate(ψ2_vals)
        local θ_2 = ψ_to_θ([ψ1, ψ2])
        push!(β1_flat, θ_2[1])
        push!(K1_flat, θ_2[2])
        push!(like_flat, like_matrix[i, j])  # like_matrix[i,j] at (ψ1_vals[i], ψ2_vals[j])
    end
end

p2 = scatter(β1_flat, K1_flat, zcolor=like_flat, c=:dense,
             xlabel="β₁", ylabel="K₁", title="Profile likelihood in θ-space",
             markersize=2, markerstrokewidth=0, label="",
             xlims=(0, 0.1), ylims=(0, 200))
scatter!([β1_true], [K1_true], mc=:darkgoldenrod, msc=:match, ms=10, markershape=:star, label="True")

# Extract 95% CI contour from ψ-space and transform to θ-space
using Contour

# Use the direct contour() API which returns (xs, ys) arrays correctly
# For Contour.contour(x, y, z, level): expects z[i,j] at (x[i], y[j])
# like_matrix[i, j] = likelihood at (ψ1_vals[i], ψ2_vals[j]) - matches!

global n_lines = 0
for lev in [lstar]
    c = Contour.contour(collect(ψ1_vals), collect(ψ2_vals), like_matrix, lev)
    for line in Contour.lines(c)
        global n_lines += 1
        ψ1_contour, ψ2_contour = Contour.coordinates(line)  # Returns (xs, ys) tuple
        
        # Transform each point to θ-space
        β1_contour = Float64[]
        K1_contour = Float64[]
        for k in 1:length(ψ1_contour)
            local θ_2 = ψ_to_θ([ψ1_contour[k], ψ2_contour[k]])
            push!(β1_contour, θ_2[1])
            push!(K1_contour, θ_2[2])
        end
        
        # Black solid line matching visualization.jl style
        plot!(p2, β1_contour, K1_contour, color=:black, lw=1, label="")
    end
end
println("Found $n_lines CI contour lines")

# Combined plot
plt = plot(p1, p2, layout=(1,2), size=(1200, 500))
savefig(plt, "minimal_2D_reparam_result.png")
println("Saved: minimal_2D_reparam_result.png")

# Cleanup
rmprocs(workers())
