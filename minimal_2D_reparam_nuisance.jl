# Minimal 2D Profile - Reparameterized Space with Nuisance Parameter
# Profile in (ψ₁ = K₁/β₁, ψ₂ = β₁) space, optimizing over β₂ as nuisance
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
β2_true = θ_true[8]   # 0.025 - nuisance parameter
ψ1_true = K1_true / β1_true  # 1500 (the identifiable ratio)

println("True values: β₁=$β1_true, K₁=$K1_true, β₂=$β2_true")
println("True ψ₁=K₁/β₁=$ψ1_true")

# === REPARAMETERIZATION using reparam() ===
# Transform: θ = [β₁, K₁] → ψ = [K₁/β₁, β₁]
A_T = [-1.0  1.0;    # col 1: ψ₁ = K/β (exponents: β^-1 * K^1)
        1.0  0.0]    # col 2: ψ₂ = β   (exponents: β^1)

θ_to_ψ, ψ_to_θ = ReparamTools.reparam(A_T)

# Test the transformation
θ_2 = [β1_true, K1_true]
ψ_2 = θ_to_ψ(θ_2)
println("\nTransformation test:")
println("  θ = [β₁, K₁] = $θ_2 → ψ = [K₁/β₁, β₁] = $ψ_2")

# === LIKELIHOOD in ψ space with β₂ nuisance ===
# Free parameters: β₁, β₂, K₁ (indices 7, 8, 10 in full model)
# We work in transformed space: [ψ₁, ψ₂, β₂] = [K₁/β₁, β₁, β₂]
free_orig_indices = [7, 8, 10]  # β₁, β₂, K₁
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
    θ_to_ψ_local, ψ_to_θ_local = ReparamTools.reparam(A_T)
end

# Likelihood in 3-param log space: [log(ψ₁), log(ψ₂), log(β₂)]
@everywhere function lnlike_ψ3_log(ψ3_log)
    # ψ3_log = [log(ψ₁), log(ψ₂), log(β₂)] where ψ₁ = K₁/β₁, ψ₂ = β₁
    ψ1 = exp(ψ3_log[1])
    ψ2 = exp(ψ3_log[2])
    β2 = exp(ψ3_log[3])
    
    # Transform ψ → θ
    θ_2 = ψ_to_θ_local([ψ1, ψ2])  # [β₁, K₁]
    β1 = θ_2[1]
    K1 = θ_2[2]
    
    # Check positivity
    if β1 <= 0 || K1 <= 0 || β2 <= 0
        return -Inf
    end
    
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

# === BOUNDS in 3-param ψ-log space ===
ψ_true = [ψ1_true, β1_true]  # [1500, 0.02]
ψ_log_true = log.(ψ_true)

# Same ranges as minimal_2D_reparam.jl for ψ₁ and ψ₂
# Add β₂ range
ψ3_log_lower = [ψ_log_true[1] - 2.5, ψ_log_true[2] - 1.5, log(0.005)]  # ψ₁, ψ₂, β₂
ψ3_log_upper = [ψ_log_true[1] + 1.5, ψ_log_true[2] + 1.5, log(0.12)]   # ψ₁, ψ₂, β₂

# Target indices: ψ₁=1, ψ₂=2 (profile over these)
# Nuisance index: β₂=3 (optimize at each grid point)
target = [1, 2]

# === 2D PROFILE in ψ space with nuisance optimization ===
println("\n2D profile in reparameterized space (ψ₁=K₁/β₁, ψ₂=β₁)...")
println("With β₂ as nuisance parameter (optimized at each grid point)")

GRID = 50  # 50×50 = 2500 points (smaller grid since optimization at each point)
t_start = time()

nuisance_guess = [log(β2_true)]  # Initial guess for β₂

ψ_vals, ll_vals = ReparamTools.profile_target(lnlike_ψ3_log, target, ψ3_log_lower, ψ3_log_upper, nuisance_guess;
                                  grid_steps=GRID,
                                  use_distributed=true,
                                  method=:LN_BOBYQA,  # Gradient-free for distributed
                                  optmaxtime=10.0)

elapsed = time() - t_start
println("Done in $(round(elapsed, digits=1)) seconds")
println("Finite values: $(sum(isfinite.(ll_vals)))/$(length(ll_vals))")
println("LL range: $(minimum(ll_vals)) to $(maximum(ll_vals))")

# === PLOT in BOTH spaces ===
using Plots
using Distributions
using Contour

# Construct grid values in ψ space (same as profile_target uses)
ψ1_log_range = range(ψ3_log_lower[1], ψ3_log_upper[1], length=GRID)
ψ2_log_range = range(ψ3_log_lower[2], ψ3_log_upper[2], length=GRID)
ψ1_vals = exp.(ψ1_log_range)
ψ2_vals = exp.(ψ2_log_range)

# Reshape
ll_matrix = reshape(ll_vals, GRID, GRID)
ll_max = maximum(ll_matrix[isfinite.(ll_matrix)])
like_matrix = exp.(ll_matrix .- ll_max)
like_for_plots = like_matrix'

# Chi-square threshold
lstar = exp(-quantile(Chisq(2), 0.95)/2)
println("95% CI threshold (likelihood): $(round(lstar, digits=3))")

# Plot 1: Reparameterized space (ψ₁ vs ψ₂)
p1 = contourf(ψ1_vals, ψ2_vals, like_for_plots, color=:dense, levels=20, lw=0,
              xlabel="ψ₁ = K₁/β₁", ylabel="ψ₂ = β₁", 
              title="Profile likelihood in ψ-space\n(β₂ optimized)",
              clims=(0,1))
scatter!([ψ1_true], [β1_true], mc=:darkgoldenrod, msc=:match, ms=10, markershape=:star, label="True")
contour!(ψ1_vals, ψ2_vals, like_for_plots, levels=[lstar], color=:black, lw=1, legend=false)

# Plot 2: Transform back to original (β₁, K₁) space
β1_flat = Float64[]
K1_flat = Float64[]
like_flat = Float64[]

for (i, ψ1) in enumerate(ψ1_vals)
    for (j, ψ2) in enumerate(ψ2_vals)
        local θ_2 = ψ_to_θ([ψ1, ψ2])
        push!(β1_flat, θ_2[1])
        push!(K1_flat, θ_2[2])
        push!(like_flat, like_matrix[i, j])
    end
end

p2 = scatter(β1_flat, K1_flat, zcolor=like_flat, c=:dense,
             xlabel="β₁", ylabel="K₁", 
             title="Profile likelihood in θ-space\n(β₂ optimized)",
             markersize=3, markerstrokewidth=0, label="",
             xlims=(0, 0.1), ylims=(0, 200), clims=(0,1))
scatter!([β1_true], [K1_true], mc=:darkgoldenrod, msc=:match, ms=10, markershape=:star, label="True")

# Extract 95% CI contour from ψ-space and transform to θ-space
n_lines = 0
for lev in [lstar]
    c = Contour.contour(collect(ψ1_vals), collect(ψ2_vals), like_matrix, lev)
    for line in Contour.lines(c)
        global n_lines += 1
        ψ1_contour, ψ2_contour = Contour.coordinates(line)
        
        # Transform each point to θ-space
        β1_contour = Float64[]
        K1_contour = Float64[]
        for k in 1:length(ψ1_contour)
            local θ_2 = ψ_to_θ([ψ1_contour[k], ψ2_contour[k]])
            push!(β1_contour, θ_2[1])
            push!(K1_contour, θ_2[2])
        end
        
        plot!(p2, β1_contour, K1_contour, color=:black, lw=1, label="")
    end
end
println("Found $n_lines CI contour lines")

# Combined plot
plt = plot(p1, p2, layout=(1,2), size=(1200, 500))
savefig(plt, "minimal_2D_reparam_nuisance_result.png")
println("Saved: minimal_2D_reparam_nuisance_result.png")

# Cleanup
rmprocs(workers())
