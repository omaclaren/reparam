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
using Distributions, LinearAlgebra, Random, ForwardDiff

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
using ScatteredInterpolation
using Contour

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

# Plot 2: Transform back to original (β₁, K₁) space using ThinPlate RBF
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
        push!(like_flat, like_matrix[i, j])
    end
end

# Create regular grid in θ-space for plotting
β1_min, β1_max = 0.004, 0.09
K1_min, K1_max = 0.5, 200.0
β1_reg = range(β1_min, β1_max, length=100)
K1_reg = range(K1_min, K1_max, length=100)

# Normalize scattered points to [0,1] for RBF interpolation
β1_norm = (β1_flat .- β1_min) ./ (β1_max - β1_min)
K1_norm = (K1_flat .- K1_min) ./ (K1_max - K1_min)

# Create ThinPlate RBF interpolant in normalized space
points_norm = hcat(β1_norm, K1_norm)'  # 2 × N matrix
itp = interpolate(ThinPlate(), points_norm, like_flat)

# Evaluate on regular grid (in normalized coordinates)
like_θ_reg = zeros(length(β1_reg), length(K1_reg))
for (i, β1) in enumerate(β1_reg)
    β1_n = (β1 - β1_min) / (β1_max - β1_min)
    for (j, K1) in enumerate(K1_reg)
        K1_n = (K1 - K1_min) / (K1_max - K1_min)
        like_θ_reg[i, j] = evaluate(itp, [β1_n, K1_n])[1]
    end
end

# Clamp to [0, 1]
like_θ_reg = clamp.(like_θ_reg, 0.0, 1.0)

p2 = contourf(collect(β1_reg), collect(K1_reg), like_θ_reg', color=:dense, levels=20, lw=0,
             xlabel="β₁", ylabel="K₁", title="Profile likelihood in θ-space",
             xlims=(0, 0.1), ylims=(0, 200), clims=(0,1))
scatter!([β1_true], [K1_true], mc=:darkgoldenrod, msc=:match, ms=10, markershape=:star, label="True")

# Extract 95% CI contour from ψ-space and transform to θ-space
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

# === 1D PROFILE PROJECTIONS (row/column maxima of 2D grid) ===
println("\n" * "=" ^ 60)
println("1D PROFILE PROJECTIONS (from 2D grid)")
println("=" ^ 60)

# --- ψ-space profiles ---
# Profile over ψ₁ (K/β): max over ψ₂ (β₁) for each ψ₁
like_ψ1 = [maximum(like_matrix[i, :]) for i in 1:length(ψ1_vals)]
println("\nψ-SPACE:")
println("  Profile(ψ₁ = K/β): max over β₁ → range [$(round(minimum(like_ψ1), digits=4)), $(round(maximum(like_ψ1), digits=4))]")

# Profile over ψ₂ (β₁): max over ψ₁ (K/β) for each ψ₂
like_ψ2 = [maximum(like_matrix[:, j]) for j in 1:length(ψ2_vals)]
println("  Profile(ψ₂ = β₁): max over K/β → range [$(round(minimum(like_ψ2), digits=4)), $(round(maximum(like_ψ2), digits=4))]")

# --- θ-space profiles (from interpolated grid) ---
# Profile over β₁: max over K₁ for each β₁
like_β1 = [maximum(like_θ_reg[i, :]) for i in 1:length(β1_reg)]
println("\nθ-SPACE:")
println("  Profile(β₁): max over K₁ → range [$(round(minimum(like_β1), digits=4)), $(round(maximum(like_β1), digits=4))]")

# Profile over K₁: max over β₁ for each K₁
like_K1 = [maximum(like_θ_reg[:, j]) for j in 1:length(K1_reg)]
println("  Profile(K₁): max over β₁ → range [$(round(minimum(like_K1), digits=4)), $(round(maximum(like_K1), digits=4))]")

# 1D CI threshold
lstar_1d = exp(-quantile(Chisq(1), 0.95)/2)
println("\n95% CI threshold (1D): $(round(lstar_1d, digits=3))")

# ψ-space 1D profile plots
p3 = plot(ψ1_vals, like_ψ1, xlabel="ψ₁ = K₁/β₁", ylabel="Profile Likelihood",
          title="ψ-space: K/β", linewidth=2, legend=false, ylims=(0, 1.05))
hline!([lstar_1d], color=:red, linestyle=:dash, linewidth=2)
vline!([ψ1_true], color=:green, linestyle=:dot, linewidth=2)

p4 = plot(ψ2_vals, like_ψ2, xlabel="ψ₂ = β₁", ylabel="Profile Likelihood",
          title="ψ-space: β₁", linewidth=2, legend=false, ylims=(0, 1.05))
hline!([lstar_1d], color=:red, linestyle=:dash, linewidth=2)
vline!([β1_true], color=:green, linestyle=:dot, linewidth=2)

# θ-space 1D profile plots
p5 = plot(collect(β1_reg), like_β1, xlabel="β₁", ylabel="Profile Likelihood",
          title="θ-space: β₁", linewidth=2, legend=false, ylims=(0, 1.05))
hline!([lstar_1d], color=:red, linestyle=:dash, linewidth=2)
vline!([β1_true], color=:green, linestyle=:dot, linewidth=2)

p6 = plot(collect(K1_reg), like_K1, xlabel="K₁", ylabel="Profile Likelihood",
          title="θ-space: K₁", linewidth=2, legend=false, ylims=(0, 1.05))
hline!([lstar_1d], color=:red, linestyle=:dash, linewidth=2)
vline!([K1_true], color=:green, linestyle=:dot, linewidth=2)

# Combined 6-panel plot: 2D plots on top, 1D profiles below
plt_all = plot(p1, p2, p3, p4, p5, p6, layout=(3,2), size=(1200, 1200))
savefig(plt_all, "minimal_2D_reparam_with_1D.png")
println("\nSaved: minimal_2D_reparam_with_1D.png")

# === HESSIAN ANALYSIS (curvature at MLE) ===
println("\n" * "=" ^ 60)
println("HESSIAN ANALYSIS (profile likelihood curvature at MLE)")
println("=" ^ 60)

# Define log-likelihood functions for Hessian computation
# θ-space: [β₁, K₁]
function nll_θ(θ_2)
    β1, K1 = θ_2
    θ_full = zeros(eltype(θ_2), 18)
    θ_full[fixed_indices] .= θ_fixed
    θ_full[7] = β1
    θ_full[10] = K1
    pred = RepressilatorModel.predict_mRNA(θ_full, t_obs, X0)
    return sum((data .- pred).^2) / (2 * σ^2)
end

# ψ-space: [ψ₁ = K/β, ψ₂ = β]
function nll_ψ(ψ_2)
    ψ1, ψ2 = ψ_2
    β1 = ψ2
    K1 = ψ1 * ψ2
    θ_full = zeros(eltype(ψ_2), 18)
    θ_full[fixed_indices] .= θ_fixed
    θ_full[7] = β1
    θ_full[10] = K1
    pred = RepressilatorModel.predict_mRNA(θ_full, t_obs, X0)
    return sum((data .- pred).^2) / (2 * σ^2)
end

# Compute Hessians at MLE
θ_mle = [β1_true, K1_true]
ψ_mle = [ψ1_true, β1_true]

H_θ = ForwardDiff.hessian(nll_θ, θ_mle)
H_ψ = ForwardDiff.hessian(nll_ψ, ψ_mle)

println("\n--- JOINT HESSIANS ---")
println("\nθ-space Hessian at MLE [β₁, K₁]:")
display(round.(H_θ, digits=2))

println("\nψ-space Hessian at MLE [K/β, β]:")
display(round.(H_ψ, digits=8))

# === PROFILE INFORMATION via Schur complement ===
# I_profile(ψ) = H_ψψ - H_ψν * H_νν^(-1) * H_νψ
println("\n--- PROFILE INFORMATION (Schur complement) ---")
println("Formula: I_profile(ψ) = H_ψψ - H_ψν * H_νν⁻¹ * H_νψ")

# θ-space: profile over β₁ (nuisance = K₁)
H_ββ = H_θ[1,1]
H_βK = H_θ[1,2]
H_KK = H_θ[2,2]
I_profile_β1 = H_ββ - H_βK^2 / H_KK
println("\nθ-space, profile over β₁ (nuisance = K₁):")
println("  H_ββ = $(round(H_ββ, digits=2))")
println("  H_βK = $(round(H_βK, digits=4))")
println("  H_KK = $(round(H_KK, digits=6))")
println("  I_profile(β₁) = H_ββ - H_βK²/H_KK = $(round(I_profile_β1, digits=4))")

# θ-space: profile over K₁ (nuisance = β₁)
I_profile_K1 = H_KK - H_βK^2 / H_ββ
println("\nθ-space, profile over K₁ (nuisance = β₁):")
println("  I_profile(K₁) = H_KK - H_βK²/H_ββ = $(round(I_profile_K1, digits=8))")

# ψ-space: profile over ψ₁=K/β (nuisance = ψ₂=β)
H_11 = H_ψ[1,1]  # K/β
H_12 = H_ψ[1,2]
H_22 = H_ψ[2,2]  # β
I_profile_ψ1 = H_11 - H_12^2 / H_22
println("\nψ-space, profile over ψ₁=K/β (nuisance = ψ₂=β):")
println("  H_ψ₁ψ₁ = $(round(H_11, sigdigits=4))")
println("  H_ψ₁ψ₂ = $(round(H_12, sigdigits=4))")
println("  H_ψ₂ψ₂ = $(round(H_22, sigdigits=4))")
if abs(H_22) > 1e-15
    println("  I_profile(K/β) = $(round(I_profile_ψ1, sigdigits=4))")
else
    println("  I_profile(K/β) = H_ψ₁ψ₁ = $(round(H_11, sigdigits=4)) (H_ψ₂ψ₂ ≈ 0)")
end

# ψ-space: profile over ψ₂=β (nuisance = ψ₁=K/β)
I_profile_ψ2 = H_22 - H_12^2 / H_11
println("\nψ-space, profile over ψ₂=β (nuisance = ψ₁=K/β):")
println("  I_profile(β) = H_ψ₂ψ₂ - H_ψ₁ψ₂²/H_ψ₁ψ₁ = $(round(I_profile_ψ2, sigdigits=4))")

println("\n" * "-" ^ 40)
println("SUMMARY:")
println("  θ-space profile info: β₁ → $(round(I_profile_β1, sigdigits=3)), K₁ → $(round(I_profile_K1, sigdigits=3))")
println("  ψ-space profile info: K/β → $(round(H_11, sigdigits=3)), β → $(round(I_profile_ψ2, sigdigits=3))")
println("\n  Near-zero profile info = flat profile = non-identifiable")

# Cleanup
rmprocs(workers())
