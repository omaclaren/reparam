# Minimal 2D Profile - IIR-derived Coordinates
# Profile over (ψ₁ = K/β, ψ₂ = β·K) where ψ₁ is identifiable, ψ₂ is non-identifiable
# This anticipates the IIR-derived transformation

if !@isdefined(ReparamTools)
    include("ReparamTools.jl")
end
if !@isdefined(RepressilatorModel)
    include("examples/RepressilatorModel.jl")
end

using .ReparamTools
using .RepressilatorModel
using Distributions, LinearAlgebra, Random, ForwardDiff

# === PLOTTING OPTIONS ===
USE_SCATTER_PLOT = false  # true = scatter plot, false = RBF interpolated contour

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

# True values for gene 1
β1_true = θ_true[7]   # 0.02
K1_true = θ_true[10]  # 30.0

# IIR-derived coordinates:
#   ψ₁ = K/β (identifiable)
#   ψ₂ = β·K (non-identifiable)
ψ1_true = K1_true / β1_true  # 1500
ψ2_true = β1_true * K1_true  # 0.6

println("=" ^ 70)
println("MINIMAL 2D PROFILE - IIR COORDINATES")
println("=" ^ 70)
println("\nTrue values:")
println("  θ-space: β₁ = $β1_true, K₁ = $K1_true")
println("  ψ-space: ψ₁ = K/β = $ψ1_true (identifiable)")
println("           ψ₂ = β·K = $ψ2_true (non-identifiable)")

# === REPARAMETERIZATION ===
# θ = [β, K] → ψ = [K/β, β·K]
#
# In log space: 
#   log(ψ₁) = log(K) - log(β) = -1×log(β) + 1×log(K)
#   log(ψ₂) = log(β) + log(K) = +1×log(β) + 1×log(K)
#
# A_T columns are the exponents for each ψ:
#       [ψ₁   ψ₂]
# A_T = [-1   +1;  # β exponents
#        +1   +1]  # K exponents

A_T = [-1.0  1.0;    # col 1: ψ₁ = K/β (β^-1 · K^1)
        1.0  1.0]    # col 2: ψ₂ = β·K (β^1 · K^1)

θ_to_ψ, ψ_to_θ = reparam(A_T)

# Verify transformation
θ_2 = [β1_true, K1_true]
ψ_2 = θ_to_ψ(θ_2)
θ_back = ψ_to_θ(ψ_2)

println("\nTransformation verification:")
println("  θ = [β, K] = $θ_2")
println("  ψ = [K/β, β·K] = $ψ_2")
println("  θ_back = $θ_back")
println("  Roundtrip OK: $(isapprox(θ_2, θ_back))")

# Verify determinant (should be -2 for this transformation)
println("  det(A_T) = $(det(A_T))")

# === BOUNDS in ψ-log space ===
# To cover a rectangular region in θ-space: β ∈ [0.004, 0.1], K ∈ [1, 200]
# We need:
#   ψ₁ = K/β: [1/0.1, 200/0.004] = [10, 50000]
#   ψ₂ = β·K: [0.004*1, 0.1*200] = [0.004, 20]
#
# Note: Not all (ψ₁, ψ₂) combinations map to valid θ within bounds!
# The transformation creates a non-rectangular region in ψ-space.

# Use bounds that give good θ coverage
ψ_log_lower = [log(10.0), log(0.004)]    # ψ₁ ≥ 10, ψ₂ ≥ 0.004
ψ_log_upper = [log(50000.0), log(20.0)]  # ψ₁ ≤ 50000, ψ₂ ≤ 20

println("\nBounds in ψ-space:")
println("  ψ₁ = K/β: [$(round(exp(ψ_log_lower[1]), sigdigits=3)), $(round(exp(ψ_log_upper[1]), sigdigits=3))]")
println("  ψ₂ = β·K: [$(round(exp(ψ_log_lower[2]), sigdigits=3)), $(round(exp(ψ_log_upper[2]), sigdigits=3))]")

# Check corresponding θ bounds at corners
println("\nCorner check (ψ → θ):")
corners = [(ψ_log_lower[1], ψ_log_lower[2]),
           (ψ_log_lower[1], ψ_log_upper[2]),
           (ψ_log_upper[1], ψ_log_lower[2]),
           (ψ_log_upper[1], ψ_log_upper[2])]
for (i, (ψ1_log, ψ2_log)) in enumerate(corners)
    ψ = exp.([ψ1_log, ψ2_log])
    θ = ψ_to_θ(ψ)
    println("  Corner $i: ψ=[$(round(ψ[1], sigdigits=3)), $(round(ψ[2], sigdigits=3))] → θ=[β=$(round(θ[1], sigdigits=3)), K=$(round(θ[2], sigdigits=3))]")
end

# === LIKELIHOOD in ψ-log space ===
fixed_indices = setdiff(1:18, [7, 10])  # All except β₁, K₁
θ_fixed = θ_true[fixed_indices]

function lnlike_ψ_log(ψ_log)
    ψ = exp.(ψ_log)
    θ_2 = ψ_to_θ(ψ)  # [β, K]
    
    # Check positivity
    if any(θ_2 .<= 0)
        return -Inf
    end
    
    # Reconstruct full parameter vector
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

# === 2D PROFILE ===
println("\n" * "=" ^ 70)
println("2D PROFILE (grid evaluation, no nuisance)")
println("=" ^ 70)

target = [1, 2]  # Both ψ₁ and ψ₂
GRID = 80

t_start = time()
ψ_vals, ll_vals = profile_target(lnlike_ψ_log, target, ψ_log_lower, ψ_log_upper, Float64[];
                                  grid_steps=GRID, use_distributed=false)
elapsed = time() - t_start

println("Done in $(round(elapsed, digits=1)) seconds")
println("Finite values: $(sum(isfinite.(ll_vals)))/$(length(ll_vals))")
println("LL range: $(minimum(ll_vals[isfinite.(ll_vals)])) to $(maximum(ll_vals))")

# === PLOTS ===
using Plots
using Contour
using ScatteredInterpolation

# Extract grid
ψ1_grid = exp.(range(ψ_log_lower[1], ψ_log_upper[1], length=GRID))
ψ2_grid = exp.(range(ψ_log_lower[2], ψ_log_upper[2], length=GRID))

ll_matrix = reshape(ll_vals, GRID, GRID)
ll_max = maximum(ll_matrix[isfinite.(ll_matrix)])
like_matrix = exp.(ll_matrix .- ll_max)

# Chi-square thresholds
lstar_2d = exp(-quantile(Chisq(2), 0.95)/2)
lstar_1d = exp(-quantile(Chisq(1), 0.95)/2)

# Plot 1: ψ-space (log scale for ψ₁)
p1 = contourf(ψ1_grid, ψ2_grid, like_matrix', color=:dense, levels=20, lw=0,
              xlabel="ψ₁ = K/β (identifiable)", ylabel="ψ₂ = β·K (non-identifiable)", 
              title="Profile likelihood in IIR coordinates",
              xscale=:log10, clims=(0,1))
scatter!([ψ1_true], [ψ2_true], mc=:darkgoldenrod, msc=:match, ms=10, markershape=:star, label="True")
contour!(ψ1_grid, ψ2_grid, like_matrix', levels=[lstar_2d], color=:black, lw=2, 
         xscale=:log10, label="95% CI")

# Plot 2: θ-space (scatter or RBF interpolation)
β_flat = Float64[]
K_flat = Float64[]
like_flat = Float64[]

for (i, ψ1) in enumerate(ψ1_grid)
    for (j, ψ2) in enumerate(ψ2_grid)
        θ = ψ_to_θ([ψ1, ψ2])
        push!(β_flat, θ[1])
        push!(K_flat, θ[2])
        push!(like_flat, like_matrix[i, j])
    end
end

# Compute axis limits for θ-space plot based on ψ₂ constraint
# For a nice rectangular plot, we want max(β)*max(K) ≤ ψ₂_upper
# This ensures all displayed (β, K) combinations are "compatible"
ψ2_upper = exp(ψ_log_upper[2])  # Upper bound on β·K
β_plot_max = 0.1                 # Desired max β for display
K_plot_max = ψ2_upper / 0.01     # K such that 0.01 * K = ψ₂_upper (for min reasonable β)
K_plot_max = min(K_plot_max, 200.0)  # Cap at reasonable value
β_plot_min = 0.004
K_plot_min = 1.0

# Ensure the product constraint is satisfied at corners
# β_plot_max * K_plot_max should be ≤ ψ₂_upper for visual consistency
if β_plot_max * K_plot_max > ψ2_upper
    # Scale down to satisfy constraint
    scale = sqrt(ψ2_upper / (β_plot_max * K_plot_max))
    β_plot_max *= scale
    K_plot_max *= scale
end

println("\nθ-space plot limits (constrained by ψ₂ = β·K ≤ $(round(ψ2_upper, sigdigits=3))):")
println("  β ∈ [$(round(β_plot_min, sigdigits=3)), $(round(β_plot_max, sigdigits=3))]")
println("  K ∈ [$(round(K_plot_min, sigdigits=3)), $(round(K_plot_max, sigdigits=3))]")

if USE_SCATTER_PLOT
    # Scatter plot version with constrained axes
    p2 = scatter(β_flat, K_flat, zcolor=like_flat, c=:dense,
                 xlabel="β₁", ylabel="K₁", title="Profile likelihood in θ-space (scatter)",
                 markersize=2, markerstrokewidth=0, label="", clims=(0,1),
                 xlims=(β_plot_min, β_plot_max), ylims=(K_plot_min, K_plot_max))
    scatter!([β1_true], [K1_true], mc=:darkgoldenrod, msc=:match, ms=10, markershape=:star, label="True")
else
    # RBF interpolated contour version
    # Use the rectangular region where we have data coverage
    β_min, β_max = 0.004, 0.09
    K_min, K_max = 1.0, 200.0
    β_reg = range(β_min, β_max, length=100)
    K_reg = range(K_min, K_max, length=100)

    # Filter scattered points to those within our rectangular region
    in_region = (β_flat .>= β_min) .& (β_flat .<= β_max) .& (K_flat .>= K_min) .& (K_flat .<= K_max)
    β_filt = β_flat[in_region]
    K_filt = K_flat[in_region]
    like_filt = like_flat[in_region]

    # Normalize scattered points to [0,1] for RBF interpolation
    β_norm = (β_filt .- β_min) ./ (β_max - β_min)
    K_norm = (K_filt .- K_min) ./ (K_max - K_min)

    # Create ThinPlate RBF interpolant in normalized space
    points_norm = hcat(β_norm, K_norm)'  # 2 × N matrix
    itp = interpolate(ThinPlate(), points_norm, like_filt)

    # Evaluate on regular grid
    like_θ_reg = zeros(length(β_reg), length(K_reg))
    for (i, β) in enumerate(β_reg)
        β_n = (β - β_min) / (β_max - β_min)
        for (j, K) in enumerate(K_reg)
            K_n = (K - K_min) / (K_max - K_min)
            like_θ_reg[i, j] = evaluate(itp, [β_n, K_n])[1]
        end
    end

    # Clamp to [0, 1]
    like_θ_reg = clamp.(like_θ_reg, 0.0, 1.0)

    p2 = contourf(collect(β_reg), collect(K_reg), like_θ_reg', color=:dense, levels=20, lw=0,
                 xlabel="β₁", ylabel="K₁", title="Profile likelihood in θ-space",
                 xlims=(β_min, β_max), ylims=(K_min, K_max), clims=(0,1))
    scatter!([β1_true], [K1_true], mc=:darkgoldenrod, msc=:match, ms=10, markershape=:star, label="True")
end

# Transform CI contour to θ-space, only plotting within rectangular region
β_plot_min, β_plot_max = 0.004, 0.09
K_plot_min, K_plot_max = 1.0, 200.0
c = Contour.contour(collect(ψ1_grid), collect(ψ2_grid), like_matrix, lstar_2d)
for line in Contour.lines(c)
    ψ1_c, ψ2_c = Contour.coordinates(line)
    β_c = Float64[]
    K_c = Float64[]
    for k in 1:length(ψ1_c)
        θ_k = ψ_to_θ([ψ1_c[k], ψ2_c[k]])
        # Only include points within the rectangular plot region
        if β_plot_min <= θ_k[1] <= β_plot_max && K_plot_min <= θ_k[2] <= K_plot_max
            push!(β_c, θ_k[1])
            push!(K_c, θ_k[2])
        end
    end
    if length(β_c) > 1
        plot!(p2, β_c, K_c, color=:black, lw=2, label="")
    end
end

# === 1D PROFILES ===
# Profile over ψ₁: max over ψ₂ for each ψ₁
like_ψ1 = [maximum(like_matrix[i, :]) for i in 1:GRID]

# Profile over ψ₂: max over ψ₁ for each ψ₂
like_ψ2 = [maximum(like_matrix[:, j]) for j in 1:GRID]

println("\n" * "=" ^ 70)
println("1D PROFILE PROJECTIONS")
println("=" ^ 70)
println("\nProfile(ψ₁ = K/β): range [$(round(minimum(like_ψ1), digits=4)), $(round(maximum(like_ψ1), digits=4))]")
println("Profile(ψ₂ = β·K): range [$(round(minimum(like_ψ2), digits=4)), $(round(maximum(like_ψ2), digits=4))]")

# Check if profile drops below threshold
ψ1_above = like_ψ1 .> lstar_1d
ψ2_above = like_ψ2 .> lstar_1d
println("\nψ₁ values above 95% threshold: $(sum(ψ1_above))/$(GRID)")
println("ψ₂ values above 95% threshold: $(sum(ψ2_above))/$(GRID)")

p3 = plot(ψ1_grid, like_ψ1, xlabel="ψ₁ = K/β", ylabel="Profile Likelihood",
          title="Profile: K/β (IDENTIFIABLE)", linewidth=2, legend=false, 
          xscale=:log10, ylims=(0, 1.05))
hline!([lstar_1d], color=:red, linestyle=:dash, linewidth=2)
vline!([ψ1_true], color=:green, linestyle=:dot, linewidth=2)

p4 = plot(ψ2_grid, like_ψ2, xlabel="ψ₂ = β·K", ylabel="Profile Likelihood",
          title="Profile: β·K (NON-IDENTIFIABLE)", linewidth=2, legend=false,
          ylims=(0, 1.05))
hline!([lstar_1d], color=:red, linestyle=:dash, linewidth=2)
vline!([ψ2_true], color=:green, linestyle=:dot, linewidth=2)

# Combined plot
plt = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 1000))
savefig(plt, "minimal_2D_IIR_coords_result.png")
println("\nSaved: minimal_2D_IIR_coords_result.png")

# === HESSIAN ANALYSIS ===
println("\n" * "=" ^ 70)
println("HESSIAN ANALYSIS")
println("=" ^ 70)

function nll_ψ(ψ)
    θ = ψ_to_θ(ψ)
    θ_full = zeros(eltype(ψ), 18)
    θ_full[fixed_indices] .= θ_fixed
    θ_full[7] = θ[1]
    θ_full[10] = θ[2]
    pred = RepressilatorModel.predict_mRNA(θ_full, t_obs, X0)
    return sum((data .- pred).^2) / (2 * σ^2)
end

H_ψ = ForwardDiff.hessian(nll_ψ, [ψ1_true, ψ2_true])
eigvals_H = eigvals(H_ψ)

println("\nHessian at MLE in ψ-space:")
println("  H[1,1] (∂²/∂ψ₁²) = $(round(H_ψ[1,1], sigdigits=4))  ← K/β curvature")
println("  H[2,2] (∂²/∂ψ₂²) = $(round(H_ψ[2,2], sigdigits=4))  ← β·K curvature")
println("  H[1,2] (cross)   = $(round(H_ψ[1,2], sigdigits=4))")
println("\nEigenvalues: $(round.(eigvals_H, sigdigits=4))")

# Profile information via Schur complement
I_profile_ψ1 = H_ψ[1,1] - H_ψ[1,2]^2 / H_ψ[2,2]
I_profile_ψ2 = H_ψ[2,2] - H_ψ[1,2]^2 / H_ψ[1,1]

println("\nProfile information (Schur complement):")
println("  I_profile(ψ₁ = K/β) = $(round(I_profile_ψ1, sigdigits=4))")
println("  I_profile(ψ₂ = β·K) = $(round(I_profile_ψ2, sigdigits=4))")
println("\n→ Near-zero profile info = flat profile = non-identifiable")

println("\n" * "=" ^ 70)
println("SUMMARY")
println("=" ^ 70)
println("""
Using IIR-derived coordinates:
  ψ₁ = K/β (identifiable)   → should have peaked 1D profile
  ψ₂ = β·K (non-identifiable) → should have flat 1D profile

This matches IIR finding that K/β ratios are identifiable
while β·K products are non-identifiable (invariant directions).
""")
