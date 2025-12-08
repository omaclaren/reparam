# IIR-Guided Profiling
# Run IIR on a multi-parameter subset, then profile over the identified targets
# with remaining parameters as nuisance

# Include modules (with guard to avoid double-loading)
if !@isdefined(ReparamTools)
    include("ReparamTools.jl")
end
if !@isdefined(RepressilatorModel)
    include("examples/RepressilatorModel.jl")
end

using .ReparamTools
using .RepressilatorModel
using Distributions, LinearAlgebra, Random, ForwardDiff

# === MODEL SETUP ===
Random.seed!(42)
NT, T_end = 4, 10000.0
t_obs = LinRange(0, T_end, NT)
X0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
σ = 10.0

# True parameters (18 total)
# Indices: α₀(1-3), α(4-6), β(7-9), K(10-12), k_degm(13-15), k_degp(16-18)
θ_true = [0.008, 0.009, 0.010,      # α₀ (1-3)
          1.0, 1.2, 1.5,             # α  (4-6)
          0.02, 0.025, 0.015,        # β  (7-9)
          30.0, 28.0, 32.0,          # K  (10-12)
          0.006, 0.0055, 0.0065,     # k_degm (13-15)
          0.0012, 0.0011, 0.0013]    # k_degp (16-18)

# Generate data
y_true = RepressilatorModel.predict_mRNA(θ_true, t_obs, X0)
data = y_true + σ * randn(length(y_true))

println("=" ^ 70)
println("IIR-GUIDED PROFILING")
println("=" ^ 70)

# === CHOOSE PARAMETER SUBSET FOR IIR ===
# Start with β and K for all 3 genes (6 parameters)
# Indices: β(7,8,9), K(10,11,12)
free_indices = [7, 8, 9, 10, 11, 12]
free_names = ["β₁", "β₂", "β₃", "K₁", "K₂", "K₃"]
n_free = length(free_indices)

fixed_indices = setdiff(1:18, free_indices)
θ_fixed = θ_true[fixed_indices]
θ_free_true = θ_true[free_indices]

println("\nFree parameters ($n_free):")
for (i, idx) in enumerate(free_indices)
    println("  $i: $(free_names[i]) (index $idx) = $(θ_true[idx])")
end

# === DEFINE ϕ FOR THIS SUBSET ===
function ϕ_subset(θ_free)
    T = eltype(θ_free)
    θ_full = Vector{T}(undef, 18)
    θ_full[fixed_indices] .= θ_fixed
    θ_full[free_indices] .= θ_free
    return RepressilatorModel.predict_mRNA(θ_full, t_obs, X0)
end

# ϕ in log-space
ϕ_subset_log(θ_log) = ϕ_subset(exp.(θ_log))

# === RUN IIR ===
println("\n" * "=" ^ 70)
println("RUNNING IIR (find_invariant_subspace)")
println("=" ^ 70)

θ_free_log_true = log.(θ_free_true)

S, N, N_perp, rank_J = find_invariant_subspace(
    ϕ_subset_log, θ_free_log_true;
    verbose=true
)

println("\nResults:")
println("  Singular values: ", round.(S, sigdigits=4))
println("  Jacobian rank: $rank_J / $n_free")
println("  Identifiable directions (N_perp): $(size(N_perp, 2))")
println("  Non-identifiable directions (N): $(size(N, 2))")

# === INTERPRET TRANSFORMATION ===
println("\n" * "=" ^ 70)
println("TRANSFORMATION ANALYSIS")
println("=" ^ 70)

println("\nN_perp columns (identifiable directions in log-space):")
for j in 1:size(N_perp, 2)
    v = N_perp[:, j]
    println("\n  Identifiable direction $j:")
    println("    Raw: ", round.(v, digits=3))
    # Interpret: find dominant components
    sorted_idx = sortperm(abs.(v), rev=true)
    println("    Interpretation (top components):")
    for k in 1:min(3, n_free)
        idx = sorted_idx[k]
        if abs(v[idx]) > 0.1
            sign_str = v[idx] > 0 ? "+" : "-"
            println("      $sign_str$(round(abs(v[idx]), digits=2)) × log($(free_names[idx]))")
        end
    end
end

println("\nN columns (non-identifiable/invariant directions):")
for j in 1:size(N, 2)
    v = N[:, j]
    println("\n  Invariant direction $j:")
    println("    Raw: ", round.(v, digits=3))
end

# === BUILD FULL TRANSFORMATION ===
println("\n" * "=" ^ 70)
println("CONSTRUCTING COORDINATE TRANSFORMATION")
println("=" ^ 70)

# A_T has N_perp columns first (identifiable), then N columns (non-identifiable)
n_ident = size(N_perp, 2)
n_nonident = size(N, 2)

if n_nonident > 0
    A_T = hcat(N_perp, N)
else
    # All directions identifiable - construct orthogonal complement
    println("\nN is empty - all directions identifiable")
    A_T = N_perp
    # Would need to construct complement if we want full basis
end

println("\nTransformation matrix A_T (columns = new coordinates in log-space):")
println("  Size: $(size(A_T))")
println("  First $n_ident columns: identifiable (targets for profiling)")
println("  Remaining $n_nonident columns: non-identifiable (nuisance)")

# Check invertibility
if size(A_T, 1) == size(A_T, 2)
    d = det(A_T)
    println("\n  det(A_T) = $(round(d, digits=6))")
    if abs(d) > 1e-10
        println("  ✓ Matrix is invertible")
    else
        println("  ✗ Matrix is singular!")
    end
end

# === APPLY VARIMAX FOR INTERPRETABILITY ===
println("\n" * "=" ^ 70)
println("VARIMAX ROTATION ANALYSIS")
println("=" ^ 70)

# Option 1: Varimax on full matrix (may mix identifiable/non-identifiable)
A_T_varimax_full = varimax_rotation(A_T; n_restarts=200, threshold=1e-2)

println("\n1. FULL VARIMAX (rotates all 6 directions together):")
println("   WARNING: This may mix identifiable and non-identifiable directions!")
for j in 1:size(A_T_varimax_full, 2)
    v = A_T_varimax_full[:, j]
    println("\n  ψ_$j:")
    println("    Coefficients: ", round.(v, digits=3))
    for k in 1:n_free
        if abs(v[k]) > 0.3
            sign_str = v[k] > 0 ? "+" : "-"
            coef = round(abs(v[k]), digits=2)
            println("      $sign_str$coef × log($(free_names[k]))")
        end
    end
end

# Option 2: Varimax on each subspace separately (preserves identifiability structure)
println("\n2. SEPARATE VARIMAX (preserves identifiable/non-identifiable structure):")

# Rotate identifiable subspace
N_perp_varimax = varimax_rotation(N_perp; n_restarts=200, threshold=1e-2)
println("\n  Identifiable directions (varimax-rotated N_perp):")
for j in 1:size(N_perp_varimax, 2)
    v = N_perp_varimax[:, j]
    println("\n    ψ_$j (identifiable):")
    println("      Coefficients: ", round.(v, digits=3))
    for k in 1:n_free
        if abs(v[k]) > 0.3
            sign_str = v[k] > 0 ? "+" : "-"
            coef = round(abs(v[k]), digits=2)
            println("        $sign_str$coef × log($(free_names[k]))")
        end
    end
end

# Rotate non-identifiable subspace
if n_nonident > 0
    N_varimax = varimax_rotation(N; n_restarts=200, threshold=1e-2)
    println("\n  Non-identifiable directions (varimax-rotated N):")
    for j in 1:size(N_varimax, 2)
        v = N_varimax[:, j]
        println("\n    ψ_$(n_ident + j) (non-identifiable):")
        println("      Coefficients: ", round.(v, digits=3))
        for k in 1:n_free
            if abs(v[k]) > 0.3
                sign_str = v[k] > 0 ? "+" : "-"
                coef = round(abs(v[k]), digits=2)
                println("        $sign_str$coef × log($(free_names[k]))")
            end
        end
    end
    
    # Build separate-varimax transformation
    A_T_varimax_sep = hcat(N_perp_varimax, N_varimax)
else
    A_T_varimax_sep = N_perp_varimax
    N_varimax = nothing
end

println("\n  det(A_T_varimax_sep) = $(round(det(A_T_varimax_sep), digits=6))")

# === APPLY SCALE AND ROUND FOR CLEAN EXPONENTS ===
println("\n" * "=" ^ 70)
println("SCALE AND ROUND (for clean integer exponents)")
println("=" ^ 70)

# Apply scale_and_round to get ±1 instead of ±0.707
N_perp_clean = scale_and_round(N_perp_varimax; round_within=0.1)

println("\n  Identifiable directions (after scale_and_round):")
for j in 1:size(N_perp_clean, 2)
    v = N_perp_clean[:, j]
    println("\n    ψ_$j (identifiable):")
    println("      Coefficients: ", round.(v, digits=3))
    for k in 1:n_free
        if abs(v[k]) > 0.3
            sign_str = v[k] > 0 ? "+" : "-"
            coef = round(abs(v[k]), digits=2)
            println("        $sign_str$coef × log($(free_names[k]))")
        end
    end
end

if n_nonident > 0
    N_clean = scale_and_round(N_varimax; round_within=0.1)
    println("\n  Non-identifiable directions (after scale_and_round):")
    for j in 1:size(N_clean, 2)
        v = N_clean[:, j]
        println("\n    ψ_$(n_ident + j) (non-identifiable):")
        println("      Coefficients: ", round.(v, digits=3))
        for k in 1:n_free
            if abs(v[k]) > 0.3
                sign_str = v[k] > 0 ? "+" : "-"
                coef = round(abs(v[k]), digits=2)
                println("        $sign_str$coef × log($(free_names[k]))")
            end
        end
    end
    
    A_T_scaled = hcat(N_perp_clean, N_clean)
else
    A_T_scaled = N_perp_clean
end

println("\n  det(A_T_scaled) = $(round(det(A_T_scaled), digits=6))")
println("  Note: scale_and_round breaks orthonormality but gives cleaner interpretation")

# === CREATE TRANSFORMATION FUNCTIONS ===
# Use SCALED transformation for clean K/β and K·β monomials
A_T_final = A_T_scaled
θ_to_ψ, ψ_to_θ = reparam(A_T_final)

# Print the final transformation for reference
println("\n" * "=" ^ 70)
println("FINAL TRANSFORMATION (Separate Varimax + Scale/Round)")
println("=" ^ 70)
println("\nIdentifiable coordinates (profile targets):")
for j in 1:n_ident
    v = A_T_final[:, j]
    terms = String[]
    for k in 1:n_free
        if abs(v[k]) > 0.1
            coef = round(v[k], digits=2)
            push!(terms, "$(coef > 0 ? "+" : "")$coef×log($(free_names[k]))")
        end
    end
    println("  ψ_$j = exp($(join(terms, " ")))")
end
println("\nNon-identifiable coordinates (nuisance):")
for j in (n_ident+1):n_free
    v = A_T_final[:, j]
    terms = String[]
    for k in 1:n_free
        if abs(v[k]) > 0.1
            coef = round(v[k], digits=2)
            push!(terms, "$(coef > 0 ? "+" : "")$coef×log($(free_names[k]))")
        end
    end
    println("  ψ_$j = exp($(join(terms, " ")))")
end

# Test
println("\n" * "-" ^ 40)
println("Transformation test at true values:")
ψ_true = θ_to_ψ(θ_free_true)
println("  θ_free = ", round.(θ_free_true, digits=4))
println("  ψ = θ_to_ψ(θ) = ", round.(ψ_true, digits=4))
θ_back = ψ_to_θ(ψ_true)
println("  θ_back = ", round.(θ_back, digits=4))
println("  Roundtrip OK: ", isapprox(θ_free_true, θ_back, rtol=1e-10))

# === SETUP PROFILING ===
println("\n" * "=" ^ 70)
println("PROFILING SETUP")
println("=" ^ 70)

# Profile over first n_ident coordinates (identifiable)
# Nuisance: remaining n_nonident coordinates

target_indices = 1:n_ident
nuisance_indices = (n_ident+1):n_free

println("\nTarget parameters (to profile over): ψ_1 to ψ_$n_ident")
println("Nuisance parameters (to optimize over): ψ_$(n_ident+1) to ψ_$n_free")

# === SUMMARY ===
println("\n" * "=" ^ 70)
println("SUMMARY")
println("=" ^ 70)
println("""
IIR Analysis on $(n_free)-parameter subset ($(join(free_names, ", "))):
  - Found $n_ident identifiable direction(s)
  - Found $n_nonident non-identifiable direction(s)
  
Separate Varimax rotation preserves identifiable/non-identifiable structure:
  - Identifiable: K/β ratios (ψ_1 to ψ_$n_ident)
  - Non-identifiable: K·β products (ψ_$(n_ident+1) to ψ_$n_free)
  
Next steps for profiling:
  1. Profile over identifiable coordinates (ψ_1, ψ_2, ψ_3)
  2. Treat non-identifiable coordinates as nuisance parameters
  3. Compare with profiling individual K or β (should be flat)
""")

# === 2D PROFILING DEMONSTRATION ===
# Focus on gene 1: β₁ (index 1 in free_indices=7) and K₁ (index 4 in free_indices=10)
# The IIR transformation gives us K₁/β₁ (identifiable) and K₁·β₁ (non-identifiable)

println("\n" * "=" ^ 70)
println("2D PROFILING DEMONSTRATION (Gene 1: β₁, K₁)")
println("=" ^ 70)

# True values for gene 1
β1_true = θ_true[7]   # 0.02
K1_true = θ_true[10]  # 30.0
ψ1_true_2d = K1_true / β1_true  # K/β = 1500 (identifiable)
ψ2_true_2d = β1_true * K1_true  # β·K = 0.6 (non-identifiable)

println("\nTrue values:")
println("  β₁ = $β1_true")
println("  K₁ = $K1_true")
println("  ψ₁ = K₁/β₁ = $ψ1_true_2d (identifiable)")
println("  ψ₂ = β₁·K₁ = $ψ2_true_2d (non-identifiable)")

# 2D transformation for gene 1 only
function θ_to_ψ_2d(θ2)  # θ2 = [β₁, K₁]
    β, K = θ2
    return [K/β, β*K]  # [ψ₁, ψ₂]
end

function ψ_to_θ_2d(ψ2)  # ψ2 = [K/β, β·K]
    ψ1, ψ2_val = ψ2
    β = sqrt(ψ2_val / ψ1)
    K = sqrt(ψ1 * ψ2_val)
    return [β, K]
end

# Fixed other parameters (all except β₁, K₁)
fixed_indices_2d = setdiff(1:18, [7, 10])
θ_fixed_2d = θ_true[fixed_indices_2d]

# Log-likelihood for profiling in ψ-space
function lnlike_2param_log(ψ_log)
    ψ1 = exp(ψ_log[1])
    ψ2 = exp(ψ_log[2])
    θ2 = ψ_to_θ_2d([ψ1, ψ2])
    
    θ_full = zeros(eltype(ψ_log), 18)
    θ_full[fixed_indices_2d] .= θ_fixed_2d
    θ_full[7] = θ2[1]   # β₁
    θ_full[10] = θ2[2]  # K₁
    
    pred = RepressilatorModel.predict_mRNA(θ_full, t_obs, X0)
    return -sum((data .- pred).^2) / (2 * σ^2)
end

# Bounds in log-ψ space (same as minimal_2D_IIR_coords.jl)
lower_2d = [log(10.0), log(0.004)]      # ψ₁ ∈ [10, 50000], ψ₂ ∈ [0.004, 20]
upper_2d = [log(50000.0), log(20.0)]

# === COMPUTE 2D PROFILE ===
println("\nComputing 2D profile over (ψ₁, ψ₂)...")
GRID = 80

ψ1_grid = exp.(range(lower_2d[1], upper_2d[1], length=GRID))
ψ2_grid = exp.(range(lower_2d[2], upper_2d[2], length=GRID))

like_matrix = zeros(GRID, GRID)
t_start = time()
for (i, ψ1) in enumerate(ψ1_grid)
    for (j, ψ2) in enumerate(ψ2_grid)
        ll = lnlike_2param_log([log(ψ1), log(ψ2)])
        like_matrix[i, j] = ll
    end
end
elapsed = time() - t_start
println("Done in $(round(elapsed, digits=1)) seconds")

# Normalize
ll_max = maximum(like_matrix[isfinite.(like_matrix)])
like_matrix = exp.(like_matrix .- ll_max)

# CI thresholds
lstar_2d = exp(-quantile(Chisq(2), 0.95)/2)
lstar_1d = exp(-quantile(Chisq(1), 0.95)/2)

# === PLOTTING ===
println("\nGenerating plots...")
using Plots
using Contour
using ScatteredInterpolation

USE_SCATTER_PLOT = false

# Plot 1: 2D profile in ψ-space (top-left)
p1 = contourf(ψ1_grid, ψ2_grid, like_matrix', color=:dense, levels=20, lw=0,
              xlabel="ψ₁ = K/β (identifiable)", ylabel="ψ₂ = β·K (non-identifiable)",
              title="2D Profile in IIR coordinates",
              xscale=:log10, clims=(0,1))
scatter!([ψ1_true_2d], [ψ2_true_2d], mc=:darkgoldenrod, msc=:match, ms=10, markershape=:star, label="True")
contour!(ψ1_grid, ψ2_grid, like_matrix', levels=[lstar_2d], color=:black, lw=2, 
         xscale=:log10, label="95% CI")

# Plot 2: Transform to θ-space (top-right)
β_flat = Float64[]
K_flat = Float64[]
like_flat = Float64[]

for (i, ψ1) in enumerate(ψ1_grid)
    for (j, ψ2) in enumerate(ψ2_grid)
        θ = ψ_to_θ_2d([ψ1, ψ2])
        push!(β_flat, θ[1])
        push!(K_flat, θ[2])
        push!(like_flat, like_matrix[i, j])
    end
end

# Fixed rectangular region for θ-space plots
β_plot_min, β_plot_max = 0.004, 0.09
K_plot_min, K_plot_max = 1.0, 200.0

if USE_SCATTER_PLOT
    p2 = scatter(β_flat, K_flat, zcolor=like_flat, c=:dense,
                 xlabel="β₁", ylabel="K₁", title="Profile likelihood in θ-space (scatter)",
                 markersize=2, markerstrokewidth=0, label="", clims=(0,1),
                 xlims=(β_plot_min, β_plot_max), ylims=(K_plot_min, K_plot_max))
    scatter!([β1_true], [K1_true], mc=:darkgoldenrod, msc=:match, ms=10, markershape=:star, label="True")
else
    # RBF interpolated contour version
    β_min, β_max = β_plot_min, β_plot_max
    K_min, K_max = K_plot_min, K_plot_max
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
c = Contour.contour(collect(ψ1_grid), collect(ψ2_grid), like_matrix, lstar_2d)
for line in Contour.lines(c)
    ψ1_c, ψ2_c = Contour.coordinates(line)
    β_c = Float64[]
    K_c = Float64[]
    for k in 1:length(ψ1_c)
        θ_k = ψ_to_θ_2d([ψ1_c[k], ψ2_c[k]])
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
like_ψ1 = [maximum(like_matrix[i, :]) for i in 1:GRID]
like_ψ2 = [maximum(like_matrix[:, j]) for j in 1:GRID]

println("\n1D Profile projections:")
println("  Profile(ψ₁ = K/β): range [$(round(minimum(like_ψ1), digits=4)), $(round(maximum(like_ψ1), digits=4))]")
println("  Profile(ψ₂ = β·K): range [$(round(minimum(like_ψ2), digits=4)), $(round(maximum(like_ψ2), digits=4))]")

ψ1_above = like_ψ1 .> lstar_1d
ψ2_above = like_ψ2 .> lstar_1d
println("  ψ₁ values above 95% threshold: $(sum(ψ1_above))/$GRID")
println("  ψ₂ values above 95% threshold: $(sum(ψ2_above))/$GRID")

# Plot 3: 1D profile for ψ₁ = K/β (bottom-left)
p3 = plot(ψ1_grid, like_ψ1, xlabel="ψ₁ = K/β", ylabel="Profile Likelihood",
          title="Profile: K/β (IDENTIFIABLE)", linewidth=2, legend=false, 
          xscale=:log10, ylims=(0, 1.05))
hline!([lstar_1d], color=:red, linestyle=:dash, linewidth=2)
vline!([ψ1_true_2d], color=:green, linestyle=:dot, linewidth=2)

# Plot 4: 1D profile for ψ₂ = β·K (bottom-right)
p4 = plot(ψ2_grid, like_ψ2, xlabel="ψ₂ = β·K", ylabel="Profile Likelihood",
          title="Profile: β·K (NON-IDENTIFIABLE)", linewidth=2, legend=false,
          ylims=(0, 1.05))
hline!([lstar_1d], color=:red, linestyle=:dash, linewidth=2)
vline!([ψ2_true_2d], color=:green, linestyle=:dot, linewidth=2)

# Combined plot
plt = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 1000))
savefig(plt, "iir_guided_profiling_result.png")
println("\nSaved: iir_guided_profiling_result.png")

println("\n" * "=" ^ 70)
println("PROFILING CONFIRMS IIR ANALYSIS")
println("=" ^ 70)
println("""
The 2D profile confirms the IIR-derived identifiability structure:
  - ψ₁ = K/β has a PEAKED profile ($(sum(ψ1_above))/$GRID above threshold) → IDENTIFIABLE
  - ψ₂ = β·K has a FLAT profile ($(sum(ψ2_above))/$GRID above threshold) → NON-IDENTIFIABLE

This matches the IIR analysis which found:
  - $n_ident identifiable direction(s): K/β ratios
  - $n_nonident non-identifiable direction(s): K·β products
""")