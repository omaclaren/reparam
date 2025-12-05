# Investigate IIR on 2-parameter subset (β₁, K₁)
# Goal: Understand how IIR finds identifiable vs non-identifiable directions
# and how to use these for profiling

using Distributed
addprocs(4)
println("Workers: ", workers())

# Load on main
include("examples/RepressilatorModel.jl")
include("ReparamTools.jl")
using .RepressilatorModel
using .ReparamTools
using Distributions, LinearAlgebra, Random, ForwardDiff
using Plots; gr()

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

# True values for gene 1
β1_true = θ_true[7]   # 0.02
K1_true = θ_true[10]  # 30.0

println("=" ^ 70)
println("IIR ANALYSIS ON 2-PARAMETER SUBSET (β₁, K₁)")
println("=" ^ 70)
println("\nTrue values: β₁ = $β1_true, K₁ = $K1_true")
println("Expected identifiable combination: K₁/β₁ = $(K1_true/β1_true)")
println("Expected orthogonal complement: K₁·β₁ = $(K1_true*β1_true)")

# === DEFINE ϕ FOR 2-PARAMETER SUBSET ===
# We fix all parameters except β₁ and K₁, then define ϕ as the model prediction

free_indices = [7, 10]  # β₁, K₁ in full 18-param space
fixed_indices = setdiff(1:18, free_indices)
θ_fixed = θ_true[fixed_indices]

# ϕ: [β₁, K₁] → model predictions (distribution parameters)
function ϕ_2param(θ_2)
    β1, K1 = θ_2
    
    # Reconstruct full parameter vector (must preserve Dual type for ForwardDiff)
    T = eltype(θ_2)
    θ_full = Vector{T}(undef, 18)
    θ_full[fixed_indices] .= θ_fixed
    θ_full[7] = β1
    θ_full[10] = K1
    
    # Return model predictions
    return RepressilatorModel.predict_mRNA(θ_full, t_obs, X0)
end

# ϕ in log-space: [log(β₁), log(K₁)] → predictions
ϕ_2param_log(θ_log) = ϕ_2param(exp.(θ_log))

# === RUN IIR ===
println("\n" * "=" ^ 70)
println("Running IIR (find_invariant_subspace)")
println("=" ^ 70)

θ_2_true = [β1_true, K1_true]
θ_2_log_true = log.(θ_2_true)

S, N, N_perp, rank_J = ReparamTools.find_invariant_subspace(
    ϕ_2param_log, θ_2_log_true;
    verbose=true
)

println("\nResults:")
println("  Singular values: ", round.(S, sigdigits=4))
println("  Jacobian rank: $rank_J / 2")
println("  N_perp dimensions: ", size(N_perp))
println("  N dimensions: ", size(N))

# === INTERPRET THE TRANSFORMATION ===
println("\n" * "=" ^ 70)
println("TRANSFORMATION MATRIX ANALYSIS")
println("=" ^ 70)

println("\nN_perp (identifiable directions):")
for j in 1:size(N_perp, 2)
    v = N_perp[:, j]
    println("  Column $j: [$(round(v[1], digits=4)), $(round(v[2], digits=4))]")
    # Interpret: v = [a, b] means ψ = β₁^a · K₁^b in log-space
    # i.e., log(ψ) = a·log(β₁) + b·log(K₁)
    if abs(v[1]) > 0.1 && abs(v[2]) > 0.1
        if sign(v[1]) != sign(v[2])
            ratio = -v[2]/v[1]
            println("    → Ratio pattern: K₁^$(round(ratio, digits=2))/β₁ (opposite signs)")
        else
            println("    → Product pattern: K₁·β₁ type (same signs)")
        end
    end
end

println("\nN (non-identifiable/invariant directions):")
if size(N, 2) == 0
    println("  Empty — all directions are identifiable (minimal image)")
else
    for j in 1:size(N, 2)
        v = N[:, j]
        println("  Column $j: [$(round(v[1], digits=4)), $(round(v[2], digits=4))]")
        if abs(v[1]) > 0.1 && abs(v[2]) > 0.1
            if sign(v[1]) == sign(v[2])
                println("    → Product pattern: K₁·β₁ type (same signs)")
            else
                println("    → Ratio pattern (opposite signs)")
            end
        end
    end
end

# === CONSTRUCT FULL TRANSFORMATION ===
println("\n" * "=" ^ 70)
println("FULL TRANSFORMATION MATRIX")
println("=" ^ 70)

# The full transformation combines N_perp and N
# A_T = [N_perp  N] gives columns that are the new coordinate directions
# ψ = A_T' · log(θ) transforms from θ to ψ

if size(N, 2) > 0
    A_T = hcat(N_perp, N)  # Full basis
else
    # If N is empty, we need to construct a complement manually
    # This happens when rank = 2 (both directions identifiable)
    println("\nN is empty - constructing orthogonal complement of N_perp...")
    # Use QR or direct orthogonal construction
    v1 = N_perp[:, 1]
    v2 = [-v1[2], v1[1]]  # Perpendicular in 2D
    v2 = v2 / norm(v2)
    A_T = hcat(N_perp, reshape(v2, 2, 1))
end

println("\nA_T (columns are new coordinate directions in log-space):")
println("  Column 1 (ψ₁): [$(round(A_T[1,1], digits=4)), $(round(A_T[2,1], digits=4))]")
println("  Column 2 (ψ₂): [$(round(A_T[1,2], digits=4)), $(round(A_T[2,2], digits=4))]")

# Verify invertibility
println("\ndet(A_T) = $(round(det(A_T), digits=6))")
if abs(det(A_T)) > 1e-10
    println("  ✓ Matrix is invertible")
else
    println("  ✗ Matrix is singular!")
end

# === VERIFY TRANSFORMATION ===
println("\n" * "=" ^ 70)
println("VERIFY TRANSFORMATION")
println("=" ^ 70)

# Create transformation functions using reparam()
θ_to_ψ, ψ_to_θ = ReparamTools.reparam(A_T)

println("\nTest at true values:")
println("  θ = [β₁, K₁] = $θ_2_true")
ψ_2 = θ_to_ψ(θ_2_true)
println("  ψ = θ_to_ψ(θ) = $ψ_2")
θ_back = ψ_to_θ(ψ_2)
println("  θ_back = ψ_to_θ(ψ) = $θ_back")
println("  Roundtrip match: $(isapprox(θ_2_true, θ_back))")

# Interpret ψ values
println("\nInterpretation:")
println("  ψ₁ = $(round(ψ_2[1], digits=4))")
println("  ψ₂ = $(round(ψ_2[2], digits=4))")
println("  For reference: K₁/β₁ = $(K1_true/β1_true)")
println("  For reference: K₁·β₁ = $(K1_true*β1_true)")
println("  For reference: β₁ = $β1_true")
println("  For reference: K₁ = $K1_true")

# === QUANTIFY IDENTIFIABILITY COMPROMISE FROM ROTATION ===
println("\n" * "=" ^ 70)
println("QUANTIFYING IDENTIFIABILITY STRUCTURE")
println("=" ^ 70)

# The key measure: σ_eff = ||J·v|| / ||v|| (normalized sensitivity)
# For SVD: v is orthonormal so ||v||=1, and σ_eff = singular value
# For rotated: v may mix identifiable/non-identifiable directions

# Compute Jacobian at θ_2_log_true
J_2 = ForwardDiff.jacobian(ϕ_2param_log, θ_2_log_true)
println("\nJacobian shape: ", size(J_2))

# For SVD basis (N_perp, N):
println("\nSVD basis identifiability:")
for j in 1:size(N_perp, 2)
    v = N_perp[:, j]
    σ_eff = norm(J_2 * v) / norm(v)
    println("  N_perp[$j]: ||J·v||/||v|| = $(round(σ_eff, digits=4)) (should ≈ σ[$j] = $(round(S[j], digits=4)))")
end
for j in 1:size(N, 2)
    v = N[:, j]
    σ_eff = norm(J_2 * v) / norm(v)
    println("  N[$j]: ||J·v||/||v|| = $(round(σ_eff, digits=6)) (should ≈ 0 for non-identifiable)")
end

# === APPLY VARIMAX ROTATION ===
println("\n" * "=" ^ 70)
println("VARIMAX ROTATION")
println("=" ^ 70)

# Varimax tries to make the transformation matrix sparser (closer to ±1, 0)
N_perp_varimax = ReparamTools.varimax_rotation(A_T; n_restarts=200, threshold=1e-2)

println("\nVarimax-rotated A_T:")
println("  Column 1: [$(round(N_perp_varimax[1,1], digits=4)), $(round(N_perp_varimax[2,1], digits=4))]")
println("  Column 2: [$(round(N_perp_varimax[1,2], digits=4)), $(round(N_perp_varimax[2,2], digits=4))]")

# Round to nearest simple fraction for interpretation
function interpret_vector(v)
    # Check for common patterns
    patterns = [
        ([-1.0, 1.0], "K/β"),
        ([1.0, -1.0], "β/K"),
        ([1.0, 1.0], "K·β"),
        ([-1.0, -1.0], "1/(K·β)"),
        ([1.0, 0.0], "β"),
        ([0.0, 1.0], "K"),
        ([-1.0, 0.0], "1/β"),
        ([0.0, -1.0], "1/K"),
    ]
    
    v_norm = v / norm(v)
    best_match = ""
    best_score = Inf
    
    for (pattern, name) in patterns
        p_norm = pattern / norm(pattern)
        score = norm(v_norm - p_norm)
        if score < best_score
            best_score = score
            best_match = name
        end
        # Also check negative
        score_neg = norm(v_norm + p_norm)
        if score_neg < best_score
            best_score = score_neg
            best_match = name * " (flipped)"
        end
    end
    
    return best_match, best_score
end

println("\nInterpretation of varimax columns:")
for j in 1:size(N_perp_varimax, 2)
    v = N_perp_varimax[:, j]
    match, score = interpret_vector(v)
    println("  Column $j: $match (score: $(round(score, digits=3)))")
end

# === QUANTIFY IDENTIFIABILITY LOSS FROM VARIMAX ===
println("\n" * "=" ^ 70)
println("IDENTIFIABILITY LOSS FROM VARIMAX ROTATION")
println("=" ^ 70)

println("\nComparing SVD vs Varimax basis identifiability:")
println("  (σ_eff = ||J·v||/||v|| measures how much output changes along direction v)")
println("")

# SVD directions are ordered by singular value
println("SVD Basis (ordered by identifiability):")
for j in 1:2
    if j <= size(N_perp, 2)
        v = N_perp[:, j]
        label = "Identifiable"
    else
        v = N[:, j - size(N_perp, 2)]
        label = "Non-identifiable"
    end
    σ_eff = norm(J_2 * v) / norm(v)
    println("  Direction $j: σ_eff = $(round(σ_eff, digits=4)) [$label]")
end

println("\nVarimax Basis (rotated for interpretability):")
for j in 1:2
    v = N_perp_varimax[:, j]
    σ_eff = norm(J_2 * v) / norm(v)
    match, _ = interpret_vector(v)
    println("  Direction $j: σ_eff = $(round(σ_eff, digits=4)) [$match]")
end

# KEY: Measure how much each varimax direction projects onto identifiable vs non-identifiable
println("\n" * "-" ^ 50)
println("PROJECTION ANALYSIS (mixing of identifiable/non-identifiable)")
println("-" ^ 50)

# Project each varimax direction onto SVD basis
for j in 1:2
    v_varimax = N_perp_varimax[:, j]
    match, _ = interpret_vector(v_varimax)
    
    # Project onto identifiable direction (N_perp[:, 1])
    proj_identifiable = abs(dot(v_varimax, N_perp[:, 1]))
    
    # Project onto non-identifiable direction (N[:, 1])
    proj_nonidentifiable = abs(dot(v_varimax, N[:, 1]))
    
    total = proj_identifiable + proj_nonidentifiable
    pct_identifiable = 100 * proj_identifiable / total
    pct_nonidentifiable = 100 * proj_nonidentifiable / total
    
    println("\nVarimax direction $j ($match):")
    println("  Identifiable component:     $(round(pct_identifiable, digits=1))%")
    println("  Non-identifiable component: $(round(pct_nonidentifiable, digits=1))%")
    
    if pct_nonidentifiable > 10
        println("  ⚠ WARNING: $(round(pct_nonidentifiable, digits=1))% non-identifiable mixing!")
    else
        println("  ✓ Mostly identifiable direction")
    end
end

# Create transformation with varimax
θ_to_ψ_var, ψ_to_θ_var = ReparamTools.reparam(N_perp_varimax)

println("\nTest varimax transformation at true values:")
ψ_var = θ_to_ψ_var(θ_2_true)
println("  ψ_varimax = $ψ_var")
θ_back_var = ψ_to_θ_var(ψ_var)
println("  θ_back = $θ_back_var")
println("  Roundtrip match: $(isapprox(θ_2_true, θ_back_var))")

# === SCALE AND ROUND FOR INTERPRETABILITY ===
println("\n" * "=" ^ 70)
println("SCALED AND ROUNDED TRANSFORMATION")
println("=" ^ 70)

# Try to get nice integer coefficients
A_scaled = ReparamTools.scale_and_round(A_T)
println("\nScaled/rounded A_T:")
display(A_scaled)

A_var_scaled = ReparamTools.scale_and_round(N_perp_varimax)
println("\nScaled/rounded varimax A_T:")
display(A_var_scaled)

# === COMPUTE 2D LIKELIHOOD AND PROJECT TO 1D PROFILES ===
println("\n" * "=" ^ 70)
println("2D LIKELIHOOD SURFACE AND 1D PROFILE PROJECTIONS")
println("=" ^ 70)

# Define negative log-likelihood for 2-param case
function nll_2param(θ_2)
    y_pred = ϕ_2param(θ_2)
    return sum((data .- y_pred).^2) / (2 * σ^2)
end

# Use LOG-SPACE grid matching the working profile (minimal_2D_reparam_nuisance.jl)
n_grid = 50

# Log-space ranges centered on true values (matching minimal_2D style)
β1_log_lower, β1_log_upper = log(β1_true) - 1.5, log(β1_true) + 1.5
K1_log_lower, K1_log_upper = log(K1_true) - 1.5, log(K1_true) + 1.5

β1_log_range = range(β1_log_lower, β1_log_upper, length=n_grid)
K1_log_range = range(K1_log_lower, K1_log_upper, length=n_grid)
β1_range = exp.(β1_log_range)
K1_range = exp.(K1_log_range)

println("\nGrid ranges (log-space, matching minimal_2D style):")
println("  β₁: [$(round(β1_range[1], digits=4)), $(round(β1_range[end], digits=4))]")
println("  K₁: [$(round(K1_range[1], digits=1)), $(round(K1_range[end], digits=1))]")

println("\nComputing 2D likelihood surface on $(n_grid)×$(n_grid) grid...")
nll_grid = zeros(n_grid, n_grid)
for (i, β1) in enumerate(β1_range)
    for (j, K1) in enumerate(K1_range)
        nll_grid[i, j] = nll_2param([β1, K1])
    end
end

nll_min = minimum(nll_grid)
println("  min(NLL) = $(round(nll_min, digits=2))")

# Convert to profile likelihood (relative to minimum)
pll_grid = nll_grid .- nll_min

# Project to 1D profiles by minimization over the other parameter
println("\n1D Profile Projections (minimize over other param):")

# DEBUG: Check grid orientation
println("\n  DEBUG: nll_grid[1,1] at (β₁=$(β1_range[1]), K₁=$(K1_range[1])) = $(round(nll_grid[1,1], digits=2))")
println("  DEBUG: nll_grid[end,1] at (β₁=$(β1_range[end]), K₁=$(K1_range[1])) = $(round(nll_grid[end,1], digits=2))")
println("  DEBUG: nll_grid[1,end] at (β₁=$(β1_range[1]), K₁=$(K1_range[end])) = $(round(nll_grid[1,end], digits=2))")

# Profile over β₁: for each β₁, find min over K₁
pll_β1 = [minimum(pll_grid[i, :]) for i in 1:n_grid]
# Find which K₁ achieves the minimum for each β₁
K1_at_min = [K1_range[argmin(pll_grid[i, :])] for i in 1:n_grid]
println("\n  Profile(β₁): range = [$(round(minimum(pll_β1), digits=4)), $(round(maximum(pll_β1), digits=4))]")
println("    K₁ at min for β₁=$(β1_range[1]): $(round(K1_at_min[1], digits=1))")
println("    K₁ at min for β₁=$(β1_range[end]): $(round(K1_at_min[end], digits=1))")
println("    Expected K₁ = K/β × β₁ = 1500 × β₁")
println("    For β₁=$(β1_range[1]): expected K₁ = $(round(1500*β1_range[1], digits=1))")
println("    For β₁=$(β1_range[end]): expected K₁ = $(round(1500*β1_range[end], digits=1))")
if maximum(pll_β1) < 0.5
    println("    → FLAT (max < 0.5) - β₁ is non-identifiable ✓")
else
    println("    → NOT FLAT (max = $(round(maximum(pll_β1), digits=2))) - β₁ has some identifiability")
end

# Profile over K₁: for each K₁, find min over β₁
pll_K1 = [minimum(pll_grid[:, j]) for j in 1:n_grid]
println("\n  Profile(K₁): range = [$(round(minimum(pll_K1), digits=4)), $(round(maximum(pll_K1), digits=4))]")
if maximum(pll_K1) < 0.5
    println("    → FLAT (max < 0.5) - K₁ is non-identifiable ✓")
else
    println("    → NOT FLAT (max = $(round(maximum(pll_K1), digits=2))) - K₁ has some identifiability")
end

# Now profile in transformed coordinates (ψ₁ = K/β, ψ₂ = K·β)
println("\n" * "-" ^ 50)
println("Profile in transformed coordinates:")

# Compute ψ grid from θ grid
ψ1_grid = [K1/β1 for β1 in β1_range, K1 in K1_range]  # K/β (identifiable)
ψ2_grid = [K1*β1 for β1 in β1_range, K1 in K1_range]  # K·β (non-identifiable)

ψ1_min, ψ1_max = extrema(ψ1_grid)
ψ2_min, ψ2_max = extrema(ψ2_grid)
println("  ψ₁ (K/β) range: [$(round(ψ1_min, digits=1)), $(round(ψ1_max, digits=1))]")
println("  ψ₂ (K·β) range: [$(round(ψ2_min, digits=3)), $(round(ψ2_max, digits=3))]")

# Bin by ψ₁ and find min pll in each bin
n_ψ_bins = 20
ψ1_edges = range(ψ1_min, ψ1_max, length=n_ψ_bins+1)
pll_ψ1 = fill(Inf, n_ψ_bins)
for i in 1:n_grid, j in 1:n_grid
    ψ1_val = ψ1_grid[i, j]
    bin = clamp(Int(floor((ψ1_val - ψ1_min) / (ψ1_max - ψ1_min) * n_ψ_bins)) + 1, 1, n_ψ_bins)
    pll_ψ1[bin] = min(pll_ψ1[bin], pll_grid[i, j])
end
pll_ψ1[isinf.(pll_ψ1)] .= NaN

println("\n  Profile(ψ₁ = K/β): range = [$(round(minimum(filter(!isnan, pll_ψ1)), digits=2)), $(round(maximum(filter(!isnan, pll_ψ1)), digits=2))]")
if maximum(filter(!isnan, pll_ψ1)) > 2.0
    println("    → CURVED - ψ₁ is identifiable ✓")
else
    println("    → FLAT - ψ₁ is non-identifiable")
end

# Bin by ψ₂ and find min pll in each bin
ψ2_edges = range(ψ2_min, ψ2_max, length=n_ψ_bins+1)
pll_ψ2 = fill(Inf, n_ψ_bins)
for i in 1:n_grid, j in 1:n_grid
    ψ2_val = ψ2_grid[i, j]
    bin = clamp(Int(floor((ψ2_val - ψ2_min) / (ψ2_max - ψ2_min) * n_ψ_bins)) + 1, 1, n_ψ_bins)
    pll_ψ2[bin] = min(pll_ψ2[bin], pll_grid[i, j])
end
pll_ψ2[isinf.(pll_ψ2)] .= NaN

println("\n  Profile(ψ₂ = K·β): range = [$(round(minimum(filter(!isnan, pll_ψ2)), digits=2)), $(round(maximum(filter(!isnan, pll_ψ2)), digits=2))]")
if maximum(filter(!isnan, pll_ψ2)) < 0.5
    println("    → FLAT - ψ₂ is non-identifiable ✓")
else
    println("    → CURVED (max = $(round(maximum(filter(!isnan, pll_ψ2)), digits=2))) - ψ₂ has some identifiability")
end

# === PLOTS ===
println("\n" * "=" ^ 70)
println("GENERATING PLOTS")
println("=" ^ 70)

# 2D heatmap + contour in (β₁, K₁) space
p1 = heatmap(β1_range, K1_range, pll_grid', 
    xlabel="β₁", ylabel="K₁",
    title="Profile Likelihood (β₁, K₁)",
    color=:viridis, clims=(0, 10))
contour!(p1, β1_range, K1_range, pll_grid', 
    levels=[1.92, 3.84, 5.99],
    color=:white, linewidth=2)
scatter!(p1, [β1_true], [K1_true], color=:red, markersize=8, label="True", legend=:topright)
# Add constant K/β line through true value (the ridge)
K_over_β_true = K1_true / β1_true
β_line = range(extrema(β1_range)..., length=100)
K_line = K_over_β_true .* β_line
plot!(p1, β_line, K_line, color=:red, linestyle=:dash, linewidth=2, label="K/β = $(Int(K_over_β_true))")
# Add the actual minimizing K₁ for each β₁
plot!(p1, collect(β1_range), K1_at_min, color=:cyan, linewidth=2, linestyle=:solid, label="Min K₁(β₁)")

# 1D profiles in original coordinates
p2 = plot(β1_range, pll_β1, 
    xlabel="β₁", ylabel="Profile Likelihood",
    title="1D Profile: β₁ (minimized over K₁)",
    linewidth=2, legend=false, ylims=(0, 5))
hline!(p2, [1.92], color=:red, linestyle=:dash, label="95% CI")
vline!(p2, [β1_true], color=:green, linestyle=:dot, label="True")

p3 = plot(K1_range, pll_K1, 
    xlabel="K₁", ylabel="Profile Likelihood",
    title="1D Profile: K₁ (minimized over β₁)",
    linewidth=2, legend=false, ylims=(0, 5))
hline!(p3, [1.92], color=:red, linestyle=:dash, label="95% CI")
vline!(p3, [K1_true], color=:green, linestyle=:dot, label="True")

# 1D profiles in transformed coordinates
ψ1_centers = [(ψ1_edges[i] + ψ1_edges[i+1])/2 for i in 1:n_ψ_bins]
ψ2_centers = [(ψ2_edges[i] + ψ2_edges[i+1])/2 for i in 1:n_ψ_bins]

p4 = plot(ψ1_centers, pll_ψ1, 
    xlabel="ψ₁ = K/β", ylabel="Profile Likelihood",
    title="1D Profile: K/β (identifiable)",
    linewidth=2, legend=false, xscale=:log10)
hline!(p4, [1.92], color=:red, linestyle=:dash, label="95% CI")
vline!(p4, [K1_true/β1_true], color=:green, linestyle=:dot, label="True")

p5 = plot(ψ2_centers, pll_ψ2, 
    xlabel="ψ₂ = K·β", ylabel="Profile Likelihood",
    title="1D Profile: K·β (non-identifiable)",
    linewidth=2, legend=false, ylims=(0, 5), xscale=:log10)
hline!(p5, [1.92], color=:red, linestyle=:dash, label="95% CI")
vline!(p5, [K1_true*β1_true], color=:green, linestyle=:dot, label="True")

# Combined plot
p_combined = plot(p1, p2, p3, p4, p5, layout=(2, 3), size=(1200, 700))
savefig(p_combined, "investigate_IIR_2param_profiles.png")
println("\nSaved: investigate_IIR_2param_profiles.png")

# === SUMMARY ===
println("\n" * "=" ^ 70)
println("SUMMARY")
println("=" ^ 70)

println("""
For the 2-parameter case (β₁, K₁):

1. IIR finds the identifiable direction (N_perp) via SVD + invariance test
2. The orthogonal complement (N or constructed) gives the non-identifiable direction
3. Together they form an invertible transformation A_T

Key findings:
- Orthogonal IIR gives: ψ₁ ∝ K/β (identifiable), ψ₂ ∝ K·β (non-identifiable complement)
- Varimax rotation can simplify to: ψ₁ ∝ K/β, ψ₂ ∝ β (more interpretable)

For profiling:
- Profile over ψ₁ (identifiable) shows tight CI
- Profile over ψ₂ (or β₁ with varimax) shows wide CI
- 2D profile in (ψ₁, ψ₂) shows vertical band (identifiable in ψ₁ direction)
- Transformed to (β₁, K₁) shows diagonal wedge

Next steps:
- Extend to more parameters (e.g., add β₂ as nuisance)
- Use IIR-derived transformation systematically for profiling
""")

# Cleanup
rmprocs(workers())
