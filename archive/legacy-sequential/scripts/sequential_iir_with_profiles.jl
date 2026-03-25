"""
Sequential IIR with Profile Likelihood Plots
Demonstrates complete workflow:
1. Find reparameterization (SVD bases)
2. Filter active directions (Jacobian)
3. Post-process (Varimax interpretation)
4. Implement transformation
5. Generate profile plots
"""

using LinearAlgebra
using Distributions
using Random
using Plots
include("../ReparamTools.jl")
using .ReparamTools

println("="^60)
println("Sequential IIR with Profile Likelihood")
println("Sum of Independent Poisson Limits")
println("="^60)

# ============================================================
# MODEL SETUP
# ============================================================

θ_true = [21.0, 0.9, 110.0, 0.18]  # [n₁, p₁, n₂, p₂]
ϕ_θ(θ) = [θ[1]*θ[2] + θ[3]*θ[4], θ[1]*θ[2] + θ[3]*θ[4]]
λ_true = θ_true[1] * θ_true[2] + θ_true[3] * θ_true[4]

# Fixed data (realistic sample size)
data = [36.4, 40.3, 36.7, 36.8, 43.8, 41.7, 33.4, 29.6, 37.4, 36.8]

distrib_θ = θ -> Normal(ϕ_θ(θ)[1], sqrt(ϕ_θ(θ)[2]))

# Construct likelihood
function construct_lnlike(distrib, data)
    return θ -> sum(logpdf.(distrib(θ), data))
end

lnlike_θ = construct_lnlike(distrib_θ, data)

# Bounds for original parameters
θ_lower = [0.1, 0.0001, 0.1, 0.0001]
θ_upper = [500.0, 1.0, 500.0, 1.0]
θ_initial = [50.0, 0.3, 50.0, 0.3]

println("\nModel: Y ~ N(μ=λ, σ²=λ) where λ = n₁p₁ + n₂p₂")
println("Data: ", length(data), " observations (fixed)")
println("True λ = ", λ_true)

# ============================================================
# FIND MLE IN ORIGINAL COORDINATES
# ============================================================

println("\n" * "="^60)
println("MLE Estimation (Original Coordinates)")
println("="^60)

θ_MLE, lnlike_MLE = profile_target(lnlike_θ, Int[], θ_lower, θ_upper, θ_initial)

println("\nMLE (original parameters):")
println("  n₁ = ", round(θ_MLE[1], digits=4))
println("  p₁ = ", round(θ_MLE[2], digits=4))
println("  n₂ = ", round(θ_MLE[3], digits=4))
println("  p₂ = ", round(θ_MLE[4], digits=4))
println("  λ_MLE = ", round(θ_MLE[1]*θ_MLE[2] + θ_MLE[3]*θ_MLE[4], digits=2))

# ============================================================
# STAGE 1: f=log (Find multiplicative structure)
# ============================================================

println("\n" * "="^60)
println("STAGE 1: f=log (SVD basis)")
println("="^60)

ϕ_log = θ_log -> ϕ_θ(exp.(θ_log))

S_s1, N_s1, N_perp_s1, rank_s1 = find_invariant_subspace(
    ϕ_log, log.(θ_MLE); rtolJ=sqrt(eps()), atolM=1e-10
)

println("  Rank: ", rank_s1, "/4")
println("  Dim(N_perp): ", size(N_perp_s1, 2))
println("  Dim(N): ", size(N_s1, 2))

# ============================================================
# STAGE 2: f=identity (Find additive structure)
# ============================================================

println("\n" * "="^60)
println("STAGE 2: f=identity (SVD basis)")
println("="^60)

# Build Stage 1 transformation (square, orthonormal)
A1 = vcat(N_perp_s1', N_s1')
θ_to_θ1(θ) = exp.(A1 * log.(θ))
θ1_to_θ(θ1) = exp.(A1' * log.(θ1))

θ1_MLE = θ_to_θ1(θ_MLE)
ϕ_stage2(θ1) = ϕ_θ(θ1_to_θ(θ1))

S_s2, N_s2, N_perp_s2, rank_s2 = find_invariant_subspace(
    ϕ_stage2, θ1_MLE; rtolJ=sqrt(eps()), atolM=1e-10
)

println("  Rank: ", rank_s2, "/4")
println("  Dim(N_perp): ", size(N_perp_s2, 2))
println("  Dim(N): ", size(N_s2, 2))

# ============================================================
# FILTER: Identify active directions
# ============================================================

println("\n" * "="^60)
println("JACOBIAN FILTERING")
println("="^60)

J_s2 = compute_ϕ_Jacobian(ϕ_stage2, θ1_MLE)

active_cols = Int[]
for col in 1:size(N_perp_s2, 2)
    v = N_perp_s2[:, col]
    Jv_norm = norm(J_s2 * v)
    is_active = Jv_norm > 1e-6
    println("  Column $col: ||J*v|| = ", round(Jv_norm, digits=2), is_active ? " ✓" : " ✗")
    if is_active
        push!(active_cols, col)
    end
end

println("\nActive directions: ", active_cols)
println("True identifiable dimension: ", length(active_cols))

# ============================================================
# POST-PROCESSING: Varimax interpretation
# ============================================================

println("\n" * "="^60)
println("POST-PROCESSING: Varimax Dictionary")
println("="^60)

N_perp_varimax = varimax_rotation(N_perp_s1; n_restarts=200, threshold=1e-2)
N_perp_var_scaled = scale_and_round(N_perp_varimax; column_scales=ones(size(N_perp_varimax, 2)))

println("\nVarimax basis (Stage 1):")
display(N_perp_var_scaled)

param_names = ["n₁", "p₁", "n₂", "p₂"]
println("\nInterpretation:")
for col in 1:size(N_perp_var_scaled, 2)
    exps = N_perp_var_scaled[:, col]
    terms = String[]
    for i in 1:4
        if abs(exps[i]) > 1e-6
            push!(terms, param_names[i])
        end
    end
    println("  Column $col: ", join(terms, "·"))
end

# ============================================================
# IMPLEMENT FINAL TRANSFORMATION
# ============================================================

println("\n" * "="^60)
println("FINAL TRANSFORMATION (using Varimax basis)")
println("="^60)

# Use Varimax basis for N_perp, keep SVD basis for N
# This gives interpretable coordinates
N_perp_final = N_perp_varimax
N_final = N_s1

A_final = vcat(N_perp_final', N_final')

# Transformation functions
θ_to_ψ(θ) = exp.(A_final * log.(θ))
ψ_to_θ(ψ) = exp.(inv(A_final) * log.(ψ))

# Transform MLE
ψ_MLE = θ_to_ψ(θ_MLE)

println("\nTransformed MLE:")
println("  ψ₁ (n₁p₁) = ", round(ψ_MLE[1], digits=4))
println("  ψ₂ (n₂p₂) = ", round(ψ_MLE[2], digits=4))
println("  ψ₃ = ", round(ψ_MLE[3], digits=4))
println("  ψ₄ = ", round(ψ_MLE[4], digits=4))
println("  Sum ψ₁+ψ₂ = ", round(ψ_MLE[1] + ψ_MLE[2], digits=2))

# Likelihood in transformed coordinates
lnlike_ψ = ψ -> lnlike_θ(ψ_to_θ(ψ))

# Transformed bounds (approximate - could be refined)
ψ_lower = θ_to_ψ(θ_lower)
ψ_upper = θ_to_ψ(θ_upper)

println("\nTransformed parameter bounds:")
for i in 1:4
    println("  ψ[$i]: [", round(ψ_lower[i], digits=4), ", ", round(ψ_upper[i], digits=4), "]")
end

# ============================================================
# PROFILE LIKELIHOOD (Original Coordinates)
# ============================================================

println("\n" * "="^60)
println("PROFILE LIKELIHOOD: Original Coordinates")
println("="^60)

# 1D profiles for each original parameter
plots_original = []

for i in 1:4
    println("\nProfiling parameter $i ($(param_names[i]))...")

    # Create grid
    margin = 0.3
    grid = collect(range(θ_MLE[i] * (1-margin), θ_MLE[i] * (1+margin), length=50))

    profile_vals = Float64[]
    for val in grid
        # Temporarily fix parameter i at val
        θ_test = copy(θ_MLE)
        θ_test[i] = val
        push!(profile_vals, lnlike_θ(θ_test))
    end

    p = plot(grid, profile_vals,
             label="Profile",
             xlabel=param_names[i],
             ylabel="Log-likelihood",
             title="Profile: $(param_names[i])",
             linewidth=2)
    vline!([θ_MLE[i]], label="MLE", linestyle=:dash)

    push!(plots_original, p)
end

plot_orig = plot(plots_original..., layout=(2,2), size=(800, 600))
savefig(plot_orig, "examples/figures/sequential_iir_profiles_original.png")
println("\n✓ Saved: examples/figures/sequential_iir_profiles_original.png")

# ============================================================
# PROFILE LIKELIHOOD (Transformed Coordinates)
# ============================================================

println("\n" * "="^60)
println("PROFILE LIKELIHOOD: Transformed Coordinates")
println("="^60)

ψ_names = ["ψ₁ (n₁p₁)", "ψ₂ (n₂p₂)", "ψ₃", "ψ₄"]
plots_transformed = []

for i in 1:4
    println("\nProfiling ψ[$i] ($(ψ_names[i]))...")

    # Create grid
    margin = 0.5
    grid = collect(range(ψ_MLE[i] * (1-margin), ψ_MLE[i] * (1+margin), length=50))

    profile_vals = Float64[]
    for val in grid
        # Temporarily fix parameter i at val
        ψ_test = copy(ψ_MLE)
        ψ_test[i] = val
        push!(profile_vals, lnlike_ψ(ψ_test))
    end

    p = plot(grid, profile_vals,
             label="Profile",
             xlabel=ψ_names[i],
             ylabel="Log-likelihood",
             title="Profile: $(ψ_names[i])",
             linewidth=2)
    vline!([ψ_MLE[i]], label="MLE", linestyle=:dash)

    push!(plots_transformed, p)
end

plot_trans = plot(plots_transformed..., layout=(2,2), size=(800, 600))
savefig(plot_trans, "examples/figures/sequential_iir_profiles_transformed.png")
println("\n✓ Saved: examples/figures/sequential_iir_profiles_transformed.png")

# ============================================================
# 2D PROFILE: Sum vs Ratio
# ============================================================

println("\n" * "="^60)
println("2D PROFILE: Sum (invariant) vs Ratio (identifiable)")
println("="^60)

# Create 2D grid for ψ₁ and ψ₂
n_grid = 30
ψ1_range = range(ψ_MLE[1] * 0.5, ψ_MLE[1] * 1.5, length=n_grid)
ψ2_range = range(ψ_MLE[2] * 0.5, ψ_MLE[2] * 1.5, length=n_grid)

ll_grid = zeros(n_grid, n_grid)

println("Computing 2D likelihood surface...")
for (i, ψ1) in enumerate(ψ1_range)
    for (j, ψ2) in enumerate(ψ2_range)
        ψ_test = copy(ψ_MLE)
        ψ_test[1] = ψ1
        ψ_test[2] = ψ2
        ll_grid[i, j] = lnlike_ψ(ψ_test)
    end
end

# Convert to sum and ratio coordinates
sum_range = ψ1_range .+ ψ2_range[1]  # Approximate
ratio_range = ψ1_range ./ ψ2_range[1]  # Approximate

p_2d = heatmap(ψ1_range, ψ2_range, ll_grid',
               xlabel="ψ₁ (n₁p₁)",
               ylabel="ψ₂ (n₂p₂)",
               title="2D Likelihood: ψ₁ vs ψ₂",
               color=:viridis)
scatter!([ψ_MLE[1]], [ψ_MLE[2]], label="MLE", markersize=8, color=:red)

# Add diagonal line (constant sum)
sum_MLE = ψ_MLE[1] + ψ_MLE[2]
ψ1_diag = collect(ψ1_range)
ψ2_diag = sum_MLE .- ψ1_diag
valid = (ψ2_diag .>= minimum(ψ2_range)) .& (ψ2_diag .<= maximum(ψ2_range))
plot!(ψ1_diag[valid], ψ2_diag[valid], label="Sum = $(round(sum_MLE, digits=1))",
      linestyle=:dash, linewidth=2, color=:white)

savefig(p_2d, "examples/figures/sequential_iir_2d_profile.png")
println("\n✓ Saved: examples/figures/sequential_iir_2d_profile.png")

# ============================================================
# SUMMARY
# ============================================================

println("\n" * "="^60)
println("SUMMARY")
println("="^60)
println("""
Sequential IIR Workflow (Clean):
1. Stage 1 (f=log): Found 2D span [n₁p₁, n₂p₂] using SVD
2. Stage 2 (f=identity): Identified 1D active via Jacobian
3. Post-process: Varimax shows interpretable basis
4. Implement: Use Varimax basis for transformation
5. Profile: Generate plots in both coordinate systems

Key Results:
- IDENTIFIABLE: Ratio n₁p₁/n₂p₂ (difference in log-space)
- INVARIANT: Sum n₁p₁ + n₂p₂ (flat profile expected)

Plots Generated:
- Original coordinates: 4 individual profiles
- Transformed coordinates: 4 individual profiles
- 2D profile: Shows ridge along constant sum

Dictionary (Varimax) used ONLY for:
- Interpretation of basis vectors
- Final transformation to readable coordinates
- NOT used during invariance detection
""")
