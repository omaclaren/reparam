"""
Sequential IIR: Final Clean Example
Matches stat_sum_model.jl data generation (Normal approximation)
Dictionary as post-processing only
"""

using LinearAlgebra
using Distributions
using Random
include("../../../ReparamTools.jl")
using .ReparamTools

println("="^60)
println("Sequential IIR: Clean Example")
println("Sum of Independent Poisson Limits (Normal Approximation)")
println("="^60)

# ============================================================
# MODEL SETUP - Match stat_model.jl approach
# ============================================================

θ_true = [21.0, 0.9, 110.0, 0.18]  # [n₁, p₁, n₂, p₂]

# Auxiliary mapping: ϕ(θ) = [μ, σ²] where μ = σ² = n₁p₁ + n₂p₂
ϕ_θ(θ) = [θ[1]*θ[2] + θ[3]*θ[4], θ[1]*θ[2] + θ[3]*θ[4]]

λ_true = θ_true[1] * θ_true[2] + θ_true[3] * θ_true[4]

println("\nModel:")
println("  Parameters: θ = [n₁, p₁, n₂, p₂]")
println("  Auxiliary: ϕ(θ) = [λ, λ] where λ = n₁p₁ + n₂p₂")
println("  Data: Y ~ N(μ=λ, σ²=λ)")
println("  True λ = ", λ_true)

# Fixed data (generated once with Random.seed!(42), N=10)
# Generated using: rand(Normal(38.7, sqrt(38.7)), 10)
data = [36.4, 40.3, 36.7, 36.8, 43.8, 41.7, 33.4, 29.6, 37.4, 36.8]

println("\nUsing fixed dataset (", length(data), " observations)")
println("  Sample mean: ", round(mean(data), digits=2))
println("  Sample variance: ", round(var(data), digits=2))

# Likelihood
function negloglik(θ)
    μ_σ² = ϕ_θ(θ)
    μ, σ² = μ_σ²[1], μ_σ²[2]
    -sum(logpdf.(Normal(μ, sqrt(σ²)), data))
end

# ============================================================
# STAGE 1: f=log
# ============================================================

println("\n" * "="^60)
println("STAGE 1: f=log (Multiplicative Structure)")
println("="^60)

ϕ_log = θ_log -> ϕ_θ(exp.(θ_log))

S_s1, N_s1, N_perp_s1, rank_s1 = find_invariant_subspace(
    ϕ_log, log.(θ_true); rtolJ=sqrt(eps()), atolM=1e-10
)

println("\nResults:")
println("  Singular values: ", round.(S_s1, digits=6))
println("  Rank: ", rank_s1, "/4")
println("  Dim(N_perp): ", size(N_perp_s1, 2), " (potentially identifiable)")
println("  Dim(N): ", size(N_s1, 2), " (invariant)")

# Build square transformation
A1 = vcat(N_perp_s1', N_s1')
θ_to_θ1(θ) = exp.(A1 * log.(θ))
θ1_to_θ(θ1) = exp.(A1' * log.(θ1))

θ1_MLE = θ_to_θ1(θ_true)
println("\nStage 1 MLE coordinates:")
for i in 1:4
    println("  θ¹[$i] = ", round(θ1_MLE[i], digits=4))
end

# ============================================================
# STAGE 2: f=identity
# ============================================================

println("\n" * "="^60)
println("STAGE 2: f=identity (Additive Structure)")
println("="^60)

ϕ_stage2(θ1) = ϕ_θ(θ1_to_θ(θ1))

S_s2, N_s2, N_perp_s2, rank_s2 = find_invariant_subspace(
    ϕ_stage2, θ1_MLE; rtolJ=sqrt(eps()), atolM=1e-10
)

println("\nResults (raw from find_invariant_subspace):")
println("  Singular values: ", round.(S_s2, digits=6))
println("  Rank: ", rank_s2, "/4")
println("  Dim(N_perp): ", size(N_perp_s2, 2))
println("  Dim(N): ", size(N_s2, 2))

# ============================================================
# FILTER: Identify active directions via Jacobian
# ============================================================

println("\n" * "="^60)
println("FILTERING: Jacobian Projection to Find Active Directions")
println("="^60)

J_s2 = compute_ϕ_Jacobian(ϕ_stage2, θ1_MLE)
println("\nJacobian shape: ", size(J_s2))

println("\nChecking N_perp columns:")
J_norms = Float64[]
active_cols = Int[]

for col in 1:size(N_perp_s2, 2)
    v = N_perp_s2[:, col]
    Jv_norm = norm(J_s2 * v)
    push!(J_norms, Jv_norm)

    is_active = Jv_norm > 1e-6
    println("  Column $col: ||J*v|| = ", round(Jv_norm, digits=6),
            is_active ? " → ACTIVE" : " → null")

    if is_active
        push!(active_cols, col)
    end
end

println("\nActive columns: ", active_cols)
println("TRUE identifiable dimension: ", length(active_cols))

# ============================================================
# POST-PROCESSING: Varimax for Interpretation
# ============================================================

println("\n" * "="^60)
println("POST-PROCESSING: Varimax Dictionary")
println("="^60)

# Compute Varimax rotation of Stage 1 basis
N_perp_varimax = varimax_rotation(N_perp_s1; n_restarts=200, threshold=1e-2)
N_perp_var_scaled = scale_and_round(N_perp_varimax; column_scales=ones(size(N_perp_varimax, 2)))

println("\nStage 1 Varimax basis (scaled):")
display(N_perp_var_scaled)

param_names = ["n₁", "p₁", "n₂", "p₂"]
println("\n\nInterpretation:")
for col in 1:size(N_perp_var_scaled, 2)
    exps = N_perp_var_scaled[:, col]
    terms = String[]
    for i in 1:4
        if abs(exps[i]) > 1e-6
            exp_str = abs(exps[i] - 1.0) < 1e-6 ? "" : "^" * string(round(exps[i], digits=2))
            push!(terms, param_names[i] * exp_str)
        end
    end
    println("  θ¹[$col]: ", join(terms, "·"))
end

# Rotation matrix
R_s1 = N_perp_s1' * N_perp_varimax

# Interpret active directions in Varimax basis
if !isempty(active_cols)
    println("\n" * "="^60)
    println("ACTIVE DIRECTIONS (Identifiable)")
    println("="^60)

    for (idx, col) in enumerate(active_cols)
        v_svd = N_perp_s2[:, col]
        n_perp_dim = size(N_perp_s1, 2)
        v_reduced = v_svd[1:n_perp_dim]

        # Rotate to Varimax
        v_varimax = R_s1' * v_reduced
        v_varimax_norm = v_varimax / norm(v_varimax)

        println("\nActive Direction $idx:")
        println("  SVD coords: ", round.(v_svd, digits=4))
        println("  Varimax coords (normalized): ", round.(v_varimax_norm, digits=4))

        # Interpret
        if n_perp_dim == 2
            sum_like = abs(v_varimax_norm[1] - v_varimax_norm[2]) < 0.1
            diff_like = abs(v_varimax_norm[1] + v_varimax_norm[2]) < 0.1

            if sum_like
                println("  → Combination: θ¹[1] + θ¹[2]")
                println("  → In parameters: n₁p₁ + n₂p₂")
                println("  → Meaning: The SUM is identifiable")
            elseif diff_like
                println("  → Combination: θ¹[1] - θ¹[2]")
                println("  → In parameters: n₁p₁ / n₂p₂ (ratio)")
                println("  → Meaning: The DIFFERENCE is identifiable")
            else
                println("  → Mixed combination")
            end
        end
    end
end

# Interpret invariant directions
if size(N_s2, 2) > 0
    println("\n" * "="^60)
    println("INVARIANT DIRECTIONS (Non-identifiable)")
    println("="^60)

    for col in 1:min(2, size(N_s2, 2))
        v_svd = N_s2[:, col]
        n_perp_dim = size(N_perp_s1, 2)
        v_reduced = v_svd[1:n_perp_dim]

        v_varimax = R_s1' * v_reduced
        v_varimax_norm = v_varimax / norm(v_varimax)

        println("\nInvariant Direction $col:")
        println("  Varimax coords (normalized): ", round.(v_varimax_norm, digits=4))

        if n_perp_dim == 2
            sum_like = abs(v_varimax_norm[1] - v_varimax_norm[2]) < 0.1
            diff_like = abs(v_varimax_norm[1] + v_varimax_norm[2]) < 0.1

            if sum_like
                println("  → Combination: θ¹[1] + θ¹[2]")
                println("  → Meaning: Can change n₁p₁ and n₂p₂ equally")
                println("              without affecting output")
            elseif diff_like
                println("  → Combination: θ¹[1] - θ¹[2]")
                println("  → In parameters: ratio n₁p₁ / n₂p₂")
                println("  → Meaning: Ratio is non-identifiable")
            end
        end
    end
end

println("\n" * "="^60)
println("SUMMARY")
println("="^60)
println("Stage 1 (f=log):")
println("  → Found ", size(N_perp_s1, 2), "D span of [n₁p₁, n₂p₂]")
println("\nStage 2 (f=identity):")
println("  → Raw basis: ", size(N_perp_s2, 2), "D")
println("  → After Jacobian filter: ", length(active_cols), "D active")
println("\nPost-processing (Varimax):")
println("  → Identified which combination(s) are identifiable")
println("\nKey Point:")
println("  Dictionary (Varimax) is ONLY for interpretation")
println("  Algorithm works entirely with SVD bases")
