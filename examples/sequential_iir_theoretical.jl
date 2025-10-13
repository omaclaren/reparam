"""
Sequential IIR: Theoretical Clean Example
Uses exact mean (no sampling noise) to show clean structure
Dictionary as post-processing only
"""

using LinearAlgebra
using Distributions
using Random
using Printf
include("../ReparamTools.jl")
using .ReparamTools

println("="^60)
println("Sequential IIR: Theoretical Clean Example")
println("Sum of Independent Poisson Limits (Exact Mean)")
println("="^60)

# ============================================================
# MODEL SETUP - Exact mean (no sampling noise)
# ============================================================

θ_true = [21.0, 0.9, 110.0, 0.18]  # [n₁, p₁, n₂, p₂]

# Auxiliary mapping: ϕ(θ) = [μ, σ²] where μ = σ² = n₁p₁ + n₂p₂
ϕ_θ(θ) = [θ[1]*θ[2] + θ[3]*θ[4], θ[1]*θ[2] + θ[3]*θ[4]]

λ_true = θ_true[1] * θ_true[2] + θ_true[3] * θ_true[4]

println("\nModel:")
println("  Parameters: θ = [n₁, p₁, n₂, p₂]")
println("  Auxiliary: ϕ(θ) = [λ, λ] where λ = n₁p₁ + n₂p₂")
println("  Data: Exact mean (no sampling noise)")
println("  True λ = ", λ_true)

# Use exact mean - no sampling noise
N_samples = 10
data = fill(λ_true, N_samples)

println("\nUsing exact mean (", N_samples, " observations)")
println("  All observations = ", λ_true)

# Likelihood (will be minimized when ϕ(θ) = λ_true)
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
println("  Singular values: ", round.(S_s1[S_s1 .> 1e-10], digits=6))
println("  Rank: ", rank_s1, "/4")
println("  Dim(N_perp): ", size(N_perp_s1, 2), " (potentially identifiable)")
println("  Dim(N): ", size(N_s1, 2), " (invariant in log-space)")

println("\n  N_perp (SVD basis, scaled):")
N_perp_s1_scaled = scale_and_round(N_perp_s1; column_scales=ones(size(N_perp_s1, 2)))
display(N_perp_s1_scaled)

# Build square transformation
A1 = vcat(N_perp_s1', N_s1')
θ_to_θ1(θ) = exp.(A1 * log.(θ))
θ1_to_θ(θ1) = exp.(A1' * log.(θ1))

θ1_MLE = θ_to_θ1(θ_true)
println("\n  Stage 1 MLE coordinates:")
for i in 1:4
    println("    θ¹[$i] = ", round(θ1_MLE[i], digits=4))
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
println("  Singular values: ", round.(S_s2[S_s2 .> 1e-10], digits=6))
println("  Rank: ", rank_s2, "/4")
println("  Dim(N_perp): ", size(N_perp_s2, 2))
println("  Dim(N): ", size(N_s2, 2))

# ============================================================
# FILTER: Identify active directions via Jacobian
# ============================================================

println("\n" * "="^60)
println("FILTERING: Jacobian Projection")
println("="^60)

J_s2 = compute_ϕ_Jacobian(ϕ_stage2, θ1_MLE)
println("\nJacobian shape: ", size(J_s2))

println("\nTesting N_perp columns:")
active_cols = Int[]

for col in 1:size(N_perp_s2, 2)
    v = N_perp_s2[:, col]
    Jv_norm = norm(J_s2 * v)
    is_active = Jv_norm > 1e-6

    println("  Column $col: ||J*v|| = ", @sprintf("%.2e", Jv_norm),
            is_active ? " ✓ ACTIVE" : " ✗ null")

    if is_active
        push!(active_cols, col)
    end
end

println("\n✓ Active columns: ", active_cols)
println("✓ True identifiable dimension: ", length(active_cols))

# ============================================================
# POST-PROCESSING: Varimax Dictionary
# ============================================================

println("\n" * "="^60)
println("POST-PROCESSING: Varimax for Interpretation")
println("="^60)

# Compute Varimax rotation of Stage 1 basis
N_perp_varimax = varimax_rotation(N_perp_s1; n_restarts=200, threshold=1e-2)
N_perp_var_scaled = scale_and_round(N_perp_varimax; column_scales=ones(size(N_perp_varimax, 2)))

println("\nStage 1 Varimax basis:")
display(N_perp_var_scaled)

param_names = ["n₁", "p₁", "n₂", "p₂"]
println("\n")
for col in 1:size(N_perp_var_scaled, 2)
    exps = N_perp_var_scaled[:, col]
    terms = String[]
    for i in 1:4
        if abs(exps[i]) > 1e-6
            exp_str = abs(exps[i] - 1.0) < 1e-6 ? "" : "^" * string(round(exps[i], digits=2))
            push!(terms, param_names[i] * exp_str)
        end
    end
    println("  θ¹[$col] = ", join(terms, "·"))
end

R_s1 = N_perp_s1' * N_perp_varimax

# Interpret active direction
if !isempty(active_cols)
    println("\n" * "="^60)
    println("IDENTIFIABLE COMBINATION")
    println("="^60)

    col = active_cols[1]
    v_svd = N_perp_s2[:, col]
    n_perp_dim = size(N_perp_s1, 2)
    v_reduced = v_svd[1:n_perp_dim]

    v_varimax = R_s1' * v_reduced
    v_varimax_norm = v_varimax / norm(v_varimax)

    println("\nActive direction (after Jacobian filter):")
    println("  Varimax coefficients: ", round.(v_varimax_norm, digits=4))

    # Clean interpretation
    if n_perp_dim == 2
        if abs(v_varimax_norm[1] - v_varimax_norm[2]) < 0.1
            println("\n  ✓ Combination: θ¹[1] + θ¹[2]")
            println("  ✓ In parameters: n₁p₁ + n₂p₂")
            println("  ✓ Meaning: SUM is identifiable")
        elseif abs(v_varimax_norm[1] + v_varimax_norm[2]) < 0.1
            println("\n  ✓ Combination: θ¹[1] - θ¹[2]")
            println("  ✓ In log-space: log(n₁p₁) - log(n₂p₂)")
            println("  ✓ Meaning: RATIO n₁p₁/n₂p₂ is identifiable")
        end
    end
end

# Interpret invariant direction
if size(N_s2, 2) > 0
    println("\n" * "="^60)
    println("INVARIANT COMBINATION (Non-identifiable)")
    println("="^60)

    # Take first invariant direction
    v_svd = N_s2[:, 1]
    n_perp_dim = size(N_perp_s1, 2)
    v_reduced = v_svd[1:n_perp_dim]

    v_varimax = R_s1' * v_reduced
    v_varimax_norm = v_varimax / norm(v_varimax)

    println("\nInvariant direction:")
    println("  Varimax coefficients: ", round.(v_varimax_norm, digits=4))

    if n_perp_dim == 2
        if abs(v_varimax_norm[1] - v_varimax_norm[2]) < 0.1
            println("\n  ✓ Combination: θ¹[1] + θ¹[2]")
            println("  ✓ In parameters: n₁p₁ + n₂p₂")
            println("  ✓ Meaning: SUM is NON-IDENTIFIABLE")
            println("\n  → Can change both n₁p₁ and n₂p₂ by same amount")
            println("    without affecting fit to data")
        elseif abs(v_varimax_norm[1] + v_varimax_norm[2]) < 0.1
            println("\n  ✓ Combination: θ¹[1] - θ¹[2]")
            println("  ✓ Meaning: RATIO is NON-IDENTIFIABLE")
        end
    end
end

println("\n" * "="^60)
println("CLEAN SEQUENTIAL IIR WORKFLOW")
println("="^60)
println("""
1. STAGE 1 (f=log):
   • Run find_invariant_subspace in log-space
   • Build square transformation A₁ = [N_perp'; N']
   • Creates coordinates θ¹ = exp(A₁ log θ)
   • Uses pure SVD basis (no Varimax yet)

2. STAGE 2 (f=identity):
   • Run find_invariant_subspace on θ¹ → ϕ(θ(θ¹))
   • Returns potentially identifiable span N_perp
   • Uses pure SVD basis (no Varimax yet)

3. FILTER (Jacobian projection):
   • Compute J = ∇_θ¹ ϕ at MLE
   • Test each column of N_perp: is ||J·v|| > 0?
   • Keep only active columns (true identifiable)

4. POST-PROCESS (Varimax dictionary):
   • Rotate Stage 1 basis to Varimax
   • Express active directions in Varimax coordinates
   • Human-readable combinations like "n₁p₁ + n₂p₂"

KEY POINT: Dictionary is ONLY for interpretation!
""")
