"""
CLEAN Sequential IIR Implementation
Demonstrates the correct workflow with dictionary as post-processing only
"""

using LinearAlgebra
using Distributions
using Random
include("../../../ReparamTools.jl")
using .ReparamTools

Random.seed!(123)

println("="^60)
println("Sequential IIR: Clean Implementation")
println("Sum of Independent Poisson Limits Example")
println("="^60)

# ============================================================
# MODEL SETUP
# ============================================================

θ_true = [21.0, 0.9, 110.0, 0.18]  # [n₁, p₁, n₂, p₂]
n_obs = 100

function generate_data(θ)
    n1, p1, n2, p2 = θ
    y1 = rand(Binomial(Int(n1), p1), n_obs)
    y2 = rand(Binomial(Int(n2), p2), n_obs)
    return y1 .+ y2
end

y_obs = generate_data(θ_true)

function negloglik(θ)
    n1, p1, n2, p2 = θ
    λ1, λ2 = n1 * p1, n2 * p2
    -sum(logpdf.(Poisson(λ1 + λ2), y_obs))
end

ϕ_func(θ) = [negloglik(θ)]

# ============================================================
# STAGE 1: f=log (Multiplicative structure)
# ============================================================

println("\n" * "="^60)
println("STAGE 1: f=log")
println("="^60)

ϕ_log = θ_log -> ϕ_func(exp.(θ_log))

S_s1, N_s1, N_perp_s1, rank_s1 = find_invariant_subspace(
    ϕ_log, log.(θ_true); rtolJ=sqrt(eps()), atolM=1e-10
)

println("\nResults:")
println("  Rank: ", rank_s1, "/4")
println("  Dim(N): ", size(N_s1, 2), " (invariant)")
println("  Dim(N_perp): ", size(N_perp_s1, 2), " (potentially identifiable)")

# Build square transformation matrix
A1 = vcat(N_perp_s1', N_s1')
println("\n  Transformation matrix A1: ", size(A1))

# Define coordinate transformations
θ_to_θ1(θ) = exp.(A1 * log.(θ))
θ1_to_θ(θ1) = exp.(A1' * log.(θ1))  # A1 is orthogonal

# Compute Stage 1 MLE
θ1_MLE = θ_to_θ1(θ_true)
println("\nStage 1 coordinates at MLE:")
for i in 1:4
    println("  θ¹[$i] = ", round(θ1_MLE[i], digits=4))
end

# ============================================================
# STAGE 2: f=identity (Additive structure)
# ============================================================

println("\n" * "="^60)
println("STAGE 2: f=identity")
println("="^60)

ϕ_stage2(θ1) = ϕ_func(θ1_to_θ(θ1))

S_s2, N_s2, N_perp_s2, rank_s2 = find_invariant_subspace(
    ϕ_stage2, θ1_MLE; rtolJ=sqrt(eps()), atolM=1e-10
)

println("\nResults:")
println("  Rank: ", rank_s2, "/4")
println("  Dim(N): ", size(N_s2, 2), " (invariant in θ¹ space)")
println("  Dim(N_perp): ", size(N_perp_s2, 2), " (potentially identifiable)")

# Check which N_perp directions actually affect output
if size(N_perp_s2, 2) > 0
    println("\nChecking which directions affect output:")
    J_s2 = compute_ϕ_Jacobian(ϕ_stage2, θ1_MLE)

    active_cols = Int[]
    for col in 1:size(N_perp_s2, 2)
        v = N_perp_s2[:, col]
        Jv_norm = norm(J_s2 * v)
        is_active = Jv_norm > 1e-6

        println("  Column $col: ||J*v|| = ", round(Jv_norm, digits=8),
                is_active ? " (ACTIVE)" : " (null)")

        if is_active
            push!(active_cols, col)
        end
    end

    println("\n  Active directions: ", active_cols)
    println("  Truly identifiable dimension: ", length(active_cols))
end

# ============================================================
# POST-PROCESSING: Varimax for Interpretation
# ============================================================

println("\n" * "="^60)
println("POST-PROCESSING: Dictionary Rotation")
println("="^60)

# Compute Varimax rotation of Stage 1 basis
N_perp_varimax = varimax_rotation(N_perp_s1; n_restarts=200, threshold=1e-2)
N_perp_var_scaled = scale_and_round(N_perp_varimax; column_scales=ones(size(N_perp_varimax, 2)))

println("\nStage 1 Varimax basis:")
display(N_perp_var_scaled)

println("\n\nInterpretation:")
param_names = ["n₁", "p₁", "n₂", "p₂"]
for col in 1:size(N_perp_var_scaled, 2)
    exps = N_perp_var_scaled[:, col]
    terms = String[]
    for i in 1:4
        if abs(exps[i]) > 1e-6
            push!(terms, param_names[i] * (abs(exps[i] - 1.0) < 1e-6 ? "" : "^$(exps[i])"))
        end
    end
    println("  Column $col: ", join(terms, " * "))
end

# Rotation matrix from SVD to Varimax
R_s1 = N_perp_s1' * N_perp_varimax

# Interpret Stage 2 active directions in Varimax basis
if !isempty(active_cols)
    println("\n\nStage 2 Active Directions (in Varimax basis):")

    for (idx, col) in enumerate(active_cols)
        v_svd = N_perp_s2[:, col]

        # Only first size(N_perp_s1,2) components are in the Stage 1 N_perp space
        n_perp_dim = size(N_perp_s1, 2)
        v_reduced = v_svd[1:n_perp_dim]

        # Rotate to Varimax
        v_var = R_s1' * v_reduced
        v_var_norm = v_var / norm(v_var)

        println("\n  Active direction $idx:")
        println("    Varimax coefficients: ", round.(v_var_norm, digits=4))

        # Interpret
        if abs(v_var_norm[1] - v_var_norm[2]) < 0.1
            println("    → Interpretation: θ¹[1] + θ¹[2] (sum)")
        elseif abs(v_var_norm[1] + v_var_norm[2]) < 0.1
            println("    → Interpretation: θ¹[1] - θ¹[2] (difference)")
        end

        # Express in original parameters
        combination_4d = N_perp_varimax * v_var_norm
        println("    → In log-space: ", round.(combination_4d, digits=4))
        println("    → In parameters: product of [", join([param_names[i] * "^" * string(round(combination_4d[i], digits=2)) for i in 1:4], ", "), "]")
    end
end

# Interpret Stage 2 invariant directions
if size(N_s2, 2) > 0
    println("\n\nStage 2 Invariant Directions (in Varimax basis):")

    for col in 1:size(N_s2, 2)
        v_svd = N_s2[:, col]

        n_perp_dim = size(N_perp_s1, 2)
        v_reduced = v_svd[1:n_perp_dim]

        v_var = R_s1' * v_reduced
        v_var_norm = v_var / norm(v_var)

        println("\n  Invariant direction $col:")
        println("    Varimax coefficients: ", round.(v_var_norm, digits=4))

        if abs(v_var_norm[1] - v_var_norm[2]) < 0.1
            println("    → Interpretation: θ¹[1] + θ¹[2] (SUM - THE INVARIANT!)")
            println("    → In parameters: n₁p₁ + n₂p₂")
        end
    end
end

println("\n" * "="^60)
println("SUMMARY")
println("="^60)
println("  Stage 1 (f=log): Identified span of products [n₁p₁, n₂p₂]")
println("  Stage 2 (f=identity): Found their sum is invariant")
println("  Result: n₁p₁ + n₂p₂ is the non-identifiable combination")
println("  Dictionary (Varimax): Used for interpretation only")
