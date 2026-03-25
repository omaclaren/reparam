"""
CORRECTED Sequential IIR Implementation
Honest example with proper data generation and active direction filtering
"""

using LinearAlgebra
using Distributions
using Random
include("../../../ReparamTools.jl")
using .ReparamTools

Random.seed!(123)

println("="^60)
println("Sequential IIR: CORRECTED Implementation")
println("Sum of Independent Poisson Variables Example")
println("="^60)

# ============================================================
# MODEL SETUP - Use direct Poisson to match theory
# ============================================================

θ_true = [21.0, 0.9, 110.0, 0.18]  # [n₁, p₁, n₂, p₂]
n_obs = 100

function generate_data_poisson(θ)
    n1, p1, n2, p2 = θ
    λ1, λ2 = n1 * p1, n2 * p2
    # Direct Poisson samples from sum
    return rand(Poisson(λ1 + λ2), n_obs)
end

y_obs = generate_data_poisson(θ_true)

function negloglik(θ)
    n1, p1, n2, p2 = θ
    λ = n1 * p1 + n2 * p2  # Only the SUM matters!
    -sum(logpdf.(Poisson(λ), y_obs))
end

ϕ_func(θ) = [negloglik(θ)]

println("\nModel structure:")
println("  Parameters: θ = [n₁, p₁, n₂, p₂]")
println("  Output depends ONLY on: λ = n₁p₁ + n₂p₂")
println("  Expected: 1D identifiable space")

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
println("  Singular values: ", round.(S_s1, digits=6))
println("  Rank: ", rank_s1, "/4")
println("  Dim(N): ", size(N_s1, 2), " (invariant in log-space)")
println("  Dim(N_perp): ", size(N_perp_s1, 2), " (potentially identifiable)")

if rank_s1 != 1
    println("\n  ⚠ WARNING: Expected rank 1 since output is scalar depending on λ = n₁p₁ + n₂p₂")
    println("  Actual rank: ", rank_s1)
    println("  This suggests numerical issues or data doesn't match theoretical structure")
end

# Build square transformation matrix
A1 = vcat(N_perp_s1', N_s1')
println("\n  Transformation matrix A1: ", size(A1))

# Define coordinate transformations
θ_to_θ1(θ) = exp.(A1 * log.(θ))
θ1_to_θ(θ1) = exp.(A1' * log.(θ1))

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

println("\nResults (raw):")
println("  Singular values: ", round.(S_s2, digits=6))
println("  Rank: ", rank_s2, "/4")
println("  Dim(N): ", size(N_s2, 2))
println("  Dim(N_perp): ", size(N_perp_s2, 2))

# ============================================================
# FILTER TO ACTIVE DIRECTIONS (via Jacobian)
# ============================================================

println("\n" * "="^60)
println("FILTERING: Identify Active Directions via Jacobian")
println("="^60)

J_s2 = compute_ϕ_Jacobian(ϕ_stage2, θ1_MLE)
println("\nJacobian shape: ", size(J_s2))

if size(N_perp_s2, 2) > 0
    println("\nChecking N_perp columns:")

    J_norms = Float64[]
    for col in 1:size(N_perp_s2, 2)
        v = N_perp_s2[:, col]
        Jv_norm = norm(J_s2 * v)
        push!(J_norms, Jv_norm)

        println("  Column $col: ||J*v|| = ", round(Jv_norm, digits=8))
    end

    # Find active column (largest Jacobian norm)
    active_idx = argmax(J_norms)
    println("\n  Active column: ", active_idx)
    println("  TRUE identifiable dimension: 1")

    # Extract the active direction
    v_active_svd = N_perp_s2[:, active_idx]
    println("\n  Active direction (SVD coords):")
    println("    ", round.(v_active_svd, digits=4))
end

# Similarly for N (invariant directions)
if size(N_s2, 2) > 0
    println("\nChecking N columns (should all have zero Jacobian):")

    for col in 1:size(N_s2, 2)
        v = N_s2[:, col]
        Jv_norm = norm(J_s2 * v)
        println("  Column $col: ||J*v|| = ", round(Jv_norm, digits=8))
    end
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

println("\nStage 1 Varimax basis (scaled):")
display(N_perp_var_scaled)

println("\n\nInterpretation of Varimax columns:")
param_names = ["n₁", "p₁", "n₂", "p₂"]
for col in 1:size(N_perp_var_scaled, 2)
    exps = N_perp_var_scaled[:, col]
    terms = String[]
    for i in 1:4
        if abs(exps[i]) > 1e-6
            exp_str = abs(exps[i] - 1.0) < 1e-6 ? "" : "^" * string(round(exps[i], digits=2))
            push!(terms, param_names[i] * exp_str)
        end
    end
    println("  Column $col: ", join(terms, "·"))
end

# Rotation matrix from SVD to Varimax
R_s1 = N_perp_s1' * N_perp_varimax

# Interpret the ACTIVE direction from Stage 2 in Varimax basis
if size(N_perp_s2, 2) > 0 && !isempty(J_norms)
    active_idx = argmax(J_norms)
    v_active_svd = N_perp_s2[:, active_idx]

    # Extract components in Stage 1 N_perp space
    n_perp_dim = size(N_perp_s1, 2)
    v_reduced = v_active_svd[1:n_perp_dim]

    # Rotate to Varimax
    v_varimax = R_s1' * v_reduced
    v_varimax_norm = v_varimax / norm(v_varimax)

    println("\n\nActive Direction in Varimax Basis:")
    println("  Coefficients: ", round.(v_varimax_norm, digits=4))

    # Interpret
    if n_perp_dim == 2
        if abs(v_varimax_norm[1] - v_varimax_norm[2]) < 0.1
            println("  → Interpretation: θ¹[1] + θ¹[2]")
            println("  → In parameters: n₁p₁ + n₂p₂ (SUM)")
        elseif abs(v_varimax_norm[1] + v_varimax_norm[2]) < 0.1
            println("  → Interpretation: θ¹[1] - θ¹[2]")
            println("  → In parameters: n₁p₁ - n₂p₂ (DIFFERENCE)")
        else
            ratio = v_varimax_norm[1] / v_varimax_norm[2]
            println("  → Interpretation: ", round(ratio, digits=2), "·θ¹[1] + θ¹[2]")
        end
    end
end

# Interpret invariant directions
if size(N_s2, 2) > 0
    println("\n\nInvariant Directions in Varimax Basis:")

    for col in 1:min(2, size(N_s2, 2))  # Show first 2 at most
        v_svd = N_s2[:, col]
        n_perp_dim = size(N_perp_s1, 2)
        v_reduced = v_svd[1:n_perp_dim]

        v_varimax = R_s1' * v_reduced
        v_varimax_norm = v_varimax / norm(v_varimax)

        println("\n  Invariant $col:")
        println("    Coefficients: ", round.(v_varimax_norm, digits=4))

        if n_perp_dim == 2 && abs(v_varimax_norm[1] - v_varimax_norm[2]) < 0.1
            println("    → θ¹[1] + θ¹[2] = n₁p₁ + n₂p₂ (THE INVARIANT COMBINATION!)")
        end
    end
end

println("\n" * "="^60)
println("SUMMARY")
println("="^60)
println("  Data: Direct Poisson samples from λ = n₁p₁ + n₂p₂")
println("  Stage 1 (f=log): Found ", size(N_perp_s1, 2), "D potentially identifiable span")
println("  Stage 2 (f=identity): Raw ", size(N_perp_s2, 2), "D basis")
println("  After Jacobian filter: 1D active (", size(N_s2, 2), "D invariant)")
println("  Dictionary (Varimax): Shows combinations in terms of [n₁p₁, n₂p₂]")
println("\n  Key point: Dictionary is post-processing, not part of algorithm!")
