"""
Clean sequential IIR with post-processing dictionary
4D → 2D → 1D with Varimax rotation as post-processing only
"""

using LinearAlgebra
using Distributions
using Random
include("../ReparamTools.jl")
using .ReparamTools

Random.seed!(123)

# Sum of two independent Poisson limit models
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

println("="^60)
println("CLEAN SEQUENTIAL IIR: 4D → 2D → 1D")
println("="^60)

# ============================================================
# STAGE 1: f=log, work in SVD basis
# ============================================================
println("\nSTAGE 1 (f=log, SVD basis):")

ϕ_log = θ_log -> ϕ_func(exp.(θ_log))
S_s1, N_s1, N_perp_s1, rank_s1 = find_invariant_subspace(
    ϕ_log, log.(θ_true); rtolJ=sqrt(eps()), atolM=1e-10
)

println("  Rank: ", rank_s1, "/4")
println("  Dim(N): ", size(N_s1, 2))
println("  Dim(N_perp): ", size(N_perp_s1, 2))

println("\n  N_perp (SVD basis, scaled for display):")
N_perp_s1_scaled = scale_and_round(N_perp_s1; column_scales=ones(size(N_perp_s1, 2)))
display(N_perp_s1_scaled)

# ============================================================
# COMPUTE DICTIONARY (once, for later interpretation)
# ============================================================
println("\n\nCOMPUTE DICTIONARY (Varimax rotation):")

N_perp_varimax = varimax_rotation(N_perp_s1; n_restarts=200, threshold=1e-2)
N_perp_var_scaled = scale_and_round(N_perp_varimax; column_scales=ones(size(N_perp_varimax, 2)))

println("  Varimax-rotated N_perp (scaled):")
display(N_perp_var_scaled)

# Rotation matrix from SVD to Varimax basis
R = N_perp_s1' * N_perp_varimax
println("\n  Rotation R = N_perp_svd' * N_perp_varimax:")
println("  R shape: ", size(R))

# ============================================================
# STAGE 2: Work in 2D SVD coordinates (NOT full 4D!)
# ============================================================
println("\n\nSTAGE 2 (f=identity, 2D SVD coordinates):")

# Map to 2D: y = N_perp' * log(θ)
function log_to_y(θ_log)
    return N_perp_s1' * θ_log
end

function y_to_log(y)
    # This is the key: need to reconstruct log(θ) from 2D y
    # We're in the reduced space, so need to pick a point in the fiber
    # Use MLE as reference point
    θ_log_ref = log.(θ_true)
    return θ_log_ref + N_perp_s1 * (y - log_to_y(θ_log_ref))
end

# Stage 2 mapping: y → ϕ
ϕ_stage2(y) = ϕ_func(exp.(y_to_log(y)))

y_true = log_to_y(log.(θ_true))
println("  y_true (2D SVD coords): ", round.(y_true, digits=4))

S_s2, N_s2, N_perp_s2, rank_s2 = find_invariant_subspace(
    ϕ_stage2, y_true; rtolJ=sqrt(eps()), atolM=1e-10
)

println("  Rank: ", rank_s2, "/2")
println("  Dim(N): ", size(N_s2, 2))
println("  Dim(N_perp): ", size(N_perp_s2, 2))

if size(N_perp_s2, 2) > 0
    println("\n  N_perp_s2 (Stage 2 basis):")
    display(N_perp_s2)

    # Check which directions affect output
    J_s2 = compute_ϕ_Jacobian(ϕ_stage2, y_true)
    J_proj = J_s2 * N_perp_s2

    U_proj, S_proj, V_proj = svd(J_proj)
    println("\n  Jacobian projection singular values:")
    for (i, s) in enumerate(S_proj)
        println("    [$i]: ", round(s, digits=10))
    end

    rank_proj = count(S_proj .> sqrt(eps()) * maximum(S_proj))
    println("  Rank(J * N_perp): ", rank_proj)

    # Keep only the principal direction
    println("\n  Taking principal direction (largest singular value):")
    v_principal_2D = N_perp_s2[:, 1]
    println("  v_principal (in 2D SVD coords): ", round.(v_principal_2D, digits=4))

    # ============================================================
    # POST-PROCESS: Rotate to dictionary basis
    # ============================================================
    println("\n\nPOST-PROCESS (rotate to dictionary):")

    # The principal direction in 2D SVD coords corresponds to a direction in 4D log-space
    v_principal_4D = N_perp_s1 * v_principal_2D
    println("  v_principal (in 4D log-space): ", round.(v_principal_4D, digits=4))

    # Now rotate to Varimax basis for interpretation
    # We want: coefficient in Varimax basis = R' * (coefficient in SVD basis)
    v_varimax = R' * v_principal_2D
    println("  v_principal (in Varimax 2D coords): ", round.(v_varimax, digits=4))

    # The combination is in the 2D Varimax space, not 4D!
    # v_varimax is the coefficient vector [a, b] where combination = a*θ¹ + b*θ²
    # where θ¹ = n₁p₁ and θ² = n₂p₂ (the Varimax columns)

    println("\n  Final identifiable combination:")
    println("  Coefficients in Varimax 2D basis: ", round.(v_varimax, digits=4))

    # Interpret: which Varimax combinations are in the sum?
    println("\n  Interpretation (Varimax basis = [n₁p₁, n₂p₂]):")
    if abs(v_varimax[1]) > 1e-6
        println("    Coefficient on n₁p₁: ", round(v_varimax[1], digits=4))
    end
    if abs(v_varimax[2]) > 1e-6
        println("    Coefficient on n₂p₂: ", round(v_varimax[2], digits=4))
    end

    # To get 4D representation: map back through N_perp_varimax
    # But normalize first
    v_varimax_norm = v_varimax / norm(v_varimax)
    v_4D_from_varimax = N_perp_varimax * v_varimax_norm
    v_4D_scaled = scale_and_round(v_4D_from_varimax; column_scales=[1.0])

    println("\n  4D log-space exponents (scaled):")
    param_names = ["n₁", "p₁", "n₂", "p₂"]
    for i in 1:4
        coef = v_4D_scaled[i]
        if abs(coef) > 1e-6
            println("    $(param_names[i]): ", round(coef, digits=4))
        end
    end
end

println("\n" * "="^60)
println("SUMMARY")
println("="^60)
println("  Sequential reduction: 4D → 2D → 1D")
println("  Dictionary (Varimax) computed once at Stage 1")
println("  Stage 2 works in 2D SVD coordinates")
println("  Final result rotated to dictionary for interpretation")
