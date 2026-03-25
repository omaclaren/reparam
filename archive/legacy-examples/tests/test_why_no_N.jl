"""
Why is Dim(N) = 0 at Stage 2?
"""

using LinearAlgebra
using Distributions
using Random
include("../../../ReparamTools.jl")
using .ReparamTools

Random.seed!(123)

θ_true = [21.0, 0.9, 110.0, 0.18]
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

# Stage 1
ϕ_log = θ_log -> [negloglik(exp.(θ_log))]
S_s1, N_s1, N_perp_s1, rank_s1 = find_invariant_subspace(
    ϕ_log, log.(θ_true); rtolJ=sqrt(eps()), atolM=1e-10
)

# Stage 2: simplified transformation
θ_to_y(θ) = N_perp_s1' * log.(θ)
y_to_θ(y) = exp.(N_perp_s1 * y)

ϕ_stage2(y) = [negloglik(y_to_θ(y))]
y_true = θ_to_y(θ_true)

println("Stage 2 setup:")
println("  y_true: ", y_true)

# The issue: y_to_θ(y) only gives part of θ!
# It maps 2D y → 4D log-space, but only spans 2D subspace

println("\nTest: does y_to_θ give valid parameters?")
θ_reconstructed = y_to_θ(y_true)
println("  Original θ: ", θ_true)
println("  Reconstructed: ", θ_reconstructed)

println("\nThe problem: we're missing the N_s1 component!")
println("  Full reconstruction needs: θ = exp(N_perp*y + N*z)")
println("  But we only have: θ = exp(N_perp*y)")
println("  This only works if we fix z (the invariant coords)")

# The CORRECT way: use full reconstruction
println("\n" * "="^60)
println("CORRECTED Stage 2")
println("="^60)

# Store reference point for the invariant directions
θ_log_ref = log.(θ_true)
z_ref = N_s1' * θ_log_ref  # Invariant component at reference

function y_to_θ_correct(y)
    # Reconstruct: log(θ) = N_perp*y + N*z_ref
    θ_log = N_perp_s1 * y + N_s1 * z_ref
    return exp.(θ_log)
end

println("\nTest corrected reconstruction:")
θ_recon_correct = y_to_θ_correct(y_true)
println("  Original θ: ", θ_true)
println("  Reconstructed: ", round.(θ_recon_correct, digits=6))
println("  Match: ", isapprox(θ_true, θ_recon_correct, rtol=1e-10))

# Now run Stage 2 with correct mapping
ϕ_stage2_correct(y) = [negloglik(y_to_θ_correct(y))]

S_s2, N_s2, N_perp_s2, rank_s2 = find_invariant_subspace(
    ϕ_stage2_correct, y_true; rtolJ=sqrt(eps()), atolM=1e-10
)

println("\nStage 2 results (corrected):")
println("  Rank: ", rank_s2, "/2")
println("  Dim(N): ", size(N_s2, 2))
println("  Dim(N_perp): ", size(N_perp_s2, 2))

if size(N_s2, 2) > 0
    println("\n  N_s2 (invariant directions in 2D):")
    display(N_s2)

    # Interpret in Varimax basis
    N_perp_varimax = varimax_rotation(N_perp_s1; n_restarts=200, threshold=1e-2)
    R = N_perp_s1' * N_perp_varimax

    for col in 1:size(N_s2, 2)
        v_svd = N_s2[:, col]
        v_var = R' * v_svd
        v_var_norm = v_var / norm(v_var)

        println("\n  Invariant direction $col:")
        println("    Varimax coords: ", round.(v_var_norm, digits=4))

        if abs(v_var_norm[1] - v_var_norm[2]) < 0.1
            println("    → SUM: n₁p₁ + n₂p₂")
        end
    end
end
