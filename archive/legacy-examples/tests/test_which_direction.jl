"""
Check which N_perp_s2 direction gives the sum
"""

using LinearAlgebra
using Distributions
using Random
include("../ReparamTools.jl")
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

N_perp_varimax = varimax_rotation(N_perp_s1; n_restarts=200, threshold=1e-2)
R = N_perp_s1' * N_perp_varimax

# Stage 2 on SVD coords
y_true = N_perp_s1' * log.(θ_true)

function y_to_θ(y)
    θ_log_ref = log.(θ_true)
    θ_log = θ_log_ref + N_perp_s1 * (y - N_perp_s1' * θ_log_ref)
    return exp.(θ_log)
end

ϕ_stage2(y) = [negloglik(y_to_θ(y))]

S_s2, N_s2, N_perp_s2, rank_s2 = find_invariant_subspace(
    ϕ_stage2, y_true; rtolJ=sqrt(eps()), atolM=1e-10
)

println("N_perp_s2 has ", size(N_perp_s2, 2), " columns")
println("N_perp_s2:")
display(N_perp_s2)

# Test: which directions actually affect the output?
J_s2 = compute_ϕ_Jacobian(ϕ_stage2, y_true)

println("\n\nTesting each direction:")
for col in 1:size(N_perp_s2, 2)
    v_svd = N_perp_s2[:, col]
    v_var = R' * v_svd
    v_var_norm = v_var / norm(v_var)

    # Check Jacobian projection
    Jv = J_s2 * v_svd
    mag = norm(Jv)

    println("\nColumn $col:")
    println("  SVD coords: ", round.(v_svd, digits=4))
    println("  Varimax coords: ", round.(v_var_norm, digits=4))
    println("  ||J*v||: ", round(mag, digits=6))

    if abs(v_var_norm[1] - v_var_norm[2]) < 0.1
        println("  → This is approximately [1,1]/√2 (THE SUM!)")
    elseif abs(v_var_norm[1] + v_var_norm[2]) < 0.1
        println("  → This is approximately [1,-1]/√2 (difference)")
    end
end

println("\n" * "="^60)
println("The column with large ||J*v|| and Varimax ~ [0.707, 0.707]")
println("is the sum n₁p₁ + n₂p₂")
println("="^60)
