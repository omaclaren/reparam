"""
Debug: what is the Hessian test seeing at Stage 2?
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

# Stage 2 with correct reconstruction
θ_log_ref = log.(θ_true)
z_ref = N_s1' * θ_log_ref

function y_to_θ(y)
    θ_log = N_perp_s1 * y + N_s1 * z_ref
    return exp.(θ_log)
end

ϕ_stage2(y) = [negloglik(y_to_θ(y))]
y_true = N_perp_s1' * log.(θ_true)

# Manual Hessian test on the [1,1] direction (sum)
println("Manual Hessian test on sum direction [1,1]:")

v_sum = [1.0, 1.0] / sqrt(2)  # Normalized
println("  Test direction: ", v_sum)

# Perturbation test
ε = 1e-5
num_samples = 5

println("\n  Testing invariance by perturbing along v:")
for i in 1:num_samples
    # Perturb in direction v
    y_pert = y_true + ε * randn() * v_sum

    # Evaluate at perturbed point
    ϕ_pert = ϕ_stage2(y_pert)
    ϕ_true = ϕ_stage2(y_true)

    println("    Δϕ = ", abs(ϕ_pert[1] - ϕ_true[1]))
end

# The key question: is the mapping y → ϕ(y) actually invariant along [1,1]?
println("\n  Testing by moving along sum direction:")
for α in [-0.1, -0.01, 0.0, 0.01, 0.1]
    y_test = y_true + α * v_sum
    ϕ_test = ϕ_stage2(y_test)[1]

    println("    α = ", α, " → ϕ = ", round(ϕ_test, digits=6))
end

# Compare with difference direction [1,-1]
println("\n\nTesting difference direction [1,-1]:")
v_diff = [1.0, -1.0] / sqrt(2)

for α in [-0.1, -0.01, 0.0, 0.01, 0.1]
    y_test = y_true + α * v_diff
    ϕ_test = ϕ_stage2(y_test)[1]

    println("    α = ", α, " → ϕ = ", round(ϕ_test, digits=6))
end

println("\n" * "="^60)
println("EXPECTED:")
println("  Sum [1,1]: ϕ should NOT change (invariant)")
println("  Difference [1,-1]: ϕ should change (identifiable)")
println("="^60)
