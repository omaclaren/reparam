"""
Test: verify data structure matches theory
"""

using LinearAlgebra
using Distributions
using Random
include("../ReparamTools.jl")
using .ReparamTools

Random.seed!(123)

θ_true = [21.0, 0.9, 110.0, 0.18]
n_obs = 100

# Generate Poisson data
λ_true = θ_true[1] * θ_true[2] + θ_true[3] * θ_true[4]
println("True λ = n₁p₁ + n₂p₂ = ", λ_true)

y_obs = rand(Poisson(λ_true), n_obs)
println("Data: ", n_obs, " samples from Poisson(", λ_true, ")")
println("Sample mean: ", mean(y_obs))

# Define likelihood
function negloglik(θ)
    n1, p1, n2, p2 = θ
    λ = n1 * p1 + n2 * p2
    -sum(logpdf.(Poisson(λ), y_obs))
end

# Test: does likelihood only depend on sum?
println("\nTest 1: Different (n₁,p₁) with same n₁p₁")
θ_test1 = [10.5, 1.8, 110.0, 0.18]  # n₁p₁ = 18.9 (same)
θ_test2 = [63.0, 0.3, 110.0, 0.18]  # n₁p₁ = 18.9 (same)

println("  θ₁: n₁p₁ = ", θ_test1[1] * θ_test1[2])
println("  θ₂: n₁p₁ = ", θ_test2[1] * θ_test2[2])
println("  NLL(θ₁) = ", round(negloglik(θ_test1), digits=6))
println("  NLL(θ₂) = ", round(negloglik(θ_test2), digits=6))
println("  Difference: ", abs(negloglik(θ_test1) - negloglik(θ_test2)))

println("\nTest 2: Different sums")
θ_test3 = [21.0, 0.9, 110.0, 0.19]  # Slightly different sum
println("  λ(true) = ", θ_true[1]*θ_true[2] + θ_true[3]*θ_true[4])
println("  λ(test) = ", θ_test3[1]*θ_test3[2] + θ_test3[3]*θ_test3[4])
println("  NLL(true) = ", round(negloglik(θ_true), digits=6))
println("  NLL(test) = ", round(negloglik(θ_test3), digits=6))
println("  Difference: ", abs(negloglik(θ_true) - negloglik(θ_test3)))

# Compute gradient
println("\nGradient at θ_true:")
ε = 1e-6
grad = zeros(4)
nll_0 = negloglik(θ_true)
for i in 1:4
    θ_pert = copy(θ_true)
    θ_pert[i] += ε
    grad[i] = (negloglik(θ_pert) - nll_0) / ε
end

println("  ∇NLL = ", round.(grad, digits=6))
println("  Direction [p₁, n₁, p₂, n₂]·∇NLL:")
println("    [", round(θ_true[2], digits=4), ", ",
           round(θ_true[1], digits=4), ", ",
           round(θ_true[4], digits=4), ", ",
           round(θ_true[3], digits=4), "]")

# The gradient should be proportional to [p₁, n₁, p₂, n₂]
# because ∂λ/∂n₁ = p₁, ∂λ/∂p₁ = n₁, etc.
expected_grad = [θ_true[2], θ_true[1], θ_true[4], θ_true[3]] * grad[1] / θ_true[2]
println("\n  Expected (if only sum matters): ", round.(expected_grad, digits=6))
println("  Actual: ", round.(grad, digits=6))
println("  Match: ", isapprox(grad, expected_grad, rtol=1e-3))
