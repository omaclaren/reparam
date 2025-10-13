"""
Generate realistic data for sum-of-Poisson example
Save to use in sequential_iir_final.jl
"""

using Distributions
using Random

Random.seed!(42)

# True parameters
θ_true = [21.0, 0.9, 110.0, 0.18]  # [n₁, p₁, n₂, p₂]
λ_true = θ_true[1] * θ_true[2] + θ_true[3] * θ_true[4]

println("True parameters: n₁=", θ_true[1], ", p₁=", θ_true[2],
        ", n₂=", θ_true[3], ", p₂=", θ_true[4])
println("True λ = n₁p₁ + n₂p₂ = ", λ_true)

# Generate data from Normal approximation
N_samples = 10
data = rand(Normal(λ_true, sqrt(λ_true)), N_samples)

println("\nGenerated ", N_samples, " samples:")
println("data = ", round.(data, digits=1))

println("\nSample statistics:")
println("  Mean: ", round(mean(data), digits=2))
println("  Variance: ", round(var(data), digits=2))
println("  Expected mean: ", λ_true)
println("  Expected variance: ", λ_true)
