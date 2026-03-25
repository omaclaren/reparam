using LinearAlgebra

# True parameters
θ_true = [2.0, 1.5, 0.2, 0.1, 0.3, 0.25, 1.0, 3.0]

# A1_full from previous run (from output)
A1_full = [
    0.0        0.0        0.0          -1.0           0.0           0.0           0.0        0.0;
   -0.166666  -0.833334   0.0           0.0           0.0           0.0          -0.166668  -0.499999;
    0.0        0.0        0.0           0.0           0.0          -1.0           0.0        0.0;
    0.0        0.0        0.0           0.0           1.0           0.0           0.0        0.0;
    0.833334   0.166666   0.0           0.0           0.0           0.0          -0.166668  -0.499999;
   -0.166666   0.166666   0.0           0.0           0.0           0.0           0.833333  -0.500001;
    0.0        0.0        1.0           0.0           0.0           0.0           0.0        0.0;
    0.5       -0.5       -0.0           -0.0          -0.0           0.0           0.5        0.5
]

A1_inv = inv(A1_full)

# Transform to Stage 1
θ1_true = exp.(A1_full * log.(θ_true))

println("Stage 1 coordinates:")
println(θ1_true)

# Reconstruct original
θ_recovered = exp.(A1_inv * log.(θ1_true))

println("\nRecovered parameters:")
println(θ_recovered)
println("\nError:")
println(θ_recovered - θ_true)

# Meshkat's combinations
b₁, c₁, k₀₁, k₀₂, k₁₂, k₂₁, V_M, K_M = θ_recovered

q₁ = b₁ * c₁
q₂ = c₁ * K_M
q₃ = k₀₂ + k₁₂
q₄ = c₁ * V_M * k₁₂ * k₂₁
q₅ = c₁ * V_M * (k₀₁ + k₂₁)

println("\nMeshkat's combinations from recovered θ:")
println("q₁ = ", q₁)
println("q₂ = ", q₂)
println("q₃ = ", q₃)
println("q₄ = ", q₄)
println("q₅ = ", q₅)

# True values
q₁_true = 3.0
q₂_true = 4.5
q₃_true = 0.4
q₄_true = 0.1125
q₅_true = 0.675

println("\nTrue values:")
println("q₁ = ", q₁_true, " error: ", abs(q₁ - q₁_true))
println("q₂ = ", q₂_true, " error: ", abs(q₂ - q₂_true))
println("q₃ = ", q₃_true, " error: ", abs(q₃ - q₃_true))
println("q₄ = ", q₄_true, " error: ", abs(q₄ - q₄_true))
println("q₅ = ", q₅_true, " error: ", abs(q₅ - q₅_true))
