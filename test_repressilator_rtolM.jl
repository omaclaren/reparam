# Quick test: Does rtolM work for repressilator?

if !@isdefined(ReparamTools)
    include("ReparamTools.jl")
end

using .ReparamTools
using LinearAlgebra
using DifferentialEquations

# Repressilator ODE
function repressilator!(dX, X, θ, t)
    m₁, m₂, m₃, p₁, p₂, p₃ = X
    α₀₁, α₀₂, α₀₃ = θ[1:3]
    α₁, α₂, α₃ = θ[4:6]
    β₁, β₂, β₃ = θ[7:9]
    K₁, K₂, K₃ = θ[10:12]
    k_degm₁, k_degm₂, k_degm₃ = θ[13:15]
    k_degp₁, k_degp₂, k_degp₃ = θ[16:18]
    n = 2.5

    dX[1] = α₀₁ + α₁ / (1 + (p₃/K₃)^n) - k_degm₁ * m₁
    dX[2] = α₀₂ + α₂ / (1 + (p₁/K₁)^n) - k_degm₂ * m₂
    dX[3] = α₀₃ + α₃ / (1 + (p₂/K₂)^n) - k_degm₃ * m₃
    dX[4] = β₁ * m₁ - k_degp₁ * p₁
    dX[5] = β₂ * m₂ - k_degp₂ * p₂
    dX[6] = β₃ * m₃ - k_degp₃ * p₃
end

function solve_repressilator(t_save, θ, X0)
    tspan = (0.0, maximum(t_save))
    prob = ODEProblem(repressilator!, X0, tspan, θ)
    sol = solve(prob, Rodas4(), saveat=t_save, abstol=1e-10, reltol=1e-8)
    return Array(sol)
end

# MLE from successful run (moderate regime K ~ 20-30)
θ_MLE = [0.0089, 0.009656, 0.005,      # α₀
         0.8668, 1.418, 1.283,          # α
         0.01773, 0.02807, 0.01345,     # β
         20.99, 25.76, 29.6,            # K
         0.006938, 0.006034, 0.004547,  # k_degm
         0.001321, 0.001165, 0.001376]  # k_degp

θ_log_MLE = log.(θ_MLE)
X0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
t_obs = [0.0, 1250.0, 2500.0, 3750.0, 5000.0, 6250.0, 7500.0, 8750.0, 10000.0]

# Prediction function
function predict_mRNA(θ, t_grid)
    sol_matrix = solve_repressilator(t_grid, θ, X0)
    mRNA = sol_matrix[1:3, :]
    return vec(mRNA)
end

ϕ_log = θ_log -> predict_mRNA(exp.(θ_log), t_obs)

println("="^70)
println("Testing rtolM on Repressilator (Hessian method)")
println("="^70)

println("\nUsing MLE from successful moderate regime run:")
println("  K values: $(round.(θ_MLE[10:12], digits=2))")
println("  β values: $(round.(θ_MLE[7:9], sigdigits=3))")

println("\nRunning IIR with default rtolM (32√eps ≈ 4.8e-7)...")
flush(stdout)

S, N, N_perp, rank_J = find_invariant_subspace(ϕ_log, θ_log_MLE; verbose=true)

println("\nResults:")
println("  Jacobian rank: $rank_J / 18")
println("  σ_max: $(round(maximum(S), sigdigits=4))")
println("  τM = rtolM * σ_max ≈ $(round(32*sqrt(eps()) * maximum(S), sigdigits=4))")
println("  Invariant null space dimension: $(size(N, 2))")
println("  Identifiable directions: $(size(N_perp, 2))")

if size(N, 2) == 3
    println("\n✓ SUCCESS: Found 3 invariant null vectors with default tolerance!")
    println("  Expected: βK products are non-identifiable")
    @assert size(N, 2) == 3 "Sanity check: Should find exactly 3 invariant directions"
    @assert rank_J == 15 "Sanity check: Jacobian rank should be 15"
else
    println("\n✗ FAILURE: Found $(size(N, 2)) invariant vectors")
    println("  Expected: 3 (for βK products)")
    error("Default tolerance test failed: Got $(size(N, 2))/3 invariant directions")
end

println("\n" * "="^70)
println("✓ ALL TESTS PASSED")
println("="^70)
println("\nDefault tolerance rtolM=32√eps is correctly calibrated for stiff ODEs!")
