# Test if Hessian-based method works for repressilator now

if !@isdefined(ReparamTools)
    include("ReparamTools.jl")
end

using .ReparamTools
using LinearAlgebra
using ForwardDiff
using DifferentialEquations

# Inline the repressilator ODE model
function repressilator_rhs!(du, u, p, t)
    m1, m2, m3, p1, p2, p3 = u
    α₀₁, α₀₂, α₀₃, α₁, α₂, α₃, β₁, β₂, β₃, K₁, K₂, K₃,
    k_degm₁, k_degm₂, k_degm₃, k_degp₁, k_degp₂, k_degp₃, n = p

    f1 = α₀₁ + α₁ / (1 + (p3/K₁)^n)
    f2 = α₀₂ + α₂ / (1 + (p1/K₂)^n)
    f3 = α₀₃ + α₃ / (1 + (p2/K₃)^n)

    du[1] = f1 - k_degm₁ * m1
    du[2] = f2 - k_degm₂ * m2
    du[3] = f3 - k_degm₃ * m3
    du[4] = β₁ * m1 - k_degp₁ * p1
    du[5] = β₂ * m2 - k_degp₂ * p2
    du[6] = β₃ * m3 - k_degp₃ * p3
end

function solve_repressilator(t_span, θ, X0, n)
    p = vcat(θ, n)
    tspan = (first(t_span), last(t_span))
    prob = ODEProblem(repressilator_rhs!, X0, tspan, p)
    sol = solve(prob, Rodas4(), saveat=t_span, abstol=1e-8, reltol=1e-6)
    return sol
end

# MLE from previous run
θ_MLE = [0.006088, 0.00944, 0.00664,  # α₀
         0.8, 1.143, 1.255,             # α
         0.01461, 0.02451, 0.01041,     # β
         20.75, 21.22, 31.47,           # K
         0.008, 0.006449, 0.004437,     # k_degm
         0.001288, 0.001123, 0.001276]  # k_degp

θ_log_MLE = log.(θ_MLE)
X0 = [5.0, 0.0, 0.0, 0.0, 0.0, 0.0]
t_obs = [0.0, 1666.7, 3333.3, 5000.0, 6666.7, 8333.3, 10000.0]

println("="^70)
println("REPRESSILATOR: Testing Hessian-Based Method")
println("="^70)

function predict_mRNA_short(θ, t_grid)
    n = 3.0
    sol = solve_repressilator(t_grid, θ, X0, n)
    if sol.retcode != :Success
        error("ODE solve failed: $(sol.retcode)")
    end
    sol_matrix = Array(sol)
    mRNA = sol_matrix[1:3, :]
    return vec(mRNA)
end

ϕ_log = θ_log -> predict_mRNA_short(exp.(θ_log), t_obs)

println("\nAttempting Hessian-based invariance test...")
println("(This may fail with nested AD error)")

try
    S, N, N_perp, rank = find_invariant_subspace(
        ϕ_log, θ_log_MLE;
        # No invariance_method = uses Hessian (default)
        verbose=true
    )

    println("\n✓ SUCCESS! Hessian-based method worked!")
    println("  Rank: $rank/18")
    println("  Invariant null vectors: $(size(N, 2))")
    println("  Non-invariant: $(size(N_perp, 2) - rank)")

catch e
    println("\n✗ FAILED as expected:")
    println("  Error type: $(typeof(e))")
    println("  Message: $(sprint(showerror, e))")

    if isa(e, MethodError)
        println("\n  This is likely the nested AD issue.")
        println("  → Finite-difference method is necessary for stiff ODEs")
    end
end
