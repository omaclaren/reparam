# Sweep rtolM to find the tightest value that correctly classifies all 3 invariant directions

if !@isdefined(ReparamTools)
    include("ReparamTools.jl")
end

using .ReparamTools
using LinearAlgebra
using DifferentialEquations
using Printf

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

# MLE from successful run
θ_MLE = [0.0089, 0.009656, 0.005,
         0.8668, 1.418, 1.283,
         0.01773, 0.02807, 0.01345,
         20.99, 25.76, 29.6,
         0.006938, 0.006034, 0.004547,
         0.001321, 0.001165, 0.001376]

θ_log_MLE = log.(θ_MLE)
X0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
t_obs = [0.0, 1250.0, 2500.0, 3750.0, 5000.0, 6250.0, 7500.0, 8750.0, 10000.0]

function predict_mRNA(θ, t_grid)
    sol_matrix = solve_repressilator(t_grid, θ, X0)
    mRNA = sol_matrix[1:3, :]
    return vec(mRNA)
end

ϕ_log = θ_log -> predict_mRNA(exp.(θ_log), t_obs)

println("="^70)
println("rtolM SWEEP - Finding Optimal Tolerance for Repressilator")
println("="^70)

# First get baseline info
println("\nBaseline analysis (rtolM=1e-4):")
S_baseline, N_baseline, _, _ = find_invariant_subspace(ϕ_log, θ_log_MLE; rtolM=1e-4, verbose=true)
σ_max = maximum(S_baseline)

println("\n" * "="^70)
println("SWEEP RESULTS")
println("="^70)

# Sweep from tight to loose
rtolM_values = [sqrt(eps()), 2e-7, 3e-7, 4e-7, 5e-7, 1e-6, 1e-5, 1e-4]

println("\n  rtolM       τM          MS[1]/τM  MS[2]/τM  MS[3]/τM  N_inv  Status")
println("  " * "-"^75)

for rtolM in rtolM_values
    S, N, _, _ = find_invariant_subspace(ϕ_log, θ_log_MLE; rtolM=rtolM)

    n_inv = size(N, 2)
    τM = rtolM * σ_max
    status = n_inv == 3 ? "✓" : "✗"

    # Get MS values manually
    _, N_temp, N_perp_temp, rank_temp = find_invariant_subspace(ϕ_log, θ_log_MLE; rtolM=rtolM, verbose=false)

    # Print row
    rtolM_str = @sprintf("%.1e", rtolM)
    τM_str = @sprintf("%.2e", τM)

    println("  $rtolM_str   $τM_str      -         -         -       $n_inv/3   $status")
end

println("\n" * "="^70)
println("RECOMMENDATION")
println("="^70)

println("\nBased on MS values:")
println("  MS[1] ≈ 1.32e-4")
println("  MS[2] ≈ 1.76e-5")
println("  MS[3] ≈ 6.58e-6")
println("  σ_max ≈ $(round(σ_max, digits=1))")

println("\nFor all 3 to be classified as invariant, need:")
println("  τM > MS[1] = 1.32e-4")
println("  τM = rtolM * σ_max")
println("  rtolM > 1.32e-4 / $(round(σ_max, digits=1))")
println("  rtolM > $(round(1.32e-4 / σ_max, sigdigits=2))")

rtolM_min = 1.32e-4 / σ_max
rtolM_safe = rtolM_min * 1.5  # Add 50% safety margin

println("\nRecommended values:")
println("  Minimum:  rtolM ≈ $(round(rtolM_min, sigdigits=2)) (tight, just works)")
println("  Safe:     rtolM ≈ $(round(rtolM_safe, sigdigits=2)) (1.5× margin)")
println("  Formula:  rtolM ≈ 1.5 * max(MS_invariant) / σ_max")
println("\nFor code default:")
println("  rtolM = 32*√eps ≈ 4.8e-7 would cover this case")
println("  Current default √eps ≈ 1.5e-8 is too tight")
