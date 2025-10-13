"""
Apply dictionary approach to PK model
Test: Can we fix sign lottery by flipping Varimax columns?
"""

using LinearAlgebra
using DifferentialEquations
include("../ReparamTools.jl")
using .ReparamTools

# Parameters: [b₁, c₁, k₀₁, k₀₂, k₁₂, k₂₁, V_M, K_M]
θ_true = [2.0, 1.5, 0.2, 0.1, 0.3, 0.25, 1.0, 3.0]

# ODE model
function pk_ode!(dx, x, θ, t)
    b₁, c₁, k₀₁, k₀₂, k₁₂, k₂₁, V_M, K_M = θ
    x₁, x₂ = x
    mm_clearance = (V_M * x₁) / (K_M + x₁)
    u_input = t < 0.1 ? 1.0 : 0.0
    dx[1] = -(k₀₁ + k₁₂)*x₁ + k₂₁*x₂ - mm_clearance + b₁*u_input
    dx[2] = k₁₂*x₁ - (k₀₂ + k₂₁)*x₂
end

t_obs = collect(range(0.1, 5.0, length=20))
x0 = [0.0, 0.0]

function ϕ_func_θ(θ)
    prob = ODEProblem(pk_ode!, x0, (0.0, maximum(t_obs)), θ)
    sol = solve(prob, Tsit5(), saveat=t_obs, abstol=1e-10, reltol=1e-10)
    return θ[2] * sol[1, :]
end

println("="^60)
println("PK MODEL: Dictionary Approach with Sign Correction")
println("="^60)

# ============================================================
# STAGE 1: Find N_perp, apply Varimax, check signs
# ============================================================

println("\nSTAGE 1: SVD + Varimax with sign checking")
println("-"^60)

ϕ_log = θ_log -> ϕ_func_θ(exp.(θ_log))
S_s1, N_s1, N_perp_s1_raw, rank_s1 = find_invariant_subspace(
    ϕ_log, log.(θ_true); rtolJ=sqrt(eps()), atolM=1e-10
)

println("Stage 1 results:")
println("  Rank: ", rank_s1, "/8")
println("  Dim(N_perp): ", size(N_perp_s1_raw, 2))

# Apply Varimax
N_perp_varimax_raw = varimax_rotation(N_perp_s1_raw; n_restarts=200, threshold=1e-2)

println("\nVarimax-rotated N_perp (before sign correction):")
N_var_display = copy(N_perp_varimax_raw)
N_var_display[abs.(N_var_display) .< 1e-2] .= 0.0
display(N_var_display)

# Check for negative exponents (reciprocals)
# Row 4 = k₀₂ (parameter index 4)
# Row 5 = k₁₂ (parameter index 5)

println("\n\nChecking for negative exponents:")
param_names = ["b₁", "c₁", "k₀₁", "k₀₂", "k₁₂", "k₂₁", "V_M", "K_M"]

N_perp_corrected = copy(N_perp_varimax_raw)

for col in 1:size(N_perp_varimax_raw, 2)
    # Find dominant parameter in this column
    v = N_perp_varimax_raw[:, col]
    max_idx = argmax(abs.(v))
    max_val = v[max_idx]

    println("  Column $col: dominant parameter $(param_names[max_idx]) with exponent $(round(max_val, digits=3))")

    # If negative (reciprocal), flip the sign
    if max_val < -0.5
        println("    → Flipping column $col to make positive exponent")
        N_perp_corrected[:, col] = -N_perp_corrected[:, col]
    end
end

println("\nCorrected N_perp (after sign flips):")
N_corr_display = copy(N_perp_corrected)
N_corr_display[abs.(N_corr_display) .< 1e-2] .= 0.0
display(N_corr_display)

# Scale and round
N_perp_scaled = scale_and_round(N_perp_corrected; column_scales=ones(size(N_perp_corrected, 2)))
N_clean = scale_and_round(N_s1; column_scales=ones(size(N_s1, 2)))

println("\nScaled N_perp:")
display(N_perp_scaled)

# Build square transformation
A1_full = transpose(hcat(N_perp_scaled, N_clean))

θ_to_stage1(θ) = exp.(A1_full * log.(θ))
stage1_to_θ(θ1) = exp.(inv(A1_full) * log.(θ1))

θ1_true = θ_to_stage1(θ_true)

println("\nStage 1 coordinates:")
for i in 1:8
    println("  θ¹[$i] = ", round(θ1_true[i], digits=4))
end

# Interpret first few Stage 1 coordinates
println("\nInterpretation (from scaled matrix rows):")
for i in 1:min(5, size(A1_full, 1))
    row = A1_full[i, :]
    nonzero = findall(x -> abs(x) > 0.1, row)
    if !isempty(nonzero)
        terms = String[]
        for j in nonzero
            c = row[j]
            if abs(c - 1) < 0.1
                push!(terms, param_names[j])
            elseif abs(c + 1) < 0.1
                push!(terms, "1/$(param_names[j])")
            else
                push!(terms, "$(param_names[j])^$(round(c, digits=2))")
            end
        end
        println("  θ¹[$i] ~ ", join(terms, " * "))
    end
end

# ============================================================
# STAGE 2: Find sums (like k₀₂ + k₁₂)
# ============================================================

println("\n\n" * "="^60)
println("STAGE 2: Find sum combinations")
println("="^60)

ϕ_stage2(θ1) = ϕ_func_θ(stage1_to_θ(θ1))

S_s2, N_s2, N_perp_s2, rank_s2 = find_invariant_subspace(
    ϕ_stage2, θ1_true; rtolJ=sqrt(eps()), atolM=1e-10
)

println("\nStage 2 results:")
println("  Rank: ", rank_s2)
println("  Dim(N_perp): ", size(N_perp_s2, 2))
println("  Dim(N): ", size(N_s2, 2))

if size(N_perp_s2, 2) > 0
    N_perp_s2_scaled = scale_and_round(N_perp_s2; column_scales=ones(size(N_perp_s2, 2)))

    println("\n  N_perp (scaled):")
    display(N_perp_s2_scaled)

    println("\n\nLooking for sums:")

    # Meshkat's q₃ = k₀₂ + k₁₂
    # If our sign correction worked, we should have:
    # θ¹[i] = k₀₂, θ¹[j] = k₁₂
    # And Stage 2 should find θ¹[i] + θ¹[j]

    for col in 1:size(N_perp_s2_scaled, 2)
        v = N_perp_s2_scaled[:, col]
        nonzero_idx = findall(x -> abs(x) > 0.1, v)

        if length(nonzero_idx) == 2 && all(v[nonzero_idx] .> 0)
            i, j = nonzero_idx
            println("  ✓ Column $col: θ¹[$i] + θ¹[$j] (a sum!)")
            println("    → This could be k₀₂ + k₁₂ if sign correction worked")
        end
    end
end

println("\n\n" * "="^60)
println("CONCLUSION")
println("="^60)
println("\nDid sign correction solve the sign lottery?")
println("  - Check if Stage 1 coordinates include k₀₂, k₁₂ (not 1/k₀₂)")
println("  - Check if Stage 2 found their sum")
