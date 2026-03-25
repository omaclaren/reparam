"""
Test dictionary approach on PK model
Question: Can augmenting with original parameters help Stage 2 find k₀₂ + k₁₂?
"""

using LinearAlgebra
using DifferentialEquations

include("../../../ReparamTools.jl")
using .ReparamTools

# True parameters
b₁, c₁, k₀₁, k₀₂, k₁₂, k₂₁, V_M, K_M = 2.0, 1.5, 0.2, 0.1, 0.3, 0.25, 1.0, 3.0
θ_true = [b₁, c₁, k₀₁, k₀₂, k₁₂, k₂₁, V_M, K_M]

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
println("DICTIONARY APPROACH: PK Model")
println("="^60)
println("Question: Can dictionary with ORIGINAL θ help find k₀₂ + k₁₂?")

# ============================================================
# STAGE 1: Core SVD basis (no Varimax yet)
# ============================================================

println("\n\nSTAGE 1: Core SVD basis")
println("-"^60)

ϕ_log = θ_log -> ϕ_func_θ(exp.(θ_log))
S_s1, N_s1, N_perp_s1, rank_s1 = find_invariant_subspace(
    ϕ_log, log.(θ_true); rtolJ=sqrt(eps()), atolM=1e-10
)

println("Stage 1 results:")
println("  Rank: ", rank_s1, "/8")
println("  Dim(N_perp): ", size(N_perp_s1, 2))

# Core transformation (SVD basis only, no rotation)
n_core = size(N_perp_s1, 2)

y_to_logθ(y) = N_perp_s1 * y
logθ_to_y(logθ) = N_perp_s1' * logθ

y_true = logθ_to_y(log.(θ_true))

println("\nCore y coordinates: ", round.(y_true, digits=4))

# ============================================================
# STAGE 2: Dictionary with ORIGINAL θ parameters
# ============================================================

println("\n\n" * "="^60)
println("STAGE 2: Augment with original θ parameters")
println("="^60)

# Augmented vector: [y; θ]
# The idea: θ includes k₀₂, k₁₂ directly, so Stage 2 can form k₀₂ + k₁₂

function θ_to_augmented(θ)
    logθ = log.(θ)
    y = logθ_to_y(logθ)
    return [y; θ]  # Combine core coordinates and original params
end

function augmented_to_θ(aug)
    # Use ONLY core y for inversion
    y = aug[1:n_core]
    logθ = y_to_logθ(y)
    return exp.(logθ)
end

aug_true = θ_to_augmented(θ_true)

println("\nAugmented vector dimension: ", length(aug_true))
println("  Components 1-", n_core, ": y (core SVD coordinates)")
println("  Components ", n_core+1, "-", n_core+8, ": θ (original parameters)")
println()
println("Key: θ[4]=k₀₂ is component ", n_core+4)
println("     θ[5]=k₁₂ is component ", n_core+5)

# Auxiliary mapping in augmented space
ϕ_augmented(aug) = ϕ_func_θ(augmented_to_θ(aug))

# Apply find_invariant_subspace with f=identity
println("\nApplying find_invariant_subspace to [y; θ]...")
S_s2, N_s2, N_perp_s2, rank_s2 = find_invariant_subspace(
    ϕ_augmented, aug_true;
    rtolJ=sqrt(eps()),
    atolM=1e-10
)

println("\nStage 2 results:")
println("  Rank: ", rank_s2)
println("  Dim(N_perp): ", size(N_perp_s2, 2))

if size(N_perp_s2, 2) > 0
    println("\n  N_perp (scaled):")
    N_perp_s2_scaled = scale_and_round(N_perp_s2; column_scales=ones(size(N_perp_s2, 2)))
    display(N_perp_s2_scaled)

    println("\n\nLooking for k₀₂ + k₁₂:")
    k02_idx = n_core + 4  # Position of k₀₂ in augmented vector
    k12_idx = n_core + 5  # Position of k₁₂ in augmented vector

    println("  k₀₂ is component ", k02_idx)
    println("  k₁₂ is component ", k12_idx)

    found_sum = false
    for col in 1:size(N_perp_s2_scaled, 2)
        v = N_perp_s2_scaled[:, col]
        if abs(v[k02_idx]) > 0.5 && abs(v[k12_idx]) > 0.5
            println("\n  Column ", col, " involves k₀₂ and k₁₂:")
            println("    k₀₂ coefficient: ", v[k02_idx])
            println("    k₁₂ coefficient: ", v[k12_idx])

            # Check if it's approximately k₀₂ + k₁₂
            if abs(v[k02_idx] - v[k12_idx]) < 0.2  # Similar coefficients
                other_terms = sum(abs.(v)) - abs(v[k02_idx]) - abs(v[k12_idx])
                if other_terms < 0.5  # Few other terms
                    println("    ✓ This appears to be a*k₀₂ + a*k₁₂ combination!")
                    found_sum = true
                end
            end
        end
    end

    if !found_sum
        println("\n  ✗ Did not find k₀₂ + k₁₂ combination")
        println("    Stage 2 may be using only y coordinates")
    end
end

println("\n\n" * "="^60)
println("CONCLUSION")
println("="^60)
println("If Stage 2 found k₀₂ + k₁₂:")
println("  ✓ Dictionary approach works - original params provide needed access")
println()
println("If Stage 2 used only y coordinates:")
println("  ✗ Redundancy in [y; θ] doesn't help")
println("  → Need different approach for sum-type combinations")
