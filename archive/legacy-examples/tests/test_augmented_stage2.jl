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
println("AUGMENTED STAGE 2 TEST")
println("="^60)

# Stage 1: Standard approach with Varimax
ϕ_log = θ_log -> ϕ_func_θ(exp.(θ_log))
S_s1, N_s1, N_perp_s1, rank_s1 = find_invariant_subspace(
    ϕ_log, log.(θ_true); rtolJ=sqrt(eps()), atolM=1e-10
)

N_perp_varimax = varimax_rotation(N_perp_s1; n_restarts=200, threshold=1e-2)
A1_full = hcat(N_perp_varimax, N_s1)'
A1_inv = inv(A1_full)

transform_s1(θ) = exp.(A1_full * log.(θ))
inverse_s1(θ1) = exp.(A1_inv * log.(θ1))

θ1_true = transform_s1(θ_true)

println("\nStage 1 complete. θ¹ at true params:")
println(round.(θ1_true, digits=4))

# AUGMENTED STAGE 2
println("\n" * "="^60)
println("AUGMENTED STAGE 2 (dictionary approach)")
println("="^60)

# Create augmented parameter vector: [θ¹; θ]
# Dimension: 8 + 8 = 16

function ϕ_augmented(θ_aug)
    # θ_aug = [θ¹; θ] (16-dim)
    θ1_part = θ_aug[1:8]
    θ_part = θ_aug[9:16]

    # Convert θ¹ back to θ for model evaluation
    θ_from_θ1 = inverse_s1(θ1_part)

    # Could use either, but use θ_part (original)
    return ϕ_func_θ(θ_part)
end

# Augmented true parameters
θ_aug_true = [θ1_true; θ_true]

println("\nAugmented vector dimension: ", length(θ_aug_true))
println("  First 8: θ¹ (Varimax transformed)")
println("  Last 8: θ (original)")

# Apply find_invariant_subspace with f=identity to augmented vector
println("\nApplying find_invariant_subspace with f=identity...")
S_s2, N_s2, N_perp_s2, rank_s2 = find_invariant_subspace(
    ϕ_augmented, θ_aug_true;
    rtolJ=sqrt(eps()),
    atolM=1e-10
)

println("\nStage 2 results:")
println("  Rank: ", rank_s2)
println("  Dim(N_perp): ", size(N_perp_s2, 2))
println("  Dim(N): ", size(N_s2, 2))

if size(N_perp_s2, 2) > 0
    println("\n  N_perp (scaled for interpretation):")
    N_perp_s2_scaled = scale_and_round(N_perp_s2; column_scales=ones(size(N_perp_s2, 2)))
    display(N_perp_s2_scaled)

    println("\n\nLooking for k₀₂ + k₁₂ combination:")
    println("  k₀₂ is component 12 of augmented vector (θ[4])")
    println("  k₁₂ is component 13 of augmented vector (θ[5])")
    println()

    for col in 1:size(N_perp_s2_scaled, 2)
        v = N_perp_s2_scaled[:, col]
        if abs(v[12]) > 0.1 && abs(v[13]) > 0.1
            println("  Column ", col, " involves components 12 (k₀₂) and 13 (k₁₂):")
            println("    Coef[12] = ", v[12])
            println("    Coef[13] = ", v[13])
            if abs(v[12] - 1) < 0.1 && abs(v[13] - 1) < 0.1 && sum(abs.(v) .> 0.1) == 2
                println("    ✓ This appears to be k₀₂ + k₁₂!")
            end
        end
    end
end

println("\n\nConclusion:")
println("If Stage 2 found a column with coefficients [0,...,0,1,1,0,...,0]")
println("at positions 12,13 (corresponding to k₀₂, k₁₂), then the augmented")
println("approach successfully identified the sum q₃ = k₀₂ + k₁₂")
