"""
Quick analysis of Stage 2 results for PK model
"""

using LinearAlgebra
using DifferentialEquations
using ForwardDiff

# Include necessary functions
include("../ReparamTools.jl")
using .ReparamTools

# True parameters
b₁_true, c₁_true, k₀₁_true, k₀₂_true = 2.0, 1.5, 0.2, 0.1
k₁₂_true, k₂₁_true, V_M_true, K_M_true = 0.3, 0.25, 1.0, 3.0
θ_true = [b₁_true, c₁_true, k₀₁_true, k₀₂_true, k₁₂_true, k₂₁_true, V_M_true, K_M_true]

# ODE model
function pk_ode!(dx, x, θ, t)
    b₁, c₁, k₀₁, k₀₂, k₁₂, k₂₁, V_M, K_M = θ
    x₁, x₂ = x
    mm_clearance = (V_M * x₁) / (K_M + x₁)
    u_input = t < 0.1 ? 1.0 : 0.0
    dx[1] = -(k₀₁ + k₁₂)*x₁ + k₂₁*x₂ - mm_clearance + b₁*u_input
    dx[2] = k₁₂*x₁ - (k₀₂ + k₂₁)*x₂
end

# Observation times and initial conditions
t_obs = collect(range(0.1, 5.0, length=20))
x0 = [0.0, 0.0]

# Output function
function ϕ_func_θ(θ)
    prob = ODEProblem(pk_ode!, x0, (0.0, maximum(t_obs)), θ)
    sol = solve(prob, Tsit5(), saveat=t_obs, abstol=1e-10, reltol=1e-10)
    c₁ = θ[2]
    return c₁ * sol[1, :]  # Observe first compartment
end

println("="^60)
println("STAGE 1 ANALYSIS")
println("="^60)

# Stage 1: f = log
ϕ_log = θ_log -> ϕ_func_θ(exp.(θ_log))
S_stage1, N_stage1, N_perp_stage1, rankJ_stage1 = find_invariant_subspace(
    ϕ_log, log.(θ_true);
    rtolJ=sqrt(eps()),
    atolM=1e-10
)

println("\nStage 1 Results:")
println("  Rank: ", rankJ_stage1)
println("  Dim(N_perp): ", size(N_perp_stage1, 2))
println("  Dim(N): ", size(N_stage1, 2))

# Apply Varimax if needed
if size(N_perp_stage1, 2) > 1
    println("\n  Applying Varimax rotation...")
    N_perp_rotated = varimax_rotation(N_perp_stage1; n_restarts=200, threshold=1e-2)
    N_perp_for_transform = N_perp_rotated
else
    N_perp_for_transform = N_perp_stage1
end

# Build Stage 1 transformation
A1_full_T = hcat(N_perp_for_transform, N_stage1)
A1_full = A1_full_T'
A1_inv = inv(A1_full)

transform_stage1(θ) = exp.(A1_full * log.(θ))
inverse_transform_stage1(θ1) = exp.(A1_inv * log.(θ1))

θ1_true = transform_stage1(θ_true)

println("\n  Stage 1 coordinates at true θ:")
for i in 1:8
    println("    θ¹[$i] = ", round(θ1_true[i], digits=4))
end

println("\n" * "="^60)
println("STAGE 2 ANALYSIS")
println("="^60)

# Stage 2: f = identity in θ¹ space
ϕ_stage1_coords = θ1 -> ϕ_func_θ(inverse_transform_stage1(θ1))

S_stage2, N_stage2, N_perp_stage2, rankJ_stage2 = find_invariant_subspace(
    ϕ_stage1_coords, θ1_true;
    rtolJ=sqrt(eps()),
    atolM=1e-10
)

println("\nStage 2 Results:")
println("  Rank: ", rankJ_stage2)
println("  Dim(N_perp): ", size(N_perp_stage2, 2))
println("  Dim(N): ", size(N_stage2, 2))

println("\n  Singular values:")
display(S_stage2)

if size(N_perp_stage2, 2) > 0
    println("\n\n  N_perp (potentially identifiable in θ¹ space):")
    display(N_perp_stage2)

    println("\n\n  Scaled N_perp for interpretation:")
    N_perp_scaled = scale_and_round(N_perp_stage2; column_scales=ones(size(N_perp_stage2, 2)))
    display(N_perp_scaled)

    println("\n\n  Interpreting Stage 2 combinations:")
    for col in 1:size(N_perp_scaled, 2)
        coeffs = N_perp_scaled[:, col]
        nonzero_idx = findall(x -> abs(x) > 0.01, coeffs)
        if !isempty(nonzero_idx)
            terms = String[]
            for i in nonzero_idx
                c = coeffs[i]
                if c ≈ 1
                    push!(terms, "θ¹[$i]")
                elseif c ≈ -1
                    push!(terms, "-θ¹[$i]")
                else
                    push!(terms, "$(c)θ¹[$i]")
                end
            end
            println("    Direction $col: ", join(terms, " + "))
        end
    end
end

println("\n" * "="^60)
println("INTERPRETATION")
println("="^60)

# From A1_full, we know:
# θ¹[1] = 1/k₀₂, θ¹[3] = 1/k₂₁, θ¹[4] = k₁₂, θ¹[7] = k₀₁

println("\nFrom Stage 1 transformation:")
println("  θ¹[1] ≈ 1/k₀₂")
println("  θ¹[3] ≈ 1/k₂₁")
println("  θ¹[4] ≈ k₁₂")
println("  θ¹[7] ≈ k₀₁")

println("\nMeshkat's identifiable combinations:")
println("  q₃ = k₀₂ + k₁₂ = 1/θ¹[1] + θ¹[4]")
println("  q₅ = c₁V_M(k₀₁ + k₂₁) involves sum k₀₁ + k₂₁ = θ¹[7] + 1/θ¹[3]")

println("\nDoes Stage 2 N_perp contain directions corresponding to these combinations?")
println("We need to check if N_perp_stage2 has columns that pick out these sums...")

# Check specific combinations
println("\n\nChecking if Stage 2 found key combinations:")

if size(N_perp_stage2, 2) > 0
    # Look for direction involving θ¹[7] and θ¹[3] (for k₀₁ + k₂₁)
    for col in 1:size(N_perp_stage2, 2)
        v = N_perp_stage2[:, col]
        if abs(v[7]) > 0.1 && abs(v[3]) > 0.1 && sum(abs.(v) .> 0.1) == 2
            println("  Column $col appears to involve θ¹[7] and θ¹[3]:")
            println("    Coefficients: ", round.(v, digits=4))
            println("    This could relate to k₀₁ + k₂₁ if nonlinear...")
        end
    end

    # Look for direction involving θ¹[1] and θ¹[4] (for k₀₂ + k₁₂)
    for col in 1:size(N_perp_stage2, 2)
        v = N_perp_stage2[:, col]
        if abs(v[1]) > 0.1 && abs(v[4]) > 0.1 && sum(abs.(v) .> 0.1) == 2
            println("  Column $col appears to involve θ¹[1] and θ¹[4]:")
            println("    Coefficients: ", round.(v, digits=4))
            println("    This could relate to k₀₂ + k₁₂ if nonlinear...")
        end
    end
end

println("\n" * "="^60)
println("CONCLUSION")
println("="^60)

println("\nKey findings:")
println("1. Stage 1 (f=log) found rank ", rankJ_stage1, " with ", size(N_perp_stage1, 2), " potentially identifiable directions")
println("2. Stage 2 (f=identity) found rank ", rankJ_stage2, " with ", size(N_perp_stage2, 2), " potentially identifiable directions")
println("3. Stage 2 should identify LINEAR combinations in θ¹ space")
println("4. But Meshkat's q₃, q₅ appear as NONLINEAR functions of θ¹:")
println("   q₃ = 1/θ¹[1] + θ¹[4]  (reciprocal + linear)")
println("   q₅ involves 1/θ¹[3] + θ¹[7]")
println("\n⚠ This suggests Stage 2 with f=identity may NOT capture these combinations!")
println("  Alternative: Need Stage 2 with different f, or symbolic analysis of invariants")
