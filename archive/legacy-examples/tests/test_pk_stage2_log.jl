"""
Test Stage 2 using f=log (not f=identity)
This might give cleaner rank determination
"""

using LinearAlgebra
using DifferentialEquations
include("../../../ReparamTools.jl")
using .ReparamTools

# Parameters: [b₁, c₁, k₀₁, k₀₂, k₁₂, k₂₁, V_M, K_M]
θ_true = [2.0, 1.5, 0.2, 0.1, 0.3, 0.25, 1.0, 3.0]

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
println("PK MODEL: Stage 2 with f=log")
println("="^60)

# Stage 1 (f=log)
ϕ_log = θ_log -> ϕ_func_θ(exp.(θ_log))
S_s1, N_s1, N_perp_s1_raw, rank_s1 = find_invariant_subspace(
    ϕ_log, log.(θ_true); rtolJ=sqrt(eps()), atolM=1e-10
)

println("\nStage 1 (f=log):")
println("  Rank: ", rank_s1, "/8")

# Varimax + sign correction
N_perp_varimax = varimax_rotation(N_perp_s1_raw; n_restarts=200, threshold=0.0)

N_perp_corrected = copy(N_perp_varimax)
for col in 1:size(N_perp_varimax, 2)
    v = N_perp_varimax[:, col]
    max_idx = argmax(abs.(v))
    max_val = v[max_idx]
    if max_val < -0.5
        N_perp_corrected[:, col] = -N_perp_corrected[:, col]
    end
end

# Orthonormal transformation
A1_full = transpose(hcat(N_perp_corrected, N_s1))

θ_to_stage1(θ) = exp.(A1_full * log.(θ))
stage1_to_θ(θ1) = exp.(A1_full' * log.(θ1))

θ1_true = θ_to_stage1(θ_true)

# Stage 2 with f=log (not f=identity!)
println("\nStage 2 (f=log):")
ϕ_stage2_log(θ1_log) = ϕ_func_θ(stage1_to_θ(exp.(θ1_log)))

S_s2, N_s2, N_perp_s2, rank_s2 = find_invariant_subspace(
    ϕ_stage2_log, log.(θ1_true); rtolJ=sqrt(eps()), atolM=1e-10
)

println("  Rank: ", rank_s2)
println("  Dim(N): ", size(N_s2, 2))
println("  Dim(N_perp): ", size(N_perp_s2, 2))

# Check singular values
J_s2_log = compute_ϕ_Jacobian(ϕ_stage2_log, log.(θ1_true))
U, S_vals, V = svd(J_s2_log)

println("\n  Singular values (f=log):")
for (i, s) in enumerate(S_vals)
    println("    [$i]: ", round(s, digits=10))
end

rtol = sqrt(eps())
rank_direct = count(S_vals .> rtol * maximum(S_vals))
println("  Rank: ", rank_direct)

println("\nCompare with f=identity:")
ϕ_stage2_id(θ1) = ϕ_func_θ(stage1_to_θ(θ1))
J_s2_id = compute_ϕ_Jacobian(ϕ_stage2_id, θ1_true)
U_id, S_vals_id, V_id = svd(J_s2_id)

println("  Singular values (f=identity):")
for (i, s) in enumerate(S_vals_id)
    println("    [$i]: ", round(s, digits=10))
end

rank_id = count(S_vals_id .> rtol * maximum(S_vals_id))
println("  Rank: ", rank_id)

println("\nConclusion:")
println("  Using f=log may give different (potentially cleaner) rank.")
