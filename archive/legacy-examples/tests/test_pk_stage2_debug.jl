"""
Debug Stage 2 Jacobian computation
"""

using LinearAlgebra
using DifferentialEquations
include("../../../ReparamTools.jl")
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

# Stage 1 - minimal version
ϕ_log = θ_log -> ϕ_func_θ(exp.(θ_log))
S_s1, N_s1, N_perp_s1_raw, rank_s1 = find_invariant_subspace(
    ϕ_log, log.(θ_true); rtolJ=sqrt(eps()), atolM=1e-10
)

N_perp_varimax_raw = varimax_rotation(N_perp_s1_raw; n_restarts=200, threshold=1e-2)
N_perp_corrected = copy(N_perp_varimax_raw)

for col in 1:size(N_perp_varimax_raw, 2)
    v = N_perp_varimax_raw[:, col]
    max_idx = argmax(abs.(v))
    max_val = v[max_idx]
    if max_val < -0.5
        N_perp_corrected[:, col] = -N_perp_corrected[:, col]
    end
end

N_perp_scaled = scale_and_round(N_perp_corrected; column_scales=ones(size(N_perp_corrected, 2)))
N_clean = scale_and_round(N_s1; column_scales=ones(size(N_s1, 2)))

A1_full = transpose(hcat(N_perp_scaled, N_clean))
θ_to_stage1(θ) = exp.(A1_full * log.(θ))
stage1_to_θ(θ1) = exp.(inv(A1_full) * log.(θ1))

θ1_true = θ_to_stage1(θ_true)

println("Stage 1 coordinates:")
for i in 1:8
    println("  θ¹[$i] = ", round(θ1_true[i], digits=4))
end

# Stage 2 - directly compute Jacobian
println("\nStage 2 Jacobian computation:")
ϕ_stage2(θ1) = ϕ_func_θ(stage1_to_θ(θ1))

J_s2 = compute_ϕ_Jacobian(ϕ_stage2, θ1_true)
println("  Jacobian shape: ", size(J_s2))

U_s2, S_s2, V_s2 = svd(J_s2)
println("  Singular values:")
for (i, s) in enumerate(S_s2)
    println("    [$i]: ", round(s, digits=8))
end

rtol = sqrt(eps())
rank_s2 = count(S_s2 .> rtol * maximum(S_s2))
println("  Rank (rtol=$(rtol)): ", rank_s2)

# Check if using identity transformation (f=identity) makes a difference
println("\nNow run find_invariant_subspace on Stage 2:")
S_s2_fis, N_s2_fis, N_perp_s2_fis, rank_s2_fis = find_invariant_subspace(
    ϕ_stage2, θ1_true; rtolJ=sqrt(eps()), atolM=1e-10
)

println("  Rank from find_invariant_subspace: ", rank_s2_fis)
println("  Dim(N): ", size(N_s2_fis, 2))
println("  Dim(N_perp): ", size(N_perp_s2_fis, 2))

if size(N_perp_s2_fis, 2) > 0
    # Check rank of Jacobian projected onto N_perp
    J_proj_s2 = J_s2 * N_perp_s2_fis
    U_proj, S_proj, V_proj = svd(J_proj_s2)

    println("\n  J * N_perp singular values:")
    for (i, s) in enumerate(S_proj)
        println("    [$i]: ", round(s, digits=8))
    end

    rank_proj = count(S_proj .> rtol * maximum(S_proj))
    println("  Rank(J * N_perp): ", rank_proj)
end
