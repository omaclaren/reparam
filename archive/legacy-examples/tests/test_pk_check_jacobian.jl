"""
Check which Stage 1 coordinates actually affect output
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

# Stage 1: Same as test_pk_dictionary.jl
ϕ_log = θ_log -> ϕ_func_θ(exp.(θ_log))
S_s1, N_s1, N_perp_s1_raw, rank_s1 = find_invariant_subspace(
    ϕ_log, log.(θ_true); rtolJ=sqrt(eps()), atolM=1e-10
)

println("Stage 1 SVD:")
println("  Rank: ", rank_s1, "/8")
println("  Dim(N_perp): ", size(N_perp_s1_raw, 2))

# Apply Varimax + sign correction (same as test_pk_dictionary.jl)
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

# Build transformation
A1_full = transpose(hcat(N_perp_scaled, N_clean))
θ_to_stage1(θ) = exp.(A1_full * log.(θ))
stage1_to_θ(θ1) = exp.(inv(A1_full) * log.(θ1))

θ1_true = θ_to_stage1(θ_true)

# Compute Jacobian at Stage 1 coordinates
ϕ_stage1(θ1) = ϕ_func_θ(stage1_to_θ(θ1))
J_stage1 = compute_ϕ_Jacobian(ϕ_stage1, θ1_true)

println("\nStage 1 Jacobian analysis:")
println("  Jacobian shape: ", size(J_stage1))
U_J, S_J, V_J = svd(J_stage1)
println("  Singular values: ", round.(S_J, digits=6))
println("  Rank: ", count(S_J .> sqrt(eps()) * maximum(S_J)))

# Check which coordinates affect output
println("\nWhich Stage 1 coordinates affect output?")
param_labels = ["θ¹[$i]" for i in 1:8]

for i in 1:8
    # Norm of i-th row of V_J weighted by singular values
    sensitivity = norm(S_J .* V_J[i, :])
    println("  $param_labels[i]: sensitivity = $(round(sensitivity, digits=6))")
end

# Also check the N_perp projection
println("\nProjection onto N_perp from Stage 1:")
# N_perp_scaled are the first 7 columns of A1_full'
N_perp_s1_scaled = (A1_full')[:, 1:7]

J_proj = J_stage1 * N_perp_s1_scaled
U_proj, S_proj, V_proj = svd(J_proj)

println("  J * N_perp shape: ", size(J_proj))
println("  Singular values: ", round.(S_proj, digits=6))
println("  Rank: ", count(S_proj .> sqrt(eps()) * maximum(S_proj)))

println("\nInterpretation:")
println("  If rank(J * N_perp) < 7, then not all 7 'potentially identifiable'")
println("  coordinates actually affect the output.")
