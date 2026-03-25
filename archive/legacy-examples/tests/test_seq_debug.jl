"""
Debug: trace through the sequential mapping carefully
"""

using LinearAlgebra
using Distributions
using Random
include("../../../ReparamTools.jl")
using .ReparamTools

Random.seed!(123)

θ_true = [21.0, 0.9, 110.0, 0.18]
n_obs = 100

function generate_data(θ)
    n1, p1, n2, p2 = θ
    y1 = rand(Binomial(Int(n1), p1), n_obs)
    y2 = rand(Binomial(Int(n2), p2), n_obs)
    return y1 .+ y2
end

y_obs = generate_data(θ_true)

function negloglik(θ)
    n1, p1, n2, p2 = θ
    λ1, λ2 = n1 * p1, n2 * p2
    -sum(logpdf.(Poisson(λ1 + λ2), y_obs))
end

ϕ_func(θ) = [negloglik(θ)]

# Stage 1
ϕ_log = θ_log -> ϕ_func(exp.(θ_log))
S_s1, N_s1, N_perp_s1, rank_s1 = find_invariant_subspace(
    ϕ_log, log.(θ_true); rtolJ=sqrt(eps()), atolM=1e-10
)

println("Stage 1 SVD basis:")
println("N_perp_s1:")
display(N_perp_s1)

N_perp_varimax = varimax_rotation(N_perp_s1; n_restarts=200, threshold=1e-2)
N_perp_var_scaled = scale_and_round(N_perp_varimax; column_scales=ones(2))

println("\nVarimax basis (scaled):")
display(N_perp_var_scaled)

println("\nVarimax columns interpretation:")
println("  Column 1: [1,1,0,0] → log(n₁p₁)")
println("  Column 2: [0,0,1,1] → log(n₂p₂)")

# Now: what should the sum n₁p₁ + n₂p₂ be in Varimax coords?
# The sum in parameter space: n₁p₁ + n₂p₂
# In log-space this is NOT linear! log(n₁p₁ + n₂p₂) ≠ log(n₁p₁) + log(n₂p₂)

# That's why we need Stage 2 with f=identity!
# After Stage 1 transform with f=log, we have coordinates:
#   θ¹[1] = n₁p₁ (exponential of [1,1,0,0]·log(θ))
#   θ¹[2] = n₂p₂ (exponential of [0,0,1,1]·log(θ))

# Stage 2 with f=identity finds: θ¹[1] + θ¹[2]
# This should have coefficient vector [1, 1] in the 2D Varimax space

println("\n" * "="^60)
println("Expected result:")
println("  In 2D Varimax coords: [1, 1] (equal weight on both)")
println("  Interpretation: θ¹[1] + θ¹[2] = n₁p₁ + n₂p₂")
println("="^60)

# Now check what Stage 2 actually gives
y_true = N_perp_s1' * log.(θ_true)

function y_to_log_approx(y)
    # Approximate inverse using SVD basis
    # This is key: we're in 2D, need to map back to 4D
    # Use θ_true as anchor point
    θ_log_ref = log.(θ_true)
    N_perp_s1 * (y - N_perp_s1' * θ_log_ref) + θ_log_ref
end

ϕ_stage2(y) = ϕ_func(exp.(y_to_log_approx(y)))

S_s2, N_s2, N_perp_s2, rank_s2 = find_invariant_subspace(
    ϕ_stage2, y_true; rtolJ=sqrt(eps()), atolM=1e-10
)

println("\nStage 2 results:")
println("  Dim(N_perp): ", size(N_perp_s2, 2))
println("  N_perp_s2 (in 2D SVD basis):")
display(N_perp_s2)

# Check Jacobian
J_s2 = compute_ϕ_Jacobian(ϕ_stage2, y_true)
J_proj = J_s2 * N_perp_s2
U, S, V = svd(J_proj)

println("\n  Singular values of J*N_perp:")
println("    ", round.(S, digits=10))
println("  Principal direction (column 1): ", N_perp_s2[:, 1])

# Now rotate to Varimax basis
R = N_perp_s1' * N_perp_varimax

println("\n  Rotation R:")
display(R)

v_principal_svd = N_perp_s2[:, 1]
v_principal_var = R' * v_principal_svd

println("\n  Principal direction in Varimax 2D coords:")
println("    ", round.(v_principal_var, digits=4))
println("\n  Normalized:")
v_norm = v_principal_var / norm(v_principal_var)
println("    ", round.(v_norm, digits=4))

println("\n  Expected: approximately [0.707, 0.707] (equal weight)")
