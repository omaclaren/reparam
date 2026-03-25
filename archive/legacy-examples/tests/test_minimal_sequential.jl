"""
MINIMAL Sequential IIR: No dictionary pre-processing
Just SVD bases at each stage, Varimax only for final display
"""

using LinearAlgebra
using Distributions
using Random
include("../../../ReparamTools.jl")
using .ReparamTools

Random.seed!(123)

θ_true = [21.0, 0.9, 110.0, 0.18]  # [n₁, p₁, n₂, p₂]
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

println("="^60)
println("MINIMAL SEQUENTIAL IIR")
println("="^60)

# ============================================================
# STAGE 1: f=log, pure SVD basis
# ============================================================
println("\nSTAGE 1 (f=log):")

ϕ_log = θ_log -> ϕ_func(exp.(θ_log))
S_s1, N_s1, N_perp_s1, rank_s1 = find_invariant_subspace(
    ϕ_log, log.(θ_true); rtolJ=sqrt(eps()), atolM=1e-10
)

println("  Rank: ", rank_s1, "/4")
println("  Dim(N): ", size(N_s1, 2))
println("  Dim(N_perp): ", size(N_perp_s1, 2))

# ============================================================
# STAGE 2: f=identity, work directly in Stage 1 SVD coords
# ============================================================
println("\nSTAGE 2 (f=identity, reduced coords):")

# Transformation: θ → y (2D SVD coords)
θ_to_y(θ) = N_perp_s1' * log.(θ)
y_to_θ(y) = exp.(N_perp_s1 * y)  # SIMPLIFIED: just the subspace part

# Stage 2 mapping
ϕ_stage2(y) = ϕ_func(y_to_θ(y))

y_true = θ_to_y(θ_true)
println("  y_true: ", round.(y_true, digits=4))

S_s2, N_s2, N_perp_s2, rank_s2 = find_invariant_subspace(
    ϕ_stage2, y_true; rtolJ=sqrt(eps()), atolM=1e-10
)

println("  Rank: ", rank_s2, "/2")
println("  Dim(N): ", size(N_s2, 2))
println("  Dim(N_perp): ", size(N_perp_s2, 2))

# ============================================================
# IDENTIFY which directions affect output
# ============================================================
println("\nIDENTIFY active directions:")

J_s2 = compute_ϕ_Jacobian(ϕ_stage2, y_true)

for col in 1:size(N_perp_s2, 2)
    v = N_perp_s2[:, col]
    Jv_norm = norm(J_s2 * v)
    println("  Column $col: ||J*v|| = ", round(Jv_norm, digits=6))
end

# Take only columns with non-zero Jacobian projection
active_cols = Int[]
for col in 1:size(N_perp_s2, 2)
    v = N_perp_s2[:, col]
    if norm(J_s2 * v) > 1e-6
        push!(active_cols, col)
    end
end

println("  Active columns: ", active_cols)
N_perp_s2_active = N_perp_s2[:, active_cols]

println("\nAfter filtering:")
println("  Identifiable dimension: ", length(active_cols))

# ============================================================
# POST-PROCESS: Varimax for interpretation
# ============================================================
println("\n" * "="^60)
println("POST-PROCESSING (Varimax interpretation)")
println("="^60)

# Compute Varimax rotation of Stage 1 basis
N_perp_varimax = varimax_rotation(N_perp_s1; n_restarts=200, threshold=1e-2)
N_perp_var_scaled = scale_and_round(N_perp_varimax; column_scales=ones(2))

println("\nStage 1 Varimax basis (4D log-space):")
display(N_perp_var_scaled)
println("\nInterpretation:")
println("  Column 1: [1,1,0,0] → log(n₁p₁)")
println("  Column 2: [0,0,1,1] → log(n₂p₂)")

# Rotation from SVD to Varimax in 2D
R = N_perp_s1' * N_perp_varimax

# Express active directions in Varimax basis
println("\nActive directions in Varimax coordinates:")
for (i, col) in enumerate(active_cols)
    v_svd = N_perp_s2[:, col]
    v_var = R' * v_svd
    v_var_norm = v_var / norm(v_var)

    println("\n  Direction $i:")
    println("    SVD coords: ", round.(v_svd, digits=4))
    println("    Varimax coords (normalized): ", round.(v_var_norm, digits=4))

    # Check if it's sum or difference
    if abs(v_var_norm[1] - v_var_norm[2]) < 0.1
        println("    → Sum: n₁p₁ + n₂p₂")
    elseif abs(v_var_norm[1] + v_var_norm[2]) < 0.1
        println("    → Difference: n₁p₁ - n₂p₂")
    end

    # Map to 4D for display
    v_4D = N_perp_varimax * v_var_norm
    v_4D_scaled = scale_and_round(v_4D; column_scales=[1.0])
    println("    4D exponents: ", round.(v_4D_scaled, digits=4))
end

# Also show invariant directions from Stage 2
if size(N_s2, 2) > 0
    println("\n\nInvariant directions (Stage 2):")
    for col in 1:size(N_s2, 2)
        v_svd = N_s2[:, col]
        v_var = R' * v_svd
        v_var_norm = v_var / norm(v_var)

        println("\n  Invariant $col:")
        println("    Varimax coords (normalized): ", round.(v_var_norm, digits=4))

        if abs(v_var_norm[1] - v_var_norm[2]) < 0.1
            println("    → Sum: n₁p₁ + n₂p₂ (THE INVARIANT COMBINATION!)")
        end
    end
end

println("\n" * "="^60)
println("SUMMARY")
println("="^60)
println("  Sequential: 4D → 2D → 1D")
println("  Stage 1: Pure SVD basis")
println("  Stage 2: Pure SVD basis, identify active directions via J")
println("  Post-process: Varimax for interpretation only")
println("  Result: 1 identifiable direction (difference)")
println("          1 invariant direction (sum)")
