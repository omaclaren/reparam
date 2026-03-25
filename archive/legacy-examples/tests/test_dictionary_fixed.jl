"""
Dictionary approach - CORRECT implementation
Stage 2 runs on SVD coordinates y, then rotate to Varimax for interpretation
"""

using LinearAlgebra
include("../../../ReparamTools.jl")
using .ReparamTools

# True parameters
n₁, p₁, n₂, p₂ = 100.0, 0.2, 20.0, 0.1
xy_true = [n₁, p₁, n₂, p₂]

# Model
ϕ_xy(xy) = [xy[1]*xy[2] + xy[3]*xy[4], xy[1]*xy[2] + xy[3]*xy[4]]

println("="^60)
println("CORRECT DICTIONARY APPROACH")
println("="^60)

# ============================================================
# STAGE 1: Build SVD basis + Varimax dictionary
# ============================================================

println("\nSTAGE 1: SVD basis + Varimax dictionary")
println("-"^60)

ϕ_log = xy_log -> ϕ_xy(exp.(xy_log))
S_s1, N_s1, N_perp_s1, rank_s1 = find_invariant_subspace(
    ϕ_log, log.(xy_true); rtolJ=sqrt(eps()), atolM=1e-10
)

println("Stage 1 results:")
println("  Rank: ", rank_s1)
println("  Dim(N_perp): ", size(N_perp_s1, 2))

# Core SVD basis
println("\nCore SVD basis N_perp:")
display(N_perp_s1)

# Build Varimax dictionary (no threshold for orthonormality)
N_perp_varimax = varimax_rotation(N_perp_s1; n_restarts=200, threshold=0.0)

# Rotation matrix
R = N_perp_s1' * N_perp_varimax

println("\n\nVarimax basis (for interpretation):")
N_var_display = copy(N_perp_varimax)
N_var_display[abs.(N_var_display) .< 1e-2] .= 0.0
display(N_var_display)

println("\n\nRotation matrix R (Varimax relative to SVD):")
display(R)

# ============================================================
# STAGE 2: Run on SVD coordinates y (CORRECT!)
# ============================================================

println("\n\n" * "="^60)
println("STAGE 2: Run on SVD coordinates y")
println("="^60)

# Define Stage 2 mapping in SVD coordinates
# y → θ via θ = exp(N_perp * y)
# ϕ(y) = ϕ(exp(N_perp * y))

ϕ_stage2_svd(y) = ϕ_xy(exp.(N_perp_s1 * y))

y_true = N_perp_s1' * log.(xy_true)

println("\nSVD coordinates at true parameters:")
println("  y = ", round.(y_true, digits=4))

# Apply find_invariant_subspace on y (NO exponentials in inverse!)
println("\nApplying find_invariant_subspace to SVD coordinates...")
S_s2, N_s2, N_perp_s2, rank_s2 = find_invariant_subspace(
    ϕ_stage2_svd, y_true;
    rtolJ=sqrt(eps()),
    atolM=1e-10
)

println("\nStage 2 results:")
println("  Rank: ", rank_s2)
println("  Dim(N_perp): ", size(N_perp_s2, 2))
println("  Dim(N): ", size(N_s2, 2))

if size(N_perp_s2, 2) > 0
    println("\n  N_perp in SVD space:")
    display(N_perp_s2)

    # NOW rotate to Varimax for interpretation
    println("\n  Rotating to Varimax dictionary for interpretation:")
    N_perp_s2_varimax = R * N_perp_s2

    println("\n  N_perp in Varimax space:")
    display(N_perp_s2_varimax)

    println("\n  Scaled for interpretation:")
    N_perp_s2_var_scaled = scale_and_round(N_perp_s2_varimax; column_scales=ones(size(N_perp_s2_varimax, 2)))
    display(N_perp_s2_var_scaled)

    println("\n\nInterpreting in Varimax coordinates:")
    for col in 1:size(N_perp_s2_var_scaled, 2)
        v = N_perp_s2_var_scaled[:, col]
        nonzero_idx = findall(x -> abs(x) > 0.1, v)

        if !isempty(nonzero_idx)
            terms = String[]
            for i in nonzero_idx
                c = v[i]
                if abs(c - 1) < 0.1
                    push!(terms, "z[$i]")
                elseif abs(c + 1) < 0.1
                    push!(terms, "-z[$i]")
                else
                    push!(terms, "$(round(c, digits=2))*z[$i]")
                end
            end
            println("  Column $col: ", join(terms, " + "))
        end
    end

    # Check for z[1] + z[2]
    println("\n\nLooking for z[1] + z[2] (sum n₁p₁ + n₂p₂):")
    for col in 1:size(N_perp_s2_var_scaled, 2)
        v = N_perp_s2_var_scaled[:, col]
        if abs(v[1]) > 0.5 && abs(v[2]) > 0.5 && abs(v[1] - v[2]) < 0.3
            println("  ✓ Column $col appears to be z[1] + z[2]!")
        end
    end
end

if size(N_s2, 2) > 0
    println("\n  N (invariant null space) in SVD space:")
    display(N_s2)

    println("\n  Rotating to Varimax:")
    N_s2_varimax = R * N_s2
    display(N_s2_varimax)
end

println("\n\n" * "="^60)
println("SUMMARY")
println("="^60)
println("\nKey insight:")
println("  ✓ Run Stage 2 on SVD coordinates y (stable inverse)")
println("  ✓ Rotate results to Varimax AFTER Stage 2 (for interpretation)")
println("  ✓ Dictionary is for display only, not for invariance test")
println()
println("This avoids exponentials in the inverse, keeping Hessian test numerically stable")
