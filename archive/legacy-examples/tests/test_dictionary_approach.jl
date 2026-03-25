"""
Test dictionary approach on sum-of-Poisson model
Stage 1: Build core SVD basis + Varimax dictionary
Stage 2: Use augmented dictionary [y; z_varimax]
"""

using LinearAlgebra
include("../ReparamTools.jl")
using .ReparamTools

# True parameters
n₁, p₁, n₂, p₂ = 100.0, 0.2, 20.0, 0.1
xy_true = [n₁, p₁, n₂, p₂]

# Auxiliary mapping (sum of two Poisson limits)
ϕ_xy(xy) = [xy[1]*xy[2] + xy[3]*xy[4], xy[1]*xy[2] + xy[3]*xy[4]]

println("="^60)
println("DICTIONARY APPROACH: Sum-of-Poisson Model")
println("="^60)

# ============================================================
# STAGE 1: Build core SVD basis + dictionary
# ============================================================

println("\nSTAGE 1: Core basis + dictionary")
println("-"^60)

# Apply find_invariant_subspace in log-space
ϕ_log = xy_log -> ϕ_xy(exp.(xy_log))
S_s1, N_s1, N_perp_s1, rank_s1 = find_invariant_subspace(
    ϕ_log, log.(xy_true); rtolJ=sqrt(eps()), atolM=1e-10
)

println("Stage 1 results:")
println("  Rank: ", rank_s1, "/", length(xy_true))
println("  Dim(N_perp): ", size(N_perp_s1, 2))
println("  Dim(N): ", size(N_s1, 2))

println("\nCore SVD basis N_perp:")
display(N_perp_s1)

# Core transformation using SVD basis
# Forward: y = N_perp' * log(θ)
# Backward: log(θ) = N_perp * y

y_to_logxy(y) = N_perp_s1 * y
logxy_to_y(logxy) = N_perp_s1' * logxy

y_true = logxy_to_y(log.(xy_true))
println("\n\nCore Stage 1 coordinates y:")
println("  y = N_perp' * log(θ)")
println("  y_true = ", round.(y_true, digits=4))

# Build dictionary: Varimax rotation
println("\n\nBuilding dictionary features:")
println("  Varimax rotation of N_perp...")

# Use Varimax WITHOUT thresholding to preserve orthonormality
N_perp_varimax = varimax_rotation(N_perp_s1; n_restarts=200, threshold=0.0)

println("\n  Varimax basis (orthonormal, no thresholding):")
display(N_perp_varimax)

# For interpretation, apply threshold to a COPY
N_perp_varimax_display = copy(N_perp_varimax)
N_perp_varimax_display[abs.(N_perp_varimax_display) .< 1e-2] .= 0.0
println("\n  Varimax for display (thresholded for interpretation only):")
display(N_perp_varimax_display)

# Dictionary features are linear combinations of y
# If N_perp_varimax = N_perp_s1 * R, then
# z_varimax = R * y = R * N_perp_s1' * log(θ) = N_perp_varimax' * log(θ)

# Find rotation matrix R: N_perp_varimax = N_perp_s1 * R
# So R = N_perp_s1' * N_perp_varimax (since both are orthonormal)
R = N_perp_s1' * N_perp_varimax

println("\n\nRotation matrix R (Varimax relative to SVD):")
display(R)

# Dictionary coordinates
z_varimax = R * y_true
println("\n\nDictionary coordinates z_varimax = R * y:")
println("  z_varimax = ", round.(z_varimax, digits=4))

# Verify: z_varimax should equal N_perp_varimax' * log(θ)
z_varimax_check = N_perp_varimax' * log.(xy_true)
println("  Check: N_perp_varimax' * log(θ) = ", round.(z_varimax_check, digits=4))
println("  Match: ", isapprox(z_varimax, z_varimax_check, rtol=1e-10))

# ============================================================
# STAGE 2: Use augmented dictionary [y; z_varimax]
# ============================================================

println("\n\n" * "="^60)
println("STAGE 2: Augmented dictionary approach")
println("="^60)

# Augmented coordinates: combine core y and dictionary z_varimax
# Dimension: size(N_perp, 2) for each = 2 + 2 = 4

n_core = size(N_perp_s1, 2)  # Should be 2 for this example

function xy_to_stage2(xy)
    logxy = log.(xy)
    y = logxy_to_y(logxy)  # Core coordinates
    z_var = R * y          # Varimax dictionary
    return z_var  # Use ONLY Varimax for Stage 2
end

function stage2_to_xy(z)
    # Invert: z = R*y, so y = R^{-1}*z
    y = R \ z
    logxy = y_to_logxy(y)
    return exp.(logxy)
end

z_s2_true = xy_to_stage2(xy_true)

println("\nStage 2 coordinates (Varimax only, no augmentation):")
println("  z_varimax = ", round.(z_s2_true, digits=4))

# Define auxiliary mapping in Varimax space
ϕ_stage2(z) = ϕ_xy(stage2_to_xy(z))

# Apply find_invariant_subspace with f=identity to Varimax coordinates
println("\nApplying find_invariant_subspace to Varimax coordinates...")
S_s2, N_s2, N_perp_s2, rank_s2 = find_invariant_subspace(
    ϕ_stage2, z_s2_true;
    rtolJ=sqrt(eps()),
    atolM=1e-10
)

println("\nStage 2 results:")
println("  Rank: ", rank_s2)
println("  Dim(N_perp): ", size(N_perp_s2, 2))
println("  Dim(N): ", size(N_s2, 2))

if size(N_perp_s2, 2) > 0
    println("\n  N_perp (potentially identifiable in augmented space):")
    N_perp_s2_scaled = scale_and_round(N_perp_s2; column_scales=ones(size(N_perp_s2, 2)))
    display(N_perp_s2_scaled)

    println("\n\nInterpreting Stage 2 results (in Varimax space):")
    println()

    for col in 1:size(N_perp_s2_scaled, 2)
        v = N_perp_s2_scaled[:, col]
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

    # Check if we found z[1] + z[2] (the sum n₁p₁ + n₂p₂)
    println("\n\nLooking for z[1] + z[2] (which would be n₁p₁ + n₂p₂):")
    for col in 1:size(N_perp_s2_scaled, 2)
        v = N_perp_s2_scaled[:, col]
        if abs(v[1] - 1) < 0.2 && abs(v[2] - 1) < 0.2
            println("  ✓ Column $col appears to be z[1] + z[2]!")
        end
    end
end

println("\n\n" * "="^60)
println("SUMMARY")
println("="^60)
println("Expected: Stage 2 should identify n₁p₁+n₂p₂ using Varimax coords")
println("  Varimax gives sparse: z[1]~n₁p₁, z[2]~n₂p₂")
println("  So z[1]+z[2] should be identifiable")
println()
println("Key advantage of dictionary approach:")
println("  - Core SVD basis ensures stable inversion")
println("  - Varimax dictionary provides interpretable combinations")
println("  - Stage 2 automatically selects which to use")
