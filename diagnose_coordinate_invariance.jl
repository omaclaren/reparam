# Diagnostic script to investigate coordinate-dependent invariance
# This will help us understand why both coordinate systems show invariant null spaces

if !@isdefined(ReparamTools)
    include("ReparamTools.jl")
    println("✓ ReparamTools module included")
end

using .ReparamTools
using LinearAlgebra
using ForwardDiff

println("="^70)
println("DIAGNOSTIC: Coordinate-Dependent Invariance Investigation")
println("="^70)

# Use a simple example where we KNOW the behavior
# Let's use the repressilator at MLE from the test

# First, load the repressilator model components
include("examples/repressilator.jl")

# We should now have θ_MLE and θ_log_MLE from the repressilator run
# But since we can't guarantee that, let's use the true parameters as a test point

println("\nTest point: θ_true (true parameters)")
println("θ_true = ", θ_true)

# Define auxiliary map
t_test = [0.0, 1000.0, 2000.0, 3000.0, 4000.0]  # Short time grid for speed
ϕ_log = θ_log -> begin
    θ = exp.(θ_log)
    sol_matrix = solve_repressilator(t_test, θ, X0)
    mRNA = extract_mrna(sol_matrix)
    return vec(mRNA)
end

ϕ_original = θ -> begin
    sol_matrix = solve_repressilator(t_test, θ, X0)
    mRNA = extract_mrna(sol_matrix)
    return vec(mRNA)
end

θ_test = θ_true
θ_log_test = log.(θ_test)

println("\n" * "="^70)
println("1. LOG-SPACE ANALYSIS")
println("="^70)

# Compute Jacobian in log-space
J_log = ForwardDiff.jacobian(ϕ_log, θ_log_test)
svd_log = svd(J_log)

println("Jacobian dimensions: ", size(J_log))
println("Singular values: ", round.(svd_log.S, sigdigits=4))
rank_log = count(svd_log.S .> sqrt(eps()) * svd_log.S[1])
println("Rank: $rank_log/18")

# Get null space
null_dim_log = 18 - rank_log
if null_dim_log > 0
    N_log = svd_log.V[:, rank_log+1:end]
    println("\nNull space basis (log-space):")
    println("Dimensions: ", size(N_log))

    param_names_short = ["α₀₁", "α₀₂", "α₀₃", "α₁", "α₂", "α₃",
                         "β₁", "β₂", "β₃", "K₁", "K₂", "K₃",
                         "kdm1", "kdm2", "kdm3", "kdp1", "kdp2", "kdp3"]

    for j in 1:min(3, size(N_log, 2))
        println("\n  Null vector $j (log-space):")
        v = N_log[:, j]
        # Show all components
        for i in 1:18
            if abs(v[i]) > 0.01
                println("    $(param_names_short[i]): $(round(v[i], digits=3))")
            end
        end

        # Check βK pattern
        println("  βK pattern check:")
        for gene in 1:3
            beta_idx = 6 + gene
            K_idx = 9 + gene
            beta_coef = v[beta_idx]
            K_coef = v[K_idx]
            match = abs(beta_coef - K_coef) < 0.05
            println("    β$gene=$(round(beta_coef, digits=3)), K$gene=$(round(K_coef, digits=3)) → $(match ? "✓" : "✗")")
        end
    end
end

println("\n" * "="^70)
println("2. ORIGINAL-SPACE ANALYSIS")
println("="^70)

# Compute Jacobian in original space
J_orig = ForwardDiff.jacobian(ϕ_original, θ_test)
svd_orig = svd(J_orig)

println("Jacobian dimensions: ", size(J_orig))
println("Singular values: ", round.(svd_orig.S, sigdigits=4))
rank_orig = count(svd_orig.S .> sqrt(eps()) * svd_orig.S[1])
println("Rank: $rank_orig/18")

# Get null space
null_dim_orig = 18 - rank_orig
if null_dim_orig > 0
    N_orig = svd_orig.V[:, rank_orig+1:end]
    println("\nNull space basis (original-space):")
    println("Dimensions: ", size(N_orig))

    for j in 1:min(3, size(N_orig, 2))
        println("\n  Null vector $j (original-space):")
        v = N_orig[:, j]
        # Show all components
        for i in 1:18
            if abs(v[i]) > 0.01
                println("    $(param_names_short[i]): $(round(v[i], digits=3))")
            end
        end

        # Check βK pattern (would be different in original space!)
        println("  βK pattern check:")
        for gene in 1:3
            beta_idx = 6 + gene
            K_idx = 9 + gene
            beta_coef = v[beta_idx]
            K_coef = v[K_idx]
            match = abs(beta_coef - K_coef) < 0.05
            println("    β$gene=$(round(beta_coef, digits=3)), K$gene=$(round(K_coef, digits=3)) → $(match ? "✓" : "✗")")
        end
    end
end

println("\n" * "="^70)
println("3. RELATIONSHIP BETWEEN NULL SPACES")
println("="^70)

# Check if null spaces span the same subspace (up to rotation)
if null_dim_log > 0 && null_dim_orig > 0 && null_dim_log == null_dim_orig
    # Compute N_log' * N_orig - if they span same space, this should have full rank
    overlap = N_log' * N_orig
    println("Overlap matrix N_log' * N_orig:")
    println(round.(overlap, digits=3))

    overlap_svd = svd(overlap)
    println("\nSingular values of overlap:")
    println(round.(overlap_svd.S, digits=4))

    if minimum(overlap_svd.S) > 0.1
        println("\n✓ Null spaces span approximately the same subspace")
        println("  (All singular values > 0.1)")
    else
        println("\n✗ Null spaces span different subspaces")
        println("  (Some singular values near 0)")
    end
else
    println("Cannot compare - null space dimensions differ")
end

println("\n" * "="^70)
println("4. JACOBIAN MAGNITUDE COMPARISON")
println("="^70)

# Check if Jacobian magnitudes are very different
println("Jacobian norms:")
println("  ||J_log||_F = ", round(norm(J_log), sigdigits=4))
println("  ||J_orig||_F = ", round(norm(J_orig), sigdigits=4))
println("  Ratio: ", round(norm(J_log) / norm(J_orig), digits=2))

println("\nMax singular value:")
println("  σ_max(J_log) = ", round(svd_log.S[1], sigdigits=4))
println("  σ_max(J_orig) = ", round(svd_orig.S[1], sigdigits=4))
println("  Ratio: ", round(svd_log.S[1] / svd_orig.S[1], digits=2))

println("\n" * "="^70)
println("CONCLUSION")
println("="^70)
println("This diagnostic will help us understand whether:")
println("  1. Both null spaces are truly the same (unexpected)")
println("  2. Null spaces differ but both happen to be invariant (very special)")
println("  3. There's a bug in the invariance test")
println("  4. The tolerance is too permissive")
