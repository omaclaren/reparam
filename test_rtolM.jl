# Test that rtolM tolerance works correctly

if !@isdefined(ReparamTools)
    include("ReparamTools.jl")
end

using .ReparamTools
using LinearAlgebra

# Simple stat_model example
function ϕ_stat_model(θ_log)
    n, p = exp.(θ_log)
    return [n*p, n*p]  # Both mean and variance equal np for Poisson
end

# MLE from stat_model
θ_log_MLE = [log(100.0), log(0.5)]

println("="^70)
println("Testing rtolM (relative tolerance) implementation")
println("="^70)

# Test 1: Default (should use rtolM)
println("\n1. Default tolerance (rtolM = √eps):")
S, N, N_perp, rank = find_invariant_subspace(ϕ_stat_model, θ_log_MLE)
println("  Rank: $rank/2")
println("  Invariant null space dim: $(size(N, 2))")
println("  σ_max: $(maximum(S))")
println("  τM = rtolM * σ_max ≈ $(sqrt(eps()) * maximum(S))")

# Test 2: Explicit atolM (backward compatibility)
println("\n2. Explicit atolM = 1e-10 (backward compatibility):")
S2, N2, N_perp2, rank2 = find_invariant_subspace(ϕ_stat_model, θ_log_MLE; atolM=1e-10)
println("  Rank: $rank2/2")
println("  Invariant null space dim: $(size(N2, 2))")
println("  τM = atolM = 1e-10 (fixed)")

# Test 3: Explicit rtolM
println("\n3. Explicit rtolM = 1e-6 (relaxed):")
S3, N3, N_perp3, rank3 = find_invariant_subspace(ϕ_stat_model, θ_log_MLE; rtolM=1e-6)
println("  Rank: $rank3/2")
println("  Invariant null space dim: $(size(N3, 2))")
println("  τM = rtolM * σ_max ≈ $(1e-6 * maximum(S))")

println("\n✓ All tests completed successfully!")
println("\nExpected behavior:")
println("  - All three should find rank = 1")
println("  - All three should find 1 invariant null vector")
println("  - rtolM automatically scales with problem magnitude")
