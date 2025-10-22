# Test stat_model: Sequential vs Distributed 2D Profiling
# Verifies that distributed profiling produces identical results to sequential

using Distributed

println("="^70)
println("TEST: stat_model Sequential vs Distributed Profiling")
println("="^70)

# Add workers
println("\n[1/5] Setting up workers...")
addprocs(2)
println("✓ Added 2 workers: ", workers())

# Load ReparamTools on main process
include("ReparamTools.jl")
using .ReparamTools
using Distributions
using LinearAlgebra
using Random
using Printf
using Statistics

# Load on workers
@everywhere begin
    include($(joinpath(@__DIR__, "ReparamTools.jl")))
    using .ReparamTools
    using Distributions
    using LinearAlgebra
end

println("\n[2/5] Setting up stat_model...")
# Set random seed for reproducibility
Random.seed!(42)

# Model parameters (Poisson limit: Normal(np, sqrt(np)))
n_true, p_true = 100.0, 0.2
θ_true = [n_true, p_true]

# Parameter bounds
θ_lower = [0.1, 0.0001]
θ_upper = [500.0, 1.0]

# Fixed data (same as stat_model.jl)
data_fixed = [21.9, 22.3, 12.8, 16.4, 16.4, 20.3, 16.2, 20.0, 19.7, 24.4]

# Define model: θ = [n, p] -> Normal(np, sqrt(np))
function predict_mean_std(θ)
    n, p = θ
    np = n * p
    return np, sqrt(np)
end

# Likelihood function
function lnlike_θ(θ)
    try
        μ, σ = predict_mean_std(θ)
        if σ <= 0 || !isfinite(μ) || !isfinite(σ)
            return -Inf
        end
        dist = Normal(μ, σ)
        return sum(logpdf(dist, d) for d in data_fixed)
    catch
        return -Inf
    end
end

# Send data and likelihood to workers
@everywhere data_global = $(data_fixed)

@everywhere function predict_mean_std_worker(θ)
    n, p = θ
    np = n * p
    return np, sqrt(np)
end

@everywhere function lnlike_θ_worker(θ)
    try
        μ, σ = predict_mean_std_worker(θ)
        if σ <= 0 || !isfinite(μ) || !isfinite(σ)
            return -Inf
        end
        dist = Normal(μ, σ)
        return sum(logpdf(dist, d) for d in data_global)
    catch
        return -Inf
    end
end

println("✓ Model setup complete")
println("  True parameters: n=$(n_true), p=$(p_true)")
println("  Data: ", length(data_fixed), " observations")
println("  Mean: ", round(mean(data_fixed), digits=2))

# Verify likelihood at true parameters
ll_true = lnlike_θ(θ_true)
println("\n  Likelihood at true params: ", round(ll_true, digits=4))

println("\n[3/5] Running SEQUENTIAL 2D profile (n, p)...")
println("  Grid: 5×5 = 25 points")
println("  Target parameters: [1, 2] (n and p)")

# Initial guess for nuisance parameters (empty for 2D case)
nuisance_indices = Int[]
nuisance_guess = Float64[]

# Sequential profiling
println("\n  Starting sequential profile...")
t_seq = @elapsed begin
    θ_seq, ll_seq = ReparamTools.profile_target(
        lnlike_θ, [1, 2],  # Profile both n and p
        θ_lower, θ_upper,
        nuisance_guess;
        grid_steps=5,
        use_distributed=false,
        optmaxtime=5.0
    )
end

println("✓ Sequential complete")
println("  Time: $(round(t_seq, digits=2))s")
println("  Grid points: ", length(ll_seq))
println("  Finite likelihoods: ", sum(isfinite.(ll_seq)), "/", length(ll_seq))
println("  Max likelihood: ", round(maximum(ll_seq[isfinite.(ll_seq)]), digits=4))

println("\n[4/5] Running DISTRIBUTED 2D profile (n, p)...")
println("  Same grid: 5×5 = 25 points")
println("  Workers: 2, chunks: 2")

println("\n  Starting distributed profile...")
t_dist = @elapsed begin
    θ_dist, ll_dist = ReparamTools.profile_target(
        lnlike_θ_worker, [1, 2],
        θ_lower, θ_upper,
        nuisance_guess;
        grid_steps=5,
        use_distributed=true,
        n_chunks=2,
        optmaxtime=5.0
    )
end

println("✓ Distributed complete")
println("  Time: $(round(t_dist, digits=2))s")
println("  Grid points: ", length(ll_dist))
println("  Finite likelihoods: ", sum(isfinite.(ll_dist)), "/", length(ll_dist))
println("  Max likelihood: ", round(maximum(ll_dist[isfinite.(ll_dist)]), digits=4))

println("\n[5/5] Comparing results...")
println("  Speedup: $(round(t_seq/t_dist, digits=2))x")

# Compare likelihoods
if length(ll_seq) != length(ll_dist)
    println("\n✗ FAIL: Grid sizes differ!")
    println("  Sequential: ", length(ll_seq))
    println("  Distributed: ", length(ll_dist))
    exit(1)
end

# Compute differences
ll_diff = abs.(ll_seq - ll_dist)
max_diff = maximum(ll_diff)
mean_diff = mean(ll_diff)

println("\n  Likelihood comparison:")
println("    Max absolute difference:  ", @sprintf("%.2e", max_diff))
println("    Mean absolute difference: ", @sprintf("%.2e", mean_diff))

# Check for matching finite/infinite status
finite_seq = isfinite.(ll_seq)
finite_dist = isfinite.(ll_dist)
finite_mismatch = sum(finite_seq .!= finite_dist)

println("    Finite/Inf status matches: ", finite_mismatch == 0 ? "✓" : "✗ $(finite_mismatch) mismatches")

# Final verdict
tolerance = 1e-8
println("\n" * "="^70)
if max_diff < tolerance && finite_mismatch == 0
    println("✓✓✓ TEST PASSED ✓✓✓")
    println("Sequential and distributed results match within tolerance $(tolerance)")
    exit_code = 0
else
    println("✗✗✗ TEST FAILED ✗✗✗")
    if max_diff >= tolerance
        println("Maximum difference $(max_diff) exceeds tolerance $(tolerance)")
    end
    if finite_mismatch > 0
        println("Finite/Inf status differs at $(finite_mismatch) points")
    end
    exit_code = 1
end
println("="^70)

# Cleanup
rmprocs(workers())

exit(exit_code)
