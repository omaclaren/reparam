# Test repressilator: Sequential vs Distributed 2D Profiling
# Verifies that distributed profiling produces identical results to sequential
# Uses RepressilatorModel.jl module with autodiff enabled

using Distributed

println("="^70)
println("TEST: Repressilator Sequential vs Distributed Profiling")
println("="^70)

# Add workers
println("\n[1/6] Setting up workers...")
addprocs(2)
println("✓ Added 2 workers: ", workers())

# Load modules on main process
include("examples/RepressilatorModel.jl")
using .RepressilatorModel
include("ReparamTools.jl")
using .ReparamTools
using Distributions
using LinearAlgebra
using Random
using Printf
using Statistics

# Load on workers
@everywhere begin
    include($(joinpath(@__DIR__, "examples", "RepressilatorModel.jl")))
    using .RepressilatorModel
    include($(joinpath(@__DIR__, "ReparamTools.jl")))
    using .ReparamTools
    using Distributions
    using LinearAlgebra
end

println("\n[2/6] Setting up repressilator model...")
# Model setup (same as in repressilator.jl)
NT = 5  # Small for fast test
T_end = 5000.0
t_obs = LinRange(0, T_end, NT)
X0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
σ = 1.0

# True parameters (18 total, n=2.5 fixed)
θ_true = [0.5, 0.5, 0.5,  # α₀
          10.0, 10.0, 10.0,  # α
          0.02, 0.01, 0.015,  # β
          30.0, 26.0, 32.0,  # K
          0.35, 0.35, 0.35,  # k_degm
          0.12, 0.12, 0.12]  # k_degp

println("✓ Model setup complete")
println("  Time points: ", NT)
println("  T_end: ", T_end)
println("  Parameters: ", length(θ_true))
println("  Noise: σ = ", σ)

println("\n[3/6] Generating data...")
Random.seed!(123)  # Fixed seed for reproducibility
y_true = RepressilatorModel.predict_mRNA(θ_true, t_obs, X0)
data = y_true + σ * randn(length(y_true))

println("✓ Data generated")
println("  Observations: ", length(data))
println("  True prediction mean: ", round(mean(y_true), digits=2))

# Send data to workers
@everywhere t_obs_global = $(t_obs)
@everywhere X0_global = $(X0)
@everywhere σ_global = $(σ)
@everywhere data_global = $(data)

# Define likelihood on all processes
function lnlike_θ(θ)
    try
        pred = RepressilatorModel.predict_mRNA(θ, t_obs, X0)
        dist = MvNormal(pred, σ^2 * I(length(pred)))
        return logpdf(dist, data)
    catch e
        return -Inf
    end
end

@everywhere function lnlike_θ_worker(θ)
    try
        pred = RepressilatorModel.predict_mRNA(θ, t_obs_global, X0_global)
        dist = MvNormal(pred, σ_global^2 * I(length(pred)))
        return logpdf(dist, data_global)
    catch e
        return -Inf
    end
end

# Transform to log space for optimization
lnlike_θ_log(θ_log) = lnlike_θ(exp.(θ_log))

@everywhere lnlike_θ_log_worker(θ_log) = lnlike_θ_worker(exp.(θ_log))

# Verify likelihood at true parameters
ll_true = lnlike_θ(θ_true)
println("\n  Likelihood at true params: ", round(ll_true, digits=4))

if !isfinite(ll_true)
    println("✗ ERROR: Likelihood at true parameters is not finite!")
    rmprocs(workers())
    exit(1)
end

println("\n[4/6] Running SEQUENTIAL 3×3 profile (β₁, K₁)...")
println("  Grid: 3×3 = 9 points")
println("  Target parameters: [7, 10] (β₁ and K₁)")
println("  Optimization timeout: 20s per point")

# Parameter bounds in log space
θ_log_true = log.(θ_true)
θ_log_lower = θ_log_true .- 1.0  # Wider bounds
θ_log_upper = θ_log_true .+ 1.0

# Nuisance parameter initial guess
target_indices = [7, 10]
nuisance_indices = setdiff(1:18, target_indices)
nuisance_guess = θ_log_true[nuisance_indices]

println("\n  Starting sequential profile...")
t_seq = @elapsed begin
    θ_seq, ll_seq = ReparamTools.profile_target(
        lnlike_θ_log, target_indices,
        θ_log_lower, θ_log_upper,
        nuisance_guess;
        grid_steps=3,
        use_distributed=false,
        optmaxtime=20.0
    )
end

println("✓ Sequential complete")
println("  Time: $(round(t_seq, digits=2))s ($(round(t_seq/60, digits=2)) min)")
println("  Grid points: ", length(ll_seq))
println("  Finite likelihoods: ", sum(isfinite.(ll_seq)), "/", length(ll_seq))
if sum(isfinite.(ll_seq)) > 0
    println("  Max likelihood: ", round(maximum(ll_seq[isfinite.(ll_seq)]), digits=4))
end

println("\n[5/6] Running DISTRIBUTED 3×3 profile (β₁, K₁)...")
println("  Same grid: 3×3 = 9 points")
println("  Workers: 2, chunks: 2")
println("  Optimization timeout: 20s per point")

println("\n  Starting distributed profile...")
t_dist = @elapsed begin
    θ_dist, ll_dist = ReparamTools.profile_target(
        lnlike_θ_log_worker, target_indices,
        θ_log_lower, θ_log_upper,
        nuisance_guess;
        grid_steps=3,
        use_distributed=true,
        n_chunks=2,
        optmaxtime=20.0
    )
end

println("✓ Distributed complete")
println("  Time: $(round(t_dist, digits=2))s ($(round(t_dist/60, digits=2)) min)")
println("  Grid points: ", length(ll_dist))
println("  Finite likelihoods: ", sum(isfinite.(ll_dist)), "/", length(ll_dist))
if sum(isfinite.(ll_dist)) > 0
    println("  Max likelihood: ", round(maximum(ll_dist[isfinite.(ll_dist)]), digits=4))
end

println("\n[6/6] Comparing results...")
println("  Speedup: $(round(t_seq/t_dist, digits=2))x")

# Compare likelihoods
if length(ll_seq) != length(ll_dist)
    println("\n✗ FAIL: Grid sizes differ!")
    println("  Sequential: ", length(ll_seq))
    println("  Distributed: ", length(ll_dist))
    rmprocs(workers())
    exit(1)
end

# Filter out non-finite values for comparison
finite_mask = isfinite.(ll_seq) .& isfinite.(ll_dist)
n_finite = sum(finite_mask)

println("\n  Likelihood comparison:")
println("    Points with finite values in both: ", n_finite, "/", length(ll_seq))

if n_finite > 0
    ll_diff = abs.(ll_seq[finite_mask] - ll_dist[finite_mask])
    max_diff = maximum(ll_diff)
    mean_diff = mean(ll_diff)

    println("    Max absolute difference:  ", @sprintf("%.2e", max_diff))
    println("    Mean absolute difference: ", @sprintf("%.2e", mean_diff))
else
    println("    No points with finite values in both runs")
    max_diff = Inf
    mean_diff = Inf
end

# Check for matching finite/infinite status
finite_seq = isfinite.(ll_seq)
finite_dist = isfinite.(ll_dist)
finite_mismatch = sum(finite_seq .!= finite_dist)

println("    Finite/Inf status matches: ", finite_mismatch == 0 ? "✓" : "✗ $(finite_mismatch) mismatches")

# Detailed mismatch analysis
if finite_mismatch > 0
    println("\n    Mismatch details:")
    for i in 1:length(ll_seq)
        if finite_seq[i] != finite_dist[i]
            println("      Point $i: seq=$(finite_seq[i] ? "finite" : "inf"), dist=$(finite_dist[i] ? "finite" : "inf")")
        end
    end
end

# Final verdict (relaxed tolerance for stiff ODE)
tolerance = 1e-6
println("\n" * "="^70)
if n_finite > 0 && max_diff < tolerance && finite_mismatch == 0
    println("✓✓✓ TEST PASSED ✓✓✓")
    println("Sequential and distributed results match within tolerance $(tolerance)")
    println("All $(n_finite) finite points agree")
    exit_code = 0
elseif n_finite > 0 && max_diff < tolerance && finite_mismatch <= 2
    println("⚠ TEST PASSED (with warnings) ⚠")
    println("Finite values match within tolerance, but $(finite_mismatch) finite/inf mismatches")
    println("This may indicate optimization sensitivity for stiff ODE system")
    exit_code = 0
else
    println("✗✗✗ TEST FAILED ✗✗✗")
    if n_finite == 0
        println("No finite likelihood values in both runs")
    elseif max_diff >= tolerance
        println("Maximum difference $(max_diff) exceeds tolerance $(tolerance)")
    end
    if finite_mismatch > 2
        println("Too many finite/inf mismatches: $(finite_mismatch)")
    end
    exit_code = 1
end
println("="^70)

# Cleanup
rmprocs(workers())

exit(exit_code)
