# Test stat_model: Concurrent 1D Profiles with Worker Pool Partitioning
# Runs 1D profiles for n and p in parallel using disjoint worker pools
# Generates comparison figures showing sequential vs concurrent execution

using Distributed

println("="^70)
println("TEST: stat_model Concurrent 1D Profiles with Figures")
println("="^70)

# Add workers
println("\n[1/6] Setting up workers...")
addprocs(4)  # Need 4 workers to partition into 2 pools of 2
println("✓ Added 4 workers: ", workers())

# Load ReparamTools on main process
include("ReparamTools.jl")
using .ReparamTools
using Distributions
using LinearAlgebra
using Random
using Printf
using Statistics
using Plots
using Distributed: WorkerPool

# Load on workers
@everywhere begin
    include($(joinpath(@__DIR__, "ReparamTools.jl")))
    using .ReparamTools
    using Distributions
    using LinearAlgebra
end

println("\n[2/6] Setting up stat_model...")
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

# Likelihood function (NO type annotation for ForwardDiff!)
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
println("\n  Likelihood at true params: ", round(ll_true, digits=3))

println("\n[3/6] Running SEQUENTIAL 1D profiles...")
println("  Grid: 500 points per parameter (same as original stat_model.jl)")

# Sequential profile for n (fixing p at center)
center = 0.5 .* (θ_lower .+ θ_upper)
println("\n  Profile 1 (n): Sequential")
t_seq_n = @elapsed begin
    θ_seq_n, ll_seq_n = ReparamTools.profile_target(
        lnlike_θ, [1],  # Profile n
        θ_lower, θ_upper,
        [center[2]];  # Fix p at center
        grid_steps=500,
        use_distributed=false,
        optmaxtime=5.0
    )
end

println("    Time: $(round(t_seq_n, digits=2))s")
println("    Finite: ", sum(isfinite.(ll_seq_n)), "/", length(ll_seq_n))

# Sequential profile for p (fixing n at center)
println("\n  Profile 2 (p): Sequential")
t_seq_p = @elapsed begin
    θ_seq_p, ll_seq_p = ReparamTools.profile_target(
        lnlike_θ, [2],  # Profile p
        θ_lower, θ_upper,
        [center[1]];  # Fix n at center
        grid_steps=500,
        use_distributed=false,
        optmaxtime=5.0
    )
end

println("    Time: $(round(t_seq_p, digits=2))s")
println("    Finite: ", sum(isfinite.(ll_seq_p)), "/", length(ll_seq_p))

t_seq_total = t_seq_n + t_seq_p
println("\n  Total sequential time: $(round(t_seq_total, digits=2))s")

println("\n[4/6] Running CONCURRENT 1D profiles (worker pool partition)...")
println("  Partitioning 4 workers into 2 pools of 2")

# Partition workers
all_workers = workers()
pool1_ids = all_workers[1:2]
pool2_ids = all_workers[3:4]
pool1 = WorkerPool(pool1_ids)
pool2 = WorkerPool(pool2_ids)

println("  Pool 1 (n profile): workers ", pool1_ids)
println("  Pool 2 (p profile): workers ", pool2_ids)

println("\n  Running concurrent profiles with @sync/@async...")
t_concurrent = @elapsed begin
    concurrent_results = @sync begin
        task1 = @async begin
            ReparamTools.profile_target(
                lnlike_θ_worker, [1],  # Profile n
                θ_lower, θ_upper,
                [center[2]];
                grid_steps=500,
                use_distributed=true,
                n_chunks=length(pool1.workers),
                worker_pool=pool1,
                optmaxtime=5.0
            )
        end
        task2 = @async begin
            ReparamTools.profile_target(
                lnlike_θ_worker, [2],  # Profile p
                θ_lower, θ_upper,
                [center[1]];
                grid_steps=500,
                use_distributed=true,
                n_chunks=length(pool2.workers),
                worker_pool=pool2,
                optmaxtime=5.0
            )
        end
        (fetch(task1), fetch(task2))
    end
end

(θ_conc_n, ll_conc_n), (θ_conc_p, ll_conc_p) = concurrent_results

println("\n  Concurrent execution time: $(round(t_concurrent, digits=2))s")
println("  Profile 1 (n): ", sum(isfinite.(ll_conc_n)), "/", length(ll_conc_n), " finite")
println("  Profile 2 (p): ", sum(isfinite.(ll_conc_p)), "/", length(ll_conc_p), " finite")

# Compute speedup
speedup = t_seq_total / t_concurrent
println("\n  Speedup: $(round(speedup, digits=2))x")
println("  Efficiency: $(round(100*speedup/2, digits=1))% (of ideal 2x)")

println("\n[5/6] Comparing results...")

# Compare n profiles
diff_n = abs.(ll_seq_n - ll_conc_n)
max_diff_n = maximum(diff_n)
println("  Profile 1 (n): max diff = ", @sprintf("%.2e", max_diff_n))
println("    Sequential vs concurrent match: ", max_diff_n < 1e-10 ? "✓" : "✗")

# Compare p profiles
diff_p = abs.(ll_seq_p - ll_conc_p)
max_diff_p = maximum(diff_p)
println("  Profile 2 (p): max diff = ", @sprintf("%.2e", max_diff_p))
println("    Sequential vs concurrent match: ", max_diff_p < 1e-10 ? "✓" : "✗")

println("\n[6/6] Generating figures...")

# Extract parameter values for plotting
n_vals_seq = [θ[1] for θ in θ_seq_n]
n_vals_conc = [θ[1] for θ in θ_conc_n]
p_vals_seq = [θ[2] for θ in θ_seq_p]
p_vals_conc = [θ[2] for θ in θ_conc_p]

# Create plots
p1 = plot(n_vals_seq, ll_seq_n,
    label="Sequential",
    xlabel="n",
    ylabel="Log-likelihood",
    title="Profile: n (fixing p)",
    marker=:circle,
    linewidth=2,
    legend=:bottomright)
plot!(p1, n_vals_conc, ll_conc_n,
    label="Concurrent",
    marker=:x,
    linewidth=2,
    linestyle=:dash)
vline!(p1, [n_true], label="True n", linestyle=:dot, linewidth=2, color=:black)

p2 = plot(p_vals_seq, ll_seq_p,
    label="Sequential",
    xlabel="p",
    ylabel="Log-likelihood",
    title="Profile: p (fixing n)",
    marker=:circle,
    linewidth=2,
    legend=:bottomright)
plot!(p2, p_vals_conc, ll_conc_p,
    label="Concurrent",
    marker=:x,
    linewidth=2,
    linestyle=:dash)
vline!(p2, [p_true], label="True p", linestyle=:dot, linewidth=2, color=:black)

# Difference plots
p3 = plot(n_vals_seq, diff_n,
    label="",
    xlabel="n",
    ylabel="Absolute difference",
    title="Difference: n profile",
    marker=:circle,
    linewidth=2,
    yscale=:log10)

p4 = plot(p_vals_seq, diff_p,
    label="",
    xlabel="p",
    ylabel="Absolute difference",
    title="Difference: p profile",
    marker=:circle,
    linewidth=2,
    yscale=:log10)

# Combine plots
plot_combined = plot(p1, p2, p3, p4,
    layout=(2,2),
    size=(1200, 800),
    plot_title="Concurrent 1D Profiles: Sequential vs Concurrent ($(round(speedup, digits=2))x speedup)")

# Save figure
output_file = "stat_model_concurrent_profiles.png"
savefig(plot_combined, output_file)
println("✓ Figure saved: ", output_file)

# Final verdict
println("\n" * "="^70)
tolerance = 1e-8
if max_diff_n < tolerance && max_diff_p < tolerance
    println("✓✓✓ TEST PASSED ✓✓✓")
    println("Concurrent profiles match sequential within tolerance $(tolerance)")
    println("Speedup: $(round(speedup, digits=2))x (efficiency: $(round(100*speedup/2, digits=1))%)")
    exit_code = 0
else
    println("✗✗✗ TEST FAILED ✗✗✗")
    println("Maximum differences exceed tolerance:")
    println("  n profile: ", @sprintf("%.2e", max_diff_n))
    println("  p profile: ", @sprintf("%.2e", max_diff_p))
    exit_code = 1
end
println("="^70)

# Cleanup
rmprocs(workers())

exit(exit_code)
