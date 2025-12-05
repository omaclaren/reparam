# Test stat_model: Full Workflow with Distributed Profiling
# Runs complete workflow: MLE + concurrent 1D profiles with worker pool partitioning
# Same settings as original stat_model.jl (grid_steps=500)

using Distributed

println("="^70)
println("TEST: stat_model Full Workflow with Distributed Profiling")
println("="^70)

# Add workers
println("\n[1/7] Setting up workers...")
addprocs(4)  # 4 workers for 2 pools of 2
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

println("\n[2/7] Setting up stat_model (Poisson limit)...")
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

# Initial guess
θ_initial = [mean(data_fixed), 0.5]

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
println("  Initial guess: n=$(round(θ_initial[1], digits=2)), p=$(θ_initial[2])")

# Verify likelihood at true parameters
ll_true = lnlike_θ(θ_true)
println("\n  Likelihood at true params: ", round(ll_true, digits=3))

println("\n[3/7] Finding MLE (sequential)...")
t_mle = @elapsed begin
    θ_MLE, ll_MLE = ReparamTools.profile_target(
        lnlike_θ, [],  # Empty target_indices = MLE
        θ_lower, θ_upper,
        θ_initial;
        grid_steps=500,
        use_distributed=false,
        optmaxtime=10.0
    )
end

println("✓ MLE found")
println("  Time: $(round(t_mle, digits=2))s")
println("  MLE: n=$(round(θ_MLE[1], digits=3)), p=$(round(θ_MLE[2], digits=3))")
println("  Log-likelihood: ", round(ll_MLE, digits=3))

println("\n[4/7] Running SEQUENTIAL 1D profiles...")
println("  Grid: 500 points per parameter (same as stat_model.jl)")

# Sequential profile for n (fixing p at MLE)
println("\n  Profile 1 (n): Sequential")
t_seq_n = @elapsed begin
    θ_seq_n, ll_seq_n = ReparamTools.profile_target(
        lnlike_θ, [1],  # Profile n
        θ_lower, θ_upper,
        [θ_MLE[2]];  # Fix p at MLE value
        grid_steps=500,
        use_distributed=false,
        optmaxtime=5.0
    )
end

println("    Time: $(round(t_seq_n, digits=2))s")
println("    Finite: ", sum(isfinite.(ll_seq_n)), "/", length(ll_seq_n))
println("    Max likelihood: ", round(maximum(ll_seq_n[isfinite.(ll_seq_n)]), digits=3))

# Sequential profile for p (fixing n at MLE)
println("\n  Profile 2 (p): Sequential")
t_seq_p = @elapsed begin
    θ_seq_p, ll_seq_p = ReparamTools.profile_target(
        lnlike_θ, [2],  # Profile p
        θ_lower, θ_upper,
        [θ_MLE[1]];  # Fix n at MLE value
        grid_steps=500,
        use_distributed=false,
        optmaxtime=5.0
    )
end

println("    Time: $(round(t_seq_p, digits=2))s")
println("    Finite: ", sum(isfinite.(ll_seq_p)), "/", length(ll_seq_p))
println("    Max likelihood: ", round(maximum(ll_seq_p[isfinite.(ll_seq_p)]), digits=3))

t_seq_total = t_seq_n + t_seq_p
println("\n  Total sequential profile time: $(round(t_seq_total, digits=2))s")
println("  Total workflow time (sequential): $(round(t_mle + t_seq_total, digits=2))s")

println("\n[5/7] Running CONCURRENT 1D profiles (worker pool partition)...")
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
                [θ_MLE[2]];  # Fix p at MLE value
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
                [θ_MLE[1]];  # Fix n at MLE value
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
println("\n  Total workflow time (concurrent): $(round(t_mle + t_concurrent, digits=2))s")

# Compute speedup
speedup = t_seq_total / t_concurrent
workflow_speedup = (t_mle + t_seq_total) / (t_mle + t_concurrent)
println("\n  Profile speedup: $(round(speedup, digits=2))x")
println("  Workflow speedup: $(round(workflow_speedup, digits=2))x")
println("  Profile efficiency: $(round(100*speedup/2, digits=1))% (of ideal 2x)")

println("\n[6/7] Comparing results...")

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

println("\n[7/7] Generating figures...")

# Extract parameter values for plotting
n_vals_seq = [θ[1] for θ in θ_seq_n]
n_vals_conc = [θ[1] for θ in θ_conc_n]
p_vals_seq = [θ[2] for θ in θ_seq_p]
p_vals_conc = [θ[2] for θ in θ_conc_p]

# Create profile plots
p1 = plot(n_vals_seq, ll_seq_n,
    label="Sequential",
    xlabel="n",
    ylabel="Log-likelihood",
    title="Profile: n (fixing p at MLE)",
    marker=:circle,
    markersize=2,
    linewidth=2,
    legend=:bottomright)
plot!(p1, n_vals_conc, ll_conc_n,
    label="Concurrent",
    marker=:x,
    markersize=2,
    linewidth=2,
    linestyle=:dash)
vline!(p1, [n_true], label="True n", linestyle=:dot, linewidth=2, color=:black)
vline!(p1, [θ_MLE[1]], label="MLE n", linestyle=:dash, linewidth=2, color=:green)

p2 = plot(p_vals_seq, ll_seq_p,
    label="Sequential",
    xlabel="p",
    ylabel="Log-likelihood",
    title="Profile: p (fixing n at MLE)",
    marker=:circle,
    markersize=2,
    linewidth=2,
    legend=:bottomright)
plot!(p2, p_vals_conc, ll_conc_p,
    label="Concurrent",
    marker=:x,
    markersize=2,
    linewidth=2,
    linestyle=:dash)
vline!(p2, [p_true], label="True p", linestyle=:dot, linewidth=2, color=:black)
vline!(p2, [θ_MLE[2]], label="MLE p", linestyle=:dash, linewidth=2, color=:green)

# Difference plots
p3 = plot(n_vals_seq, diff_n,
    label="",
    xlabel="n",
    ylabel="Absolute difference",
    title="Difference: n profile",
    marker=:circle,
    markersize=2,
    linewidth=2,
    yscale=:log10,
    ylims=(1e-16, 1e-13))

p4 = plot(p_vals_seq, diff_p,
    label="",
    xlabel="p",
    ylabel="Absolute difference",
    title="Difference: p profile",
    marker=:circle,
    markersize=2,
    linewidth=2,
    yscale=:log10,
    ylims=(1e-16, 1e-13))

# Combine plots
plot_combined = plot(p1, p2, p3, p4,
    layout=(2,2),
    size=(1200, 800),
    plot_title="Full Workflow: MLE + Concurrent 1D Profiles ($(round(workflow_speedup, digits=2))x workflow speedup)")

# Save figure
output_file = "stat_model_full_workflow_distributed.png"
savefig(plot_combined, output_file)
println("✓ Figure saved: ", output_file)

# Summary table
println("\n" * "="^70)
println("WORKFLOW SUMMARY")
println("="^70)
println(@sprintf("%-30s %10s %10s", "Step", "Sequential", "Concurrent"))
println("-"^70)
println(@sprintf("%-30s %9.2fs %9.2fs", "MLE", t_mle, t_mle))
println(@sprintf("%-30s %9.2fs %9s", "Profile n", t_seq_n, "-"))
println(@sprintf("%-30s %9.2fs %9s", "Profile p", t_seq_p, "-"))
println(@sprintf("%-30s %9.2fs %9.2fs", "Both profiles (concurrent)", t_seq_total, t_concurrent))
println("-"^70)
println(@sprintf("%-30s %9.2fs %9.2fs", "TOTAL WORKFLOW", t_mle + t_seq_total, t_mle + t_concurrent))
println("="^70)
println(@sprintf("Profile speedup: %.2fx (%.1f%% efficiency)", speedup, 100*speedup/2))
println(@sprintf("Workflow speedup: %.2fx", workflow_speedup))
println("="^70)

# Final verdict
println("\n" * "="^70)
tolerance = 1e-8
if max_diff_n < tolerance && max_diff_p < tolerance
    println("✓✓✓ TEST PASSED ✓✓✓")
    println("Full workflow with concurrent profiles validated!")
    println("  - Numerical accuracy: machine precision")
    println("  - Profile speedup: $(round(speedup, digits=2))x")
    println("  - Workflow speedup: $(round(workflow_speedup, digits=2))x")
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
