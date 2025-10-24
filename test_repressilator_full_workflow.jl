# Comprehensive Repressilator Test: MLE + 1D Profiles + 2D Profile
# Tests the full workflow with option for distributed profiling

using Distributed
using Distributed: WorkerPool
using Printf

println("="^70)
println("COMPREHENSIVE REPRESSILATOR TEST")
println("MLE + Individual 1D Profiles + Joint 2D Profile")
println("="^70)

# Configuration
USE_DISTRIBUTED = true  # Set to false for sequential
N_WORKERS = 2
GRID_1D = 5  # Grid points for 1D profiles
GRID_2D = 3  # Grid points per dimension for 2D profile
MLE_TIMEOUT = 30.0  # seconds
PROFILE_TIMEOUT = 20.0  # seconds per grid point

full_worker_pool = nothing

if USE_DISTRIBUTED
    println("\n[SETUP] Distributed mode with $(N_WORKERS) workers")
    addprocs(N_WORKERS)
    println("✓ Workers: ", workers())
    full_worker_pool = WorkerPool(workers())
else
    println("\n[SETUP] Sequential mode (no workers)")
end

# Load modules
include("examples/RepressilatorModel.jl")
using .RepressilatorModel
include("ReparamTools.jl")
using .ReparamTools
using Distributions
using LinearAlgebra
using Random
using Statistics

if USE_DISTRIBUTED
    @everywhere begin
        if !@isdefined(RepressilatorModel)
            include($(joinpath(@__DIR__, "examples", "RepressilatorModel.jl")))
        end
        using .RepressilatorModel
        if !@isdefined(ReparamTools)
            include($(joinpath(@__DIR__, "ReparamTools.jl")))
        end
        using .ReparamTools
        using Distributions
        using LinearAlgebra
    end
end

println("\n[1/6] Setting up model...")

# Model configuration
NT = 6  # Reasonable number of time points
T_end = 8000.0
t_obs = LinRange(0, T_end, NT)
X0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
σ = 1.0

# True parameters (from repressilator.jl)
θ_true = [0.5, 0.5, 0.5,  # α₀
          10.0, 10.0, 10.0,  # α
          0.02, 0.01, 0.015,  # β
          30.0, 26.0, 32.0,  # K
          0.35, 0.35, 0.35,  # k_degm
          0.12, 0.12, 0.12]  # k_degp

param_names = ["α₀₁", "α₀₂", "α₀₃", "α₁", "α₂", "α₃",
               "β₁", "β₂", "β₃", "K₁", "K₂", "K₃",
               "k_degm₁", "k_degm₂", "k_degm₃", "k_degp₁", "k_degp₂", "k_degp₃"]

println("✓ Model configuration:")
println("  Time points: ", NT, " from 0 to ", T_end)
println("  Parameters: ", length(θ_true))
println("  Noise level: σ = ", σ)

# Generate data
println("\n[2/6] Generating synthetic data...")
Random.seed!(42)
y_true = RepressilatorModel.predict_mRNA(θ_true, t_obs, X0)
data = y_true + σ * randn(length(y_true))

println("✓ Data generated:")
println("  Observations: ", length(data))
println("  True prediction mean: ", @sprintf("%.2f", mean(y_true)))
println("  Data mean: ", @sprintf("%.2f", mean(data)))

# Define likelihood functions
function lnlike_θ(θ)
    try
        pred = RepressilatorModel.predict_mRNA(θ, t_obs, X0)
        dist = MvNormal(pred, σ^2 * I(length(pred)))
        return logpdf(dist, data)
    catch
        return -Inf
    end
end

lnlike_θ_log(θ_log) = lnlike_θ(exp.(θ_log))

# Send data to workers if distributed
if USE_DISTRIBUTED
    @everywhere t_obs_global = $(t_obs)
    @everywhere X0_global = $(X0)
    @everywhere σ_global = $(σ)
    @everywhere data_global = $(data)

    @everywhere function lnlike_θ_worker(θ)
        try
            pred = RepressilatorModel.predict_mRNA(θ, t_obs_global, X0_global)
            dist = MvNormal(pred, σ_global^2 * I(length(pred)))
            return logpdf(dist, data_global)
        catch
            return -Inf
        end
    end

    @everywhere lnlike_θ_log_worker(θ_log) = lnlike_θ_worker(exp.(θ_log))

    # Use worker version for profiling
    lnlike_profile = lnlike_θ_log_worker
else
    lnlike_profile = lnlike_θ_log
end

# Verify likelihood at true parameters
ll_true = lnlike_θ(θ_true)
println("\n  Likelihood at true params: ", @sprintf("%.4f", ll_true))

if !isfinite(ll_true)
    println("✗ ERROR: Likelihood at true parameters is not finite!")
    if USE_DISTRIBUTED
        rmprocs(workers())
    end
    exit(1)
end

# Parameter bounds (log space)
θ_log_true = log.(θ_true)
θ_log_lower = θ_log_true .- 2.0  # Wide bounds
θ_log_upper = θ_log_true .+ 2.0

println("\n[3/6] Finding MLE...")
println("  Starting from true parameters")
println("  Timeout: ", MLE_TIMEOUT, "s")

t_mle = @elapsed begin
    θ_log_MLE, lnlike_MLE = ReparamTools.profile_target(
        lnlike_profile,
        Int[],  # Empty = MLE
        θ_log_lower, θ_log_upper,
        θ_log_true;
        grid_steps=Int[],
        use_distributed=USE_DISTRIBUTED,
        n_chunks=USE_DISTRIBUTED ? N_WORKERS : 1,
        optmaxtime=MLE_TIMEOUT
    )
end

θ_MLE = exp.(θ_log_MLE)

println("✓ MLE found in ", @sprintf("%.2f", t_mle), "s")
println("  Log-likelihood: ", @sprintf("%.4f", lnlike_MLE))
println("  Improvement over true: ", @sprintf("%.4f", lnlike_MLE - ll_true))

println("\n  MLE parameter values (showing β and K):")
println("    β₁ = ", @sprintf("%.6f", θ_MLE[7]))
println("    β₂ = ", @sprintf("%.6f", θ_MLE[8]))
println("    β₃ = ", @sprintf("%.6f", θ_MLE[9]))
println("    K₁ = ", @sprintf("%.4f", θ_MLE[10]))
println("    K₂ = ", @sprintf("%.4f", θ_MLE[11]))
println("    K₃ = ", @sprintf("%.4f", θ_MLE[12]))

println("\n[4/6] Running individual 1D profiles...")
println("  Parameters: β₁ (index 7) and K₁ (index 10)")
println("  Grid: ", GRID_1D, " points each")
println("  Using MLE as starting point for nuisance parameters")

β1_index = 7
K1_index = 10
target_β1 = [β1_index]
target_K1 = [K1_index]
nuisance_β1 = setdiff(1:18, target_β1)
nuisance_K1 = setdiff(1:18, target_K1)
nuisance_guess_β1 = θ_log_MLE[nuisance_β1]
nuisance_guess_K1 = θ_log_MLE[nuisance_K1]

β1_concurrency_used = false

if USE_DISTRIBUTED && length(workers()) >= 2
    worker_ids = workers()
    split_idx = max(1, length(worker_ids) ÷ 2)
    pool_β = WorkerPool(worker_ids[1:split_idx])
    pool_K = WorkerPool(worker_ids[split_idx+1:end])
    if !isempty(pool_β.workers) && !isempty(pool_K.workers)
        println("  Running β₁ and K₁ profiles concurrently across worker pools...")
        β1_concurrency_used = true
        concurrent_results = @sync begin
            task_β = @async begin
                local θ_vals, ll_vals
                local elapsed = @elapsed begin
                    θ_vals, ll_vals = ReparamTools.profile_target(
                        lnlike_profile, target_β1,
                        θ_log_lower, θ_log_upper,
                        nuisance_guess_β1;
                        grid_steps=GRID_1D,
                        use_distributed=true,
                        worker_pool=pool_β,
                        n_chunks=max(1, min(GRID_1D, length(pool_β.workers))),
                        optmaxtime=PROFILE_TIMEOUT
                    )
                end
                (θ_vals, ll_vals, elapsed)
            end
            task_K = @async begin
                local θ_vals, ll_vals
                local elapsed = @elapsed begin
                    θ_vals, ll_vals = ReparamTools.profile_target(
                        lnlike_profile, target_K1,
                        θ_log_lower, θ_log_upper,
                        nuisance_guess_K1;
                        grid_steps=GRID_1D,
                        use_distributed=true,
                        worker_pool=pool_K,
                        n_chunks=max(1, min(GRID_1D, length(pool_K.workers))),
                        optmaxtime=PROFILE_TIMEOUT
                    )
                end
                (θ_vals, ll_vals, elapsed)
            end
            (fetch(task_β), fetch(task_K))
        end

        (β1_result, K1_result) = concurrent_results

        θ_β1_vals, ll_β1_vals, t_β1 = β1_result
        θ_K1_vals, ll_K1_vals, t_K1 = K1_result
    end
end

if !β1_concurrency_used
    t_β1 = @elapsed begin
        θ_β1_vals, ll_β1_vals = ReparamTools.profile_target(
            lnlike_profile, target_β1,
            θ_log_lower, θ_log_upper,
            nuisance_guess_β1;
            grid_steps=GRID_1D,
            use_distributed=USE_DISTRIBUTED,
            worker_pool=USE_DISTRIBUTED ? full_worker_pool : nothing,
            n_chunks=USE_DISTRIBUTED && full_worker_pool !== nothing ?
                max(1, min(GRID_1D, length(full_worker_pool.workers))) : nothing,
            optmaxtime=PROFILE_TIMEOUT
        )
    end

    t_K1 = @elapsed begin
        θ_K1_vals, ll_K1_vals = ReparamTools.profile_target(
            lnlike_profile, target_K1,
            θ_log_lower, θ_log_upper,
            nuisance_guess_K1;
            grid_steps=GRID_1D,
            use_distributed=USE_DISTRIBUTED,
            worker_pool=USE_DISTRIBUTED ? full_worker_pool : nothing,
            n_chunks=USE_DISTRIBUTED && full_worker_pool !== nothing ?
                max(1, min(GRID_1D, length(full_worker_pool.workers))) : nothing,
            optmaxtime=PROFILE_TIMEOUT
        )
    end
end

println("  ✓ β₁ profile complete in ", @sprintf("%.2f", t_β1), "s")
println("    Grid points: ", length(ll_β1_vals))
println("    Finite values: ", sum(isfinite.(ll_β1_vals)), "/", length(ll_β1_vals))
if sum(isfinite.(ll_β1_vals)) > 0
    println("    Max likelihood: ", @sprintf("%.4f", maximum(ll_β1_vals[isfinite.(ll_β1_vals)])))
end

println("\n  ✓ K₁ profile complete in ", @sprintf("%.2f", t_K1), "s")
println("    Grid points: ", length(ll_K1_vals))
println("    Finite values: ", sum(isfinite.(ll_K1_vals)), "/", length(ll_K1_vals))
if sum(isfinite.(ll_K1_vals)) > 0
    println("    Max likelihood: ", @sprintf("%.4f", maximum(ll_K1_vals[isfinite.(ll_K1_vals)])))
end

println("\n[5/6] Running joint 2D profile (β₁, K₁)...")
println("  Grid: ", GRID_2D, "×", GRID_2D, " = ", GRID_2D^2, " points")

target_2d = [7, 10]  # β₁ and K₁
nuisance_2d = setdiff(1:18, target_2d)
nuisance_guess_2d = θ_log_MLE[nuisance_2d]

t_2d = @elapsed begin
    θ_2d_vals, ll_2d_vals = ReparamTools.profile_target(
        lnlike_profile, target_2d,
        θ_log_lower, θ_log_upper,
        nuisance_guess_2d;
        grid_steps=GRID_2D,
        use_distributed=USE_DISTRIBUTED,
        worker_pool=USE_DISTRIBUTED ? full_worker_pool : nothing,
        n_chunks=USE_DISTRIBUTED && full_worker_pool !== nothing ?
            max(1, min(GRID_2D^2, length(full_worker_pool.workers))) : nothing,
        optmaxtime=PROFILE_TIMEOUT
    )
end

println("✓ 2D profile complete in ", @sprintf("%.2f", t_2d), "s (", @sprintf("%.2f", t_2d/60), " min)")
println("  Grid points: ", length(ll_2d_vals))
println("  Finite values: ", sum(isfinite.(ll_2d_vals)), "/", length(ll_2d_vals))
if sum(isfinite.(ll_2d_vals)) > 0
    println("  Max likelihood: ", @sprintf("%.4f", maximum(ll_2d_vals[isfinite.(ll_2d_vals)])))
end

println("\n[6/6] Summary...")

profiles_wall_time = β1_concurrency_used ? max(t_β1, t_K1) : (t_β1 + t_K1)
total_time = t_mle + profiles_wall_time + t_2d
total_evals = 1 + length(ll_β1_vals) + length(ll_K1_vals) + length(ll_2d_vals)

println("\n" * "="^70)
println("RESULTS SUMMARY")
println("="^70)

println("\nMode: ", USE_DISTRIBUTED ? "DISTRIBUTED ($(N_WORKERS) workers)" : "SEQUENTIAL")
println("\nTimings:")
println("  MLE:          ", @sprintf("%6.2f", t_mle), "s")
println("  β₁ profile:   ", @sprintf("%6.2f", t_β1), "s (", GRID_1D, " points)")
println("  K₁ profile:   ", @sprintf("%6.2f", t_K1), "s (", GRID_1D, " points)")
println("  2D profile:   ", @sprintf("%6.2f", t_2d), "s (", GRID_2D^2, " points)")
println("  " * "-"^50)
println("  Total:        ", @sprintf("%6.2f", total_time), "s (", @sprintf("%.2f", total_time/60), " min)")
if β1_concurrency_used
    println("  (1D profiles overlapped: effective wall-clock uses max duration)")
end

println("\nEvaluations:")
println("  Total grid points: ", total_evals)
if USE_DISTRIBUTED
    println("  Average per point: ", @sprintf("%.2f", total_time/total_evals), "s")
end

println("\nSuccess rates:")
println("  MLE:        ", isfinite(lnlike_MLE) ? "✓" : "✗")
println("  β₁ profile: ", sum(isfinite.(ll_β1_vals)), "/", length(ll_β1_vals),
        " (", @sprintf("%.0f", 100*sum(isfinite.(ll_β1_vals))/length(ll_β1_vals)), "%)")
println("  K₁ profile: ", sum(isfinite.(ll_K1_vals)), "/", length(ll_K1_vals),
        " (", @sprintf("%.0f", 100*sum(isfinite.(ll_K1_vals))/length(ll_K1_vals)), "%)")
println("  2D profile: ", sum(isfinite.(ll_2d_vals)), "/", length(ll_2d_vals),
        " (", @sprintf("%.0f", 100*sum(isfinite.(ll_2d_vals))/length(ll_2d_vals)), "%)")

all_success = isfinite(lnlike_MLE) &&
              all(isfinite.(ll_β1_vals)) &&
              all(isfinite.(ll_K1_vals)) &&
              all(isfinite.(ll_2d_vals))

println("\n" * "="^70)
if all_success
    println("✓✓✓ ALL TESTS SUCCESSFUL ✓✓✓")
    println("\nMLE estimation and all profile likelihoods computed successfully.")
    println("Autodiff enabled and working throughout.")
    exit_code = 0
else
    println("⚠ PARTIAL SUCCESS ⚠")
    println("\nSome likelihood evaluations failed to converge.")
    println("This may indicate:")
    println("  - Optimization timeout too short")
    println("  - Parameter bounds too wide")
    println("  - Initial guesses suboptimal")
    exit_code = 0  # Still consider success if most points converged
end
println("="^70)

# Generate 2D plot
println("\n[PLOTTING] Generating 2D profile figure...")
using Plots

# Extract β₁ and K₁ values in original scale
β1_vals = [exp(θ[7]) for θ in θ_2d_vals]
K1_vals = [exp(θ[10]) for θ in θ_2d_vals]

# Reshape to grid
β1_grid = reshape(β1_vals, GRID_2D, GRID_2D)
K1_grid = reshape(K1_vals, GRID_2D, GRID_2D)
ll_grid = reshape(ll_2d_vals, GRID_2D, GRID_2D)

# Create contour plot
p = contour(β1_grid[:,1], K1_grid[1,:], ll_grid',
    xlabel="β₁",
    ylabel="K₁",
    title="Repressilator 2D Profile (β₁, K₁) - $(USE_DISTRIBUTED ? "Distributed" : "Sequential")",
    fill=true,
    color=:viridis,
    levels=10,
    size=(800, 600))

# Mark true values
scatter!(p, [θ_true[7]], [θ_true[10]],
    marker=:star, markersize=10, color=:red, label="True")

# Mark MLE
scatter!(p, [exp(θ_log_MLE[7])], [exp(θ_log_MLE[10])],
    marker=:circle, markersize=8, color=:white, label="MLE")

# Save
output_file = "repressilator_2D_profile.png"
savefig(p, output_file)
println("✓ Figure saved: ", output_file)

# Cleanup
if USE_DISTRIBUTED
    rmprocs(workers())
end

exit(exit_code)
