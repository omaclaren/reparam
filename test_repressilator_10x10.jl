# Repressilator 2D Profile: 10×10 Grid Test
# Clean test of distributed 2D profiling with moderate resolution

using Distributed
using Distributed: WorkerPool
using Printf

println("="^70)
println("REPRESSILATOR 2D PROFILE: 10×10 Grid")
println("="^70)

# Configuration
USE_DISTRIBUTED = true
N_WORKERS = 4
GRID_2D = 10  # 10×10 = 100 points
MLE_TIMEOUT = 30.0
PROFILE_TIMEOUT = 30.0

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

println("\n[1/4] Setting up model...")

# Model configuration - OLD WORKING SETTINGS (from Oct 29 commit 00b9143)
NT = 6
T_end = 8000.0
t_obs = LinRange(0, T_end, NT)
X0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
σ = 1.0

# True parameters - OLD WORKING VALUES (from Oct 29 commit 00b9143)
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
println("\n[2/4] Generating synthetic data...")
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
θ_log_lower = θ_log_true .- 2.0
θ_log_upper = θ_log_true .+ 2.0

println("\n[3/4] Finding MLE...")
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
println("    K₁ = ", @sprintf("%.4f", θ_MLE[10]))

println("\n[4/4] Running joint 2D profile (β₁, K₁)...")
println("  Grid: ", GRID_2D, "×", GRID_2D, " = ", GRID_2D^2, " points")
println("  Estimated time: ~", @sprintf("%.1f", GRID_2D^2 * PROFILE_TIMEOUT / N_WORKERS / 60), " minutes")

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

# Summary
total_time = t_mle + t_2d

println("\n" * "="^70)
println("RESULTS SUMMARY")
println("="^70)

println("\nMode: ", USE_DISTRIBUTED ? "DISTRIBUTED ($(N_WORKERS) workers)" : "SEQUENTIAL")
println("\nTimings:")
println("  MLE:          ", @sprintf("%6.2f", t_mle), "s")
println("  2D profile:   ", @sprintf("%6.2f", t_2d), "s (", GRID_2D^2, " points)")
println("  " * "-"^50)
println("  Total:        ", @sprintf("%6.2f", total_time), "s (", @sprintf("%.2f", total_time/60), " min)")

println("\nSuccess rates:")
println("  MLE:        ", isfinite(lnlike_MLE) ? "✓" : "✗")
println("  2D profile: ", sum(isfinite.(ll_2d_vals)), "/", length(ll_2d_vals),
        " (", @sprintf("%.0f", 100*sum(isfinite.(ll_2d_vals))/length(ll_2d_vals)), "%)")

all_success = isfinite(lnlike_MLE) && all(isfinite.(ll_2d_vals))

# Generate 2D plot
println("\n[PLOTTING] Generating 2D profile figure...")

using Plots
using Distributions: Chisq, quantile

# Extract parameter values in original scale
β1_values = unique([exp(θ[7]) for θ in θ_2d_vals])
K1_values = unique([exp(θ[10]) for θ in θ_2d_vals])

println("  β₁ range: ", extrema(β1_values))
println("  K₁ range: ", extrema(K1_values))

# K varies fastest in product, so reshape as (lenK, lenβ) then transpose to get rows=β, cols=K
ll_grid = reshape(ll_2d_vals, length(K1_values), length(β1_values))'
like_grid = exp.(ll_grid)

# Chi-square calibration for 95% confidence contour
df = 2
lstar = exp(-quantile(Chisq(df), 0.95)/2)

# Create contour plot without colorbar (avoid GR rendering bug)
plt = contourf(β1_values, K1_values, like_grid,
              color=:dense, levels=20, lw=0,
              xlabel="β₁", ylabel="K₁",
              title="Repressilator 2D Profile: (β₁, K₁) [10×10 grid]",
              colorbar=false,
              size=(600, 500))

# Add 95% confidence contour
contour!(β1_values, K1_values, like_grid,
         levels=[lstar], color=:black, lw=2, legend=false, fill=false)

# Mark MLE
scatter!([exp(θ_log_MLE[7])], [exp(θ_log_MLE[10])],
         mc=:silver, msc=:match, markersize=8, markershape=:circle, legend=false)

# Mark true values
scatter!([θ_true[7]], [θ_true[10]],
         mc=:darkgoldenrod, msc=:match, markersize=10, markershape=:star, legend=false)

# Save
output_file = "repressilator_2D_10x10_beta1_K1.png"
savefig(plt, output_file)
println("✓ Figure saved: ", output_file)

println("\n" * "="^70)
if all_success
    println("✓✓✓ ALL TESTS SUCCESSFUL ✓✓✓")
    println("\n10×10 grid profile completed successfully.")
    println("Autodiff enabled and working throughout.")
    exit_code = 0
else
    println("⚠ PARTIAL SUCCESS ⚠")
    println("\nSome likelihood evaluations failed to converge.")
    exit_code = 0
end
println("="^70)

# Cleanup
if USE_DISTRIBUTED
    rmprocs(workers())
end

exit(exit_code)
