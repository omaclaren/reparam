# Comprehensive test: Distributed profiling with autodiff ENABLED
using Distributed
addprocs(2)
println("✓ Added 2 workers: ", workers())

# Load modules
include("examples/RepressilatorModel.jl")
using .RepressilatorModel
include("ReparamTools.jl")
using .ReparamTools

@everywhere begin
    include($(joinpath(@__DIR__, "examples", "RepressilatorModel.jl")))
    using .RepressilatorModel
    include($(joinpath(@__DIR__, "ReparamTools.jl")))
    using .ReparamTools
    using Distributions
    using LinearAlgebra
end

# Setup
NT = 4
t_obs = LinRange(0.0, 5000.0, NT)
X0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
σ = 1.0

θ_true = [0.5, 0.5, 0.5, 10.0, 10.0, 10.0, 0.02, 0.01, 0.015,
          30.0, 26.0, 32.0, 0.35, 0.35, 0.35, 0.12, 0.12, 0.12]

println("\n=== Step 1: Verify autodiff works on master ===")
@time y_true = RepressilatorModel.predict_mRNA(θ_true, t_obs, X0)
println("✓ Master autodiff working, prediction size: ", length(y_true))

# Generate data
data = y_true + σ * randn(length(y_true))

# Send to workers
@everywhere t_obs_global = $(t_obs)
@everywhere X0_global = $(X0)
@everywhere σ_global = $(σ)
@everywhere data_global = $(data)
@everywhere NT_global = $(NT)

@everywhere function lnlike_θ(θ)
    try
        pred = RepressilatorModel.predict_mRNA(θ, t_obs_global, X0_global)
        dist = MvNormal(pred, σ_global^2 * I(length(pred)))
        return logpdf(dist, data_global)
    catch e
        println("Error in lnlike: ", e)
        return -Inf
    end
end

@everywhere lnlike_θ_log(θ_log) = lnlike_θ(exp.(θ_log))

println("\n=== Step 2: Verify likelihood at true parameters ===")
ll_true = lnlike_θ(θ_true)
println("Likelihood at true params: ", ll_true)
if isfinite(ll_true)
    println("✓ Likelihood is finite")
else
    println("✗ ERROR: Likelihood is not finite!")
    exit(1)
end

println("\n=== Step 3: Test worker evaluation ===")
θ_true_copy = copy(θ_true)
test_result = @spawnat 2 lnlike_θ(θ_true_copy)
ll_worker = fetch(test_result)
println("Likelihood on worker 2: ", ll_worker)
if isfinite(ll_worker) && abs(ll_worker - ll_true) < 1e-6
    println("✓ Worker evaluation matches master")
else
    println("✗ ERROR: Worker result differs or is invalid!")
    println("  Master: ", ll_true, ", Worker: ", ll_worker)
end

println("\n=== Step 4: Run tiny 2×2 distributed profile (β₁, K₁) ===")
θ_log_true = log.(θ_true)
θ_log_lower = θ_log_true .- 0.5
θ_log_upper = θ_log_true .+ 0.5
nuisance_guess = θ_log_true[setdiff(1:18, [7, 10])]

println("Running 2×2 grid with 15s optimization timeout per point...")
t_dist = @elapsed begin
    θ_vals, ll_vals = ReparamTools.profile_target(
        lnlike_θ_log, [7, 10],  # β₁, K₁
        θ_log_lower, θ_log_upper,
        nuisance_guess;
        grid_steps=2,
        use_distributed=true,
        n_chunks=2,
        optmaxtime=15.0
    )
end

println("\n=== Results ===")
println("Time: $(round(t_dist, digits=2))s")
println("Grid: ", length(θ_vals))
println("Likelihoods: ", ll_vals)
println("Max: ", maximum(ll_vals))
println("Min: ", minimum(ll_vals))

if all(isfinite.(ll_vals))
    println("\n✓✓✓ SUCCESS - All likelihoods finite!")
    println("✓ Distributed profiling works with autodiff enabled")
else
    println("\n✗ WARNING: Some likelihoods are NaN/Inf")
    println("  This may indicate optimization timeout too short")
end

rmprocs(workers())
