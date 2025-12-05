# Diagnostic Script: Investigate Flat Profile
# Checks if profile_point respects fixed parameters in sequential and distributed modes
# USING TIGHTER BOUNDS (from repressilator.jl)

using Distributed
using Printf

println("="^70)
println("DIAGNOSTIC: FLAT PROFILE INVESTIGATION (TIGHT BOUNDS)")
println("="^70)

# Add workers
if length(workers()) < 2
    addprocs(2)
end
println("✓ Workers: ", workers())

# Load modules locally first
include("examples/RepressilatorModel.jl")
using .RepressilatorModel
include("ReparamTools.jl")
using .ReparamTools
using Distributions
using LinearAlgebra
using Random

# Load modules on workers
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

# Model setup (same as production)
NT = 6
T_end = 8000.0
t_obs = LinRange(0, T_end, NT)
X0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
σ = 1.0

# True parameters
θ_true = [0.5, 0.5, 0.5, 10.0, 10.0, 10.0, 0.02, 0.01, 0.015, 30.0, 26.0, 32.0, 0.35, 0.35, 0.35, 0.12, 0.12, 0.12]
θ_log_true = log.(θ_true)

# Generate data
Random.seed!(42)
y_true = RepressilatorModel.predict_mRNA(θ_true, t_obs, X0)
data = y_true + σ * randn(length(y_true))

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

# MLE (approx)
ll_true = lnlike_θ_worker(θ_true)
println("Likelihood at true parameters: ", ll_true)

# Tighter bounds (from repressilator.jl)
θ_lower = similar(θ_true)
θ_upper = similar(θ_true)

# Basal transcription α₀ᵢ (indices 1-3): [0.005, 0.015]
θ_lower[1:3] .= 0.005; θ_upper[1:3] .= 0.015

# Regulated transcription αᵢ (indices 4-6): [0.8, 2.0]
θ_lower[4:6] .= 0.8; θ_upper[4:6] .= 2.0

# Translation βᵢ (indices 7-9): [0.01, 0.03]
θ_lower[7:9] .= 0.01; θ_upper[7:9] .= 0.03

# Repression threshold Kᵢ (indices 10-12): [20, 40]
θ_lower[10:12] .= 20.0; θ_upper[10:12] .= 40.0

# mRNA degradation k_degmᵢ (indices 13-15): [0.004, 0.008]
θ_lower[13:15] .= 0.004; θ_upper[13:15] .= 0.008

# Protein degradation k_degpᵢ (indices 16-18): [0.001, 0.0015]
θ_lower[16:18] .= 0.001; θ_upper[16:18] .= 0.0015

θ_log_lower = log.(θ_lower)
θ_log_upper = log.(θ_upper)

# Setup profiling
target_2d = [7, 10]
nuisance_2d = setdiff(1:18, target_2d)
nuisance_guess = θ_log_true[nuisance_2d]

# Fixed values for profile (log scale)
# Testing point: β₁=0.01, K₁=40.0 (Ratio 4000 vs True 1500)
ψ_fixed_log = [log(0.01), log(40.0)]
println("Testing point: β₁=0.01, K₁=40.0 (Ratio 4000 vs True 1500)")

println("\n[1/2] Running SEQUENTIAL profile_point...")
θ_opt_seq, _, ll_seq, _ = ReparamTools.profile_point(
    lnlike_θ_log_worker,
    ψ_fixed_log, target_2d,
    θ_log_lower, θ_log_upper,
    nuisance_guess;
    optmaxtime=5.0
)
println("Sequential Result: ", ll_seq)
println("  β₁ optimized: ", exp(θ_opt_seq[7]))
println("  K₁ optimized: ", exp(θ_opt_seq[10]))

if abs(ll_seq - ll_true) < 1.0
    println("  FAIL: Sequential optimized to MLE likelihood (ignored constraints?)")
else
    println("  PASS: Sequential likelihood is lower than MLE.")
end

# Cleanup
rmprocs(workers())
