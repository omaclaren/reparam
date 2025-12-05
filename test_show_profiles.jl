#!/usr/bin/env julia
# Show actual 1D and 2D profile values

# Load on main process first
include("ReparamTools.jl")
include("examples/RepressilatorModel.jl")
using .ReparamTools
using .RepressilatorModel
using Random
using Statistics
using Distributions
using LinearAlgebra
using Printf

# Add workers
using Distributed
addprocs(4)

@everywhere begin
    include("ReparamTools.jl")
    include("examples/RepressilatorModel.jl")
    using .ReparamTools
    using .RepressilatorModel
    using Random
end

Random.seed!(42)

println("="^70)
println("Repressilator Profile Likelihood Analysis")
println("="^70)

# Setup (matching full workflow test)
T_end = 8000.0
NT = 6
t_obs = range(0, T_end, length=NT)
X0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
σ = 1.0

θ_true = [0.5, 0.5, 0.5,  # α₀
          10.0, 10.0, 10.0,  # α
          0.02, 0.01, 0.015,  # β
          30.0, 26.0, 32.0,  # K
          0.35, 0.35, 0.35,  # k_degm
          0.12, 0.12, 0.12]  # k_degp

# Generate data
y_true = RepressilatorModel.predict_mRNA(θ_true, t_obs, X0)
data = y_true + σ * randn(length(y_true))

println("\n[SETUP]")
println("  Timepoints: ", NT)
println("  Data points: ", length(data))
println("  True β₁ = ", θ_true[7])
println("  True K₁ = ", θ_true[10])

# Likelihood function
function lnlike_θ(θ)
    try
        pred = RepressilatorModel.predict_mRNA(θ, t_obs, X0)
        dist = MvNormal(pred, σ^2 * I(length(pred)))
        return logpdf(dist, data)
    catch
        return -Inf
    end
end

# Log-space likelihood
lnlike_θ_log = θ_log -> lnlike_θ(exp.(θ_log))

# Setup for distributed
@everywhere function lnlike_θ_log_worker(θ_log)
    try
        θ = exp.(θ_log)
        pred = RepressilatorModel.predict_mRNA(θ, $t_obs, $X0)
        dist = MvNormal(pred, $(σ^2) * I(length(pred)))
        return logpdf(dist, $data)
    catch
        return -Inf
    end
end

# Bounds
θ_log_true = log.(θ_true)
θ_log_lower = θ_log_true .- 2.0
θ_log_upper = θ_log_true .+ 2.0

println("\n[1/3] Running 1D profile for β₁ (index 7)...")
target_beta = [7]
nuisance_beta = setdiff(1:18, target_beta)
nuisance_guess_beta = θ_log_true[nuisance_beta]

θ_1d_beta, ll_1d_beta = ReparamTools.profile_target(
    lnlike_θ_log_worker, target_beta,
    θ_log_lower, θ_log_upper,
    nuisance_guess_beta;
    grid_steps=20,
    use_distributed=true,
    optmaxtime=30.0
)

beta_values = [exp(θ[7]) for θ in θ_1d_beta]

println("\n[2/3] Running 1D profile for K₁ (index 10)...")
target_K = [10]
nuisance_K = setdiff(1:18, target_K)
nuisance_guess_K = θ_log_true[nuisance_K]

θ_1d_K, ll_1d_K = ReparamTools.profile_target(
    lnlike_θ_log_worker, target_K,
    θ_log_lower, θ_log_upper,
    nuisance_guess_K;
    grid_steps=20,
    use_distributed=true,
    optmaxtime=30.0
)

K_values = [exp(θ[10]) for θ in θ_1d_K]

println("\n[3/3] Running 2D profile for (β₁, K₁)...")
target_2d = [7, 10]
nuisance_2d = setdiff(1:18, target_2d)
nuisance_guess_2d = θ_log_true[nuisance_2d]

θ_2d, ll_2d = ReparamTools.profile_target(
    lnlike_θ_log_worker, target_2d,
    θ_log_lower, θ_log_upper,
    nuisance_guess_2d;
    grid_steps=20,
    use_distributed=true,
    optmaxtime=30.0
)

println("\n" * "="^70)
println("1D PROFILE FOR β₁")
println("="^70)
println("β₁ value              Log-likelihood    Likelihood")
println("-"^70)
for i in 1:length(beta_values)
    println(@sprintf("%.6f              %+.4e     %.8f",
                     beta_values[i], ll_1d_beta[i], exp(ll_1d_beta[i])))
end

println("\n" * "="^70)
println("1D PROFILE FOR K₁")
println("="^70)
println("K₁ value              Log-likelihood    Likelihood")
println("-"^70)
for i in 1:length(K_values)
    println(@sprintf("%.6f              %+.4e     %.8f",
                     K_values[i], ll_1d_K[i], exp(ll_1d_K[i])))
end

println("\n" * "="^70)
println("2D PROFILE SUMMARY")
println("="^70)
ll_2d_grid = reshape(ll_2d, 20, 20)
println("Grid size: 20 × 20")
println("Log-likelihood range: ", extrema(ll_2d))
println("Likelihood range: ", extrema(exp.(ll_2d)))
println("\nFirst row (β₁ = ", @sprintf("%.6f", beta_values[1]), ", varying K₁):")
for j in 1:20
    println(@sprintf("  K₁=%.2f: ll=%.4e, like=%.8f",
                     K_values[j], ll_2d_grid[1,j], exp(ll_2d_grid[1,j])))
end

println("\nMiddle row (β₁ = ", @sprintf("%.6f", beta_values[10]), ", varying K₁):")
for j in 1:20
    println(@sprintf("  K₁=%.2f: ll=%.4e, like=%.8f",
                     K_values[j], ll_2d_grid[10,j], exp(ll_2d_grid[10,j])))
end

println("\n" * "="^70)
println("INTERPRETATION")
println("="^70)
ll_range = maximum(ll_2d) - minimum(ll_2d)
println("Log-likelihood range: ", @sprintf("%.4e", ll_range))
println("This is ", @sprintf("%.2e", ll_range), " away from zero")

if ll_range < 1e-10
    println("\n⚠️  FLAT PROFILE LIKELIHOOD!")
    println("All parameter combinations give essentially the same likelihood.")
    println("This means β₁ and K₁ are NOT IDENTIFIABLE from this data.")
    println("\nPossible reasons:")
    println("  - Too few timepoints (", NT, " observations)")
    println("  - Observing only mRNA (not proteins)")
    println("  - Parameters are structurally non-identifiable")
    println("  - Need longer time series or different observables")
else
    println("\nParameters show variation in likelihood.")
    println("95% confidence threshold: ", exp(-quantile(Chisq(2), 0.95)/2))
end

rmprocs(workers())
