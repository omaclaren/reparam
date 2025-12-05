#!/usr/bin/env julia
# Test: Compare 1D and 2D profile likelihoods for repressilator

# Load on main process first
include("ReparamTools.jl")
include("examples/RepressilatorModel.jl")
using .ReparamTools
using .RepressilatorModel
using Random

# Now add workers and load there too
using Distributed
addprocs(4)

@everywhere begin
    include("ReparamTools.jl")
    include("examples/RepressilatorModel.jl")
    using .ReparamTools
    using .RepressilatorModel
    using Random
end
Random.seed!(123)

println("="^70)
println("Comparing 1D and 2D Profile Likelihoods")
println("="^70)

# Generate data
t_obs = range(0, 8000, length=6)
θ_true = [0.02, 0.01, 0.015, 8.0, 8.0, 8.0, 0.02, 0.01, 0.015, 30.0, 26.0, 32.0, 0.025, 0.02, 0.03, 0.015, 0.01, 0.0125]
Y_data, _ = generate_data(θ_true, collect(t_obs), σ=1.0)

# Build likelihood
lnlike = construct_lnlike_repressilator(Y_data, collect(t_obs))

# Find MLE (cheat using true params as starting point)
println("\n[1/4] Finding MLE...")
θ_log_lower = log.(θ_true) .- 1.0
θ_log_upper = log.(θ_true) .+ 1.0
θ_log_MLE = log.(θ_true)  # Start from true

# Run 1D profile for β₁ (index 7)
println("\n[2/4] Running 1D profile for β₁...")
target_1d_beta = [7]
nuisance_1d_beta = setdiff(1:18, target_1d_beta)
nuisance_guess_beta = θ_log_MLE[nuisance_1d_beta]

θ_1d_beta, ll_1d_beta = profile_target(
    lnlike, target_1d_beta,
    θ_log_lower, θ_log_upper,
    nuisance_guess_beta;
    grid_steps=20,
    use_distributed=true,
    optmaxtime=30.0
)

println("  β₁ 1D profile:")
println("    Points: ", length(ll_1d_beta))
println("    Log-likelihood range: ", extrema(ll_1d_beta))
println("    Values: ", ll_1d_beta)

# Run 1D profile for K₁ (index 10)
println("\n[3/4] Running 1D profile for K₁...")
target_1d_K = [10]
nuisance_1d_K = setdiff(1:18, target_1d_K)
nuisance_guess_K = θ_log_MLE[nuisance_1d_K]

θ_1d_K, ll_1d_K = profile_target(
    lnlike, target_1d_K,
    θ_log_lower, θ_log_upper,
    nuisance_guess_K;
    grid_steps=20,
    use_distributed=true,
    optmaxtime=30.0
)

println("  K₁ 1D profile:")
println("    Points: ", length(ll_1d_K))
println("    Log-likelihood range: ", extrema(ll_1d_K))
println("    Values: ", ll_1d_K)

# Run 2D profile for (β₁, K₁)
println("\n[4/4] Running 2D profile for (β₁, K₁)...")
target_2d = [7, 10]
nuisance_2d = setdiff(1:18, target_2d)
nuisance_guess_2d = θ_log_MLE[nuisance_2d]

θ_2d, ll_2d = profile_target(
    lnlike, target_2d,
    θ_log_lower, θ_log_upper,
    nuisance_guess_2d;
    grid_steps=20,
    use_distributed=true,
    optmaxtime=30.0
)

println("  2D profile:")
println("    Points: ", length(ll_2d))
println("    Log-likelihood range: ", extrema(ll_2d))
println("    First 20 values: ", ll_2d[1:20])

# Reshape 2D profile to grid
ll_2d_grid = reshape(ll_2d, 20, 20)

# Extract β₁ profile from 2D (max over K₁ for each β₁)
ll_beta_from_2d = [maximum(ll_2d_grid[i, :]) for i in 1:20]

# Extract K₁ profile from 2D (max over β₁ for each K₁)
ll_K_from_2d = [maximum(ll_2d_grid[:, j]) for j in 1:20]

println("\n" * "="^70)
println("COMPARISON")
println("="^70)

println("\nβ₁ Profile:")
println("  1D direct:  ", ll_1d_beta)
println("  From 2D:    ", ll_beta_from_2d)
println("  Difference: ", abs.(ll_1d_beta .- ll_beta_from_2d))
println("  Max diff:   ", maximum(abs.(ll_1d_beta .- ll_beta_from_2d)))

println("\nK₁ Profile:")
println("  1D direct:  ", ll_1d_K)
println("  From 2D:    ", ll_K_from_2d)
println("  Difference: ", abs.(ll_1d_K .- ll_K_from_2d))
println("  Max diff:   ", maximum(abs.(ll_1d_K .- ll_K_from_2d)))

println("\n" * "="^70)
if maximum(abs.(ll_1d_beta .- ll_beta_from_2d)) < 1e-6 &&
   maximum(abs.(ll_1d_K .- ll_K_from_2d)) < 1e-6
    println("✓ PROFILES MATCH - 2D data is consistent!")
else
    println("✗ PROFILES DON'T MATCH - Something is wrong!")
    println("  This suggests the 2D profile data is incorrect.")
end
println("="^70)

rmprocs(workers())
