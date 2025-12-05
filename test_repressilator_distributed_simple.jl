# Simple test of distributed profiling on repressilator model
# Uses RepressilatorModel module

using Distributed

# Add workers
addprocs(3)
println("✓ Added 3 workers: ", workers())

# Load RepressilatorModel and ReparamTools on master
include("examples/RepressilatorModel.jl")
using .RepressilatorModel
include("ReparamTools.jl")
using .ReparamTools
using Distributions
using LinearAlgebra
using Random

# Load on workers
@everywhere begin
    include($(joinpath(@__DIR__, "examples", "RepressilatorModel.jl")))
    using .RepressilatorModel
    include($(joinpath(@__DIR__, "ReparamTools.jl")))
    using .ReparamTools
    using Distributions
    using LinearAlgebra
end

println("\n=== Setting up repressilator model ===")

# Model configuration
Random.seed!(1234)
NT = 8
T_end = 10000.0
t_obs = LinRange(0, T_end, NT)
X0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
σ = 1.0

# True parameters (from repressilator.jl)
θ_true = [
    0.5, 0.5, 0.5,           # α₀₁, α₀₂, α₀₃
    10.0, 10.0, 10.0,        # α₁, α₂, α₃
    0.02, 0.01, 0.015,       # β₁, β₂, β₃
    30.0, 26.0, 32.0,        # K₁, K₂, K₃
    0.35, 0.35, 0.35,        # k_degm
    0.12, 0.12, 0.12         # k_degp
]

# Generate data
y_true = RepressilatorModel.predict_mRNA(θ_true, t_obs, X0)
N_obs = length(y_true)
data = y_true + σ * randn(N_obs)

println("  Time points: $NT")
println("  Observations: $N_obs")
println("  Noise: σ = $σ")

# Send data to workers
@everywhere t_obs_global = $(t_obs)
@everywhere X0_global = $(X0)
@everywhere σ_global = $(σ)
@everywhere data_global = $(data)
@everywhere NT_global = $(NT)

# Define likelihood on all workers
@everywhere function lnlike_θ(θ)
    try
        pred = RepressilatorModel.predict_mRNA(θ, t_obs_global, X0_global)
        dist = MvNormal(pred, σ_global^2 * I(3*NT_global))
        return sum(logpdf(dist, d) for d in data_global)
    catch
        return -Inf
    end
end

# Likelihood in log space (also on all workers)
@everywhere lnlike_θ_log(θ_log) = lnlike_θ(exp.(θ_log))

# Parameter bounds (log space)
θ_log_lower = log.([
    0.01, 0.01, 0.01,     # α₀
    0.1, 0.1, 0.1,        # α
    0.001, 0.001, 0.001,  # β
    1.0, 1.0, 1.0,        # K
    0.01, 0.01, 0.01,     # k_degm
    0.01, 0.01, 0.01      # k_degp
])

θ_log_upper = log.([
    10.0, 10.0, 10.0,     # α₀
    100.0, 100.0, 100.0,  # α
    1.0, 1.0, 1.0,        # β
    100.0, 100.0, 100.0,  # K
    10.0, 10.0, 10.0,     # k_degm
    10.0, 10.0, 10.0      # k_degp
])

println("\n=== Testing 2D profile: (β₁, K₁) ===")
println("Grid: 3×3 = 9 points (test mode)")

# Parameters to profile (β₁=7, K₁=10)
β1_index = 7
K1_index = 10
target_indices = [β1_index, K1_index]
nuisance_indices = setdiff(1:18, target_indices)

# Initial guess for nuisance (use true values)
θ_log_init = log.(θ_true)
nuisance_guess = θ_log_init[nuisance_indices]

println("\nRunning distributed profiling...")
println("Workers: ", nworkers())

t_dist = @elapsed begin
    θ_vals, ll_vals = ReparamTools.profile_target(
        lnlike_θ_log, target_indices,
        θ_log_lower, θ_log_upper,
        nuisance_guess;
        grid_steps=3,
        use_distributed=true,
        n_chunks=3,
        optmaxtime=10.0
    )
end

println("\n=== Results ===")
println("Time: $(round(t_dist, digits=2))s ($(round(t_dist/60, digits=2)) min)")
println("Grid size: ", size(θ_vals))
println("Max likelihood: ", maximum(ll_vals))
println("Min likelihood: ", minimum(ll_vals))

rmprocs(workers())
println("\n✓ Test complete!")
