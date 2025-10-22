# Test script for distributed 2D profiling
# Simple 5x5 grid test with 2 workers

using Distributed

# Add 2 workers
println("Adding 2 workers...")
addprocs(2)
println("Workers: ", workers())

# Load ReparamTools on main process first (with guard)
if !@isdefined(ReparamTools)
    include("ReparamTools.jl")
end
using .ReparamTools: profile_grid_distributed, profile_grid_sequential
using Random
using Distributions
using LinearAlgebra

# Simple test model: 2D Gaussian (define BEFORE loading on workers)
θ_true = [3.0, 5.0]
Σ_global = [1.0 0.5; 0.5 2.0]
n_obs = 100

Random.seed!(123)
data_global = [rand(MvNormal(θ_true, Σ_global)) for _ in 1:n_obs]

# Load ReparamTools and define likelihood on all workers
@everywhere begin
    if !@isdefined(ReparamTools)
        include($(joinpath(@__DIR__, "ReparamTools.jl")))
    end
    using .ReparamTools

    # Define likelihood function on all workers
    function lnlike_θ(θ::Vector{Float64})
        if length(θ) != 2
            return -Inf
        end
        try
            dist = MvNormal(θ, $(Σ_global))
            return sum(logpdf(dist, d) for d in $(data_global))
        catch
            return -Inf
        end
    end
end

# Define on main process too
function lnlike_θ(θ::Vector{Float64})
    if length(θ) != 2
        return -Inf
    end
    try
        dist = MvNormal(θ, Σ_global)
        return sum(logpdf(dist, d) for d in data_global)
    catch
        return -Inf
    end
end

# Setup for 2D profiling
θ_lower = [0.0, 0.0]
θ_upper = [6.0, 10.0]

# Build 5x5 grid
θ1_grid = collect(LinRange(θ_lower[1], θ_upper[1], 5))
θ2_grid = collect(LinRange(θ_lower[2], θ_upper[2], 5))
ψ_grid = vec([collect([θ1, θ2]) for θ1 in θ1_grid, θ2 in θ2_grid])

println("\nGrid size: ", length(ψ_grid), " points")
println("ψ_indices: [1, 2] (both parameters of interest, no nuisance params)")

# Test distributed profiling
println("\n=== Testing Distributed Profiling ===")
ψ_indices = [1, 2]
ω_initial = Float64[]  # No nuisance parameters

@time θ_vals_dist, ll_vals_dist = profile_grid_distributed(
    lnlike_θ, ψ_grid, ψ_indices,
    θ_lower, θ_upper, ω_initial;
    method=:LN_BOBYQA,
    optmaxtime=10.0,
    n_chunks=2,
    chunk_strategy=:stripes,
    track_convergence=false
)

# Test sequential for comparison
println("\n=== Testing Sequential Profiling (for comparison) ===")
@time θ_vals_seq, ll_vals_seq = profile_grid_sequential(
    lnlike_θ, ψ_grid, ψ_indices,
    θ_lower, θ_upper, ω_initial;
    method=:LN_BOBYQA,
    optmaxtime=10.0,
    track_convergence=false
)

# Compare results
println("\n=== Results Comparison ===")
println("Distributed max ll: ", maximum(ll_vals_dist))
println("Sequential max ll:  ", maximum(ll_vals_seq))
println("Difference: ", maximum(abs.(ll_vals_dist .- ll_vals_seq)))

if maximum(abs.(ll_vals_dist .- ll_vals_seq)) < 1e-6
    println("\n✓ SUCCESS: Distributed and sequential results match!")
else
    println("\n⚠ WARNING: Results differ!")
end

# Cleanup
rmprocs(workers())
println("\nWorkers removed. Test complete.")
