# Quick test: Compare sequential vs distributed on stat_model 2D profile

using Distributed
addprocs(2)

# Load on main first
if !@isdefined(ReparamTools)
    include("ReparamTools.jl")
end
using .ReparamTools
using Distributions
using LinearAlgebra
using Random

# Then load on workers
@everywhere begin
    if !@isdefined(ReparamTools)
        include($(joinpath(@__DIR__, "ReparamTools.jl")))
    end
    using .ReparamTools
    using Distributions
    using LinearAlgebra
end

Random.seed!(12)

# Simple Poisson limit model from stat_model.jl
n_true = 1000.0
p_true = 0.01
n_obs = 100
data = rand(Poisson(n_true * p_true), n_obs)

# Define likelihood everywhere
@everywhere function lnlike_xy(xy::Vector{Float64})
    n, p = xy
    if n <= 0 || p <= 0 || p >= 1
        return -Inf
    end
    lambda = n * p
    return sum(logpdf(Poisson(lambda), d) for d in $(data))
end

# Parameter bounds (from stat_model.jl)
xy_lower_bounds = [1.0, 0.001]
xy_upper_bounds = [10000.0, 0.999]

println("Testing 2D profile (both parameters)...")
println("Grid: 20x20 = 400 points\n")

# Sequential
println("=== Sequential ===")
t_seq = @elapsed begin
    θ_seq, ll_seq = ReparamTools.profile_target(
        lnlike_xy, [1,2], xy_lower_bounds, xy_upper_bounds, Float64[];
        grid_steps=20, use_distributed=false
    )
end
println("Time: $(round(t_seq, digits=2))s")

# Distributed
println("\n=== Distributed (2 workers) ===")
t_dist = @elapsed begin
    θ_dist, ll_dist = ReparamTools.profile_target(
        lnlike_xy, [1,2], xy_lower_bounds, xy_upper_bounds, Float64[];
        grid_steps=20, use_distributed=true, n_chunks=2
    )
end
println("Time: $(round(t_dist, digits=2))s")

println("\n=== Results ===")
println("Speedup: $(round(t_seq/t_dist, digits=2))x")
println("Max diff: $(maximum(abs.(ll_seq .- ll_dist)))")

rmprocs(workers())
