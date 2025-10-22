# Test profile_target with use_distributed=true parameter

using Distributed

# Add workers
println("Adding 2 workers...")
addprocs(2)
println("Workers: ", workers())

# Load on main
include("ReparamTools.jl")
using .ReparamTools
using Distributions
using LinearAlgebra

# Load on workers
@everywhere begin
    include($(joinpath(@__DIR__, "ReparamTools.jl")))
    using .ReparamTools
    using Distributions
    using LinearAlgebra

    # Define likelihood on all workers
    function lnlike_θ(θ::Vector{Float64})
        if length(θ) != 2
            return -Inf
        end
        try
            dist = MvNormal(θ, $(diagm([1.0, 2.0])))
            return sum(logpdf(dist, d) for d in $([[3.0, 5.0] for _ in 1:100]))
        catch
            return -Inf
        end
    end
end

# Test parameters
θ_lower = [0.1, 0.1]
θ_upper = [10.0, 10.0]
ψ_indices = [1, 2]  # 2D profile
ω_initial = Float64[]  # No nuisance parameters

println("\nTesting profile_target API with 2D grid (10x10)...\n")

# Sequential
println("=== Sequential (use_distributed=false) ===")
t_seq = @elapsed begin
    θ_seq, ll_seq = ReparamTools.profile_target(lnlike_θ, ψ_indices, θ_lower, θ_upper, ω_initial;
        grid_steps=10, use_distributed=false)
end
println("Time: $(round(t_seq, digits=2))s")
println("Grid size: ", size(θ_seq))

# Distributed
println("\n=== Distributed (use_distributed=true, n_chunks=2) ===")
t_dist = @elapsed begin
    θ_dist, ll_dist = ReparamTools.profile_target(lnlike_θ, ψ_indices, θ_lower, θ_upper, ω_initial;
        grid_steps=10, use_distributed=true, n_chunks=2)
end
println("Time: $(round(t_dist, digits=2))s")
println("Grid size: ", size(θ_dist))

# Compare
println("\n=== Comparison ===")
println("Speedup: $(round(t_seq/t_dist, digits=2))x")
println("Max likelihood diff: $(maximum(abs.(ll_seq .- ll_dist)))")
println("Results match: ", maximum(abs.(ll_seq .- ll_dist)) < 1e-10)

rmprocs(workers())
println("\nTest complete!")
