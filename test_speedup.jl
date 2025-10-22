# Test realistic speedup with expensive likelihood function

using Distributed

# Add workers
println("Adding 3 workers...")
addprocs(3)
println("Workers: ", workers())

# Load on main process first
include("ReparamTools.jl")
using .ReparamTools
using Distributions
using LinearAlgebra

# Load everything on all worker processes
@everywhere begin
    include($(joinpath(@__DIR__, "ReparamTools.jl")))
    using .ReparamTools
    using Distributions
    using LinearAlgebra

    # Define expensive likelihood on all workers (actual computational work)
    function lnlike_θ(θ::Vector{Float64})
        # Do actual computation that keeps CPU busy
        # Matrix operations similar to what happens in ODE solves
        n = 1000
        A = randn(n, n)
        B = randn(n, n)
        C = A * B  # Expensive matrix multiply

        # Also do some nonlinear function evaluations
        result = 0.0
        for i in 1:10000
            result += exp(-sum((θ .- [2.0, 3.0]).^2) * i / 10000)
        end

        return -sum((θ .- [2.0, 3.0]).^2) + log(result) - sum(C[1:10])
    end
end

# Build grid: 10x10 = 100 points
ψ_grid = vec([[x, y] for x in range(1.0, 3.0, length=10), y in range(2.0, 4.0, length=10)])

println("\nGrid size: ", length(ψ_grid), " points")
println("Expected speedup with 3 workers: 3x (ideal)\n")

# Test sequential
println("=== Sequential Profiling ===")
t_seq = @elapsed begin
    θ_seq, ll_seq = ReparamTools.profile_grid_sequential(
        lnlike_θ, ψ_grid, [1, 2],
        [0.1, 0.1], [10.0, 10.0], Float64[]
    )
end
println("Sequential time: ", round(t_seq, digits=2), " seconds")

# Test distributed
println("\n=== Distributed Profiling (3 workers) ===")
t_dist = @elapsed begin
    θ_dist, ll_dist = ReparamTools.profile_grid_distributed(
        lnlike_θ, ψ_grid, [1, 2],
        [0.1, 0.1], [10.0, 10.0], Float64[];
        n_chunks=3
    )
end
println("Distributed time: ", round(t_dist, digits=2), " seconds")

# Calculate speedup
speedup = t_seq / t_dist
println("\n=== Results ===")
println("Actual speedup: ", round(speedup, digits=2), "x")
println("Efficiency: ", round(speedup/3 * 100, digits=1), "%")
println("Results match: ", maximum(abs.(ll_seq .- ll_dist)) < 1e-10)

# Cleanup
rmprocs(workers())
println("\nTest complete.")
