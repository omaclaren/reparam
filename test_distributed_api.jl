# Test profile_target with use_distributed=true parameter

using Distributed
using Distributed: WorkerPool

# Add workers
println("Adding 4 workers...")
addprocs(4)
println("Workers: ", workers())

# Load on main
if !@isdefined(ReparamTools)
    include("ReparamTools.jl")
end
using .ReparamTools
using Distributions
using LinearAlgebra

# Load on workers
@everywhere begin
    if !@isdefined(ReparamTools)
        include($(joinpath(@__DIR__, "ReparamTools.jl")))
    end
    using .ReparamTools
    using Distributions
    using LinearAlgebra

    # Define likelihood on all workers (NO type annotation for ForwardDiff compatibility)
    function lnlike_θ(θ)
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

println("\n=== WorkerPool partition (concurrent 1D profiles) ===")
worker_ids = workers()
if length(worker_ids) >= 2
    split_idx = max(1, length(worker_ids) ÷ 2)
    pool1_ids = worker_ids[1:split_idx]
    pool2_ids = worker_ids[split_idx+1:end]
    if isempty(pool1_ids) || isempty(pool2_ids)
        println("Not enough workers to form two pools; skipping partition test.")
    else
        pool1 = WorkerPool(pool1_ids)
        pool2 = WorkerPool(pool2_ids)

        center = 0.5 .* (θ_lower .+ θ_upper)
        ψ_index_1 = [1]
        ψ_index_2 = [2]
        ω_init_1 = [center[2]]
        ω_init_2 = [center[1]]

        θ_seq_1, ll_seq_1 = ReparamTools.profile_target(
            lnlike_θ, ψ_index_1, θ_lower, θ_upper, ω_init_1;
            grid_steps=10, use_distributed=false)
        θ_seq_2, ll_seq_2 = ReparamTools.profile_target(
            lnlike_θ, ψ_index_2, θ_lower, θ_upper, ω_init_2;
            grid_steps=10, use_distributed=false)

        concurrent_results = @sync begin
            task1 = @async begin
                ReparamTools.profile_target(
                    lnlike_θ, ψ_index_1, θ_lower, θ_upper, ω_init_1;
                    grid_steps=10, use_distributed=true,
                    n_chunks=length(pool1.workers), worker_pool=pool1)
            end
            task2 = @async begin
                ReparamTools.profile_target(
                    lnlike_θ, ψ_index_2, θ_lower, θ_upper, ω_init_2;
                    grid_steps=10, use_distributed=true,
                    n_chunks=length(pool2.workers), worker_pool=pool2)
            end
            (fetch(task1), fetch(task2))
        end

        (θ_dist_1, ll_dist_1), (θ_dist_2, ll_dist_2) = concurrent_results

        println("  Profile 1 matches sequential: ",
            maximum(abs.(ll_seq_1 .- ll_dist_1)) < 1e-10)
        println("  Profile 2 matches sequential: ",
            maximum(abs.(ll_seq_2 .- ll_dist_2)) < 1e-10)
        println("  Pool sizes: ", (length(pool1.workers), length(pool2.workers)))
    end
else
    println("Not enough workers for partition test; skipping.")
end

rmprocs(workers())
println("\nTest complete!")
