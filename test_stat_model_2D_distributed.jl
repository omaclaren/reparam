# Test stat_model: 2D Profile Likelihood with Distributed Execution
# Replicates original stat_model.jl workflow exactly, but with distributed option
# Same settings: grid_steps=500 → 500×500 = 250,000 points

using Distributed

println("="^70)
println("TEST: stat_model 2D Profile (Sequential vs Distributed)")
println("="^70)

# Add workers for distributed version
println("\n[1/5] Setting up workers...")
addprocs(4)
println("✓ Added 4 workers: ", workers())

# Load ReparamTools on main process
include("ReparamTools.jl")
using .ReparamTools
using Distributions
using LinearAlgebra
using Random
using Printf
using Statistics
using Plots
using LaTeXStrings
using Distributions: Chisq, quantile

# Load on workers
@everywhere begin
    include($(joinpath(@__DIR__, "ReparamTools.jl")))
    using .ReparamTools
    using Distributions
    using LinearAlgebra
end

println("\n[2/5] Setting up stat_model (Poisson limit)...")
Random.seed!(12)  # Same as stat_model.jl

# Model definition (exactly as stat_model.jl)
poisson_limit = true
n_true, p_true = 100.0, 0.2
xy_true = [n_true, p_true]

# Parameter bounds
n_min, n_max = 0.1, 500.0
p_min, p_max = 0.0001, 1.0
xy_lower_bounds = [n_min, p_min]
xy_upper_bounds = [n_max, p_max]

# Variable names
varnames = Dict("ψ1" => "n", "ψ2" => "p")
varnames["ψ1_save"] = "n"
varnames["ψ2_save"] = "p"

# Generate data (exactly as stat_model.jl)
if poisson_limit
    ϕ_xy = xy -> [xy[1]*xy[2], xy[1]*xy[2]]
else
    ϕ_xy = xy -> [xy[1]*xy[2], xy[1]*xy[2]*(1-xy[2])]
end

distrib_xy = xy -> Normal(ϕ_xy(xy)[1], sqrt(ϕ_xy(xy)[2]))

# Generate data
data_xy = rand(distrib_xy(xy_true), 10)

# Likelihood function
function lnlike_xy(xy)
    try
        dist = distrib_xy(xy)
        return sum(logpdf(dist, d) for d in data_xy)
    catch
        return -Inf
    end
end

# Send to workers
@everywhere data_global = $(data_xy)
@everywhere ϕ_xy_worker(xy) = [xy[1]*xy[2], xy[1]*xy[2]]
@everywhere function lnlike_xy_worker(xy)
    try
        μ = ϕ_xy_worker(xy)[1]
        σ = sqrt(ϕ_xy_worker(xy)[2])
        dist = Normal(μ, σ)
        return sum(logpdf(dist, d) for d in data_global)
    catch
        return -Inf
    end
end

println("✓ Model setup complete")
println("  Model: Poisson limit - Normal(np, sqrt(np))")
println("  True parameters: n=$(n_true), p=$(p_true)")
println("  Data: ", length(data_xy), " observations")
println("  Mean: ", round(mean(data_xy), digits=2))

# Grid settings (exactly as stat_model.jl)
grid_steps = [500]
target_indices_ij = [1, 2]  # Profile both n and p
nuisance_guess = Float64[]  # No nuisance parameters

println("\n[3/5] Running SEQUENTIAL 2D profile...")
println("  Grid: 500×500 = 250,000 points")

t_seq = @elapsed begin
    ψω_seq, lnlike_seq = ReparamTools.profile_target(
        lnlike_xy, target_indices_ij,
        xy_lower_bounds, xy_upper_bounds,
        nuisance_guess;
        grid_steps=grid_steps
    )
end

println("✓ Sequential complete")
println("  Time: $(round(t_seq, digits=2))s")
println("  Points: ", length(lnlike_seq))
println("  Finite: ", sum(isfinite.(lnlike_seq)), "/", length(lnlike_seq))

# Extract ψ values for plotting (exactly as stat_model.jl)
ψ_seq = [ψω[target_indices_ij] for ψω in ψω_seq]

println("\n[4/5] Running DISTRIBUTED 2D profile...")
println("  Grid: 500×500 = 250,000 points")
println("  Workers: 4, chunks: 4")

t_dist = @elapsed begin
    ψω_dist, lnlike_dist = ReparamTools.profile_target(
        lnlike_xy_worker, target_indices_ij,
        xy_lower_bounds, xy_upper_bounds,
        nuisance_guess;
        grid_steps=grid_steps,
        use_distributed=true,
        n_chunks=4
    )
end

println("✓ Distributed complete")
println("  Time: $(round(t_dist, digits=2))s")
println("  Points: ", length(lnlike_dist))
println("  Finite: ", sum(isfinite.(lnlike_dist)), "/", length(lnlike_dist))

# Extract ψ values for plotting
ψ_dist = [ψω[target_indices_ij] for ψω in ψω_dist]

# Compare results
ll_diff = abs.(lnlike_seq - lnlike_dist)
max_diff = maximum(ll_diff)
println("\n  Comparison:")
println("    Max likelihood diff: ", @sprintf("%.2e", max_diff))
println("    Speedup: $(round(t_seq/t_dist, digits=2))x")
println("    Efficiency: $(round(100*(t_seq/t_dist)/4, digits=1))%")

println("\n[5/5] Generating comparison plots...")

# Helper function to plot 2D contour (replicating visualization.jl)
function plot_2D_profile(ψ_values, lnlike_values, title_text;
                         ψ_true=[], l_level=95, nshade_levels=20)

    # Extract unique grid values (undo Cartesian product)
    ψ1_values = unique([ψ1 for (ψ1, _) in ψ_values])
    ψ2_values = unique([ψ2 for (_, ψ2) in ψ_values])

    # Reshape to 2D grid
    lnlike_grid = reshape(lnlike_values, length(ψ1_values), length(ψ2_values))

    # Convert to likelihood scale (normalized)
    like_grid = exp.(lnlike_grid)

    # Chi-square calibration for confidence contour
    df = 2
    lstar = exp(-quantile(Chisq(df), l_level/100)/2)

    # Filled contours
    plt = contourf(ψ1_values, ψ2_values, like_grid',
                   color=:dense, levels=nshade_levels, lw=0,
                   colorbar=true,
                   xlabel=latexstring(varnames["ψ1"]),
                   ylabel=latexstring(varnames["ψ2"]),
                   title=title_text)

    # Add confidence level contour
    contour!(ψ1_values, ψ2_values, like_grid',
             levels=[lstar], color=:black, lw=2, legend=false, fill=false)

    # Mark MLE (grid maximum)
    max_idx = argmax(like_grid)
    ψ1_max = ψ1_values[max_idx[1]]
    ψ2_max = ψ2_values[max_idx[2]]
    scatter!([ψ1_max], [ψ2_max],
             mc=:silver, msc=:match, markersize=8,
             markershape=:circle, legend=false)

    # Mark true values
    if length(ψ_true) > 0
        scatter!([ψ_true[1]], [ψ_true[2]],
                 mc=:darkgoldenrod, msc=:match, markersize=10,
                 markershape=:star, legend=false)
    end

    return plt
end

# Create comparison plots
p1 = plot_2D_profile(ψ_seq, lnlike_seq, "Sequential ($(round(t_seq, digits=2))s)",
                     ψ_true=xy_true)

p2 = plot_2D_profile(ψ_dist, lnlike_dist, "Distributed ($(round(t_dist, digits=2))s)",
                     ψ_true=xy_true)

# Difference heatmap
ψ1_vals = unique([ψ1 for (ψ1, _) in ψ_seq])
ψ2_vals = unique([ψ2 for (_, ψ2) in ψ_seq])
diff_grid = reshape(ll_diff, length(ψ1_vals), length(ψ2_vals))

p3 = heatmap(ψ1_vals, ψ2_vals, log10.(diff_grid' .+ 1e-16),
             xlabel=latexstring(varnames["ψ1"]),
             ylabel=latexstring(varnames["ψ2"]),
             title="log₁₀(Absolute Difference)",
             color=:thermal,
             colorbar=true)

# Timing comparison
p4 = plot(title="Performance",
          xlabel="Method",
          ylabel="Time (s)",
          legend=false,
          xticks=(1:2, ["Sequential", "Distributed"]),
          ylims=(0, max(t_seq, t_dist)*1.2))
bar!([1, 2], [t_seq, t_dist], color=[:blue, :green], alpha=0.7)
annotate!(1, t_seq + 0.05*max(t_seq, t_dist),
          text("$(round(t_seq, digits=2))s", 10))
annotate!(2, t_dist + 0.05*max(t_seq, t_dist),
          text("$(round(t_dist, digits=2))s\n$(round(t_seq/t_dist, digits=2))x", 10))

# Combine
plot_combined = plot(p1, p2, p3, p4,
                     layout=(2,2),
                     size=(1400, 1000),
                     plot_title="stat_model 2D Profile: Sequential vs Distributed")

# Save
output_file = "stat_model_2D_distributed.png"
savefig(plot_combined, output_file)
println("✓ Figure saved: ", output_file)

# Summary table
println("\n" * "="^70)
println("SUMMARY")
println("="^70)
println(@sprintf("%-25s %15s %15s", "Metric", "Sequential", "Distributed"))
println("-"^70)
println(@sprintf("%-25s %15.2fs %15.2fs", "Time", t_seq, t_dist))
println(@sprintf("%-25s %15d %15d", "Grid points", length(lnlike_seq), length(lnlike_dist)))
println(@sprintf("%-25s %15.2e %15.2e", "Max diff", max_diff, max_diff))
println(@sprintf("%-25s %15.2fx", "Speedup", t_seq/t_dist))
println(@sprintf("%-25s %15.1f%%", "Efficiency (4 workers)", 100*(t_seq/t_dist)/4))
println("="^70)

# Test verdict
tolerance = 1e-8
if max_diff < tolerance
    println("\n✓✓✓ TEST PASSED ✓✓✓")
    println("Distributed 2D profiling matches sequential!")
    exit_code = 0
else
    println("\n✗✗✗ TEST FAILED ✗✗✗")
    println("Max difference $(max_diff) exceeds tolerance $(tolerance)")
    exit_code = 1
end

# Cleanup
rmprocs(workers())

exit(exit_code)
