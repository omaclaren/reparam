using Serialization
using Statistics
using Plots
gr()

# Load the 100x100 results
results = deserialize("nesi/repressilator_16nuisance_100x100_results.jls")

ll_vals = results["ll_vals"]
GRID = results["GRID"]
ψ_lower = results["ψ_lower"]
ψ_upper = results["ψ_upper"]
target_2d = results["target_2d"]
ψ_MLE = results["ψ_MLE"]

# Reshape to matrix
ll_matrix = reshape(ll_vals, GRID, GRID)
ll_max = maximum(filter(isfinite, ll_matrix))
like_matrix = exp.(ll_matrix .- ll_max)

# Build grids
ψ_log_lower = log.(ψ_lower)
ψ_log_upper = log.(ψ_upper)
ψ1_grid = exp.(range(ψ_log_lower[target_2d[1]], ψ_log_upper[target_2d[1]], length=GRID))
ψ2_grid = exp.(range(ψ_log_lower[target_2d[2]], ψ_log_upper[target_2d[2]], length=GRID))

# Find dips - points much worse than neighbors
dip_i = Int[]
dip_j = Int[]
dip_gaps = Float64[]
for i in 2:GRID-1
    for j in 2:GRID-1
        ll_here = ll_matrix[i, j]
        neighbors = [ll_matrix[i-1,j], ll_matrix[i+1,j], ll_matrix[i,j-1], ll_matrix[i,j+1]]
        neighbor_avg = mean(filter(isfinite, neighbors))
        if isfinite(ll_here) && isfinite(neighbor_avg)
            gap = neighbor_avg - ll_here
            if gap > 5
                push!(dip_i, i)
                push!(dip_j, j)
                push!(dip_gaps, gap)
            end
        end
    end
end

dip_ψ1 = ψ1_grid[dip_i]
dip_ψ2 = ψ2_grid[dip_j]

println("Found $(length(dip_i)) dip points (gap > 5 LL units)")
println("Max gap: $(round(maximum(dip_gaps), digits=1))")

# === EXAMPLE DIP POINT ===
dip_example_i, dip_example_j = 36, 55  # ψ1~260, ψ2~2.5, within plotting region [0,8]
dip_ψ1_val = ψ1_grid[dip_example_i]
dip_ψ2_val = ψ2_grid[dip_example_j]
dip_ll = ll_matrix[dip_example_i, dip_example_j]
neighbor_ll = ll_matrix[dip_example_i, dip_example_j+1]

println("\n=== EXAMPLE DIP ===")
println("Point (i=$dip_example_i, j=$dip_example_j)")
println("  ψ1 = $(round(dip_ψ1_val, digits=2)) (K₁/β₁)")
println("  ψ2 = $(round(dip_ψ2_val, digits=4)) (β₁·K₁)")
println("  LL = $(round(dip_ll, digits=2))")
println("  Neighbor LL = $(round(neighbor_ll, digits=2))")
println("  Gap = $(round(neighbor_ll - dip_ll, digits=1)) log-likelihood units")

# === PLOTTING BOUNDS (match replot_profile_results.jl) ===
ψ1_plot_min, ψ1_plot_max = 1e1, 1e5      # K₁/β₁ range
ψ2_plot_min, ψ2_plot_max = 0.0, 8.0      # β₁·K₁ range

# Filter dips to plotting region
in_plot_region = (dip_ψ1 .>= ψ1_plot_min) .& (dip_ψ1 .<= ψ1_plot_max) .&
                 (dip_ψ2 .>= ψ2_plot_min) .& (dip_ψ2 .<= ψ2_plot_max)
n_dips_in_region = sum(in_plot_region)
println("\nDips in plotting region: $n_dips_in_region / $(length(dip_i))")

# === 3-PANEL FIGURE ===
# Panel 1: 2D profile using scatter (no coordinate ambiguity)
# Flatten the grid for scatter
like_flat = vec(like_matrix)
ψ1_flat = repeat(ψ1_grid, outer=GRID)
ψ2_flat = repeat(ψ2_grid, inner=GRID)

# Filter to plotting region
in_bounds = (ψ1_flat .>= ψ1_plot_min) .& (ψ1_flat .<= ψ1_plot_max) .&
            (ψ2_flat .>= ψ2_plot_min) .& (ψ2_flat .<= ψ2_plot_max)

p1 = scatter(ψ1_flat[in_bounds], ψ2_flat[in_bounds], zcolor=like_flat[in_bounds],
            xscale=:log10, color=:dense, clims=(0,1),
            ms=3, ma=0.8, markerstrokewidth=0, label="",
            xlims=(ψ1_plot_min, ψ1_plot_max),
            ylims=(ψ2_plot_min, ψ2_plot_max),
            xlabel="ψ (K₁/β₁)",
            ylabel="ψ (β₁·K₁)",
            title="Profile ($n_dips_in_region dips in region)")
scatter!(dip_ψ1[in_plot_region], dip_ψ2[in_plot_region], mc=:red, ms=2, ma=0.5, label="")
scatter!([dip_ψ1_val], [dip_ψ2_val], mc=:yellow, msc=:black, ms=8, markershape=:circle, label="example")
scatter!([ψ_MLE[target_2d[1]]], [ψ_MLE[target_2d[2]]], mc=:darkgoldenrod, msc=:match, ms=10, markershape=:star, label="MLE")

# Panel 2: 1D slice showing log-likelihood (clipped to plotting region)
slice_ll = ll_matrix[dip_example_i, :]
p2 = plot(ψ2_grid, slice_ll,
          xlabel="ψ (β₁·K₁)", ylabel="Log-likelihood",
          title="1D slice at ψ₁=$(round(dip_ψ1_val, digits=1))",
          xlims=(ψ2_plot_min, ψ2_plot_max),
          lw=2, legend=false, ylims=(-35, 5))
scatter!([dip_ψ2_val], [dip_ll], mc=:yellow, msc=:black, ms=10, label="")
hline!([0], color=:gray, ls=:dash, lw=1)
annotate!(dip_ψ2_val + 0.5, dip_ll + 3, text("dip", :left, 8))

# Panel 3: Zoomed view - use scatter to show discrete values truthfully
zoom_i_range = max(1, dip_example_i-5):min(GRID, dip_example_i+5)
zoom_j_range = max(1, dip_example_j-5):min(GRID, dip_example_j+5)
zoom_like = like_matrix[zoom_i_range, zoom_j_range]
zoom_ψ1 = ψ1_grid[zoom_i_range]
zoom_ψ2 = ψ2_grid[zoom_j_range]

# Flatten for scatter
like_flat = vec(zoom_like')
ψ1_flat = repeat(zoom_ψ1, outer=length(zoom_ψ2))
ψ2_flat = repeat(zoom_ψ2, inner=length(zoom_ψ1))

p3 = scatter(ψ1_flat, ψ2_flat, zcolor=like_flat,
             xscale=:log10, color=:dense, clims=(0,1), ms=10, ma=0.9,
             markerstrokewidth=0, label="",
             xlabel="ψ (K₁/β₁)", ylabel="ψ (β₁·K₁)",
             title="Zoomed: dip point (dark) vs neighbors")
scatter!([dip_ψ1_val], [dip_ψ2_val], mc=:red, msc=:white, ms=14, markershape=:circle, label="dip")

plt = plot(p1, p2, p3, layout=(1, 3), size=(1400, 400))
savefig(plt, "dip_visualization.png")
println("\nSaved: dip_visualization.png")
