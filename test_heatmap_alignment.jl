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

# Reshape to matrix
ll_matrix = reshape(ll_vals, GRID, GRID)
ll_max = maximum(filter(isfinite, ll_matrix))
like_matrix = exp.(ll_matrix .- ll_max)

# Build grids (cell centers)
ψ_log_lower = log.(ψ_lower)
ψ_log_upper = log.(ψ_upper)
ψ1_centers = exp.(range(ψ_log_lower[target_2d[1]], ψ_log_upper[target_2d[1]], length=GRID))
ψ2_centers = exp.(range(ψ_log_lower[target_2d[2]], ψ_log_upper[target_2d[2]], length=GRID))

# Dip point
dip_i, dip_j = 36, 55
dip_ψ1 = ψ1_centers[dip_i]
dip_ψ2 = ψ2_centers[dip_j]

println("Dip at (i=$dip_i, j=$dip_j)")
println("  ψ1 = $dip_ψ1")
println("  ψ2 = $dip_ψ2")
println("  LL = $(ll_matrix[dip_i, dip_j])")

# Zoom region
zoom_i = 31:41
zoom_j = 50:60
zoom_like = like_matrix[zoom_i, zoom_j]
zoom_ψ1 = ψ1_centers[zoom_i]
zoom_ψ2 = ψ2_centers[zoom_j]

# Test 1: heatmap with cell centers (default behavior)
p1 = heatmap(zoom_ψ1, zoom_ψ2, zoom_like',
             xscale=:log10, color=:dense, clims=(0,1),
             title="heatmap (centers)", xlabel="ψ1", ylabel="ψ2")
scatter!([dip_ψ1], [dip_ψ2], mc=:red, ms=10, label="dip")

# Test 2: contourf with cell centers
p2 = contourf(zoom_ψ1, zoom_ψ2, zoom_like',
              xscale=:log10, color=:dense, clims=(0,1), levels=20, lw=0,
              title="contourf (centers)", xlabel="ψ1", ylabel="ψ2")
scatter!([dip_ψ1], [dip_ψ2], mc=:red, ms=10, label="dip")

# Test 3: Just scatter the actual values with color
like_flat = vec(zoom_like')
ψ1_flat = repeat(zoom_ψ1, outer=length(zoom_ψ2))
ψ2_flat = repeat(zoom_ψ2, inner=length(zoom_ψ1))
p3 = scatter(ψ1_flat, ψ2_flat, zcolor=like_flat,
             xscale=:log10, color=:dense, clims=(0,1), ms=8, ma=0.8,
             markerstrokewidth=0, label="",
             title="scatter (truth)", xlabel="ψ1", ylabel="ψ2")
scatter!([dip_ψ1], [dip_ψ2], mc=:red, msc=:white, ms=12, label="dip")

plt = plot(p1, p2, p3, layout=(1, 3), size=(1200, 400))
savefig(plt, "test_heatmap_alignment.png")
println("\nSaved: test_heatmap_alignment.png")
