using Plots

# Simulate creating a simple contourf plot with normalized data
ll_norm = [range(-5, 0, length=20) for _ in 1:20]
ll_grid = hcat(ll_norm...)'
like_grid = exp.(ll_grid)

println("Data ranges:")
println("  Log-likelihood: ", extrema(ll_grid))
println("  Likelihood: ", extrema(like_grid))

# Try to make a simple plot
try
    println("\nTrying plot with colorbar...")
    p = contourf(1:20, 1:20, like_grid, colorbar=true)
    savefig(p, "/tmp/test_colorbar.png")
    println("  ✓ Success with colorbar")
catch e
    println("  ✗ Failed: ", e)
end

try
    println("\nTrying plot without colorbar...")
    p = contourf(1:20, 1:20, like_grid, colorbar=false)
    savefig(p, "/tmp/test_no_colorbar.png")
    println("  ✓ Success without colorbar")
catch e
    println("  ✗ Failed: ", e)
end
