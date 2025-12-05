# Plot 2D profile likelihood from saved data
# Matches the style from examples/repressilator.jl
using Serialization
using Plots
using Printf
using Distributions

# Load the data
println("Loading data...")
data = deserialize("repressilator_2D_intermediate_data.jls")
θ_2d_vals, ll_2d_vals, θ_true, θ_log_MLE = data

println("Loaded $(length(ll_2d_vals)) profile points")
println("Log-likelihood range: $(minimum(ll_2d_vals)) to $(maximum(ll_2d_vals))")

# Parameter indices
β1_index = 7
K1_index = 10

# Extract β₁ and K₁ values in ORIGINAL scale (exp of log values)
β1_values_2d = unique([exp(θ[β1_index]) for θ in θ_2d_vals])
K1_values_2d = unique([exp(θ[K1_index]) for θ in θ_2d_vals])

# Sort the grid values
sort!(β1_values_2d)
sort!(K1_values_2d)

# Determine grid size
n_β1 = length(β1_values_2d)
n_K1 = length(K1_values_2d)
println("Grid size: $(n_β1)x$(n_K1)")

# Reshape likelihoods to 2D grid
# In Base.product, first argument varies fastest, so β₁ varies fastest
# ll_2d_vals is ordered: (β1[1],K1[1]), (β1[2],K1[1]), ..., (β1[n],K1[1]), (β1[1],K1[2]), ...
# Reshape to (n_β1, n_K1) where rows=β₁, cols=K₁
# For contourf(x, y, Z), Z should be (length(y), length(x)) = (n_K1, n_β1)
ll_grid_2d = reshape(ll_2d_vals, n_β1, n_K1)'

# Normalize to max = 0, then convert to likelihood scale
ll_grid_2d_norm = ll_grid_2d .- maximum(ll_grid_2d)
like_grid_2d = exp.(ll_grid_2d_norm)

# Chi-square calibration for 95% confidence contour (df=2 for 2D)
df_2d = 2
lstar_2d = exp(-quantile(Chisq(df_2d), 0.95)/2)

# True values and MLE (in original scale)
θ_MLE = exp.(θ_log_MLE)
β1_true = θ_true[β1_index]
K1_true = θ_true[K1_index]
β1_MLE = θ_MLE[β1_index]
K1_MLE = θ_MLE[K1_index]

println("True: β₁=$(round(β1_true, digits=4)), K₁=$(round(K1_true, digits=2))")
println("MLE:  β₁=$(round(β1_MLE, digits=4)), K₁=$(round(K1_MLE, digits=2))")

# Create the plot (matching examples/repressilator.jl style)
gr()
plt_2d = contourf(β1_values_2d, K1_values_2d, like_grid_2d,
                  color=:dense, levels=20, lw=0,
                  xlabel="β₁", ylabel="K₁",
                  title="Repressilator 2D Profile: (β₁, K₁)",
                  colorbar=false,
                  size=(800, 600))

# Add 95% confidence contour
contour!(plt_2d, β1_values_2d, K1_values_2d, like_grid_2d,
         levels=[lstar_2d], color=:black, lw=2, legend=false, fill=false)

# Mark MLE
scatter!(plt_2d, [β1_MLE], [K1_MLE],
         mc=:silver, msc=:match, markersize=8, markershape=:circle, 
         label="MLE", legend=:topright)

# Mark true values
scatter!(plt_2d, [β1_true], [K1_true],
         mc=:darkgoldenrod, msc=:match, markersize=10, markershape=:star,
         label="True")

savefig(plt_2d, "figures/repressilator_2D_distributed_beta1_K1.png")
println("\n✓ Saved: figures/repressilator_2D_distributed_beta1_K1.png")
