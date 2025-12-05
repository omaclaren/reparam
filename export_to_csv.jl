# Export Data to CSV for Python Plotting

using Serialization
using DelimitedFiles

println("Loading data...")
if !isfile("repressilator_2D_production_data.jls")
    println("Error: Data file not found!")
    exit(1)
end

(θ_2d_vals, ll_2d_vals, θ_true, θ_log_MLE) = deserialize("repressilator_2D_production_data.jls")

# Convert log parameters to natural scale
θ_2d_natural = [exp.(θ) for θ in θ_2d_vals]
# Extract just the interest parameters (β₁ is index 7, K₁ is index 10)
beta1_vals = [θ[7] for θ in θ_2d_natural]
K1_vals = [θ[10] for θ in θ_2d_natural]

# Prepare data matrix: beta1, K1, log_likelihood
data_matrix = hcat(beta1_vals, K1_vals, ll_2d_vals)

# Export grid data
writedlm("profile_data.csv", data_matrix, ',')
println("Grid data exported to profile_data.csv")

# Export metadata (MLE and True values)
# Format: label, beta1, K1
mle_beta1 = exp(θ_log_MLE[7])
mle_K1 = exp(θ_log_MLE[10])
true_beta1 = θ_true[7]
true_K1 = θ_true[10]

open("metadata.csv", "w") do io
    println(io, "type,beta1,K1")
    println(io, "MLE,$mle_beta1,$mle_K1")
    println(io, "TRUE,$true_beta1,$true_K1")
end
println("Metadata exported to metadata.csv")
