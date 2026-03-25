# Quick test of stat_sum_model sequential IIR without profiling
include("../ReparamTools.jl")
using .ReparamTools
using Distributions
using LinearAlgebra
using Random

Random.seed!(42)

# Model setup
ϕ_xy = xy -> [xy[1]*xy[2] + xy[3]*xy[4], xy[1]*xy[2] + xy[3]*xy[4]]
distrib_xy = xy -> Normal(ϕ_xy(xy)[1], sqrt(ϕ_xy(xy)[2]))

# True parameters and data
# Use asymmetric products to avoid SVD rotation
xy_true = [80.0, 0.1, 40.0, 0.3]  # Products: 8 and 12, sum = 20
data = rand(distrib_xy(xy_true), 10)

println("True sum: ", xy_true[1]*xy_true[2] + xy_true[3]*xy_true[4])

# Log space
xytoXY_log(xy) = log.(xy)
XYtoxy_log(XY) = exp.(XY)
ϕ_XY_log = construct_ϕ_XY(ϕ_xy, XYtoxy_log)

# Construct likelihood and find MLE
lnlike_xy = construct_lnlike_xy(distrib_xy, data)
lnlike_XY_log = construct_lnlike_XY(lnlike_xy, XYtoxy_log)

# Find MLE in log coordinates
xy_lower = [0.1, 0.0001, 0.1, 0.0001]
xy_upper = [500.0, 1.0, 500.0, 1.0]
XY_log_lower = log.(xy_lower)
XY_log_upper = log.(xy_upper)
XY_log_initial = log.([50.0, 0.3, 50.0, 0.3])

# Quick MLE
using Optim
result = optimize(θ -> -lnlike_XY_log(θ), XY_log_lower, XY_log_upper, XY_log_initial)
XY_log_MLE = Optim.minimizer(result)

println("MLE in log coords: ", round.(XY_log_MLE, digits=4))
println("MLE in original: ", round.(exp.(XY_log_MLE), digits=4))

println("\n=== STAGE 1: Monomial Identification ===")
# Use true parameters for invariance analysis to get clean structure
XY_log_true = log.(xy_true)
S1, N1, N_perp1, rank1 = find_invariant_subspace(ϕ_XY_log, XY_log_true)
println("Rank: ", rank1)
println("N_perp dimensions: ", size(N_perp1, 2))
println("N dimensions: ", size(N1, 2))

# Build A1 - scale identifiable block only
N_perp1_scaled = scale_and_round(N_perp1)

# Clean invariant null space
N1_clean = copy(N1)
for j in 1:size(N1_clean, 2)
    col = N1_clean[:, j]
    col_norm = norm(col)
    col[abs.(col) .< eps() * col_norm] .= 0.0
    N1_clean[:, j] = col
end

A1_full_T = hcat(N_perp1_scaled, N1_clean)
A1_full = A1_full_T'

println("\nA1 transformation matrix:")
display(A1_full)

# Stage 1 transformations
xy_to_stage1(xy) = exp.(A1_full * log.(xy))
stage1_to_xy(θ1) = exp.(inv(A1_full) * log.(θ1))

stage1_MLE = xy_to_stage1(xy_true)
println("\nStage 1 at true params:")
display(stage1_MLE)

println("\n=== STAGE 2: Linear Combination Identification ===")
ϕ_stage2 = θ1 -> ϕ_xy(stage1_to_xy(θ1))

S2, N2, N_perp2, rank2 = find_invariant_subspace(ϕ_stage2, stage1_MLE)
println("Rank: ", rank2)
println("N_perp dimensions: ", size(N_perp2, 2))
println("N dimensions: ", size(N2, 2))

# Build A2 - scale identifiable block only
N_perp2_scaled = scale_and_round(N_perp2)

# Clean invariant null space
N2_clean = copy(N2)
for j in 1:size(N2_clean, 2)
    col = N2_clean[:, j]
    col_norm = norm(col)
    col[abs.(col) .< eps() * col_norm] .= 0.0
    N2_clean[:, j] = col
end

A2_full_T = hcat(N_perp2_scaled, N2_clean)
A2_full = A2_full_T'

println("\nA2 transformation matrix:")
display(A2_full)

# Overall transformation
xy_to_final(xy) = A2_full * exp.(A1_full * log.(xy))

# Inverse transformation (corrected): θ¹ = A2 \ ψ, then θ = exp(inv(A1) * log(θ¹))
function final_to_xy_test(ψ)
    θ1 = A2_full \ ψ
    if any(θ1 .<= 0)
        println("WARNING: θ¹ has non-positive components: ", θ1)
        return fill(NaN, length(θ1))
    end
    return exp.(inv(A1_full) * log.(θ1))
end

final_at_true = xy_to_final(xy_true)
println("\nFinal transformation at true params:")
display(final_at_true)

println("\nExpected first component ≈ ", xy_true[1]*xy_true[2] + xy_true[3]*xy_true[4])
println("Actual first component:    ", final_at_true[1])

# Test inverse
println("\n=== TESTING INVERSE ===")
xy_reconstructed = final_to_xy_test(final_at_true)
println("Original xy:      ", round.(xy_true, digits=4))
println("Reconstructed xy: ", round.(xy_reconstructed, digits=4))
println("Match: ", isapprox(xy_true, xy_reconstructed, rtol=1e-6))

println("\n=== SUCCESS ===")
println("Sequential IIR application completed successfully!")
