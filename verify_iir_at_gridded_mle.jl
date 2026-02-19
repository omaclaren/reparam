# Verify IIR transformation is the same at gridded MLE vs original MLE
# This demonstrates robustness of IIR to small MLE variations

using Serialization
using LinearAlgebra
using DifferentialEquations
include("ReparamTools.jl")
include("examples/RepressilatorModel.jl")
using .RepressilatorModel

# Load gridded MLE (properly extracted)
θ_grid = deserialize("gridded_mle_theta_full.jls")

# Load original results for comparison
results = deserialize("nesi/repressilator_16nuisance_50x50_results.jls")
A_T_original = results["A_T_final"]
θ_MLE_original = results["θ_MLE"]

println("=== IIR Verification at Gridded MLE ===\n")
println("Original MLE θ[7,10] (β₁, K₁): $(round(θ_MLE_original[7], digits=4)), $(round(θ_MLE_original[10], digits=2))")
println("Gridded MLE θ[7,10] (β₁, K₁): $(round(θ_grid[7], digits=4)), $(round(θ_grid[10], digits=2))")

# Set up repressilator model (same as examples/repressilator.jl)
m0 = [0.5, 0.3, 0.1]
P0 = [2.0, 1.5, 1.0]
u0 = vcat(m0, P0)
tspan = (0.0, 200.0)
t_obs_fine = range(0, 200, length=501)  # Fine grid for IIR

# Create ODE problem with gridded MLE
prob = ODEProblem(repressilator!, u0, tspan, θ_grid)

# Auxiliary mapping in LOG-SPACE (as in original repressilator.jl)
function ϕ_log(θ_log)
    θ = exp.(θ_log)
    prob_θ = remake(prob, p=θ)
    sol = solve(prob_θ, Rodas5P(), saveat=t_obs_fine, abstol=1e-10, reltol=1e-8)
    if sol.retcode != :Success
        return fill(NaN, 3 * length(t_obs_fine))
    end
    m_obs = hcat([sol.u[i][1:3] for i in 1:length(sol.t)]...)'
    return vec(m_obs)
end

# Run IIR analysis in log-space (same as original)
θ_log_grid = log.(θ_grid)
println("\nRunning IIR analysis at gridded MLE (log-space)...")
S, N, N_perp, rank_J = ReparamTools.find_invariant_subspace(
    ϕ_log, θ_log_grid
    # Uses default rtol_invariance = 32√eps, calibrated for stiff ODEs
)

println("Rank: $rank_J / $(length(θ_grid))")
println("Identifiable: $(size(N_perp, 2)), Non-identifiable: $(size(N, 2))")

# Show singular values to check for gap
println("\nSingular values (look for gap around position 15):")
for (i, s) in enumerate(S)
    marker = i == 15 ? " <-- expected gap" : ""
    println("  σ[$i] = $(round(s, digits=6))$marker")
end
if length(S) >= 16
    gap = S[15] / S[16]
    println("\nGap σ[15]/σ[16] = $(round(gap, digits=1))×")
end

# Build transformation matrix
A_T_new = hcat(N_perp, N)
A_T_scaled = ReparamTools.scale_and_round(A_T_new)

println("\n=== Transformation Comparison ===")
println("Original A_T shape: $(size(A_T_original))")
println("New A_T shape: $(size(A_T_scaled))")

# Check if subspaces are the same (up to basis rotation)
# Compare the null space (non-identifiable) directions
N_original = A_T_original[:, end-2:end]  # Last 3 columns
N_new = A_T_scaled[:, end-2:end]

# Compute angle between subspaces
function subspace_angle(A, B)
    # Orthonormalize
    QA = Matrix(qr(Float64.(A)).Q)[:, 1:size(A,2)]
    QB = Matrix(qr(Float64.(B)).Q)[:, 1:size(B,2)]
    # Singular values of Q_A' * Q_B give cosines of principal angles
    svs = svd(QA' * QB).S
    # Convert to angles in degrees
    angles = acosd.(clamp.(svs, -1, 1))
    return angles
end

angles = subspace_angle(N_original, N_new)
println("\nPrincipal angles between null spaces (non-identifiable):")
println("  $(round.(angles, digits=2))°")
println("  (0° = identical subspaces)")

# Compare identifiable subspaces too
N_perp_original = A_T_original[:, 1:end-3]
N_perp_new = A_T_scaled[:, 1:end-3]
angles_perp = subspace_angle(N_perp_original, N_perp_new)
println("\nPrincipal angles between identifiable subspaces:")
println("  Range: $(round(minimum(angles_perp), digits=2))° - $(round(maximum(angles_perp), digits=2))°")

# Also check if transformation gives similar coordinates
ψ_original = exp.(A_T_original * log.(θ_grid))
ψ_new = exp.(A_T_scaled * log.(θ_grid))

println("\n=== Key coordinates at gridded MLE ===")
# Find which row corresponds to K₁/β₁ and β₁K₁ in each transformation
# by checking exponents (row 9 should have K/β pattern, row 17 should have βK pattern)

println("Using original transformation (indices 9, 17):")
println("  ψ₉ (K₁/β₁) = $(round(ψ_original[9], digits=2))")
println("  ψ₁₇ (β₁K₁) = $(round(ψ_original[17], digits=4))")

# For new transformation, find matching rows
println("\nUsing new transformation:")
println("  ψ₉ = $(round(ψ_new[9], digits=2))")
println("  ψ₁₇ = $(round(ψ_new[17], digits=4))")

println("\n=== Conclusion ===")
max_null_angle = maximum(angles)
if max_null_angle < 5.0
    println("✓ Non-identifiable subspaces match (max angle $(round(max_null_angle, digits=2))°)")
    println("✓ IIR is robust to MLE variation")
else
    println("⚠ Non-identifiable subspaces differ (max angle $(round(max_null_angle, digits=2))°)")
end
