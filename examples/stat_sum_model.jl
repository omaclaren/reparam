# Include ReparamTools.jl code if not already loaded
if !@isdefined(ReparamTools)
    include("../ReparamTools.jl")
    println("✓ ReparamTools module included")
else
    println("✓ ReparamTools module already included")
end

# Load required packages
using .ReparamTools
using Plots
using Distributions
using LinearAlgebra
using Random

# Set random seed for reproducibility
Random.seed!(42)

# ============================================================
# Model Definition: Sum of Two Independent Poisson Limit Models
# ============================================================
# Parameters: θ = [n₁, p₁, n₂, p₂]
# Auxiliary mapping: ϕ(θ) = [n₁p₁ + n₂p₂, n₁p₁ + n₂p₂]
# Distribution: Y ~ N(μ, σ²) where μ = σ² = n₁p₁ + n₂p₂
#
# Nested non-identifiability structure:
# - Only sum n₁p₁ + n₂p₂ is identifiable
# - Individual products n₁p₁ and n₂p₂ are not identifiable
# - Components within each product are not identifiable
# ============================================================

println("\n" * "="^60)
println("Sum of Two Independent Poisson Limit Models")
println("="^60)
println("Parameters: θ = [n₁, p₁, n₂, p₂]")
println("Auxiliary mapping: ϕ(θ) = [n₁p₁ + n₂p₂, n₁p₁ + n₂p₂]")
println("Distribution: Y ~ N(μ, σ²) where μ = σ² = n₁p₁ + n₂p₂")
println("="^60)

# Parameter -> data parameter mapping
ϕ_xy = xy -> [xy[1]*xy[2] + xy[3]*xy[4], xy[1]*xy[2] + xy[3]*xy[4]]

# --------------------------------------------------------
# Setup and Data Generation
# --------------------------------------------------------

# Parameter -> distribution mapping
distrib_xy = xy -> Normal(ϕ_xy(xy)[1], sqrt(ϕ_xy(xy)[2]))

# Variables and bounds
varnames = Dict(
    "ψ1" => "n_1", "ψ2" => "p_1", "ψ3" => "n_2", "ψ4" => "p_2",
    "ψ1_save" => "n1", "ψ2_save" => "p1", "ψ3_save" => "n2", "ψ4_save" => "p2"
)

# Parameter bounds
n_min, n_max = 0.1, 500.0
p_min, p_max = 1e-4, 1.0

xy_lower_bounds = [n_min, p_min, n_min, p_min]
xy_upper_bounds = [n_max, p_max, n_max, p_max]

# Initial guess for optimization
xy_initial = [50.0, 0.3, 50.0, 0.3]

# True parameter values
# Use asymmetric products to avoid SVD finding rotated geometric-mean basis
n1_true, p1_true = 80.0, 0.1   # Product 1: n₁p₁ = 8
n2_true, p2_true = 40.0, 0.3   # Product 2: n₂p₂ = 12
                                # Sum: n₁p₁ + n₂p₂ = 20
xy_true = [n1_true, p1_true, n2_true, p2_true]

# Note: At symmetric parameter sets (e.g., n₁p₁ = n₂p₂), the SVD can pick
# rotated bases in the identifiable subspace. Using asymmetric values ensures
# Stage 1 aligns with individual products [n₁p₁, n₂p₂].

println("\nTrue parameters:")
println("  n₁ = ", n1_true, ", p₁ = ", p1_true, " → n₁p₁ = ", n1_true*p1_true)
println("  n₂ = ", n2_true, ", p₂ = ", p2_true, " → n₂p₂ = ", n2_true*p2_true)
println("  Sum: n₁p₁ + n₂p₂ = ", n1_true*p1_true + n2_true*p2_true)

# Generate synthetic data
N_samples = 10
Random.seed!(42)
data = rand(distrib_xy(xy_true), N_samples)
println("\nGenerated ", N_samples, " observations")
println("Sample mean: ", round(mean(data), digits=2))
println("Sample variance: ", round(var(data), digits=2))

# --------------------------------------------------------
# Original Parameterization Analysis
# --------------------------------------------------------
model_name = "stat_sum_model_xy"
grid_steps = [300]  # Reduced from 500 for runtime
dim_all = length(xy_initial)
indices_all = 1:dim_all

println("\n" * "="^60)
println("Original Parameterization: [n₁, p₁, n₂, p₂]")
println("="^60)

# Construct likelihood
lnlike_xy = construct_lnlike_xy(distrib_xy, data)

# Point estimation (MLE)
target_indices = []  # empty for MLE
xy_MLE, lnlike_xy_MLE = profile_target(lnlike_xy, target_indices,
    xy_lower_bounds, xy_upper_bounds,
    xy_initial; grid_steps=grid_steps)

println("\nMLE in original coordinates:")
println("  n₁ = ", round(xy_MLE[1], digits=2))
println("  p₁ = ", round(xy_MLE[2], digits=4))
println("  n₂ = ", round(xy_MLE[3], digits=2))
println("  p₂ = ", round(xy_MLE[4], digits=4))
println("  n₁p₁ = ", round(xy_MLE[1]*xy_MLE[2], digits=2))
println("  n₂p₂ = ", round(xy_MLE[3]*xy_MLE[4], digits=2))
println("  Sum = ", round(xy_MLE[1]*xy_MLE[2] + xy_MLE[3]*xy_MLE[4], digits=2))

# Quadratic approximation at MLE
lnlike_xy_ellipse, H_xy_ellipse = construct_ellipse_lnlike_approx(lnlike_xy, xy_MLE)

# Eigenanalysis of Fisher Information
evals, evecs = eigen(H_xy_ellipse; sortby = x -> -real(x))
println("\nFisher Information eigenvalues:")
for (i, eval_i) in enumerate(evals)
    println("  λ", i, " = ", round(eval_i, digits=6))
end

# SVD of auxiliary mapping
J_ϕ_xy = compute_ϕ_Jacobian(ϕ_xy, xy_MLE)
U_xy, S_xy, Vt_xy = svd(J_ϕ_xy)
println("\nSVD of Jacobian in original coordinates:")
println("  Jacobian size: ", size(J_ϕ_xy))
println("  Singular values: ", round.(S_xy, digits=6))
println("  Expected rank: 1 (only sum is identifiable)")
println("  Null space dimension: ", dim_all - count(>(sqrt(eps())*maximum(S_xy)), S_xy))

# 1D Profiles for all 4 parameters
println("\nComputing 1D profiles for all 4 parameters...")
for i in 1:dim_all
    target_index = i
    nuisance_indices = setdiff(indices_all, target_index)
    nuisance_guess = xy_MLE[nuisance_indices]

    # Profile full likelihood
    ψω_values, lnlike_ψ_values = profile_target(lnlike_xy, target_index,
        xy_lower_bounds, xy_upper_bounds,
        nuisance_guess; grid_steps=grid_steps)

    # Extract profiled parameter values
    ψ_values = [ψω[target_index] for ψω in ψω_values]

    # Plot profiles
    plot_1D_profile(model_name, ψ_values, lnlike_ψ_values,
        varnames["ψ"*string(i)];
        varname_save=varnames["ψ"*string(i)*"_save"],
        ψ_true=xy_true[i], ψ_MLE=xy_MLE[i])
end

# 2D Profiles - selective pairs to manage runtime
# Commented out for debugging - uncomment when needed
# println("\nComputing selective 2D profiles...")
#
# # Define key pairs: (n₁,p₁), (n₂,p₂), (n₁,n₂), (p₁,p₂)
# key_pairs = [(1,2), (3,4), (1,3), (2,4)]
# pair_descriptions = [
#     "(n₁, p₁) - Ridge within first product",
#     "(n₂, p₂) - Ridge within second product",
#     "(n₁, n₂) - n constraint",
#     "(p₁, p₂) - p constraint"
# ]
#
# for (pair_idx, (i,j)) in enumerate(key_pairs)
#     println("  ", pair_descriptions[pair_idx])
#
#     target_indices_ij = [i,j]
#     nuisance_indices = setdiff(indices_all, target_indices_ij)
#     nuisance_guess = xy_MLE[nuisance_indices]
#     ψ_true_pair = xy_true[target_indices_ij]
#
#     # Create varnames for this pair
#     current_varnames = deepcopy(varnames)
#     current_varnames["ψ1"] = varnames["ψ"*string(i)]
#     current_varnames["ψ2"] = varnames["ψ"*string(j)]
#     current_varnames["ψ1_save"] = varnames["ψ"*string(i)*"_save"]
#     current_varnames["ψ2_save"] = varnames["ψ"*string(j)*"_save"]
#
#     # Profile full likelihood
#     ψω_values, lnlike_ψ_values = profile_target(lnlike_xy, target_indices_ij,
#         xy_lower_bounds, xy_upper_bounds,
#         nuisance_guess; grid_steps=grid_steps)
#
#     # Extract profiled parameter values
#     ψ_values = [ψω[target_indices_ij] for ψω in ψω_values]
#
#     # Plot contours
#     plot_2D_contour(model_name, ψ_values, lnlike_ψ_values,
#         current_varnames; ψ_true=ψ_true_pair, ψ_MLE=xy_MLE)
# end

println("\nOriginal parameterization analysis complete.")
println("Note: All parameters show poor individual identifiability")
println("      due to nested non-identifiability structure.")

# --------------------------------------------------------
# Log Parameterization Analysis
# --------------------------------------------------------
model_name = "stat_sum_model_log"

println("\n" * "="^60)
println("Log Parameterization: [ln n₁, ln p₁, ln n₂, ln p₂]")
println("="^60)

# Coordinate transformation
xytoXY_log(xy) = log.(xy)
XYtoxy_log(XY) = exp.(XY)

# Transform bounds and parameters
XY_log_lower_bounds = log.(xy_lower_bounds)
XY_log_upper_bounds = log.(xy_upper_bounds)
XY_log_initial = xytoXY_log(xy_initial)
XY_log_true = xytoXY_log(xy_true)

# Transform likelihood, distribution, and phi mapping
lnlike_XY_log = construct_lnlike_XY(lnlike_xy, XYtoxy_log)
distrib_XY_log = construct_distrib_XY(distrib_xy, XYtoxy_log)
ϕ_XY_log = construct_ϕ_XY(ϕ_xy, XYtoxy_log)

# Update variable names for log coordinates
varnames["ψ1"] = "\\ln\\ n_1"
varnames["ψ2"] = "\\ln\\ p_1"
varnames["ψ3"] = "\\ln\\ n_2"
varnames["ψ4"] = "\\ln\\ p_2"
varnames["ψ1_save"] = "ln_n1"
varnames["ψ2_save"] = "ln_p1"
varnames["ψ3_save"] = "ln_n2"
varnames["ψ4_save"] = "ln_p2"

# Point estimation in log coordinates
target_indices = []  # empty for MLE
XY_log_MLE, lnlike_XY_log_MLE = profile_target(lnlike_XY_log, target_indices,
    XY_log_lower_bounds, XY_log_upper_bounds,
    XY_log_initial; grid_steps=grid_steps)

println("\nMLE in log coordinates:")
for i in 1:dim_all
    println("  ", varnames["ψ"*string(i)], " = ", round(XY_log_MLE[i], digits=4))
end

# Back-transform to check
xy_MLE_from_log = exp.(XY_log_MLE)
println("\nBack-transformed MLE:")
println("  n₁ = ", round(xy_MLE_from_log[1], digits=2))
println("  p₁ = ", round(xy_MLE_from_log[2], digits=4))
println("  n₂ = ", round(xy_MLE_from_log[3], digits=2))
println("  p₂ = ", round(xy_MLE_from_log[4], digits=4))
println("  Sum = ", round(xy_MLE_from_log[1]*xy_MLE_from_log[2] + xy_MLE_from_log[3]*xy_MLE_from_log[4], digits=2))

# Quadratic approximation at MLE
lnlike_XY_log_ellipse, H_XY_log_ellipse = construct_ellipse_lnlike_approx(lnlike_XY_log, XY_log_MLE)

# Eigenanalysis in log coordinates
evals_log, evecs_log = eigen(H_XY_log_ellipse; sortby = x -> -real(x))
println("\nFisher Information eigenvalues in log coordinates:")
for (i, eval_i) in enumerate(evals_log)
    println("  λ", i, " = ", round(eval_i, digits=6))
end

# SVD of auxiliary mapping in log coordinates
J_ϕ_XY_log, U_XY_log, S_XY_log, Vt_XY_log = compute_ϕ_Jacobian(ϕ_XY_log, XY_log_MLE, compute_svd=true)
println("\nSVD of Jacobian in log coordinates:")
println("  Jacobian size: ", size(J_ϕ_XY_log))
println("  Singular values: ", round.(S_XY_log, digits=6))
rtol_J = sqrt(eps())
rankJ_log = count(>(rtol_J * maximum(S_XY_log)), S_XY_log)
println("  Numerical rank: ", rankJ_log)
println("  Null space dimension: ", dim_all - rankJ_log)

println("\nLog parameterization analysis complete.")
println("Ready for invariant subspace analysis...")

# ============================================================
# STAGE 1: Invariant Subspace Analysis in Log Coordinates
# Goal: Identify monomial combinations (products like n₁p₁, n₂p₂)
# ============================================================

println("\n" * "="^60)
println("STAGE 1: Monomial Identification (f=log, f⁻¹=exp)")
println("="^60)
println("Applying Algorithm 1 to identify monomial (product) combinations")
println("Expected: Rank 1, 3D invariant null space (image reparameterization)")

# Apply find_invariant_subspace at true parameters for clean structure
XY_log_true = log.(xy_true)
S_stage1, N_stage1, N_perp_stage1, rank_stage1 = find_invariant_subspace(ϕ_XY_log, XY_log_true)

println("\nStage 1 Results:")
println("  Singular values: ", round.(S_stage1, digits=6))
println("  Numerical rank: ", rank_stage1)
println("  Potentially identifiable space dimension (N_perp): ", size(N_perp_stage1, 2))
println("  Invariant null space dimension (N): ", size(N_stage1, 2))

# Determine type of reparameterization
null_space_dim_stage1 = length(XY_log_MLE) - rank_stage1
if size(N_stage1, 2) == null_space_dim_stage1 && null_space_dim_stage1 > 0
    println("\nReparameterization type: Minimal image")
    println("  (Full null space is invariant)")
    reparam_type_stage1 = "minimal_image"
elseif size(N_stage1, 2) > 0 && size(N_stage1, 2) < null_space_dim_stage1
    println("\nReparameterization type: Image (not minimal)")
    println("  (Partial null space is invariant: ", size(N_stage1, 2), " of ", null_space_dim_stage1, " dimensions)")
    reparam_type_stage1 = "image"
elseif size(N_stage1, 2) == 0 && null_space_dim_stage1 > 0
    println("\nReparameterization type: Image (not minimal)")
    println("  (No invariant null space detected)")
    reparam_type_stage1 = "image"
else
    println("\nReparameterization type: Appears structurally identifiable")
    reparam_type_stage1 = "identifiable"
end

# Display subspaces
if size(N_stage1, 2) > 0
    println("\nInvariant null space basis N (columns):")
    display(N_stage1)
end

println("\nIdentifiable space basis N_perp (columns):")
display(N_perp_stage1)

# Interpretability pass: rotate N_perp to align with sparse parameter-local combinations
println("\nApplying interpretability pass to N_perp...")
k = size(N_perp_stage1, 2)  # Number of potentially identifiable directions
n_params = size(N_perp_stage1, 1)

# Step 1: Cluster parameters based on their participation in N_perp columns
# Simple greedy clustering: assign each parameter to the column where it has highest absolute loading
row_loadings = abs.(N_perp_stage1)
clusters = zeros(Int, n_params)
for i in 1:n_params
    # Find which column this parameter loads most strongly on
    clusters[i] = argmax(row_loadings[i, :])
end

println("  Parameter clusters: ", clusters)

# Step 2: Build template directions - one per cluster
T = zeros(n_params, k)
for i in 1:k
    # Find parameters in cluster i
    cluster_params = findall(clusters .== i)
    T[cluster_params, i] .= 1.0
end
println("  Template matrix T (columns are cluster indicators):")
display(T)

# Step 3: Project templates into span(N_perp)
P = N_perp_stage1 * (N_perp_stage1' * T)
println("  Projected templates P:")
display(P)

# Zero out tiny entries
for j in 1:size(P, 2)
    col = P[:, j]
    col[abs.(col) .< 1e-10] .= 0.0
    P[:, j] = col
end

# Step 4: Orthonormalize the projections
Q, R = qr(P)
N_perp_rotated = Matrix(Q[:, 1:k])
println("  Rotated N_perp (sparse, orthonormal):")
display(N_perp_rotated)

# Use the rotated basis instead of the original
N_perp_stage1 = N_perp_rotated

# Construct full transformation matrix A1
# CRITICAL: For rotated (sparse) N_perp, just normalize to {0, ±1} pattern
println("\nConstructing Stage 1 transformation matrix A1...")

# Apply scale_and_round to get {0, ±1} pattern
N_perp_scaled = scale_and_round(N_perp_stage1; round_within=0.5)
println("  Scaled N_perp block (after scale_and_round): ", size(N_perp_scaled))
display(N_perp_scaled)

# Keep invariant null space orthonormal (don't scale - keeps transformation well-conditioned)
N_stage1_clean = copy(N_stage1)
for j in 1:size(N_stage1_clean, 2)
    col = N_stage1_clean[:, j]
    col_norm = norm(col)
    # Zero out entries below eps() * norm
    col[abs.(col) .< eps() * col_norm] .= 0.0
    N_stage1_clean[:, j] = col
end
println("  Cleaned N block (orthonormal): ", size(N_stage1_clean))

# Combine scaled potentially identifiable and orthonormal invariant blocks
A1_full_T = hcat(N_perp_scaled, N_stage1_clean)
println("  Combined matrix size: ", size(A1_full_T))

# Transpose to get transformation matrix (rows = parameter combinations)
A1_full = A1_full_T'
println("  Transformation matrix A1 size: ", size(A1_full))

println("\nStage 1 Transformation Matrix A1 (rows = combinations):")
display(A1_full)

# Display combinations
println("\nStage 1 combinations (rows of A1):")
for i in 1:size(A1_full, 1)
    row = A1_full[i, :]
    if i <= size(N_perp_stage1, 2)
        println("  Combination ", i, " (potentially identifiable): ", round.(row, digits=3))
    else
        println("  Combination ", i, " (invariant): ", round.(row, digits=3))
    end
end

# Symbolic representation of Stage 1 transformation
println("\nSymbolic Stage 1 transformation: θ¹ = exp(A1 * log(θ))")
for i in 1:size(A1_full, 1)
    row = A1_full[i, :]
    terms = String[]
    for j in 1:length(row)
        coef = row[j]
        if abs(coef) > 1e-6
            var_name = ["n₁", "p₁", "n₂", "p₂"][j]
            if abs(coef - 1.0) < 1e-6
                push!(terms, var_name)
            elseif abs(coef + 1.0) < 1e-6
                push!(terms, var_name * "⁻¹")
            else
                push!(terms, var_name * "^" * string(round(coef, digits=3)))
            end
        end
    end
    if !isempty(terms)
        println("  θ¹[$i] = ", join(terms, " * "))
    else
        println("  θ¹[$i] = 1")
    end
end

# Define Stage 1 coordinate transformations
xy_to_stage1(xy) = exp.(A1_full * log.(xy))
stage1_to_xy(θ1) = exp.(inv(A1_full) * log.(θ1))

# Verify transformation at MLE
xy_MLE_check = exp.(XY_log_MLE)
stage1_MLE = xy_to_stage1(xy_MLE_check)
println("\nStage 1 MLE (θ¹ = exp(A1 * log(θ))):")
for i in 1:length(stage1_MLE)
    println("  θ¹[", i, "] = ", round(stage1_MLE[i], digits=4))
end

println("\n" * "-"^60)
println("Stage 1 Summary:")
println("  Rank: ", rank_stage1)
println("  Potentially identifiable dimensions: ", size(N_perp_stage1, 2))
println("  Invariant null dimensions: ", size(N_stage1, 2))
println("  Type: ", reparam_type_stage1)
println("-"^60)

# ============================================================
# STAGE 2: Linear Combination Identification
# Goal: Identify sum of monomials (n₁p₁ + n₂p₂)
# ============================================================

println("\n" * "="^60)
println("STAGE 2: Linear Combination Identification (f=id)")
println("="^60)

# Define auxiliary mapping in Stage 1 coordinates

ϕ_stage2 = θ1 -> ϕ_xy(stage1_to_xy(θ1))

# Compute Jacobian of Stage 2 auxiliary mapping at Stage 1 MLE
J_ϕ_stage2 = compute_ϕ_Jacobian(ϕ_stage2, stage1_MLE)
println("\nStage 2 Jacobian:")
println("  Size: ", size(J_ϕ_stage2))
println("  Jacobian matrix:")
display(J_ϕ_stage2)

# For f=identity, find_invariant_subspace should work directly
# The algorithm computes Jacobian via automatic differentiation, which works for any f
S_stage2, N_stage2, N_perp_stage2, rank_stage2 = find_invariant_subspace(ϕ_stage2, stage1_MLE)

println("\nStage 2 Results:")
println("  Singular values: ", round.(S_stage2, digits=6))
println("  Numerical rank: ", rank_stage2)
println("  Potentially identifiable space dimension (N_perp): ", size(N_perp_stage2, 2))
println("  Invariant null space dimension (N): ", size(N_stage2, 2))

# Determine type of reparameterization
null_space_dim_stage2 = length(stage1_MLE) - rank_stage2
if size(N_stage2, 2) == null_space_dim_stage2 && null_space_dim_stage2 > 0
    println("\nReparameterization type: Minimal image")
    println("  (Full null space is invariant)")
    reparam_type_stage2 = "minimal_image"
elseif size(N_stage2, 2) > 0 && size(N_stage2, 2) < null_space_dim_stage2
    println("\nReparameterization type: Image (not minimal)")
    println("  (Partial null space is invariant: ", size(N_stage2, 2), " of ", null_space_dim_stage2, " dimensions)")
    reparam_type_stage2 = "image"
else
    println("\nReparameterization type: Appears structurally identifiable")
    reparam_type_stage2 = "identifiable"
end

# Display subspaces
if size(N_stage2, 2) > 0
    println("\nInvariant null space basis N (columns):")
    display(N_stage2)
end

println("\nIdentifiable space basis N_perp (columns):")
display(N_perp_stage2)

# Construct full transformation matrix A2
# Apply scale_and_round to N_perp, keep N orthonormal
println("\nConstructing Stage 2 transformation matrix A2...")

# Apply scale_and_round to potentially identifiable block
N_perp2_scaled = scale_and_round(N_perp_stage2; round_within=0.5)
println("  Scaled N_perp block: ", size(N_perp2_scaled))

# Keep invariant null space orthonormal (don't scale - keeps transformation well-conditioned)
N_stage2_clean = copy(N_stage2)
for j in 1:size(N_stage2_clean, 2)
    col = N_stage2_clean[:, j]
    col_norm = norm(col)
    # Zero out entries below eps() * norm
    col[abs.(col) .< eps() * col_norm] .= 0.0
    # Unit normalize
    col = col / norm(col)
    # Enforce sign convention: make largest absolute entry positive
    max_idx = argmax(abs.(col))
    if col[max_idx] < 0
        col = -col
    end
    N_stage2_clean[:, j] = col
end
println("  Cleaned N block (orthonormal): ", size(N_stage2_clean))

# Combine blocks
A2_full_T = hcat(N_perp2_scaled, N_stage2_clean)
A2_full = A2_full_T'

println("\nStage 2 Transformation Matrix A2 (rows = combinations):")
display(A2_full)

# Interpret the Stage 2 combinations
println("\nStage 2 combinations (rows of A2):")
for i in 1:size(A2_full, 1)
    row = A2_full[i, :]
    if i <= size(N_perp_stage2, 2)
        println("  Combination ", i, " (potentially identifiable): ", round.(row, digits=2))
    else
        println("  Combination ", i, " (invariant): ", round.(row, digits=2))
    end
end

# Symbolic representation of Stage 2 transformation
println("\nSymbolic Stage 2 transformation: ψ = A2 * θ¹")
for i in 1:size(A2_full, 1)
    row = A2_full[i, :]
    terms = String[]
    for j in 1:length(row)
        coef = row[j]
        if abs(coef) > 1e-6
            if abs(coef - 1.0) < 1e-6
                push!(terms, "θ¹[$j]")
            elseif abs(coef + 1.0) < 1e-6
                push!(terms, "-θ¹[$j]")
            else
                push!(terms, string(round(coef, digits=2)) * "*θ¹[$j]")
            end
        end
    end
    if !isempty(terms)
        println("  ψ[$i] = ", join(terms, " + "))
    else
        println("  ψ[$i] = 0")
    end
end

# Define overall transformation: ψ(θ) = A2 * exp(A1 * log(θ))
xy_to_final(xy) = A2_full * exp.(A1_full * log.(xy))

# Inverse: θ¹ = A2 \ ψ, then θ = exp(inv(A1) * log(θ¹))
# Need to check that θ¹ > 0 for log to be defined
function final_to_xy(ψ)
    θ1 = A2_full \ ψ
    # Check if all components are positive
    if any(θ1 .<= 0)
        # Return invalid point that will give -Inf likelihood
        return fill(NaN, length(θ1))
    end
    return exp.(inv(A1_full) * log.(θ1))
end

# Compute final MLE and true values
final_MLE = xy_to_final(xy_MLE_check)
final_true = xy_to_final(xy_true)

println("\nFinal (Stage 2) MLE:")
for i in 1:length(final_MLE)
    println("  ψ[", i, "] = ", round(final_MLE[i], digits=4))
end

println("\n" * "-"^60)
println("Stage 2 Summary:")
println("  Rank: ", rank_stage2)
println("  Potentially identifiable dimensions: ", size(N_perp_stage2, 2))
println("  Invariant null dimensions: ", size(N_stage2, 2))
println("  Type: ", reparam_type_stage2)
println("-"^60)

println("\n" * "="^60)
println("Sequential Application Complete")
println("="^60)
println("Stage 1: Rank ", rank_stage1, ", ", size(N_stage1, 2), "D invariant null space")
println("Stage 2: Rank ", rank_stage2, ", ", size(N_stage2, 2), "D invariant null space")
println("Overall: ψ(θ) = A2 * exp(A1 * log(θ))")
println("="^60)

# ============================================================
# Final Diagnostics: Analysis in Two-Stage IIR Coordinates
# ============================================================

model_name = "stat_sum_model_final"

println("\n" * "="^60)
println("Final Diagnostics: Two-Stage IIR Parameterization")
println("="^60)

# Transform likelihood to final coordinates with safety check
function lnlike_final_safe(ψ)
    xy = final_to_xy(ψ)
    # Check if transformation returned NaN (invalid point)
    if any(isnan.(xy))
        return -Inf
    end
    return lnlike_xy(xy)
end
lnlike_final = lnlike_final_safe

# Compute Fisher Information to determine which parameters are actually identifiable
lnlike_final_ellipse_prelim, H_final_ellipse_prelim = construct_ellipse_lnlike_approx(lnlike_final, final_MLE)
evals_prelim = eigvals(H_final_ellipse_prelim; sortby = x -> -real(x))

# Count truly identifiable parameters (eigenvalue significantly > 0)
n_identifiable = count(evals_prelim .> 1e-3)
println("\nFisher eigenvalues (preliminary): ", round.(evals_prelim, digits=6))
println("Number of identifiable parameters: ", n_identifiable)

# Set bounds for final coordinates
# Need to ensure A2_full \ ψ > 0 for all ψ in bounds
final_lower_bounds = fill(-50.0, dim_all)
final_upper_bounds = fill(200.0, dim_all)

# Set bounds based on Fisher eigenvalues
for i in 1:dim_all
    if i <= n_identifiable
        # Identifiable: use tighter bounds around MLE
        if final_MLE[i] > 0
            margin = abs(final_MLE[i]) * 0.5
            final_lower_bounds[i] = max(final_MLE[i] - margin, final_MLE[i] * 0.1)
            final_upper_bounds[i] = final_MLE[i] + margin * 2
        else
            # Near zero: use symmetric range appropriate to scale
            final_lower_bounds[i] = -1.0
            final_upper_bounds[i] = 1.0
        end
    else
        # Non-identifiable: use very wide bounds (likelihood should be flat)
        # Distinguish between potentially identifiable (can be negative) and invariant null space (must be positive)
        n_pot_id = size(N_perp_stage2, 2)
        if i <= n_pot_id
            # From potentially identifiable space: can be negative
            if abs(final_MLE[i]) < 0.1
                # Near-zero MLE: use symmetric range
                final_lower_bounds[i] = -10.0
                final_upper_bounds[i] = 10.0
            else
                # Non-zero MLE: wide range around it
                final_lower_bounds[i] = final_MLE[i] - 20.0
                final_upper_bounds[i] = final_MLE[i] + 20.0
            end
        else
            # From invariant null space - check if needs positive-only or can be negative
            # Test: for ψ[2] (often a difference), allow negative; for others check MLE
            if i == 2
                # Special case: ψ[2] is often a difference, allow negative
                final_lower_bounds[i] = -10.0
                final_upper_bounds[i] = 10.0
            elseif final_MLE[i] > 0.1
                # Positive MLE: use wide range
                final_lower_bounds[i] = max(final_MLE[i] * 0.01, 1e-6)
                final_upper_bounds[i] = max(final_MLE[i] * 10.0, 200.0)
            else
                # Near-zero but must stay positive
                final_lower_bounds[i] = 1e-6
                final_upper_bounds[i] = 1.0
            end
        end
    end
end

# Verify bounds are valid at MLE
θ1_at_MLE = A2_full \ final_MLE
println("\nBounds check at MLE:")
println("  θ¹ = A2 \\ ψ_MLE: ", round.(θ1_at_MLE, digits=4))
if any(θ1_at_MLE .<= 0)
    println("  WARNING: MLE maps to non-positive θ¹")
else
    println("  All components positive ✓")
end

println("\nFinal parameter bounds:")
for i in 1:dim_all
    println("  ψ[$i]: [", final_lower_bounds[i], ", ", final_upper_bounds[i], "]")
end

final_initial = final_MLE

# Update variable names for final coordinates with symbolic expressions
varnames["ψ1"] = "\$n_1p_1 + n_2p_2\$"
varnames["ψ2"] = "\$0.71(n_1p_1 - n_2p_2)\$"
varnames["ψ3"] = "\$(n_1/p_1)^{0.71}\$"
varnames["ψ4"] = "\$(p_2/n_2)^{0.71}\$"
varnames["ψ1_save"] = "n1p1_plus_n2p2"
varnames["ψ2_save"] = "n2p2_minus_n1p1_scaled"
varnames["ψ3_save"] = "n1_over_p1_pow"
varnames["ψ4_save"] = "n2_over_p2_pow_inv"

# Point estimation in final coordinates
target_indices = []
final_MLE_opt, lnlike_final_MLE = profile_target(lnlike_final, target_indices,
    final_lower_bounds, final_upper_bounds,
    final_initial; grid_steps=grid_steps)

println("\nMLE in final coordinates:")
for i in 1:dim_all
    println("  ψ[", i, "] = ", round(final_MLE_opt[i], digits=4))
end

# Quadratic approximation
lnlike_final_ellipse, H_final_ellipse = construct_ellipse_lnlike_approx(lnlike_final, final_MLE_opt)

# Eigenanalysis
evals_final, evecs_final = eigen(H_final_ellipse; sortby = x -> -real(x))
println("\nFisher Information eigenvalues in final coordinates:")
for (i, eval_i) in enumerate(evals_final)
    println("  λ", i, " = ", round(eval_i, digits=6))
    if i <= size(N_perp_stage2, 2)
        println("    (potentially identifiable)")
    else
        println("    (non-identifiable)")
    end
end


# 1D Profiles for all 4 final parameters
println("\nComputing 1D profiles in final coordinates...")
for i in 1:dim_all
    target_index = i
    nuisance_indices = setdiff(indices_all, target_index)
    nuisance_guess = final_MLE_opt[nuisance_indices]

    # Profile full likelihood
    ψω_values, lnlike_ψ_values = profile_target(lnlike_final, target_index,
        final_lower_bounds, final_upper_bounds,
        nuisance_guess; grid_steps=grid_steps)

    # Extract profiled parameter values
    ψ_values = [ψω[target_index] for ψω in ψω_values]

    # Plot profile
    plot_1D_profile(model_name, ψ_values, lnlike_ψ_values,
        varnames["ψ"*string(i)];
        varname_save=varnames["ψ"*string(i)*"_save"],
        ψ_true=final_true[i], ψ_MLE=final_MLE_opt[i])

    println("  Profile ", i, ": complete")
end

# 2D Profile: Commented out for now
# println("\nComputing key 2D profile: (ψ1, ψ2)")
# i, j = 1, 2
# target_indices_ij = [i, j]
# nuisance_indices = setdiff(indices_all, target_indices_ij)
# nuisance_guess = final_MLE_opt[nuisance_indices]
# ψ_true_pair = final_true[target_indices_ij]
#
# current_varnames = deepcopy(varnames)
# current_varnames["ψ1"] = varnames["ψ"*string(i)]
# current_varnames["ψ2"] = varnames["ψ"*string(j)]
# current_varnames["ψ1_save"] = varnames["ψ"*string(i)*"_save"]
# current_varnames["ψ2_save"] = varnames["ψ"*string(j)*"_save"]
#
# # Profile full likelihood
# ψω_values, lnlike_ψ_values = profile_target(lnlike_final, target_indices_ij,
#     final_lower_bounds, final_upper_bounds,
#     nuisance_guess; grid_steps=[100])  # Lower resolution for 2D
#
# # Extract profiled parameter values
# ψ_values = [ψω[target_indices_ij] for ψω in ψω_values]
#
# # Plot contour
# plot_2D_contour(model_name, ψ_values, lnlike_ψ_values,
#     current_varnames; ψ_true=ψ_true_pair, ψ_MLE=final_MLE_opt)

println("\n" * "="^60)
println("Summary")
println("="^60)
println("Transformation matrices:")
println("  A1: ", size(A1_full))
println("  A2: ", size(A2_full))
println("\nDimensional reduction:")
println("  Stage 1: ", rank_stage1, "D potentially identifiable, ", size(N_stage1, 2), "D invariant")
println("  Stage 2: ", rank_stage2, "D potentially identifiable, ", size(N_stage2, 2), "D invariant")
println("\nOverall transformation: ψ(θ) = A2 * exp(A1 * log(θ))")

# Compute and display the full symbolic transformation
println("\nFull symbolic transformation:")
for i in 1:size(A2_full, 1)
    # For ψ[i], we need to compose: A2[i,:] * exp(A1 * log(θ))
    # This means ψ[i] = Σⱼ A2[i,j] * exp(A1[j,:] ⋅ log(θ))
    # = Σⱼ A2[i,j] * exp(Σₖ A1[j,k] * log(θₖ))
    # = Σⱼ A2[i,j] * ∏ₖ θₖ^A1[j,k]

    a2_row = A2_full[i, :]
    terms = String[]

    for j in 1:length(a2_row)
        coef = a2_row[j]
        if abs(coef) > 1e-6
            # Get A1[j,:] to build the monomial
            a1_row = A1_full[j, :]
            monomial_parts = String[]
            for k in 1:length(a1_row)
                exp = a1_row[k]
                if abs(exp) > 1e-6
                    var_name = ["n₁", "p₁", "n₂", "p₂"][k]
                    if abs(exp - 1.0) < 1e-6
                        push!(monomial_parts, var_name)
                    elseif abs(exp + 1.0) < 1e-6
                        push!(monomial_parts, var_name * "⁻¹")
                    else
                        push!(monomial_parts, var_name * "^" * string(round(exp, digits=2)))
                    end
                end
            end

            if !isempty(monomial_parts)
                monomial = join(monomial_parts, "*")
                if abs(coef - 1.0) < 1e-6
                    push!(terms, monomial)
                elseif abs(coef + 1.0) < 1e-6
                    push!(terms, "-(" * monomial * ")")
                else
                    push!(terms, string(round(coef, digits=2)) * "*(" * monomial * ")")
                end
            end
        end
    end

    if !isempty(terms)
        println("  ψ[$i] = ", join(terms, " + "))
    else
        println("  ψ[$i] = 0")
    end
end

println("="^60)
