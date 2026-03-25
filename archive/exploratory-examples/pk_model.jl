# Two-Compartment Pharmacokinetic Model with Michaelis-Menten Clearance
# Based on Meshkat, Anderson & DiStefano (2011, 2014)
#
# Model has 8 parameters with 5 identifiable combinations:
#   q₁ = b₁c₁           (input-output scaling product)
#   q₂ = c₁K_M          (output scaling × MM constant)
#   q₃ = k₀₂ + k₁₂      (sum of rates)
#   q₄ = c₁V_M k₁₂k₂₁   (4-parameter product)
#   q₅ = c₁V_M(k₀₁+k₂₁) (product × sum)

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
using DifferentialEquations
using FactorLoadingMatrices

# Set random seed for reproducibility
Random.seed!(1234)

# --------------------------------------------------------
# Model Definition
# --------------------------------------------------------

# Define the 2-compartment PK model with Michaelis-Menten clearance
function pk_ode!(dx, x, θ, t)
    """
    Two-compartment PK model with Michaelis-Menten elimination.

    States:
    - x[1]: Amount in compartment 1 (central)
    - x[2]: Amount in compartment 2 (peripheral)

    Parameters:
    - θ[1] = b₁:   Input scaling (dose rate)
    - θ[2] = c₁:   Output scaling (concentration observation)
    - θ[3] = k₀₁:  Linear elimination from compartment 1
    - θ[4] = k₀₂:  Linear elimination from compartment 2
    - θ[5] = k₁₂:  Transfer rate 1→2
    - θ[6] = k₂₁:  Transfer rate 2→1
    - θ[7] = V_M:  Michaelis-Menten maximum velocity
    - θ[8] = K_M:  Michaelis-Menten constant
    """
    b₁, c₁, k₀₁, k₀₂, k₁₂, k₂₁, V_M, K_M = θ
    x₁, x₂ = x

    # Michaelis-Menten saturable clearance from compartment 1
    mm_clearance = (V_M * x₁) / (K_M + x₁)

    # Input function (simple bolus at t=0, then zero)
    u_input = t < 0.1 ? 1.0 : 0.0

    dx[1] = -(k₀₁ + k₁₂)*x₁ + k₂₁*x₂ - mm_clearance + b₁*u_input
    dx[2] = k₁₂*x₁ - (k₀₂ + k₂₁)*x₂
end

# ODE model solver
function solve_pk_ode(t_save, θ, x0; solver=Rodas4())
    """
    Solve PK model and return output measurements (y = c₁*x₁).

    Parameters:
    - t_save: Time grid points
    - θ: Parameter vector [b₁, c₁, k₀₁, k₀₂, k₁₂, k₂₁, V_M, K_M]
    - x0: Initial condition [x₁(0), x₂(0)]
    - solver: ODE solver (default: Rodas4() for stiff problems)

    Returns:
    - Vector of observed concentrations y(t) = c₁*x₁(t)
    """
    c₁ = θ[2]
    tspan = (0.0, maximum(t_save))
    prob = ODEProblem(pk_ode!, x0, tspan, θ)
    sol = solve(prob, solver, saveat=t_save, abstol=1e-12, reltol=1e-9)

    # Return observed output: y = c₁ * x₁
    return c₁ * sol[1, :]
end

# Creates a ϕ mapping function with fixed grid parameters
function create_ϕ_mapping(t, x0)
    """
    Create a ϕ mapping function from model parameters to observed outputs
    with fixed grid parameters.

    Parameters:
    - t: Time grid points
    - x0: Initial condition [x₁(0), x₂(0)]

    Returns:
    - ϕ mapping function from θ to observed concentrations
    """
    return θ -> solve_pk_ode(t, θ, x0)
end

# --------------------------------------------------------
# Setup and Data Generation
# --------------------------------------------------------

# Fine grid setup
T = 24.0  # hours
NT = 241
t = LinRange(0, T, NT)
indices_fine = 1:NT

# Observation grid setup
NT_obs = 13
indices_obs = 1:Int((NT-1)/(NT_obs-1)):NT
obs_matrix = construct_observation_matrix(indices_obs, indices_fine)
t_obs = t[indices_obs]

# Initial condition and observation parameters
x0 = [0.0, 0.0]  # Both compartments start empty
σ = 0.05  # Observation noise std dev

# --------------------------------------------------------
# True Parameters and Known Identifiable Combinations
# --------------------------------------------------------

# True parameter values (chosen to be identifiable combinations)
b₁_true = 2.0
c₁_true = 1.5
k₀₁_true = 0.2
k₀₂_true = 0.1
k₁₂_true = 0.3
k₂₁_true = 0.25
V_M_true = 1.0
K_M_true = 2.0

θ_true = [b₁_true, c₁_true, k₀₁_true, k₀₂_true, k₁₂_true, k₂₁_true, V_M_true, K_M_true]

# Known identifiable combinations (from Meshkat et al.)
println("\n" * "="^60)
println("Known Identifiable Combinations (Meshkat et al. 2014)")
println("="^60)
q₁_true = b₁_true * c₁_true
q₂_true = c₁_true * K_M_true
q₃_true = k₀₂_true + k₁₂_true
q₄_true = c₁_true * V_M_true * k₁₂_true * k₂₁_true
q₅_true = c₁_true * V_M_true * (k₀₁_true + k₂₁_true)

println("q₁ = b₁c₁             = ", q₁_true)
println("q₂ = c₁K_M            = ", q₂_true)
println("q₃ = k₀₂ + k₁₂        = ", q₃_true)
println("q₄ = c₁V_M k₁₂k₂₁     = ", q₄_true)
println("q₅ = c₁V_M(k₀₁ + k₂₁) = ", q₅_true)
println("="^60 * "\n")

# --------------------------------------------------------
# Define ϕ mapping and generate data
# --------------------------------------------------------

# Define ϕ mapping in original coordinates on fine grid
ϕ_func_θ = create_ϕ_mapping(t, x0)

# Parameter → data distribution (forward) mapping on fine grid
solver = Rodas4()
distrib_fine_θ = θ -> MvNormal(solve_pk_ode(t, θ, x0; solver=solver), σ^2*I(NT))

# Parameter → data distribution (forward) mapping on observation grid
distrib_θ = θ -> MvNormal(solve_pk_ode(t_obs, θ, x0; solver=solver), σ^2*I(NT_obs))

# Generate synthetic data
data = rand(distrib_θ(θ_true))

# Visualize data and true solution
p_data = scatter(t_obs, data, label="Data", markersize=4)
plot!(t, solve_pk_ode(t, θ_true, x0), label="True Solution",
      xlabel="Time (hours)", ylabel="Concentration",
      legend=:topright, linewidth=2)
display(p_data)

# --------------------------------------------------------
# Sequential IIR Analysis
# --------------------------------------------------------

println("\n" * "="^60)
println("SEQUENTIAL IIR ANALYSIS")
println("="^60)

# Stage 1 (f = log) - Identify monomial combinations
println("\n--- STAGE 1: Log transformation (monomials) ---")
println("Expected to find combinations: q₁, q₂, q₄")
println("These are products: b₁c₁, c₁K_M, c₁V_M k₁₂k₂₁")

# Apply find_invariant_subspace with f = log
# IMPORTANT: find_invariant_subspace works in TRANSFORMED space
# For f=log, we evaluate at log(θ), so create a composed function
ϕ_log = θ_log -> ϕ_func_θ(exp.(θ_log))

# Use log of true parameters for clean structure (in practice would use log(MLE))
θ_log_stage1 = log.(θ_true)

S_stage1, N_stage1, N_perp_stage1, rankJ_stage1 = find_invariant_subspace(
    ϕ_log, θ_log_stage1;
    rtolJ=sqrt(eps()),
    atolM=1e-10
)

println("\nStage 1 Results:")
println("  Rank of Jacobian: ", rankJ_stage1)
println("  Dimension of invariant null space: ", size(N_stage1, 2))
println("  Dimension of potentially identifiable subspace: ", size(N_perp_stage1, 2))
println("\n  Singular values:")
display(S_stage1)

println("\n  N_perp (potentially identifiable in log-space):")
display(N_perp_stage1)

println("\n  N (invariant null space in log-space):")
display(N_stage1)

# Apply Varimax rotation for interpretability
# CRITICAL: Keep orthonormal version for actual transformations!
if size(N_perp_stage1, 2) > 1
    println("\n  Applying Varimax rotation to N_perp for sparse structure...")
    N_perp_rotated_ortho = varimax_rotation(N_perp_stage1; n_restarts=200, threshold=1e-2)
    println("\n  Rotated N_perp (orthonormal, after Varimax):")
    display(N_perp_rotated_ortho)

    # For interpretation only: scale and round (breaks orthogonality!)
    println("\n  Applying scale_and_round for interpretation...")
    N_perp_scaled_interp = scale_and_round(N_perp_rotated_ortho; column_scales=ones(size(N_perp_rotated_ortho, 2)))
    println("\n  Scaled N_perp (for interpretation only):")
    display(N_perp_scaled_interp)

    # Use orthonormal version for actual transformation
    N_perp_for_transform = N_perp_rotated_ortho
else
    println("\n  Only one N_perp column - no Varimax rotation needed")
    N_perp_for_transform = N_perp_stage1
    N_perp_scaled_interp = scale_and_round(N_perp_stage1; column_scales=ones(size(N_perp_stage1, 2)))
    println("\n  Scaled N_perp (for interpretation):")
    display(N_perp_scaled_interp)
end

# Interpret Stage 1 combinations (using scaled version)
println("\n  Interpreting Stage 1 monomial combinations:")
param_names = ["b₁", "c₁", "k₀₁", "k₀₂", "k₁₂", "k₂₁", "V_M", "K_M"]
for col in 1:size(N_perp_scaled_interp, 2)
    coeffs = N_perp_scaled_interp[:, col]
    nonzero_idx = findall(x -> abs(x) > 0.01, coeffs)
    if !isempty(nonzero_idx)
        terms = [coeffs[i] ≈ 1 ? param_names[i] :
                 coeffs[i] ≈ -1 ? "1/$(param_names[i])" :
                 coeffs[i] > 0 ? "$(param_names[i])^$(coeffs[i])" :
                 "1/$(param_names[i])^$(abs(coeffs[i]))"
                 for i in nonzero_idx]
        println("    θ¹[$col] ~ ", join(terms, " × "))
    end
end

# Stage 2 (f = identity) - Identify sum combinations
println("\n--- STAGE 2: Identity transformation (sums) ---")
println("Expected to find combinations: q₃, q₅")
println("These involve sums: k₀₂+k₁₂, c₁V_M(k₀₁+k₂₁)")

# Construct transformation matrix from Stage 1
# CRITICAL: Use orthonormal version (not scaled) for proper transformation!
A1_full_T = hcat(N_perp_for_transform, N_stage1)
A1_full = A1_full_T'  # Transpose to get transformation matrix
A1_full_inv = inv(A1_full)  # Proper matrix inverse (not just transpose!)

println("\nStage 1 Transformation Matrix A1:")
println("  Size: ", size(A1_full))
display(A1_full)

# Create Stage 1 transformed ϕ function
# θ¹ = exp(A1 * log(θ))
# ϕ_stage1(θ¹) = ϕ(exp(A1_inv * log(θ¹)))
function transform_stage1(θ)
    """Transform θ → θ¹ using Stage 1 transformation"""
    return exp.(A1_full * log.(θ))
end

function inverse_transform_stage1(θ1)
    """Transform θ¹ → θ using inverse of Stage 1 transformation"""
    # Use proper matrix inverse (not transpose, since A1_full may not be orthogonal)
    return exp.(A1_full_inv * log.(θ1))
end

# Composed ϕ in Stage 1 coordinates: ϕ(θ¹) where θ¹ = exp(A1*log(θ))
ϕ_stage1_coords = θ1 -> ϕ_func_θ(inverse_transform_stage1(θ1))

# Apply find_invariant_subspace with f = identity (no transformation)
# Evaluate at transformed true parameters
θ1_true = transform_stage1(θ_true)

println("\nTransformed parameters θ¹ at true values:")
for i in 1:length(θ1_true)
    println("  θ¹[$i] = ", round(θ1_true[i], digits=4))
end

S_stage2, N_stage2, N_perp_stage2, rankJ_stage2 = find_invariant_subspace(
    ϕ_stage1_coords, θ1_true;
    rtolJ=sqrt(eps()),
    atolM=1e-10
)

println("\nStage 2 Results:")
println("  Rank of Jacobian: ", rankJ_stage2)
println("  Dimension of invariant null space: ", size(N_stage2, 2))
println("  Dimension of potentially identifiable subspace: ", size(N_perp_stage2, 2))
println("\n  Singular values:")
display(S_stage2)

if size(N_perp_stage2, 2) > 0
    println("\n  N_perp (potentially identifiable in Stage 1 space):")
    display(N_perp_stage2)

    println("\n  N (invariant null space in Stage 1 space):")
    display(N_stage2)

    # Apply scale_and_round to N_perp_stage2
    println("\n  Applying scale_and_round to Stage 2 N_perp...")
    N_perp_stage2_scaled = scale_and_round(N_perp_stage2; column_scales=ones(size(N_perp_stage2, 2)))
    println("\n  Scaled Stage 2 N_perp:")
    display(N_perp_stage2_scaled)

    # Interpret Stage 2 combinations
    println("\n  Interpreting Stage 2 (linear) combinations of Stage 1 coordinates:")
    for col in 1:size(N_perp_stage2_scaled, 2)
        coeffs = N_perp_stage2_scaled[:, col]
        nonzero_idx = findall(x -> abs(x) > 0.01, coeffs)
        if !isempty(nonzero_idx)
            terms = ["$(coeffs[i] ≈ 1 ? "" : coeffs[i] < 0 ? "-" : "+")θ¹[$i]"
                     for i in nonzero_idx]
            println("    ψ[$col] ~ ", join(terms, " "))
        end
    end
end

# --------------------------------------------------------
# Validation
# --------------------------------------------------------

println("\n" * "="^60)
println("VALIDATION AGAINST MESHKAT ET AL. (2014)")
println("="^60)

# Compute the actual values of the known identifiable combinations
println("\nKnown identifiable combinations from Meshkat et al.:")
println("  q₁ = b₁c₁             = ", q₁_true)
println("  q₂ = c₁K_M            = ", q₂_true)
println("  q₃ = k₀₂ + k₁₂        = ", q₃_true)
println("  q₄ = c₁V_M k₁₂k₂₁     = ", q₄_true)
println("  q₅ = c₁V_M(k₀₁ + k₂₁) = ", q₅_true)

# Check what we recovered from sequential IIR
println("\nSequential IIR Results:")
println("  Stage 1: Found ", size(N_perp_stage1, 2), " potentially identifiable monomial combinations")
println("  Stage 2: Found ", size(N_perp_stage2, 2), " potentially identifiable linear combinations")
println("\nTotal identifiable structure:")
println("  Rank at Stage 1: ", rankJ_stage1, " (expected ~5-6)")
println("  Rank at Stage 2: ", rankJ_stage2, " (expected ~5-6)")

# The final effective rank should match Meshkat's 5 identifiable combinations
# Note: We may have over-parameterized with 7 Stage 1 + 7 Stage 2 directions
# but the underlying rank should still be ~5

println("\nInterpretation:")
println("  - Stage 1 (monomials) separated individual rates and some products")
println("  - Stage 2 (sums) should reveal linear combinations like k₀₂+k₁₂")
println("  - The Hessian invariance test identified which combinations are truly invariant")

# --------------------------------------------------------
# Mapping Sequential IIR to Meshkat's Canonical Combinations
# --------------------------------------------------------

println("\n" * "="^60)
println("MAPPING TO MESHKAT'S CANONICAL COMBINATIONS")
println("="^60)

# Evaluate Stage 1 coordinates at true parameters
θ1_true = transform_stage1(θ_true)

println("\nStage 1 coordinates at true θ:")
for i in 1:8
    println("  θ¹[$i] = ", round(θ1_true[i], digits=4))
end

# From the scaled interpretation (lines 86-93), we have approximately:
#   θ¹[1] ~ k₀₂
#   θ¹[2] ~ c₁ × K_M^0.5
#   θ¹[3] ~ k₂₁
#   θ¹[4] ~ k₁₂
#   θ¹[5] ~ b₁ × K_M^(-0.5)
#   θ¹[6] ~ b₁^0.5 × c₁^(-0.5) × V_M^(-1.5) × K_M
#   θ¹[7] ~ k₀₁

# Read actual transformation from A1_full matrix (rows give exponents)
# θ¹ = exp(A1_full * log(θ))
# Row i of A1_full gives the exponents for computing θ¹[i]

println("\nActual Stage 1 transformations from A1_full matrix:")
println("  (Rows of A1_full give exponents in: θ¹[i] = ∏ⱼ θⱼ^(A1[i,j]))")

# Extract actual transformation for a few key coordinates
A1 = Matrix(A1_full)  # Convert from adjoint to regular matrix

# Verify by computing expected values:
println("\nVerifying Stage 1 interpretations:")
println("  θ¹[1] = k₀₂^(-1)      : ", round(θ1_true[1], digits=4), " vs ", round(1/θ_true[4], digits=4))
println("  θ¹[3] = k₂₁^(-1)      : ", round(θ1_true[3], digits=4), " vs ", round(1/θ_true[6], digits=4))
println("  θ¹[4] = k₁₂           : ", round(θ1_true[4], digits=4), " vs ", round(θ_true[5], digits=4))
println("  θ¹[7] = k₀₁           : ", round(θ1_true[7], digits=4), " vs ", round(θ_true[3], digits=4))

# Row 2: [-1/6, -5/6, 0, 0, 0, 0, -1/6, -1/2]
# θ¹[2] = b₁^(-1/6) × c₁^(-5/6) × V_M^(-1/6) × K_M^(-1/2)
expected_theta1_2 = θ_true[1]^(-1/6) * θ_true[2]^(-5/6) * θ_true[7]^(-1/6) * θ_true[8]^(-1/2)
println("  θ¹[2] complex product : ", round(θ1_true[2], digits=4), " vs ", round(expected_theta1_2, digits=4))

# Systematic reconstruction of all q₁...q₅ from θ¹ coordinates
# Since θ = exp(A1_inv * log(θ¹)), we can express original parameters in terms of θ¹

function reconstruct_meshkat_from_stage1(θ1, A1_inv_matrix)
    """
    Reconstruct Meshkat's q₁...q₅ from Stage 1 coordinates θ¹
    Returns: (q₁, q₂, q₃, q₄, q₅) or nothing if reconstruction fails
    """
    # First recover original parameters: θ = exp(A1_inv * log(θ¹))
    θ_recovered = exp.(A1_inv_matrix * log.(θ1))

    b₁, c₁, k₀₁, k₀₂, k₁₂, k₂₁, V_M, K_M = θ_recovered

    q₁_recon = b₁ * c₁
    q₂_recon = c₁ * K_M
    q₃_recon = k₀₂ + k₁₂
    q₄_recon = c₁ * V_M * k₁₂ * k₂₁
    q₅_recon = c₁ * V_M * (k₀₁ + k₂₁)

    return (q₁_recon, q₂_recon, q₃_recon, q₄_recon, q₅_recon)
end

println("\n" * "="^60)
println("SYSTEMATIC RECONSTRUCTION OF MESHKAT'S q₁...q₅")
println("="^60)

# Extract A1_inv as a regular matrix
A1_inv_matrix = Matrix(A1_full_inv)

# Reconstruct all q values from Stage 1 coordinates
q_recon = reconstruct_meshkat_from_stage1(θ1_true, A1_inv_matrix)

println("\nReconstruction from Stage 1 coordinates:")
println("  q₁ = b₁c₁             : ", round(q_recon[1], digits=6), " vs ", round(q₁_true, digits=6),
        " ", abs(q_recon[1] - q₁_true) < 1e-10 ? "✓" : "✗")
println("  q₂ = c₁K_M            : ", round(q_recon[2], digits=6), " vs ", round(q₂_true, digits=6),
        " ", abs(q_recon[2] - q₂_true) < 1e-10 ? "✓" : "✗")
println("  q₃ = k₀₂ + k₁₂        : ", round(q_recon[3], digits=6), " vs ", round(q₃_true, digits=6),
        " ", abs(q_recon[3] - q₃_true) < 1e-10 ? "✓" : "✗")
println("  q₄ = c₁V_Mk₁₂k₂₁      : ", round(q_recon[4], digits=6), " vs ", round(q₄_true, digits=6),
        " ", abs(q_recon[4] - q₄_true) < 1e-10 ? "✓" : "✗")
println("  q₅ = c₁V_M(k₀₁ + k₂₁) : ", round(q_recon[5], digits=6), " vs ", round(q₅_true, digits=6),
        " ", abs(q_recon[5] - q₅_true) < 1e-10 ? "✓" : "✗")

println("\nInterpretation:")
println("  ✓ All q₁...q₅ can be perfectly reconstructed from θ¹ via inverse transform")
println("  ✓ This confirms Stage 1 transformation is invertible and preserves information")
println("  ⚠ BUT: The q's are NOT simple combinations of θ¹ components")
println("     They require the full inverse transformation A1_inv")

println("\nCritical question for Stage 2:")
println("  Can Stage 2 identify combinations of θ¹ that correspond to q₁...q₅?")
println("  If Varimax has separated parameters well, some q's might appear as")
println("  simple products or ratios of θ¹ components. Let's investigate...")

# --------------------------------------------------------
# Parameter Bounds for Profiling
# --------------------------------------------------------

# Variable names for plotting
varnames = Dict(
    "ψ1" => "b₁", "ψ2" => "c₁", "ψ3" => "k₀₁", "ψ4" => "k₀₂",
    "ψ5" => "k₁₂", "ψ6" => "k₂₁", "ψ7" => "V_M", "ψ8" => "K_M"
)
for i in 1:8
    varnames["ψ$(i)_save"] = varnames["ψ$i"]
end

# Parameter bounds (generous to show identifiability issues)
θ_lower_bounds = [0.5, 0.5, 0.05, 0.05, 0.1, 0.1, 0.2, 0.5]
θ_upper_bounds = [5.0, 5.0, 1.0, 0.5, 1.0, 1.0, 5.0, 10.0]

# Initial guess for optimization
θ_initial = 0.5 * (θ_lower_bounds + θ_upper_bounds)

# Construct log-likelihood in original parameterization
lnlike_θ = construct_lnlike_xy(distrib_θ, data; dist_type=:multi)

# --------------------------------------------------------
# MLE Estimation
# --------------------------------------------------------

println("\n" * "="^60)
println("MAXIMUM LIKELIHOOD ESTIMATION")
println("="^60)

point_estimation_method = :LN_BOBYQA
target_indices = [] # Empty for MLE
n_guesses = 5

# Generate multiple initial guesses
nuisance_guesses = generate_initial_guesses(θ_lower_bounds, θ_upper_bounds, n_guesses)

θ_MLE, lnlike_θ_MLE = profile_target(lnlike_θ, target_indices,
    θ_lower_bounds, θ_upper_bounds,
    θ_initial; grid_steps=[500], ω_initial_extras=nuisance_guesses,
    method=point_estimation_method)

println("\nMLE estimate:")
for i in 1:8
    println("  $(varnames["ψ$i"]) = $(round(θ_MLE[i], digits=4)) (true: $(round(θ_true[i], digits=4)))")
end

# --------------------------------------------------------
# Jacobian Analysis at MLE
# --------------------------------------------------------

println("\n" * "="^60)
println("JACOBIAN SVD ANALYSIS")
println("="^60)

J_ϕ_θ, U_θ, S_θ, Vt_θ = compute_ϕ_Jacobian(ϕ_func_θ, θ_MLE; method_type=:auto, compute_svd=true)

println("\nSingular values of Jacobian J_ϕ(θ):")
for (i, s) in enumerate(S_θ)
    println("  σ[$i] = ", round(s, sigdigits=4))
end

println("\nRight singular vectors (parameter combinations):")
display(Vt_θ')

println("\nCondition number: ", S_θ[1]/S_θ[end])

# Check for small singular values indicating non-identifiability
rtol = sqrt(eps())
rank_J = sum(S_θ .> rtol * S_θ[1])
println("\nEffective rank: $rank_J / $(length(S_θ))")

if rank_J < length(θ_MLE)
    println("\n⚠ Model is structurally non-identifiable!")
    println("  Expected: 5 identifiable combinations from 8 parameters")
    println("  Found: $rank_J singular values above threshold")
end

# --------------------------------------------------------
# TODO: Profile Likelihood Analysis
# --------------------------------------------------------

println("\n" * "="^60)
println("PROFILE LIKELIHOOD ANALYSIS")
println("="^60)
println("TODO: Compute 1D profiles for all 8 parameters")
println("Expected: Flat profiles for non-identifiable parameters")
println("         Curved profiles for identifiable combinations")

# Grid sizes for profiling
# grid_steps = [500]
# dim_all = length(θ_initial)
# indices_all = 1:dim_all
# profile_method = :LN_BOBYQA

# for i in 1:dim_all
#     # TODO: Compute and plot profiles
# end

println("\n" * "="^60)
println("END OF ANALYSIS")
println("="^60)
