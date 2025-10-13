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

# Set random seed for reproducibility
Random.seed!(42)

# --------------------------------------------------------
# Model Definition: Repressilator
# --------------------------------------------------------
# Three-gene negative feedback loop (Elowitz & Leibler 2000)
# Gene 1 represses Gene 2, Gene 2 represses Gene 3, Gene 3 represses Gene 1
#
# State variables: m₁, m₂, m₃ (mRNA concentrations), p₁, p₂, p₃ (protein concentrations)
#
# Key parameters:
# - α₁, α₂, α₃: Basal transcription rates
# - β₁, β₂, β₃: Translation rates
# - K₁, K₂, K₃: Hill repression constants
# - n: Hill coefficient (cooperative binding)
# - γₘ: mRNA degradation rate
# - γₚ: Protein degradation rate
#
# Identifiable combinations (from literature): K₁/β₁, K₂/β₂, K₃/β₃
# These are the "effective regulation strengths"

function repressilator!(dX, X, θ, t)
    """
    Repressilator ODE system.

    State vector X = [m₁, m₂, m₃, p₁, p₂, p₃]
    Parameter vector θ = [α₁, α₂, α₃, β₁, β₂, β₃, K₁, K₂, K₃, n, γₘ, γₚ]

    Parameters:
    - dX: Rate of change vector (modified in-place)
    - X: Current state vector (6 states)
    - θ: Parameter vector (12 parameters)
    - t: Current time
    """

    # Unpack state variables
    m₁, m₂, m₃, p₁, p₂, p₃ = X

    # Unpack parameters
    α₁, α₂, α₃, β₁, β₂, β₃, K₁, K₂, K₃, n, γₘ, γₚ = θ

    # mRNA dynamics: repression by previous gene's protein
    # Gene 1 is repressed by protein 3
    dX[1] = α₁ / (1 + (p₃/K₁)^n) - γₘ * m₁

    # Gene 2 is repressed by protein 1
    dX[2] = α₂ / (1 + (p₁/K₂)^n) - γₘ * m₂

    # Gene 3 is repressed by protein 2
    dX[3] = α₃ / (1 + (p₂/K₃)^n) - γₘ * m₃

    # Protein dynamics: translation from mRNA, degradation
    dX[4] = β₁ * m₁ - γₚ * p₁
    dX[5] = β₂ * m₂ - γₚ * p₂
    dX[6] = β₃ * m₃ - γₚ * p₃
end

# ODE model solver
function solve_repressilator(t_save, θ, X0; solver=Rodas4())
    """
    Solve the repressilator ODE system.

    Parameters:
    - t_save: Time grid points
    - θ: Parameter vector (12 parameters)
    - X0: Initial condition (6 states)
    - solver: ODE solver (default: Rodas4())

    Returns:
    - Matrix of solution values: 6 states × NT time points
    """
    tspan = (0.0, maximum(t_save))
    prob = ODEProblem(repressilator!, X0, tspan, θ)
    sol = solve(prob, solver, saveat=t_save, abstol=1e-10, reltol=1e-8)

    # Return as matrix: rows are states, columns are time points
    return Array(sol)
end

# Extract mRNA observations - ALL THREE (match Eisenberg)
function extract_mrna(solution_matrix)
    """
    Extract all three mRNA concentrations from full state vector.

    Eisenberg & Hayashi observe all three mRNAs to get K/β structure.

    Parameters:
    - solution_matrix: 6 × NT matrix of states

    Returns:
    - 3 × NT matrix of mRNA concentrations [m₁, m₂, m₃]
    """
    return solution_matrix[1:3, :]  # All three mRNAs
end

# Creates a ϕ mapping function from parameters to mRNA observations
function create_ϕ_mapping(t, X0)
    """
    Create a ϕ mapping function from model parameters to mRNA observations.

    Parameters:
    - t: Time grid points
    - X0: Initial condition (6 states)

    Returns:
    - ϕ mapping function from θ to flattened mRNA observations
    """
    function ϕ(θ)
        sol_matrix = solve_repressilator(t, θ, X0)
        mrna_matrix = extract_mrna(sol_matrix)
        # Flatten to vector: [m₁(t₁), m₂(t₁), m₃(t₁), m₁(t₂), ...]
        return vec(mrna_matrix')
    end
    return ϕ
end

# --------------------------------------------------------
# Setup and Data Generation
# --------------------------------------------------------

println(repeat("=", 60))
println("Repressilator Model: Invariant Image Reparameterization")
println(repeat("=", 60))

# Time grid setup - match Eisenberg with dense sampling
T_end = 100.0  # Final time
NT = 21        # Dense time points
t = LinRange(0, T_end, NT)

# Initial conditions (6 states: m₁, m₂, m₃, p₁, p₂, p₃)
# Start with small perturbation from symmetry
X0 = [1.0, 0.5, 0.3, 2.0, 1.0, 0.5]

# Observation noise
σ = 0.1

# --------------------------------------------------------
# True parameter values
# --------------------------------------------------------
# We'll use symmetric parameters except for the identifiable ratios K/β
# This makes the identifiability pattern clear

α₁_true = 1.0
α₂_true = 1.0
α₃_true = 1.0

β₁_true = 2.0
β₂_true = 2.5
β₃_true = 3.0

K₁_true = 10.0
K₂_true = 15.0
K₃_true = 18.0

n_true = 2.0
γₘ_true = 1.0
γₚ_true = 0.5

θ_true = [α₁_true, α₂_true, α₃_true,
          β₁_true, β₂_true, β₃_true,
          K₁_true, K₂_true, K₃_true,
          n_true, γₘ_true, γₚ_true]

# Print true identifiable combinations
println("\nTrue identifiable combinations (K/β ratios):")
println("  K₁/β₁ = $(K₁_true/β₁_true)")
println("  K₂/β₂ = $(K₂_true/β₂_true)")
println("  K₃/β₃ = $(K₃_true/β₃_true)")

# --------------------------------------------------------
# Generate synthetic data
# --------------------------------------------------------

# Create ϕ mapping
ϕ_func = create_ϕ_mapping(t, X0)

# Generate noiseless solution
y_true = ϕ_func(θ_true)

# Number of observations
N_obs = length(y_true)  # 3 mRNAs × NT time points

# Add Gaussian noise
data = y_true + σ * randn(N_obs)

println("\nData generated:")
println("  Time points: $NT")
println("  Observables: 3 mRNAs (m₁, m₂, m₃)")
println("  Total observations: $N_obs")
println("  Noise level σ = $σ")

# --------------------------------------------------------
# Parameter Bounds and Initial Guess
# --------------------------------------------------------

# We'll focus on parameters that appear in the identifiable combinations:
# β₁, β₂, β₃, K₁, K₂, K₃
# Fix the other parameters at their true values for simplicity

# Create reduced parameter vector: θ_red = [β₁, β₂, β₃, K₁, K₂, K₃]
# and ϕ mapping that incorporates fixed parameters

function create_ϕ_reduced(ϕ_full, θ_fixed_indices, θ_fixed_values)
    """
    Create reduced ϕ mapping with some parameters fixed.

    Parameters:
    - ϕ_full: Full ϕ mapping function
    - θ_fixed_indices: Indices of parameters to fix
    - θ_fixed_values: Values of fixed parameters

    Returns:
    - ϕ_red: Reduced ϕ mapping function
    """
    function ϕ_red(θ_red)
        # Reconstruct full parameter vector
        # Use eltype to handle ForwardDiff.Dual numbers
        T = promote_type(eltype(θ_red), Float64)
        θ_full = Vector{T}(undef, 12)
        red_idx = 1
        for i in 1:12
            if i in θ_fixed_indices
                idx_in_fixed = findfirst(==(i), θ_fixed_indices)
                θ_full[i] = θ_fixed_values[idx_in_fixed]
            else
                θ_full[i] = θ_red[red_idx]
                red_idx += 1
            end
        end
        return ϕ_full(θ_full)
    end
    return ϕ_red
end

# Fix parameters: α₁, α₂, α₃, n, γₘ, γₚ (match Eisenberg setup)
# Focus on β and K parameters where K/β ratios are identifiable
θ_fixed_indices = [1, 2, 3, 10, 11, 12]
θ_fixed_values = [α₁_true, α₂_true, α₃_true, n_true, γₘ_true, γₚ_true]

# Reduced parameter vector: θ_red = [β₁, β₂, β₃, K₁, K₂, K₃] (indices 4,5,6,7,8,9)
ϕ_reduced = create_ϕ_reduced(ϕ_func, θ_fixed_indices, θ_fixed_values)

# True values for reduced parameters
θ_red_true = [β₁_true, β₂_true, β₃_true, K₁_true, K₂_true, K₃_true]

# Parameter bounds for reduced parameters
θ_red_lower = [0.5, 0.5, 0.5, 2.0, 2.0, 2.0]
θ_red_upper = [5.0, 5.0, 5.0, 30.0, 30.0, 30.0]

# Initial guess (perturbed from true)
θ_red_initial = θ_red_true .* (1.0 .+ 0.2*randn(6))
θ_red_initial = max.(θ_red_lower, min.(θ_red_upper, θ_red_initial))

println("\nReduced parameter space (6 parameters):")
println("  Parameters: [β₁, β₂, β₃, K₁, K₂, K₃]")
println("  Fixed: α, n, γₘ, γₚ (Eisenberg setup)")
println("  True values: ", round.(θ_red_true, digits=3))
println("  Initial guess: ", round.(θ_red_initial, digits=3))

# --------------------------------------------------------
# Apply IIR: Find Invariant Subspace
# --------------------------------------------------------

println("\n" * repeat("=", 60))
println("Applying Invariant Image Reparameterization")
println(repeat("=", 60))

# Wrap ϕ_reduced in log-space for Stage 1 (f=log transformation)
# This is critical: the algorithm operates on log-transformed parameters
ϕ_log(θ_log) = ϕ_reduced(exp.(θ_log))
θ_log_true = log.(θ_red_true)

# Compute Jacobian in log-space
J_θ_log = compute_ϕ_Jacobian(ϕ_log, θ_log_true)
println("\nJacobian dimensions (in log-space): ", size(J_θ_log))

# SVD of Jacobian
U_θ, S_θ, Vt_θ = svd(J_θ_log)
println("\nSingular values of Jacobian (in log-space):")
for (i, s) in enumerate(S_θ)
    println("  σ[$i] = ", round(s, sigdigits=6))
end

# Apply find_invariant_subspace with f = log transformation
println("\nApplying find_invariant_subspace() with f=log...")
println("(Operating in log-space: θ_log = log(θ_red))")

# Note: We expect K/β combinations to be invariant
# In log space: log(K) - log(β) correspond to linear combinations

S_inv, N_inv, N_perp_inv, rank_J = find_invariant_subspace(
    ϕ_log, θ_log_true;
    rtolJ=sqrt(eps()),
    atolM=1e-10
)

n_params = length(θ_red_true)  # Should be 6
param_names = ["β₁", "β₂", "β₃", "K₁", "K₂", "K₃"]

println("\nInvariant Subspace Analysis:")
println("  Jacobian rank: $rank_J / $n_params")
println("  Identifiable directions: ", size(N_perp_inv, 2))
println("  Non-identifiable directions: ", size(N_inv, 2))

# Print the invariant null space vectors
if size(N_inv, 2) > 0
    println("\nNon-identifiable (invariant) directions in log-space:")
    println("(These should correspond to K/β ratios)")
    for j in 1:size(N_inv, 2)
        v = N_inv[:, j]
        println("  Direction $j: ", round.(v, digits=3))

        # Interpret: v[i] * log(θ[i]) are the coefficients
        # For K₁/β₁: expect v ∝ [-1, 0, 0, 1, 0, 0, 0, 0] → β₁^(-1) * K₁
        active_terms = [(i, v[i]) for i in 1:n_params if abs(v[i]) > 0.01]
        if !isempty(active_terms)
            println("    Interpretation: exp(",
                    join(["$(round(coef, digits=2))*log($(param_names[idx]))" for (idx, coef) in active_terms], " + "),
                    ")")
        end
    end
end

# Print the identifiable directions
if size(N_perp_inv, 2) > 0
    println("\nIdentifiable directions in log-space:")
    for j in 1:size(N_perp_inv, 2)
        v = N_perp_inv[:, j]
        println("  Direction $j: ", round.(v, digits=3))
    end
end

# --------------------------------------------------------
# Construct Transformation Matrix
# --------------------------------------------------------

# Following stat_model.jl pattern: build full square transformation
# A_full = [N_perp'; N']^T after proper scaling

println("\n" * repeat("=", 60))
println("Constructing Transformation Matrix")
println(repeat("=", 60))

# Build column-stacked matrix (columns are basis vectors)
A_full_T = hcat(N_perp_inv, N_inv)

# Scale columns for cleaner interpretation (must match number of parameters!)
A_full_T_scaled = scale_and_round(A_full_T; column_scales=ones(n_params))

# Transpose to get transformation matrix (rows are parameter combinations)
A_full = A_full_T_scaled'

println("\nTransformation matrix A (rows define new parameters):")
display(round.(A_full, digits=2))

# Compute inverse for θ_red = exp(A_inv * log(ψ))
A_inv = inv(A_full)

# Define transformation functions
θ_to_ψ(θ) = exp.(A_full * log.(θ))
ψ_to_θ(ψ) = exp.(A_inv * log.(ψ))

# Verify transformation at true parameters
ψ_true = θ_to_ψ(θ_red_true)
θ_reconstructed = ψ_to_θ(ψ_true)

println("\nTransformation verification:")
println("  Original θ: ", round.(θ_red_true, digits=3))
println("  Transformed ψ: ", round.(ψ_true, digits=3))
println("  Reconstructed θ: ", round.(θ_reconstructed, digits=3))
println("  Max error: ", maximum(abs.(θ_red_true - θ_reconstructed)))

# --------------------------------------------------------
# Identify which ψ correspond to K/β ratios
# --------------------------------------------------------

# The non-identifiable parameters should be K₁/β₁, K₂/β₂, K₃/β₃
# Let's check which ψ indices these correspond to

println("\n" * repeat("=", 60))
println("Identifiable Combinations")
println(repeat("=", 60))

n_ident = size(N_perp_inv, 2)
n_non_ident = size(N_inv, 2)

println("\nIdentifiable combinations (ψ[1:$n_ident]):")
for i in 1:n_ident
    row = A_full[i, :]
    # Find dominant contributions
    max_abs_idx = argmax(abs.(row))
    println("  ψ[$i] ≈ $(param_names[max_abs_idx])^$(round(row[max_abs_idx], digits=2))")
end

println("\nNon-identifiable combinations (ψ[$(n_ident+1):$n_params]):")
println("(Expected: K₁/β₁, K₂/β₂, K₃/β₃)")
for i in (n_ident+1):n_params
    row = A_full[i, :]
    # Interpret as products of powers
    terms = String[]
    for j in 1:n_params
        if abs(row[j]) > 0.01
            push!(terms, "$(param_names[j])^$(round(row[j], digits=2))")
        end
    end
    println("  ψ[$i] = ", join(terms, " * "))

    # Compute value
    println("    Value = ", round(ψ_true[i], digits=3))
end

println("\n" * repeat("=", 60))
println("Analysis Complete")
println(repeat("=", 60))

# --------------------------------------------------------
# Save results for later use
# --------------------------------------------------------

# Store key results in a dictionary for potential profiling/plotting
results = Dict(
    "θ_true" => θ_red_true,
    "ψ_true" => ψ_true,
    "A_full" => A_full,
    "A_inv" => A_inv,
    "N_perp" => N_perp_inv,
    "N" => N_inv,
    "rank_J" => rank_J,
    "data" => data,
    "ϕ_reduced" => ϕ_reduced,
    "θ_to_ψ" => θ_to_ψ,
    "ψ_to_θ" => ψ_to_θ
)

println("\nResults saved to 'results' dictionary")
println("Ready for profile likelihood analysis and plotting")
