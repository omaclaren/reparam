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
using ForwardDiff

# Set random seed for reproducibility
Random.seed!(42)

# Global index helpers for parameter groups (used throughout analysis)
const BETA_INDICES = [7, 8, 9]
const K_INDICES = [10, 11, 12]
const PARAM_INDICES = collect(1:18)

# ========================================================================
# CONFIGURATION: Profiling Settings
# ========================================================================
# Three modes: "test" (~1 min), "paper" (~4 min), "full" (~60 min)
const PROFILE_MODE = "paper"  # Options: "test", "paper", "full"

# Mode configurations
const PROFILE_CONFIGS = Dict(
    "test"  => (grid_1d=3, grid_2d=[3,3], timeout=10.0, n_guesses=1, do_2d=false),
    "paper" => (grid_1d=7, grid_2d=[5,5], timeout=10.0, n_guesses=2, do_2d=false),
    "full"  => (grid_1d=10, grid_2d=[7,7], timeout=60.0, n_guesses=3, do_2d=true)
)

# Parallelization note:
# Current implementation uses sequential profiling with previous point's solution
# as initial guess for next point (warm start). This improves convergence but
# requires sequential execution.
#
# ALTERNATIVE: Use MLE as initial guess for all points → enables parallelization
# - Each optimization becomes independent → can use Julia's @threads or Distributed
# - Trade-off: potentially slower convergence per point, but massive speedup from parallel
# - For 18-param ODE: ~6x speedup on 6-core machine could reduce "paper" mode from 4→<1 min
#
# To implement: modify profile_target() to accept parallel=true flag and use MLE
# as ω_initial for all grid points instead of chaining solutions.

const CONFIG = PROFILE_CONFIGS[PROFILE_MODE]
println("=" ^ 70)
println("PROFILE MODE: $(PROFILE_MODE)")
println("  1D grid: $(CONFIG.grid_1d) points")
println("  2D grid: $(CONFIG.grid_2d) points")
println("  Timeout: $(CONFIG.timeout)s per optimization")
println("  Initial guesses: $(CONFIG.n_guesses)")
println("  2D profiling: $(CONFIG.do_2d)")
est_time = CONFIG.do_2d ?
    (2 * CONFIG.grid_1d + prod(CONFIG.grid_2d)) * CONFIG.n_guesses * CONFIG.timeout / 60 :
    2 * CONFIG.grid_1d * CONFIG.n_guesses * CONFIG.timeout / 60
println("  Estimated runtime: ~$(round(Int, est_time)) minutes")
println("=" ^ 70)

# --------------------------------------------------------
# Model Definition: Repressilator (Eisenberg & Hayashi Setup)
# --------------------------------------------------------
# Eisenberg & Hayashi exact parameter setup with n=2 (Hill coefficient fixed)
#
# For i=1,2,3 (modulo 3):
#   ṁᵢ = α₀ᵢ + αᵢ/(1 + (pᵢ₋₁/Kᵢ₋₁)²) - k_degmᵢ·mᵢ
#   ṗᵢ = βᵢ·mᵢ - k_degpᵢ·pᵢ
#   yᵢ = mᵢ
#
# 18 parameters with rank 15/18 (3 non-identifiable).
# Expected identifiable: K₁/β₁, K₂/β₂, K₃/β₃ ratios.

function repressilator!(dX, X, θ, t)
    """
    Repressilator ODE system - Eisenberg & Hayashi formulation with n=2 fixed.

    State vector X = [m₁, m₂, m₃, p₁, p₂, p₃]
    Parameter vector θ = [α₀₁, α₀₂, α₀₃, α₁, α₂, α₃,
                          β₁, β₂, β₃, K₁, K₂, K₃,
                          k_degm₁, k_degm₂, k_degm₃, k_degp₁, k_degp₂, k_degp₃]

    18 parameters (Hill coefficient n fixed at 2)
    """

    # Unpack state variables
    m₁, m₂, m₃, p₁, p₂, p₃ = X

    # Unpack parameters (18 total)
    α₀₁, α₀₂, α₀₃ = θ[1:3]      # Basal transcription
    α₁, α₂, α₃ = θ[4:6]          # Regulated transcription
    β₁, β₂, β₃ = θ[7:9]          # Translation
    K₁, K₂, K₃ = θ[10:12]        # Inhibition constants
    k_degm₁, k_degm₂, k_degm₃ = θ[13:15]  # mRNA degradation
    k_degp₁, k_degp₂, k_degp₃ = θ[16:18]  # Protein degradation

    # Hill coefficient fixed at 2
    n = 2.0

    # mRNA dynamics: basal + regulated transcription - degradation
    # Gene i is repressed by protein i-1 (modulo 3)
    dX[1] = α₀₁ + α₁ / (1 + (p₃/K₃)^n) - k_degm₁ * m₁
    dX[2] = α₀₂ + α₂ / (1 + (p₁/K₁)^n) - k_degm₂ * m₂
    dX[3] = α₀₃ + α₃ / (1 + (p₂/K₂)^n) - k_degm₃ * m₃

    # Protein dynamics: translation - degradation
    dX[4] = β₁ * m₁ - k_degp₁ * p₁
    dX[5] = β₂ * m₂ - k_degp₂ * p₂
    dX[6] = β₃ * m₃ - k_degp₃ * p₃
end

# ODE model solver
function solve_repressilator(t_save, θ, X0; solver=Rodas4())
    """
    Solve the repressilator ODE system.
    """
    tspan = (0.0, maximum(t_save))
    prob = ODEProblem(repressilator!, X0, tspan, θ)
    sol = solve(prob, solver, saveat=t_save, abstol=1e-10, reltol=1e-8)
    return Array(sol)
end

# Extract mRNA observations (all three as in Eisenberg)
function extract_mrna(solution_matrix)
    """Extract all three mRNA concentrations."""
    return solution_matrix[1:3, :]
end

# Creates ϕ mapping function. Assume map to mRNA only (all three).
function create_ϕ_mapping(t, X0)
    """Create ϕ mapping from parameters to mRNA observations."""
    function ϕ(θ)
        sol_matrix = solve_repressilator(t, θ, X0)
        mrna_matrix = extract_mrna(sol_matrix)
        # Flatten: [m₁(t₁), m₂(t₁), m₃(t₁), m₁(t₂), ...]
        return vec(mrna_matrix')
    end
    return ϕ
end

# --------------------------------------------------------
# Setup and Data Generation
# --------------------------------------------------------

println(repeat("=", 70))
println("Repressilator Model (Eisenberg & Hayashi Setup)")
println("18 Free Parameters (n fixed at 2)")
println("Expected identifiable: K₁/β₁, K₂/β₂, K₃/β₃ ratios")
println(repeat("=", 70))

# Time grid - dense sampling as in Eisenberg
T_end = 100.0
NT = 21
t = LinRange(0, T_end, NT)

# Initial conditions from Eisenberg paper
X0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Observation noise
σ = 0.5  # Moderate noise for visible prediction intervals

# --------------------------------------------------------
# True parameter values - EISENBERG & HAYASHI EXACT
# --------------------------------------------------------

# Basal transcription rates
α₀₁_true = 5e-4
α₀₂_true = 1e-4
α₀₃_true = 9e-4

# Regulated transcription rates
α₁_true = 0.5
α₂_true = 0.7
α₃_true = 0.8

# Translation rates
β₁_true = 0.1155
β₂_true = 0.23
β₃_true = 0.0789

# Inhibition constants
K₁_true = 40.0
K₂_true = 30.0
K₃_true = 50.0

# mRNA degradation rates
k_degm₁_true = 0.005776
k_degm₂_true = 0.00987
k_degm₃_true = 0.00345

# Protein degradation rates
k_degp₁_true = 0.001155
k_degp₂_true = 0.00059
k_degp₃_true = 0.004982

# Hill coefficient (fixed at 2.0 in model)
n_true = 2.0

# Parameter names
param_names = ["α₀₁", "α₀₂", "α₀₃",
               "α₁", "α₂", "α₃",
               "β₁", "β₂", "β₃",
               "K₁", "K₂", "K₃",
               "k_degm₁", "k_degm₂", "k_degm₃",
               "k_degp₁", "k_degp₂", "k_degp₃"]

# Parameter vector (18 parameters, n not included since it's fixed)
θ_true = [α₀₁_true, α₀₂_true, α₀₃_true,
          α₁_true, α₂_true, α₃_true,
          β₁_true, β₂_true, β₃_true,
          K₁_true, K₂_true, K₃_true,
          k_degm₁_true, k_degm₂_true, k_degm₃_true,
          k_degp₁_true, k_degp₂_true, k_degp₃_true]

println("\nTrue parameter values (18 total, n=2 fixed):")
for (i, (name, val)) in enumerate(zip(param_names, θ_true))
    println("  $name = $val")
end

println("\nExpected identifiable combinations:")
println("  K₁/β₁ = $(K₁_true/β₁_true)")
println("  K₂/β₂ = $(K₂_true/β₂_true)")
println("  K₃/β₃ = $(K₃_true/β₃_true)")

function predict_mRNA(θ, t_grid=t)
    sol_matrix = solve_repressilator(t_grid, θ, X0)
    mRNA = extract_mrna(sol_matrix)  # 3×NT matrix
    return vec(mRNA')  # Flatten to vector (time-major order)
end

# --------------------------------------------------------
# Generate synthetic data
# --------------------------------------------------------

ϕ_func = create_ϕ_mapping(t, X0)
y_true = predict_mRNA(θ_true, t)
N_obs = length(y_true)
data = y_true + σ * randn(N_obs)

println("\nData generated:")
println("  Time points: $NT")
println("  Observables: 3 mRNAs (m₁, m₂, m₃)")
println("  Total observations: $N_obs")
println("  Noise level σ = $σ")

# IIR analysis moved to after MLE computation (see below)


println("PROFILE-WISE PREDICTION: Individual vs Ratio Comparison")
println(repeat("=", 70))

# --------------------------------------------------------
# Setup: Compare prediction uncertainty for:
# Option B: Individual parameters K₁, β₁ vs ratio K₁/β₁
# --------------------------------------------------------

# Use existing time grid from data (T_end=100, NT=21)
# For predictions, we want finer resolution
t_pred = LinRange(0, T_end, 101)  # Fine grid for smooth prediction bands

# Observation noise model (same as fitting)
σ_pred = σ

# Define prediction distribution: mRNA trajectories m₁, m₂, m₃
# Use data grid (t) for likelihood, prediction grid (t_pred) for visualization

# Distribution for predictions (mRNA only, not proteins)
# Use fine grid for smooth prediction bands; additive Gaussian noise model
distrib_fine_θ = θ -> MvNormal(predict_mRNA(θ, t_pred), σ_pred^2 * I(3*length(t_pred)))

# Log-space wrapper for profiling (profiles are in log-space)
distrib_fine_θ_log = θ_log -> distrib_fine_θ(exp.(θ_log))

println("\nPrediction setup:")
println("  Parameters: 18 (n=2 fixed in model)")
println("  Time points: $(length(t_pred)) over [0, $T_end]")
println("  Observables: 3 mRNA species (m₁, m₂, m₃)")
println("  Total prediction dimension: $(3*length(t_pred))")
println("  Noise level: σ = $σ_pred")

# --------------------------------------------------------
# Option B: Individual K₁, β₁ vs ratio K₁/β₁
# --------------------------------------------------------

println("\n" * repeat("-", 70))
println("Option B: Individual Parameters vs Identifiable Ratio")
println(repeat("-", 70))

# We need to profile:
# 1. K₁ individually (non-identifiable, βK product in null space)
# 2. β₁ individually (non-identifiable, βK product in null space)
# 3. K₁/β₁ ratio (identifiable, in complement space)

# Define log-likelihood for profiling
# We need synthetic "data" first - generate from true parameters
sol_matrix_data = solve_repressilator(t, θ_true, X0)
data_mRNA = extract_mrna(sol_matrix_data)  # 3×NT
data_obs = vec(data_mRNA')  # Flatten (time-major order)

# Log-likelihood function with failure tracking
mutable struct LikelihoodStats
    n_calls::Int
    n_failures::Int
end
const lnlike_stats = LikelihoodStats(0, 0)

function lnlike_θ(θ)
    lnlike_stats.n_calls += 1

    if any(θ .<= 0)
        lnlike_stats.n_failures += 1
        return -Inf
    end
    try
        pred = predict_mRNA(θ)
        if length(pred) != length(data_obs)
            lnlike_stats.n_failures += 1
            return -Inf  # Solver failed
        end
        # Simple Gaussian log-likelihood
        return -0.5 * sum(((data_obs .- pred) ./ σ).^2)
    catch e
        lnlike_stats.n_failures += 1
        # Uncomment for debugging:
        # @warn "Solver error" exception=e
        return -Inf  # Any solver failure
    end
end

# Find MLE by optimization
println("\n" * repeat("=", 70))
println("FINDING MLE")
println(repeat("=", 70))

# Bounds for optimization (log scale for positivity)
# Tighter bounds to avoid unstable ODE regions
θ_log_lower = log.(θ_true .* 0.7)  # 30% smaller
θ_log_upper = log.(θ_true .* 1.5)  # 50% larger
# Start from midpoint of bounds (NOT true parameters - we don't know those in real data!)
θ_log_initial = 0.5 * (θ_log_lower + θ_log_upper)
θ_initial = exp.(θ_log_initial)

println("\nOptimization setup:")
println("  Method: LN_BOBYQA (gradient-free, bound-constrained)")
println("  Parameters: 18")
println("  Initial guesses: 3 (from midpoint and random)")
println("  Max time: 30 seconds per guess")
println("  Bounds: 0.7×θ_true to 1.5×θ_true (log-space)")
println("  Starting point: midpoint = (0.7 + 1.5)/2 ≈ 1.1×θ_true")

# Test initial point
lnlike_initial = lnlike_θ(θ_initial)
lnlike_true_start = lnlike_θ(θ_true)

println("\nInitial point evaluation:")
println("  Starting from: 1.1×θ_true (midpoint of bounds)")
println("  Log-likelihood at start: $(round(lnlike_initial, digits=2))")
println("  Log-likelihood at truth: $(round(lnlike_true_start, digits=2))")
println("  Gap to close: $(round(lnlike_true_start - lnlike_initial, digits=2))")

if lnlike_initial == -Inf
    @error "Initial point has -Inf likelihood! Cannot optimize. Check:\n" *
           "  - ODE solver settings (tolerances, time span)\n" *
           "  - Parameter bounds\n" *
           "  - Data generation"
    error("Cannot proceed with -Inf initial likelihood")
end

# Log-likelihood in log-parameter space
lnlike_θ_log = θ_log -> lnlike_θ(exp.(θ_log))

# Optimize to find MLE (empty target_indices means find MLE)
target_indices = Int[]  # Empty for MLE
n_guesses_mle = 3
grid_steps_mle = Int[]  # Empty for MLE

# Generate multiple initial guesses
nuisance_guesses_mle = generate_initial_guesses(θ_log_lower, θ_log_upper, n_guesses_mle)

println("\nRunning optimization...")
println("  (This may take up to 90 seconds with 3 initial guesses)")
flush(stdout)

t_mle_start = time()
θ_log_MLE, lnlike_MLE = profile_target(
    lnlike_θ_log, target_indices,
    θ_log_lower, θ_log_upper,
    θ_log_initial;
    grid_steps=grid_steps_mle,
    ω_initial_extras=nuisance_guesses_mle,
    method=:LN_BOBYQA,
    optmaxtime=30.0)
t_mle_elapsed = time() - t_mle_start

println("Optimization complete!")
println("  Time: $(round(t_mle_elapsed, digits=1)) seconds")

θ_MLE = exp.(θ_log_MLE)

# Report likelihood evaluation statistics
println("\nLikelihood evaluation statistics:")
println("  Total evaluations: $(lnlike_stats.n_calls)")
println("  Failures (returned -Inf): $(lnlike_stats.n_failures)")
println("  Success rate: $(round((1 - lnlike_stats.n_failures/lnlike_stats.n_calls)*100, digits=1))%")
if lnlike_stats.n_failures / lnlike_stats.n_calls > 0.5
    @warn "High failure rate (>50%) during optimization. Consider:\n" *
          "  - Widening bounds\n" *
          "  - Using more robust ODE solver\n" *
          "  - Checking for numerical stability issues"
end

# Verify MLE is better than true parameters
lnlike_true = lnlike_θ(θ_true)
improvement = lnlike_MLE - lnlike_true

println("\nMLE verification:")
println("  Log-likelihood at true parameters: $(round(lnlike_true, digits=4))")
println("  Log-likelihood at MLE: $(round(lnlike_MLE, digits=4))")
println("  Improvement: $(round(improvement, digits=4))")

# Only warn if improvement is clearly negative (accounting for numerical noise)
if improvement < -0.01
    @warn "MLE has significantly worse likelihood than true parameters! Optimization may have failed."
elseif abs(improvement) < 0.01
    println("  (Improvement ≈0: MLE very close to true parameters, as expected with low noise)")
end

println("\nMLE parameter values:")
for (i, (name, val)) in enumerate(zip(param_names, θ_MLE))
    rel_error = abs(val - θ_true[i]) / θ_true[i] * 100
    println("  $name = $(round(val, sigdigits=4)) (true: $(round(θ_true[i], sigdigits=4)), error: $(round(rel_error, digits=1))%)")
end

# Compute MLE predictions for plotting (on fine grid) using distribution mean
pred_mean_MLE = mean(distrib_fine_θ(θ_MLE))

# --------------------------------------------------------
# Apply IIR at MLE (18 parameters)
# --------------------------------------------------------

println("\n" * repeat("=", 70))
println("Applying IIR at MLE (18 parameters)")
println(repeat("=", 70))

# Wrap in log-space for IIR
ϕ_log(θ_log) = ϕ_func(exp.(θ_log))
θ_log_true = log.(θ_true)
n_params = 18

println("\n" * repeat("=", 70))
println("Applying IIR with finite-difference invariance test...")
println(repeat("=", 70))

t_iir_start = time()
S, N, N_perp, rank_J = find_invariant_subspace(
    ϕ_log, θ_log_MLE;  # ✓ CORRECT - using MLE
    invariance_method=:finite_difference,
    fd_epsilon=1e-5,
    fd_n_probes=5,
    atolM=1e-6  # Relaxed tolerance for approximate invariance
)
t_iir_elapsed = time() - t_iir_start
println("\nIIR analysis time: $(round(t_iir_elapsed, digits=1)) seconds")

println("\nSingular values of Jacobian (in log-space):")
for (i, s) in enumerate(S)
    if i <= 10 || i > length(S) - 3
        println("  σ[$i] = ", round(s, sigdigits=6))
    elseif i == 11
        println("  ...")
    end
end

println("\nInvariant Subspace Analysis:")
println("  Jacobian rank: $rank_J / $n_params")
println("  Expected rank: 15 or 16 (with 2-3 non-identifiable combinations)")
println("  Identifiable directions: ", size(N_perp, 2))
println("  Non-identifiable directions: ", size(N, 2))

# Compute Jacobian at MLE for later use
J_θ_log = ForwardDiff.jacobian(ϕ_log, θ_log_MLE)

# Print the invariant null space vectors
if size(N, 2) > 0
    println("\n" * repeat("=", 70))
    println("NON-IDENTIFIABLE (INVARIANT) DIRECTIONS - SVD Basis (Mixed)")
    println("Expected structure: Products of K/β ratios")
    println(repeat("=", 70))

    for j in 1:size(N, 2)
        v = N[:, j]
        println("\nSVD Direction $j:")

        # Find significant coefficients
        active_terms = [(i, v[i], param_names[i]) for i in 1:n_params if abs(v[i]) > 0.05]

        if !isempty(active_terms)
            println("  Significant coefficients:")
            for (idx, coef, name) in active_terms
                println("    $(name): $(round(coef, digits=3))")
            end

            # Check if β and K have same coefficients (βK product pattern)
            println("  Pattern check (βK product if coefficients match):")
            all_match = true
            for (gene_idx, beta_idx) in enumerate(BETA_INDICES)
                K_idx = K_INDICES[gene_idx]
                beta_coef = v[beta_idx]
                K_coef = v[K_idx]
                match = abs(beta_coef - K_coef) < 0.01
                println("    β$(gene_idx): $(round(beta_coef, digits=3)), K$(gene_idx): $(round(K_coef, digits=3)) → $(match ? "✓ Same" : "✗ Different")")
                all_match = all_match && match
            end
            if all_match
                println("  → Confirms βK product structure (invariant null space)!")
            end
        end
    end

    # KEY INSIGHT from IIR theory:
    # "Invariant" refers to structural invariance of the Jacobian
    # - Invariant null space: subspace where Jacobian structure is preserved under perturbation
    # - Complement N_perp: contains parameter combinations output depends on (identifiable)
    # See IIR paper line 271: "identifiable parameter combinations are defined by
    # the orthogonal complement of V₀"
    #
    # For repressilator: invariant null space contains βK products
    #                    complement contains K/β ratios (identifiable per Eisenberg)

    println("\n" * repeat("=", 70))
    println("INVARIANT NULL SPACE (βK products)")
    println(repeat("=", 70))

    # Now examine N_perp to verify it contains K/β ratios
    println("\n" * repeat("=", 70))
    println("COMPLEMENT SPACE N_perp (Should contain K/β ratios)")
    println(repeat("=", 70))
    println("Dimensions: ", size(N_perp))

    # Apply Varimax to N_perp to see if we can find K/β ratios
    println("\nApplying Varimax rotation to N_perp...")
    N_perp_varimax = varimax_rotation(N_perp; n_restarts=200, threshold=1e-2)

    # Helper functions for identifying ratio patterns
    ratio_score(vec, gene_idx) = begin
        beta_coef = vec[BETA_INDICES[gene_idx]]
        K_coef = vec[K_INDICES[gene_idx]]
        magnitude_penalty = abs(abs(beta_coef) - abs(K_coef))
        other_energy = sum(abs.(vec)) - abs(beta_coef) - abs(K_coef)
        sign_penalty = beta_coef * K_coef > 0 ? 1.0 : 0.0
        base_score = magnitude_penalty + other_energy + sign_penalty
        # Discourage selecting trivial zero vectors
        if abs(beta_coef) + abs(K_coef) < 1e-6
            return Inf
        end
        return base_score
    end

    best_ratio_column(matrix, gene_idx) = begin
        best_ratio_score = Inf
        best_j = nothing
        for j in 1:size(matrix, 2)
            score = ratio_score(matrix[:, j], gene_idx)
            if score < best_ratio_score
                best_ratio_score = score
                best_j = j
            end
        end
        return best_j, best_ratio_score
    end

    # Look for directions with opposite signs for β and K (ratios)
    println("\nSearching for K/β ratio patterns in N_perp...")
    ratio_threshold_primary = 0.2
    ratio_directions = Tuple{Int, Int, Float64, Float64}[]
    for j in 1:size(N_perp_varimax, 2)
        v = N_perp_varimax[:, j]

        # Check each potential K/β pair
        for gene_idx in eachindex(BETA_INDICES)
            beta_coef = v[BETA_INDICES[gene_idx]]
            K_coef = v[K_INDICES[gene_idx]]

            # Ratio pattern: opposite signs, both non-negligible
            if abs(beta_coef) > ratio_threshold_primary &&
               abs(K_coef) > ratio_threshold_primary &&
               sign(beta_coef) != sign(K_coef)
                push!(ratio_directions, (j, gene_idx, beta_coef, K_coef))
                println("  Direction $j: β$(gene_idx)=$(round(beta_coef, digits=3)), K$(gene_idx)=$(round(K_coef, digits=3))")
                println("    → K$(gene_idx)/β$(gene_idx) ratio pattern!")
            end
        end
    end

    if isempty(ratio_directions)
        println("  ⚠ No clear K/β ratio patterns found with threshold $(ratio_threshold_primary). Relaxing threshold...")
        ratio_threshold_relaxed = 0.1
        for j in 1:size(N_perp_varimax, 2)
            v = N_perp_varimax[:, j]
            for gene_idx in eachindex(BETA_INDICES)
                beta_coef = v[BETA_INDICES[gene_idx]]
                K_coef = v[K_INDICES[gene_idx]]
                if abs(beta_coef) > ratio_threshold_relaxed &&
                   abs(K_coef) > ratio_threshold_relaxed &&
                   sign(beta_coef) != sign(K_coef)
                    push!(ratio_directions, (j, gene_idx, beta_coef, K_coef))
                    println("  (relaxed) Direction $j: β$(gene_idx)=$(round(beta_coef, digits=3)), K$(gene_idx)=$(round(K_coef, digits=3))")
                end
            end
        end
    end

    if isempty(ratio_directions)
        println("  ⚠ Still no obvious ratio patterns — applying heuristic search.")
        for gene_idx in eachindex(BETA_INDICES)
            best_tuple = nothing
            best_ratio_score_local = Inf
            for j in 1:size(N_perp_varimax, 2)
                v = N_perp_varimax[:, j]
                score = ratio_score(v, gene_idx)
                if score < best_ratio_score_local
                    best_ratio_score_local = score
                    best_tuple = (j, gene_idx, v[BETA_INDICES[gene_idx]], v[K_INDICES[gene_idx]])
                end
            end
            if !isnothing(best_tuple)
                push!(ratio_directions, best_tuple)
                println("  (heuristic) Gene $(gene_idx) best match at column $(best_tuple[1]) with score $(round(best_ratio_score_local, digits=3))")
            end
        end
    end

    if isempty(ratio_directions)
        println("  ⚠ Unable to identify K/β ratio patterns in N_perp. Downstream heuristics will attempt recovery.")
    else
        println("\nFound $(length(ratio_directions)) K/β ratio patterns in N_perp ✓")
    end

    ratio_direction_map = Dict{Tuple{Int, Int}, Tuple{Int, Int, Float64, Float64}}()
    for entry in ratio_directions
        ratio_direction_map[(entry[1], entry[2])] = entry
    end
    ratio_directions = sort(collect(values(ratio_direction_map)), by=x->(x[2], x[1]))

    # Measure degree of identifiability
    println("\n" * repeat("=", 70))
    println("DEGREE OF IDENTIFIABILITY Analysis")
    println(repeat("=", 70))

    # Use singular values from find_invariant_subspace
    println("\nOriginal SVD basis (N_perp):")
    println("  All singular values (identifiable directions):")
    for i in 1:rank_J
        println("    σ[$i] = $(round(S[i], digits=3))")
    end
    println("  Condition number: $(round(S[1]/S[rank_J], digits=2))")

    # Check which SVD directions have K/β structure
    # Note: N_perp already contains the identifiable directions (columns of V)
    println("\n  Checking SVD basis for K/β ratio patterns:")
    V_r_from_svd = N_perp

    svd_kb_directions = []
    for i in 1:rank_J
        v = V_r_from_svd[:, i]
        # Check if this direction shows K/β ratio pattern (opposite signs)
        for gene_idx in eachindex(BETA_INDICES)
            beta_coef = v[BETA_INDICES[gene_idx]]
            K_coef = v[K_INDICES[gene_idx]]
            if abs(beta_coef) > 0.3 && abs(K_coef) > 0.3 && sign(beta_coef) != sign(K_coef)
                push!(svd_kb_directions, (i, gene_idx, S[i]))
                println("    SVD[$i] (σ=$(round(S[i], digits=3))): K$(gene_idx)/β$(gene_idx) ratio (β=$(round(beta_coef, digits=3)), K=$(round(K_coef, digits=3)))")
            end
        end
    end

    if isempty(svd_kb_directions)
        println("    No clear K/β ratio patterns in SVD basis (mixed combinations)")
    end

    println("\nVarimax-rotated N_perp basis:")
    println("  Effective singular values ||J·v|| for each rotated direction:")

    # Compute ||J·v|| for specific K/β ratio directions
    for (j, i, beta_coef, K_coef) in ratio_directions
        v = N_perp_varimax[:, j]
        eff_sigma = norm(J_θ_log * v)
        println("    K$i/β$i (Direction $j): σ_eff = $(round(eff_sigma, digits=3))")
    end

    # For comparison, compute for all N_perp_varimax directions
    all_eff_sigmas = [norm(J_θ_log * N_perp_varimax[:, j]) for j in 1:size(N_perp_varimax, 2)]
    ratio_indices = unique([r[1] for r in ratio_directions])

    println("\n  Effective σ range for all N_perp_varimax: [$(round(minimum(all_eff_sigmas), digits=3)), $(round(maximum(all_eff_sigmas), digits=3))]")
    if isempty(ratio_indices)
        println("  No K/β ratio directions identified; skipping ratio-specific σ range.")
    else
        ratio_eff_sigmas = all_eff_sigmas[ratio_indices]
        println("  Effective σ range for K/β ratios: [$(round(minimum(ratio_eff_sigmas), digits=3)), $(round(maximum(ratio_eff_sigmas), digits=3))]")
        println("  K/β ratios condition number: $(round(maximum(ratio_eff_sigmas)/minimum(ratio_eff_sigmas), digits=2))")
    end

    # Build full transformation matrices (SVD vs Varimax)
    println("\n" * repeat("=", 70))
    println("REPARAMETERIZATION COMPARISON: SVD vs Varimax")
    println(repeat("=", 70))

    # SVD-based transformation (as returned by find_invariant_subspace)
    A_svd_T = hcat(N_perp, N)
    A_svd = A_svd_T'

    println("\nSVD-based transformation matrix A_svd:")
    println("  Dimensions: ", size(A_svd))
    println("  First $rank_J rows (identifiable): from N_perp")
    println("  Last $(n_params - rank_J) rows (non-identifiable): from N")

    # Show the non-identifiable combinations from SVD
    println("\n  Non-identifiable directions (SVD basis):")
    param_names_short = ["α₀₁", "α₀₂", "α₀₃", "α₁", "α₂", "α₃", "β₁", "β₂", "β₃", "K₁", "K₂", "K₃",
                         "k_dm₁", "k_dm₂", "k_dm₃", "k_dp₁", "k_dp₂", "k_dp₃"]

    for i in 1:size(N, 2)
        row = A_svd[rank_J+i, :]
        println("    ψ[$(rank_J+i)] = ", join([round(row[j], digits=3) for j in 1:n_params], ", "))
    end

    # Varimax-based transformation
    println("\n" * repeat("=", 70))
    println("Applying Varimax rotation separately to N_perp and N...")
    println(repeat("=", 70))

    N_varimax = varimax_rotation(N; n_restarts=200, threshold=1e-2)

    # Apply scale_and_round to get clean integer/half-integer patterns
    # IMPORTANT: This does NOT change identifiability! The normalized sensitivity
    # ||J·v||/||v|| remains the same. It only rescales vectors to have nice
    # integer exponents (±1 instead of ±0.707 = ±1/√2)
    #
    # WARNING: scale_and_round destroys orthonormality! The scaled bases are no
    # longer orthonormal. A_varimax is for presentation/interpretation only.
    # For actual transformations requiring orthogonality, use A_svd or re-orthonormalize.
    println("\nApplying scale_and_round to Varimax-rotated bases...")
    println("  (Note: This rescales for clean exponents but doesn't change identifiability)")
    println("  (Warning: This breaks orthonormality - A_varimax is for presentation only)")
    N_perp_clean = scale_and_round(N_perp_varimax; round_within=0.1)
    N_clean = scale_and_round(N_varimax; round_within=0.1)

    # Build Varimax transformation matrix (for presentation/interpretation)
    A_varimax_T = hcat(N_perp_clean, N_clean)
    A_varimax = A_varimax_T'

    println("\nVarimax-based transformation matrix A_varimax:")
    println("  Dimensions: ", size(A_varimax))

    # ======================================================================
    # CREATE IIR TRANSFORMATIONS FOR PROFILING
    # ======================================================================
    println("\n" * repeat("=", 70))
    println("CREATING IIR TRANSFORMATIONS FOR PROFILING")
    println(repeat("=", 70))

    # Create forward and inverse transformations: ψ = exp(A * log(θ))
    θ_to_ψ, ψ_to_θ = reparam(A_varimax_T)

    # Transform MLE to ψ space
    ψ_MLE = θ_to_ψ(θ_MLE)
    ψ_log_MLE = log.(ψ_MLE)
    println("\nMLE in ψ coordinates:")
    for i in 1:min(5, length(ψ_MLE))
        println("  ψ[$i] = $(round(ψ_MLE[i], digits=4))")
    end

    # Derive ψ-space bounds by mapping θ bounds through the log-linear transform
    ψ_log_lower_bounds = similar(ψ_MLE)
    ψ_log_upper_bounds = similar(ψ_MLE)
    for i in 1:n_params
        row = A_varimax[i, :]
        lower_val = 0.0
        upper_val = 0.0
        for j in 1:n_params
            coef = row[j]
            if coef >= 0
                lower_val += coef * θ_log_lower[j]
                upper_val += coef * θ_log_upper[j]
            else
                lower_val += coef * θ_log_upper[j]
                upper_val += coef * θ_log_lower[j]
            end
        end
        if lower_val > upper_val
            lower_val, upper_val = upper_val, lower_val
        end
        ψ_log_lower_bounds[i] = lower_val
        ψ_log_upper_bounds[i] = upper_val
    end
    expansion_margin = 2.0
    ψ_log_lower_bounds .= min.(ψ_log_lower_bounds, ψ_log_MLE .- expansion_margin)
    ψ_log_upper_bounds .= max.(ψ_log_upper_bounds, ψ_log_MLE .+ expansion_margin)
    ψ_lower_bounds = exp.(ψ_log_lower_bounds)
    ψ_upper_bounds = exp.(ψ_log_upper_bounds)

    # Create likelihood in ψ space
    function lnlike_ψ(ψ)
        θ = ψ_to_θ(ψ)
        return lnlike_θ(θ)
    end

    lnlike_ψ_log(ψ_log) = lnlike_ψ(exp.(ψ_log))

    println("\nLikelihood in ψ space created")
    println("  lnlike_ψ(ψ_MLE) = $(round(lnlike_ψ(ψ_MLE), digits=4))")
    println("  lnlike_θ(θ_MLE) = $(round(lnlike_θ(θ_MLE), digits=4))")
    println("  Match: $(isapprox(lnlike_ψ(ψ_MLE), lnlike_θ(θ_MLE), atol=1e-6))")

    # Identify which ψ index corresponds to the β₁/K₁ combination (use heuristic if needed)
    candidate_columns = [j for (j, gene_idx, _, _) in ratio_directions if gene_idx == 1]
    best_column, best_ratio_score = best_ratio_column(N_perp_clean, 1)
    if isnothing(best_column)
        error("Could not identify a K₁/β₁ ratio direction in N_perp!")
    end

    if isempty(candidate_columns)
        println("\nNo direct β₁/K₁ hits in ratio_directions; using heuristic best column $(best_column) with score $(round(best_ratio_score, digits=3)).")
    elseif best_column ∉ candidate_columns
        println("\nHeuristic refined β₁/K₁ combination assignment from columns $(candidate_columns) to $best_column (score $(round(best_ratio_score, digits=3))).")
    else
        println("\nFound β₁/K₁ combination at N_perp column $best_column (score $(round(best_ratio_score, digits=3))).")
    end

    # The ψ index is the column index in N_perp (since A = [N_perp'; N'])
    ψ_K1_β1_index = best_column
    println("  β₁/K₁ combination is ψ[$ψ_K1_β1_index] in IIR coordinates")
    println("  ψ[$ψ_K1_β1_index](MLE) = $(round(ψ_MLE[ψ_K1_β1_index], digits=6))")
    println("  True β₁/K₁ = $(round(θ_true[7]/θ_true[10], digits=6))  (inverse K₁/β₁ = $(round(θ_true[10]/θ_true[7], digits=2)))")

    # Tighten profiling bounds around β₁/K₁ by focusing on a small multiplicative band about the MLE
    β1_index_local = 7
    K1_index_local = 10
    ratio_beta_over_K_mle = exp(θ_log_MLE[β1_index_local] - θ_log_MLE[K1_index_local])
    log_center = log(ratio_beta_over_K_mle)
    log_halfwidth = log(5.0)  # allow 5× variation in each direction
    ψ_log_lower_bounds[ψ_K1_β1_index] = max(ψ_log_lower_bounds[ψ_K1_β1_index], log_center - log_halfwidth)
    ψ_log_upper_bounds[ψ_K1_β1_index] = min(ψ_log_upper_bounds[ψ_K1_β1_index], log_center + log_halfwidth)

    # Create distribution function in ψ space for prediction intervals
    distrib_fine_ψ(ψ) = distrib_fine_θ(ψ_to_θ(ψ))
    distrib_fine_ψ_log(ψ_log) = distrib_fine_ψ(exp.(ψ_log))

    println("\nDistribution in ψ space created for predictions")

    # Show the non-identifiable combinations from Varimax
    println("\n  Non-identifiable directions (Varimax basis):")
    for i in 1:size(N_varimax, 2)
        row = A_varimax[rank_J+i, :]
        println("    ψ[$(rank_J+i)] = ", join([round(row[j], digits=3) for j in 1:n_params], ", "))
    end

    # Show symbolic monomials for both
    println("\n" * repeat("=", 70))
    println("SYMBOLIC MONOMIAL COMPARISONS")
    println(repeat("=", 70))

    function format_monomial(row, param_names, threshold=0.05)
        terms = String[]
        for j in 1:length(row)
            coef = row[j]
            if abs(coef) > threshold
                name = param_names[j]
                if abs(coef - 1.0) < 0.01
                    push!(terms, name)
                elseif abs(coef + 1.0) < 0.01
                    push!(terms, name * "⁻¹")
                else
                    push!(terms, name * "^" * string(round(coef, digits=2)))
                end
            end
        end
        return isempty(terms) ? "1" : join(terms, "·")
    end

    println("\nNon-identifiable monomials (SVD basis):")
    for i in 1:size(N, 2)
        row = A_svd[rank_J+i, :]
        mono = format_monomial(row, param_names_short)
        println("  ψ[$(rank_J+i)] = $mono")
    end

    println("\nNon-identifiable monomials (Varimax basis):")
    for i in 1:size(N_varimax, 2)
        row = A_varimax[rank_J+i, :]
        mono = format_monomial(row, param_names_short)
        println("  ψ[$(rank_J+i)] = $mono")
    end

    # Compare degree of identifiability: compute ||J·ψ|| for transformed parameters
    println("\n" * repeat("=", 70))
    println("IDENTIFIABILITY COMPARISON: SVD vs Varimax monomials")
    println(repeat("=", 70))

    # For SVD non-identifiable combos
    println("\nSVD non-identifiable combinations:")
    for i in 1:size(N, 2)
        v_orig = N[:, i]  # Direction in original parameter space
        Jv_norm = norm(J_θ_log * v_orig)
        row = A_svd[rank_J+i, :]
        mono = format_monomial(row, param_names_short)
        println("  ψ[$(rank_J+i)] = $mono")
        println("    ||J·v|| = $(round(Jv_norm, digits=6)) (should be ≈0 for non-identifiable)")
    end

    # For Varimax non-identifiable combos
    println("\nVarimax non-identifiable combinations (after scale_and_round):")
    for i in 1:size(N_clean, 2)
        v_orig = N_clean[:, i]
        Jv_norm = norm(J_θ_log * v_orig)
        row = A_varimax[rank_J+i, :]
        mono = format_monomial(row, param_names_short)
        println("  ψ[$(rank_J+i)] = $mono")
        println("    ||J·v|| = $(round(Jv_norm, digits=6)) (should be ≈0 for non-identifiable)")
    end

    # NOW THE KEY PART: Compare identifiable directions (N_perp)
    println("\n" * repeat("=", 70))
    println("IDENTIFIABLE DIRECTIONS (N_perp): SVD vs Varimax")
    println(repeat("=", 70))

    println("\nSVD identifiable combinations (all $rank_J):")
    for i in 1:size(N_perp, 2)
        v_orig = N_perp[:, i]
        Jv_norm = norm(J_θ_log * v_orig)
        row = A_svd[i, :]
        mono = format_monomial(row, param_names_short, 0.1)
        println("  ψ[$i] = $mono")
        println("    σ_eff = $(round(Jv_norm, digits=3))")
    end

    println("\nVarimax identifiable combinations - K/β ratios:")
    println("  Comparing before/after scale_and_round:")
    println("  (Note: ||J·v||/||v|| is the identifiability measure - should stay constant)")
    for (j, i, beta_coef, K_coef) in ratio_directions
        v_clean = N_perp_clean[:, j]
        v_varimax = N_perp_varimax[:, j]

        Jv_norm_clean = norm(J_θ_log * v_clean)
        Jv_norm_varimax = norm(J_θ_log * v_varimax)

        v_norm_clean = norm(v_clean)
        v_norm_varimax = norm(v_varimax)

        row = A_varimax[j, :]
        mono = format_monomial(row, param_names_short, 0.01)
        println("\n  ψ[$j] (K$i/β$i) = $mono")
        println("    Varimax:        ||v||=$(round(v_norm_varimax, digits=3)), ||J·v||=$(round(Jv_norm_varimax, digits=3)), σ_eff=||J·v||/||v||=$(round(Jv_norm_varimax/v_norm_varimax, digits=3))")
        println("    After scaling:  ||v||=$(round(v_norm_clean, digits=3)), ||J·v||=$(round(Jv_norm_clean, digits=3)), σ_eff=||J·v||/||v||=$(round(Jv_norm_clean/v_norm_clean, digits=3))")
    end

    println("\n  All other Varimax identifiable directions:")
    for j in 1:size(N_perp_clean, 2)
        if !(j in ratio_indices)
            v_orig = N_perp_clean[:, j]
            Jv_norm = norm(J_θ_log * v_orig)
            v_norm = norm(v_orig)
            row = A_varimax[j, :]
            mono = format_monomial(row, param_names_short, 0.01)
            println("  ψ[$j] = $mono")
            println("    σ_eff = $(round(Jv_norm/v_norm, digits=3))")
        end
    end

    # SORTED COMPARISON TABLE
    println("\n" * repeat("=", 70))
    println("SORTED IDENTIFIABILITY COMPARISON: SVD vs Varimax")
    println(repeat("=", 70))

    # Collect and sort SVD results
    svd_results = []
    for i in 1:size(N_perp, 2)
        v_orig = N_perp[:, i]
        sigma = norm(J_θ_log * v_orig)
        row = A_svd[i, :]
        mono = format_monomial(row, param_names_short, 0.1)
        push!(svd_results, (i, mono, sigma))
    end
    sort!(svd_results, by=x->x[3], rev=true)

    # Collect and sort Varimax results
    varimax_results = []
    for j in 1:size(N_perp_clean, 2)
        v_orig = N_perp_clean[:, j]
        sigma = norm(J_θ_log * v_orig) / norm(v_orig)  # Normalized
        row = A_varimax[j, :]
        mono = format_monomial(row, param_names_short, 0.01)

        # Classify type
        nonzero_count = count(abs.(row) .> 0.01)
        if j in ratio_indices
            ptype = "β/K ratio"
        elseif nonzero_count == 1
            ptype = "single"
        elseif nonzero_count == 2
            ptype = "2-param"
        else
            ptype = "mixed"
        end

        push!(varimax_results, (j, mono, sigma, ptype))
    end
    sort!(varimax_results, by=x->x[3], rev=true)

    # Print sorted comparison
    println("\nRank | SVD Combination (σ_eff) | Varimax Combination (σ_eff) | Type")
    println(repeat("-", 70))

    for rank in 1:rank_J
        svd_idx, svd_mono, svd_sigma = svd_results[rank]
        var_idx, var_mono, var_sigma, var_type = varimax_results[rank]

        # Truncate SVD mono if too long
        svd_display = length(svd_mono) > 25 ? svd_mono[1:22]*"..." : svd_mono

        println("$(lpad(rank,2)) | $(rpad(svd_display,25)) ($(rpad(round(svd_sigma,digits=1),6))) | $(rpad(var_mono,20)) ($(rpad(round(var_sigma,digits=1),6))) | $var_type")
    end

    # Summary statistics
    println("\n" * repeat("-", 70))
    println("Summary:")

    svd_singles = count(x -> occursin("⁻¹", x[2]) && count(c->c=='·', x[2])==0, svd_results)
    var_singles = count(x -> x[4] == "single", varimax_results)
    var_ratios = count(x -> x[4] == "β/K ratio", varimax_results)

    println("  SVD: $(rank_J-svd_singles) mixed + $svd_singles single = $rank_J total")
    println("  Varimax: $var_singles single + $var_ratios β/K ratios + $(rank_J-var_singles-var_ratios) other = $rank_J total")
    println("  Varimax interpretability: $(var_singles+var_ratios)/$rank_J simple combinations ($(round(100*(var_singles+var_ratios)/rank_J, digits=1))%)")

    println("\nVarimax-rotated null directions (βK products):")

    for j in 1:size(N_varimax, 2)
        v = N_varimax[:, j]
        println("\nVarimax Direction $j:")
        println("  All coefficients: ", [round(v[i], digits=3) for i in 1:n_params])

        # Look for pattern where one β and one K dominate
        for (gene_idx, beta_idx) in enumerate(BETA_INDICES)
            K_idx = K_INDICES[gene_idx]
            beta_coef = v[beta_idx]
            K_coef = v[K_idx]

            if abs(beta_coef) > 0.3 && abs(K_coef) > 0.3
                # Found dominant β,K pair
                # Same sign → product, opposite sign → ratio
                if sign(beta_coef) == sign(K_coef)
                    println("  → (β$(gene_idx)·K$(gene_idx)) product")
                else
                    ratio_str = beta_coef > 0 ? "β$(gene_idx)/K$(gene_idx)" : "K$(gene_idx)/β$(gene_idx)"
                    println("  → ($ratio_str) ratio")
                end

                # Compute the product and ratio values (compare MLE vs true)
                K_val_MLE = exp(θ_log_MLE[K_idx])
                beta_val_MLE = exp(θ_log_MLE[beta_idx])
                K_val_true = exp(θ_log_true[K_idx])
                beta_val_true = exp(θ_log_true[beta_idx])

                product_val_MLE = beta_val_MLE * K_val_MLE
                ratio_val_MLE = K_val_MLE / beta_val_MLE
                beta_over_K_MLE = beta_val_MLE / K_val_MLE
                product_val_true = beta_val_true * K_val_true
                ratio_val_true = K_val_true / beta_val_true
                beta_over_K_true = beta_val_true / K_val_true

                println("    MLE:  K$(gene_idx) = $(round(K_val_MLE, digits=4)), β$(gene_idx) = $(round(beta_val_MLE, digits=4))")
                println("    True: K$(gene_idx) = $(round(K_val_true, digits=4)), β$(gene_idx) = $(round(beta_val_true, digits=4))")
                println("    β$(gene_idx)·K$(gene_idx) (MLE) = $(round(product_val_MLE, digits=2)) (in invariant null space)")
                println("    β$(gene_idx)/K$(gene_idx) (MLE) = $(round(beta_over_K_MLE, digits=6)) (identifiable combination)")
                println("    K$(gene_idx)/β$(gene_idx) (MLE) = $(round(ratio_val_MLE, digits=2)) (inverse)")
                println("    β$(gene_idx)/K$(gene_idx) (true) = $(round(beta_over_K_true, digits=6))")
                println("    K$(gene_idx)/β$(gene_idx) (true) = $(round(ratio_val_true, digits=2))  (Eisenberg reference)")
            end
        end
    end
else
    println("\n⚠ WARNING: No invariant null space found!")
    println("This suggests the model doesn't have IIR-compatible invariant structure.")
end

# Profile K₁ and β₁ in parallel using threading (for comparison)
println("\nProfiling K₁ and β₁ individually in θ-space (comparison, using $(Threads.nthreads()) threads)...")

K1_index = 10
β1_index = 7
n_guesses = CONFIG.n_guesses

# Pre-allocate result containers
profile_results = Vector{Any}(undef, 2)

Threads.@threads for i in 1:2
    param_index = i == 1 ? K1_index : β1_index
    param_name = i == 1 ? "K₁" : "β₁"

    println("\n$(i). Profiling $param_name (parameter $param_index, non-identifiable)...")

    nuisance_indices = setdiff(1:n_params, param_index)
    nuisance_guess = θ_log_MLE[nuisance_indices]

    nuisance_extras = generate_initial_guesses(
        θ_log_lower[nuisance_indices],
        θ_log_upper[nuisance_indices],
        n_guesses)

    ψ_values, lnlike_values = profile_target(
        lnlike_θ_log, param_index,
        θ_log_lower, θ_log_upper,
        nuisance_guess;
        grid_steps=[CONFIG.grid_1d],
        ω_initial_extras=nuisance_extras,
        method=:LN_BOBYQA,
        optmaxtime=CONFIG.timeout)

    profile_vals = [ψ[param_index] for ψ in ψ_values]
    println("  Profiled $param_name range: [$(round(exp(minimum(profile_vals)), digits=2)), $(round(exp(maximum(profile_vals)), digits=2))]")

    # Compute prediction intervals
    lower, upper, _ = construct_upper_lower_profile_wise_CIs_for_mean(
        distrib_fine_θ_log, ψ_values, lnlike_values; l_level=95, df=rank_J)

    # Store results
    profile_results[i] = (ψ_values, lnlike_values, lower, upper, profile_vals)
end

# Extract results from parallel computation
ψK1_values, lnlike_K1_values, lower_K1, upper_K1, K1_profile_vals = profile_results[1]
ψβ1_values, lnlike_β1_values, lower_β1, upper_β1, β1_profile_vals = profile_results[2]

# Profile β₁/K₁ combination directly in ψ-space (identifiable combination)
println("\nProfiling identifiable β₁/K₁ combination directly in ψ-space (reporting inverse K₁/β₁ for comparison)...")
nuisance_indices_ratio = setdiff(1:n_params, ψ_K1_β1_index)
nuisance_guess_ratio = ψ_log_MLE[nuisance_indices_ratio]

nuisance_extras_ratio = generate_initial_guesses(
    ψ_log_lower_bounds[nuisance_indices_ratio],
    ψ_log_upper_bounds[nuisance_indices_ratio],
    n_guesses)

ψ_ratio_log_values, lnlike_ratio_values = profile_target(
    lnlike_ψ_log, ψ_K1_β1_index,
    ψ_log_lower_bounds, ψ_log_upper_bounds,
    nuisance_guess_ratio;
    grid_steps=[CONFIG.grid_1d],
    ω_initial_extras=nuisance_extras_ratio,
    method=:LN_BOBYQA,
    optmaxtime=CONFIG.timeout)

ratio_log_vals = [ψ[ψ_K1_β1_index] for ψ in ψ_ratio_log_values]
beta_over_K_values = exp.(ratio_log_vals)
K_over_beta_values = 1 ./ beta_over_K_values
println("  β₁/K₁ range: [$(round(minimum(beta_over_K_values), digits=6)), $(round(maximum(beta_over_K_values), digits=6))]")
println("  Equivalent K₁/β₁ range: [$(round(minimum(K_over_beta_values), digits=2)), $(round(maximum(K_over_beta_values), digits=2))]")
println("  True β₁/K₁ = $(round(θ_true[β1_index]/θ_true[K1_index], digits=6))  (K₁/β₁ = $(round(θ_true[K1_index]/θ_true[β1_index], digits=2)))")
ratio_log_mle = ψ_log_MLE[ψ_K1_β1_index]
ratio_mle_distance = minimum(abs.(ratio_log_vals .- ratio_log_mle))
println("  Distance from profiled grid to log-MLE: $(ratio_mle_distance)")
if ratio_mle_distance > 1e-2
    println("  ⚠ MLE log-value lies outside current grid resolution; consider increasing CONFIG.grid_1d or providing better initial guesses.")
end

# Plot 1D profiles for diagnostics (ratio plus individual parameters)
ratio_plot_idx = sortperm(K_over_beta_values)
ratio_values_sorted = K_over_beta_values[ratio_plot_idx]
lnlike_ratio_sorted = lnlike_ratio_values[ratio_plot_idx]
lnlike_ratio_norm = lnlike_ratio_sorted .- maximum(lnlike_ratio_sorted)

ratio_mle = θ_MLE[K1_index] / θ_MLE[β1_index]
plot_1D_profile("repressilator",
    ratio_values_sorted, lnlike_ratio_norm, "K_{1}/\\beta_{1}";
    varname_save="K1_over_beta1_ratio",
    ψ_true=θ_true[K1_index]/θ_true[β1_index],
    ψ_MLE=ratio_mle,
    save_dir=joinpath(@__DIR__, "..", "figures") * "/")

K1_values = exp.([ψ[K1_index] for ψ in ψK1_values])
K1_plot_idx = sortperm(K1_values)
K1_values_sorted = K1_values[K1_plot_idx]
lnlike_K1_sorted = lnlike_K1_values[K1_plot_idx]
lnlike_K1_norm = lnlike_K1_sorted .- maximum(lnlike_K1_sorted)

plot_1D_profile("repressilator",
    K1_values_sorted, lnlike_K1_norm, "K_{1}";
    varname_save="K1_theta_space",
    ψ_true=θ_true[K1_index],
    ψ_MLE=θ_MLE[K1_index],
    save_dir=joinpath(@__DIR__, "..", "figures") * "/")

β1_values = exp.([ψ[β1_index] for ψ in ψβ1_values])
β1_plot_idx = sortperm(β1_values)
β1_values_sorted = β1_values[β1_plot_idx]
lnlike_β1_sorted = lnlike_β1_values[β1_plot_idx]
lnlike_β1_norm = lnlike_β1_sorted .- maximum(lnlike_β1_sorted)

plot_1D_profile("repressilator",
    β1_values_sorted, lnlike_β1_norm, "\\beta_{1}";
    varname_save="beta1_theta_space",
    ψ_true=θ_true[β1_index],
    ψ_MLE=θ_MLE[β1_index],
    save_dir=joinpath(@__DIR__, "..", "figures") * "/")

# Prediction intervals using ψ-space distribution
lower_ratio, upper_ratio, _ = construct_upper_lower_profile_wise_CIs_for_mean(
    distrib_fine_ψ_log, ψ_ratio_log_values, lnlike_ratio_values; l_level=95, df=rank_J)

# Optional: retain legacy 2D joint profile for validation if requested
if CONFIG.do_2d
    println("\n(Optional) Running 2D joint profile of (β₁, K₁) in θ-space for validation...")
    target_indices_K1β1 = [β1_index, K1_index]
    nuisance_indices_2d = setdiff(1:n_params, target_indices_K1β1)
    nuisance_guess_2d = θ_log_MLE[nuisance_indices_2d]

    nuisance_extras_2d = generate_initial_guesses(
        θ_log_lower[nuisance_indices_2d],
        θ_log_upper[nuisance_indices_2d],
        n_guesses)

    ψK1β1_values, lnlike_K1β1_values = profile_target(
        lnlike_θ_log, target_indices_K1β1,
        θ_log_lower, θ_log_upper,
        nuisance_guess_2d;
        grid_steps=CONFIG.grid_2d,
        ω_initial_extras=nuisance_extras_2d,
        method=:LN_BOBYQA,
        optmaxtime=CONFIG.timeout)

    ratio_values_joint = [exp(ψ[K1_index] - ψ[β1_index]) for ψ in ψK1β1_values]
    println("  Joint profile ratio range: [$(round(minimum(ratio_values_joint), digits=2)), $(round(maximum(ratio_values_joint), digits=2))]")
end

println("\nPrediction interval widths (mean across time/species):")
width_K1 = mean(upper_K1 - lower_K1)
width_β1 = mean(upper_β1 - lower_β1)
width_ratio = mean(upper_ratio - lower_ratio)
println("  K₁ individual:   ", round(width_K1, digits=4))
println("  β₁ individual:   ", round(width_β1, digits=4))
println("  β₁/K₁ identifiable: ", round(width_ratio, digits=4), "  (inverse K₁/β₁ shares the same prediction band)")

# Reshape predictions for plotting (currently flattened)
# Shape: 3 mRNA × NT_pred time points
reshape_pred(v) = reshape(v, 3, length(t_pred))

lower_K1_mat = reshape_pred(lower_K1)
upper_K1_mat = reshape_pred(upper_K1)
lower_β1_mat = reshape_pred(lower_β1)
upper_β1_mat = reshape_pred(upper_β1)
lower_ratio_mat = reshape_pred(lower_ratio)
upper_ratio_mat = reshape_pred(upper_ratio)
mle_mat = reshape_pred(pred_mean_MLE)

# Extract actual noisy data (data is flattened in time-major order)
data_mat = reshape(data, 3, NT)'  # NT×3

# Plot predictions for all three mRNA species
species_names = ["m_{1}", "m_{2}", "m_{3}"]
species_subscripts = ["1", "2", "3"]

for (i, (name, subscript)) in enumerate(zip(species_names, species_subscripts))
    ci_intervals = [
        (lower_K1_mat[i,:], upper_K1_mat[i,:], "K₁ individual", :red),
        (lower_β1_mat[i,:], upper_β1_mat[i,:], "β₁ individual", :orange),
        (lower_ratio_mat[i,:], upper_ratio_mat[i,:], "β₁/K₁ identifiable (IIR)", :blue)
    ]

    plot_profile_wise_CI_comparison(
        t_pred, mle_mat[i,:],
        ci_intervals,
        "repressilator_$(subscript)", "$name", "t", "t";
        data_indep=t, data_dep=data_mat[:, i],
        title="Repressilator $name: Individual vs Ratio Prediction Intervals",
        save_dir=joinpath(@__DIR__, "..", "figures") * "/",
        show_legend=false,
        show_title=false
    )
end

println("\n" * repeat("=", 70))
println("Analysis Complete")
println(repeat("=", 70))
