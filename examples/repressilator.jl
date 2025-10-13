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

# ========================================================================
# CONFIGURATION: Profiling Settings
# ========================================================================
# Three modes: "test" (~1 min), "paper" (~4 min), "full" (~60 min)
const PROFILE_MODE = "paper"  # Change to "test" or "full" as needed

# Mode configurations
const PROFILE_CONFIGS = Dict(
    "test" => (grid_1d=3, grid_2d=[3,3], timeout=10.0, n_guesses=1, do_2d=false),
    "paper" => (grid_1d=5, grid_2d=[5,5], timeout=10.0, n_guesses=1, do_2d=true),
    "full" => (grid_1d=10, grid_2d=[7,7], timeout=60.0, n_guesses=2, do_2d=true)
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

# --------------------------------------------------------
# Generate synthetic data
# --------------------------------------------------------

ϕ_func = create_ϕ_mapping(t, X0)
y_true = ϕ_func(θ_true)
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
function predict_mRNA(θ, t_grid=t)
    sol_matrix = solve_repressilator(t_grid, θ, X0)
    mRNA = extract_mrna(sol_matrix)  # 3×NT matrix
    return vec(mRNA')  # Flatten to vector (time-major order)
end

# Distribution for predictions (mRNA only, not proteins)
# Use fine grid for smooth prediction bands
distrib_fine_θ = θ -> MvLogNormal(log.(abs.(predict_mRNA(θ, t_pred)) .+ 1e-10), σ_pred^2*I(3*length(t_pred)))

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
    method=:LD_LBFGS,
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

# Compute MLE predictions for plotting (on fine grid)
pred_mean_MLE = predict_mRNA(θ_MLE, t_pred)

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
            # In 18-param space: β₁, β₂, β₃ are at positions 7, 8, 9
            #                    K₁, K₂, K₃ are at positions 10, 11, 12
            beta_indices = [7, 8, 9]
            K_indices = [10, 11, 12]

            println("  Pattern check (βK product if coefficients match):")
            all_match = true
            for i in 1:3
                beta_coef = v[beta_indices[i]]
                K_coef = v[K_indices[i]]
                match = abs(beta_coef - K_coef) < 0.01
                println("    β$i: $(round(beta_coef, digits=3)), K$i: $(round(K_coef, digits=3)) → $(match ? "✓ Same" : "✗ Different")")
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

    # Look for directions with opposite signs for β and K (ratios)
    println("\nSearching for K/β ratio patterns in N_perp...")
    beta_indices = [7, 8, 9]
    K_indices = [10, 11, 12]

    ratio_directions = []
    for j in 1:size(N_perp_varimax, 2)
        v = N_perp_varimax[:, j]

        # Check each potential K/β pair
        for i in 1:3
            beta_coef = v[beta_indices[i]]
            K_coef = v[K_indices[i]]

            # Ratio pattern: opposite signs, both non-negligible
            if abs(beta_coef) > 0.3 && abs(K_coef) > 0.3 && sign(beta_coef) != sign(K_coef)
                push!(ratio_directions, (j, i, beta_coef, K_coef))
                println("  Direction $j: β$i=$(round(beta_coef, digits=3)), K$i=$(round(K_coef, digits=3))")
                println("    → K$i/β$i ratio pattern!")
            end
        end
    end

    if isempty(ratio_directions)
        println("  ⚠ No clear K/β ratio patterns found in N_perp")
        println("  This needs further investigation...")
    else
        println("\nFound $(length(ratio_directions)) K/β ratio patterns in N_perp ✓")
    end

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
    beta_indices = [7, 8, 9]
    K_indices = [10, 11, 12]

    svd_kb_directions = []
    for i in 1:rank_J
        v = V_r_from_svd[:, i]
        # Check if this direction shows K/β ratio pattern (opposite signs)
        for j in 1:3
            beta_coef = v[beta_indices[j]]
            K_coef = v[K_indices[j]]
            if abs(beta_coef) > 0.3 && abs(K_coef) > 0.3 && sign(beta_coef) != sign(K_coef)
                push!(svd_kb_directions, (i, j, S[i]))
                println("    SVD[$i] (σ=$(round(S[i], digits=3))): K$j/β$j ratio (β=$((round(beta_coef, digits=3))), K=$(round(K_coef, digits=3)))")
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
    ratio_indices = [r[1] for r in ratio_directions]
    ratio_eff_sigmas = all_eff_sigmas[ratio_indices]

    println("\n  Effective σ range for all N_perp_varimax: [$(round(minimum(all_eff_sigmas), digits=3)), $(round(maximum(all_eff_sigmas), digits=3))]")
    println("  Effective σ range for K/β ratios: [$(round(minimum(ratio_eff_sigmas), digits=3)), $(round(maximum(ratio_eff_sigmas), digits=3))]")
    println("  K/β ratios condition number: $(round(maximum(ratio_eff_sigmas)/minimum(ratio_eff_sigmas), digits=2))")

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
            ptype = "K/β ratio"
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
    var_ratios = count(x -> x[4] == "K/β ratio", varimax_results)

    println("  SVD: $(rank_J-svd_singles) mixed + $svd_singles single = $rank_J total")
    println("  Varimax: $var_singles single + $var_ratios K/β ratios + $(rank_J-var_singles-var_ratios) other = $rank_J total")
    println("  Varimax interpretability: $(var_singles+var_ratios)/$rank_J simple combinations ($(round(100*(var_singles+var_ratios)/rank_J, digits=1))%)")

    println("\nVarimax-rotated null directions (βK products):")

    for j in 1:size(N_varimax, 2)
        v = N_varimax[:, j]
        println("\nVarimax Direction $j:")
        println("  All coefficients: ", [round(v[i], digits=3) for i in 1:n_params])

        # Check for individual K/β pattern
        # In 18-param space: β₁, β₂, β₃ are at positions 7, 8, 9
        #                    K₁, K₂, K₃ are at positions 10, 11, 12
        beta_indices = [7, 8, 9]
        K_indices = [10, 11, 12]

        # Look for pattern where one β and one K dominate
        for i in 1:3
            beta_coef = v[beta_indices[i]]
            K_coef = v[K_indices[i]]

            if abs(beta_coef) > 0.3 && abs(K_coef) > 0.3
                # Found dominant β,K pair
                # Same sign → product, opposite sign → ratio
                if sign(beta_coef) == sign(K_coef)
                    println("  → (β$i·K$i) product")
                else
                    ratio_str = beta_coef > 0 ? "β$i/K$i" : "K$i/β$i"
                    println("  → ($ratio_str) ratio")
                end

                # Compute the product and ratio values (compare MLE vs true)
                K_val_MLE = exp(θ_log_MLE[K_indices[i]])
                beta_val_MLE = exp(θ_log_MLE[beta_indices[i]])
                K_val_true = exp(θ_log_true[K_indices[i]])
                beta_val_true = exp(θ_log_true[beta_indices[i]])

                product_val_MLE = beta_val_MLE * K_val_MLE
                ratio_val_MLE = K_val_MLE / beta_val_MLE
                product_val_true = beta_val_true * K_val_true
                ratio_val_true = K_val_true / beta_val_true

                println("    MLE:  K$i = $(round(K_val_MLE, digits=4)), β$i = $(round(beta_val_MLE, digits=4))")
                println("    True: K$i = $(round(K_val_true, digits=4)), β$i = $(round(beta_val_true, digits=4))")
                println("    β$i·K$i (MLE) = $(round(product_val_MLE, digits=2)) (in invariant null space)")
                println("    K$i/β$i (MLE) = $(round(ratio_val_MLE, digits=2)) (in complement, identifiable)")
                println("    K$i/β$i (true) = $(round(ratio_val_true, digits=2))")
                println("    Expected K$i/β$i per Eisenberg: $(round([346.32, 130.43, 633.71][i], digits=2))")
            end
        end
    end
else
    println("\n⚠ WARNING: No invariant null space found!")
    println("This suggests the model doesn't have IIR-compatible invariant structure.")
end

# Profile K₁ (parameter index 10)
println("\n1. Profiling K₁ (parameter 10, non-identifiable)...")
K1_index = 10
n_params = 18
nuisance_indices_K1 = setdiff(1:n_params, K1_index)
nuisance_guess_K1 = θ_log_MLE[nuisance_indices_K1]

# Generate multiple initial guesses
n_guesses = CONFIG.n_guesses
nuisance_extras_K1 = generate_initial_guesses(
    θ_log_lower[nuisance_indices_K1],
    θ_log_upper[nuisance_indices_K1],
    n_guesses)

ψK1_values, lnlike_K1_values = profile_target(
    lnlike_θ_log, K1_index,
    θ_log_lower, θ_log_upper,
    nuisance_guess_K1;
    grid_steps=[CONFIG.grid_1d],
    ω_initial_extras=nuisance_extras_K1,
    method=:LN_BOBYQA,
    optmaxtime=CONFIG.timeout)

K1_profile_vals = [ψ[K1_index] for ψ in ψK1_values]
println("  Profiled K₁ range: [$(round(exp(minimum(K1_profile_vals)), digits=2)), $(round(exp(maximum(K1_profile_vals)), digits=2))]")

# Prediction intervals from K₁ profile
lower_K1, upper_K1, _ = construct_upper_lower_profile_wise_CIs_for_mean(
    distrib_fine_θ_log, ψK1_values, lnlike_K1_values; l_level=95, df=18)

# Profile β₁ (parameter index 7)
println("\n2. Profiling β₁ (parameter 7, non-identifiable)...")
β1_index = 7
nuisance_indices_β1 = setdiff(1:n_params, β1_index)
nuisance_guess_β1 = θ_log_MLE[nuisance_indices_β1]

nuisance_extras_β1 = generate_initial_guesses(
    θ_log_lower[nuisance_indices_β1],
    θ_log_upper[nuisance_indices_β1],
    n_guesses)

ψβ1_values, lnlike_β1_values = profile_target(
    lnlike_θ_log, β1_index,
    θ_log_lower, θ_log_upper,
    nuisance_guess_β1;
    grid_steps=[CONFIG.grid_1d],
    ω_initial_extras=nuisance_extras_β1,
    method=:LN_BOBYQA,
    optmaxtime=CONFIG.timeout)

β1_profile_vals = [ψ[β1_index] for ψ in ψβ1_values]
println("  Profiled β₁ range: [$(round(exp(minimum(β1_profile_vals)), digits=4)), $(round(exp(maximum(β1_profile_vals)), digits=4))]")

# Prediction intervals from β₁ profile
lower_β1, upper_β1, _ = construct_upper_lower_profile_wise_CIs_for_mean(
    distrib_fine_θ_log, ψβ1_values, lnlike_β1_values; l_level=95, df=18)

# Profile K₁/β₁ ratio (conditional on CONFIG.do_2d)
if CONFIG.do_2d
    println("\n3. Profiling K₁/β₁ ratio (identifiable)...")
    # Need to parameterize as: ψ₁ = log(K₁/β₁), with K₁ = exp(ψ₁) * β₁
    # This requires constrained profiling - implement simplified version

    # For now, profile the ratio by fixing the product β₁*K₁ and varying the ratio
    # Transformation: [K₁, β₁] → [K₁/β₁, β₁*K₁]
    # In log space: [log K₁, log β₁] → [log K₁ - log β₁, log K₁ + log β₁]

    println("  Using 2D joint profile of (β₁, K₁) to extract ratio...")
    target_indices_K1β1 = [β1_index, K1_index]  # [7, 10]
    nuisance_indices_ratio = setdiff(1:n_params, target_indices_K1β1)
    nuisance_guess_ratio = θ_log_MLE[nuisance_indices_ratio]

    nuisance_extras_ratio = generate_initial_guesses(
        θ_log_lower[nuisance_indices_ratio],
        θ_log_upper[nuisance_indices_ratio],
        n_guesses)

    ψK1β1_values, lnlike_K1β1_values = profile_target(
        lnlike_θ_log, target_indices_K1β1,
        θ_log_lower, θ_log_upper,
        nuisance_guess_ratio;
        grid_steps=CONFIG.grid_2d,
        ω_initial_extras=nuisance_extras_ratio,
        method=:LN_BOBYQA,
        optmaxtime=CONFIG.timeout)

    # Extract ratio values: log(K₁/β₁) = log(K₁) - log(β₁)
    ratio_values = [ψ[K1_index] - ψ[β1_index] for ψ in ψK1β1_values]
    println("  Profiled ratio range: [$(round(exp(minimum(ratio_values)), digits=2)), $(round(exp(maximum(ratio_values)), digits=2))]")
    println("  True ratio: $(round(θ_true[K1_index]/θ_true[β1_index], digits=2))")

    # Prediction intervals from joint (K₁,β₁) profile
    lower_ratio, upper_ratio, _ = construct_upper_lower_profile_wise_CIs_for_mean(
        distrib_fine_θ_log, ψK1β1_values, lnlike_K1β1_values; l_level=95, df=18)
else
    println("\n3. Skipping 2D profile (CONFIG.do_2d = false)")
    # Create dummy variables for plotting section
    lower_ratio = lower_K1  # Use K1 as placeholder
    upper_ratio = upper_K1
end

println("\nPrediction interval widths (mean across time/species):")
width_K1 = mean(upper_K1 - lower_K1)
width_β1 = mean(upper_β1 - lower_β1)
width_ratio = mean(upper_ratio - lower_ratio)
println("  K₁ individual:   ", round(width_K1, digits=4))
println("  β₁ individual:   ", round(width_β1, digits=4))
println("  K₁/β₁ ratio:     ", round(width_ratio, digits=4))

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

# Plot m₁ predictions for comparison using library function
ci_intervals = [
    (lower_K1_mat[1,:], upper_K1_mat[1,:], "K₁ individual", :red),
    (lower_β1_mat[1,:], upper_β1_mat[1,:], "β₁ individual", :orange),
    (lower_ratio_mat[1,:], upper_ratio_mat[1,:], "K₁/β₁ ratio (joint)", :blue)
]

plot_profile_wise_CI_comparison(
    t_pred, mle_mat[1,:],
    ci_intervals,
    "repressilator", "m₁ concentration", "Time", "t";
    data_indep=t, data_dep=vec(data_mRNA[1,:]),
    title="Repressilator: Individual vs Ratio Prediction Intervals",
    save_dir=joinpath(@__DIR__, "..", "figures") * "/"
)

println("\n" * repeat("=", 70))
println("Analysis Complete")
println(repeat("=", 70))
