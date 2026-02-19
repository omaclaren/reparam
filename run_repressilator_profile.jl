# Unified Repressilator Profile Likelihood Script
# Supports slice, hybrid (linear path), and full profile modes
#
# Usage:
#   julia run_repressilator_profile.jl --nuisance=0              # Slice at MLE
#   julia run_repressilator_profile.jl --mode=hybrid --grid=100  # Hybrid (linear path approx)
#   julia run_repressilator_profile.jl --nuisance=16 --grid=50   # Full profile
#   julia run_repressilator_profile.jl --nuisance=16 --workers=7 # Parallel
#
# Modes:
#   slice (--nuisance=0): Fix all nuisance at MLE, evaluate likelihood on grid
#   hybrid (--mode=hybrid): Linear path approximation for nuisance (fast, anti-conservative)
#   profile (--nuisance>0): Full optimization over nuisance (slow, accurate)
#
# On NeSI/SLURM, workers auto-detected from SLURM_CPUS_PER_TASK

using Pkg
Pkg.activate(".")
Pkg.instantiate()

# === PARSE ARGUMENTS ===
function parse_int_arg(args, prefix, default)
    for arg in args
        if startswith(arg, prefix)
            return parse(Int, split(arg, "=")[2])
        end
    end
    return default
end

function parse_bool_arg(args, flag)
    return flag in args
end

function parse_string_arg(args, prefix, default)
    for arg in args
        if startswith(arg, prefix)
            return split(arg, "=")[2]
        end
    end
    return default
end

N_NUISANCE = parse_int_arg(ARGS, "--nuisance=", 16)  # Default: full profile
GRID = parse_int_arg(ARGS, "--grid=", 50)
MODE = parse_string_arg(ARGS, "--mode=", "auto")  # auto, slice, hybrid, profile

# Determine actual mode
if MODE == "hybrid"
    USE_HYBRID = true
    N_NUISANCE = 16  # Hybrid handles all nuisance via linear path
elseif MODE == "slice" || N_NUISANCE == 0
    USE_HYBRID = false
    N_NUISANCE = 0
else
    USE_HYBRID = false
end

# Validate nuisance count
if N_NUISANCE < 0 || N_NUISANCE > 16
    error("--nuisance must be between 0 and 16")
end

# === DISTRIBUTED SETUP ===
using Distributed

USE_DISTRIBUTED = N_NUISANCE > 0 && !USE_HYBRID  # Only need distributed for full profiling

if USE_DISTRIBUTED
    if haskey(ENV, "SLURM_CPUS_PER_TASK")
        n_cpus = parse(Int, ENV["SLURM_CPUS_PER_TASK"])
        println("SLURM detected: $n_cpus CPUs")
    else
        n_cpus = parse_int_arg(ARGS, "--workers=", 7) + 1
        println("Local mode: $(n_cpus - 1) workers")
    end

    n_workers = n_cpus - 1
    if nworkers() == 1 && n_workers > 1
        addprocs(n_workers)
    end
    println("Workers ready: $(nworkers())")

    # Load modules on all workers
    const PROJECT_DIR = pwd()

    @everywhere begin
        using Pkg
        Pkg.activate(".")
    end

    @everywhere PROJECT_DIR = $PROJECT_DIR
    @everywhere include(joinpath(PROJECT_DIR, "ReparamTools.jl"))
    @everywhere include(joinpath(PROJECT_DIR, "examples/RepressilatorModel.jl"))

    @everywhere begin
        using .ReparamTools
        using .RepressilatorModel
        using Distributions, LinearAlgebra, Random, ForwardDiff
    end
    println("Modules loaded on all workers")
else
    # Single-threaded for slice or hybrid (no workers added)
    include(joinpath(@__DIR__, "ReparamTools.jl"))
    include(joinpath(@__DIR__, "examples/RepressilatorModel.jl"))
    using .ReparamTools
    using .RepressilatorModel
    using Distributions, LinearAlgebra, Random, ForwardDiff, Statistics
    if USE_HYBRID
        println("Single-threaded mode (hybrid/linear path)")
    else
        println("Single-threaded mode (slice)")
    end
end

# === MODE DESCRIPTION ===
mode_str = if USE_HYBRID
    "HYBRID (linear path approximation)"
elseif N_NUISANCE == 0
    "SLICE (no nuisance, fixed at MLE)"
elseif N_NUISANCE == 16
    "FULL PROFILE (16 nuisance)"
else
    "PARTIAL PROFILE ($N_NUISANCE nuisance profiled, $(16 - N_NUISANCE) fixed at MLE)"
end

println("\n" * "=" ^ 70)
println("REPRESSILATOR PROFILE LIKELIHOOD")
println("=" ^ 70)
println("Mode: $mode_str")
println("Grid: $GRID × $GRID = $(GRID^2) points")
if USE_DISTRIBUTED
    println("Workers: $(nworkers())")
end

# === MODEL SETUP ===
Random.seed!(42)

NT, T_end = 8, 10000.0
t_obs = LinRange(0, T_end, NT)
X0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
σ = 10.0

θ_true = [0.008, 0.009, 0.010,      # α₀ (1-3)
          1.0, 1.2, 1.5,             # α  (4-6)
          0.02, 0.025, 0.015,        # β  (7-9)
          30.0, 28.0, 32.0,          # K  (10-12)
          0.006, 0.0055, 0.0065,     # k_degm (13-15)
          0.0012, 0.0011, 0.0013]    # k_degp (16-18)

param_names = [
    "α₀₁", "α₀₂", "α₀₃", "α₁", "α₂", "α₃",
    "β₁", "β₂", "β₃", "K₁", "K₂", "K₃",
    "k_degm₁", "k_degm₂", "k_degm₃", "k_degp₁", "k_degp₂", "k_degp₃"
]

n_params = 18
β1_idx, K1_idx = 7, 10

# Generate data
y_true = RepressilatorModel.predict_mRNA(θ_true, t_obs, X0)
data = y_true + σ * randn(length(y_true))

# === PARAMETER BOUNDS ===
θ_lower = [0.005, 0.005, 0.005, 0.8, 0.8, 0.8, 0.01, 0.01, 0.01,
           20.0, 20.0, 20.0, 0.004, 0.004, 0.004, 0.001, 0.001, 0.001]
θ_upper = [0.015, 0.015, 0.015, 2.0, 2.0, 2.0, 0.03, 0.03, 0.03,
           40.0, 40.0, 40.0, 0.008, 0.008, 0.008, 0.0015, 0.0015, 0.0015]
θ_log_lower = log.(θ_lower)
θ_log_upper = log.(θ_upper)

# Wider bounds for profiling (to capture full uncertainty region)
θ_lower_profile = [0.003, 0.003, 0.003, 0.5, 0.5, 0.5, 0.002, 0.002, 0.002,
                   3.0, 3.0, 3.0, 0.003, 0.003, 0.003, 0.0008, 0.0008, 0.0008]
θ_upper_profile = [0.020, 0.020, 0.020, 3.0, 3.0, 3.0, 0.5, 0.5, 0.5,
                   100.0, 100.0, 100.0, 0.010, 0.010, 0.010, 0.002, 0.002, 0.002]

# === LIKELIHOOD FUNCTION ===
distrib_θ = θ -> MvNormal(RepressilatorModel.predict_mRNA(θ, t_obs, X0), σ^2 * I(3*NT))
lnlike_θ = ReparamTools.construct_lnlike_xy(distrib_θ, data; dist_type=:multi)
lnlike_θ_log = θ_log -> lnlike_θ(exp.(θ_log))

# === FIND MLE (always needed) ===
println("\n" * "=" ^ 70)
println("FINDING MLE")
println("=" ^ 70)

θ_log_initial = 0.5 * (θ_log_lower + θ_log_upper)
n_mle_guesses = 20  # Slightly more than original 15 for wider bounds
mle_guesses = ReparamTools.generate_initial_guesses(θ_log_lower, θ_log_upper, n_mle_guesses)

println("Running MLE optimization with $n_mle_guesses restarts...")
flush(stdout)

t_mle_start = time()
θ_log_MLE, lnlike_MLE = ReparamTools.profile_target(
    lnlike_θ_log, Int[], θ_log_lower, θ_log_upper, mle_guesses[1];
    grid_steps=Int[], ω_initial_extras=mle_guesses[2:end],
    method=:LN_BOBYQA, optmaxtime=150.0)  # Slightly more than original 120s
t_mle_elapsed = time() - t_mle_start

θ_MLE = exp.(θ_log_MLE)
println("MLE found in $(round(t_mle_elapsed, digits=1)) seconds")
println("Log-likelihood at MLE: $(round(lnlike_MLE, digits=2))")
println("β₁_MLE = $(round(θ_MLE[β1_idx], digits=4)), K₁_MLE = $(round(θ_MLE[K1_idx], digits=2))")

# === IIR ANALYSIS (always needed) ===
println("\n" * "=" ^ 70)
println("RUNNING IIR ANALYSIS")
println("=" ^ 70)

t_iir = LinRange(0, T_end, 501)
function ϕ_iir_highprec(θ)
    sol_matrix = RepressilatorModel.solve_repressilator(t_iir, θ, X0; abstol=1e-10, reltol=1e-8)
    mRNA = sol_matrix[1:3, :]
    return vec(mRNA)
end
ϕ_iir_log(θ_log) = ϕ_iir_highprec(exp.(θ_log))

t_iir_start = time()
S, N, N_perp, rank_J = ReparamTools.find_invariant_subspace(
    ϕ_iir_log, θ_log_MLE; rtol_rank=1e-7, verbose=true)
println("IIR analysis completed in $(round(time() - t_iir_start, digits=1)) seconds")

n_ident = size(N_perp, 2)
n_nonident = size(N, 2)

println("\nRank: $rank_J / $n_params")
println("Identifiable: $n_ident, Non-identifiable: $n_nonident")

if n_nonident == 0
    error("No non-identifiable directions found!")
end

# === BUILD TRANSFORMATION ===
N_perp_varimax = ReparamTools.varimax_rotation(N_perp; n_restarts=200, threshold=1e-2)
N_perp_clean = ReparamTools.scale_and_round(N_perp_varimax; round_within=0.15)

# Fix K/β signs to get K/β (not β/K)
for j in 1:n_ident
    v = N_perp_clean[:, j]
    for gene in 1:3
        β_idx = 6 + gene
        K_idx = 9 + gene
        β_coef = round(Int, v[β_idx])
        K_coef = round(Int, v[K_idx])
        other_sum = sum(abs.(round.(Int, v[[i for i in 1:n_params if i != β_idx && i != K_idx]])))
        if abs(β_coef) == 1 && abs(K_coef) == 1 && β_coef != K_coef && other_sum == 0
            if β_coef == 1 && K_coef == -1
                N_perp_clean[:, j] *= -1
            end
            break
        end
    end
end

N_varimax = ReparamTools.varimax_rotation(N; n_restarts=200, threshold=1e-2)
N_clean = ReparamTools.scale_and_round(N_varimax; round_within=0.15)

A_T_final = hcat(N_perp_clean, N_clean)

θ_to_ψ, ψ_to_θ = ReparamTools.reparam(A_T_final)
ψ_MLE = θ_to_ψ(θ_MLE)

# === FIND GENE 1 COORDINATES ===
println("\nIdentifying gene 1 coordinates...")

gene1_ident_idx = nothing
gene1_nonident_idx = nothing

# Find K₁/β₁ (identifiable)
for j in 1:n_ident
    v = A_T_final[:, j]
    k_rounded = round(Int, v[K1_idx])
    b_rounded = round(Int, v[β1_idx])
    other_sum = sum(abs.(round.(Int, v[[i for i in 1:n_params if i != β1_idx && i != K1_idx]])))
    if k_rounded == 1 && b_rounded == -1 && other_sum == 0
        global gene1_ident_idx = j
        println("  ψ_$j = K₁/β₁ (identifiable)")
        break
    end
end

# Find β₁·K₁ (non-identifiable)
for j in 1:n_nonident
    v = N_clean[:, j]
    k_rounded = round(Int, v[K1_idx])
    b_rounded = round(Int, v[β1_idx])
    other_sum = sum(abs.(round.(Int, v[[i for i in 1:n_params if i != β1_idx && i != K1_idx]])))
    if b_rounded == k_rounded && abs(b_rounded) >= 1 && other_sum == 0
        global gene1_nonident_idx = n_ident + j
        println("  ψ_$(n_ident + j) = β₁·K₁ (non-identifiable)")
        break
    end
end

if isnothing(gene1_ident_idx) || isnothing(gene1_nonident_idx)
    error("Could not identify gene 1 coordinates!")
end

target_2d = [gene1_ident_idx, gene1_nonident_idx]
println("\nTarget: ψ_$(target_2d[1]) (K₁/β₁), ψ_$(target_2d[2]) (β₁·K₁)")

# === COMPUTE ψ-SPACE BOUNDS ===
function compute_ψ_bounds(θ_lo, θ_hi, θ_to_ψ_func, n_samples=10000)
    n = length(θ_lo)
    ψ_samples = [θ_to_ψ_func(θ_lo .+ rand(n) .* (θ_hi - θ_lo)) for _ in 1:n_samples]
    ψ_mat = hcat(ψ_samples...)
    return vec(minimum(ψ_mat, dims=2)), vec(maximum(ψ_mat, dims=2))
end

ψ_lower, ψ_upper = compute_ψ_bounds(θ_lower_profile, θ_upper_profile, θ_to_ψ, 10000)
ψ_log_lower = log.(ψ_lower)
ψ_log_upper = log.(ψ_upper)

# === SELECT NUISANCE PARAMETERS ===
# Order by: other genes' K/β ratios first, then products, then remaining
# This gives a natural ordering for partial profiling

all_other_ψ = setdiff(1:n_params, target_2d)

# For partial nuisance: prioritize gene 2 and 3 K/β ratios (most important)
# then gene 2/3 products, then everything else
# This is a heuristic - the identifiable combos are more "important" to profile

if USE_HYBRID
    # Hybrid: all nuisance handled via linear path (conceptually "profiled")
    nuisance_to_profile = all_other_ψ
    fixed_at_mle = Int[]
elseif N_NUISANCE == 0
    nuisance_to_profile = Int[]
    fixed_at_mle = all_other_ψ
elseif N_NUISANCE >= 16
    nuisance_to_profile = all_other_ψ
    fixed_at_mle = Int[]
else
    # Partial: profile first N_NUISANCE of the others
    nuisance_to_profile = all_other_ψ[1:N_NUISANCE]
    fixed_at_mle = all_other_ψ[N_NUISANCE+1:end]
end

println("\n" * "=" ^ 70)
println("PROFILING SETUP")
println("=" ^ 70)
println("Mode: $mode_str")
println("Target (2D grid): ψ_$(target_2d[1]), ψ_$(target_2d[2])")
println("Profiled nuisance ($(length(nuisance_to_profile))): $(isempty(nuisance_to_profile) ? "none" : "ψ_" * join(nuisance_to_profile, ", ψ_"))")
println("Fixed at MLE ($(length(fixed_at_mle))): $(isempty(fixed_at_mle) ? "none" : "ψ_" * join(fixed_at_mle, ", ψ_"))")

# === BUILD GRID ===
target1_log_grid = range(ψ_log_lower[target_2d[1]], ψ_log_upper[target_2d[1]], length=GRID)
target2_log_grid = range(ψ_log_lower[target_2d[2]], ψ_log_upper[target_2d[2]], length=GRID)
ψ_target1_grid = exp.(collect(target1_log_grid))
ψ_target2_grid = exp.(collect(target2_log_grid))

println("\nGrid ranges:")
println("  K₁/β₁: [$(round(ψ_target1_grid[1], sigdigits=3)), $(round(ψ_target1_grid[end], sigdigits=3))]")
println("  β₁·K₁: [$(round(ψ_target2_grid[1], sigdigits=3)), $(round(ψ_target2_grid[end], sigdigits=3))]")

# === RUN PROFILING ===
println("\n" * "=" ^ 70)
if USE_HYBRID
    println("COMPUTING HYBRID PROFILE (linear path approximation)")
elseif N_NUISANCE == 0
    println("COMPUTING SLICE (evaluating likelihood on grid)")
else
    println("COMPUTING PROFILE LIKELIHOOD")
end
println("=" ^ 70)

t_profile_start = time()

# Storage for diagnostics (hybrid mode)
gradient_norms = Float64[]

if USE_HYBRID
    # === HYBRID MODE: Linear path approximation for nuisance ===
    #
    # Algorithm:
    # 1. Compute Hessian H of log-likelihood in ψ-space at MLE
    # 2. Partition into interest (I) and nuisance (N)
    # 3. At each grid point: δψ_N = -H_NN⁺ H_NI (ψ_I - ψ_I_MLE)
    # 4. Evaluate true likelihood at (ψ_I, ψ_N_MLE + δψ_N)
    # 5. Compute gradient diagnostic

    println("Setting up hybrid profiling...")
    println("  Computing Hessian at MLE (in log-ψ space)...")
    flush(stdout)

    # Define log-likelihood in log-ψ space (ensures positivity)
    function lnlike_ψ_log_full(ψ_log)
        try
            ψ = exp.(ψ_log)
            θ = ψ_to_θ(ψ)
            if any(θ .<= 0) || any(!isfinite, θ)
                return -Inf
            end
            return lnlike_θ(θ)
        catch
            return -Inf
        end
    end

    # Work in log-ψ space
    ψ_log_MLE = log.(ψ_MLE)

    # Compute Hessian via ForwardDiff in log-ψ space
    H_full = -ForwardDiff.hessian(lnlike_ψ_log_full, ψ_log_MLE)

    # Symmetrize (numerical safety)
    H_full = 0.5 * (H_full + H_full')

    # Partition indices
    interest_idx = target_2d
    nuisance_idx = setdiff(1:n_params, target_2d)

    # Extract blocks
    H_II = H_full[interest_idx, interest_idx]
    H_IN = H_full[interest_idx, nuisance_idx]
    H_NI = H_full[nuisance_idx, interest_idx]
    H_NN = H_full[nuisance_idx, nuisance_idx]

    # Eigendecompose H_NN to handle non-identifiable directions
    eigen_NN = eigen(Symmetric(H_NN))
    λ_NN = eigen_NN.values
    U_NN = eigen_NN.vectors

    # Check PSD: H = -∇²ℓ should be PSD at a maximum
    n_negative = sum(λ_NN .< -1e-10 * maximum(abs.(λ_NN)))
    if n_negative > 0
        println("  WARNING: $n_negative negative eigenvalues in H_NN (not at local max?)")
        println("    Min eigenvalue: $(minimum(λ_NN))")
    end

    # Use POSITIVE-part eigenvalues only (safer than abs - avoids inverting saddle directions)
    λ_pos = max.(λ_NN, 0.0)
    λ_max = maximum(λ_pos)
    rtol_eig = 1e-8
    ident_mask = λ_pos .> rtol_eig * λ_max
    n_ident_nuisance = sum(ident_mask)
    n_flat_nuisance = sum(.!ident_mask)

    println("  H_NN eigenvalue spectrum:")
    println("    Identifiable (positive curvature): $n_ident_nuisance directions")
    println("    Flat/saddle (non-identifiable): $n_flat_nuisance directions")
    if n_flat_nuisance > 0 && any(λ_pos[ident_mask] .> 0)
        gap = minimum(λ_pos[ident_mask]) / max(maximum(λ_pos[.!ident_mask]), 1e-15)
        println("    Spectral gap: $(round(gap, sigdigits=2))×")
    end

    # Compute pseudoinverse using only identifiable (positive curvature) directions
    # H_NN⁺ = U_r * Λ_r⁻¹ * U_rᵀ
    U_r = U_NN[:, ident_mask]
    Λ_r = λ_pos[ident_mask]  # Strictly positive
    H_NN_pinv = U_r * Diagonal(1.0 ./ Λ_r) * U_r'

    # Precompute path matrix (explicit projection form for clarity)
    # path_matrix = -H_NN⁺ H_NI = -U_r Λ_r⁻¹ (U_rᵀ H_NI)
    path_matrix = -(U_r * Diagonal(1.0 ./ Λ_r) * (U_r' * H_NI))

    # For gradient diagnostic: curvature-scaled projected score
    Λ_r_sqrt_inv = Diagonal(1.0 ./ sqrt.(Λ_r))

    println("\nEvaluating $(GRID^2) grid points with linear path approximation...")
    flush(stdout)

    # Pre-allocate
    ll_vals = zeros(GRID^2)
    ψ_vals = zeros(2, GRID^2)
    gradient_norms = zeros(GRID^2)

    # Reference values in log-ψ space
    ψ_log_I_MLE = ψ_log_MLE[interest_idx]
    ψ_log_N_MLE = ψ_log_MLE[nuisance_idx]

    # Gradient function for diagnostics (in log-ψ space)
    function grad_N_lnlike_log(ψ_log_full)
        try
            g = ForwardDiff.gradient(lnlike_ψ_log_full, ψ_log_full)
            return g[nuisance_idx]
        catch
            return fill(NaN, length(nuisance_idx))
        end
    end

    # Loop in column-major order (target1 fast, target2 slow) - matches profile_target output
    # No snake ordering needed since we're not optimizing (just evaluating likelihood)
    local k = 0
    local n_debug = 3  # Debug first few points
    for (j, ψ2) in enumerate(ψ_target2_grid)   # j = target2 index (outer/slow)
        for (i, ψ1) in enumerate(ψ_target1_grid)  # i = target1 index (inner/fast)
            k += 1

            # Interest parameter deviation in LOG-ψ space
            ψ_log_I = [log(ψ1), log(ψ2)]
            δψ_log_I = ψ_log_I - ψ_log_I_MLE

            # Linear path approximation for nuisance in LOG-ψ space
            δψ_log_N = path_matrix * δψ_log_I
            ψ_log_N = ψ_log_N_MLE + δψ_log_N

            # Build full log-ψ vector
            ψ_log_full_k = copy(ψ_log_MLE)
            ψ_log_full_k[interest_idx] = ψ_log_I
            ψ_log_full_k[nuisance_idx] = ψ_log_N

            # Debug output for first few points
            if k <= n_debug
                ψ_full_k = exp.(ψ_log_full_k)
                println("\n  Debug point $k:")
                println("    ψ_I = $([ψ1, ψ2])")
                println("    δψ_log_I = $δψ_log_I")
                println("    ||δψ_log_N|| = $(norm(δψ_log_N))")
                println("    min(ψ) = $(minimum(ψ_full_k)), max(ψ) = $(maximum(ψ_full_k))")
                θ_test = ψ_to_θ(ψ_full_k)
                println("    min(θ) = $(minimum(θ_test)), max(θ) = $(maximum(θ_test))")
            end

            # Evaluate true likelihood (function takes log-ψ)
            ll_vals[k] = lnlike_ψ_log_full(ψ_log_full_k)
            ψ_vals[:, k] = ψ_log_I  # Already in log space

            # Gradient diagnostic: ||g_scaled|| = ||Λ_r^{-1/2} U_rᵀ ∇_N log L||
            if isfinite(ll_vals[k])
                g_N = grad_N_lnlike_log(ψ_log_full_k)
                if all(isfinite.(g_N))
                    g_proj = U_r' * g_N  # Project onto identifiable subspace
                    g_scaled = Λ_r_sqrt_inv * g_proj
                    gradient_norms[k] = norm(g_scaled)
                else
                    gradient_norms[k] = NaN
                end
            else
                gradient_norms[k] = NaN
            end
        end
        # Progress (after each target2 column)
        if j % max(1, GRID ÷ 10) == 0
            println("  Column $j/$GRID complete")
            flush(stdout)
        end
    end

    # Convert to match expected format
    ψ_vals = ψ_vals'  # Transpose to N×2

    # Report gradient diagnostics
    # The diagnostic is ||Λ_r^{-1/2} U_rᵀ ∇_N ℓ|| = curvature-scaled projected nuisance score
    # Interpretation: size of implied one-step Newton correction in identifiable nuisance subspace
    # Values << 1: near the nuisance optimum (linear path approximation valid)
    # Values >> 1: far from nuisance optimum (approximation breaking down)
    valid_grads = gradient_norms[isfinite.(gradient_norms)]
    println("\nProjected nuisance score diagnostic (||Λ_r^{-1/2} U_rᵀ ∇_N ℓ||):")
    println("  (Measures distance from nuisance optimum in curvature-scaled units)")
    if isempty(valid_grads)
        println("  WARNING: All gradient computations failed (NaN)")
        println("  This may indicate issues with autodiff or parameter bounds")
    else
        println("  Valid points: $(length(valid_grads))/$(length(gradient_norms))")
        println("  Median: $(round(median(valid_grads), sigdigits=3))")
        println("  Max: $(round(maximum(valid_grads), sigdigits=3))")
        n_good = sum(valid_grads .< 1.0)
        println("  Points with diagnostic < 1 (good approximation): $n_good/$(length(valid_grads))")
        if n_good < length(valid_grads)
            println("  Points with diagnostic > 10 (poor approximation): $(sum(valid_grads .> 10))/$(length(valid_grads))")
        end
    end

elseif N_NUISANCE == 0
    # SLICE: Use profile_target with empty nuisance (matches working minimal_2D_IIR_coords.jl)
    # This ensures correct grid ordering for reshape
    println("Evaluating $(GRID^2) grid points via profile_target...")
    flush(stdout)

    # Build likelihood that fixes other parameters at MLE
    function lnlike_slice_ψ_log(ψ_log_targets)
        try
            ψ_full = copy(ψ_MLE)
            ψ_full[target_2d[1]] = exp(ψ_log_targets[1])
            ψ_full[target_2d[2]] = exp(ψ_log_targets[2])
            θ = ψ_to_θ(ψ_full)
            if any(θ .<= 0) || any(!isfinite, θ)
                return -Inf
            end
            return lnlike_θ(θ)
        catch
            return -Inf
        end
    end

    # Bounds for just the two targets
    ψ_log_lower_targets = [ψ_log_lower[target_2d[1]], ψ_log_lower[target_2d[2]]]
    ψ_log_upper_targets = [ψ_log_upper[target_2d[1]], ψ_log_upper[target_2d[2]]]

    # Use profile_target with empty nuisance array (Float64[])
    # This is the pattern from working minimal_2D_IIR_coords.jl
    ψ_vals, ll_vals = ReparamTools.profile_target(
        lnlike_slice_ψ_log, [1, 2], ψ_log_lower_targets, ψ_log_upper_targets, Float64[];
        grid_steps=GRID, use_distributed=false)
else
    # PROFILE: Optimize over nuisance parameters

    # Broadcast to workers if distributed
    if USE_DISTRIBUTED
        @everywhere A_T_global = $A_T_final
        @everywhere data_global = $data
        @everywhere t_obs_global = $(collect(t_obs))
        @everywhere X0_global = $X0
        @everywhere σ_global = $σ
        @everywhere NT_global = $NT
        @everywhere ψ_MLE_global = $ψ_MLE
        @everywhere fixed_at_mle_global = $fixed_at_mle
        @everywhere target_2d_global = $target_2d
        @everywhere nuisance_to_profile_global = $nuisance_to_profile

        @everywhere function lnlike_ψ_log_worker(ψ_log_partial)
            # ψ_log_partial contains: [target1, target2, nuisance_to_profile...]
            # Need to build full ψ vector
            try
                ψ_full = copy(ψ_MLE_global)

                # Set targets
                ψ_full[target_2d_global[1]] = exp(ψ_log_partial[1])
                ψ_full[target_2d_global[2]] = exp(ψ_log_partial[2])

                # Set profiled nuisance
                for (k, idx) in enumerate(nuisance_to_profile_global)
                    ψ_full[idx] = exp(ψ_log_partial[2 + k])
                end

                # fixed_at_mle stays at ψ_MLE values

                θ = exp.(A_T_global' \ log.(ψ_full))
                if any(θ .<= 0) || any(!isfinite, θ)
                    return -Inf
                end
                pred = RepressilatorModel.predict_mRNA(θ, t_obs_global, X0_global)
                if any(!isfinite, pred)
                    return -Inf
                end
                dist = MvNormal(pred, σ_global^2 * I(3*NT_global))
                return logpdf(dist, data_global)
            catch
                return -Inf
            end
        end
    else
        # Non-distributed version
        function lnlike_ψ_log_local(ψ_log_partial)
            try
                ψ_full = copy(ψ_MLE)
                ψ_full[target_2d[1]] = exp(ψ_log_partial[1])
                ψ_full[target_2d[2]] = exp(ψ_log_partial[2])
                for (k, idx) in enumerate(nuisance_to_profile)
                    ψ_full[idx] = exp(ψ_log_partial[2 + k])
                end
                θ = ψ_to_θ(ψ_full)
                if any(θ .<= 0) || any(!isfinite, θ)
                    return -Inf
                end
                return lnlike_θ(θ)
            catch
                return -Inf
            end
        end
    end

    # Build bounds for optimization
    # Order: [target1, target2, nuisance...]
    opt_indices = vcat(target_2d, nuisance_to_profile)
    ψ_log_lower_opt = ψ_log_lower[opt_indices]
    ψ_log_upper_opt = ψ_log_upper[opt_indices]

    # Nuisance indices within the optimization vector (positions 3 onwards)
    nuisance_opt_indices = collect(3:(2 + length(nuisance_to_profile)))

    # Initial guess for nuisance: MLE values
    nuisance_log_lower = ψ_log_lower_opt[nuisance_opt_indices]
    nuisance_log_upper = ψ_log_upper_opt[nuisance_opt_indices]
    nuisance_log_guess = log.(ψ_MLE[nuisance_to_profile])
    nuisance_log_guess = clamp.(nuisance_log_guess, nuisance_log_lower .+ 1e-6, nuisance_log_upper .- 1e-6)

    # Extra starting points (snake_direction=:row handles warm-starting effectively)
    n_extra_guesses = N_NUISANCE <= 4 ? 10 : 15
    nuisance_extras = ReparamTools.generate_initial_guesses(nuisance_log_lower, nuisance_log_upper, n_extra_guesses)

    # Fewer chunks = better warm-starting continuity, but need enough for parallelism
    # 36 chunks balances warm-starting vs utilizing available workers
    n_chunks_profile = 36

    println("Grid: $GRID × $GRID = $(GRID^2) points")
    println("Nuisance to optimize: $(length(nuisance_to_profile))")
    println("Extra restarts: $n_extra_guesses")
    println("Chunks: $n_chunks_profile")
    if USE_DISTRIBUTED
        println("Workers: $(nworkers())")
    end
    println("\nStarting profiling...")
    flush(stdout)

    lnlike_func = USE_DISTRIBUTED ? lnlike_ψ_log_worker : lnlike_ψ_log_local

    ψ_vals, ll_vals = ReparamTools.profile_target(
        lnlike_func, [1, 2], ψ_log_lower_opt, ψ_log_upper_opt, nuisance_log_guess;
        grid_steps=GRID, use_distributed=USE_DISTRIBUTED,
        ω_initial_extras=nuisance_extras,
        method=:LN_BOBYQA, optmaxtime= N_NUISANCE <= 4 ? 60.0 : 150.0,
        n_chunks=n_chunks_profile,
        snake_direction=:row  # ψ₂ (non-identifiable) varies faster - smoother nuisance landscape
    )
end

t_profile_elapsed = time() - t_profile_start
println("\nCompleted in $(round(t_profile_elapsed/60, digits=1)) minutes")
println("Finite values: $(sum(isfinite.(ll_vals)))/$(length(ll_vals))")

# === SAVE RESULTS ===
using Serialization

output_base = if USE_HYBRID
    "repressilator_hybrid_$(GRID)x$(GRID)"
else
    "repressilator_$(N_NUISANCE)nuisance_$(GRID)x$(GRID)"
end

results = Dict(
    "ψ_vals" => ψ_vals,
    "ll_vals" => ll_vals,
    "ψ_MLE" => ψ_MLE,
    "θ_MLE" => θ_MLE,
    "A_T_final" => A_T_final,
    "target_2d" => target_2d,
    "nuisance_to_profile" => nuisance_to_profile,
    "fixed_at_mle" => fixed_at_mle,
    "GRID" => GRID,
    "N_NUISANCE" => N_NUISANCE,
    "ψ_lower" => ψ_lower,
    "ψ_upper" => ψ_upper,
    "param_names" => param_names,
    "rank_J" => rank_J,
    "n_ident" => n_ident,
    "n_nonident" => n_nonident,
    "mode" => mode_str,
    "gradient_norms" => USE_HYBRID ? gradient_norms : Float64[]
)

serialize("$(output_base)_results.jls", results)
println("\nResults saved to $(output_base)_results.jls")

# === ANALYSIS ===
ll_matrix = reshape(ll_vals, GRID, GRID)
ll_max = maximum(ll_matrix[isfinite.(ll_matrix)])
like_matrix = exp.(ll_matrix .- ll_max)

lstar_1d = exp(-quantile(Chisq(1), 0.95)/2)
lstar_2d = exp(-quantile(Chisq(2), 0.95)/2)

like_ψ_target1 = [maximum(like_matrix[i, :]) for i in 1:GRID]
like_ψ_target2 = [maximum(like_matrix[:, j]) for j in 1:GRID]

n_above_1 = sum(like_ψ_target1 .> lstar_1d)
n_above_2 = sum(like_ψ_target2 .> lstar_1d)

println("\n" * "=" ^ 70)
println("RESULTS SUMMARY")
println("=" ^ 70)
println("K₁/β₁ (identifiable): $n_above_1/$GRID above 95% threshold")
println("β₁·K₁ (non-identifiable): $n_above_2/$GRID above 95% threshold")

println("\n" * "=" ^ 70)
println("COMPLETE")
println("=" ^ 70)
println("""
Summary:
  Mode: $mode_str
  Grid: $GRID × $GRID
  K₁/β₁ (identifiable): $n_above_1/$GRID above threshold (peaked profile expected)
  β₁·K₁ (non-identifiable): $n_above_2/$GRID above threshold (flat profile expected)

Output: $(output_base)_results.jls

To generate plots, run:
  julia replot_profile_results.jl $(output_base)_results.jls
""")
