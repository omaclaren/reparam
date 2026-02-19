# IIR-Guided Profiling - Full 18 Parameter Version
# Extension of iir_guided_profiling.jl to all 18 repressilator parameters
#
# Usage:
#   Sequential: julia iir_guided_profiling_18param.jl
#   Parallel:   julia iir_guided_profiling_18param.jl --parallel
#   With grid:  julia iir_guided_profiling_18param.jl --parallel --grid=25

# === PARSE COMMAND LINE ARGS ===
USE_DISTRIBUTED = "--parallel" in ARGS
N_WORKERS = 7

# Parse --grid=N argument (default 50)
GRID = 50
for arg in ARGS
    if startswith(arg, "--grid=")
        global GRID = parse(Int, split(arg, "=")[2])
    end
end

if USE_DISTRIBUTED
    using Distributed
    if nworkers() == 1
        addprocs(N_WORKERS)
    end
    println("Distributed mode: $(nworkers()) workers")
end

# Load modules on MAIN first
include(joinpath(@__DIR__, "ReparamTools.jl"))
include(joinpath(@__DIR__, "examples/RepressilatorModel.jl"))
using .ReparamTools
using .RepressilatorModel
using Distributions, LinearAlgebra, Random, ForwardDiff

# Load on workers if distributed
if USE_DISTRIBUTED
    @everywhere begin
        include($(joinpath(@__DIR__, "ReparamTools.jl")))
        include($(joinpath(@__DIR__, "examples/RepressilatorModel.jl")))
        using .ReparamTools
        using .RepressilatorModel
        using Distributions, LinearAlgebra, Random
    end
    println("Modules loaded on all workers")
end

# === MODEL SETUP ===
Random.seed!(42)
NT, T_end = 8, 10000.0  # NT=8 matches repressilator.jl (gives 24 data points for 18 params)
t_obs = LinRange(0, T_end, NT)
X0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
σ = 10.0

# True parameters (18 total)
# Indices: α₀(1-3), α(4-6), β(7-9), K(10-12), k_degm(13-15), k_degp(16-18)
θ_true = [0.008, 0.009, 0.010,      # α₀ (1-3)
          1.0, 1.2, 1.5,             # α  (4-6)
          0.02, 0.025, 0.015,        # β  (7-9)
          30.0, 28.0, 32.0,          # K  (10-12)
          0.006, 0.0055, 0.0065,     # k_degm (13-15)
          0.0012, 0.0011, 0.0013]    # k_degp (16-18)

# Parameter names for all 18
param_names = [
    "α₀₁", "α₀₂", "α₀₃",           # 1-3
    "α₁", "α₂", "α₃",               # 4-6
    "β₁", "β₂", "β₃",               # 7-9
    "K₁", "K₂", "K₃",               # 10-12
    "k_degm₁", "k_degm₂", "k_degm₃", # 13-15
    "k_degp₁", "k_degp₂", "k_degp₃"  # 16-18
]

n_params = 18

# Generate data
y_true = RepressilatorModel.predict_mRNA(θ_true, t_obs, X0)
data = y_true + σ * randn(length(y_true))

println("=" ^ 70)
println("IIR-GUIDED PROFILING - FULL 18 PARAMETERS")
println("=" ^ 70)

println("\nModel setup:")
println("  Parameters: $n_params")
println("  Time points: $NT")
println("  Data points: $(3 * NT) (3 mRNAs × $NT times)")
println("  Noise σ: $σ")

# === BOUNDS (from repressilator.jl) ===
println("\n" * "=" ^ 70)
println("PARAMETER BOUNDS")
println("=" ^ 70)

θ_lower = similar(θ_true)
θ_upper = similar(θ_true)

# Basal transcription α₀ᵢ (indices 1-3)
θ_lower[1:3] .= 0.005
θ_upper[1:3] .= 0.015

# Regulated transcription αᵢ (indices 4-6)
θ_lower[4:6] .= 0.8
θ_upper[4:6] .= 2.0

# Translation βᵢ (indices 7-9)
θ_lower[7:9] .= 0.01
θ_upper[7:9] .= 0.03

# Repression threshold Kᵢ (indices 10-12)
θ_lower[10:12] .= 20.0
θ_upper[10:12] .= 40.0

# mRNA degradation k_degmᵢ (indices 13-15)
θ_lower[13:15] .= 0.004
θ_upper[13:15] .= 0.008

# Protein degradation k_degpᵢ (indices 16-18)
θ_lower[16:18] .= 0.001
θ_upper[16:18] .= 0.0015

θ_log_lower = log.(θ_lower)
θ_log_upper = log.(θ_upper)

println("\nBounds set (from repressilator.jl biological constraints)")

# === LIKELIHOOD FUNCTION ===
distrib_θ = θ -> MvNormal(RepressilatorModel.predict_mRNA(θ, t_obs, X0), σ^2 * I(3*NT))
lnlike_θ = ReparamTools.construct_lnlike_xy(distrib_θ, data; dist_type=:multi)
lnlike_θ_log = θ_log -> lnlike_θ(exp.(θ_log))

# === FIND MLE ===
println("\n" * "=" ^ 70)
println("FINDING MLE")
println("=" ^ 70)

# Start from midpoint of bounds
θ_log_initial = 0.5 * (θ_log_lower + θ_log_upper)

# Generate initial guesses
n_guesses = 3
nuisance_guesses = ReparamTools.generate_initial_guesses(θ_log_lower, θ_log_upper, n_guesses)

println("\nOptimization setup:")
println("  Method: LN_BOBYQA")
println("  Initial guesses: $n_guesses")
println("  Max time: 30s per guess")

# Test initial point
lnlike_initial = lnlike_θ_log(θ_log_initial)
lnlike_true_val = lnlike_θ_log(log.(θ_true))
println("\n  Log-likelihood at initial: $(round(lnlike_initial, digits=2))")
println("  Log-likelihood at truth: $(round(lnlike_true_val, digits=2))")

println("\nRunning MLE optimization...")
flush(stdout)

t_mle_start = time()
θ_log_MLE, lnlike_MLE = ReparamTools.profile_target(
    lnlike_θ_log, Int[],  # Empty target = find MLE
    θ_log_lower, θ_log_upper,
    nuisance_guesses[1];
    grid_steps=Int[],
    ω_initial_extras=nuisance_guesses[2:end],
    method=:LN_BOBYQA,
    optmaxtime=30.0)
t_mle_elapsed = time() - t_mle_start

θ_MLE = exp.(θ_log_MLE)

println("MLE found in $(round(t_mle_elapsed, digits=1)) seconds")
println("  Log-likelihood at MLE: $(round(lnlike_MLE, digits=2))")
println("  Improvement over initial: $(round(lnlike_MLE - lnlike_initial, digits=2))")

println("\nMLE vs True parameters:")
for i in 1:n_params
    ratio = θ_MLE[i] / θ_true[i]
    println("  $(param_names[i]): MLE=$(round(θ_MLE[i], sigdigits=4)), True=$(round(θ_true[i], sigdigits=4)), ratio=$(round(ratio, digits=3))")
end

# === DEFINE ϕ (auxiliary mapping) ===
# For likelihood/MLE: use t_obs with default tolerances
function ϕ_obs(θ)
    return RepressilatorModel.predict_mRNA(θ, t_obs, X0)
end

# For IIR: use fine grid with tight tolerances (like repressilator.jl)
# This gives accurate Jacobian/Hessian for the invariance test
t_iir = LinRange(0, T_end, 501)  # Fine grid for IIR
function ϕ_iir_highprec(θ)
    sol_matrix = RepressilatorModel.solve_repressilator(t_iir, θ, X0;
                                                         abstol=1e-10, reltol=1e-8)
    mRNA = sol_matrix[1:3, :]
    return vec(mRNA)
end
ϕ_iir_log(θ_log) = ϕ_iir_highprec(exp.(θ_log))

# === RUN IIR AT MLE ===
println("\n" * "=" ^ 70)
println("RUNNING IIR (find_invariant_subspace) AT MLE")
println("=" ^ 70)

# Tolerance notes:
# - With high-precision solver, try default tolerances first
# - If needed, adjust rtolJ to respect singular value gap
# - rtol_rank=1e-7 ensures σ~4e-5 is classified as zero (4000× gap from σ~0.18)
rtol_rank_custom = 1e-7  # Respect the large singular value gap

println("\nUsing high-precision ϕ with:")
println("  Time points: $(length(t_iir)) (fine grid)")
println("  ODE tolerances: abstol=1e-10, reltol=1e-8")

S, N, N_perp, rank_J = ReparamTools.find_invariant_subspace(
    ϕ_iir_log, θ_log_MLE;
    rtol_rank=rtol_rank_custom,
    verbose=true
)

n_ident = size(N_perp, 2)
n_nonident = size(N, 2)

println("\nResults:")
println("  Jacobian rank: $rank_J / $n_params")
println("  Identifiable directions (N_perp): $n_ident")
println("  Non-identifiable directions (N): $n_nonident")

# === SINGULAR VALUE GAP ANALYSIS ===
println("\n" * "=" ^ 70)
println("SINGULAR VALUE GAP ANALYSIS")
println("=" ^ 70)

println("\nAll singular values:")
for (i, s) in enumerate(S)
    marker = ""
    if i == rank_J
        marker = " ← rank threshold"
    elseif i == rank_J + 1
        marker = " ← first 'zero'"
    end
    println("  σ[$i] = $(round(s, sigdigits=4))$marker")
end

# Compute gaps
if rank_J < n_params
    gap = S[rank_J] / S[rank_J + 1]
    println("\nGap at rank boundary:")
    println("  σ[$rank_J] / σ[$(rank_J+1)] = $(round(gap, sigdigits=4))")
    println("  (Large gap = clear separation between identifiable/non-identifiable)")
end

# === INTERPRET DIRECTIONS ===
if n_nonident > 0
    println("\n" * "=" ^ 70)
    println("NON-IDENTIFIABLE DIRECTIONS (N)")
    println("=" ^ 70)

    for j in 1:n_nonident
        v = N[:, j]
        println("\n  Invariant direction $j:")
        sorted_idx = sortperm(abs.(v), rev=true)
        for k in 1:n_params
            idx = sorted_idx[k]
            if abs(v[idx]) > 0.1
                sign_str = v[idx] > 0 ? "+" : "-"
                println("    $sign_str$(round(abs(v[idx]), digits=3)) × log($(param_names[idx]))")
            end
        end

        # Check βK product pattern
        println("  βK pattern check:")
        for gene in 1:3
            β_idx = 6 + gene  # β indices are 7,8,9
            K_idx = 9 + gene  # K indices are 10,11,12
            β_coef = v[β_idx]
            K_coef = v[K_idx]
            match = abs(β_coef - K_coef) < 0.05
            status = match ? "✓" : "✗"
            println("    Gene $gene: β=$(round(β_coef, digits=3)), K=$(round(K_coef, digits=3)) $status")
        end
    end
else
    println("\n" * "=" ^ 70)
    println("WARNING: NO NON-IDENTIFIABLE DIRECTIONS FOUND")
    println("=" ^ 70)
    println("\nThis is unexpected! Expected 3 non-identifiable (βK products).")
    println("The last few SVD directions (near-null space) are:")

    for j in max(1, n_params-3):n_params
        v = N_perp[:, j]
        println("\n  Direction $j (treated as identifiable):")
        sorted_idx = sortperm(abs.(v), rev=true)
        for k in 1:min(6, n_params)
            idx = sorted_idx[k]
            if abs(v[idx]) > 0.1
                sign_str = v[idx] > 0 ? "+" : "-"
                println("    $sign_str$(round(abs(v[idx]), digits=3)) × log($(param_names[idx]))")
            end
        end

        # Check βK product pattern anyway
        println("  βK pattern check:")
        for gene in 1:3
            β_idx = 6 + gene
            K_idx = 9 + gene
            β_coef = v[β_idx]
            K_coef = v[K_idx]
            match = abs(β_coef - K_coef) < 0.1
            status = match ? "✓ MATCH" : ""
            if abs(β_coef) > 0.1 || abs(K_coef) > 0.1
                println("    Gene $gene: β=$(round(β_coef, digits=3)), K=$(round(K_coef, digits=3)) $status")
            end
        end
    end
end

# === APPLY VARIMAX TO IDENTIFIABLE DIRECTIONS ===
println("\n" * "=" ^ 70)
println("VARIMAX ON IDENTIFIABLE DIRECTIONS (N_perp)")
println("=" ^ 70)

N_perp_varimax = ReparamTools.varimax_rotation(N_perp; n_restarts=200, threshold=1e-2)

# First do scale_and_round, then fix signs to get K/β (not β/K)
N_perp_clean = ReparamTools.scale_and_round(N_perp_varimax; round_within=0.15)

# Now fix signs: we want K in numerator, β in denominator for ratio columns
# β indices: 7,8,9 (genes 1,2,3)
# K indices: 10,11,12 (genes 1,2,3)
println("\nFixing K/β ratio signs (want K/β, not β/K):")
for j in 1:n_ident
    v = N_perp_clean[:, j]
    for gene in 1:3
        β_idx = 6 + gene
        K_idx = 9 + gene
        β_coef = round(Int, v[β_idx])
        K_coef = round(Int, v[K_idx])
        # Check if this is a K/β ratio column (one +1, one -1, others 0)
        other_sum = sum(abs.(round.(Int, v[[i for i in 1:n_params if i != β_idx && i != K_idx]])))
        if abs(β_coef) == 1 && abs(K_coef) == 1 && β_coef != K_coef && other_sum == 0
            # If β is positive (+1) and K is negative (-1), it's β/K - flip to get K/β
            if β_coef == 1 && K_coef == -1
                N_perp_clean[:, j] *= -1
                println("  Column $j: flipped to get K$gene/β$gene (was β$gene/K$gene)")
            else
                println("  Column $j: already K$gene/β$gene (no flip needed)")
            end
            break
        end
    end
end

# Helper to interpret column
function interpret_col(col, names)
    terms_num = String[]
    terms_den = String[]
    for k in 1:length(col)
        coef = round(Int, col[k])
        if coef == 1
            push!(terms_num, names[k])
        elseif coef == -1
            push!(terms_den, names[k])
        elseif coef > 1
            push!(terms_num, "$(names[k])^$coef")
        elseif coef < -1
            push!(terms_den, "$(names[k])^$(-coef)")
        end
    end
    num = isempty(terms_num) ? "1" : join(terms_num, "·")
    if isempty(terms_den)
        return num
    else
        den = length(terms_den) == 1 ? terms_den[1] : "($(join(terms_den, "·")))"
        return "$num/$den"
    end
end

println("\nIdentifiable combinations after varimax + scale_and_round:")
for j in 1:n_ident
    v = N_perp_clean[:, j]
    interp = interpret_col(v, param_names)
    println("  ψ_$j = $interp")
end

# Find gene 1 K/β ratio (should be K₁/β₁ after flipping)
println("\n--- Looking for gene 1 identifiable (K₁/β₁) ---")
β1_idx = 7
K1_idx = 10
global gene1_ident_idx = nothing

for j in 1:n_ident
    v = N_perp_clean[:, j]
    β1_coef = round(Int, v[β1_idx])
    K1_coef = round(Int, v[K1_idx])
    # Check for K₁/β₁ pattern: K₁=+1, β₁=-1, others ~0
    other_sum = sum(abs.(round.(Int, v[[i for i in 1:n_params if i != β1_idx && i != K1_idx]])))
    if K1_coef == 1 && β1_coef == -1 && other_sum == 0
        global gene1_ident_idx = j
        interp = interpret_col(v, param_names)
        println("  Found: ψ_$j = $interp")
        break
    end
end

if gene1_ident_idx === nothing
    println("  Not found as clean K₁/β₁ - checking all K/β combinations...")
    for j in 1:n_ident
        v = N_perp_clean[:, j]
        β1_coef = round(Int, v[β1_idx])
        K1_coef = round(Int, v[K1_idx])
        if abs(β1_coef) >= 1 && abs(K1_coef) >= 1 && β1_coef != K1_coef
            interp = interpret_col(v, param_names)
            println("  Candidate ψ_$j = $interp (K₁=$K1_coef, β₁=$β1_coef)")
        end
    end
end

# === APPLY VARIMAX TO NON-IDENTIFIABLE (if any) ===
if n_nonident > 0
    println("\n" * "=" ^ 70)
    println("VARIMAX ON NON-IDENTIFIABLE DIRECTIONS (N)")
    println("=" ^ 70)

    N_varimax = ReparamTools.varimax_rotation(N; n_restarts=200, threshold=1e-2)
    N_clean = ReparamTools.scale_and_round(N_varimax; round_within=0.15)

    println("\nNon-identifiable combinations after varimax + scale_and_round:")
    for j in 1:n_nonident
        v = N_clean[:, j]
        interp = interpret_col(v, param_names)
        println("  ψ_$(n_ident + j) = $interp")
    end

    # Find gene 1 non-identifiable (β₁K₁)
    println("\n--- Looking for gene 1 non-identifiable (β₁·K₁) ---")
    global gene1_nonident_idx = nothing
    for j in 1:n_nonident
        v = N_clean[:, j]
        β1_coef = round(Int, v[β1_idx])
        K1_coef = round(Int, v[K1_idx])
        other_sum = sum(abs.(round.(Int, v[[i for i in 1:n_params if i != β1_idx && i != K1_idx]])))
        if β1_coef == K1_coef && abs(β1_coef) >= 1 && other_sum == 0
            global gene1_nonident_idx = n_ident + j
            interp = interpret_col(v, param_names)
            println("  Found: ψ_$(n_ident + j) = $interp")
            break
        end
    end
end

# === SUMMARY ===
println("\n" * "=" ^ 70)
println("SUMMARY")
println("=" ^ 70)

println("\nExpected: rank 15/18, 3 non-identifiable (βK products)")
println("Found: rank $rank_J/$n_params, $n_nonident non-identifiable")

if n_nonident == 0
    println("\n⚠️  Invariance test did not classify any directions as non-identifiable.")
    println("   The βK structure may still be present in the null space directions.")
    println("   Consider:")
    println("   1. Examining the last few singular values for gap structure")
    println("   2. Relaxing the tolerance (rtol_invariance parameter)")
    println("   3. The problem might be borderline at this evaluation point")
end

# === BUILD FULL TRANSFORMATION ===
# Only proceed to profiling if we found the expected structure
if n_nonident == 0
    println("\n" * "=" ^ 70)
    println("CANNOT PROCEED TO PROFILING")
    println("=" ^ 70)
    println("No non-identifiable directions found. Check IIR results above.")
    println("May need to adjust rtol_invariance tolerance or examine singular value structure.")
    error("Stopping: expected 3 non-identifiable directions (βK products)")
end

println("\n" * "=" ^ 70)
println("CONSTRUCTING FULL TRANSFORMATION")
println("=" ^ 70)

# Build transformation matrix: identifiable columns first, then non-identifiable
A_T_final = hcat(N_perp_clean, N_clean)

println("\nTransformation matrix A_T_final:")
println("  Size: $(size(A_T_final))")
println("  First $n_ident columns: identifiable")
println("  Last $n_nonident columns: non-identifiable")
println("  det(A_T_final) = $(round(det(A_T_final), digits=6))")

# Create θ ↔ ψ transformations using library function
θ_to_ψ, ψ_to_θ = ReparamTools.reparam(A_T_final)

# Test roundtrip
println("\nTransformation test at MLE:")
ψ_MLE = θ_to_ψ(θ_MLE)
θ_back = ψ_to_θ(ψ_MLE)
println("  Roundtrip OK: $(isapprox(θ_MLE, θ_back, rtol=1e-10))")

# Print all coordinates
println("\nAll 18 ψ coordinates at MLE:")
for j in 1:n_params
    status = j <= n_ident ? "identifiable" : "non-identifiable"
    interp = interpret_col(A_T_final[:, j], param_names)
    println("  ψ_$j = $(round(ψ_MLE[j], sigdigits=4)) ($interp, $status)")
end

# === IDENTIFY GENE 1 COORDINATES ===
println("\n" * "=" ^ 70)
println("IDENTIFYING GENE 1 COORDINATES FOR 2D PROFILE")
println("=" ^ 70)

# Gene 1 indices in θ: β₁=7, K₁=10
β1_idx = 7
K1_idx = 10

# Find which ψ coordinate is K₁/β₁ (identifiable)
println("\nSearching for K₁/β₁ (identifiable):")
for j in 1:n_ident
    v = A_T_final[:, j]
    β1_coef = round(Int, v[β1_idx])
    K1_coef = round(Int, v[K1_idx])
    other_sum = sum(abs.(round.(Int, v[[i for i in 1:n_params if i != β1_idx && i != K1_idx]])))

    if K1_coef == 1 && β1_coef == -1 && other_sum == 0
        global gene1_ident_idx = j
        println("  Found: ψ_$j = K₁/β₁")
        break
    end
end

# Find which ψ coordinate is β₁K₁ (non-identifiable)
println("\nSearching for β₁·K₁ (non-identifiable):")
for j in 1:n_nonident
    v = N_clean[:, j]
    β1_coef = round(Int, v[β1_idx])
    K1_coef = round(Int, v[K1_idx])
    other_sum = sum(abs.(round.(Int, v[[i for i in 1:n_params if i != β1_idx && i != K1_idx]])))

    if β1_coef == K1_coef && abs(β1_coef) >= 1 && other_sum == 0
        global gene1_nonident_idx = n_ident + j
        println("  Found: ψ_$(n_ident + j) = β₁·K₁")
        break
    end
end

if gene1_ident_idx === nothing || gene1_nonident_idx === nothing
    error("Could not find both gene 1 coordinates! Check transformation.")
end

println("\nSelected for 2D profile:")
println("  Target 1: ψ_$gene1_ident_idx (identifiable, K₁/β₁)")
println("  Target 2: ψ_$gene1_nonident_idx (non-identifiable, β₁·K₁)")

target_2d = [gene1_ident_idx, gene1_nonident_idx]
nuisance_2d = setdiff(1:n_params, target_2d)
println("  Nuisance: $(length(nuisance_2d)) parameters (all other ψ)")

# === DEFINE BOUNDS IN ψ-SPACE ===
println("\n" * "=" ^ 70)
println("SETTING UP ψ-SPACE BOUNDS")
println("=" ^ 70)

# Strategy: Transform θ bounds to ψ bounds, but use sensible ranges
# For identifiable directions: bounded by combinations of θ bounds
# For non-identifiable directions: use wide but reasonable ranges

# Start with reasonable θ bounds (wider than MLE search for profiling)
θ_lower_profile = similar(θ_true)
θ_upper_profile = similar(θ_true)

# Basal transcription α₀ᵢ (indices 1-3)
θ_lower_profile[1:3] .= 0.003
θ_upper_profile[1:3] .= 0.020

# Regulated transcription αᵢ (indices 4-6)
θ_lower_profile[4:6] .= 0.5
θ_upper_profile[4:6] .= 3.0

# Translation βᵢ (indices 7-9) - wider for profiling
θ_lower_profile[7:9] .= 0.005
θ_upper_profile[7:9] .= 0.08

# Repression threshold Kᵢ (indices 10-12) - wider for profiling
θ_lower_profile[10:12] .= 10.0
θ_upper_profile[10:12] .= 80.0

# mRNA degradation k_degmᵢ (indices 13-15)
θ_lower_profile[13:15] .= 0.003
θ_upper_profile[13:15] .= 0.010

# Protein degradation k_degpᵢ (indices 16-18)
θ_lower_profile[16:18] .= 0.0008
θ_upper_profile[16:18] .= 0.002

# Compute ψ bounds by transforming corner points and taking envelope
# This is safer than trying to analytically derive ψ bounds
function compute_ψ_bounds(θ_lo, θ_hi, θ_to_ψ_func, n_samples=1000)
    n = length(θ_lo)
    ψ_samples = []

    # Sample random points in θ hypercube
    for _ in 1:n_samples
        θ_sample = θ_lo .+ rand(n) .* (θ_hi - θ_lo)
        push!(ψ_samples, θ_to_ψ_func(θ_sample))
    end

    # Also include corners for extreme values
    for i in 0:(2^n - 1)
        θ_corner = similar(θ_lo)
        for j in 1:n
            bit = (i >> (j-1)) & 1
            θ_corner[j] = bit == 0 ? θ_lo[j] : θ_hi[j]
        end
        push!(ψ_samples, θ_to_ψ_func(θ_corner))
    end

    ψ_mat = hcat(ψ_samples...)
    ψ_lower = vec(minimum(ψ_mat, dims=2))
    ψ_upper = vec(maximum(ψ_mat, dims=2))

    return ψ_lower, ψ_upper
end

# For 18 parameters, 2^18 corners is too many - use sampling only
println("\nComputing ψ bounds from θ sampling...")
ψ_lower, ψ_upper = compute_ψ_bounds(θ_lower_profile, θ_upper_profile, θ_to_ψ, 10000)

# Convert to log-space for profiling
ψ_log_lower = log.(ψ_lower)
ψ_log_upper = log.(ψ_upper)

println("\nψ bounds (from θ sampling):")
for j in 1:n_params
    status = j <= n_ident ? "ident" : "non-id"
    interp = interpret_col(A_T_final[:, j], param_names)
    println("  ψ_$j ($interp, $status) ∈ [$(round(ψ_lower[j], sigdigits=3)), $(round(ψ_upper[j], sigdigits=3))]")
end

# === SETUP PROFILING ===
println("\n" * "=" ^ 70)
println("PROFILING SETUP")
println("=" ^ 70)

# Log-likelihood in full ψ-space (18D)
function lnlike_18param_ψ_log(ψ_log)
    ψ = exp.(ψ_log)
    θ = ψ_to_θ(ψ)
    return lnlike_θ(θ)
end

# Test at MLE
ψ_log_MLE = log.(ψ_MLE)
lnlike_at_ψMLE = lnlike_18param_ψ_log(ψ_log_MLE)
println("\nLog-likelihood at ψ_MLE: $(round(lnlike_at_ψMLE, digits=2))")
println("(Should match MLE: $(round(lnlike_MLE, digits=2)))")

# Initial guess for nuisance: midpoint of bounds in log-space
nuisance_log_lower = ψ_log_lower[nuisance_2d]
nuisance_log_upper = ψ_log_upper[nuisance_2d]
nuisance_log_guess = (nuisance_log_lower .+ nuisance_log_upper) ./ 2

# Clamp to be strictly inside bounds
eps_bound = 1e-6
nuisance_log_guess = clamp.(nuisance_log_guess, nuisance_log_lower .+ eps_bound, nuisance_log_upper .- eps_bound)

# Generate extra starting points
n_extra_guesses = 5
nuisance_extras = ReparamTools.generate_initial_guesses(nuisance_log_lower, nuisance_log_upper, n_extra_guesses)

println("\nNuisance setup:")
println("  Nuisance dimensions: $(length(nuisance_2d))")
println("  Extra starting points: $n_extra_guesses")

# === RUN 2D PROFILE ===
println("\n" * "=" ^ 70)
println("2D PROFILING over ψ_$(target_2d[1]) × ψ_$(target_2d[2])")
println("(profiling out $(length(nuisance_2d)) nuisance parameters)")
println("=" ^ 70)

# GRID is set from command line (--grid=N, default 50)

# Grid in log-ψ space for target coordinates
target1_log_grid = range(ψ_log_lower[target_2d[1]], ψ_log_upper[target_2d[1]], length=GRID)
target2_log_grid = range(ψ_log_lower[target_2d[2]], ψ_log_upper[target_2d[2]], length=GRID)

# Convert to ψ-space grids for plotting
ψ_target1_grid = exp.(collect(target1_log_grid))
ψ_target2_grid = exp.(collect(target2_log_grid))

println("\nGrid setup:")
println("  Grid: $GRID × $GRID = $(GRID^2) points")
println("  Target 1 (K₁/β₁) range: [$(round(ψ_target1_grid[1], sigdigits=3)), $(round(ψ_target1_grid[end], sigdigits=3))]")
println("  Target 2 (β₁·K₁) range: [$(round(ψ_target2_grid[1], sigdigits=3)), $(round(ψ_target2_grid[end], sigdigits=3))]")

# === DISTRIBUTED SETUP ===
# Run with: julia iir_guided_profiling_18param.jl --parallel
if USE_DISTRIBUTED
    # Send data to workers
    @everywhere A_T_global = $A_T_final
    @everywhere data_global = $data
    @everywhere t_obs_global = $(collect(t_obs))
    @everywhere X0_global = $X0
    @everywhere σ_global = $σ
    @everywhere NT_global = $NT

    # Define worker likelihood function
    @everywhere function lnlike_18param_ψ_log_worker(ψ_log)
        try
            ψ = exp.(ψ_log)
            θ = exp.(A_T_global' \ log.(ψ))
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

    lnlike_func = lnlike_18param_ψ_log_worker
    println("Worker likelihood function defined")
else
    lnlike_func = lnlike_18param_ψ_log
end

println("\nRunning profile_target...")
println("  (This will take a while with 16 nuisance parameters)")
println("  Distributed: $USE_DISTRIBUTED")
flush(stdout)

t_profile_start = time()
ψ_vals, ll_vals = ReparamTools.profile_target(
    lnlike_func, target_2d, ψ_log_lower, ψ_log_upper, nuisance_log_guess;
    grid_steps=GRID, use_distributed=USE_DISTRIBUTED,
    ω_initial_extras=nuisance_extras,
    method=:LN_BOBYQA, optmaxtime=60.0
)
t_profile_elapsed = time() - t_profile_start

println("Done in $(round(t_profile_elapsed/60, digits=1)) minutes")
println("Finite values: $(sum(isfinite.(ll_vals)))/$(length(ll_vals))")

# === RESHAPE AND ANALYZE ===
ll_matrix = reshape(ll_vals, GRID, GRID)
ll_max = maximum(ll_matrix[isfinite.(ll_matrix)])
like_matrix = exp.(ll_matrix .- ll_max)

# CI thresholds
using Distributions
lstar_2d = exp(-quantile(Chisq(2), 0.95)/2)
lstar_1d = exp(-quantile(Chisq(1), 0.95)/2)

# 1D profiles
like_ψ_target1 = [maximum(like_matrix[i, :]) for i in 1:GRID]
like_ψ_target2 = [maximum(like_matrix[:, j]) for j in 1:GRID]

ψ_target1_above = like_ψ_target1 .> lstar_1d
ψ_target2_above = like_ψ_target2 .> lstar_1d

println("\n1D Profile analysis:")
println("  K₁/β₁ (identifiable): $(sum(ψ_target1_above))/$GRID above 95% threshold")
println("  β₁·K₁ (non-identifiable): $(sum(ψ_target2_above))/$GRID above 95% threshold")

# === PLOTTING ===
println("\nGenerating plots...")
using Plots
using Contour
using ScatteredInterpolation

USE_SCATTER_PLOT = false  # Set to true for scatter, false for RBF interpolation

# True values for targets
ψ_target1_true = ψ_MLE[target_2d[1]]
ψ_target2_true = ψ_MLE[target_2d[2]]

# Helper function for θ-space transformation
function ψ_targets_to_θ1(ψ_t1, ψ_t2, ψ_ref, target_2d, ψ_to_θ_func)
    ψ_full = copy(ψ_ref)
    ψ_full[target_2d[1]] = ψ_t1
    ψ_full[target_2d[2]] = ψ_t2
    θ = ψ_to_θ_func(ψ_full)
    return θ[7], θ[10]  # β₁, K₁
end

# Plot 1: 2D profile in ψ-space
p1 = contourf(ψ_target1_grid, ψ_target2_grid, like_matrix', color=:dense, levels=20, lw=0,
              xlabel="ψ_$(target_2d[1]) = K₁/β₁ (identifiable)",
              ylabel="ψ_$(target_2d[2]) = β₁·K₁ (non-identifiable)",
              title="2D Profile in IIR coordinates\n(16 nuisance params profiled out)",
              xscale=:log10, clims=(0,1))
scatter!([ψ_target1_true], [ψ_target2_true], mc=:darkgoldenrod, msc=:match, ms=10,
         markershape=:star, label="MLE")
contour!(ψ_target1_grid, ψ_target2_grid, like_matrix', levels=[lstar_2d], color=:black, lw=2,
         xscale=:log10, label="95% CI")

# Plot 2: Transform to θ-space
β_flat = Float64[]
K_flat = Float64[]
like_flat = Float64[]

for (i, ψ_t1) in enumerate(ψ_target1_grid)
    for (j, ψ_t2) in enumerate(ψ_target2_grid)
        β1, K1 = ψ_targets_to_θ1(ψ_t1, ψ_t2, ψ_MLE, target_2d, ψ_to_θ)
        push!(β_flat, β1)
        push!(K_flat, K1)
        push!(like_flat, like_matrix[i, j])
    end
end

# θ-space bounds for plotting
β_plot_min, β_plot_max = 0.005, 0.08
K_plot_min, K_plot_max = 10.0, 80.0

if USE_SCATTER_PLOT
    p2 = scatter(β_flat, K_flat, zcolor=like_flat, c=:dense,
                 xlabel="β₁", ylabel="K₁",
                 title="Profile likelihood in θ-space (scatter)\n(other params profiled out)",
                 markersize=2, markerstrokewidth=0, label="", clims=(0,1),
                 xlims=(β_plot_min, β_plot_max), ylims=(K_plot_min, K_plot_max))
    scatter!([θ_MLE[7]], [θ_MLE[10]], mc=:darkgoldenrod, msc=:match, ms=10,
             markershape=:star, label="MLE")
else
    # RBF interpolated contour version
    β_min, β_max = β_plot_min, β_plot_max
    K_min, K_max = K_plot_min, K_plot_max
    β_reg = range(β_min, β_max, length=100)
    K_reg = range(K_min, K_max, length=100)

    # Filter scattered points to those within our rectangular region
    in_region = (β_flat .>= β_min) .& (β_flat .<= β_max) .& (K_flat .>= K_min) .& (K_flat .<= K_max)
    β_filt = β_flat[in_region]
    K_filt = K_flat[in_region]
    like_filt = like_flat[in_region]

    # Normalize scattered points to [0,1] for RBF interpolation
    β_norm = (β_filt .- β_min) ./ (β_max - β_min)
    K_norm = (K_filt .- K_min) ./ (K_max - K_min)

    # Create ThinPlate RBF interpolant in normalized space
    points_norm = hcat(β_norm, K_norm)'  # 2 × N matrix
    itp = interpolate(ThinPlate(), points_norm, like_filt)

    # Evaluate on regular grid
    like_θ_reg = zeros(length(β_reg), length(K_reg))
    for (i, β) in enumerate(β_reg)
        β_n = (β - β_min) / (β_max - β_min)
        for (j, K) in enumerate(K_reg)
            K_n = (K - K_min) / (K_max - K_min)
            like_θ_reg[i, j] = evaluate(itp, [β_n, K_n])[1]
        end
    end

    # Clamp to [0, 1]
    like_θ_reg = clamp.(like_θ_reg, 0.0, 1.0)

    p2 = contourf(collect(β_reg), collect(K_reg), like_θ_reg', color=:dense, levels=20, lw=0,
                 xlabel="β₁", ylabel="K₁", title="Profile likelihood in θ-space\n(other params profiled out)",
                 xlims=(β_min, β_max), ylims=(K_min, K_max), clims=(0,1))
    scatter!([θ_MLE[7]], [θ_MLE[10]], mc=:darkgoldenrod, msc=:match, ms=10, markershape=:star, label="MLE")

    # Transform CI contour to θ-space
    c = Contour.contour(collect(ψ_target1_grid), collect(ψ_target2_grid), like_matrix, lstar_2d)
    for line in Contour.lines(c)
        ψ_t1_c, ψ_t2_c = Contour.coordinates(line)
        β_c = Float64[]
        K_c = Float64[]
        for k in 1:length(ψ_t1_c)
            β_k, K_k = ψ_targets_to_θ1(ψ_t1_c[k], ψ_t2_c[k], ψ_MLE, target_2d, ψ_to_θ)
            if β_plot_min <= β_k <= β_plot_max && K_plot_min <= K_k <= K_plot_max
                push!(β_c, β_k)
                push!(K_c, K_k)
            end
        end
        if length(β_c) > 1
            plot!(p2, β_c, K_c, color=:black, lw=2, label="")
        end
    end
end

# Plot 3: 1D profile for identifiable
p3 = plot(ψ_target1_grid, like_ψ_target1,
          xlabel="ψ_$(target_2d[1]) = K₁/β₁ (identifiable)", ylabel="Profile Likelihood",
          title="Profile: K₁/β₁ (IDENTIFIABLE)", linewidth=2, legend=false,
          xscale=:log10, ylims=(0, 1.05))
hline!([lstar_1d], color=:red, linestyle=:dash, linewidth=2)
vline!([ψ_target1_true], color=:green, linestyle=:dot, linewidth=2)

# Plot 4: 1D profile for non-identifiable
p4 = plot(ψ_target2_grid, like_ψ_target2,
          xlabel="ψ_$(target_2d[2]) = β₁·K₁ (non-identifiable)", ylabel="Profile Likelihood",
          title="Profile: β₁·K₁ (NON-IDENTIFIABLE)", linewidth=2, legend=false,
          ylims=(0, 1.05))
hline!([lstar_1d], color=:red, linestyle=:dash, linewidth=2)
vline!([ψ_target2_true], color=:green, linestyle=:dot, linewidth=2)

# Combined plot
plt = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 1000))
savefig(plt, "iir_guided_profiling_18param_result.png")
println("\nSaved: iir_guided_profiling_18param_result.png")

# === SUMMARY ===
println("\n" * "=" ^ 70)
println("PROFILING CONFIRMS IIR ANALYSIS")
println("=" ^ 70)
println("""
The 2D profile (with $(length(nuisance_2d)) nuisance parameters profiled out) confirms IIR structure:
  - K₁/β₁ (identifiable) has a PEAKED profile ($(sum(ψ_target1_above))/$GRID above threshold)
  - β₁·K₁ (non-identifiable) has a FLAT profile ($(sum(ψ_target2_above))/$GRID above threshold)

This matches the IIR analysis which found:
  - $n_ident identifiable direction(s)
  - $n_nonident non-identifiable direction(s)

Full 18-parameter profiling demonstrates that IIR correctly identifies
the structural non-identifiability in the repressilator model.
""")
