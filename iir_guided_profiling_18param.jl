# IIR-Guided Profiling - Full 18 Parameter Version
# Extension of iir_guided_profiling.jl to all 18 repressilator parameters
#
# Development approach:
# 1. First: Find MLE, run IIR, examine results (this file, current state)
# 2. Next: Add profiling once IIR results are understood

# Include modules (with guard to avoid double-loading)
if !@isdefined(ReparamTools)
    include("ReparamTools.jl")
end
if !@isdefined(RepressilatorModel)
    include("examples/RepressilatorModel.jl")
end

using .ReparamTools
using .RepressilatorModel
using Distributions, LinearAlgebra, Random, ForwardDiff

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
lnlike_θ = construct_lnlike_xy(distrib_θ, data; dist_type=:multi)
lnlike_θ_log = θ_log -> lnlike_θ(exp.(θ_log))

# === FIND MLE ===
println("\n" * "=" ^ 70)
println("FINDING MLE")
println("=" ^ 70)

# Start from midpoint of bounds
θ_log_initial = 0.5 * (θ_log_lower + θ_log_upper)

# Generate initial guesses
n_guesses = 3
nuisance_guesses = generate_initial_guesses(θ_log_lower, θ_log_upper, n_guesses)

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
θ_log_MLE, lnlike_MLE = profile_target(
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
# - rtolJ=1e-7 ensures σ~4e-5 is classified as zero (4000× gap from σ~0.18)
rtolJ_custom = 1e-7  # Respect the large singular value gap

println("\nUsing high-precision ϕ with:")
println("  Time points: $(length(t_iir)) (fine grid)")
println("  ODE tolerances: abstol=1e-10, reltol=1e-8")

S, N, N_perp, rank_J = find_invariant_subspace(
    ϕ_iir_log, θ_log_MLE;
    rtolJ=rtolJ_custom,
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

N_perp_varimax = varimax_rotation(N_perp; n_restarts=200, threshold=1e-2)

# First do scale_and_round, then fix signs to get K/β (not β/K)
N_perp_clean = scale_and_round(N_perp_varimax; round_within=0.15)

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

    N_varimax = varimax_rotation(N; n_restarts=200, threshold=1e-2)
    N_clean = scale_and_round(N_varimax; round_within=0.15)

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
    println("   2. Relaxing the tolerance (rtolM parameter)")
    println("   3. The problem might be borderline at this evaluation point")
end

println("\n" * "=" ^ 70)
println("NEXT STEPS")
println("=" ^ 70)
println("""
If IIR found expected structure (3 non-identifiable):
  - Proceed to profiling setup
  - Identify gene 1 coordinates for 2D profile

If IIR found 0 non-identifiable:
  - Check singular value gap
  - May need to adjust rtolM tolerance
  - Compare with repressilator.jl results at same MLE
""")
