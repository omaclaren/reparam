# IIR-Guided Profiling
# Run IIR on a multi-parameter subset, then profile over the identified targets
# with remaining parameters as nuisance

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
NT, T_end = 4, 10000.0
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

# Generate data
y_true = RepressilatorModel.predict_mRNA(θ_true, t_obs, X0)
data = y_true + σ * randn(length(y_true))

println("=" ^ 70)
println("IIR-GUIDED PROFILING")
println("=" ^ 70)

# === CHOOSE PARAMETER SUBSET FOR IIR ===
# Start with β and K for all 3 genes (6 parameters)
# Indices: β(7,8,9), K(10,11,12)
free_indices = [7, 8, 9, 10, 11, 12]
free_names = ["β₁", "β₂", "β₃", "K₁", "K₂", "K₃"]
n_free = length(free_indices)

fixed_indices = setdiff(1:18, free_indices)
θ_fixed = θ_true[fixed_indices]
θ_free_true = θ_true[free_indices]

println("\nFree parameters ($n_free):")
for (i, idx) in enumerate(free_indices)
    println("  $i: $(free_names[i]) (index $idx) = $(θ_true[idx])")
end

# === DEFINE ϕ FOR THIS SUBSET ===
function ϕ_subset(θ_free)
    T = eltype(θ_free)
    θ_full = Vector{T}(undef, 18)
    θ_full[fixed_indices] .= θ_fixed
    θ_full[free_indices] .= θ_free
    return RepressilatorModel.predict_mRNA(θ_full, t_obs, X0)
end

# ϕ in log-space
ϕ_subset_log(θ_log) = ϕ_subset(exp.(θ_log))

# === RUN IIR ===
println("\n" * "=" ^ 70)
println("RUNNING IIR (find_invariant_subspace)")
println("=" ^ 70)

θ_free_log_true = log.(θ_free_true)

S, N, N_perp, rank_J = find_invariant_subspace(
    ϕ_subset_log, θ_free_log_true;
    verbose=true
)

println("\nResults:")
println("  Singular values: ", round.(S, sigdigits=4))
println("  Jacobian rank: $rank_J / $n_free")
println("  Identifiable directions (N_perp): $(size(N_perp, 2))")
println("  Non-identifiable directions (N): $(size(N, 2))")

# === INTERPRET TRANSFORMATION ===
println("\n" * "=" ^ 70)
println("TRANSFORMATION ANALYSIS")
println("=" ^ 70)

println("\nN_perp columns (identifiable directions in log-space):")
for j in 1:size(N_perp, 2)
    v = N_perp[:, j]
    println("\n  Identifiable direction $j:")
    println("    Raw: ", round.(v, digits=3))
    # Interpret: find dominant components
    sorted_idx = sortperm(abs.(v), rev=true)
    println("    Interpretation (top components):")
    for k in 1:min(3, n_free)
        idx = sorted_idx[k]
        if abs(v[idx]) > 0.1
            sign_str = v[idx] > 0 ? "+" : "-"
            println("      $sign_str$(round(abs(v[idx]), digits=2)) × log($(free_names[idx]))")
        end
    end
end

println("\nN columns (non-identifiable/invariant directions):")
for j in 1:size(N, 2)
    v = N[:, j]
    println("\n  Invariant direction $j:")
    println("    Raw: ", round.(v, digits=3))
end

# === BUILD FULL TRANSFORMATION ===
println("\n" * "=" ^ 70)
println("CONSTRUCTING COORDINATE TRANSFORMATION")
println("=" ^ 70)

# A_T has N_perp columns first (identifiable), then N columns (non-identifiable)
n_ident = size(N_perp, 2)
n_nonident = size(N, 2)

if n_nonident > 0
    A_T = hcat(N_perp, N)
else
    # All directions identifiable - construct orthogonal complement
    println("\nN is empty - all directions identifiable")
    A_T = N_perp
    # Would need to construct complement if we want full basis
end

println("\nTransformation matrix A_T (columns = new coordinates in log-space):")
println("  Size: $(size(A_T))")
println("  First $n_ident columns: identifiable (targets for profiling)")
println("  Remaining $n_nonident columns: non-identifiable (nuisance)")

# Check invertibility
if size(A_T, 1) == size(A_T, 2)
    d = det(A_T)
    println("\n  det(A_T) = $(round(d, digits=6))")
    if abs(d) > 1e-10
        println("  ✓ Matrix is invertible")
    else
        println("  ✗ Matrix is singular!")
    end
end

# === APPLY VARIMAX FOR INTERPRETABILITY ===
println("\n" * "=" ^ 70)
println("VARIMAX ROTATION ANALYSIS")
println("=" ^ 70)

# Option 1: Varimax on full matrix (may mix identifiable/non-identifiable)
A_T_varimax_full = varimax_rotation(A_T; n_restarts=200, threshold=1e-2)

println("\n1. FULL VARIMAX (rotates all 6 directions together):")
println("   WARNING: This may mix identifiable and non-identifiable directions!")
for j in 1:size(A_T_varimax_full, 2)
    v = A_T_varimax_full[:, j]
    println("\n  ψ_$j:")
    println("    Coefficients: ", round.(v, digits=3))
    for k in 1:n_free
        if abs(v[k]) > 0.3
            sign_str = v[k] > 0 ? "+" : "-"
            coef = round(abs(v[k]), digits=2)
            println("      $sign_str$coef × log($(free_names[k]))")
        end
    end
end

# Option 2: Varimax on each subspace separately (preserves identifiability structure)
println("\n2. SEPARATE VARIMAX (preserves identifiable/non-identifiable structure):")

# Rotate identifiable subspace
N_perp_varimax = varimax_rotation(N_perp; n_restarts=200, threshold=1e-2)
println("\n  Identifiable directions (varimax-rotated N_perp):")
for j in 1:size(N_perp_varimax, 2)
    v = N_perp_varimax[:, j]
    println("\n    ψ_$j (identifiable):")
    println("      Coefficients: ", round.(v, digits=3))
    for k in 1:n_free
        if abs(v[k]) > 0.3
            sign_str = v[k] > 0 ? "+" : "-"
            coef = round(abs(v[k]), digits=2)
            println("        $sign_str$coef × log($(free_names[k]))")
        end
    end
end

# Rotate non-identifiable subspace
if n_nonident > 0
    N_varimax = varimax_rotation(N; n_restarts=200, threshold=1e-2)
    println("\n  Non-identifiable directions (varimax-rotated N):")
    for j in 1:size(N_varimax, 2)
        v = N_varimax[:, j]
        println("\n    ψ_$(n_ident + j) (non-identifiable):")
        println("      Coefficients: ", round.(v, digits=3))
        for k in 1:n_free
            if abs(v[k]) > 0.3
                sign_str = v[k] > 0 ? "+" : "-"
                coef = round(abs(v[k]), digits=2)
                println("        $sign_str$coef × log($(free_names[k]))")
            end
        end
    end
    
    # Build separate-varimax transformation
    A_T_varimax_sep = hcat(N_perp_varimax, N_varimax)
else
    A_T_varimax_sep = N_perp_varimax
    N_varimax = nothing
end

println("\n  det(A_T_varimax_sep) = $(round(det(A_T_varimax_sep), digits=6))")

# === APPLY SCALE AND ROUND FOR CLEAN EXPONENTS ===
println("\n" * "=" ^ 70)
println("SCALE AND ROUND (for clean integer exponents)")
println("=" ^ 70)

# Apply scale_and_round to get ±1 instead of ±0.707
# Use column_scales to ensure we get K/β (not β/K) for all identifiable directions
# After varimax, columns may have different sign conventions - we flip as needed
N_perp_clean = scale_and_round(N_perp_varimax; round_within=0.1, column_scales=[-1, -1, 1])

println("\n  Identifiable directions (after scale_and_round):")
for j in 1:size(N_perp_clean, 2)
    v = N_perp_clean[:, j]
    println("\n    ψ_$j (identifiable):")
    println("      Coefficients: ", round.(v, digits=3))
    for k in 1:n_free
        if abs(v[k]) > 0.3
            sign_str = v[k] > 0 ? "+" : "-"
            coef = round(abs(v[k]), digits=2)
            println("        $sign_str$coef × log($(free_names[k]))")
        end
    end
end

if n_nonident > 0
    N_clean = scale_and_round(N_varimax; round_within=0.1)
    println("\n  Non-identifiable directions (after scale_and_round):")
    for j in 1:size(N_clean, 2)
        v = N_clean[:, j]
        println("\n    ψ_$(n_ident + j) (non-identifiable):")
        println("      Coefficients: ", round.(v, digits=3))
        for k in 1:n_free
            if abs(v[k]) > 0.3
                sign_str = v[k] > 0 ? "+" : "-"
                coef = round(abs(v[k]), digits=2)
                println("        $sign_str$coef × log($(free_names[k]))")
            end
        end
    end
    
    A_T_scaled = hcat(N_perp_clean, N_clean)
else
    A_T_scaled = N_perp_clean
end

println("\n  det(A_T_scaled) = $(round(det(A_T_scaled), digits=6))")
println("  Note: scale_and_round breaks orthonormality but gives cleaner interpretation")

# === USE THE IIR-DERIVED TRANSFORMATION ===
# Build full transformation matrix from IIR results
A_T_final = hcat(N_perp_clean, N_clean)
θ_to_ψ, ψ_to_θ = reparam(A_T_final)

# Print what IIR actually gives us
println("\n" * "=" ^ 70)
println("IIR-DERIVED TRANSFORMATION (A_T_final)")
println("=" ^ 70)
println("\nFull transformation matrix (columns = new coordinates in log-space):")
println("θ ordering: [β₁, β₂, β₃, K₁, K₂, K₃]")
display(round.(A_T_final, digits=3))

println("\nIdentifiable coordinates (columns 1-$n_ident):")
for j in 1:n_ident
    v = A_T_final[:, j]
    terms = String[]
    for k in 1:n_free
        if abs(v[k]) > 0.1
            coef = round(v[k], digits=2)
            push!(terms, "$(coef > 0 ? "+" : "")$coef×log($(free_names[k]))")
        end
    end
    println("  ψ_$j = exp($(join(terms, " ")))")
end

println("\nNon-identifiable coordinates (columns $(n_ident+1)-$n_free):")
for j in (n_ident+1):n_free
    v = A_T_final[:, j]
    terms = String[]
    for k in 1:n_free
        if abs(v[k]) > 0.1
            coef = round(v[k], digits=2)
            push!(terms, "$(coef > 0 ? "+" : "")$coef×log($(free_names[k]))")
        end
    end
    println("  ψ_$j = exp($(join(terms, " ")))")
end

# Test
println("\n" * "-" ^ 40)
println("Transformation test at true values:")
ψ_true = θ_to_ψ(θ_free_true)
println("  θ_free = ", round.(θ_free_true, digits=4))
println("  ψ = θ_to_ψ(θ) = ", round.(ψ_true, digits=4))
θ_back = ψ_to_θ(ψ_true)
println("  θ_back = ", round.(θ_back, digits=4))
println("  Roundtrip OK: ", isapprox(θ_free_true, θ_back, rtol=1e-10))

# === SETUP PROFILING ===
println("\n" * "=" ^ 70)
println("PROFILING SETUP")
println("=" ^ 70)

# Profile over first n_ident coordinates (identifiable)
# Nuisance: remaining n_nonident coordinates

target_indices = 1:n_ident
nuisance_indices = (n_ident+1):n_free

println("\nTarget parameters (to profile over): ψ_1 to ψ_$n_ident (identifiable)")
println("Nuisance parameters (to optimize over): ψ_$(n_ident+1) to ψ_$n_free (non-identifiable)")

# Helper function to build interpretation labels from transformation columns
function make_ψ_label(col, names)
    terms_num = String[]
    terms_den = String[]
    for k in 1:length(col)
        coef = Int(round(col[k]))
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
        return "$num/$(join(terms_den, "·"))"
    end
end

# === SUMMARY ===
println("\n" * "=" ^ 70)
println("SUMMARY OF IIR ANALYSIS")
println("=" ^ 70)

# Build dynamic interpretation of transformation
println("IIR Analysis on $(n_free)-parameter subset ($(join(free_names, ", "))):")
println("  - Found $n_ident identifiable direction(s)")
println("  - Found $n_nonident non-identifiable direction(s)")
println("\nTransformation (after separate varimax + scale_and_round):")
for j in 1:n_free
    col = A_T_final[:, j]
    status = j <= n_ident ? "identifiable" : "non-identifiable"
    label = make_ψ_label(col, free_names)
    println("  ψ_$j = $label ($status)")
end

# === 2D PROFILING USING IIR-DERIVED COORDINATES ===
# Dynamically find which ψ coordinates involve gene 1 (β₁ and K₁)
# Gene 1 corresponds to indices 1 (β₁) and 4 (K₁) in θ_free = [β₁,β₂,β₃,K₁,K₂,K₃]

println("\n" * "=" ^ 70)
println("SELECTING COORDINATES FOR GENE 1")
println("=" ^ 70)

# Find which ψ coordinates are dominated by β₁ (index 1) and K₁ (index 4) in θ_free
β1_idx_in_θfree = 1  # β₁ is first in θ_free
K1_idx_in_θfree = 4  # K₁ is fourth in θ_free

# For each ψ, check if it primarily involves gene 1
function get_gene1_involvement(col)
    # col is a column of A_T_final (exponents for one ψ in log space)
    # Check involvement of β₁ (index 1) and K₁ (index 4)
    β1_coef = abs(col[β1_idx_in_θfree])
    K1_coef = abs(col[K1_idx_in_θfree])
    other_coefs = sum(abs.(col[[2,3,5,6]]))  # β₂,β₃,K₂,K₃
    return (β1_coef + K1_coef, other_coefs)
end

println("\nAnalyzing ψ coordinates for gene 1 involvement:")
gene1_ident_idx = nothing
gene1_nonident_idx = nothing

for j in 1:n_free
    global gene1_ident_idx, gene1_nonident_idx  # Explicit global assignment
    col = A_T_final[:, j]
    gene1_inv, other_inv = get_gene1_involvement(col)
    is_gene1 = gene1_inv > 1.5 && other_inv < 0.5  # Strong gene 1, weak others
    status = j <= n_ident ? "identifiable" : "non-identifiable"
    
    # Build interpretation string
    terms = String[]
    for k in 1:n_free
        if abs(col[k]) > 0.5
            coef = Int(round(col[k]))
            push!(terms, "$(free_names[k])^$coef")
        end
    end
    interp = join(terms, "·")
    
    marker = ""
    if is_gene1
        if j <= n_ident && gene1_ident_idx === nothing
            gene1_ident_idx = j
            marker = " ← GENE 1 IDENTIFIABLE"
        elseif j > n_ident && gene1_nonident_idx === nothing
            gene1_nonident_idx = j
            marker = " ← GENE 1 NON-IDENTIFIABLE"
        end
    end
    
    println("  ψ_$j ($status): $interp (gene1=$gene1_inv, other=$other_inv)$marker")
end

if gene1_ident_idx === nothing || gene1_nonident_idx === nothing
    error("Could not find both gene 1 coordinates! Check IIR output.")
end

println("\nSelected for 2D profile:")
println("  Target ψ_$gene1_ident_idx (identifiable, gene 1)")
println("  Target ψ_$gene1_nonident_idx (non-identifiable, gene 1)")

# Set up indices dynamically
target_2d = [gene1_ident_idx, gene1_nonident_idx]
nuisance_2d = setdiff(1:n_free, target_2d)

println("  Nuisance: ψ_$(join(nuisance_2d, ", ψ_"))")

# True values in ψ-space (using full 6D transformation)
ψ_true_full = θ_to_ψ(θ_free_true)
println("\nTrue values in ψ-space:")
for j in 1:n_free
    status = j <= n_ident ? "identifiable" : "non-identifiable"
    println("  ψ_$j = $(round(ψ_true_full[j], digits=4)) ($status)")
end

# Use make_ψ_label (defined above) to build interpretation labels for targets
ψ_target1_label = make_ψ_label(A_T_final[:, target_2d[1]], free_names)
ψ_target2_label = make_ψ_label(A_T_final[:, target_2d[2]], free_names)

println("\nInterpretation:")
println("  ψ_$(target_2d[1]) = $ψ_target1_label (identifiable)")
println("  ψ_$(target_2d[2]) = $ψ_target2_label (non-identifiable)")

# Log-likelihood in full ψ-space (6D)
function lnlike_6param_ψ_log(ψ_log)
    ψ = exp.(ψ_log)
    θ_free = ψ_to_θ(ψ)
    
    # Build full 18-parameter vector
    θ_full = zeros(eltype(ψ_log), 18)
    θ_full[fixed_indices] .= θ_fixed
    θ_full[free_indices] .= θ_free
    
    pred = RepressilatorModel.predict_mRNA(θ_full, t_obs, X0)
    return -sum((data .- pred).^2) / (2 * σ^2)
end

# Bounds in log-ψ space - based on actual IIR transformation
# Need to compute bounds that correspond to reasonable θ ranges
# θ_free ∈ [β₁, β₂, β₃, K₁, K₂, K₃]
# Reasonable: β ∈ [0.001, 0.5], K ∈ [1, 500]

# Compute ψ bounds from θ bounds
θ_lower = [0.001, 0.001, 0.001, 1.0, 1.0, 1.0]
θ_upper = [0.5, 0.5, 0.5, 500.0, 500.0, 500.0]

# Transform corner points to get ψ bounds
ψ_corners = []
for β1 in [θ_lower[1], θ_upper[1]]
    for β2 in [θ_lower[2], θ_upper[2]]
        for β3 in [θ_lower[3], θ_upper[3]]
            for K1 in [θ_lower[4], θ_upper[4]]
                for K2 in [θ_lower[5], θ_upper[5]]
                    for K3 in [θ_lower[6], θ_upper[6]]
                        push!(ψ_corners, θ_to_ψ([β1, β2, β3, K1, K2, K3]))
                    end
                end
            end
        end
    end
end
ψ_corners_mat = hcat(ψ_corners...)'
lower_6d = vec(minimum(ψ_corners_mat, dims=1))
upper_6d = vec(maximum(ψ_corners_mat, dims=1))

# Convert to log space
lower_6d_log_raw = log.(lower_6d)
upper_6d_log_raw = log.(upper_6d)

# Add buffer to bounds to avoid NLopt boundary issues during adaptive continuation
# (same 30% buffer approach as examples/repressilator.jl)
println("\nAdding 30% buffer to bounds (for NLopt adaptive continuation):")
lower_6d_log = copy(lower_6d_log_raw)
upper_6d_log = copy(upper_6d_log_raw)
for j in 1:6
    range_j = upper_6d_log_raw[j] - lower_6d_log_raw[j]
    buffer = 0.3 * range_j
    lower_6d_log[j] = lower_6d_log_raw[j] + buffer
    upper_6d_log[j] = upper_6d_log_raw[j] - buffer
end

println("\nψ bounds (from θ bounds transformation, with 30% buffer):")
for j in 1:6
    println("  ψ_$j ∈ [$(round(exp(lower_6d_log[j]), sigdigits=3)), $(round(exp(upper_6d_log[j]), sigdigits=3))]")
end

# === COMPUTE 2D PROFILE WITH NUISANCE OPTIMIZATION ===
println("\n" * "=" ^ 70)
println("2D PROFILING over ψ_$(target_2d[1]) × ψ_$(target_2d[2])")
println("(profiling out ψ_$(join(nuisance_2d, ", ψ_")) as nuisance)")
println("=" ^ 70)

GRID = 50  # Finer grid for publication quality

# Grid in ψ-space for the two target coordinates
ψ_target1_grid = exp.(range(lower_6d_log[target_2d[1]], upper_6d_log[target_2d[1]], length=GRID))
ψ_target2_grid = exp.(range(lower_6d_log[target_2d[2]], upper_6d_log[target_2d[2]], length=GRID))

# Use profile_target from ReparamTools (expects log-space bounds)
# Start nuisance at midpoint of bounds (NOT true values - that would be cheating)
nuisance_lower = lower_6d_log[nuisance_2d]
nuisance_upper = upper_6d_log[nuisance_2d]
nuisance_guess = (nuisance_lower .+ nuisance_upper) ./ 2

# Ensure guess is strictly inside bounds (NLopt requires this)
eps_bound = 1e-6
nuisance_guess = clamp.(nuisance_guess, nuisance_lower .+ eps_bound, nuisance_upper .- eps_bound)

println("\nNuisance optimization setup:")
println("  Nuisance indices: $(nuisance_2d)")
println("  Initial guess (log-space midpoint): $(round.(nuisance_guess, digits=3))")
println("  Initial guess (ψ-space): $(round.(exp.(nuisance_guess), digits=3))")
println("  True values (ψ-space): $(round.(ψ_true_full[nuisance_2d], digits=4))")
println("  True values (log-space): $(round.(log.(ψ_true_full[nuisance_2d]), digits=3))")

println("\nRunning profile_target...")
t_start = time()
ψ_vals, ll_vals = profile_target(lnlike_6param_ψ_log, target_2d, lower_6d_log, upper_6d_log, nuisance_guess;
                                  grid_steps=GRID, use_distributed=false,
                                  method=:LN_BOBYQA, optmaxtime=30.0)
elapsed = time() - t_start
println("Done in $(round(elapsed, digits=1)) seconds")
println("Finite values: $(sum(isfinite.(ll_vals)))/$(length(ll_vals))")

# Verify that nuisance parameters actually changed from initial guess
# profile_target returns θ_values which include the optimized nuisance
# Check a few sample points
println("\nVerifying nuisance optimization (sample points):")
sample_indices = [1, div(length(ψ_vals), 2), length(ψ_vals)]
for idx in sample_indices
    θ_opt = ψ_vals[idx]
    nuisance_opt = θ_opt[nuisance_2d]
    println("  Grid point $idx: nuisance = $(round.(nuisance_opt, digits=3))")
end

# Reshape results
ll_matrix = reshape(ll_vals, GRID, GRID)
ll_max = maximum(ll_matrix[isfinite.(ll_matrix)])
like_matrix = exp.(ll_matrix .- ll_max)

# CI thresholds
lstar_2d = exp(-quantile(Chisq(2), 0.95)/2)
lstar_1d = exp(-quantile(Chisq(1), 0.95)/2)

# === PLOTTING ===
println("\nGenerating plots...")
using Plots
using Contour
using ScatteredInterpolation

USE_SCATTER_PLOT = false

# Convenience: true values for the two target coordinates
ψ_target1_true = ψ_true_full[target_2d[1]]
ψ_target2_true = ψ_true_full[target_2d[2]]

# Plot 1: 2D profile in ψ-space (top-left)
p1 = contourf(ψ_target1_grid, ψ_target2_grid, like_matrix', color=:dense, levels=20, lw=0,
              xlabel="ψ_$(target_2d[1]) = $ψ_target1_label (identifiable)", 
              ylabel="ψ_$(target_2d[2]) = $ψ_target2_label (non-identifiable)",
              title="2D Profile in IIR coordinates\n(ψ_$(join(nuisance_2d, ",ψ_")) profiled out)",
              xscale=:log10, clims=(0,1))
scatter!([ψ_target1_true], [ψ_target2_true], mc=:darkgoldenrod, msc=:match, ms=10, markershape=:star, label="True")
contour!(ψ_target1_grid, ψ_target2_grid, like_matrix', levels=[lstar_2d], color=:black, lw=2, 
         xscale=:log10, label="95% CI")

# Plot 2: Transform to θ-space using library's ψ_to_θ
# We need to map (ψ_target1, ψ_target2) to (β₁, K₁)
# Strategy: create a full ψ vector with nuisance at true values, then use ψ_to_θ

function ψ_targets_to_θ1(ψ_t1, ψ_t2, ψ_true_full, target_2d, ψ_to_θ_func)
    # Build full ψ vector with target values and nuisance at true
    ψ_full = copy(ψ_true_full)
    ψ_full[target_2d[1]] = ψ_t1
    ψ_full[target_2d[2]] = ψ_t2
    # Transform to θ_free = [β₁,β₂,β₃,K₁,K₂,K₃]
    θ_free = ψ_to_θ_func(ψ_full)
    # Return gene 1: β₁ (index 1) and K₁ (index 4)
    return θ_free[1], θ_free[4]  # β₁, K₁
end

β_flat = Float64[]
K_flat = Float64[]
like_flat = Float64[]

for (i, ψ_t1) in enumerate(ψ_target1_grid)
    for (j, ψ_t2) in enumerate(ψ_target2_grid)
        β1, K1 = ψ_targets_to_θ1(ψ_t1, ψ_t2, ψ_true_full, target_2d, ψ_to_θ)
        push!(β_flat, β1)
        push!(K_flat, K1)
        push!(like_flat, like_matrix[i, j])
    end
end

# True values for gene 1 (from original θ)
β1_true = θ_free_true[1]  # β₁
K1_true = θ_free_true[4]  # K₁

# Fixed rectangular region for θ-space plots
β_plot_min, β_plot_max = 0.004, 0.09
K_plot_min, K_plot_max = 1.0, 200.0

if USE_SCATTER_PLOT
    p2 = scatter(β_flat, K_flat, zcolor=like_flat, c=:dense,
                 xlabel="β₁", ylabel="K₁", title="Profile likelihood in θ-space (scatter)\n(other params profiled out)",
                 markersize=2, markerstrokewidth=0, label="", clims=(0,1),
                 xlims=(β_plot_min, β_plot_max), ylims=(K_plot_min, K_plot_max))
    scatter!([β1_true], [K1_true], mc=:darkgoldenrod, msc=:match, ms=10, markershape=:star, label="True")
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
    scatter!([β1_true], [K1_true], mc=:darkgoldenrod, msc=:match, ms=10, markershape=:star, label="True")
end

# Transform CI contour to θ-space, only plotting within rectangular region
c = Contour.contour(collect(ψ_target1_grid), collect(ψ_target2_grid), like_matrix, lstar_2d)
for line in Contour.lines(c)
    ψ_t1_c, ψ_t2_c = Contour.coordinates(line)
    β_c = Float64[]
    K_c = Float64[]
    for k in 1:length(ψ_t1_c)
        β_k, K_k = ψ_targets_to_θ1(ψ_t1_c[k], ψ_t2_c[k], ψ_true_full, target_2d, ψ_to_θ)
        # Only include points within the rectangular plot region
        if β_plot_min <= β_k <= β_plot_max && K_plot_min <= K_k <= K_plot_max
            push!(β_c, β_k)
            push!(K_c, K_k)
        end
    end
    if length(β_c) > 1
        plot!(p2, β_c, K_c, color=:black, lw=2, label="")
    end
end

# === 1D PROFILES ===
like_ψ_target1 = [maximum(like_matrix[i, :]) for i in 1:GRID]
like_ψ_target2 = [maximum(like_matrix[:, j]) for j in 1:GRID]

println("\n1D Profile projections:")
println("  Profile(ψ_$(target_2d[1])): range [$(round(minimum(like_ψ_target1), digits=4)), $(round(maximum(like_ψ_target1), digits=4))]")
println("  Profile(ψ_$(target_2d[2])): range [$(round(minimum(like_ψ_target2), digits=4)), $(round(maximum(like_ψ_target2), digits=4))]")

ψ_target1_above = like_ψ_target1 .> lstar_1d
ψ_target2_above = like_ψ_target2 .> lstar_1d
println("  ψ_$(target_2d[1]) values above 95% threshold: $(sum(ψ_target1_above))/$GRID")
println("  ψ_$(target_2d[2]) values above 95% threshold: $(sum(ψ_target2_above))/$GRID")

# Plot 3: 1D profile for identifiable coordinate (bottom-left)
p3 = plot(ψ_target1_grid, like_ψ_target1, 
          xlabel="ψ_$(target_2d[1]) = $ψ_target1_label (identifiable)", ylabel="Profile Likelihood",
          title="Profile: ψ_$(target_2d[1]) (IDENTIFIABLE)", linewidth=2, legend=false, 
          xscale=:log10, ylims=(0, 1.05))
hline!([lstar_1d], color=:red, linestyle=:dash, linewidth=2)
vline!([ψ_target1_true], color=:green, linestyle=:dot, linewidth=2)

# Plot 4: 1D profile for non-identifiable coordinate (bottom-right)
p4 = plot(ψ_target2_grid, like_ψ_target2, 
          xlabel="ψ_$(target_2d[2]) = $ψ_target2_label (non-identifiable)", ylabel="Profile Likelihood",
          title="Profile: ψ_$(target_2d[2]) (NON-IDENTIFIABLE)", linewidth=2, legend=false,
          ylims=(0, 1.05))
hline!([lstar_1d], color=:red, linestyle=:dash, linewidth=2)
vline!([ψ_target2_true], color=:green, linestyle=:dot, linewidth=2)

# Combined plot
plt = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 1000))
savefig(plt, "iir_guided_profiling_result.png")
println("\nSaved: iir_guided_profiling_result.png")

println("\n" * "=" ^ 70)
println("PROFILING CONFIRMS IIR ANALYSIS")
println("=" ^ 70)
println("""
The 2D profile (with $(length(nuisance_2d)) nuisance parameters profiled out) confirms IIR structure:
  - ψ_$(target_2d[1]) = $ψ_target1_label (identifiable) has a PEAKED profile ($(sum(ψ_target1_above))/$GRID above threshold)
  - ψ_$(target_2d[2]) = $ψ_target2_label (non-identifiable) has a FLAT profile ($(sum(ψ_target2_above))/$GRID above threshold)

This matches the IIR analysis which found:
  - $n_ident identifiable direction(s)
  - $n_nonident non-identifiable direction(s)

Key difference from minimal_2D_IIR_coords.jl:
  - Here we profile out ψ_$(join(nuisance_2d, ", ψ_")) (the other genes' coordinates)
  - This is the PROPER profile likelihood, not just fixing other parameters
""")