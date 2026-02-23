# ============================================================================
# KNOWN INCORRECT — DO NOT USE FOR PAPER
# ============================================================================
# Issues identified (2025-02-23):
# 1. Parameterisation likely wrong: ψ_vals are in log(ψ) space (IIR coordinates).
#    Must convert ψ-log → ψ (natural) → θ (original model params: β, K, etc.)
#    RepressilatorModel.predict_mRNA takes θ (original params), NOT ψ.
#    β and K are original model parameters; β·K and K/β are IIR coordinates.
#    The inverse IIR transform (ψ → θ) must be correct for predictions to work.
# 2. Method is wrong: uses fixed MLE row/column slices instead of proper 1D
#    profile extraction (maximize over other direction). This conflates nuisance
#    variation with target parameter effects — evidenced by m₂ showing reversed
#    pattern (non-identifiable width > identifiable width).
# 3. Results NOT verified by user before committing.
# See context/context_20260223_phase6_prediction_intervals_broken.md for details.
# ============================================================================
#
# Compute prediction intervals from saved 2D profile likelihood results
#
# Post-processes .jls files from run_repressilator_profile.jl (NeSI output).
# No re-optimization needed — uses the saved parameter vectors directly.
#
# For each surviving parameter vector (above χ²(rank_J) threshold), solves the
# ODE on a fine time grid and takes pointwise min/max to form prediction bands.
#
# The key demonstration: profiling over the non-identifiable combination (β₁·K₁)
# produces ~zero prediction uncertainty, while profiling over the identifiable
# combination (K₁/β₁) produces non-zero prediction uncertainty.
#
# Usage:
#   julia --project=. compute_prediction_intervals.jl <results.jls>
#   julia --project=. compute_prediction_intervals.jl nesi/repressilator_16nuisance_50x50_results.jls
#
# Output:
#   <results>_predictions.png  — 2×3 panel figure (2 directions × 3 mRNA species)
#   <results>_predictions.jls  — saved prediction data for further analysis

using Serialization
using Distributions
using LinearAlgebra
using Statistics
using Plots

# Load model
include("examples/RepressilatorModel.jl")
using .RepressilatorModel

# === PARSE ARGUMENTS ===
if length(ARGS) < 1
    error("Usage: julia compute_prediction_intervals.jl <results.jls>")
end

input_file = ARGS[1]
output_base = replace(input_file, ".jls" => "")

println("Loading results from: $input_file")
results = deserialize(input_file)

# === EXTRACT SAVED DATA ===
ψ_vals = results["ψ_vals"]       # Vector of 18-dim ψ vectors (log-space, optimised nuisance)
ll_vals = results["ll_vals"]     # Normalised log-likelihoods (max = 0)
ψ_MLE = results["ψ_MLE"]
θ_MLE = results["θ_MLE"]
A_T_final = results["A_T_final"]
target_2d = results["target_2d"]
GRID = results["GRID"]
ψ_lower = results["ψ_lower"]
ψ_upper = results["ψ_upper"]
rank_J = results["rank_J"]
n_ident = results["n_ident"]
n_nonident = results["n_nonident"]

n_params = length(θ_MLE)

println("Grid: $GRID × $GRID = $(length(ψ_vals)) points")
println("Rank: $rank_J, Identifiable: $n_ident, Non-identifiable: $n_nonident")
println("Target coordinates: ψ_$(target_2d[1]) (K₁/β₁), ψ_$(target_2d[2]) (β₁·K₁)")

# === RECONSTRUCT TRANSFORMATIONS ===
function ψ_to_θ(ψ)
    exp.(A_T_final' \ log.(ψ))
end

# === MODEL SETUP (must match run_repressilator_profile.jl) ===
NT, T_end = 8, 10000.0
t_obs = LinRange(0, T_end, NT)
X0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
σ = 10.0

# Fine time grid for prediction bands
t_pred = collect(LinRange(0, T_end, 501))
n_time = length(t_pred)
n_species = 3

# True parameters (for reference / data regeneration)
θ_true = [0.008, 0.009, 0.010,      # α₀ (1-3)
          1.0, 1.2, 1.5,            # α  (4-6)
          0.02, 0.025, 0.015,       # β  (7-9)
          30.0, 28.0, 32.0,         # K  (10-12)
          0.006, 0.0055, 0.0065,    # k_degm (13-15)
          0.0012, 0.0011, 0.0013]   # k_degp (16-18)

# Regenerate data (same seed as run_repressilator_profile.jl)
using Random
Random.seed!(42)
y_true = RepressilatorModel.predict_mRNA(θ_true, t_obs, X0)
data = y_true + σ * randn(length(y_true))

# Reshape data for plotting (3 species × NT times → species per column)
data_mat = reshape(data, 3, NT)  # 3 × NT: rows = species, cols = time

# === PREDICTION FUNCTION ===
function predict_mRNA_fine(θ)
    RepressilatorModel.predict_mRNA(θ, t_pred, X0)
end

# MLE predictions on fine grid
pred_MLE = predict_mRNA_fine(θ_MLE)
pred_MLE_mat = reshape(pred_MLE, n_species, n_time)  # 3 × n_time

# === THRESHOLD AND FILTERING ===
# Use df = rank_J for joint confidence region over all identifiable directions
df = rank_J
threshold = -quantile(Chisq(df), 0.95) / 2

println("\nPrediction interval settings:")
println("  df = rank_J = $df")
println("  Threshold: $(round(threshold, digits=2))")
n_above = sum(ll_vals .> threshold)
println("  Points above threshold: $n_above / $(length(ll_vals))")

# === RECONSTRUCT GRID INDICES ===
# The 2D grid is stored in column-major order: target_2d[1] varies fastest (rows),
# target_2d[2] varies slowest (columns)
ψ_log_lower = log.(ψ_lower)
ψ_log_upper = log.(ψ_upper)
target1_log_grid = range(ψ_log_lower[target_2d[1]], ψ_log_upper[target_2d[1]], length=GRID)
target2_log_grid = range(ψ_log_lower[target_2d[2]], ψ_log_upper[target_2d[2]], length=GRID)

# Reshape likelihood to matrix for row/column operations
ll_matrix = reshape(ll_vals, GRID, GRID)  # rows = target1 (K₁/β₁), cols = target2 (β₁·K₁)

# === COMPUTE PREDICTIONS AT ALL SURVIVING POINTS ===
println("\nComputing predictions at $n_above parameter vectors...")
println("  ($(n_species) species × $(n_time) time points per solve)")
flush(stdout)

t_start = time()
n_computed = 0
n_failed = 0

# Store predictions for all surviving points, indexed by grid position
# pred_all[linear_idx] = 3 × n_time matrix (or nothing if failed/below threshold)
pred_all = Vector{Union{Nothing, Matrix{Float64}}}(nothing, length(ψ_vals))

for (k, (ψ_log, ll)) in enumerate(zip(ψ_vals, ll_vals))
    if ll <= threshold || !isfinite(ll)
        continue
    end

    try
        ψ = exp.(ψ_log)
        θ = ψ_to_θ(ψ)

        if any(θ .<= 0) || any(!isfinite, θ)
            global n_failed += 1
            continue
        end

        pred = predict_mRNA_fine(θ)

        if any(!isfinite, pred)
            global n_failed += 1
            continue
        end

        pred_all[k] = reshape(pred, n_species, n_time)
        global n_computed += 1
    catch e
        global n_failed += 1
    end

    if n_computed % 200 == 0
        elapsed = time() - t_start
        rate = n_computed / elapsed
        remaining = (n_above - n_computed) / rate
        println("  $n_computed / $n_above computed ($(round(remaining, digits=0))s remaining)")
        flush(stdout)
    end
end

elapsed = time() - t_start
println("Done: $n_computed predictions in $(round(elapsed, digits=1))s ($n_failed failed)")

# === COMPUTE PREDICTION BANDS ===
# Two comparisons:
#   1. Profile over K₁/β₁ (identifiable): for each row i, take envelope across columns
#   2. Profile over β₁·K₁ (non-identifiable): for each column j, take envelope across rows
#
# For each direction, we compute:
#   - The 1D profile likelihood (max over other direction)
#   - The prediction band at each grid value of the target coordinate
#   - The "marginal" prediction band (union over all surviving points)

println("\nComputing prediction bands...")

# --- Direction 1: Identifiable (K₁/β₁) ---
# For each value of K₁/β₁ (row i), collect predictions from all columns above threshold
lower_ident = fill(Inf, n_species, n_time)
upper_ident = fill(-Inf, n_species, n_time)

for i in 1:GRID
    for j in 1:GRID
        k = (j - 1) * GRID + i  # column-major index
        if pred_all[k] !== nothing
            lower_ident .= min.(lower_ident, pred_all[k])
            upper_ident .= max.(upper_ident, pred_all[k])
        end
    end
end

# --- Direction 2: Non-identifiable (β₁·K₁) ---
# Same envelope — all surviving points contribute
# (Conceptually we're profiling over β₁·K₁, but since predictions are ~invariant
# to it, the band should be ~the same width regardless of which direction we "fix")
lower_nonident = copy(lower_ident)
upper_nonident = copy(upper_ident)

# BUT the interesting comparison is: how much does the prediction vary when we
# move along each direction SEPARATELY?
#
# For the identifiable direction: fix β₁·K₁ at the MLE column, vary K₁/β₁
# For the non-identifiable direction: fix K₁/β₁ at the MLE row, vary β₁·K₁

# Find the grid indices closest to MLE
ψ_MLE_target1_log = log(ψ_MLE[target_2d[1]])
ψ_MLE_target2_log = log(ψ_MLE[target_2d[2]])
mle_row = argmin(abs.(collect(target1_log_grid) .- ψ_MLE_target1_log))
mle_col = argmin(abs.(collect(target2_log_grid) .- ψ_MLE_target2_log))
println("  MLE grid position: row $mle_row, col $mle_col")

# Identifiable direction: vary K₁/β₁ (rows), fix β₁·K₁ at MLE column
lower_vary_ident = fill(Inf, n_species, n_time)
upper_vary_ident = fill(-Inf, n_species, n_time)
n_survive_ident = 0

for i in 1:GRID
    k = (mle_col - 1) * GRID + i
    if pred_all[k] !== nothing
        lower_vary_ident .= min.(lower_vary_ident, pred_all[k])
        upper_vary_ident .= max.(upper_vary_ident, pred_all[k])
        global n_survive_ident += 1
    end
end

# Non-identifiable direction: vary β₁·K₁ (columns), fix K₁/β₁ at MLE row
lower_vary_nonident = fill(Inf, n_species, n_time)
upper_vary_nonident = fill(-Inf, n_species, n_time)
n_survive_nonident = 0

for j in 1:GRID
    k = (j - 1) * GRID + mle_row
    if pred_all[k] !== nothing
        lower_vary_nonident .= min.(lower_vary_nonident, pred_all[k])
        upper_vary_nonident .= max.(upper_vary_nonident, pred_all[k])
        global n_survive_nonident += 1
    end
end

# === REPORT ===
width_ident = [mean(upper_vary_ident[s, :] - lower_vary_ident[s, :]) for s in 1:n_species]
width_nonident = [mean(upper_vary_nonident[s, :] - lower_vary_nonident[s, :]) for s in 1:n_species]

println("\n" * "=" ^ 70)
println("PREDICTION INTERVAL SUMMARY")
println("=" ^ 70)
println("Varying K₁/β₁ (identifiable), β₁·K₁ fixed at MLE column:")
println("  Surviving points: $n_survive_ident / $GRID")
println("\nVarying β₁·K₁ (non-identifiable), K₁/β₁ fixed at MLE row:")
println("  Surviving points: $n_survive_nonident / $GRID")

# Per-species widths
species_names = ["m₁", "m₂", "m₃"]
println("\nPer-species mean prediction widths:")
println("  Species   Identifiable   Non-identifiable   Ratio")
for s in 1:n_species
    ratio = width_ident[s] / max(width_nonident[s], 1e-10)
    println("  $(species_names[s])      $(round(width_ident[s], digits=4))         $(round(width_nonident[s], digits=4))             $(round(ratio, digits=1))×")
end

# === PLOTTING ===
println("\nGenerating prediction interval plots...")
gr(size=(1200, 600), dpi=150)

species_labels = ["mRNA 1 (m₁)", "mRNA 2 (m₂)", "mRNA 3 (m₃)"]
species_colors = [:red, :blue, :green]

# Clamp y-axis to reasonable range based on MLE + data
# (extreme parameter values can produce divergent ODE solutions)
function species_ylims(s, pred_MLE_mat, data_mat)
    y_max = max(maximum(pred_MLE_mat[s, :]), maximum(data_mat[s, :]))
    return (0, y_max * 1.5)  # 50% headroom above max of MLE/data
end

plots_array = []

for s in 1:n_species
    ylim = species_ylims(s, pred_MLE_mat, data_mat)

    # Clamp prediction bands to plot range for clean rendering
    lo_id = clamp.(lower_vary_ident[s, :], ylim[1], ylim[2])
    hi_id = clamp.(upper_vary_ident[s, :], ylim[1], ylim[2])
    lo_ni = clamp.(lower_vary_nonident[s, :], ylim[1], ylim[2])
    hi_ni = clamp.(upper_vary_nonident[s, :], ylim[1], ylim[2])

    # Identifiable direction
    p_id = plot(t_pred, lo_id, lw=0,
                fillrange=hi_id, fillalpha=0.25, color=species_colors[s],
                label="95% CI", xlabel="Time (s)", ylabel="Concentration",
                title="Vary K₁/β₁ (identifiable)\n$(species_labels[s])",
                legend=:topright, grid=false, ylims=ylim)
    plot!(t_pred, pred_MLE_mat[s, :], lw=2, color=species_colors[s], label="MLE")
    scatter!(collect(t_obs), data_mat[s, :], mc=:black, msc=:match, ms=3,
             markershape=:xcross, label="Data")

    # Non-identifiable direction
    p_ni = plot(t_pred, lo_ni, lw=0,
                fillrange=hi_ni, fillalpha=0.25, color=species_colors[s],
                label="95% CI", xlabel="Time (s)", ylabel="Concentration",
                title="Vary β₁·K₁ (non-identifiable)\n$(species_labels[s])",
                legend=:topright, grid=false, ylims=ylim)
    plot!(t_pred, pred_MLE_mat[s, :], lw=2, color=species_colors[s], label="MLE")
    scatter!(collect(t_obs), data_mat[s, :], mc=:black, msc=:match, ms=3,
             markershape=:xcross, label="Data")

    push!(plots_array, p_id)
    push!(plots_array, p_ni)
end

# Layout: 3 rows (species) × 2 columns (identifiable, non-identifiable)
plt = plot(plots_array..., layout=(3, 2), size=(1200, 900))

output_png = "$(output_base)_predictions.png"
savefig(plt, output_png)
println("Saved: $output_png")

# === SAVE PREDICTION DATA ===
pred_results = Dict(
    "lower_vary_ident" => lower_vary_ident,
    "upper_vary_ident" => upper_vary_ident,
    "lower_vary_nonident" => lower_vary_nonident,
    "upper_vary_nonident" => upper_vary_nonident,
    "pred_MLE" => pred_MLE_mat,
    "t_pred" => t_pred,
    "t_obs" => collect(t_obs),
    "data" => data_mat,
    "df" => df,
    "threshold" => threshold,
    "n_computed" => n_computed,
    "n_survive_ident" => n_survive_ident,
    "n_survive_nonident" => n_survive_nonident,
    "width_ident" => width_ident,          # per-species vector
    "width_nonident" => width_nonident,    # per-species vector
    "mle_row" => mle_row,
    "mle_col" => mle_col,
    "rank_J" => rank_J,
)

output_jls = "$(output_base)_predictions.jls"
serialize(output_jls, pred_results)
println("Saved: $output_jls")

println("\n" * "=" ^ 70)
println("DONE")
println("=" ^ 70)
