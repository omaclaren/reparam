# Compute prediction intervals from saved 2D profile likelihood results
#
# Post-processes .jls files from run_repressilator_profile.jl.
# No re-optimization is done here.
#
# Key points:
# 1) Saved ψ_vals are in log-space and (for profiled runs) stored in optimization
#    order [target_2d; nuisance_to_profile], not canonical ψ index order.
# 2) Directional intervals are extracted from 1D profile paths, not MLE slices:
#      - profile over ψ_target1: max over ψ_target2 at each ψ_target1 value
#      - profile over ψ_target2: max over ψ_target1 at each ψ_target2 value
#
# Usage:
#   julia --project=. compute_prediction_intervals.jl <results.jls>
#   julia --project=. compute_prediction_intervals.jl nesi/repressilator_16nuisance_50x50_results.jls
#
# Output:
#   <results>_predictions.png  — 3×2 panel figure (3 mRNA species × 2 directions)
#   <results>_predictions.jls  — saved prediction data for further analysis

using Serialization
using Distributions
using LinearAlgebra
using Statistics
using Random
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
ψ_vals_raw = results["ψ_vals"]
ll_vals = results["ll_vals"]
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

# Optional metadata (present in run_repressilator_profile.jl outputs)
nuisance_to_profile = get(results, "nuisance_to_profile", Int[])
fixed_at_mle = get(results, "fixed_at_mle", Int[])
mode = get(results, "mode", "unknown")

# Normalize ψ storage to Vector{Vector{Float64}} with one row per grid point.
# Some files store ψ_vals as Vector{Vector}; others as Matrix.
function normalize_ψ_rows(ψ_vals_raw, n_points_expected)
    if ψ_vals_raw isa AbstractVector
        if isempty(ψ_vals_raw)
            return Vector{Vector{Float64}}()
        elseif ψ_vals_raw[1] isa AbstractVector
            rows = [collect(Float64.(v)) for v in ψ_vals_raw]
            return rows
        else
            if n_points_expected != 1
                error("ψ_vals is a flat vector but ll_vals has $n_points_expected points")
            end
            return [collect(Float64.(ψ_vals_raw))]
        end
    elseif ψ_vals_raw isa AbstractMatrix
        nr, nc = size(ψ_vals_raw)
        if nr == n_points_expected
            return [vec(Float64.(ψ_vals_raw[i, :])) for i in 1:nr]
        elseif nc == n_points_expected
            return [vec(Float64.(ψ_vals_raw[:, i])) for i in 1:nc]
        else
            error("Cannot align ψ_vals matrix size ($nr, $nc) with ll_vals length $n_points_expected")
        end
    else
        error("Unsupported ψ_vals container type: $(typeof(ψ_vals_raw))")
    end
end

ψ_vals = normalize_ψ_rows(ψ_vals_raw, length(ll_vals))

n_params = length(θ_MLE)
ψ_log_MLE = log.(ψ_MLE)

println("Grid: $GRID × $GRID")
println("Stored points: ll=$(length(ll_vals)), ψ=$(length(ψ_vals))")
if GRID^2 != length(ll_vals)
    println("  WARNING: GRID^2 = $(GRID^2) but ll_vals has $(length(ll_vals)) entries")
end
println("Mode: $mode")
println("Rank: $rank_J, Identifiable: $n_ident, Non-identifiable: $n_nonident")
println("Target coordinates: ψ_$(target_2d[1]) (K₁/β₁), ψ_$(target_2d[2]) (β₁·K₁)")
println("Nuisance profiled: $(length(nuisance_to_profile)), fixed at MLE: $(length(fixed_at_mle))")

# === TRANSFORMS ===
# ψ is in natural (not log) scale here
ψ_to_θ(ψ) = exp.(A_T_final' \ log.(ψ))

# Saved ψ rows can be in different layouts depending on profiling mode:
# - full/partial profile: log-ψ in optimization order [target_2d; nuisance_to_profile]
# - slice/hybrid: log-ψ for targets only (length 2)
# This helper reconstructs full canonical log-ψ vector (length n_params).
function reconstruct_full_ψ_log(ψ_log_saved::Vector{Float64}, ψ_log_MLE::Vector{Float64},
                                target_2d::Vector{Int}, nuisance_to_profile::Vector{Int})
    n_params = length(ψ_log_MLE)
    n_saved = length(ψ_log_saved)

    opt_indices = vcat(target_2d, nuisance_to_profile)

    if n_saved == length(opt_indices)
        # Most common for profile runs (including full 16-nuisance profile)
        ψ_log_full = copy(ψ_log_MLE)
        ψ_log_full[opt_indices] = ψ_log_saved
        return ψ_log_full
    elseif n_saved == length(target_2d)
        # Slice/hybrid style: only the two targets are stored
        ψ_log_full = copy(ψ_log_MLE)
        ψ_log_full[target_2d] = ψ_log_saved
        return ψ_log_full
    elseif n_saved == n_params
        # Fallback: assume already canonical full vector
        return copy(ψ_log_saved)
    else
        error("Cannot reconstruct ψ layout: saved length $n_saved, opt_indices length $(length(opt_indices)), n_params $n_params")
    end
end

# Quick sanity check: MLE round-trip should be exact up to numerical precision
θ_roundtrip = ψ_to_θ(ψ_MLE)
rt_err = maximum(abs.((θ_roundtrip .- θ_MLE) ./ θ_MLE))
println("MLE round-trip max relative error: $(round(rt_err, sigdigits=3))")

# === MODEL SETUP (must match run_repressilator_profile.jl) ===
NT, T_end = 8, 10000.0
t_obs = LinRange(0, T_end, NT)
X0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
σ = 10.0

# Fine time grid for prediction bands
t_pred = collect(LinRange(0, T_end, 501))
n_time = length(t_pred)
n_species = 3

# True parameters (for data regeneration)
θ_true = [0.008, 0.009, 0.010,      # α₀ (1-3)
          1.0, 1.2, 1.5,            # α  (4-6)
          0.02, 0.025, 0.015,       # β  (7-9)
          30.0, 28.0, 32.0,         # K  (10-12)
          0.006, 0.0055, 0.0065,    # k_degm (13-15)
          0.0012, 0.0011, 0.0013]   # k_degp (16-18)

# Regenerate data (same seed as run_repressilator_profile.jl)
Random.seed!(42)
y_true = RepressilatorModel.predict_mRNA(θ_true, t_obs, X0)
data = y_true + σ * randn(length(y_true))

data_mat = reshape(data, 3, NT)  # rows=species, cols=time

# Prediction function
predict_mRNA_fine(θ) = RepressilatorModel.predict_mRNA(θ, t_pred, X0)

# MLE predictions on fine grid
pred_MLE = predict_mRNA_fine(θ_MLE)
pred_MLE_mat = reshape(pred_MLE, n_species, n_time)

# === THRESHOLD ===
# Use df = rank_J for joint confidence region over identifiable directions
df = rank_J
threshold = -quantile(Chisq(df), 0.95) / 2

println("\nPrediction interval settings:")
println("  df = rank_J = $df")
println("  Threshold: $(round(threshold, digits=3))")
println("  Points above threshold (2D joint region): $(sum(ll_vals .> threshold)) / $(length(ll_vals))")

# === GRID / PROFILE EXTRACTION ===
ll_matrix = reshape(ll_vals, GRID, GRID)  # rows=target1, cols=target2 (column-major storage)

# Log-grid values (for diagnostics and saved output)
ψ_log_lower = log.(ψ_lower)
ψ_log_upper = log.(ψ_upper)
target1_log_grid = collect(range(ψ_log_lower[target_2d[1]], ψ_log_upper[target_2d[1]], length=GRID))
target2_log_grid = collect(range(ψ_log_lower[target_2d[2]], ψ_log_upper[target_2d[2]], length=GRID))

# Original (continuous optimization) MLE nearest grid location (diagnostic)
ψ_MLE_target1_log = log(ψ_MLE[target_2d[1]])
ψ_MLE_target2_log = log(ψ_MLE[target_2d[2]])
mle_row = argmin(abs.(target1_log_grid .- ψ_MLE_target1_log))
mle_col = argmin(abs.(target2_log_grid .- ψ_MLE_target2_log))
println("  Original MLE nearest grid position: row $mle_row, col $mle_col")

# Gridded MLE (argmax over saved profile surface)
k_gridded_mle = argmax(ll_vals)
grid_mle_row = ((k_gridded_mle - 1) % GRID) + 1
grid_mle_col = ((k_gridded_mle - 1) ÷ GRID) + 1
ψ_log_full_grid_mle = reconstruct_full_ψ_log(ψ_vals[k_gridded_mle], ψ_log_MLE, target_2d, nuisance_to_profile)
ψ_full_grid_mle = exp.(ψ_log_full_grid_mle)
θ_gridded_MLE = ψ_to_θ(ψ_full_grid_mle)
pred_gridded_MLE = predict_mRNA_fine(θ_gridded_MLE)
pred_gridded_MLE_mat = reshape(pred_gridded_MLE, n_species, n_time)
println("  Gridded MLE position: row $grid_mle_row, col $grid_mle_col, linear index $k_gridded_mle")
println("  ll at original-MLE-nearest grid point: $(round(ll_matrix[mle_row, mle_col], digits=4))")
println("  ll at gridded MLE: $(round(ll_vals[k_gridded_mle], digits=4))")

# 1D profile over target1 (identifiable): maximize over columns for each row
j_star_ident = [argmax(view(ll_matrix, i, :)) for i in 1:GRID]
ll_profile_ident = [ll_matrix[i, j_star_ident[i]] for i in 1:GRID]

# 1D profile over target2 (non-identifiable): maximize over rows for each column
i_star_nonident = [argmax(view(ll_matrix, :, j)) for j in 1:GRID]
ll_profile_nonident = [ll_matrix[i_star_nonident[j], j] for j in 1:GRID]

# Column-major index helper
linear_idx(i, j, GRID) = (j - 1) * GRID + i

# Profile-path linear indices
k_star_ident = [linear_idx(i, j_star_ident[i], GRID) for i in 1:GRID]
k_star_nonident = [linear_idx(i_star_nonident[j], j, GRID) for j in 1:GRID]

# Values surviving joint threshold along each 1D profile
active_ident = findall(ll_profile_ident .> threshold)
active_nonident = findall(ll_profile_nonident .> threshold)

println("\nProfile extraction:")
println("  Identifiable profile points above threshold: $(length(active_ident)) / $GRID")
println("  Non-identifiable profile points above threshold: $(length(active_nonident)) / $GRID")

if isempty(active_ident)
    error("No identifiable-profile grid values exceed threshold; cannot build interval")
end
if isempty(active_nonident)
    error("No non-identifiable-profile grid values exceed threshold; cannot build interval")
end

# We only need predictions at profile-optimal points that pass threshold
needed_indices = unique(vcat(Int.(k_star_ident[active_ident]), Int.(k_star_nonident[active_nonident])))
println("  Unique grid points requiring ODE solves: $(length(needed_indices))")

# === COMPUTE PREDICTIONS AT NEEDED POINTS ===
println("\nComputing predictions at profile-optimal points...")
println("  ($(n_species) species × $(n_time) time points per solve)")
flush(stdout)

t_start = time()
n_computed = 0
n_failed = 0

# pred_lookup[k] = 3 × n_time matrix
pred_lookup = Dict{Int, Matrix{Float64}}()

for (n, k) in enumerate(needed_indices)
    try
        ψ_log_saved = ψ_vals[k]
        ψ_log_full = reconstruct_full_ψ_log(ψ_log_saved, ψ_log_MLE, target_2d, nuisance_to_profile)
        ψ_full = exp.(ψ_log_full)
        θ = ψ_to_θ(ψ_full)

        if any(θ .<= 0) || any(!isfinite, θ)
            global n_failed += 1
            continue
        end

        pred = predict_mRNA_fine(θ)
        if any(!isfinite, pred)
            global n_failed += 1
            continue
        end

        pred_lookup[k] = reshape(pred, n_species, n_time)
        global n_computed += 1
    catch
        global n_failed += 1
    end

    if n % 25 == 0 || n == length(needed_indices)
        elapsed = time() - t_start
        rate = n / max(elapsed, 1e-9)
        remaining = (length(needed_indices) - n) / rate
        println("  $n / $(length(needed_indices)) processed ($(round(remaining, digits=1))s remaining)")
        flush(stdout)
    end
end

elapsed = time() - t_start
println("Done: $n_computed successful predictions in $(round(elapsed, digits=1))s ($n_failed failed)")

# === BUILD ENVELOPES FROM PROFILE PATHS ===
function envelope_from_profile_indices(profile_linear_indices::Vector{Int},
                                       pred_lookup::Dict{Int, Matrix{Float64}},
                                       n_species::Int, n_time::Int)
    lower = fill(Inf, n_species, n_time)
    upper = fill(-Inf, n_species, n_time)
    n_used = 0

    for k in profile_linear_indices
        if haskey(pred_lookup, k)
            pred = pred_lookup[k]
            lower .= min.(lower, pred)
            upper .= max.(upper, pred)
            n_used += 1
        end
    end

    return lower, upper, n_used
end

ident_profile_indices = Int[k_star_ident[i] for i in active_ident]
nonident_profile_indices = Int[k_star_nonident[j] for j in active_nonident]

lower_vary_ident, upper_vary_ident, n_survive_ident = envelope_from_profile_indices(
    ident_profile_indices, pred_lookup, n_species, n_time)
lower_vary_nonident, upper_vary_nonident, n_survive_nonident = envelope_from_profile_indices(
    nonident_profile_indices, pred_lookup, n_species, n_time)

if n_survive_ident == 0
    error("No valid prediction points for identifiable profile after thresholding")
end
if n_survive_nonident == 0
    error("No valid prediction points for non-identifiable profile after thresholding")
end

# === REPORT ===
width_ident = [mean(upper_vary_ident[s, :] - lower_vary_ident[s, :]) for s in 1:n_species]
width_nonident = [mean(upper_vary_nonident[s, :] - lower_vary_nonident[s, :]) for s in 1:n_species]

println("\n" * "=" ^ 72)
println("PREDICTION INTERVAL SUMMARY (PROFILE-BASED)")
println("=" ^ 72)
println("Profile over K₁/β₁ (max over β₁·K₁):")
println("  Surviving profile points: $n_survive_ident / $(length(active_ident)) threshold-passing rows")
println("Profile over β₁·K₁ (max over K₁/β₁):")
println("  Surviving profile points: $n_survive_nonident / $(length(active_nonident)) threshold-passing columns")

species_names = ["m₁", "m₂", "m₃"]
println("\nPer-species mean prediction widths:")
println("  Species   Identifiable   Non-identifiable   Ratio")
for s in 1:n_species
    ratio = width_ident[s] / max(width_nonident[s], 1e-12)
    println("  $(species_names[s])      $(round(width_ident[s], digits=4))         $(round(width_nonident[s], digits=4))             $(round(ratio, digits=1))×")
end

# Diagnostic: is a reference trajectory inside each interval band?
function outside_counts(pred_ref, lower_band, upper_band)
    [count((pred_ref[s, :] .< lower_band[s, :]) .| (pred_ref[s, :] .> upper_band[s, :])) for s in 1:size(pred_ref, 1)]
end

outside_ident_orig = outside_counts(pred_MLE_mat, lower_vary_ident, upper_vary_ident)
outside_nonident_orig = outside_counts(pred_MLE_mat, lower_vary_nonident, upper_vary_nonident)
outside_ident_grid = outside_counts(pred_gridded_MLE_mat, lower_vary_ident, upper_vary_ident)
outside_nonident_grid = outside_counts(pred_gridded_MLE_mat, lower_vary_nonident, upper_vary_nonident)

println("\nReference-curve inclusion diagnostic (outside count / $(n_time)):")
println("  Original MLE vs identifiable band: $(outside_ident_orig)")
println("  Original MLE vs non-identifiable band: $(outside_nonident_orig)")
println("  Gridded MLE vs identifiable band: $(outside_ident_grid)")
println("  Gridded MLE vs non-identifiable band: $(outside_nonident_grid)")

# === PLOTTING ===
println("\nGenerating prediction interval plots...")
gr(size=(1200, 600), dpi=150)

species_labels = ["mRNA 1 (m₁)", "mRNA 2 (m₂)", "mRNA 3 (m₃)"]
species_colors = [:red, :blue, :green]

function species_ylims(s, pred_gridded_MLE_mat, data_mat)
    y_max = max(maximum(pred_gridded_MLE_mat[s, :]), maximum(data_mat[s, :]))
    return (0, y_max * 1.5)
end

plots_array = []

for s in 1:n_species
    ylim = species_ylims(s, pred_gridded_MLE_mat, data_mat)

    lo_id = clamp.(lower_vary_ident[s, :], ylim[1], ylim[2])
    hi_id = clamp.(upper_vary_ident[s, :], ylim[1], ylim[2])
    lo_ni = clamp.(lower_vary_nonident[s, :], ylim[1], ylim[2])
    hi_ni = clamp.(upper_vary_nonident[s, :], ylim[1], ylim[2])

    p_id = plot(t_pred, lo_id, lw=0,
                fillrange=hi_id, fillalpha=0.25, color=species_colors[s],
                label="95% CI", xlabel="Time (s)", ylabel="Concentration",
                title="Profile over K₁/β₁ (identifiable)\n$(species_labels[s])",
                legend=:topright, grid=false, ylims=ylim)
    plot!(p_id, t_pred, pred_gridded_MLE_mat[s, :], lw=2.2, color=species_colors[s], label="MLE")
    scatter!(p_id, collect(t_obs), data_mat[s, :], mc=:black, msc=:match, ms=3,
             markershape=:xcross, label="Data")

    p_ni = plot(t_pred, lo_ni, lw=0,
                fillrange=hi_ni, fillalpha=0.25, color=species_colors[s],
                label="95% CI", xlabel="Time (s)", ylabel="Concentration",
                title="Profile over β₁·K₁ (non-identifiable)\n$(species_labels[s])",
                legend=:topright, grid=false, ylims=ylim)
    plot!(p_ni, t_pred, pred_gridded_MLE_mat[s, :], lw=2.2, color=species_colors[s], label="MLE")
    scatter!(p_ni, collect(t_obs), data_mat[s, :], mc=:black, msc=:match, ms=3,
             markershape=:xcross, label="Data")

    push!(plots_array, p_id)
    push!(plots_array, p_ni)
end

# Layout: 3 rows × 2 columns
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
    "pred_MLE" => pred_gridded_MLE_mat,
    "pred_gridded_MLE" => pred_gridded_MLE_mat,
    "pred_original_MLE" => pred_MLE_mat,
    "t_pred" => t_pred,
    "t_obs" => collect(t_obs),
    "data" => data_mat,
    "df" => df,
    "threshold" => threshold,
    "n_computed" => n_computed,
    "n_failed" => n_failed,
    "n_survive_ident" => n_survive_ident,
    "n_survive_nonident" => n_survive_nonident,
    "width_ident" => width_ident,
    "width_nonident" => width_nonident,
    "mle_row" => mle_row,
    "mle_col" => mle_col,
    "k_gridded_mle" => k_gridded_mle,
    "grid_mle_row" => grid_mle_row,
    "grid_mle_col" => grid_mle_col,
    "θ_original_MLE" => θ_MLE,
    "θ_gridded_MLE" => θ_gridded_MLE,
    "outside_ident_original" => outside_ident_orig,
    "outside_nonident_original" => outside_nonident_orig,
    "outside_ident_gridded" => outside_ident_grid,
    "outside_nonident_gridded" => outside_nonident_grid,
    "rank_J" => rank_J,
    "ll_profile_ident" => ll_profile_ident,
    "ll_profile_nonident" => ll_profile_nonident,
    "active_ident_rows" => active_ident,
    "active_nonident_cols" => active_nonident,
    "j_star_ident" => j_star_ident,
    "i_star_nonident" => i_star_nonident,
    "profile_indices_ident" => ident_profile_indices,
    "profile_indices_nonident" => nonident_profile_indices,
)

output_jls = "$(output_base)_predictions.jls"
serialize(output_jls, pred_results)
println("Saved: $output_jls")

println("\n" * "=" ^ 72)
println("DONE (needs user verification)")
println("=" ^ 72)
