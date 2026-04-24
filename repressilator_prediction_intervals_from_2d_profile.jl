# Repressilator prediction intervals from a saved 2D profile-likelihood run
#
# This is a specialized post-processing utility for outputs from:
#   run_repressilator_profile.jl
#
# It computes prediction envelopes for three sets of accepted points:
#   1) Full accepted 2D set (ll > threshold)
#   2) 1D profile over identifiable target (row-wise argmax)
#   3) 1D profile over non-identifiable target (column-wise argmax)
#
# No nuisance re-optimization is done here; it reuses saved profile results.
#
# Usage:
#   julia --project=. repressilator_prediction_intervals_from_2d_profile.jl <results.jls> [--states=mrna|protein] [--protein-yaxis=row-inset|panel|full|clipped]
#
# Example:
#   julia --project=. repressilator_prediction_intervals_from_2d_profile.jl nesi/repressilator_16nuisance_50x50_results.jls
#
# Outputs:
#   <results>_predictions_full2d_vs_profiles_pub.png
#   <results>_predictions_pub.jls
#
# With --states=protein:
#   <results>_protein_predictions_row_inset_full2d_vs_profiles_pub.png
#   <results>_protein_predictions_pub.jls

using Serialization
using Distributions
using LinearAlgebra
using Statistics
using Random
ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")
using Plots
using Measures
using LaTeXStrings

include(joinpath(@__DIR__, "examples", "RepressilatorModel.jl"))
using .RepressilatorModel

if length(ARGS) < 1
    error("Usage: julia repressilator_prediction_intervals_from_2d_profile.jl <results.jls> [--states=mrna|protein]")
end

input_file = ARGS[1]
output_base = replace(input_file, ".jls" => "")

state_kind = get(ENV, "REPRESSILATOR_PREDICTION_STATES", "mrna")
protein_yaxis_mode = get(ENV, "REPRESSILATOR_PROTEIN_YAXIS", "row-inset")
for arg in ARGS[2:end]
    if startswith(arg, "--states=")
        global state_kind = lowercase(split(arg, "=", limit=2)[2])
    elseif startswith(arg, "--protein-yaxis=")
        global protein_yaxis_mode = lowercase(split(arg, "=", limit=2)[2])
    else
        error("Unknown argument: $arg")
    end
end
state_kind in ("mrna", "protein") || error("--states must be 'mrna' or 'protein', got '$state_kind'")
protein_yaxis_mode == "rowinset" && (protein_yaxis_mode = "row-inset")
protein_yaxis_mode in ("row-inset", "full", "panel", "clipped") ||
    error("--protein-yaxis must be 'row-inset', 'panel', 'full', or 'clipped', got '$protein_yaxis_mode'")

println("Loading results from: $input_file")
results = deserialize(input_file)

required_keys = [
    "ψ_vals", "ll_vals", "ψ_MLE", "θ_MLE", "A_T_final", "target_2d",
    "GRID", "ψ_lower", "ψ_upper", "rank_J", "n_ident", "n_nonident"
]
for key in required_keys
    haskey(results, key) || error("Missing required key '$key' in results file")
end

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

# Optional metadata
nuisance_to_profile = get(results, "nuisance_to_profile", Int[])
fixed_at_mle = get(results, "fixed_at_mle", Int[])
mode = get(results, "mode", "unknown")
profile_chart = get(results, "profile_chart", "full_sparse_psi")
ψ_vals_layout = get(results, "ψ_vals_layout", "legacy")
θ_vals_raw = get(results, "θ_vals", nothing)

function standardize_row_container(rows_raw, n_points_expected::Int, name::String)
    rows = if rows_raw isa AbstractVector
        if isempty(rows_raw)
            Vector{Vector{Float64}}()
        elseif rows_raw[1] isa AbstractVector
            [collect(Float64.(v)) for v in rows_raw]
        else
            n_points_expected == 1 || error("$name is flat but expected $n_points_expected points")
            [collect(Float64.(rows_raw))]
        end
    elseif rows_raw isa AbstractMatrix
        nr, nc = size(rows_raw)
        if nr == n_points_expected
            [vec(Float64.(rows_raw[i, :])) for i in 1:nr]
        elseif nc == n_points_expected
            [vec(Float64.(rows_raw[:, i])) for i in 1:nc]
        else
            error("Cannot align $name size ($nr, $nc) with expected length $n_points_expected")
        end
    else
        error("Unsupported $name container type: $(typeof(rows_raw))")
    end
    return rows
end

"""
standardize_ψ_grid_layout(ψ_vals_raw, n_points_expected, ψ_log_MLE, target_2d, nuisance_to_profile)
    -> (ψ_log_grid, ψ_saved_rows)

Convert saved ψ containers into a single, human-readable canonical layout.

Output format (both vectors have length = number of grid points):
- ψ_saved_rows[k] : raw saved log-ψ row for grid point `k` (as stored in file)
- ψ_log_grid[k]   : full canonical log-ψ vector for grid point `k`
                    (length = n_params, indexed in canonical ψ coordinate order)

So conceptually:
- row 1 = parameter vector at grid point 1
- row 2 = parameter vector at grid point 2
- ...

This function also expands partial layouts (e.g., slice files storing only 2 targets)
by filling non-stored coordinates from ψ_MLE.
"""
function standardize_ψ_grid_layout(
    ψ_vals_raw,
    n_points_expected::Int,
    ψ_log_MLE::Vector{Float64},
    target_2d::Vector{Int},
    nuisance_to_profile::Vector{Int}
)
    # 1) Standardize container to vector-of-rows in saved layout
    ψ_saved_rows = standardize_row_container(ψ_vals_raw, n_points_expected, "ψ_vals")

    # 2) Expand each row to full canonical log-ψ layout
    n_params = length(ψ_log_MLE)
    opt_indices = vcat(target_2d, nuisance_to_profile)
    ψ_log_grid = Vector{Vector{Float64}}(undef, length(ψ_saved_rows))

    for k in eachindex(ψ_saved_rows)
        ψ_log_saved = ψ_saved_rows[k]
        n_saved = length(ψ_log_saved)

        if ψ_vals_layout == "canonical_full_log"
            n_saved == n_params || error("Expected canonical full ψ rows of length $n_params, got $n_saved at grid point $k")
            ψ_log_grid[k] = copy(ψ_log_saved)
        # IMPORTANT: check opt_indices layout before generic n_params fallback.
        # For legacy full-profile repressilator files, n_saved == length(opt_indices) == n_params,
        # but rows are stored in optimization order [target_2d; nuisance_to_profile],
        # not canonical ψ index order.
        elseif n_saved == length(opt_indices)
            ψ_log_full = copy(ψ_log_MLE)
            ψ_log_full[opt_indices] = ψ_log_saved
            ψ_log_grid[k] = ψ_log_full
        elseif n_saved == length(target_2d)
            ψ_log_full = copy(ψ_log_MLE)
            ψ_log_full[target_2d] = ψ_log_saved
            ψ_log_grid[k] = ψ_log_full
        elseif n_saved == n_params
            ψ_log_grid[k] = copy(ψ_log_saved)
        else
            error("Cannot standardize ψ layout at grid point $k: saved length $n_saved, opt_indices length $(length(opt_indices)), n_params $n_params")
        end
    end

    return ψ_log_grid, ψ_saved_rows
end

n_params = length(θ_MLE)
ψ_log_MLE = log.(ψ_MLE)
ψ_log_grid, ψ_saved_rows = standardize_ψ_grid_layout(
    ψ_vals_raw, length(ll_vals), ψ_log_MLE, target_2d, nuisance_to_profile)
θ_grid = isnothing(θ_vals_raw) ? nothing : standardize_row_container(θ_vals_raw, length(ll_vals), "θ_vals")

println("Grid: $GRID × $GRID")
println("Stored points: ll=$(length(ll_vals)), ψ=$(length(ψ_log_grid))")
GRID^2 == length(ll_vals) || println("  WARNING: GRID^2 = $(GRID^2), ll_vals has $(length(ll_vals)) entries")
println("Mode: $mode")
println("Profile chart: $profile_chart")
println("Rank: $rank_J, Identifiable: $n_ident, Non-identifiable: $n_nonident")
println("Target coordinates: ψ_$(target_2d[1]) (identifiable), ψ_$(target_2d[2]) (non-identifiable)")
println("Nuisance profiled: $(length(nuisance_to_profile)), fixed at MLE: $(length(fixed_at_mle))")
println("Prediction states: $state_kind")
if state_kind == "protein"
    println("Protein y-axis mode: $protein_yaxis_mode")
end
println("Saved ψ row lengths: $(sort(unique(length.(ψ_saved_rows))))")
println("Standardized ψ row length: $n_params")
if !isnothing(θ_grid)
    println("Stored θ row lengths: $(sort(unique(length.(θ_grid))))")
end

# ψ in natural scale -> θ
ψ_to_θ(ψ) = exp.(A_T_final' \ log.(ψ))

# MLE round-trip check
θ_roundtrip = ψ_to_θ(ψ_MLE)
rt_err = maximum(abs.((θ_roundtrip .- θ_MLE) ./ θ_MLE))
println("MLE round-trip max relative error: $(round(rt_err, sigdigits=3))")

# === MODEL / DATA SETUP ===
# Preferred path: load stored observation data/metadata from results file.
# Fallback for legacy result files: regenerate synthetic data with legacy constants.

const n_species = 3

has_stored_data = all(haskey(results, k) for k in ["data", "t_obs", "X0", "σ"])

if has_stored_data
    data = vec(Float64.(results["data"]))
    t_obs = collect(Float64.(results["t_obs"]))
    X0 = collect(Float64.(results["X0"]))
    σ = Float64(results["σ"])
    NT = haskey(results, "NT") ? Int(results["NT"]) : length(t_obs)
    T_end = haskey(results, "T_end") ? Float64(results["T_end"]) : maximum(t_obs)

    length(t_obs) == NT || error("Stored NT=$NT does not match length(t_obs)=$(length(t_obs))")
    length(data) == n_species * NT || error("Stored data length $(length(data)) incompatible with n_species*NT=$(n_species*NT)")

    println("Loaded observation data/metadata from results file.")
else
    println("WARNING: Results file does not include stored data. Falling back to legacy synthetic-data regeneration.")

    NT, T_end = 8, 10000.0
    t_obs = collect(LinRange(0, T_end, NT))
    X0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    σ = 10.0

    θ_true = [0.008, 0.009, 0.010,
              1.0, 1.2, 1.5,
              0.02, 0.025, 0.015,
              30.0, 28.0, 32.0,
              0.006, 0.0055, 0.0065,
              0.0012, 0.0011, 0.0013]

    Random.seed!(42)
    y_true = RepressilatorModel.predict_mRNA(θ_true, t_obs, X0)
    data = y_true + σ * randn(length(y_true))
end

data_mat = reshape(data, n_species, NT)

t_pred = collect(LinRange(0, T_end, 501))
n_time = length(t_pred)

state_rows = state_kind == "mrna" ? (1:3) : (4:6)
state_label = state_kind == "mrna" ? "mRNA" : "protein"
state_symbol_prefix = state_kind == "mrna" ? "m" : "p"

function predict_state_fine(θ)
    sol_matrix = RepressilatorModel.solve_repressilator(t_pred, θ, X0)
    return sol_matrix[state_rows, :]
end

pred_original_MLE_mat = predict_state_fine(θ_MLE)
pred_true_mat = state_kind == "protein" && haskey(results, "θ_true") ?
    predict_state_fine(Float64.(results["θ_true"])) : nothing

# === THRESHOLD POLICY ===
# Use df = rank_J (identifiable-space joint region)
df = rank_J
threshold = -quantile(Chisq(df), 0.95) / 2

println("\nPrediction interval settings:")
println("  df = rank_J = $df")
println("  threshold = $(round(threshold, digits=4))")
println("  points above threshold (2D): $(sum(ll_vals .> threshold)) / $(length(ll_vals))")

# === GRID / PROFILE EXTRACTION ===
lnlike_grid_ψ1_ψ2 = reshape(ll_vals, GRID, GRID)  # rows=target1, cols=target2 (column-major)

ψ_log_lower = log.(ψ_lower)
ψ_log_upper = log.(ψ_upper)
target1_log_grid = collect(range(ψ_log_lower[target_2d[1]], ψ_log_upper[target_2d[1]], length=GRID))
target2_log_grid = collect(range(ψ_log_lower[target_2d[2]], ψ_log_upper[target_2d[2]], length=GRID))

# Original continuous MLE nearest grid point (diagnostic)
ψ_MLE_target1_log = log(ψ_MLE[target_2d[1]])
ψ_MLE_target2_log = log(ψ_MLE[target_2d[2]])
mle_row = argmin(abs.(target1_log_grid .- ψ_MLE_target1_log))
mle_col = argmin(abs.(target2_log_grid .- ψ_MLE_target2_log))

# Gridded MLE (argmax over saved surface)
k_gridded_mle = argmax(ll_vals)
grid_mle_row = ((k_gridded_mle - 1) % GRID) + 1
grid_mle_col = ((k_gridded_mle - 1) ÷ GRID) + 1
θ_gridded_MLE = if isnothing(θ_grid)
    ψ_log_full_grid_mle = ψ_log_grid[k_gridded_mle]
    ψ_full_grid_mle = exp.(ψ_log_full_grid_mle)
    ψ_to_θ(ψ_full_grid_mle)
else
    θ_grid[k_gridded_mle]
end
pred_gridded_MLE_mat = predict_state_fine(θ_gridded_MLE)

println("  original MLE nearest grid: row $mle_row, col $mle_col")
println("  gridded MLE: row $grid_mle_row, col $grid_mle_col, idx $k_gridded_mle")
println("  ll(original-MLE-nearest) = $(round(lnlike_grid_ψ1_ψ2[mle_row, mle_col], digits=4))")
println("  ll(gridded MLE)          = $(round(ll_vals[k_gridded_mle], digits=4))")

# 1D profiles from direct argmax on saved 2D grid
ψ1_argmax_ψ2_indices = [argmax(view(lnlike_grid_ψ1_ψ2, i, :)) for i in 1:GRID]
lnlike_profile_ψ1 = [lnlike_grid_ψ1_ψ2[i, ψ1_argmax_ψ2_indices[i]] for i in 1:GRID]

ψ2_argmax_ψ1_indices = [argmax(view(lnlike_grid_ψ1_ψ2, :, j)) for j in 1:GRID]
lnlike_profile_ψ2 = [lnlike_grid_ψ1_ψ2[ψ2_argmax_ψ1_indices[j], j] for j in 1:GRID]

# accepted_ψ1_indices / accepted_ψ2_indices are 1D axis indices (ψ1 rows or ψ2 cols).
# accepted_ψ1ψ2_indices are indices into the full joint ψ1-ψ2 grid (ll_vals ordering).
accepted_ψ1_indices = findall(lnlike_profile_ψ1 .> threshold)
accepted_ψ2_indices = findall(lnlike_profile_ψ2 .> threshold)
accepted_ψ1ψ2_indices = findall(ll_vals .> threshold)

println("\nAccepted sets:")
println("  full 2D accepted points: $(length(accepted_ψ1ψ2_indices)) / $(length(ll_vals))")
println("  identifiable 1D profile points: $(length(accepted_ψ1_indices)) / $GRID")
println("  non-identifiable 1D profile points: $(length(accepted_ψ2_indices)) / $GRID")

isempty(accepted_ψ1ψ2_indices) && error("No full-2D points exceed threshold")
isempty(accepted_ψ1_indices) && error("No identifiable-profile points exceed threshold")
isempty(accepted_ψ2_indices) && error("No non-identifiable-profile points exceed threshold")

linear_idx(i, j, GRID) = (j - 1) * GRID + i
# accepted_ψ*_profile_path_indices are full-grid indices for accepted points along each profile path.
accepted_ψ1_profile_path_indices = Int[linear_idx(i, ψ1_argmax_ψ2_indices[i], GRID) for i in accepted_ψ1_indices]
accepted_ψ2_profile_path_indices = Int[linear_idx(ψ2_argmax_ψ1_indices[j], j, GRID) for j in accepted_ψ2_indices]
accepted_ψ1ψ2_indices = Int.(accepted_ψ1ψ2_indices)

# One solve pass for all needed points
prediction_eval_indices = unique(vcat(accepted_ψ1ψ2_indices, accepted_ψ1_profile_path_indices, accepted_ψ2_profile_path_indices))
println("  unique points requiring ODE solves: $(length(prediction_eval_indices))")

# === PREDICTIONS AT ACCEPTED POINTS ===
println("\nComputing predictions...")
println("  ($n_species species × $n_time time points per solve)")
flush(stdout)

t_start = time()
n_computed = 0
failure_bad_theta = 0
failure_bad_pred = 0
failure_exception = 0

pred_lookup = Dict{Int, Matrix{Float64}}()  # k -> (3 × n_time)

for (n, k) in enumerate(prediction_eval_indices)
    try
        θ = if isnothing(θ_grid)
            ψ_log_full = ψ_log_grid[k]
            ψ_full = exp.(ψ_log_full)
            ψ_to_θ(ψ_full)
        else
            θ_grid[k]
        end

        if any(θ .<= 0) || any(!isfinite, θ)
            global failure_bad_theta += 1
            continue
        end

        pred = predict_state_fine(θ)
        if any(!isfinite, pred)
            global failure_bad_pred += 1
            continue
        end

        pred_lookup[k] = pred
        global n_computed += 1
    catch
        global failure_exception += 1
    end

    if n % 50 == 0 || n == length(prediction_eval_indices)
        elapsed = time() - t_start
        rate = n / max(elapsed, 1e-9)
        remaining = (length(prediction_eval_indices) - n) / rate
        println("  $n / $(length(prediction_eval_indices)) processed ($(round(remaining, digits=1))s remaining)")
        flush(stdout)
    end
end

elapsed = time() - t_start
n_failed = failure_bad_theta + failure_bad_pred + failure_exception
println("Done: $n_computed successful in $(round(elapsed, digits=1))s; failures=$n_failed")
if n_failed > 0
    println("  failure breakdown: bad_theta=$failure_bad_theta, bad_pred=$failure_bad_pred, exception=$failure_exception")
end

function envelope_from_indices(indices::Vector{Int}, pred_lookup::Dict{Int, Matrix{Float64}}, n_species::Int, n_time::Int)
    lower = fill(Inf, n_species, n_time)
    upper = fill(-Inf, n_species, n_time)
    n_used = 0

    for k in indices
        if haskey(pred_lookup, k)
            pred = pred_lookup[k]
            lower .= min.(lower, pred)
            upper .= max.(upper, pred)
            n_used += 1
        end
    end

    return lower, upper, n_used
end

lower_pred_from_accepted_ψ1ψ2, upper_pred_from_accepted_ψ1ψ2, n_survive_full2d = envelope_from_indices(
    accepted_ψ1ψ2_indices, pred_lookup, n_species, n_time)
lower_pred_from_accepted_ψ1_profile, upper_pred_from_accepted_ψ1_profile, n_survive_ident = envelope_from_indices(
    accepted_ψ1_profile_path_indices, pred_lookup, n_species, n_time)
lower_pred_from_accepted_ψ2_profile, upper_pred_from_accepted_ψ2_profile, n_survive_nonident = envelope_from_indices(
    accepted_ψ2_profile_path_indices, pred_lookup, n_species, n_time)

n_survive_full2d > 0 || error("No valid prediction points for full 2D accepted set")
n_survive_ident > 0 || error("No valid prediction points for identifiable profile")
n_survive_nonident > 0 || error("No valid prediction points for non-identifiable profile")

# === SUMMARY ===
width_full2d = [mean(upper_pred_from_accepted_ψ1ψ2[s, :] - lower_pred_from_accepted_ψ1ψ2[s, :]) for s in 1:n_species]
width_ident = [mean(upper_pred_from_accepted_ψ1_profile[s, :] - lower_pred_from_accepted_ψ1_profile[s, :]) for s in 1:n_species]
width_nonident = [mean(upper_pred_from_accepted_ψ2_profile[s, :] - lower_pred_from_accepted_ψ2_profile[s, :]) for s in 1:n_species]

species_names = ["$(state_symbol_prefix)₁", "$(state_symbol_prefix)₂", "$(state_symbol_prefix)₃"]

println("\n" * "="^74)
println("PREDICTION INTERVAL SUMMARY ($state_label)")
println("="^74)
println("Accepted points used:")
println("  full 2D:          $n_survive_full2d / $(length(accepted_ψ1ψ2_indices))")
println("  identifiable 1D:  $n_survive_ident / $(length(accepted_ψ1_profile_path_indices))")
println("  non-identifiable 1D: $n_survive_nonident / $(length(accepted_ψ2_profile_path_indices))")

println("\nPer-species mean width comparison:")
println("  Species   Full-2D      1D Ident      1D Non-ident   Ident/Full   Non-ident/Full")
for s in 1:n_species
    r_ident = width_ident[s] / max(width_full2d[s], 1e-12)
    r_nonident = width_nonident[s] / max(width_full2d[s], 1e-12)
    println("  $(species_names[s])      $(round(width_full2d[s], digits=4))      $(round(width_ident[s], digits=4))      $(round(width_nonident[s], digits=4))      $(round(r_ident, digits=3))      $(round(r_nonident, digits=3))")
end

function outside_counts(pred_ref, lower_band, upper_band)
    [count((pred_ref[s, :] .< lower_band[s, :]) .| (pred_ref[s, :] .> upper_band[s, :]))
     for s in 1:size(pred_ref, 1)]
end

outside_full2d_grid = outside_counts(pred_gridded_MLE_mat, lower_pred_from_accepted_ψ1ψ2, upper_pred_from_accepted_ψ1ψ2)
outside_ident_grid = outside_counts(pred_gridded_MLE_mat, lower_pred_from_accepted_ψ1_profile, upper_pred_from_accepted_ψ1_profile)
outside_nonident_grid = outside_counts(pred_gridded_MLE_mat, lower_pred_from_accepted_ψ2_profile, upper_pred_from_accepted_ψ2_profile)

println("\nGridded-MLE outside counts (out of $n_time):")
println("  full 2D band: $outside_full2d_grid")
println("  1D identifiable band: $outside_ident_grid")
println("  1D non-identifiable band: $outside_nonident_grid")

# === PLOTTING ===
println("\nGenerating figure...")

# 3 columns (cases) × 3 rows (species):
# [Full 2D | 1D identifiable | 1D non-identifiable]
# Publication styling matched to visualization.jl/stat_model figures.
gr(size=(2200, 1400), dpi=300)

LINE_W = 2
BAND_ALPHA = 0.20

default(
    xguidefontsize=24,
    yguidefontsize=24,
    xtickfontsize=16,
    ytickfontsize=16,
    legendfontsize=16,
    bottom_margin=6mm,
    left_margin=6mm,
    top_margin=4.5mm,
    right_margin=4.5mm,
    framestyle=:box,
    grid=false,
)

species_labels = [latexstring("$(state_symbol_prefix)_$s") for s in 1:n_species]

case_specs = [
    (latexstring("95\\%\\ \\mathrm{CI}\\ (\\mathrm{full\\ 2D})"), lower_pred_from_accepted_ψ1ψ2, upper_pred_from_accepted_ψ1ψ2, :purple),
    (latexstring("95\\%\\ \\mathrm{CI}\\ (K_1/\\beta_1)"), lower_pred_from_accepted_ψ1_profile, upper_pred_from_accepted_ψ1_profile, :steelblue3),
    (latexstring("95\\%\\ \\mathrm{CI}\\ (\\beta_1 K_1)"), lower_pred_from_accepted_ψ2_profile, upper_pred_from_accepted_ψ2_profile, :orange),
]

function protein_row_maxima()
    maxima = zeros(n_species)
    for s in 1:n_species
        maxima[s] = max(
            maximum(pred_gridded_MLE_mat[s, :]),
            maximum(upper_pred_from_accepted_ψ1ψ2[s, :]),
            maximum(upper_pred_from_accepted_ψ1_profile[s, :]),
            maximum(upper_pred_from_accepted_ψ2_profile[s, :]),
        )
        if !isnothing(pred_true_mat)
            maxima[s] = max(maxima[s], maximum(pred_true_mat[s, :]))
        end
    end
    return maxima
end

protein_row_ymax = state_kind == "protein" ? protein_row_maxima() : zeros(n_species)

function species_ylims(s, lower_band, upper_band, pred_gridded_MLE_mat, data_mat,
                       upper_pred_from_accepted_ψ1ψ2, pred_true_mat, state_kind)
    if state_kind == "protein"
        if protein_yaxis_mode == "clipped"
            return s == 1 ? (0.0, 1.5e4) : (0.0, 500.0)
        elseif protein_yaxis_mode == "row-inset"
            return s == 1 ? (0.0, 1.5e4) : (0.0, 2.5e4)
        elseif protein_yaxis_mode == "panel"
            if s == 1
                return (0.0, 1.5e4)
            end

            y_max = max(maximum(pred_gridded_MLE_mat[s, :]), maximum(upper_band[s, :]))
            if !isnothing(pred_true_mat)
                y_max = max(y_max, maximum(pred_true_mat[s, :]))
            end
            return (0.0, y_max * 1.1)
        end
    end

    y_max = max(maximum(pred_gridded_MLE_mat[s, :]), maximum(upper_pred_from_accepted_ψ1ψ2[s, :]))
    if state_kind == "mrna"
        y_max = max(y_max, maximum(data_mat[s, :]))
    end
    if !isnothing(pred_true_mat)
        y_max = max(y_max, maximum(pred_true_mat[s, :]))
    end
    return (0.0, y_max * 1.25)
end

function zoom_ylims(s, upper_band, pred_gridded_MLE_mat, pred_true_mat)
    y_max = max(maximum(pred_gridded_MLE_mat[s, :]), maximum(upper_band[s, :]))
    if !isnothing(pred_true_mat)
        y_max = max(y_max, maximum(pred_true_mat[s, :]))
    end
    return (0.0, y_max * 1.12)
end

function add_zoom_inset!(p, s, lower_band, upper_band, case_color)
    zoom_ylim = zoom_ylims(s, upper_band, pred_gridded_MLE_mat, pred_true_mat)
    lo_zoom = clamp.(lower_band[s, :], zoom_ylim[1], zoom_ylim[2])
    hi_zoom = clamp.(upper_band[s, :], zoom_ylim[1], zoom_ylim[2])
    ytick_max = 100.0 * floor(zoom_ylim[2] / 100.0)
    zoom_yticks = ytick_max >= 200.0 ? collect(0.0:100.0:ytick_max) : :auto

    plot!(p, inset=(1, bbox(0.31, 0.33, 0.64, 0.57, :bottom, :left)))
    plot!(p, t_pred, lo_zoom, lw=0, fillrange=hi_zoom, fillalpha=BAND_ALPHA,
          color=case_color, label=false, subplot=2,
          xlims=(0.0, T_end), ylims=zoom_ylim, xticks=false,
          yticks=zoom_yticks, xtickfontsize=7, ytickfontsize=9,
          framestyle=:box, grid=false,
          foreground_color_subplot=:gray35,
          background_color_inside=:white,
          left_margin=1.0mm, right_margin=1.0mm,
          top_margin=1.0mm, bottom_margin=1.0mm)
    plot!(p, t_pred, pred_gridded_MLE_mat[s, :], lw=1.1, color=:black,
          label=false, subplot=2)
    if !isnothing(pred_true_mat)
        plot!(p, t_pred, pred_true_mat[s, :], lw=1.6, color=:darkgoldenrod,
              linestyle=:dash, label=false, subplot=2)
    end
    return p
end

plots_array = Any[]

for s in 1:n_species
    for (c, (ci_label, lower_band, upper_band, case_color)) in enumerate(case_specs)
        ylim = species_ylims(s, lower_band, upper_band, pred_gridded_MLE_mat, data_mat,
                             upper_pred_from_accepted_ψ1ψ2, pred_true_mat, state_kind)
        lo = clamp.(lower_band[s, :], ylim[1], ylim[2])
        hi = clamp.(upper_band[s, :], ylim[1], ylim[2])

        left_pad = c == 1 ? 12Plots.mm : 4.5Plots.mm
        right_pad = 4.5Plots.mm
        top_pad = 4.5Plots.mm
        bottom_pad = s == n_species ? 10Plots.mm : 3Plots.mm

        p = plot(t_pred, lo, lw=0,
                 fillrange=hi, fillalpha=BAND_ALPHA, color=case_color,
                 label=s == 1 ? ci_label : "",
                 xlabel=s == n_species ? latexstring("t") : "",
                 ylabel=c == 1 ? species_labels[s] : "",
                 legend=s == 1 ? :topleft : false,
                 framestyle=:box,
                 grid=false, ylims=ylim, xlims=(0.0, T_end), xticks=0:2500:10000,
                 left_margin=left_pad, right_margin=right_pad,
                 top_margin=top_pad, bottom_margin=bottom_pad)

        plot!(p, t_pred, pred_gridded_MLE_mat[s, :], lw=LINE_W, color=:black,
              label=s == 1 ? "MLE" : "")

        if state_kind == "mrna"
            scatter!(p, collect(t_obs), data_mat[s, :],
                     mc=:black, msc=:match, ms=3, markershape=:x,
                     label=s == 1 ? "data" : "")
        end

        if !isnothing(pred_true_mat)
            plot!(p, t_pred, pred_true_mat[s, :], lw=3, color=:darkgoldenrod,
                  linestyle=:dash, label=s == 1 ? "truth" : "")
        end

        if state_kind == "protein" && protein_yaxis_mode == "row-inset" && c == 3 && s in (2, 3)
            add_zoom_inset!(p, s, lower_band, upper_band, case_color)
        end

        push!(plots_array, p)
    end
end

plt = plot(plots_array..., layout=(3, 3), size=(2200, 1400), margin=2.5mm)
protein_yaxis_file_label = replace(protein_yaxis_mode, "-" => "_")
output_png = state_kind == "mrna" ?
    "$(output_base)_predictions_full2d_vs_profiles_pub.png" :
    (protein_yaxis_mode == "full" ?
     "$(output_base)_protein_predictions_full2d_vs_profiles_pub.png" :
     "$(output_base)_protein_predictions_$(protein_yaxis_file_label)_full2d_vs_profiles_pub.png")
savefig(plt, output_png)
println("Saved: $output_png")

# === SAVE DATA ===
pred_results = Dict(
    "lower_pred_from_accepted_ψ1ψ2" => lower_pred_from_accepted_ψ1ψ2,
    "upper_pred_from_accepted_ψ1ψ2" => upper_pred_from_accepted_ψ1ψ2,
    "lower_pred_from_accepted_ψ1_profile" => lower_pred_from_accepted_ψ1_profile,
    "upper_pred_from_accepted_ψ1_profile" => upper_pred_from_accepted_ψ1_profile,
    "lower_pred_from_accepted_ψ2_profile" => lower_pred_from_accepted_ψ2_profile,
    "upper_pred_from_accepted_ψ2_profile" => upper_pred_from_accepted_ψ2_profile,
    "pred_gridded_MLE" => pred_gridded_MLE_mat,
    "pred_original_MLE" => pred_original_MLE_mat,
    "pred_true" => pred_true_mat,
    "states" => state_kind,
    "protein_yaxis_mode" => protein_yaxis_mode,
    "t_pred" => t_pred,
    "t_obs" => collect(t_obs),
    "data" => data_mat,
    "df" => df,
    "threshold" => threshold,
    "n_computed" => n_computed,
    "n_failed" => n_failed,
    "failure_bad_theta" => failure_bad_theta,
    "failure_bad_pred" => failure_bad_pred,
    "failure_exception" => failure_exception,
    "n_survive_full2d" => n_survive_full2d,
    "n_survive_ident" => n_survive_ident,
    "n_survive_nonident" => n_survive_nonident,
    "width_full2d" => width_full2d,
    "width_ident" => width_ident,
    "width_nonident" => width_nonident,
    "full2d_indices" => accepted_ψ1ψ2_indices,
    "profile_indices_ident" => accepted_ψ1_profile_path_indices,
    "profile_indices_nonident" => accepted_ψ2_profile_path_indices,
    "ll_profile_ident" => lnlike_profile_ψ1,
    "ll_profile_nonident" => lnlike_profile_ψ2,
    "active_ident_rows" => accepted_ψ1_indices,
    "active_nonident_cols" => accepted_ψ2_indices,
    "outside_full2d_gridded" => outside_full2d_grid,
    "outside_ident_gridded" => outside_ident_grid,
    "outside_nonident_gridded" => outside_nonident_grid,
    "mle_row" => mle_row,
    "mle_col" => mle_col,
    "k_gridded_mle" => k_gridded_mle,
    "grid_mle_row" => grid_mle_row,
    "grid_mle_col" => grid_mle_col,
    "θ_original_MLE" => θ_MLE,
    "θ_gridded_MLE" => θ_gridded_MLE,
    "output_png" => output_png,
    "rank_J" => rank_J,
)

output_jls = state_kind == "mrna" ?
    "$(output_base)_predictions_pub.jls" :
    "$(output_base)_protein_predictions_pub.jls"
serialize(output_jls, pred_results)
println("Saved: $output_jls")

println("\n" * "="^74)
println("DONE")
println("="^74)
