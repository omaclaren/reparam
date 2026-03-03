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
#   julia --project=. repressilator_prediction_intervals_from_2d_profile.jl <results.jls>
#
# Example:
#   julia --project=. repressilator_prediction_intervals_from_2d_profile.jl nesi/repressilator_16nuisance_50x50_results.jls
#
# Outputs:
#   <results>_predictions_full2d_vs_profiles.png
#   <results>_predictions.jls

using Serialization
using Distributions
using LinearAlgebra
using Statistics
using Random
using Plots
using LaTeXStrings

include(joinpath(@__DIR__, "examples", "RepressilatorModel.jl"))
using .RepressilatorModel

if length(ARGS) < 1
    error("Usage: julia repressilator_prediction_intervals_from_2d_profile.jl <results.jls>")
end

input_file = ARGS[1]
output_base = replace(input_file, ".jls" => "")

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

function normalize_ψ_rows(ψ_vals_raw, n_points_expected)
    if ψ_vals_raw isa AbstractVector
        if isempty(ψ_vals_raw)
            return Vector{Vector{Float64}}()
        elseif ψ_vals_raw[1] isa AbstractVector
            return [collect(Float64.(v)) for v in ψ_vals_raw]
        else
            n_points_expected == 1 || error("ψ_vals is flat but ll_vals has $n_points_expected points")
            return [collect(Float64.(ψ_vals_raw))]
        end
    elseif ψ_vals_raw isa AbstractMatrix
        nr, nc = size(ψ_vals_raw)
        if nr == n_points_expected
            return [vec(Float64.(ψ_vals_raw[i, :])) for i in 1:nr]
        elseif nc == n_points_expected
            return [vec(Float64.(ψ_vals_raw[:, i])) for i in 1:nc]
        else
            error("Cannot align ψ_vals size ($nr, $nc) with ll_vals length $n_points_expected")
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
GRID^2 == length(ll_vals) || println("  WARNING: GRID^2 = $(GRID^2), ll_vals has $(length(ll_vals)) entries")
println("Mode: $mode")
println("Rank: $rank_J, Identifiable: $n_ident, Non-identifiable: $n_nonident")
println("Target coordinates: ψ_$(target_2d[1]) (identifiable), ψ_$(target_2d[2]) (non-identifiable)")
println("Nuisance profiled: $(length(nuisance_to_profile)), fixed at MLE: $(length(fixed_at_mle))")

# ψ in natural scale -> θ
ψ_to_θ(ψ) = exp.(A_T_final' \ log.(ψ))

# Saved ψ rows can have different layouts depending on run mode.
# Reconstruct full canonical log-ψ vector (length n_params).
function reconstruct_full_ψ_log(
    ψ_log_saved::Vector{Float64}, ψ_log_MLE::Vector{Float64},
    target_2d::Vector{Int}, nuisance_to_profile::Vector{Int}
)
    n_params = length(ψ_log_MLE)
    n_saved = length(ψ_log_saved)
    opt_indices = vcat(target_2d, nuisance_to_profile)

    if n_saved == length(opt_indices)
        ψ_log_full = copy(ψ_log_MLE)
        ψ_log_full[opt_indices] = ψ_log_saved
        return ψ_log_full
    elseif n_saved == length(target_2d)
        ψ_log_full = copy(ψ_log_MLE)
        ψ_log_full[target_2d] = ψ_log_saved
        return ψ_log_full
    elseif n_saved == n_params
        return copy(ψ_log_saved)
    else
        error("Cannot reconstruct ψ layout: saved length $n_saved, opt_indices length $(length(opt_indices)), n_params $n_params")
    end
end

# MLE round-trip check
θ_roundtrip = ψ_to_θ(ψ_MLE)
rt_err = maximum(abs.((θ_roundtrip .- θ_MLE) ./ θ_MLE))
println("MLE round-trip max relative error: $(round(rt_err, sigdigits=3))")

# === MODEL / DATA SETUP (must match run_repressilator_profile.jl) ===
NT, T_end = 8, 10000.0
t_obs = LinRange(0, T_end, NT)
X0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
σ = 10.0

t_pred = collect(LinRange(0, T_end, 501))
n_time = length(t_pred)
n_species = 3

θ_true = [0.008, 0.009, 0.010,
          1.0, 1.2, 1.5,
          0.02, 0.025, 0.015,
          30.0, 28.0, 32.0,
          0.006, 0.0055, 0.0065,
          0.0012, 0.0011, 0.0013]

Random.seed!(42)
y_true = RepressilatorModel.predict_mRNA(θ_true, t_obs, X0)
data = y_true + σ * randn(length(y_true))
data_mat = reshape(data, 3, NT)

predict_mRNA_fine(θ) = RepressilatorModel.predict_mRNA(θ, t_pred, X0)

pred_original_MLE = predict_mRNA_fine(θ_MLE)
pred_original_MLE_mat = reshape(pred_original_MLE, n_species, n_time)

# === THRESHOLD POLICY ===
# Use df = rank_J (identifiable-space joint region)
df = rank_J
threshold = -quantile(Chisq(df), 0.95) / 2

println("\nPrediction interval settings:")
println("  df = rank_J = $df")
println("  threshold = $(round(threshold, digits=4))")
println("  points above threshold (2D): $(sum(ll_vals .> threshold)) / $(length(ll_vals))")

# === GRID / PROFILE EXTRACTION ===
ll_matrix = reshape(ll_vals, GRID, GRID)  # rows=target1, cols=target2 (column-major)

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
ψ_log_full_grid_mle = reconstruct_full_ψ_log(ψ_vals[k_gridded_mle], ψ_log_MLE, target_2d, nuisance_to_profile)
ψ_full_grid_mle = exp.(ψ_log_full_grid_mle)
θ_gridded_MLE = ψ_to_θ(ψ_full_grid_mle)
pred_gridded_MLE = predict_mRNA_fine(θ_gridded_MLE)
pred_gridded_MLE_mat = reshape(pred_gridded_MLE, n_species, n_time)

println("  original MLE nearest grid: row $mle_row, col $mle_col")
println("  gridded MLE: row $grid_mle_row, col $grid_mle_col, idx $k_gridded_mle")
println("  ll(original-MLE-nearest) = $(round(ll_matrix[mle_row, mle_col], digits=4))")
println("  ll(gridded MLE)          = $(round(ll_vals[k_gridded_mle], digits=4))")

# 1D profiles from direct argmax on saved 2D grid
j_star_ident = [argmax(view(ll_matrix, i, :)) for i in 1:GRID]
ll_profile_ident = [ll_matrix[i, j_star_ident[i]] for i in 1:GRID]

i_star_nonident = [argmax(view(ll_matrix, :, j)) for j in 1:GRID]
ll_profile_nonident = [ll_matrix[i_star_nonident[j], j] for j in 1:GRID]

active_ident = findall(ll_profile_ident .> threshold)
active_nonident = findall(ll_profile_nonident .> threshold)
active_full2d = findall(ll_vals .> threshold)

println("\nAccepted sets:")
println("  full 2D accepted points: $(length(active_full2d)) / $(length(ll_vals))")
println("  identifiable 1D profile points: $(length(active_ident)) / $GRID")
println("  non-identifiable 1D profile points: $(length(active_nonident)) / $GRID")

isempty(active_full2d) && error("No full-2D points exceed threshold")
isempty(active_ident) && error("No identifiable-profile points exceed threshold")
isempty(active_nonident) && error("No non-identifiable-profile points exceed threshold")

linear_idx(i, j, GRID) = (j - 1) * GRID + i
ident_profile_indices = Int[linear_idx(i, j_star_ident[i], GRID) for i in active_ident]
nonident_profile_indices = Int[linear_idx(i_star_nonident[j], j, GRID) for j in active_nonident]
full2d_indices = Int.(active_full2d)

# One solve pass for all needed points
needed_indices = unique(vcat(full2d_indices, ident_profile_indices, nonident_profile_indices))
println("  unique points requiring ODE solves: $(length(needed_indices))")

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

for (n, k) in enumerate(needed_indices)
    try
        ψ_log_saved = ψ_vals[k]
        ψ_log_full = reconstruct_full_ψ_log(ψ_log_saved, ψ_log_MLE, target_2d, nuisance_to_profile)
        ψ_full = exp.(ψ_log_full)
        θ = ψ_to_θ(ψ_full)

        if any(θ .<= 0) || any(!isfinite, θ)
            global failure_bad_theta += 1
            continue
        end

        pred = predict_mRNA_fine(θ)
        if any(!isfinite, pred)
            global failure_bad_pred += 1
            continue
        end

        pred_lookup[k] = reshape(pred, n_species, n_time)
        global n_computed += 1
    catch
        global failure_exception += 1
    end

    if n % 50 == 0 || n == length(needed_indices)
        elapsed = time() - t_start
        rate = n / max(elapsed, 1e-9)
        remaining = (length(needed_indices) - n) / rate
        println("  $n / $(length(needed_indices)) processed ($(round(remaining, digits=1))s remaining)")
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

lower_full2d, upper_full2d, n_survive_full2d = envelope_from_indices(
    full2d_indices, pred_lookup, n_species, n_time)
lower_ident, upper_ident, n_survive_ident = envelope_from_indices(
    ident_profile_indices, pred_lookup, n_species, n_time)
lower_nonident, upper_nonident, n_survive_nonident = envelope_from_indices(
    nonident_profile_indices, pred_lookup, n_species, n_time)

n_survive_full2d > 0 || error("No valid prediction points for full 2D accepted set")
n_survive_ident > 0 || error("No valid prediction points for identifiable profile")
n_survive_nonident > 0 || error("No valid prediction points for non-identifiable profile")

# === SUMMARY ===
width_full2d = [mean(upper_full2d[s, :] - lower_full2d[s, :]) for s in 1:n_species]
width_ident = [mean(upper_ident[s, :] - lower_ident[s, :]) for s in 1:n_species]
width_nonident = [mean(upper_nonident[s, :] - lower_nonident[s, :]) for s in 1:n_species]

species_names = ["m₁", "m₂", "m₃"]

println("\n" * "="^74)
println("PREDICTION INTERVAL SUMMARY")
println("="^74)
println("Accepted points used:")
println("  full 2D:          $n_survive_full2d / $(length(full2d_indices))")
println("  identifiable 1D:  $n_survive_ident / $(length(ident_profile_indices))")
println("  non-identifiable 1D: $n_survive_nonident / $(length(nonident_profile_indices))")

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

outside_full2d_grid = outside_counts(pred_gridded_MLE_mat, lower_full2d, upper_full2d)
outside_ident_grid = outside_counts(pred_gridded_MLE_mat, lower_ident, upper_ident)
outside_nonident_grid = outside_counts(pred_gridded_MLE_mat, lower_nonident, upper_nonident)

println("\nGridded-MLE outside counts (out of $n_time):")
println("  full 2D band: $outside_full2d_grid")
println("  1D identifiable band: $outside_ident_grid")
println("  1D non-identifiable band: $outside_nonident_grid")

# === PLOTTING ===
println("\nGenerating figure...")

# 3 columns (cases) × 3 rows (species):
# [Full 2D | 1D identifiable | 1D non-identifiable]
# Style aligned with visualization.jl conventions (clear guides/ticks, black MLE, soft CI fills).
gr(size=(2100, 1200), dpi=200)

TITLE_FS = 18
GUIDE_FS = 15
TICK_FS = 11
LEGEND_FS = 11

species_labels = ["mRNA 1 (m₁)", "mRNA 2 (m₂)", "mRNA 3 (m₃)"]

case_specs = [
    (latexstring("\\mathrm{Full\\ 2D\\ pushforward}:\\ (K_1/\\beta_1,\\ \\beta_1 K_1)"), lower_full2d, upper_full2d, :purple),
    (latexstring("\\mathrm{1D\\ profile\\ over}\\ K_1/\\beta_1"), lower_ident, upper_ident, :deepskyblue3),
    (latexstring("\\mathrm{1D\\ profile\\ over}\\ \\beta_1 K_1"), lower_nonident, upper_nonident, :darkorange2),
]

function species_ylims(s, pred_gridded_MLE_mat, data_mat, upper_full2d)
    y_max = max(maximum(pred_gridded_MLE_mat[s, :]), maximum(data_mat[s, :]), maximum(upper_full2d[s, :]))
    return (0.0, y_max * 1.25)
end

plots_array = Any[]

for s in 1:n_species
    ylim = species_ylims(s, pred_gridded_MLE_mat, data_mat, upper_full2d)

    for (c, (case_title, lower_band, upper_band, case_color)) in enumerate(case_specs)
        lo = clamp.(lower_band[s, :], ylim[1], ylim[2])
        hi = clamp.(upper_band[s, :], ylim[1], ylim[2])

        left_pad = c == 1 ? 18Plots.mm : 5Plots.mm
        right_pad = 4Plots.mm
        top_pad = s == 1 ? 11Plots.mm : 3Plots.mm
        bottom_pad = s == n_species ? 11Plots.mm : 3Plots.mm

        p = plot(t_pred, lo, lw=0,
                 fillrange=hi, fillalpha=0.20, color=case_color,
                 label=s == 1 ? "95% CI" : "",
                 xlabel=s == n_species ? "Time (s)" : "",
                 ylabel=c == 1 ? species_labels[s] : "",
                 title=s == 1 ? case_title : "",
                 legend=s == 1 ? :topright : false,
                 legend_background_color=:white,
                 legend_foreground_color=:gray35,
                 titlefontsize=TITLE_FS,
                 guidefontsize=GUIDE_FS,
                 tickfontsize=TICK_FS,
                 legendfontsize=LEGEND_FS,
                 framestyle=:box,
                 grid=false, ylims=ylim, xlims=(0.0, T_end), xticks=0:2500:10000,
                 left_margin=left_pad, right_margin=right_pad,
                 top_margin=top_pad, bottom_margin=bottom_pad)

        plot!(p, t_pred, pred_gridded_MLE_mat[s, :], lw=2.2, color=:black,
              label=s == 1 ? "Gridded MLE" : "")

        scatter!(p, collect(t_obs), data_mat[s, :],
                 mc=:black, msc=:match, ms=3, markershape=:x,
                 label=s == 1 ? "Data" : "")

        push!(plots_array, p)
    end
end

plt = plot(plots_array..., layout=(3, 3), size=(2100, 1200), margin=2Plots.mm)
output_png = "$(output_base)_predictions_full2d_vs_profiles.png"
savefig(plt, output_png)
println("Saved: $output_png")

# === SAVE DATA ===
pred_results = Dict(
    "lower_full2d" => lower_full2d,
    "upper_full2d" => upper_full2d,
    "lower_ident" => lower_ident,
    "upper_ident" => upper_ident,
    "lower_nonident" => lower_nonident,
    "upper_nonident" => upper_nonident,
    "pred_gridded_MLE" => pred_gridded_MLE_mat,
    "pred_original_MLE" => pred_original_MLE_mat,
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
    "full2d_indices" => full2d_indices,
    "profile_indices_ident" => ident_profile_indices,
    "profile_indices_nonident" => nonident_profile_indices,
    "ll_profile_ident" => ll_profile_ident,
    "ll_profile_nonident" => ll_profile_nonident,
    "active_ident_rows" => active_ident,
    "active_nonident_cols" => active_nonident,
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

output_jls = "$(output_base)_predictions.jls"
serialize(output_jls, pred_results)
println("Saved: $output_jls")

println("\n" * "="^74)
println("DONE")
println("="^74)
