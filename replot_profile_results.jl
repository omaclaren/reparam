# Replot profile likelihood results from saved .jls files
# Generates 4-panel figure: IIR coords, θ-space, and 1D profiles
#
# Usage: julia replot_profile_results.jl <results.jls> [output.png]

using Serialization
ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")
using Plots
using Distributions
using LinearAlgebra
using Contour
using ScatteredInterpolation
using Measures
using LaTeXStrings

# === LOAD RESULTS ===
if length(ARGS) < 1
    error("Usage: julia replot_profile_results.jl <results.jls> [output.png]")
end

input_file = ARGS[1]
output_file = length(ARGS) >= 2 ? ARGS[2] : replace(input_file, ".jls" => "_replot_pub.png")

println("Loading results from: $input_file")
results = deserialize(input_file)

# Extract saved data (with defaults for backwards compatibility)
ψ_vals_raw = results["ψ_vals"]
ll_vals = results["ll_vals"]
ψ_MLE = results["ψ_MLE"]
θ_MLE = results["θ_MLE"]
θ_true = haskey(results, "θ_true") ? Float64.(results["θ_true"]) : nothing
A_T_final = results["A_T_final"]
target_2d = results["target_2d"]
GRID = results["GRID"]
ψ_lower = results["ψ_lower"]
ψ_upper = results["ψ_upper"]
n_ident = results["n_ident"]
n_nonident = results["n_nonident"]

# New fields (with defaults for old format)
N_NUISANCE = get(results, "N_NUISANCE", 16)
mode_str = get(results, "mode", "PROFILE (16 nuisance)")
profile_chart = get(results, "profile_chart", "full_sparse_psi")
chart_interest_matrix = get(results, "chart_interest_matrix", nothing)
chart_drop_idx = get(results, "chart_drop_idx", nothing)
chart_keep_idx = get(results, "chart_keep_idx", nothing)
chart_eta_keep_ref = get(results, "chart_eta_keep_ref", nothing)
ψ_vals_layout = get(results, "ψ_vals_layout", "legacy")
θ_vals_raw = get(results, "θ_vals", nothing)
nuisance_to_profile = get(results, "nuisance_to_profile", Int[])

n_params = length(θ_MLE)
β1_idx, K1_idx = 7, 10

# Plotting bounds - use computation bounds for full coverage
# IIR coordinates (ψ-space) - both interest coordinates are positive monomials,
# so use their actual positive ranges and display them on log axes.
ψ1_plot_min, ψ1_plot_max = ψ_lower[target_2d[1]], ψ_upper[target_2d[1]]
ψ2_plot_min, ψ2_plot_max = ψ_lower[target_2d[2]], ψ_upper[target_2d[2]]

# θ-space bounds - display zoom into the dense mapped region used for the main figure
β_plot_min = 0.0
β_plot_max = 0.2
K_plot_min = 0.0
K_plot_max = 100.0

println("Grid: $GRID × $GRID")
println("Mode: $mode_str")
println("Target coordinates: ψ_$(target_2d[1]), ψ_$(target_2d[2])")

# === RECONSTRUCT TRANSFORMATIONS ===
function ψ_to_θ(ψ)
    exp.(A_T_final' \ log.(ψ))
end

function θ_to_ψ(θ)
    exp.(A_T_final' * log.(θ))
end

ψ_true = isnothing(θ_true) ? nothing : θ_to_ψ(θ_true)

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

function standardize_ψ_grid_layout(
    ψ_vals_raw,
    n_points_expected::Int,
    ψ_log_MLE::Vector{Float64},
    target_2d::Vector{Int},
    nuisance_to_profile::Vector{Int},
    ψ_vals_layout::AbstractString,
)
    ψ_saved_rows = standardize_row_container(ψ_vals_raw, n_points_expected, "ψ_vals")

    n_params = length(ψ_log_MLE)
    opt_indices = vcat(target_2d, nuisance_to_profile)
    ψ_log_grid = Vector{Vector{Float64}}(undef, length(ψ_saved_rows))

    for k in eachindex(ψ_saved_rows)
        ψ_log_saved = ψ_saved_rows[k]
        n_saved = length(ψ_log_saved)

        if ψ_vals_layout == "canonical_full_log"
            n_saved == n_params || error("Expected canonical full ψ rows of length $n_params, got $n_saved at grid point $k")
            ψ_log_grid[k] = copy(ψ_log_saved)
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

# === RECONSTRUCT GRID ===
ψ_log_lower = log.(ψ_lower)
ψ_log_upper = log.(ψ_upper)
ψ_log_MLE = log.(ψ_MLE)
ψ_log_grid, ψ_saved_rows = standardize_ψ_grid_layout(
    ψ_vals_raw, length(ll_vals), ψ_log_MLE, Int.(target_2d), Int.(nuisance_to_profile), ψ_vals_layout)
θ_grid = isnothing(θ_vals_raw) ? nothing : standardize_row_container(θ_vals_raw, length(ll_vals), "θ_vals")

target1_log_grid = range(ψ_log_lower[target_2d[1]], ψ_log_upper[target_2d[1]], length=GRID)
target2_log_grid = range(ψ_log_lower[target_2d[2]], ψ_log_upper[target_2d[2]], length=GRID)
ψ_target1_grid = exp.(collect(target1_log_grid))
ψ_target2_grid = exp.(collect(target2_log_grid))

# === RESHAPE TO MATRIX ===
ll_matrix = reshape(ll_vals, GRID, GRID)
ll_max = maximum(ll_matrix[isfinite.(ll_matrix)])
like_matrix = exp.(ll_matrix .- ll_max)

# === GRIDDED MLE DIAGNOSTICS ===
k_gridded_mle = argmax(ll_vals)
grid_mle_row = ((k_gridded_mle - 1) % GRID) + 1
grid_mle_col = ((k_gridded_mle - 1) ÷ GRID) + 1
ψ_gridded_MLE = exp.(Float64.(ψ_log_grid[k_gridded_mle]))
θ_gridded_MLE = if isnothing(θ_grid)
    ψ_to_θ(ψ_gridded_MLE)
else
    Float64.(θ_grid[k_gridded_mle])
end

println("Gridded MLE diagnostics:")
println("  index: $k_gridded_mle")
println("  row,col: ($grid_mle_row, $grid_mle_col)")
println("  ll(gridded MLE): $(ll_vals[k_gridded_mle])")
println("  ψ target at gridded MLE: ", ψ_gridded_MLE[target_2d])
println("  θ β1,K1 at gridded MLE: ", θ_gridded_MLE[[β1_idx, K1_idx]])

# === COMPUTE 1D PROFILES ===
like_ψ_target1 = [maximum(like_matrix[i, :]) for i in 1:GRID]
like_ψ_target2 = [maximum(like_matrix[:, j]) for j in 1:GRID]

# === THRESHOLDS ===
lstar_1d = exp(-quantile(Chisq(1), 0.95)/2)
lstar_2d = exp(-quantile(Chisq(2), 0.95)/2)

# Analysis
n_above_1 = sum(like_ψ_target1 .> lstar_1d)
n_above_2 = sum(like_ψ_target2 .> lstar_1d)
println("\n1D Profile analysis:")
println("  K₁/β₁ (identifiable): $n_above_1/$GRID above 95% threshold")
println("  β₁·K₁ (non-identifiable): $n_above_2/$GRID above 95% threshold")

# === HELPER: Transform profiled interest targets to (β₁, K₁) in θ-space ===
if profile_chart == "interest_plus_original_complement" &&
   !isnothing(chart_interest_matrix) && !isnothing(chart_drop_idx) &&
   !isnothing(chart_keep_idx) && !isnothing(chart_eta_keep_ref)

    C_interest = Matrix{Float64}(chart_interest_matrix)
    drop_idx = Int.(chart_drop_idx)
    keep_idx = Int.(chart_keep_idx)
    η_keep_ref = Float64.(chart_eta_keep_ref)
    Cj_inv = inv(Matrix(C_interest[:, drop_idx]))
    Ck = Matrix(C_interest[:, keep_idx])
    θ_log_ref = log.(θ_MLE)

    function ψ_targets_to_β1K1(ψ_t1, ψ_t2, ψ_ref, target_ψ_indices, ψ_to_θ_map)
        log_interest = log.([ψ_t1, ψ_t2])
        η_drop = Cj_inv * (log_interest - Ck * η_keep_ref)
        η = copy(θ_log_ref)
        η[keep_idx] = η_keep_ref
        η[drop_idx] = η_drop
        θ = exp.(η)
        return θ[β1_idx], θ[K1_idx]
    end
else
    function ψ_targets_to_β1K1(ψ_t1, ψ_t2, ψ_ref, target_ψ_indices, ψ_to_θ_map)
        ψ_full = copy(ψ_ref)
        ψ_full[target_ψ_indices[1]] = ψ_t1
        ψ_full[target_ψ_indices[2]] = ψ_t2
        θ = ψ_to_θ_map(ψ_full)
        return θ[β1_idx], θ[K1_idx]
    end
end

# === PLOTTING ===
println("\nGenerating plots...")
gr(size=(1200, 800), dpi=600)

const LINE_W = 2
const CI_W = 1
const MARKER_SIZE_MLE = 8
const MARKER_SIZE_TRUE = 10

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

identified_axis_label = latexstring("K_1/\\beta_1")
nonidentified_axis_label = latexstring("\\beta_1 K_1")
profile_likelihood_label = "profile likelihood"

ψ_target1_mle = ψ_gridded_MLE[target_2d[1]]
ψ_target2_mle = ψ_gridded_MLE[target_2d[2]]

# Plot 1: 2D profile in ψ-space
p1 = contourf(ψ_target1_grid, ψ_target2_grid, like_matrix', color=:dense, levels=20, lw=0,
              xlabel=identified_axis_label,
              ylabel=nonidentified_axis_label,
              xscale=:log10, yscale=:log10,
              xlims=(ψ1_plot_min, ψ1_plot_max), ylims=(ψ2_plot_min, ψ2_plot_max),
              clims=(0,1), legend=false)
scatter!([ψ_target1_mle], [ψ_target2_mle], mc=:silver, msc=:match, ms=MARKER_SIZE_MLE,
         markershape=:circle, label=false)
if !isnothing(ψ_true)
    scatter!([ψ_true[target_2d[1]]], [ψ_true[target_2d[2]]], mc=:darkgoldenrod,
             msc=:match, ms=MARKER_SIZE_TRUE, markershape=:star, label=false)
end
contour!(ψ_target1_grid, ψ_target2_grid, like_matrix', levels=[lstar_2d], color=:black, lw=CI_W,
         xscale=:log10, yscale=:log10, label=false)

# Plot 2: Transform to θ-space
β1_points_θ = Float64[]
K1_points_θ = Float64[]
like_values_ψ_grid_mapped_to_θ_points = Float64[]

for (i, ψ_t1) in enumerate(ψ_target1_grid)
    for (j, ψ_t2) in enumerate(ψ_target2_grid)
        β1_val, K1_val = ψ_targets_to_β1K1(ψ_t1, ψ_t2, ψ_MLE, target_2d, ψ_to_θ)
        push!(β1_points_θ, β1_val)
        push!(K1_points_θ, K1_val)
        push!(like_values_ψ_grid_mapped_to_θ_points, like_matrix[i, j])
    end
end

# Filter to fixed plotting region
in_region = (β1_points_θ .>= β_plot_min) .& (β1_points_θ .<= β_plot_max) .&
            (K1_points_θ .>= K_plot_min) .& (K1_points_θ .<= K_plot_max)
β1_points_θ_inbounds = β1_points_θ[in_region]
K1_points_θ_inbounds = K1_points_θ[in_region]
like_values_ψ_grid_mapped_to_θ_points_inbounds = like_values_ψ_grid_mapped_to_θ_points[in_region]

# Interpolate to regular grid for contourf
β1_grid_θ = range(β_plot_min, β_plot_max, length=100)
K1_grid_θ = range(K_plot_min, K_plot_max, length=100)

# Normalize scattered points to [0,1] for RBF interpolation
β1_norm = (β1_points_θ_inbounds .- β_plot_min) ./ (β_plot_max - β_plot_min)
K1_norm = (K1_points_θ_inbounds .- K_plot_min) ./ (K_plot_max - K_plot_min)

# Create ThinPlate RBF interpolant in normalized space
points_norm = hcat(β1_norm, K1_norm)'  # 2 × N matrix
itp = interpolate(ThinPlate(), points_norm, like_values_ψ_grid_mapped_to_θ_points_inbounds)

# Evaluate on regular grid
like_θ_reg = zeros(length(β1_grid_θ), length(K1_grid_θ))
for (i, β1_val) in enumerate(β1_grid_θ)
    β1_n = (β1_val - β_plot_min) / (β_plot_max - β_plot_min)
    for (j, K1_val) in enumerate(K1_grid_θ)
        K1_n = (K1_val - K_plot_min) / (K_plot_max - K_plot_min)
        like_θ_reg[i, j] = evaluate(itp, [β1_n, K1_n])[1]
    end
end

# Clamp to [0, 1]
like_θ_reg = clamp.(like_θ_reg, 0.0, 1.0)

p2 = contourf(collect(β1_grid_θ), collect(K1_grid_θ), like_θ_reg', color=:dense, levels=20, lw=0,
             xlabel=latexstring("\\beta_1"), ylabel=latexstring("K_1"),
             xlims=(β_plot_min, β_plot_max), ylims=(K_plot_min, K_plot_max), clims=(0,1),
             legend=false)
scatter!([θ_gridded_MLE[β1_idx]], [θ_gridded_MLE[K1_idx]], mc=:silver, msc=:match,
         ms=MARKER_SIZE_MLE, markershape=:circle, label=false)
if !isnothing(θ_true)
    scatter!([θ_true[β1_idx]], [θ_true[K1_idx]], mc=:darkgoldenrod, msc=:match,
             ms=MARKER_SIZE_TRUE, markershape=:star, label=false)
end

# Transform CI contour to θ-space
try
    c = Contour.contour(collect(ψ_target1_grid), collect(ψ_target2_grid), like_matrix, lstar_2d)
    for line in Contour.lines(c)
        ψ_t1_c, ψ_t2_c = Contour.coordinates(line)
        β1_contour = Float64[]
        K1_contour = Float64[]
        for (pt1, pt2) in zip(ψ_t1_c, ψ_t2_c)
            β1_val, K1_val = ψ_targets_to_β1K1(pt1, pt2, ψ_MLE, target_2d, ψ_to_θ)
            # Only include points within plot bounds
            if β_plot_min <= β1_val <= β_plot_max && K_plot_min <= K1_val <= K_plot_max
                push!(β1_contour, β1_val)
                push!(K1_contour, K1_val)
            end
        end
        if length(β1_contour) > 1
            plot!(p2, β1_contour, K1_contour, color=:black, lw=CI_W, label="")
        end
    end
catch e
    println("Warning: Could not draw θ-space contour: $e")
end

# Plot 3: 1D profile for identifiable
p3 = plot(ψ_target1_grid, like_ψ_target1,
          xlabel=identified_axis_label, ylabel=profile_likelihood_label,
          linewidth=LINE_W, color=:black, label=false,
          xscale=:log10, xlims=(ψ1_plot_min, ψ1_plot_max), ylims=(0, 1.05))
hline!([lstar_1d], color=:black, linewidth=CI_W, label=false)
vline!([ψ_target1_mle], color=:silver, linewidth=3, label=false)
if !isnothing(ψ_true)
    vline!([ψ_true[target_2d[1]]], color=:darkgoldenrod, linestyle=:dash, linewidth=3, label=false)
end

# Plot 4: 1D profile for non-identifiable
p4 = plot(ψ_target2_grid, like_ψ_target2,
          xlabel=nonidentified_axis_label, ylabel=profile_likelihood_label,
          linewidth=LINE_W, color=:black, label=false,
          xscale=:log10, xlims=(ψ2_plot_min, ψ2_plot_max), ylims=(0, 1.05))
hline!([lstar_1d], color=:black, linewidth=CI_W, label=false)
vline!([ψ_target2_mle], color=:silver, linewidth=3, label=false)
if !isnothing(ψ_true)
    vline!([ψ_true[target_2d[2]]], color=:darkgoldenrod, linestyle=:dash, linewidth=3, label=false)
end

# Combine
plt = plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 800))
savefig(plt, output_file)
println("\nSaved: $output_file")

println("""

Summary:
  - IIR identified $(n_ident) identifiable, $(n_nonident) non-identifiable directions
  - K₁/β₁ (identifiable): peaked profile ($n_above_1/$GRID above threshold)
  - β₁·K₁ (non-identifiable): flat profile ($n_above_2/$GRID above threshold)
""")
