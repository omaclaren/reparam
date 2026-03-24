# Replot profile likelihood results from saved .jls files
# Generates 4-panel figure: IIR coords, θ-space, and 1D profiles
#
# Usage: julia replot_profile_results.jl <results.jls> [output.png]

using Serialization
using Plots
using Distributions
using LinearAlgebra
using Contour
using ScatteredInterpolation

# === LOAD RESULTS ===
if length(ARGS) < 1
    error("Usage: julia replot_profile_results.jl <results.jls> [output.png]")
end

input_file = ARGS[1]
output_file = length(ARGS) >= 2 ? ARGS[2] : replace(input_file, ".jls" => "_replot.png")

println("Loading results from: $input_file")
results = deserialize(input_file)

# Extract saved data (with defaults for backwards compatibility)
ψ_vals = results["ψ_vals"]
ll_vals = results["ll_vals"]
ψ_MLE = results["ψ_MLE"]
θ_MLE = results["θ_MLE"]
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

n_params = length(θ_MLE)
β1_idx, K1_idx = 7, 10

# Plotting bounds - use computation bounds for full coverage
# IIR coordinates (ψ-space) - use actual grid range
ψ1_plot_min, ψ1_plot_max = ψ_lower[target_2d[1]], ψ_upper[target_2d[1]]
ψ2_plot_min, ψ2_plot_max = 0.0, ψ_upper[target_2d[2]]  # start at 0 for linear axis

# θ-space bounds - restricted display crop showing the well-resolved evaluated region
β_plot_min = 0.0
β_plot_max = 0.6
K_plot_min = 0.0
K_plot_max = 400.0

println("Grid: $GRID × $GRID")
println("Mode: $mode_str")
println("Target coordinates: ψ_$(target_2d[1]), ψ_$(target_2d[2])")

# === RECONSTRUCT TRANSFORMATIONS ===
function ψ_to_θ(ψ)
    exp.(A_T_final' \ log.(ψ))
end

function θ_to_ψ(θ)
    exp.(A_T_final * log.(θ))
end

# === RECONSTRUCT GRID ===
ψ_log_lower = log.(ψ_lower)
ψ_log_upper = log.(ψ_upper)

target1_log_grid = range(ψ_log_lower[target_2d[1]], ψ_log_upper[target_2d[1]], length=GRID)
target2_log_grid = range(ψ_log_lower[target_2d[2]], ψ_log_upper[target_2d[2]], length=GRID)
ψ_target1_grid = exp.(collect(target1_log_grid))
ψ_target2_grid = exp.(collect(target2_log_grid))

# === RESHAPE TO MATRIX ===
ll_matrix = reshape(ll_vals, GRID, GRID)
ll_max = maximum(ll_matrix[isfinite.(ll_matrix)])
like_matrix = exp.(ll_matrix .- ll_max)

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
gr(size=(1200, 900))

ψ_target1_true = ψ_MLE[target_2d[1]]
ψ_target2_true = ψ_MLE[target_2d[2]]

# Build subtitle based on mode
subtitle = if N_NUISANCE == 0
    "(slice: other params fixed at MLE)"
elseif N_NUISANCE == 16
    "(16 nuisance params profiled)"
else
    "($N_NUISANCE profiled, $(16-N_NUISANCE) fixed at MLE)"
end

# Plot 1: 2D profile in ψ-space
p1 = contourf(ψ_target1_grid, ψ_target2_grid, like_matrix', color=:dense, levels=20, lw=0,
              xlabel="ψ_$(target_2d[1]) = K₁/β₁ (identifiable)",
              ylabel="ψ_$(target_2d[2]) = β₁·K₁ (non-identifiable)",
              title="Profile in IIR coordinates\n$subtitle",
              xscale=:log10, xlims=(ψ1_plot_min, ψ1_plot_max), ylims=(ψ2_plot_min, ψ2_plot_max),
              clims=(0,1))
scatter!([ψ_target1_true], [ψ_target2_true], mc=:darkgoldenrod, msc=:match, ms=10,
         markershape=:star, label="MLE")
contour!(ψ_target1_grid, ψ_target2_grid, like_matrix', levels=[lstar_2d], color=:black, lw=2,
         xscale=:log10, label="95% CI")

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
             xlabel="β₁", ylabel="K₁", title="Profile in θ-space\n$subtitle",
             xlims=(β_plot_min, β_plot_max), ylims=(K_plot_min, K_plot_max), clims=(0,1))
scatter!([θ_MLE[β1_idx]], [θ_MLE[K1_idx]], mc=:darkgoldenrod, msc=:match, ms=10, markershape=:star, label="MLE")

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
            plot!(p2, β1_contour, K1_contour, color=:black, lw=2, label="")
        end
    end
catch e
    println("Warning: Could not draw θ-space contour: $e")
end

# Plot 3: 1D profile for identifiable
p3 = plot(ψ_target1_grid, like_ψ_target1,
          xlabel="ψ_$(target_2d[1]) = K₁/β₁ (identifiable)", ylabel="Profile Likelihood",
          title="Profile: K₁/β₁ (IDENTIFIABLE)", linewidth=2, legend=false,
          xscale=:log10, xlims=(ψ1_plot_min, ψ1_plot_max), ylims=(0, 1.05))
hline!([lstar_1d], color=:red, linestyle=:dash, linewidth=2)
vline!([ψ_target1_true], color=:green, linestyle=:dot, linewidth=2)

# Plot 4: 1D profile for non-identifiable
p4 = plot(ψ_target2_grid, like_ψ_target2,
          xlabel="ψ_$(target_2d[2]) = β₁·K₁ (non-identifiable)", ylabel="Profile Likelihood",
          title="Profile: β₁·K₁ (NON-IDENTIFIABLE)", linewidth=2, legend=false,
          xlims=(ψ2_plot_min, ψ2_plot_max), ylims=(0, 1.05))
hline!([lstar_1d], color=:red, linestyle=:dash, linewidth=2)
vline!([ψ_target2_true], color=:green, linestyle=:dot, linewidth=2)

# Combine
plt = plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 900))
savefig(plt, output_file)
println("\nSaved: $output_file")

println("""

Summary:
  - IIR identified $(n_ident) identifiable, $(n_nonident) non-identifiable directions
  - K₁/β₁ (identifiable): peaked profile ($n_above_1/$GRID above threshold)
  - β₁·K₁ (non-identifiable): flat profile ($n_above_2/$GRID above threshold)
""")
