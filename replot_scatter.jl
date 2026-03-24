# Scatter plot version of profile likelihood results
# Shows actual computed grid points without interpolation
#
# Usage: julia replot_scatter.jl <results.jls> [output.png]

using Serialization
using Plots
using Distributions
using LinearAlgebra

# === LOAD RESULTS ===
if length(ARGS) < 1
    error("Usage: julia replot_scatter.jl <results.jls> [output.png]")
end

input_file = ARGS[1]
output_file = length(ARGS) >= 2 ? ARGS[2] : replace(input_file, ".jls" => "_scatter.png")

println("Loading results from: $input_file")
results = deserialize(input_file)

# Extract saved data
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
N_NUISANCE = get(results, "N_NUISANCE", 16)
mode_str = get(results, "mode", "PROFILE (16 nuisance)")
profile_chart = get(results, "profile_chart", "full_sparse_psi")
chart_interest_matrix = get(results, "chart_interest_matrix", nothing)
chart_drop_idx = get(results, "chart_drop_idx", nothing)
chart_keep_idx = get(results, "chart_keep_idx", nothing)
chart_eta_keep_ref = get(results, "chart_eta_keep_ref", nothing)

n_params = length(θ_MLE)
β1_idx, K1_idx = 7, 10

println("Grid: $GRID × $GRID")
println("Mode: $mode_str")

# === RECONSTRUCT TRANSFORMATIONS ===
function ψ_to_θ(ψ)
    exp.(A_T_final' \ log.(ψ))
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

# === TRANSFORM TO THETA-SPACE ===
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

    function ψ_targets_to_θ1(ψ_t1, ψ_t2, ψ_ref, target_idx)
        log_interest = log.([ψ_t1, ψ_t2])
        η_drop = Cj_inv * (log_interest - Ck * η_keep_ref)
        η = copy(θ_log_ref)
        η[keep_idx] = η_keep_ref
        η[drop_idx] = η_drop
        θ = exp.(η)
        return θ[β1_idx], θ[K1_idx]
    end
else
    function ψ_targets_to_θ1(ψ_t1, ψ_t2, ψ_ref, target_idx)
        ψ_full = copy(ψ_ref)
        ψ_full[target_idx[1]] = ψ_t1
        ψ_full[target_idx[2]] = ψ_t2
        θ = ψ_to_θ(ψ_full)
        return θ[β1_idx], θ[K1_idx]
    end
end

# Build scatter arrays for both spaces
ψ1_all = Float64[]
ψ2_all = Float64[]
β_all = Float64[]
K_all = Float64[]
like_all = Float64[]

for (i, ψ_t1) in enumerate(ψ_target1_grid)
    for (j, ψ_t2) in enumerate(ψ_target2_grid)
        push!(ψ1_all, ψ_t1)
        push!(ψ2_all, ψ_t2)
        β, K = ψ_targets_to_θ1(ψ_t1, ψ_t2, ψ_MLE, target_2d)
        push!(β_all, β)
        push!(K_all, K)
        push!(like_all, like_matrix[i, j])
    end
end

println("\nCoverage:")
println("  ψ₁ (K₁/β₁): [$(round(minimum(ψ1_all), digits=1)), $(round(maximum(ψ1_all), digits=1))]")
println("  ψ₂ (β₁·K₁): [$(round(minimum(ψ2_all), digits=2)), $(round(maximum(ψ2_all), digits=1))]")
println("  β₁: [$(round(minimum(β_all), digits=4)), $(round(maximum(β_all), digits=4))]")
println("  K₁: [$(round(minimum(K_all), digits=1)), $(round(maximum(K_all), digits=1))]")

# === PLOTTING ===
println("\nGenerating scatter plots...")
gr(size=(1200, 900))

ψ_target1_true = ψ_MLE[target_2d[1]]
ψ_target2_true = ψ_MLE[target_2d[2]]

subtitle = if N_NUISANCE == 0
    "(slice: other params fixed at MLE)"
elseif N_NUISANCE == 16
    "(16 nuisance params profiled)"
else
    "($N_NUISANCE profiled, $(16-N_NUISANCE) fixed at MLE)"
end

# Marker size based on grid density
ms = GRID <= 20 ? 6 : (GRID <= 50 ? 4 : 2)

# Plot 1: Scatter in ψ-space (log scale on both axes)
p1 = scatter(ψ1_all, ψ2_all, zcolor=like_all, c=:dense, ms=ms, msw=0,
             xlabel="ψ_$(target_2d[1]) = K₁/β₁ (identifiable)",
             ylabel="ψ_$(target_2d[2]) = β₁·K₁ (non-identifiable)",
             title="Profile in IIR coordinates\n$subtitle",
             xscale=:log10, yscale=:log10, clims=(0,1), label="", colorbar=true)
scatter!([ψ_target1_true], [ψ_target2_true], mc=:darkgoldenrod, msc=:match, ms=10,
         markershape=:star, label="MLE")

# Plot 2: Scatter in θ-space (same display crop as replot_profile_results.jl)
β_plot_max = 0.6
K_plot_max = 400.0
in_bounds = (β_all .<= β_plot_max) .& (K_all .<= K_plot_max)

p2 = scatter(β_all[in_bounds], K_all[in_bounds], zcolor=like_all[in_bounds],
             c=:dense, ms=ms, msw=0,
             xlabel="β₁", ylabel="K₁",
             title="Profile in θ-space\n$subtitle",
             xlims=(0, β_plot_max), ylims=(0, K_plot_max),
             clims=(0,1), label="", colorbar=true)
scatter!([θ_MLE[β1_idx]], [θ_MLE[K1_idx]], mc=:darkgoldenrod, msc=:match, ms=10,
         markershape=:star, label="MLE")

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
