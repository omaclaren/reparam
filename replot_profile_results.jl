# Replot profile likelihood results from saved .jls files
# Generates 4-panel figure: IIR coords, θ-space, and 1D profiles
#
# Usage: julia replot_profile_results.jl <results.jls> [output.png]

using Serialization
using Plots
using Distributions
using LinearAlgebra
using Contour

# === LOAD RESULTS ===
if length(ARGS) < 1
    error("Usage: julia replot_profile_results.jl <results.jls> [output.png]")
end

input_file = ARGS[1]
output_file = length(ARGS) >= 2 ? ARGS[2] : replace(input_file, ".jls" => "_replot.png")

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

n_params = length(θ_MLE)

println("Grid: $GRID × $GRID")
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

# === HELPER: Transform ψ targets to θ-space ===
function ψ_targets_to_θ1(ψ_t1, ψ_t2, ψ_ref, target_idx, ψ_to_θ_func)
    ψ_full = copy(ψ_ref)
    ψ_full[target_idx[1]] = ψ_t1
    ψ_full[target_idx[2]] = ψ_t2
    θ = ψ_to_θ_func(ψ_full)
    return θ[7], θ[10]  # β₁, K₁
end

# === PLOTTING ===
println("\nGenerating plots...")
gr(size=(1200, 900))

ψ_target1_true = ψ_MLE[target_2d[1]]
ψ_target2_true = ψ_MLE[target_2d[2]]

# Plot 1: 2D profile in ψ-space
p1 = contourf(ψ_target1_grid, ψ_target2_grid, like_matrix', color=:dense, levels=20, lw=0,
              xlabel="ψ_$(target_2d[1]) = K₁/β₁ (identifiable)",
              ylabel="ψ_$(target_2d[2]) = β₁·K₁ (non-identifiable)",
              title="2D Profile in IIR coordinates\n($(n_params - 2) nuisance params profiled out)",
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
        β, K = ψ_targets_to_θ1(ψ_t1, ψ_t2, ψ_MLE, target_2d, ψ_to_θ)
        push!(β_flat, β)
        push!(K_flat, K)
        push!(like_flat, like_matrix[i, j])
    end
end

# θ-space bounds for plotting (use data range with margin)
β_min, β_max = extrema(β_flat)
K_min, K_max = extrema(K_flat)
β_margin = 0.1 * (β_max - β_min)
K_margin = 0.1 * (K_max - K_min)

# Interpolate to regular grid for contourf
β_reg = range(β_min, β_max, length=100)
K_reg = range(K_min, K_max, length=100)

# Simple nearest-neighbor interpolation for θ-space
like_θ_reg = fill(NaN, length(β_reg), length(K_reg))
for (idx, (β_val, K_val, like_val)) in enumerate(zip(β_flat, K_flat, like_flat))
    i_β = argmin(abs.(collect(β_reg) .- β_val))
    i_K = argmin(abs.(collect(K_reg) .- K_val))
    if isnan(like_θ_reg[i_β, i_K]) || like_val > like_θ_reg[i_β, i_K]
        like_θ_reg[i_β, i_K] = like_val
    end
end

# Fill NaNs with nearest valid value (simple approach)
for i in 1:length(β_reg), j in 1:length(K_reg)
    if isnan(like_θ_reg[i, j])
        like_θ_reg[i, j] = 0.0
    end
end
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
    for (pt1, pt2) in zip(ψ_t1_c, ψ_t2_c)
        β, K = ψ_targets_to_θ1(pt1, pt2, ψ_MLE, target_2d, ψ_to_θ)
        push!(β_c, β)
        push!(K_c, K)
    end
    plot!(p2, β_c, K_c, color=:black, lw=2, label="")
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
