# Minimal 2D Profile - IIR Coordinates with External Nuisance
# Profile over ψ₁ = K₁/β₁ (identifiable target)
# With ψ₂ = β₁·K₁ (non-identifiable) and β₂ (external nuisance) optimized

if !@isdefined(ReparamTools)
    include("ReparamTools.jl")
end
if !@isdefined(RepressilatorModel)
    include("examples/RepressilatorModel.jl")
end

using .ReparamTools
using .RepressilatorModel
using Distributions, LinearAlgebra, Random, ForwardDiff

# === PLOTTING OPTIONS ===
USE_SCATTER_PLOT = false  # true = scatter plot, false = RBF interpolated contour

# === MODEL SETUP ===
Random.seed!(42)
NT, T_end = 4, 10000.0
t_obs = LinRange(0, T_end, NT)
X0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
σ = 10.0

# True parameters (18 total)
θ_true = [0.008, 0.009, 0.010,      # α₀ (1-3)
          1.0, 1.2, 1.5,             # α  (4-6)
          0.02, 0.025, 0.015,        # β  (7-9)
          30.0, 28.0, 32.0,          # K  (10-12)
          0.006, 0.0055, 0.0065,     # k_degm (13-15)
          0.0012, 0.0011, 0.0013]    # k_degp (16-18)

# Generate data
y_true = RepressilatorModel.predict_mRNA(θ_true, t_obs, X0)
data = y_true + σ * randn(length(y_true))

# True values
β1_true = θ_true[7]   # 0.02
K1_true = θ_true[10]  # 30.0
β2_true = θ_true[8]   # 0.025 (external nuisance)

# IIR-derived coordinates for gene 1
ψ1_true = K1_true / β1_true  # 1500 (identifiable)
ψ2_true = β1_true * K1_true  # 0.6 (non-identifiable)

println("=" ^ 70)
println("MINIMAL 2D PROFILE - IIR COORDS + EXTERNAL NUISANCE")
println("=" ^ 70)
println("\nTrue values:")
println("  Gene 1: β₁ = $β1_true, K₁ = $K1_true")
println("  Gene 2: β₂ = $β2_true (external nuisance)")
println("  ψ₁ = K₁/β₁ = $ψ1_true (identifiable)")
println("  ψ₂ = β₁·K₁ = $ψ2_true (non-identifiable)")

# === REPARAMETERIZATION ===
# For gene 1: θ = [β₁, K₁] → ψ = [K₁/β₁, β₁·K₁]
A_T = [-1.0  1.0;    # col 1: ψ₁ = K/β
        1.0  1.0]    # col 2: ψ₂ = β·K

θ_to_ψ, ψ_to_θ = reparam(A_T)

# === BOUNDS ===
# 3D parameter space: [ψ₁, ψ₂, β₂] in log space
# To cover θ-space: β ∈ [0.004, 0.1], K ∈ [1, 200]
#   ψ₁ = K/β needs: [10, 50000]
#   ψ₂ = β·K needs: [0.004, 20]

lower = [log(10.0),     # ψ₁ ≥ 10
         log(0.004),    # ψ₂ ≥ 0.004
         log(0.005)]    # β₂ down to 0.005

upper = [log(50000.0),  # ψ₁ ≤ 50000
         log(20.0),     # ψ₂ ≤ 20
         log(0.12)]     # β₂ up to 0.12

println("\nBounds:")
println("  ψ₁ = K/β:  [$(round(exp(lower[1]), sigdigits=3)), $(round(exp(upper[1]), sigdigits=3))]")
println("  ψ₂ = β·K:  [$(round(exp(lower[2]), sigdigits=3)), $(round(exp(upper[2]), sigdigits=3))]")
println("  β₂:        [$(round(exp(lower[3]), sigdigits=3)), $(round(exp(upper[3]), sigdigits=3))]")

# === LIKELIHOOD ===
# Free: β₁, β₂, K₁ (indices 7, 8, 10)
# Fixed: everything else
fixed_indices = setdiff(1:18, [7, 8, 10])
θ_fixed = θ_true[fixed_indices]

function lnlike_3param_log(p_log)
    # p_log = [log(ψ₁), log(ψ₂), log(β₂)]
    ψ1, ψ2, β2 = exp.(p_log)
    
    # Transform ψ → θ for gene 1
    θ_gene1 = ψ_to_θ([ψ1, ψ2])  # [β₁, K₁]
    β1, K1 = θ_gene1
    
    # Check positivity
    if β1 <= 0 || K1 <= 0 || β2 <= 0
        return -Inf
    end
    
    # Reconstruct full parameter vector
    θ_full = zeros(18)
    θ_full[fixed_indices] = θ_fixed
    θ_full[7] = β1
    θ_full[8] = β2
    θ_full[10] = K1
    
    try
        pred = RepressilatorModel.predict_mRNA(θ_full, t_obs, X0)
        return logpdf(MvNormal(pred, σ^2 * I(length(pred))), data)
    catch
        return -Inf
    end
end

# === SCENARIO 1: Profile over ψ₁ only (optimize over ψ₂ and β₂) ===
println("\n" * "=" ^ 70)
println("SCENARIO 1: 1D Profile over ψ₁ = K/β")
println("            (ψ₂ = β·K and β₂ as nuisance)")
println("=" ^ 70)

target_1 = [1]  # Profile over ψ₁ only
nuisance_guess_1 = [log(ψ2_true), log(β2_true)]
GRID_1D = 60

t_start = time()
ψ1_vals_1d, ll_vals_1d = profile_target(lnlike_3param_log, target_1, lower, upper, nuisance_guess_1;
                                         grid_steps=GRID_1D, use_distributed=false,
                                         method=:LN_BOBYQA, optmaxtime=30.0)
elapsed = time() - t_start

println("Done in $(round(elapsed, digits=1)) seconds")
println("Finite values: $(sum(isfinite.(ll_vals_1d)))/$(length(ll_vals_1d))")

# Extract grid
ψ1_grid_1d = exp.(range(lower[1], upper[1], length=GRID_1D))
ll_max_1d = maximum(ll_vals_1d[isfinite.(ll_vals_1d)])
like_1d = exp.(ll_vals_1d .- ll_max_1d)

lstar_1d = exp(-quantile(Chisq(1), 0.95)/2)
println("95% CI threshold: $(round(lstar_1d, digits=3))")

# Find CI bounds
above_thresh = like_1d .> lstar_1d
ci_indices = findall(above_thresh)
if !isempty(ci_indices)
    ci_lower = ψ1_grid_1d[first(ci_indices)]
    ci_upper = ψ1_grid_1d[last(ci_indices)]
    println("95% CI for ψ₁ = K/β: [$(round(ci_lower, sigdigits=3)), $(round(ci_upper, sigdigits=3))]")
    println("True value: $ψ1_true")
else
    println("WARNING: Profile never drops below threshold - check bounds")
end

# === SCENARIO 2: 2D Profile over (ψ₁, ψ₂) with β₂ as nuisance ===
println("\n" * "=" ^ 70)
println("SCENARIO 2: 2D Profile over (ψ₁, ψ₂)")
println("            (β₂ as nuisance)")
println("=" ^ 70)

target_2 = [1, 2]  # Profile over ψ₁ and ψ₂
nuisance_guess_2 = [log(β2_true)]
GRID_2D = 50

t_start = time()
ψ12_vals_2d, ll_vals_2d = profile_target(lnlike_3param_log, target_2, lower, upper, nuisance_guess_2;
                                          grid_steps=GRID_2D, use_distributed=false,
                                          method=:LN_BOBYQA, optmaxtime=30.0)
elapsed = time() - t_start

println("Done in $(round(elapsed, digits=1)) seconds")
println("Finite values: $(sum(isfinite.(ll_vals_2d)))/$(length(ll_vals_2d))")

# === PLOTS ===
using Plots
using Contour
using ScatteredInterpolation

# 2D profile plots - grids
ψ1_grid_2d = exp.(range(lower[1], upper[1], length=GRID_2D))
ψ2_grid_2d = exp.(range(lower[2], upper[2], length=GRID_2D))

ll_matrix_2d = reshape(ll_vals_2d, GRID_2D, GRID_2D)
ll_max_2d = maximum(ll_matrix_2d[isfinite.(ll_matrix_2d)])
like_matrix_2d = exp.(ll_matrix_2d .- ll_max_2d)

lstar_2d = exp(-quantile(Chisq(2), 0.95)/2)

# Plot 1: 2D profile in ψ-space (top-left)
p1 = contourf(ψ1_grid_2d, ψ2_grid_2d, like_matrix_2d', color=:dense, levels=20, lw=0,
              xlabel="ψ₁ = K/β (identifiable)", ylabel="ψ₂ = β·K (non-identifiable)",
              title="2D Profile in IIR coordinates\n(β₂ optimized)",
              xscale=:log10, clims=(0,1))
scatter!([ψ1_true], [ψ2_true], mc=:darkgoldenrod, msc=:match, ms=10, markershape=:star, label="True")
contour!(ψ1_grid_2d, ψ2_grid_2d, like_matrix_2d', levels=[lstar_2d], color=:black, lw=2, 
         xscale=:log10, label="95% CI")

# Plot 2: Transform to θ-space (top-right)
β_flat = Float64[]
K_flat = Float64[]
like_flat = Float64[]

for (i, ψ1) in enumerate(ψ1_grid_2d)
    for (j, ψ2) in enumerate(ψ2_grid_2d)
        θ = ψ_to_θ([ψ1, ψ2])
        push!(β_flat, θ[1])
        push!(K_flat, θ[2])
        push!(like_flat, like_matrix_2d[i, j])
    end
end

# Compute axis limits for θ-space plot based on ψ₂ constraint
# Fixed rectangular region for θ-space plots (matching IIR_coords.jl)
β_plot_min, β_plot_max = 0.004, 0.09
K_plot_min, K_plot_max = 1.0, 200.0

if USE_SCATTER_PLOT
    # Scatter plot version with constrained axes
    p2 = scatter(β_flat, K_flat, zcolor=like_flat, c=:dense,
                 xlabel="β₁", ylabel="K₁", title="2D Profile in θ-space (scatter)\n(β₂ optimized)",
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
                 xlabel="β₁", ylabel="K₁", title="2D Profile in θ-space\n(β₂ optimized)",
                 xlims=(β_min, β_max), ylims=(K_min, K_max), clims=(0,1))
    scatter!([β1_true], [K1_true], mc=:darkgoldenrod, msc=:match, ms=10, markershape=:star, label="True")
end

# Transform CI contour to θ-space, only plotting within rectangular region
c = Contour.contour(collect(ψ1_grid_2d), collect(ψ2_grid_2d), like_matrix_2d, lstar_2d)
for line in Contour.lines(c)
    ψ1_c, ψ2_c = Contour.coordinates(line)
    β_c = Float64[]
    K_c = Float64[]
    for k in 1:length(ψ1_c)
        θ_k = ψ_to_θ([ψ1_c[k], ψ2_c[k]])
        # Only include points within the rectangular plot region
        if β_plot_min <= θ_k[1] <= β_plot_max && K_plot_min <= θ_k[2] <= K_plot_max
            push!(β_c, θ_k[1])
            push!(K_c, θ_k[2])
        end
    end
    if length(β_c) > 1
        plot!(p2, β_c, K_c, color=:black, lw=2, label="")
    end
end

# === 1D PROFILES (bottom row) ===
like_ψ1_proj = [maximum(like_matrix_2d[i, :]) for i in 1:GRID_2D]
like_ψ2_proj = [maximum(like_matrix_2d[:, j]) for j in 1:GRID_2D]

# Plot 3: 1D profile for ψ₁ = K/β (bottom-left)
p3 = plot(ψ1_grid_2d, like_ψ1_proj, xlabel="ψ₁ = K/β", ylabel="Profile Likelihood",
          title="Profile: K/β (IDENTIFIABLE)", linewidth=2, legend=false, 
          xscale=:log10, ylims=(0, 1.05))
hline!([lstar_1d], color=:red, linestyle=:dash, linewidth=2)
vline!([ψ1_true], color=:green, linestyle=:dot, linewidth=2)

# Plot 4: 1D profile for ψ₂ = β·K (bottom-right)
p4 = plot(ψ2_grid_2d, like_ψ2_proj, xlabel="ψ₂ = β·K", ylabel="Profile Likelihood",
          title="Profile: β·K (NON-IDENTIFIABLE)", linewidth=2, legend=false,
          ylims=(0, 1.05))
hline!([lstar_1d], color=:red, linestyle=:dash, linewidth=2)
vline!([ψ2_true], color=:green, linestyle=:dot, linewidth=2)

# Combined plot
plt = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 1000))
savefig(plt, "minimal_2D_IIR_nuisance_result.png")
println("\nSaved: minimal_2D_IIR_nuisance_result.png")

# === SUMMARY ===
println("\n" * "=" ^ 70)
println("SUMMARY")
println("=" ^ 70)

# Profile projections
println("\n1D Profile projections from 2D grid:")
println("  Profile(ψ₁ = K/β): range [$(round(minimum(like_ψ1_proj), digits=4)), $(round(maximum(like_ψ1_proj), digits=4))]")
println("  Profile(ψ₂ = β·K): range [$(round(minimum(like_ψ2_proj), digits=4)), $(round(maximum(like_ψ2_proj), digits=4))]")

# Check flatness
ψ1_drops = maximum(like_ψ1_proj) - minimum(like_ψ1_proj)
ψ2_drops = maximum(like_ψ2_proj) - minimum(like_ψ2_proj)
println("\nProfile drop (max - min):")
println("  ψ₁ = K/β: $(round(ψ1_drops, digits=4)) → ", ψ1_drops > 0.5 ? "PEAKED (identifiable)" : "FLAT (non-identifiable)")
println("  ψ₂ = β·K: $(round(ψ2_drops, digits=4)) → ", ψ2_drops > 0.5 ? "PEAKED (identifiable)" : "FLAT (non-identifiable)")

println("""

INTERPRETATION:
- ψ₁ = K/β should have a peaked profile (identifiable from IIR)
- ψ₂ = β·K should have a flat profile (non-identifiable from IIR)
- This confirms the IIR-derived structure before profiling
""")
