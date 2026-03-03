# Compare Wald (quadratic) profile likelihood with hybrid and slice
# Wald uses Schur complement: H_profile = H_II - H_IN H_NN⁺ H_NI

using Pkg; Pkg.activate(".")
using Serialization, LinearAlgebra, ForwardDiff, Plots, Distributions

# Load hybrid results to get MLE and setup
println("Loading results for MLE and setup...")
results = deserialize("repressilator_hybrid_10x10_results.jls")
ψ_MLE = results["ψ_MLE"]
A_T_final = results["A_T_final"]
target_2d = results["target_2d"]
ψ_lower = results["ψ_lower"]
ψ_upper = results["ψ_upper"]
GRID = 100  # Use fine grid for Wald (it's cheap)

# Setup transformations
include("examples/RepressilatorModel.jl")
ψ_to_θ(ψ) = exp.(A_T_final' \ log.(ψ))

# Load data setup (same as main script)
NT = 8
t_obs = range(0, 40, length=NT)
X0 = [0.1, 0.0, 0.0, 0.1, 0.0, 0.0]
σ = 0.5

# Generate synthetic data at MLE
θ_MLE = ψ_to_θ(ψ_MLE)
pred_MLE = RepressilatorModel.predict_mRNA(θ_MLE, collect(t_obs), X0)
using Random; Random.seed!(42)
data = pred_MLE + σ * randn(length(pred_MLE))

# Log-likelihood in log-ψ space
function lnlike_ψ_log(ψ_log)
    try
        ψ = exp.(ψ_log)
        θ = ψ_to_θ(ψ)
        if any(θ .<= 0) || any(!isfinite, θ)
            return -Inf
        end
        pred = RepressilatorModel.predict_mRNA(θ, collect(t_obs), X0)
        if any(!isfinite, pred)
            return -Inf
        end
        return -0.5 * sum((data .- pred).^2) / σ^2
    catch
        return -Inf
    end
end

ψ_log_MLE = log.(ψ_MLE)

# Compute full Hessian
println("Computing Hessian at MLE...")
H_full = -ForwardDiff.hessian(lnlike_ψ_log, ψ_log_MLE)
H_full = 0.5 * (H_full + H_full')

# Partition
n_params = length(ψ_MLE)
interest_idx = target_2d
nuisance_idx = setdiff(1:n_params, target_2d)

H_II = H_full[interest_idx, interest_idx]
H_IN = H_full[interest_idx, nuisance_idx]
H_NI = H_full[nuisance_idx, interest_idx]
H_NN = H_full[nuisance_idx, nuisance_idx]

# Eigendecompose H_NN
eigen_NN = eigen(Symmetric(H_NN))
λ_NN = eigen_NN.values
U_NN = eigen_NN.vectors

println("\nH_NN eigenvalue spectrum:")
println("  Min: $(minimum(λ_NN))")
println("  Max: $(maximum(λ_NN))")
n_neg = sum(λ_NN .< 0)
n_pos = sum(λ_NN .> 1e-8 * maximum(abs.(λ_NN)))
println("  Negative: $n_neg, Positive: $n_pos, Near-zero: $(length(λ_NN) - n_neg - n_pos)")

# Pseudoinverse using positive eigenvalues
λ_pos = max.(λ_NN, 0.0)
rtol = 1e-8
mask = λ_pos .> rtol * maximum(λ_pos)
U_r = U_NN[:, mask]
Λ_r = λ_pos[mask]
H_NN_pinv = U_r * Diagonal(1.0 ./ Λ_r) * U_r'

# Schur complement: H_profile = H_II - H_IN H_NN⁺ H_NI
H_profile = H_II - H_IN * H_NN_pinv * H_NI

println("\nProfile Hessian (Schur complement):")
println("  H_profile = ")
display(H_profile)

# Eigendecompose H_profile
eigen_prof = eigen(Symmetric(H_profile))
println("\nH_profile eigenvalues: $(eigen_prof.values)")
println("  (Zero eigenvalue = non-identifiable direction)")

# Wald profile likelihood on grid
println("\nComputing Wald profile on $GRID × $GRID grid...")
ψ_log_lower = log.(ψ_lower)
ψ_log_upper = log.(ψ_upper)
target1_grid = range(ψ_log_lower[target_2d[1]], ψ_log_upper[target_2d[1]], length=GRID)
target2_grid = range(ψ_log_lower[target_2d[2]], ψ_log_upper[target_2d[2]], length=GRID)

ψ_log_I_MLE = ψ_log_MLE[interest_idx]
ll_MLE = lnlike_ψ_log(ψ_log_MLE)

ll_wald = zeros(GRID, GRID)
for (i, ψ1) in enumerate(target1_grid)
    for (j, ψ2) in enumerate(target2_grid)
        δψ_I = [ψ1, ψ2] - ψ_log_I_MLE
        # Quadratic approximation: ℓ ≈ ℓ_MLE - 0.5 * δψᵀ H_profile δψ
        ll_wald[i, j] = ll_MLE - 0.5 * dot(δψ_I, H_profile * δψ_I)
    end
end

# Convert to profile likelihood ratio
ll_max = maximum(ll_wald)
like_wald = exp.(ll_wald .- ll_max)

# 1D profiles (marginalize by taking max)
like_wald_1 = [maximum(like_wald[i, :]) for i in 1:GRID]
like_wald_2 = [maximum(like_wald[:, j]) for j in 1:GRID]

# Threshold
lstar = exp(-quantile(Chisq(1), 0.95)/2)
n_above_1 = sum(like_wald_1 .> lstar)
n_above_2 = sum(like_wald_2 .> lstar)

println("\n" * "="^60)
println("WALD PROFILE RESULTS")
println("="^60)
println("  K₁/β₁ (identifiable): $n_above_1/$GRID above 95% threshold")
println("  β₁·K₁ (non-identifiable): $n_above_2/$GRID above 95% threshold")

# Load other results for comparison
println("\nLoading other results for comparison...")
slice_results = deserialize("repressilator_0nuisance_100x100_results.jls")
hybrid_results = deserialize("repressilator_hybrid_100x100_results.jls")
full_results = deserialize("nesi/repressilator_16nuisance_50x50_results.jls")

# Extract 1D profiles from stored results
function extract_1d_profiles(res)
    ll = res["ll_vals"]
    g = res["GRID"]
    ll_mat = reshape(ll, g, g)
    ll_max = maximum(ll_mat[isfinite.(ll_mat)])
    like_mat = exp.(ll_mat .- ll_max)
    like_1 = [maximum(like_mat[i, :]) for i in 1:g]
    like_2 = [maximum(like_mat[:, j]) for j in 1:g]
    return like_1, like_2
end

function get_grid(res)
    g = res["GRID"]
    target = res["target_2d"]
    grid1 = range(log(res["ψ_lower"][target[1]]), log(res["ψ_upper"][target[1]]), length=g)
    grid2 = range(log(res["ψ_lower"][target[2]]), log(res["ψ_upper"][target[2]]), length=g)
    return grid1, grid2
end

slice_1, slice_2 = extract_1d_profiles(slice_results)
hybrid_1, hybrid_2 = extract_1d_profiles(hybrid_results)
full_1, full_2 = extract_1d_profiles(full_results)
full_grid1, full_grid2 = get_grid(full_results)

# Get grids from stored results
slice_grid1 = range(log(slice_results["ψ_lower"][slice_results["target_2d"][1]]),
                    log(slice_results["ψ_upper"][slice_results["target_2d"][1]]),
                    length=slice_results["GRID"])
slice_grid2 = range(log(slice_results["ψ_lower"][slice_results["target_2d"][2]]),
                    log(slice_results["ψ_upper"][slice_results["target_2d"][2]]),
                    length=slice_results["GRID"])

println("\nComparison (above 95% threshold):")
println("  Full:   K₁/β₁ = $(sum(full_1 .> lstar))/50, β₁·K₁ = $(sum(full_2 .> lstar))/50")
println("  Wald:   K₁/β₁ = $n_above_1/100, β₁·K₁ = $n_above_2/100")
println("  Hybrid: K₁/β₁ = $(sum(hybrid_1 .> lstar))/100, β₁·K₁ = $(sum(hybrid_2 .> lstar))/100")
println("  Slice:  K₁/β₁ = $(sum(slice_1 .> lstar))/100, β₁·K₁ = $(sum(slice_2 .> lstar))/100")

# Fixed plotting bounds (matching replot_profile_results.jl)
ψ1_plot_min, ψ1_plot_max = 1e1, 1e5      # K₁/β₁ range
ψ2_plot_min, ψ2_plot_max = 0.0, 8.0      # β₁·K₁ range

# 2D threshold
lstar_2d = exp(-quantile(Chisq(2), 0.95)/2)

# === 1D PROFILE COMPARISON ===
println("\nGenerating 1D profile comparison...")
p1d = plot(layout=(1,2), size=(1200, 400), margin=5Plots.mm)

# K₁/β₁ profile (identifiable)
plot!(p1d[1], exp.(collect(full_grid1)), full_1, label="Full (16 nuisance)", lw=3, color=:black)
plot!(p1d[1], exp.(collect(target1_grid)), like_wald_1, label="Wald (quadratic)", lw=2, color=:blue)
plot!(p1d[1], exp.(collect(slice_grid1)), hybrid_1, label="Hybrid (linear path)", lw=2, color=:green, ls=:dash)
plot!(p1d[1], exp.(collect(slice_grid1)), slice_1, label="Slice (fixed)", lw=2, color=:orange, ls=:dot)
hline!(p1d[1], [lstar], color=:red, ls=:dash, label="95% threshold", lw=1)
vline!(p1d[1], [exp(ψ_log_I_MLE[1])], color=:gray, ls=:dot, label="MLE", lw=1)
xlabel!(p1d[1], "K₁/β₁")
ylabel!(p1d[1], "Profile Likelihood")
title!(p1d[1], "K₁/β₁ (IDENTIFIABLE)")
plot!(p1d[1], xscale=:log10, xlims=(ψ1_plot_min, ψ1_plot_max), ylims=(0, 1.05), legend=:topright)

# β₁·K₁ profile (non-identifiable)
plot!(p1d[2], exp.(collect(full_grid2)), full_2, label="Full (16 nuisance)", lw=3, color=:black)
plot!(p1d[2], exp.(collect(target2_grid)), like_wald_2, label="Wald (quadratic)", lw=2, color=:blue)
plot!(p1d[2], exp.(collect(slice_grid2)), hybrid_2, label="Hybrid (linear path)", lw=2, color=:green, ls=:dash)
plot!(p1d[2], exp.(collect(slice_grid2)), slice_2, label="Slice (fixed)", lw=2, color=:orange, ls=:dot)
hline!(p1d[2], [lstar], color=:red, ls=:dash, label="95% threshold", lw=1)
vline!(p1d[2], [exp(ψ_log_I_MLE[2])], color=:gray, ls=:dot, label="MLE", lw=1)
xlabel!(p1d[2], "β₁·K₁")
ylabel!(p1d[2], "Profile Likelihood")
title!(p1d[2], "β₁·K₁ (NON-IDENTIFIABLE)")
plot!(p1d[2], xlims=(ψ2_plot_min, ψ2_plot_max), ylims=(0, 1.05), legend=:topright)

savefig(p1d, "wald_profile_comparison_1d.png")
println("Saved: wald_profile_comparison_1d.png")

# === 2D PROFILE COMPARISON ===
println("\nGenerating 2D profile comparison...")

# Wald 2D grid (already computed as ll_wald)
ψ1_grid = exp.(collect(target1_grid))
ψ2_grid = exp.(collect(target2_grid))

# Get slice and hybrid 2D grids
slice_ll = slice_results["ll_vals"]
slice_g = slice_results["GRID"]
slice_ll_mat = reshape(slice_ll, slice_g, slice_g)
slice_ll_max = maximum(slice_ll_mat[isfinite.(slice_ll_mat)])
slice_like_mat = exp.(slice_ll_mat .- slice_ll_max)

hybrid_ll = hybrid_results["ll_vals"]
hybrid_g = hybrid_results["GRID"]
hybrid_ll_mat = reshape(hybrid_ll, hybrid_g, hybrid_g)
hybrid_ll_max = maximum(hybrid_ll_mat[isfinite.(hybrid_ll_mat)])
hybrid_like_mat = exp.(hybrid_ll_mat .- hybrid_ll_max)

# Slice/hybrid grids
slice_ψ1_grid = exp.(collect(slice_grid1))
slice_ψ2_grid = exp.(collect(slice_grid2))

# MLE location
ψ1_MLE = exp(ψ_log_I_MLE[1])
ψ2_MLE = exp(ψ_log_I_MLE[2])

gr(size=(1400, 400))
p2d = plot(layout=(1,3), margin=5Plots.mm)

# Wald 2D
contourf!(p2d[1], ψ1_grid, ψ2_grid, like_wald', color=:dense, levels=20, lw=0,
          xlabel="K₁/β₁", ylabel="β₁·K₁", title="Wald (quadratic)",
          xscale=:log10, xlims=(ψ1_plot_min, ψ1_plot_max), ylims=(ψ2_plot_min, ψ2_plot_max),
          clims=(0,1))
scatter!(p2d[1], [ψ1_MLE], [ψ2_MLE], mc=:darkgoldenrod, msc=:match, ms=10,
         markershape=:star, label="MLE")
contour!(p2d[1], ψ1_grid, ψ2_grid, like_wald', levels=[lstar_2d], color=:black, lw=2,
         xscale=:log10, label="")

# Slice 2D
contourf!(p2d[2], slice_ψ1_grid, slice_ψ2_grid, slice_like_mat', color=:dense, levels=20, lw=0,
          xlabel="K₁/β₁", ylabel="β₁·K₁", title="Slice (fixed nuisance)",
          xscale=:log10, xlims=(ψ1_plot_min, ψ1_plot_max), ylims=(ψ2_plot_min, ψ2_plot_max),
          clims=(0,1))
scatter!(p2d[2], [ψ1_MLE], [ψ2_MLE], mc=:darkgoldenrod, msc=:match, ms=10,
         markershape=:star, label="MLE")
contour!(p2d[2], slice_ψ1_grid, slice_ψ2_grid, slice_like_mat', levels=[lstar_2d], color=:black, lw=2,
         xscale=:log10, label="")

# Hybrid 2D
contourf!(p2d[3], slice_ψ1_grid, slice_ψ2_grid, hybrid_like_mat', color=:dense, levels=20, lw=0,
          xlabel="K₁/β₁", ylabel="β₁·K₁", title="Hybrid (linear path)",
          xscale=:log10, xlims=(ψ1_plot_min, ψ1_plot_max), ylims=(ψ2_plot_min, ψ2_plot_max),
          clims=(0,1))
scatter!(p2d[3], [ψ1_MLE], [ψ2_MLE], mc=:darkgoldenrod, msc=:match, ms=10,
         markershape=:star, label="MLE")
contour!(p2d[3], slice_ψ1_grid, slice_ψ2_grid, hybrid_like_mat', levels=[lstar_2d], color=:black, lw=2,
         xscale=:log10, label="")

savefig(p2d, "wald_slice_hybrid_comparison_2d.png")
println("Saved: wald_slice_hybrid_comparison_2d.png")
