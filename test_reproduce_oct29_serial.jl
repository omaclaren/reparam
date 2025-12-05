# Reproduce Oct 29 "Good" Plot - SERIAL BASELINE
# Exact settings from prompt_20251029_002945.md:
# - EXACT parameters (NT=8, T_end=10000.0)
# - 7×7 grid (49 points)
# - Should show clean diagonal ridge structure

using Printf

println("="^70)
println("REPRODUCING OCT 29 REPRESSILATOR 2D PROFILE (SERIAL BASELINE)")
println("Settings: EXACT params + 7×7 grid")
println("="^70)

# Configuration - EXACT FROM OCT 29
GRID_2D = 7  # 7×7 = 49 points (exact Oct 29 setting)
MLE_TIMEOUT = 30.0
PROFILE_TIMEOUT = 30.0

# Load modules
include("examples/RepressilatorModel.jl")
using .RepressilatorModel
include("ReparamTools.jl")
using .ReparamTools
using Distributions
using LinearAlgebra
using Random
using Statistics

println("\n[1/3] Setting up model...")

# EXACT PARAMETERS FROM examples/repressilator.jl
NT = 8
T_end = 10000.0
t_obs = LinRange(0, T_end, NT)
X0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
σ = 1.0

θ_true = [0.008, 0.009, 0.010,      # α₀
          1.0, 1.2, 1.5,             # α
          0.02, 0.025, 0.015,        # β
          30.0, 28.0, 32.0,          # K
          0.006, 0.0055, 0.0065,     # k_degm
          0.0012, 0.0011, 0.0013]    # k_degp

println("✓ Model configuration:")
println("  NT = ", NT, ", T_end = ", T_end)
println("  β₁ = ", θ_true[7], ", K₁ = ", θ_true[10])
println("  Ratio K₁/β₁ = ", θ_true[10]/θ_true[7])

# Generate data
println("\n[2/3] Generating synthetic data...")
Random.seed!(42)
y_true = RepressilatorModel.predict_mRNA(θ_true, t_obs, X0)
data = y_true + σ * randn(length(y_true))

# Define likelihood functions
function lnlike_θ(θ)
    try
        pred = RepressilatorModel.predict_mRNA(θ, t_obs, X0)
        dist = MvNormal(pred, σ^2 * I(length(pred)))
        return logpdf(dist, data)
    catch
        return -Inf
    end
end

lnlike_θ_log(θ_log) = lnlike_θ(exp.(θ_log))
lnlike_profile = lnlike_θ_log

# Verify likelihood at true parameters
ll_true = lnlike_θ(θ_true)
println("  Likelihood at true params: ", @sprintf("%.4f", ll_true))

# Parameter bounds
θ_log_true = log.(θ_true)
θ_log_lower = θ_log_true .- 2.0
θ_log_upper = θ_log_true .+ 2.0

# Find MLE
println("\n  Finding MLE...")
θ_log_MLE, lnlike_MLE = ReparamTools.profile_target(
    lnlike_profile,
    Int[],
    θ_log_lower, θ_log_upper,
    θ_log_true;
    grid_steps=Int[],
    use_distributed=false,
    optmaxtime=MLE_TIMEOUT
)

θ_MLE = exp.(θ_log_MLE)
println("✓ MLE: β₁ = ", @sprintf("%.6f", θ_MLE[7]), ", K₁ = ", @sprintf("%.4f", θ_MLE[10]))

# Run 2D profile
println("\n[3/3] Running 2D profile (β₁, K₁) with $(GRID_2D)×$(GRID_2D) grid...")
target_2d = [7, 10]
nuisance_2d = setdiff(1:18, target_2d)
nuisance_guess_2d = θ_log_MLE[nuisance_2d]

t_2d = @elapsed begin
    θ_2d_vals, ll_2d_vals = ReparamTools.profile_target(
        lnlike_profile, target_2d,
        θ_log_lower, θ_log_upper,
        nuisance_guess_2d;
        grid_steps=GRID_2D,
        use_distributed=false,
        optmaxtime=PROFILE_TIMEOUT
    )
end

println("✓ 2D profile complete in ", @sprintf("%.2f", t_2d), "s")
println("  Finite values: ", sum(isfinite.(ll_2d_vals)), "/", length(ll_2d_vals))
println("  Likelihood range: ", extrema(ll_2d_vals[isfinite.(ll_2d_vals)]))

# Generate plot
println("\n[PLOTTING] Generating figure...")
using Plots
using Distributions: Chisq, quantile

β1_values = unique([exp(θ[7]) for θ in θ_2d_vals])
K1_values = unique([exp(θ[10]) for θ in θ_2d_vals])

# Julia reshape is column-major, so this produces matrix[i,j] = value at (β1[i], K1[j])
ll_grid = reshape(ll_2d_vals, length(β1_values), length(K1_values))
like_grid = exp.(ll_grid)

df = 2
lstar = exp(-quantile(Chisq(df), 0.95)/2)

plt = contourf(β1_values, K1_values, like_grid,
              color=:dense, levels=20, lw=0,
              xlabel="β₁", ylabel="K₁",
              title="Repressilator 2D Profile: (β₁, K₁) [SERIAL BASELINE]",
              colorbar=false,
              size=(800, 700))

contour!(β1_values, K1_values, like_grid,
         levels=[lstar], color=:black, lw=2, legend=false, fill=false)

scatter!([exp(θ_log_MLE[7])], [exp(θ_log_MLE[10])],
         mc=:white, msc=:black, markersize=8, markershape=:circle,
         label="MLE")

scatter!([θ_true[7]], [θ_true[10]],
         mc=:darkgoldenrod, msc=:match, markersize=10,
         markershape=:star, label="True")

output_file = "repressilator_2D_SERIAL_BASELINE.png"
savefig(plt, output_file)
println("✓ Figure saved: ", output_file)# filepath: /Users/omac010/Git-Working/reparam/test_reproduce_oct29_serial.jl
# Reproduce Oct 29 "Good" Plot - SERIAL BASELINE
# Exact settings from prompt_20251029_002945.md:
# - EXACT parameters (NT=8, T_end=10000.0)
# - 7×7 grid (49 points)
# - Should show clean diagonal ridge structure

