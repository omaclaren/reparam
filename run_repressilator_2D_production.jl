# ======================================================================
# REPRESSILATOR: 10x10 2D PROFILE (using 3x3 settings)
# ======================================================================
# Uses Distributed.jl for parallelization. All settings match the original
# 3×3 example except the grid resolution is increased to 10×10.

using Distributed
using Printf
using Serialization

# --------------------------------------------------------
# 1. Configuration (same as 3×3 example)
# --------------------------------------------------------
const N_WORKERS = 7               # use most available cores (8 total - 1 for main)
const GRID_2D = 10                # smaller grid
const PROFILE_TIMEOUT = 120.0     # even longer timeout for better optimization
const MLE_TIMEOUT = 120.0         # longer MLE timeout

println("="^70)
println("REPRESSILATOR: 10x10 2D PROFILE (using 3×3 settings)")
println("  Workers: $N_WORKERS")
println("  Grid: $GRID_2D x $GRID_2D")
println("  Profile timeout per point: $PROFILE_TIMEOUT s")
println("  MLE timeout: $MLE_TIMEOUT s")
println("="^70)

# --------------------------------------------------------
# 2. Setup Distributed Environment
# --------------------------------------------------------
println("\n[1/5] Setting up distributed environment...\n")
if length(workers()) < N_WORKERS
    addprocs(N_WORKERS - length(workers()))
end
println("✓ Workers: ", workers())

# Load modules locally first
if !@isdefined(RepressilatorModel)
    include(joinpath(@__DIR__, "examples", "RepressilatorModel.jl"))
end
using .RepressilatorModel

if !@isdefined(ReparamTools)
    include(joinpath(@__DIR__, "ReparamTools.jl"))
end
using .ReparamTools

# Load code on all workers
@everywhere begin
    import Pkg
    Pkg.activate(".")
    using Distributed, LinearAlgebra, Distributions, Random
    # Load local modules
    if !@isdefined(RepressilatorModel)
        include(joinpath(@__DIR__, "examples", "RepressilatorModel.jl"))
    end
    using .RepressilatorModel
    if !@isdefined(ReparamTools)
        include(joinpath(@__DIR__, "ReparamTools.jl"))
    end
    using .ReparamTools
end

# --------------------------------------------------------
# 3. Model and Data Setup (matching repressilator.jl)
# --------------------------------------------------------
println("\n[2/5] Setting up model and data...\n")

# Time grid (same as example)
T_end = 10000.0
NT = 8
t_obs = LinRange(0, T_end, NT)
X0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
σ = 1.0

# True parameters (same as example)
α₀_true = [0.008, 0.009, 0.010]
α_true   = [1.0, 1.2, 1.5]
β_true   = [0.02, 0.025, 0.015]
K_true   = [30.0, 28.0, 32.0]
k_degm_true = [0.006, 0.0055, 0.0065]
k_degp_true = [0.0012, 0.0011, 0.0013]
θ_true = vcat(α₀_true, α_true, β_true, K_true, k_degm_true, k_degp_true)
θ_log_true = log.(θ_true)

# Synthetic data (same seed as 3×3 example)
Random.seed!(42)
y_true = RepressilatorModel.predict_mRNA(θ_true, t_obs, X0)
data = y_true + σ * randn(length(y_true))

# Broadcast data to workers
@everywhere t_obs_global = $(t_obs)
@everywhere X0_global = $(X0)
@everywhere σ_global = $(σ)
@everywhere data_global = $(data)

@everywhere function lnlike_θ_worker(θ)
    try
        pred = RepressilatorModel.predict_mRNA(θ, t_obs_global, X0_global)
        dist = MvNormal(pred, σ_global^2 * I(length(pred)))
        return logpdf(dist, data_global)
    catch
        return -Inf
    end
end

@everywhere lnlike_θ_log_worker(θ_log) = lnlike_θ_worker(exp.(θ_log))

# --------------------------------------------------------
# 4. Bounds Setup
# --------------------------------------------------------
# Target parameters (β₁=7, K₁=10): ±2 in log space - defines the grid range
# Nuisance parameters: ±4 in log space - allows more flexibility to compensate
target_2d = [7, 10]                     # β₁ and K₁ indices
nuisance_2d = setdiff(1:18, target_2d)

θ_log_lower = copy(θ_log_true)
θ_log_upper = copy(θ_log_true)

# Target bounds: ±2 (same grid range as before)
θ_log_lower[target_2d] .-= 2.0
θ_log_upper[target_2d] .+= 2.0

# Nuisance bounds: ±4 (wider to avoid truncation)
θ_log_lower[nuisance_2d] .-= 4.0
θ_log_upper[nuisance_2d] .+= 4.0

# --------------------------------------------------------
# 5. Find MLE
# --------------------------------------------------------
println("\n[3/5] Finding MLE (starting point)...\n")
# Use true parameters as initial guess
θ_log_MLE, lnlike_MLE = ReparamTools.profile_target(
    lnlike_θ_log_worker,
    Int[],               # empty target → MLE
    θ_log_lower,
    θ_log_upper,
    θ_log_true;          # start at true values
    optmaxtime=MLE_TIMEOUT
)
println(@sprintf("✓ MLE found: % .4f", lnlike_MLE))

# --------------------------------------------------------
# 6. Run 2D profile (β₁, K₁) on a 10×10 grid
# --------------------------------------------------------
println("\n[4/5] Running 2D profile (β₁, K₁)...\n")

nuisance_guess_2d = θ_log_MLE[nuisance_2d]

n_points = GRID_2D^2
est_min = (n_points / N_WORKERS) * PROFILE_TIMEOUT / 60
println(@sprintf("  Grid: %dx%d = %d points", GRID_2D, GRID_2D, n_points))
println(@sprintf("  Estimated time: %.1f min (upper bound)", est_min))

t_start = time()
θ_2d_vals, ll_2d_vals = ReparamTools.profile_target(
    lnlike_θ_log_worker,
    target_2d,
    θ_log_lower,
    θ_log_upper,
    nuisance_guess_2d;
    grid_steps=GRID_2D,
    use_distributed=true,
    optmaxtime=PROFILE_TIMEOUT
)
elapsed = time() - t_start
println(@sprintf("✓ Profiling complete in %.1f s", elapsed))

# --------------------------------------------------------
# 7. Save results
# --------------------------------------------------------
println("\n[5/5] Saving results...\n")
serialize("repressilator_2D_intermediate_data.jls", (θ_2d_vals, ll_2d_vals, θ_true, θ_log_MLE))
println("✓ Data saved to repressilator_2D_intermediate_data.jls")

println("\nDONE.")
