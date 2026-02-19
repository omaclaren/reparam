# 18-param Parallel Profiling Runner
# Following the stat_model distributed pattern

using Distributed

println("="^70)
println("18-PARAM PROFILING WITH DISTRIBUTED")
println("="^70)

# Add workers FIRST before any other imports
n_workers = 4
println("\n[1/5] Adding $n_workers workers...")
addprocs(n_workers)
println("Workers: ", workers())

# Load on MAIN first
println("\n[2/5] Loading modules...")
include(joinpath(@__DIR__, "ReparamTools.jl"))
include(joinpath(@__DIR__, "examples/RepressilatorModel.jl"))
using .ReparamTools
using .RepressilatorModel
using Distributions, LinearAlgebra, Random, ForwardDiff

# Then load on workers
@everywhere begin
    include($(joinpath(@__DIR__, "ReparamTools.jl")))
    include($(joinpath(@__DIR__, "examples/RepressilatorModel.jl")))
    using .ReparamTools
    using .RepressilatorModel
    using Distributions, LinearAlgebra, Random
end
println("Modules loaded on main and $(nworkers()) workers")

# === MODEL SETUP ===
println("\n[3/5] Setting up model...")
Random.seed!(42)
NT, T_end = 8, 10000.0
t_obs = LinRange(0, T_end, NT)
X0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
σ = 10.0

θ_true = [0.008, 0.009, 0.010, 1.0, 1.2, 1.5, 0.02, 0.025, 0.015,
          30.0, 28.0, 32.0, 0.006, 0.0055, 0.0065, 0.0012, 0.0011, 0.0013]

y_true = RepressilatorModel.predict_mRNA(θ_true, t_obs, X0)
data = y_true + σ * randn(length(y_true))

# Parameter bounds
θ_lower = [0.005, 0.005, 0.005, 0.8, 0.8, 0.8, 0.01, 0.01, 0.01,
           20.0, 20.0, 20.0, 0.004, 0.004, 0.004, 0.001, 0.001, 0.001]
θ_upper = [0.015, 0.015, 0.015, 2.0, 2.0, 2.0, 0.03, 0.03, 0.03,
           40.0, 40.0, 40.0, 0.008, 0.008, 0.008, 0.0015, 0.0015, 0.0015]
θ_log_lower, θ_log_upper = log.(θ_lower), log.(θ_upper)

# Likelihood
function lnlike_θ(θ)
    pred = RepressilatorModel.predict_mRNA(θ, t_obs, X0)
    dist = MvNormal(pred, σ^2 * I(3*NT))
    return logpdf(dist, data)
end
lnlike_θ_log(θ_log) = lnlike_θ(exp.(θ_log))

println("Model setup complete")
println("  Data points: $(length(data))")
println("  Parameters: 18")

# Find MLE
println("\nFinding MLE...")
nuisance_guesses = ReparamTools.generate_initial_guesses(θ_log_lower, θ_log_upper, 3)
t_mle = @elapsed begin
    θ_log_MLE, lnlike_MLE = ReparamTools.profile_target(lnlike_θ_log, Int[], θ_log_lower, θ_log_upper,
                                            nuisance_guesses[1]; grid_steps=Int[],
                                            ω_initial_extras=nuisance_guesses[2:end],
                                            method=:LN_BOBYQA, optmaxtime=30.0)
end
θ_MLE = exp.(θ_log_MLE)
println("MLE found in $(round(t_mle, digits=1))s, ll=$(round(lnlike_MLE, digits=2))")

# === RUN IIR ===
println("\n[4/5] Running IIR analysis...")

# High-precision ϕ in log-space (matches sequential script)
t_iir = LinRange(0, T_end, 501)
function ϕ_iir_highprec(θ)
    sol_matrix = RepressilatorModel.solve_repressilator(t_iir, θ, X0;
                                                         abstol=1e-10, reltol=1e-8)
    mRNA = sol_matrix[1:3, :]
    return vec(mRNA)
end
ϕ_iir_log(θ_log) = ϕ_iir_highprec(exp.(θ_log))

# Use rtol_rank=1e-7 to respect the large singular value gap (matches sequential script)
S, N, N_perp, rank_J = ReparamTools.find_invariant_subspace(ϕ_iir_log, θ_log_MLE;
    rtol_rank=1e-7, verbose=true)
println("Rank: $rank_J/18, Identifiable: $(size(N_perp, 2)), Non-ident: $(size(N, 2))")

# Build transformation
N_perp_rot = ReparamTools.varimax_rotation(N_perp; n_restarts=200)
N_rot = size(N, 2) > 0 ? ReparamTools.varimax_rotation(N; n_restarts=100) : N
A_T_final = hcat(ReparamTools.scale_and_round(N_perp_rot), ReparamTools.scale_and_round(N_rot))
θ_to_ψ, ψ_to_θ = ReparamTools.reparam(A_T_final)
ψ_MLE = θ_to_ψ(θ_MLE)
println("Transformation built, det=$(round(det(A_T_final), digits=3))")

# Find target coordinates (K₁/β₁ identifiable, β₁·K₁ non-identifiable)
target_2d = Int[]
for (i, col) in enumerate(eachcol(A_T_final))
    if abs(col[7]) ≈ 1 && abs(col[10]) ≈ 1 && sum(abs.(col)) ≈ 2
        push!(target_2d, i)
    end
end
if length(target_2d) != 2
    target_2d = [6, 17]  # Fallback from previous successful run
end
println("Target coordinates: ψ_$(target_2d[1]), ψ_$(target_2d[2])")

# Compute ψ bounds
n_samples = 10000
ψ_samples = [θ_to_ψ(exp.(θ_log_lower + rand(18) .* (θ_log_upper - θ_log_lower))) for _ in 1:n_samples]
ψ_matrix = hcat(ψ_samples...)
ψ_lower, ψ_upper = vec(minimum(ψ_matrix, dims=2)), vec(maximum(ψ_matrix, dims=2))
ψ_log_lower, ψ_log_upper = log.(ψ_lower), log.(ψ_upper)

# === DISTRIBUTED SETUP ===
println("\n[5/5] Setting up distributed profiling...")

# Send data to workers
@everywhere A_T_global = $A_T_final
@everywhere data_global = $data
@everywhere t_obs_global = $(collect(t_obs))
@everywhere X0_global = $X0
@everywhere σ_global = $σ
@everywhere NT_global = $NT

# Define worker likelihood function
@everywhere function lnlike_ψ_log_worker(ψ_log)
    try
        ψ = exp.(ψ_log)
        θ = exp.(A_T_global' \ log.(ψ))
        if any(θ .<= 0) || any(!isfinite, θ)
            return -Inf
        end
        pred = RepressilatorModel.predict_mRNA(θ, t_obs_global, X0_global)
        if any(!isfinite, pred)
            return -Inf
        end
        dist = MvNormal(pred, σ_global^2 * I(3*NT_global))
        return logpdf(dist, data_global)
    catch
        return -Inf
    end
end

# Verify worker function
ψ_log_MLE = log.(ψ_MLE)
@everywhere ψ_log_MLE_test = $ψ_log_MLE
ll_worker = @fetchfrom workers()[1] lnlike_ψ_log_worker(ψ_log_MLE_test)
println("Worker test: ll=$(round(ll_worker, digits=2)) (should be $(round(lnlike_MLE, digits=2)))")

# Set up profiling
nuisance_2d = setdiff(1:18, target_2d)
nuisance_log_guess = ψ_log_MLE[nuisance_2d]
nuisance_extras = [nuisance_log_guess .+ 0.1 .* randn(16) for _ in 1:3]

GRID = 10  # Start with 10x10 test; change to 50 for production
println("\n" * "="^70)
println("RUNNING $(GRID)×$(GRID) = $(GRID^2) DISTRIBUTED PROFILE")
println("="^70)
println("Target: ψ_$(target_2d[1]), ψ_$(target_2d[2])")
println("Nuisance: 16 parameters")
println("Workers: $n_workers")
flush(stdout)

t_profile = @elapsed begin
    ψ_vals, ll_vals = ReparamTools.profile_target(
        lnlike_ψ_log_worker, target_2d, ψ_log_lower, ψ_log_upper, nuisance_log_guess;
        grid_steps=GRID, use_distributed=true,
        ω_initial_extras=nuisance_extras,
        method=:LN_BOBYQA, optmaxtime=60.0
    )
end

n_finite = sum(isfinite.(ll_vals))
println("\nDone in $(round(t_profile/60, digits=1)) minutes")
println("Finite values: $n_finite/$(length(ll_vals))")

# === RESULTS ===
using Plots, Contour, ScatteredInterpolation
gr()

ll_matrix = reshape(ll_vals, GRID, GRID)
ll_max = maximum(ll_matrix[isfinite.(ll_matrix)])
like_matrix = exp.(ll_matrix .- ll_max)
lstar_2d = exp(-quantile(Chisq(2), 0.95)/2)
lstar_1d = exp(-quantile(Chisq(1), 0.95)/2)

# 1D profiles
ψ_target1_grid = exp.(range(ψ_log_lower[target_2d[1]], ψ_log_upper[target_2d[1]], length=GRID))
ψ_target2_grid = exp.(range(ψ_log_lower[target_2d[2]], ψ_log_upper[target_2d[2]], length=GRID))
like_target1 = [maximum(like_matrix[i, :]) for i in 1:GRID]
like_target2 = [maximum(like_matrix[:, j]) for j in 1:GRID]

above1 = sum(like_target1 .>= lstar_1d)
above2 = sum(like_target2 .>= lstar_1d)
println("\n1D Profile: ψ_$(target_2d[1]) $(above1)/$GRID above 95%, ψ_$(target_2d[2]) $(above2)/$GRID above 95%")

# Plot
p1 = contourf(ψ_target1_grid, ψ_target2_grid, like_matrix', color=:dense, levels=20,
              xlabel="ψ_$(target_2d[1]) (identifiable)", ylabel="ψ_$(target_2d[2]) (non-identifiable)",
              title="2D Profile (IIR coords)", xscale=:log10)
contour!(ψ_target1_grid, ψ_target2_grid, like_matrix', levels=[lstar_2d], color=:black, lw=2)
scatter!([ψ_MLE[target_2d[1]]], [ψ_MLE[target_2d[2]]], color=:gold, ms=8, label="MLE")

p3 = plot(ψ_target1_grid, like_target1, lw=2, xscale=:log10,
          xlabel="ψ_$(target_2d[1])", ylabel="Profile Likelihood", title="Identifiable")
hline!([lstar_1d], ls=:dash, color=:red, label="95%")
vline!([ψ_MLE[target_2d[1]]], ls=:dot, color=:green, label="MLE")

p4 = plot(ψ_target2_grid, like_target2, lw=2,
          xlabel="ψ_$(target_2d[2])", ylabel="Profile Likelihood", title="Non-identifiable")
hline!([lstar_1d], ls=:dash, color=:red, label="95%")
vline!([ψ_MLE[target_2d[2]]], ls=:dot, color=:green, label="MLE")

# θ-space with RBF - handle invalid transformations
θ_grid = Vector{Float64}[]
for k in 1:length(ψ_vals)
    try
        θ = ψ_to_θ(exp.(ψ_vals[k]))
        if all(θ .> 0) && all(isfinite.(θ))
            push!(θ_grid, θ)
        else
            push!(θ_grid, fill(NaN, 18))
        end
    catch
        push!(θ_grid, fill(NaN, 18))
    end
end
β_vals_grid = [θ[7] for θ in θ_grid]
K_vals_grid = [θ[10] for θ in θ_grid]
like_flat = exp.(ll_vals .- ll_max)
valid = isfinite.(like_flat) .& (like_flat .> 0) .& isfinite.(β_vals_grid) .& isfinite.(K_vals_grid)

β_filt, K_filt, like_filt = β_vals_grid[valid], K_vals_grid[valid], like_flat[valid]
println("Valid points for θ-space plot: $(sum(valid))/$(length(valid))")
β_min, β_max = extrema(β_filt)
K_min, K_max = extrema(K_filt)
β_norm = (β_filt .- β_min) ./ (β_max - β_min)
K_norm = (K_filt .- K_min) ./ (K_max - K_min)
itp = interpolate(ThinPlate(), hcat(β_norm, K_norm)', like_filt)

β_reg = range(β_min, β_max, length=100)
K_reg = range(K_min, K_max, length=100)
like_θ_reg = [evaluate(itp, [(β - β_min)/(β_max - β_min), (K - K_min)/(K_max - K_min)])[1] for β in β_reg, K in K_reg]
like_θ_reg = clamp.(like_θ_reg, 0.0, 1.0)

p2 = contourf(collect(β_reg), collect(K_reg), like_θ_reg', color=:dense, levels=20,
              xlabel="β₁", ylabel="K₁", title="Profile in θ-space (RBF)")
contour!(collect(β_reg), collect(K_reg), like_θ_reg', levels=[lstar_2d], color=:black, lw=2)
scatter!([θ_MLE[7]], [θ_MLE[10]], color=:gold, ms=8, label="MLE")

p_final = plot(p1, p2, p3, p4, layout=(2,2), size=(1000, 800))
savefig(p_final, "iir_18param_parallel_result.png")
println("\nSaved: iir_18param_parallel_result.png")

println("\n" * "="^70)
println("RESULTS:")
println("="^70)
println("  Identifiable (ψ_$(target_2d[1])): $(above1)/$GRID above threshold - $(above1 < GRID/2 ? "PEAKED" : "FLAT")")
println("  Non-identifiable (ψ_$(target_2d[2])): $(above2)/$GRID above threshold - $(above2 >= GRID/2 ? "FLAT" : "PEAKED")")

if n_finite == length(ll_vals)
    println("\nSUCCESS: All $(length(ll_vals)) points converged")
else
    println("\nPARTIAL: $n_finite/$(length(ll_vals)) points converged")
end

rmprocs(workers())
println("\nDone.")
