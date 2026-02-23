# SUPERSEDED — orphaned inline copy, never used by any NeSI submit script.
# The active NeSI script is run_repressilator_profile.jl (called by nesi/submit_50x50.sl etc).
# This file has stale profiling bounds and no snake_direction/chunking support.
#
# Original description:
# NeSI runner for 100x100 IIR profiling
# Designed to run on Mahuika with 36 cores

using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Distributed

# Get number of CPUs from SLURM (or default to 36)
n_cpus = parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "36"))
n_workers = n_cpus - 1  # Leave 1 for main process

println("Setting up $(n_workers) workers...")
addprocs(n_workers)
println("Workers ready: $(nworkers())")

# Load modules on all workers
const PROJECT_DIR = pwd()

@everywhere begin
  using Pkg
  Pkg.activate(".")
end

@everywhere PROJECT_DIR = $PROJECT_DIR
@everywhere include(joinpath(PROJECT_DIR, "ReparamTools.jl"))
@everywhere include(joinpath(PROJECT_DIR, "examples/RepressilatorModel.jl"))

@everywhere begin
  using .ReparamTools
  using .RepressilatorModel
  using Distributions, LinearAlgebra, Random, ForwardDiff
end
  
println("Modules loaded on all workers")

# Now run the main script with settings for 100x100
# We'll inline the key parts with NeSI-optimized settings

Random.seed!(42)

# === MODEL SETUP ===
NT, T_end = 8, 10000.0
t_obs = LinRange(0, T_end, NT)
X0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
σ = 10.0

θ_true = [0.008, 0.009, 0.010,      # α₀ (1-3)
          1.0, 1.2, 1.5,             # α  (4-6)
          0.02, 0.025, 0.015,        # β  (7-9)
          30.0, 28.0, 32.0,          # K  (10-12)
          0.006, 0.0055, 0.0065,     # k_degm (13-15)
          0.0012, 0.0011, 0.0013]    # k_degp (16-18)

param_names = [
    "α₀₁", "α₀₂", "α₀₃", "α₁", "α₂", "α₃",
    "β₁", "β₂", "β₃", "K₁", "K₂", "K₃",
    "k_degm₁", "k_degm₂", "k_degm₃", "k_degp₁", "k_degp₂", "k_degp₃"
]

n_params = 18

y_true = RepressilatorModel.predict_mRNA(θ_true, t_obs, X0)
data = y_true + σ * randn(length(y_true))

println("\n" * "=" ^ 70)
println("IIR 100x100 PROFILING ON NESI")
println("=" ^ 70)
println("Workers: $(nworkers())")
println("Grid: 100 × 100 = 10000 points")
println("Parameters: $n_params")

# === BOUNDS ===
θ_lower = [0.005, 0.005, 0.005, 0.8, 0.8, 0.8, 0.01, 0.01, 0.01,
           20.0, 20.0, 20.0, 0.004, 0.004, 0.004, 0.001, 0.001, 0.001]
θ_upper = [0.015, 0.015, 0.015, 2.0, 2.0, 2.0, 0.03, 0.03, 0.03,
           40.0, 40.0, 40.0, 0.008, 0.008, 0.008, 0.0015, 0.0015, 0.0015]
θ_log_lower = log.(θ_lower)
θ_log_upper = log.(θ_upper)

# === LIKELIHOOD ===
distrib_θ = θ -> MvNormal(RepressilatorModel.predict_mRNA(θ, t_obs, X0), σ^2 * I(3*NT))
lnlike_θ = ReparamTools.construct_lnlike_xy(distrib_θ, data; dist_type=:multi)
lnlike_θ_log = θ_log -> lnlike_θ(exp.(θ_log))

# === FIND MLE ===
println("\n" * "=" ^ 70)
println("FINDING MLE")
println("=" ^ 70)

θ_log_initial = 0.5 * (θ_log_lower + θ_log_upper)
n_guesses = 5  # More restarts for better MLE
nuisance_guesses = ReparamTools.generate_initial_guesses(θ_log_lower, θ_log_upper, n_guesses)

println("Running MLE optimization with $n_guesses restarts...")
flush(stdout)

t_mle_start = time()
θ_log_MLE, lnlike_MLE = ReparamTools.profile_target(
    lnlike_θ_log, Int[], θ_log_lower, θ_log_upper, nuisance_guesses[1];
    grid_steps=Int[], ω_initial_extras=nuisance_guesses[2:end],
    method=:LN_BOBYQA, optmaxtime=60.0)
t_mle_elapsed = time() - t_mle_start

θ_MLE = exp.(θ_log_MLE)
println("MLE found in $(round(t_mle_elapsed, digits=1)) seconds")
println("Log-likelihood at MLE: $(round(lnlike_MLE, digits=2))")

# === IIR ANALYSIS ===
println("\n" * "=" ^ 70)
println("RUNNING IIR ANALYSIS")
println("=" ^ 70)

t_iir = LinRange(0, T_end, 501)
function ϕ_iir_highprec(θ)
    sol_matrix = RepressilatorModel.solve_repressilator(t_iir, θ, X0; abstol=1e-10, reltol=1e-8)
    mRNA = sol_matrix[1:3, :]
    return vec(mRNA)
end
ϕ_iir_log(θ_log) = ϕ_iir_highprec(exp.(θ_log))

t_iir_start = time()
S, N, N_perp, rank_J = ReparamTools.find_invariant_subspace(
    ϕ_iir_log, θ_log_MLE; rtol_rank=1e-7, verbose=true)
println("IIR analysis completed in $(round(time() - t_iir_start, digits=1)) seconds")

n_ident = size(N_perp, 2)
n_nonident = size(N, 2)

println("\nRank: $rank_J / $n_params")
println("Identifiable: $n_ident, Non-identifiable: $n_nonident")

if n_nonident == 0
    error("No non-identifiable directions found!")
end

# === BUILD TRANSFORMATION ===
N_perp_varimax = ReparamTools.varimax_rotation(N_perp; n_restarts=200, threshold=1e-2)
N_perp_clean = ReparamTools.scale_and_round(N_perp_varimax; round_within=0.15)

# Fix K/β signs
for j in 1:n_ident
    v = N_perp_clean[:, j]
    for gene in 1:3
        β_idx = 6 + gene
        K_idx = 9 + gene
        β_coef = round(Int, v[β_idx])
        K_coef = round(Int, v[K_idx])
        other_sum = sum(abs.(round.(Int, v[[i for i in 1:n_params if i != β_idx && i != K_idx]])))
        if abs(β_coef) == 1 && abs(K_coef) == 1 && β_coef != K_coef && other_sum == 0
            if β_coef == 1 && K_coef == -1
                N_perp_clean[:, j] *= -1
            end
            break
        end
    end
end

N_varimax = ReparamTools.varimax_rotation(N; n_restarts=200, threshold=1e-2)
N_clean = ReparamTools.scale_and_round(N_varimax; round_within=0.15)

A_T_final = hcat(N_perp_clean, N_clean)

# DEBUG: Print transformation matrix structure
println("\n=== DEBUG: Transformation Matrix ===")
println("β₁ (idx 7) and K₁ (idx 10) coefficients in each column:")
for j in 1:size(A_T_final, 2)
  v = A_T_final[:, j]
  β1_coef = v[7]
  K1_coef = v[10]
  other_sum = sum(abs.(v[[i for i in 1:n_params if i != 7 && i != 10]]))
  ident_str = j <= n_ident ? "identifiable" : "non-identifiable"
  println("  ψ_$j ($ident_str): β₁=$(round(β1_coef, digits=2)), K₁=$(round(K1_coef, digits=2)), other=$(round(other_sum, digits=2))")
end
println("=====================================\n")
  
θ_to_ψ, ψ_to_θ = ReparamTools.reparam(A_T_final)
ψ_MLE = θ_to_ψ(θ_MLE)

# Find gene 1 coordinates - DEBUG VERSION
β1_idx, K1_idx = 7, 10
gene1_ident_idx = nothing
gene1_nonident_idx = nothing

println("DEBUG: Searching for K₁/β₁ in identifiable columns...")
for j in 1:n_ident
  v = A_T_final[:, j]
  k_rounded = round(Int, v[K1_idx])
  b_rounded = round(Int, v[β1_idx])
  other_sum = sum(abs.(round.(Int, v[[i for i in 1:n_params if i != β1_idx && i != K1_idx]])))
  println("  j=$j: K=$(k_rounded), β=$(b_rounded), other=$other_sum")
  if k_rounded == 1 && b_rounded == -1 && other_sum == 0
      global gene1_ident_idx = j
      println("  → MATCH! gene1_ident_idx = $j")
      break
  end
end

println("DEBUG: Searching for β₁K₁ in non-identifiable columns...")
for j in 1:n_nonident
  v = N_clean[:, j]
  k_rounded = round(Int, v[K1_idx])
  b_rounded = round(Int, v[β1_idx])
  other_sum = sum(abs.(round.(Int, v[[i for i in 1:n_params if i != β1_idx && i != K1_idx]])))
  println("  j=$j: K=$(k_rounded), β=$(b_rounded), other=$other_sum")
  if b_rounded == k_rounded && abs(b_rounded) >= 1 && other_sum == 0
      global gene1_nonident_idx = n_ident + j
      println("  → MATCH! gene1_nonident_idx = $(n_ident + j)")
      break
  end
end

println("\nGene 1 coordinates: ψ_$gene1_ident_idx (K₁/β₁), ψ_$gene1_nonident_idx (β₁·K₁)")

target_2d = [gene1_ident_idx, gene1_nonident_idx]
nuisance_2d = setdiff(1:n_params, target_2d)

# === ψ-SPACE BOUNDS ===
θ_lower_profile = [0.003, 0.003, 0.003, 0.5, 0.5, 0.5, 0.005, 0.005, 0.005,
                   10.0, 10.0, 10.0, 0.003, 0.003, 0.003, 0.0008, 0.0008, 0.0008]
θ_upper_profile = [0.020, 0.020, 0.020, 3.0, 3.0, 3.0, 0.08, 0.08, 0.08,
                   80.0, 80.0, 80.0, 0.010, 0.010, 0.010, 0.002, 0.002, 0.002]

function compute_ψ_bounds(θ_lo, θ_hi, θ_to_ψ_func, n_samples=10000)
    n = length(θ_lo)
    ψ_samples = [θ_to_ψ_func(θ_lo .+ rand(n) .* (θ_hi - θ_lo)) for _ in 1:n_samples]
    ψ_mat = hcat(ψ_samples...)
    return vec(minimum(ψ_mat, dims=2)), vec(maximum(ψ_mat, dims=2))
end

ψ_lower, ψ_upper = compute_ψ_bounds(θ_lower_profile, θ_upper_profile, θ_to_ψ, 10000)
ψ_log_lower = log.(ψ_lower)
ψ_log_upper = log.(ψ_upper)

# === DISTRIBUTED PROFILING ===
println("\n" * "=" ^ 70)
println("100x100 PROFILING")
println("=" ^ 70)

GRID = 50

@everywhere A_T_global = $A_T_final
@everywhere data_global = $data
@everywhere t_obs_global = $(collect(t_obs))
@everywhere X0_global = $X0
@everywhere σ_global = $σ
@everywhere NT_global = $NT

@everywhere function lnlike_18param_ψ_log_worker(ψ_log)
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

nuisance_log_lower = ψ_log_lower[nuisance_2d]
nuisance_log_upper = ψ_log_upper[nuisance_2d]
nuisance_log_guess = clamp.((nuisance_log_lower .+ nuisance_log_upper) ./ 2,
                            nuisance_log_lower .+ 1e-6, nuisance_log_upper .- 1e-6)

n_extra_guesses = 15  # More restarts for smoother results
nuisance_extras = ReparamTools.generate_initial_guesses(nuisance_log_lower, nuisance_log_upper, n_extra_guesses)

println("Grid: $GRID × $GRID = $(GRID^2) points")
println("Workers: $(nworkers())")
println("Extra restarts: $n_extra_guesses")
println("\nStarting profiling...")
flush(stdout)

t_profile_start = time()
ψ_vals, ll_vals = ReparamTools.profile_target(
    lnlike_18param_ψ_log_worker, target_2d, ψ_log_lower, ψ_log_upper, nuisance_log_guess;
    grid_steps=GRID, use_distributed=true,
    ω_initial_extras=nuisance_extras,
    method=:LN_BOBYQA, optmaxtime=150.0  # More time per point
)
t_profile_elapsed = time() - t_profile_start

println("\nDone in $(round(t_profile_elapsed/60, digits=1)) minutes")
println("Finite values: $(sum(isfinite.(ll_vals)))/$(length(ll_vals))")

# === SAVE RESULTS ===
using Serialization

results = Dict(
    "ψ_vals" => ψ_vals,
    "ll_vals" => ll_vals,
    "ψ_MLE" => ψ_MLE,
    "θ_MLE" => θ_MLE,
    "A_T_final" => A_T_final,
    "target_2d" => target_2d,
    "GRID" => GRID,
    "ψ_lower" => ψ_lower,
    "ψ_upper" => ψ_upper,
    "param_names" => param_names,
    "rank_J" => rank_J,
    "n_ident" => n_ident,
    "n_nonident" => n_nonident
)

serialize("iir_50x50_results.jls", results)
println("\nResults saved to iir_50x50_results.jls")

# === QUICK ANALYSIS ===
ll_matrix = reshape(ll_vals, GRID, GRID)
ll_max = maximum(ll_matrix[isfinite.(ll_matrix)])
like_matrix = exp.(ll_matrix .- ll_max)

using Distributions
lstar_1d = exp(-quantile(Chisq(1), 0.95)/2)

like_ψ_target1 = [maximum(like_matrix[i, :]) for i in 1:GRID]
like_ψ_target2 = [maximum(like_matrix[:, j]) for j in 1:GRID]

println("\n" * "=" ^ 70)
println("RESULTS SUMMARY")
println("=" ^ 70)
println("K₁/β₁ (identifiable): $(sum(like_ψ_target1 .> lstar_1d))/$GRID above 95% threshold")
println("β₁·K₁ (non-identifiable): $(sum(like_ψ_target2 .> lstar_1d))/$GRID above 95% threshold")

# === GENERATE PLOT ===
println("\nGenerating plot...")
using Plots

target1_log_grid = range(ψ_log_lower[target_2d[1]], ψ_log_upper[target_2d[1]], length=GRID)
target2_log_grid = range(ψ_log_lower[target_2d[2]], ψ_log_upper[target_2d[2]], length=GRID)
ψ_target1_grid = exp.(collect(target1_log_grid))
ψ_target2_grid = exp.(collect(target2_log_grid))

lstar_2d = exp(-quantile(Chisq(2), 0.95)/2)

p1 = contourf(ψ_target1_grid, ψ_target2_grid, like_matrix', color=:dense, levels=20, lw=0,
              xlabel="K₁/β₁ (identifiable)", ylabel="β₁·K₁ (non-identifiable)",
              title="100×100 Profile (NeSI)", xscale=:log10, clims=(0,1))
scatter!([ψ_MLE[target_2d[1]]], [ψ_MLE[target_2d[2]]], mc=:gold, ms=10, markershape=:star, label="MLE")
contour!(ψ_target1_grid, ψ_target2_grid, like_matrix', levels=[lstar_2d], color=:black, lw=2, xscale=:log10, label="")

p2 = plot(ψ_target1_grid, like_ψ_target1, xlabel="K₁/β₁", ylabel="Profile Likelihood",
          title="K₁/β₁ (IDENTIFIABLE)", lw=2, legend=false, xscale=:log10, ylims=(0,1.05))
hline!([lstar_1d], color=:red, ls=:dash, lw=2)

p3 = plot(ψ_target2_grid, like_ψ_target2, xlabel="β₁·K₁", ylabel="Profile Likelihood",
          title="β₁·K₁ (NON-IDENTIFIABLE)", lw=2, legend=false, ylims=(0,1.05))
hline!([lstar_1d], color=:red, ls=:dash, lw=2)

plt = plot(p1, p2, p3, layout=(1,3), size=(1500, 400))
savefig(plt, "iir_50x50_result.png")
println("Saved: iir_50x50_result.png")

println("\n" * "=" ^ 70)
println("JOB COMPLETE")
println("=" ^ 70)
