# Unified Repressilator Profile Likelihood Script
# Supports slice and profile modes (hybrid mode removed)
#
# Usage:
#   julia run_repressilator_profile.jl --nuisance=0              # Slice at MLE
#   julia run_repressilator_profile.jl --nuisance=16 --grid=50   # Full profile
#   julia run_repressilator_profile.jl --nuisance=16 --workers=7 # Parallel
#
# Modes:
#   slice (--nuisance=0 or --mode=slice): Fix all nuisance at MLE, evaluate likelihood on grid
#   profile (--nuisance>0 or --mode=profile): Optimize nuisance at each grid point
#
# On NeSI/SLURM, workers auto-detected from SLURM_CPUS_PER_TASK

using Pkg
Pkg.activate(".")
Pkg.instantiate()

# === PARSE ARGUMENTS ===
function parse_kv_arg(args, prefix)
    for arg in args
        if startswith(arg, prefix)
            value = arg[length(prefix)+1:end]
            isempty(value) && error("Missing value for argument prefix '$prefix'")
            return value
        end
    end
    return nothing
end

function parse_int_arg(args, prefix, default)
    value = parse_kv_arg(args, prefix)
    value === nothing && return default
    try
        return parse(Int, value)
    catch
        error("Invalid integer for argument '$prefix': '$value'")
    end
end

function parse_string_arg(args, prefix, default)
    value = parse_kv_arg(args, prefix)
    return value === nothing ? default : value
end

N_NUISANCE = parse_int_arg(ARGS, "--nuisance=", 16)  # Default: full profile
GRID = parse_int_arg(ARGS, "--grid=", 50)
MODE = parse_string_arg(ARGS, "--mode=", "auto")  # auto, slice, profile

# Determine actual mode
if MODE == "hybrid"
    error("--mode=hybrid has been removed. Use --nuisance=0 (slice) or --nuisance>0 (profile).")
elseif MODE == "slice"
    N_NUISANCE = 0
elseif MODE == "profile"
    # keep N_NUISANCE as provided
elseif MODE != "auto"
    error("Unknown --mode=$MODE. Allowed: auto, slice, profile")
end

# Validate nuisance count
if N_NUISANCE < 0 || N_NUISANCE > 16
    error("--nuisance must be between 0 and 16")
end

# === DISTRIBUTED SETUP ===
using Distributed

USE_DISTRIBUTED = N_NUISANCE > 0  # Need distributed for nuisance profiling

if USE_DISTRIBUTED
    if haskey(ENV, "SLURM_CPUS_PER_TASK")
        n_cpus = parse(Int, ENV["SLURM_CPUS_PER_TASK"])
        println("SLURM detected: $n_cpus CPUs")
    else
        n_cpus = parse_int_arg(ARGS, "--workers=", 7) + 1
        println("Local mode: $(n_cpus - 1) workers")
    end

    n_workers = n_cpus - 1
    if nworkers() == 1 && n_workers > 1
        addprocs(n_workers)
    end
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
else
    # Single-threaded for slice (no workers added)
    include(joinpath(@__DIR__, "ReparamTools.jl"))
    include(joinpath(@__DIR__, "examples/RepressilatorModel.jl"))
    using .ReparamTools
    using .RepressilatorModel
    using Distributions, LinearAlgebra, Random, ForwardDiff, Statistics
    println("Single-threaded mode (slice)")
end

# === MODE DESCRIPTION ===
mode_str = if N_NUISANCE == 0
    "SLICE (no nuisance, fixed at MLE)"
elseif N_NUISANCE == 16
    "FULL PROFILE (16 nuisance)"
else
    "PARTIAL PROFILE ($N_NUISANCE nuisance profiled, $(16 - N_NUISANCE) fixed at MLE)"
end

println("\n" * "=" ^ 70)
println("REPRESSILATOR PROFILE LIKELIHOOD")
println("=" ^ 70)
println("Mode: $mode_str")
println("Grid: $GRID × $GRID = $(GRID^2) points")
if USE_DISTRIBUTED
    println("Workers: $(nworkers())")
end

# === MODEL SETUP ===
Random.seed!(42)

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
β1_idx, K1_idx = 7, 10

# Generate data
y_true = RepressilatorModel.predict_mRNA(θ_true, t_obs, X0)
data = y_true + σ * randn(length(y_true))

# === PARAMETER BOUNDS ===
θ_lower = [0.005, 0.005, 0.005, 0.8, 0.8, 0.8, 0.01, 0.01, 0.01,
           20.0, 20.0, 20.0, 0.004, 0.004, 0.004, 0.001, 0.001, 0.001]
θ_upper = [0.015, 0.015, 0.015, 2.0, 2.0, 2.0, 0.03, 0.03, 0.03,
           40.0, 40.0, 40.0, 0.008, 0.008, 0.008, 0.0015, 0.0015, 0.0015]
θ_log_lower = log.(θ_lower)
θ_log_upper = log.(θ_upper)

# Wider bounds for profiling (to capture full uncertainty region)
θ_lower_profile = [0.003, 0.003, 0.003, 0.5, 0.5, 0.5, 0.002, 0.002, 0.002,
                   3.0, 3.0, 3.0, 0.003, 0.003, 0.003, 0.0008, 0.0008, 0.0008]
θ_upper_profile = [0.020, 0.020, 0.020, 3.0, 3.0, 3.0, 0.5, 0.5, 0.5,
                   100.0, 100.0, 100.0, 0.010, 0.010, 0.010, 0.002, 0.002, 0.002]
θ_log_lower_profile = log.(θ_lower_profile)
θ_log_upper_profile = log.(θ_upper_profile)

# === LIKELIHOOD FUNCTION ===
distrib_θ = θ -> MvNormal(RepressilatorModel.predict_mRNA(θ, t_obs, X0), σ^2 * I(3*NT))
lnlike_θ = ReparamTools.construct_lnlike_xy(distrib_θ, data; dist_type=:multi)
lnlike_θ_log = θ_log -> lnlike_θ(exp.(θ_log))

# === FIND MLE (always needed) ===
println("\n" * "=" ^ 70)
println("FINDING MLE")
println("=" ^ 70)

θ_log_initial = 0.5 * (θ_log_lower + θ_log_upper)
n_mle_guesses = 20
mle_guesses = ReparamTools.generate_initial_guesses(θ_log_lower, θ_log_upper, n_mle_guesses)

println("Running MLE optimization with $n_mle_guesses restarts...")
flush(stdout)

t_mle_start = time()
θ_log_MLE, lnlike_MLE = ReparamTools.profile_target(
    lnlike_θ_log, Int[], θ_log_lower, θ_log_upper, mle_guesses[1];
    grid_steps=Int[], ω_initial_extras=mle_guesses[2:end],
    method=:LN_BOBYQA, optmaxtime=150.0)  # Slightly more than original 120s
t_mle_elapsed = time() - t_mle_start

θ_MLE = exp.(θ_log_MLE)
println("MLE found in $(round(t_mle_elapsed, digits=1)) seconds")
println("Log-likelihood at MLE: $(round(lnlike_MLE, digits=2))")
println("β₁_MLE = $(round(θ_MLE[β1_idx], digits=4)), K₁_MLE = $(round(θ_MLE[K1_idx], digits=2))")

# === IIR ANALYSIS (always needed) ===
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
J_iir_log = ReparamTools.compute_ϕ_Jacobian(ϕ_iir_log, θ_log_MLE)
σ1_sq = svdvals(J_iir_log)[1]^2
residual_cap = 1e-2

identified_basis_result = ReparamTools.informed_monomial_basis_search(
    N_perp,
    J_iir_log' * J_iir_log,
    σ1_sq,
    param_names;
    s_max=2,
    c_max=1,
    residual_cap=residual_cap,
)

null_basis_result = ReparamTools.simple_search_with_support_retry(
    N,
    param_names;
    s_max=2,
    c_max=1,
    residual_cap=residual_cap,
)

identified_basis_result.basis_ok || error("Could not construct identifiable-side sparse basis for repressilator")
null_basis_result.basis_ok || error("Could not construct invariant-null sparse basis for repressilator")

N_perp_clean = ReparamTools.basis_candidate_matrix(identified_basis_result.selected, n_params)
N_clean = ReparamTools.basis_candidate_matrix(null_basis_result.selected, n_params)
N_perp_labels = ReparamTools.basis_labels(identified_basis_result.selected)
N_clean_labels = ReparamTools.basis_labels(null_basis_result.selected)

function monomial_label_from_column(v, param_names)
    vr = round.(Int, v)
    num = String[]
    den = String[]
    for (name, exp) in zip(param_names, vr)
        if exp > 0
            push!(num, exp == 1 ? name : string(name, "^", exp))
        elseif exp < 0
            nexp = -exp
            push!(den, nexp == 1 ? name : string(name, "^", nexp))
        end
    end
    num_str = isempty(num) ? "1" : join(num, "*")
    den_str = isempty(den) ? "" : join(den, "*")
    return isempty(den_str) ? num_str : string(num_str, "/(", den_str, ")")
end

# Fix K/β signs to get K/β (not β/K)
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

N_perp_labels = [monomial_label_from_column(N_perp_clean[:, j], param_names) for j in 1:n_ident]

println("\nSelected identifiable-side basis labels:")
for (j, label) in enumerate(N_perp_labels)
    println("  ψ_$j = ", label)
end

println("\nSelected invariant-null basis labels:")
for (j, label) in enumerate(N_clean_labels)
    println("  ψ_$(n_ident + j) = ", label)
end

A_T_final = hcat(N_perp_clean, N_clean)

θ_to_ψ, ψ_to_θ = ReparamTools.reparam(A_T_final)
ψ_MLE = θ_to_ψ(θ_MLE)

# === FIND TARGET COORDINATES (GENE 1) ===
println("\nIdentifying gene 1 coordinates...")

# Helper: rounded coefficient pattern for exactly two active indices
function two_term_pattern(v, idx_a, idx_b)
    vr = round.(Int, v)
    a = vr[idx_a]
    b = vr[idx_b]
    others = sum(abs, vr) - abs(a) - abs(b)
    return a, b, others
end

# Find K₁/β₁ (identifiable)
target_ident_idx = let idx = nothing
    for j in 1:n_ident
        k_coef, b_coef, others = two_term_pattern(A_T_final[:, j], K1_idx, β1_idx)
        if k_coef == 1 && b_coef == -1 && others == 0
            println("  ψ_$j = K₁/β₁ (identifiable)")
            idx = j
            break
        end
    end
    idx
end

# Find β₁·K₁ (non-identifiable)
target_nonident_idx = let idx = nothing
    for j in 1:n_nonident
        k_coef, b_coef, others = two_term_pattern(N_clean[:, j], K1_idx, β1_idx)
        if b_coef == k_coef && abs(b_coef) >= 1 && others == 0
            println("  ψ_$(n_ident + j) = β₁·K₁ (non-identifiable)")
            idx = n_ident + j
            break
        end
    end
    idx
end

if isnothing(target_ident_idx) || isnothing(target_nonident_idx)
    error("Could not identify gene 1 coordinates!")
end

target_2d = Int[target_ident_idx, target_nonident_idx]
println("\nTarget: ψ_$(target_2d[1]) (K₁/β₁), ψ_$(target_2d[2]) (β₁·K₁)")

# === PROFILE CHART: exact interest + original-coordinate complement ===
function compute_ψ_bounds(θ_lo, θ_hi, θ_to_ψ_func, n_samples=10000)
    n = length(θ_lo)
    ψ_samples = [θ_to_ψ_func(θ_lo .+ rand(n) .* (θ_hi - θ_lo)) for _ in 1:n_samples]
    ψ_mat = hcat(ψ_samples...)
    return vec(minimum(ψ_mat, dims=2)), vec(maximum(ψ_mat, dims=2))
end

function linear_image_bounds(C::AbstractMatrix{<:Real}, lower::AbstractVector{<:Real}, upper::AbstractVector{<:Real})
    m, p = size(C)
    lo = zeros(Float64, m)
    hi = zeros(Float64, m)
    for i in 1:m
        for j in 1:p
            coef = C[i, j]
            if coef >= 0
                lo[i] += coef * lower[j]
                hi[i] += coef * upper[j]
            else
                lo[i] += coef * upper[j]
                hi[i] += coef * lower[j]
            end
        end
    end
    return lo, hi
end

function choose_drop_indices(C::AbstractMatrix{<:Real})
    m, p = size(C)
    F = qr(Matrix(C), ColumnNorm())
    J = sort(Vector(F.p[1:m]))
    rank(Matrix(C[:, J])) == m || error("Could not find an invertible original-coordinate complement for the chosen interests")
    return J
end

C_interest = Matrix(transpose(A_T_final[:, target_2d]))
drop_idx = choose_drop_indices(C_interest)
keep_idx = setdiff(1:n_params, drop_idx)
Cj = Matrix(C_interest[:, drop_idx])
Ck = Matrix(C_interest[:, keep_idx])
Cj_inv = inv(Cj)
chart_decoupled = maximum(abs.(Ck)) < 1e-12
chart_eta_keep_ref = θ_log_MLE[keep_idx]

println("\nProfile chart: exact interest + original-coordinate complement")
println("  Dropped original coordinates: $(join(param_names[drop_idx], ", "))")
println("  Kept original coordinates: $(length(keep_idx))")
println("  Decoupled interest block: $(chart_decoupled)")

# Keep full ψ-space bounds for metadata / postprocessing, but reuse the legacy
# varimax-era 2D target window that previously gave acceptable repressilator profiles.
# Only the nuisance optimization chart is being changed here.
ψ_lower, ψ_upper = compute_ψ_bounds(θ_lower_profile, θ_upper_profile, θ_to_ψ, 10000)
ψ_log_lower = log.(ψ_lower)
ψ_log_upper = log.(ψ_upper)
legacy_target_lower = [6.5638417359811925, 0.019558192562089023]   # [K₁/β₁, β₁·K₁]
legacy_target_upper = [37504.4697128395, 49.11316897036603]        # [K₁/β₁, β₁·K₁]
interest_log_lower = log.(legacy_target_lower)
interest_log_upper = log.(legacy_target_upper)
ψ_log_lower[target_2d] = interest_log_lower
ψ_log_upper[target_2d] = interest_log_upper
ψ_lower[target_2d] = legacy_target_lower
ψ_upper[target_2d] = legacy_target_upper

# === SELECT NUISANCE PARAMETERS ===
# Profile in original log-parameters for the nuisance complement. The current gene-1
# interest pair decouples from the chosen complement, so the nuisance feasible set stays box-constrained.
all_other_params = keep_idx

if N_NUISANCE == 0
    nuisance_to_profile = Int[]
    fixed_at_mle = all_other_params
elseif N_NUISANCE >= length(all_other_params)
    nuisance_to_profile = all_other_params
    fixed_at_mle = Int[]
else
    nuisance_to_profile = all_other_params[1:N_NUISANCE]
    fixed_at_mle = all_other_params[N_NUISANCE+1:end]
end

if !chart_decoupled && N_NUISANCE > 0
    error("The chosen interest + original-coordinate complement is coupled. A general constrained nuisance optimizer is still needed for this case.")
end

function reconstruct_η(log_interest::AbstractVector{<:Real}, η_profiled::AbstractVector{<:Real})
    η = copy(θ_log_MLE)
    for (k, idx) in enumerate(nuisance_to_profile)
        η[idx] = η_profiled[k]
    end
    η_keep = η[keep_idx]
    η_drop = Cj_inv * (Float64.(log_interest) - Ck * η_keep)
    η[drop_idx] = η_drop
    if any(!isfinite, η) || any(η .< θ_log_lower_profile) || any(η .> θ_log_upper_profile)
        return nothing
    end
    return η
end

println("\n" * "=" ^ 70)
println("PROFILING SETUP")
println("=" ^ 70)
println("Mode: $mode_str")
println("Target chart: exact [K₁/β₁, β₁·K₁] interests + original-coordinate complement")
println("Target window: reused legacy varimax-era 2D range")
println("Profiled nuisance θ ($(length(nuisance_to_profile))): $(isempty(nuisance_to_profile) ? "none" : join(param_names[nuisance_to_profile], ", "))")
println("Fixed at MLE θ ($(length(fixed_at_mle))): $(isempty(fixed_at_mle) ? "none" : join(param_names[fixed_at_mle], ", "))")

# === BUILD GRID ===
target1_log_grid = range(interest_log_lower[1], interest_log_upper[1], length=GRID)
target2_log_grid = range(interest_log_lower[2], interest_log_upper[2], length=GRID)
ψ_target1_grid = exp.(collect(target1_log_grid))
ψ_target2_grid = exp.(collect(target2_log_grid))

println("\nGrid ranges:")
println("  K₁/β₁: [$(round(ψ_target1_grid[1], sigdigits=3)), $(round(ψ_target1_grid[end], sigdigits=3))]")
println("  β₁·K₁: [$(round(ψ_target2_grid[1], sigdigits=3)), $(round(ψ_target2_grid[end], sigdigits=3))]")

# === RUN PROFILING ===
println("\n" * "=" ^ 70)
if N_NUISANCE == 0
    println("COMPUTING SLICE (evaluating likelihood on grid)")
else
    println("COMPUTING PROFILE LIKELIHOOD")
end
println("=" ^ 70)

t_profile_start = time()

if N_NUISANCE == 0
    println("Evaluating $(GRID^2) grid points via profile_target...")
    flush(stdout)

    function lnlike_slice_interest(log_interest)
        try
            η = reconstruct_η(log_interest, Float64[])
            isnothing(η) && return -Inf
            θ = exp.(η)
            if any(θ .<= 0) || any(!isfinite, θ)
                return -Inf
            end
            return lnlike_θ(θ)
        catch
            return -Inf
        end
    end

    chart_vals_raw, ll_vals = ReparamTools.profile_target(
        lnlike_slice_interest, [1, 2], interest_log_lower, interest_log_upper, Float64[];
        grid_steps=GRID, use_distributed=false)
else
    if USE_DISTRIBUTED
        @everywhere data_global = $data
        @everywhere t_obs_global = $(collect(t_obs))
        @everywhere X0_global = $X0
        @everywhere σ_global = $σ
        @everywhere NT_global = $NT
        @everywhere θ_log_MLE_global = $θ_log_MLE
        @everywhere θ_log_lower_global = $θ_log_lower_profile
        @everywhere θ_log_upper_global = $θ_log_upper_profile
        @everywhere drop_idx_global = $drop_idx
        @everywhere keep_idx_global = $keep_idx
        @everywhere nuisance_param_indices_global = $nuisance_to_profile
        @everywhere Cj_inv_global = $(Matrix(Cj_inv))
        @everywhere Ck_global = $(Matrix(Ck))

        @everywhere function reconstruct_η_global(log_interest, η_profiled)
            η = copy(θ_log_MLE_global)
            for (k, idx) in enumerate(nuisance_param_indices_global)
                η[idx] = η_profiled[k]
            end
            η_keep = η[keep_idx_global]
            η_drop = Cj_inv_global * (Float64.(log_interest) - Ck_global * η_keep)
            η[drop_idx_global] = η_drop
            if any(!isfinite, η) || any(η .< θ_log_lower_global) || any(η .> θ_log_upper_global)
                return nothing
            end
            return η
        end

        @everywhere function lnlike_interest_keep_worker(chart_partial)
            try
                log_interest = chart_partial[1:2]
                η_profiled = chart_partial[3:end]
                η = reconstruct_η_global(log_interest, η_profiled)
                isnothing(η) && return -Inf
                θ = exp.(η)
                if any(θ .<= 0) || any(!isfinite, θ)
                    return -Inf
                end
                pred = RepressilatorModel.predict_mRNA(θ, t_obs_global, X0_global)
                if any(!isfinite, pred)
                    return -Inf
                end
                dist = MvNormal(pred, σ_global^2 * I(3 * NT_global))
                return logpdf(dist, data_global)
            catch
                return -Inf
            end
        end
    else
        function lnlike_interest_keep_local(chart_partial)
            try
                log_interest = chart_partial[1:2]
                η_profiled = chart_partial[3:end]
                η = reconstruct_η(log_interest, η_profiled)
                isnothing(η) && return -Inf
                θ = exp.(η)
                if any(θ .<= 0) || any(!isfinite, θ)
                    return -Inf
                end
                return lnlike_θ(θ)
            catch
                return -Inf
            end
        end
    end

    nuisance_log_lower = θ_log_lower_profile[nuisance_to_profile]
    nuisance_log_upper = θ_log_upper_profile[nuisance_to_profile]
    nuisance_log_guess = θ_log_MLE[nuisance_to_profile]
    nuisance_log_guess = clamp.(nuisance_log_guess, nuisance_log_lower .+ 1e-6, nuisance_log_upper .- 1e-6)

    n_extra_guesses = N_NUISANCE <= 4 ? 10 : 15
    nuisance_extras = ReparamTools.generate_initial_guesses(nuisance_log_lower, nuisance_log_upper, n_extra_guesses)

    n_chunks_profile = 36

    println("Grid: $GRID × $GRID = $(GRID^2) points")
    println("Nuisance to optimize: $(length(nuisance_to_profile))")
    println("Extra restarts: $n_extra_guesses")
    println("Chunks: $n_chunks_profile")
    if USE_DISTRIBUTED
        println("Workers: $(nworkers())")
    end
    println("\nStarting profiling...")
    flush(stdout)

    lnlike_func = USE_DISTRIBUTED ? lnlike_interest_keep_worker : lnlike_interest_keep_local
    chart_log_lower_opt = vcat(interest_log_lower, nuisance_log_lower)
    chart_log_upper_opt = vcat(interest_log_upper, nuisance_log_upper)

    chart_vals_raw, ll_vals = ReparamTools.profile_target(
        lnlike_func, [1, 2], chart_log_lower_opt, chart_log_upper_opt, nuisance_log_guess;
        grid_steps=GRID, use_distributed=USE_DISTRIBUTED,
        ω_initial_extras=nuisance_extras,
        method=:LN_BOBYQA, optmaxtime=N_NUISANCE <= 4 ? 60.0 : 150.0,
        n_chunks=n_chunks_profile,
        snake_direction=:row
    )
end

# Convert mixed chart values back to full θ and canonical full ψ (log-scale) for saving.
θ_vals = Vector{Vector{Float64}}(undef, length(chart_vals_raw))
ψ_vals = Vector{Vector{Float64}}(undef, length(chart_vals_raw))
for i in eachindex(chart_vals_raw)
    row = chart_vals_raw[i]
    log_interest = row[1:2]
    η_profiled = length(row) > 2 ? row[3:end] : Float64[]
    η = reconstruct_η(log_interest, η_profiled)
    if isnothing(η)
        θ_vals[i] = fill(NaN, n_params)
        ψ_vals[i] = fill(NaN, n_params)
    else
        θ = exp.(η)
        θ_vals[i] = collect(θ)
        ψ_vals[i] = collect(log.(θ_to_ψ(θ)))
    end
end

t_profile_elapsed = time() - t_profile_start
println("\nCompleted in $(round(t_profile_elapsed/60, digits=1)) minutes")
println("Finite values: $(sum(isfinite.(ll_vals)))/$(length(ll_vals))")

# === SAVE RESULTS ===
using Serialization

output_base = "repressilator_$(N_NUISANCE)nuisance_$(GRID)x$(GRID)"

results = Dict(
    "ψ_vals" => ψ_vals,
    "ψ_vals_layout" => "canonical_full_log",
    "θ_vals" => θ_vals,
    "chart_vals" => chart_vals_raw,
    "ll_vals" => ll_vals,
    "ψ_MLE" => ψ_MLE,
    "θ_MLE" => θ_MLE,
    "A_T_final" => A_T_final,
    "identified_basis_labels" => N_perp_labels,
    "null_basis_labels" => N_clean_labels,
    "target_2d" => target_2d,
    "nuisance_to_profile" => nuisance_to_profile,
    "fixed_at_mle" => fixed_at_mle,
    "GRID" => GRID,
    "N_NUISANCE" => N_NUISANCE,
    "ψ_lower" => ψ_lower,
    "ψ_upper" => ψ_upper,
    "target_log_lower" => interest_log_lower,
    "target_log_upper" => interest_log_upper,
    "target_bounds_source" => "legacy_varimax_window",
    "param_names" => param_names,
    "rank_J" => rank_J,
    "n_ident" => n_ident,
    "n_nonident" => n_nonident,
    "mode" => mode_str,
    "profile_chart" => "interest_plus_original_complement",
    "chart_interest_matrix" => C_interest,
    "chart_drop_idx" => drop_idx,
    "chart_keep_idx" => keep_idx,
    "chart_eta_keep_ref" => chart_eta_keep_ref,
    "chart_is_decoupled" => chart_decoupled,
    # Stored observation data + metadata for reproducible post-processing
    "data" => data,
    "y_true" => y_true,
    "t_obs" => collect(t_obs),
    "X0" => X0,
    "σ" => σ,
    "θ_true" => θ_true,
    "data_seed" => 42,
    "NT" => NT,
    "T_end" => T_end
)

serialize("$(output_base)_results.jls", results)
println("\nResults saved to $(output_base)_results.jls")

# === ANALYSIS ===
ll_matrix = reshape(ll_vals, GRID, GRID)
ll_max = maximum(ll_matrix[isfinite.(ll_matrix)])
like_matrix = exp.(ll_matrix .- ll_max)

lstar_1d = exp(-quantile(Chisq(1), 0.95)/2)
lstar_2d = exp(-quantile(Chisq(2), 0.95)/2)

like_ψ_target1 = [maximum(like_matrix[i, :]) for i in 1:GRID]
like_ψ_target2 = [maximum(like_matrix[:, j]) for j in 1:GRID]

n_above_1 = sum(like_ψ_target1 .> lstar_1d)
n_above_2 = sum(like_ψ_target2 .> lstar_1d)

println("\n" * "=" ^ 70)
println("RESULTS SUMMARY")
println("=" ^ 70)
println("K₁/β₁ (identifiable): $n_above_1/$GRID above 95% threshold")
println("β₁·K₁ (non-identifiable): $n_above_2/$GRID above 95% threshold")

println("\n" * "=" ^ 70)
println("COMPLETE")
println("=" ^ 70)
println("""
Summary:
  Mode: $mode_str
  Grid: $GRID × $GRID
  K₁/β₁ (identifiable): $n_above_1/$GRID above threshold (peaked profile expected)
  β₁·K₁ (non-identifiable): $n_above_2/$GRID above threshold (flat profile expected)

Output: $(output_base)_results.jls

To generate plots, run:
  julia replot_profile_results.jl $(output_base)_results.jls
""")
