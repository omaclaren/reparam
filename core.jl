# ----------------------------------------------------------------
# NLopt parallelism model used in this file
# ----------------------------------------------------------------
# NLopt is not thread-safe in a single Julia process.
# In practice: do NOT call optimize(...) concurrently from multiple threads.
# Therefore, all NLopt loops here are sequential within each process.
# Parallel speedup is done with Distributed.jl (multiple worker processes),
# where each worker runs its own sequential NLopt calls on a chunk.

using Distributed
using Distributed: WorkerPool

# ----------------------------------------------------------------
# Likelihood in Original (xy) Coordinates (dimension independent)
# ----------------------------------------------------------------
function construct_lnlike_xy(distrib_xy, data; dist_type=:uni)
    """
    Construct log-likelihood function for parameters given (iid) data
    in original coordinates.

    Note xy is original parameterization but of arbitrary dimension
    (not necessarily two).

    Parameters:
    - distrib_xy: Function mapping parameters xy to distribution
    - data: Vector of observations
    - dist_type: :uni for univariate, :multi for multivariate distributions

    Returns: Function computing log-likelihood for given parameters xy
    """
    if dist_type === :uni
        return xy -> sum(logpdf.(distrib_xy(xy),data))
    else
        return xy -> sum(logpdf(distrib_xy(xy),data))
    end
end

function construct_lnlike_to_max(lnlike)
    """
    Wrap log-likelihood function for optimization with gradients.

    Parameters:
    - lnlike: Log-likelihood function taking parameter vector θ

    Returns: Function suitable for NLopt maximization that computes both
    function value and gradient at θ. The returned function takes parameters:
    - θ: Parameter vector
    - grad: Gradient vector to be filled
    - grad_type: :auto for automatic differentiation (default), otherwise finite differences
    """
    # closure to pass gradient to optimizer
    function lnlike_to_max(θ, grad; grad_type=:auto)
        if length(grad) > 0  # Only compute gradient if vector provided
            if grad_type === :auto
                grad[:] = ForwardDiff.gradient(lnlike, θ)
            else
                grad[:] = finite_diff_gradient(lnlike, θ)
            end
        end
        return lnlike(θ)
    end

    return lnlike_to_max
end

# --------------------------------------------------------
# Parameter mapping methods (dimension independent)
# --------------------------------------------------------

function compute_ϕ_Jacobian(ϕ_func, θ; method_type=:auto, compute_svd=false)
    """
    Compute Jacobian of ϕ mapping at given parameters, optionally with SVD.
    Works in any coordinate system.

    Parameters:
    - ϕ_func: Function implementing the ϕ mapping. Should be function of θ only.
    - θ: Parameter vector at which to evaluate the Jacobian
    - method_type: Differentiation method (currently only :auto is supported)
    - compute_svd: Whether to compute and return SVD (default: false)

    Returns:
    - If compute_svd=false: Just the Jacobian matrix
    - If compute_svd=true: Tuple of (Jacobian, SVD factorization)
    """
    if method_type === :auto
        J = ForwardDiff.jacobian(ϕ_func, θ)
    else
        error("compute_ϕ_Jacobian: method_type=$method_type is not supported. Use method_type=:auto.")
    end

    if compute_svd
        println("Computing and returning SVD of Jacobian of φ mapping")
        U, S, Vt = svd(J)
        return (J, U, S, Vt)
    else
        return J
    end
end

# --------------------------------------------------------
# Likelihood-based Statistical Inference Methods
# --------------------------------------------------------

function construct_ellipse_lnlike_approx(lnlike, θ_est; method_type=:auto, return_h=true)
    """
    Construct quadratic approximation to log-likelihood at maximum.
    Works in any coordinate system.

    Parameters:
    - lnlike: Log-likelihood function taking parameter vector θ
    - θ_est: Parameter vector at which to make approximation
    - method_type: Differentiation method (currently only :auto is supported)
    - return_h: Whether to return Hessian matrix (default: true)

    Returns:
    - If return_h=true: Tuple of (quadratic approximation function, Hessian matrix)
    - If return_h=false: Quadratic approximation function only
    """
    if method_type === :auto
        H = -ForwardDiff.hessian(lnlike, θ_est)
    else
        error("construct_ellipse_lnlike_approx: method_type=$method_type is not supported. Use method_type=:auto.")
    end
    if return_h
        return θ -> -0.5*(θ-θ_est)'*H*(θ-θ_est), H
    else
        return θ -> -0.5*(θ-θ_est)'*H*(θ-θ_est)
    end
end

"""
    profile_point(lnlike_θ, ψ_fixed, ψ_indices, θ_bounds_lower, θ_bounds_upper, ω_initial;
                  ω_initial_extras=nothing, method=:LN_BOBYQA, local_method=:LD_TNEWTON_PRECOND,
                  xtol_rel=1e-9, ftol_rel=1e-9, optmaxtime=60.0, popsize=50, track_convergence=false)

Optimize log-likelihood at a single fixed point in interest parameter space.

This is the primitive operation for profile likelihood: given fixed values for
parameters of interest (ψ_fixed), find the values of nuisance parameters (ω)
that maximize the likelihood. Uses multi-start sequential optimization.

# Arguments
- `lnlike_θ`: Log-likelihood function taking full parameter vector θ
- `ψ_fixed`: Fixed values for interest parameters (vector matching length of ψ_indices)
- `ψ_indices`: Indices of interest parameters in full θ vector
- `θ_bounds_lower`: Lower bounds for all parameters
- `θ_bounds_upper`: Upper bounds for all parameters
- `ω_initial`: Initial guess for nuisance parameters
- `ω_initial_extras`: Additional starting guesses for nuisance parameters (default: nothing)
- `method`: NLopt method for optimization (default: :LN_BOBYQA)
- `local_method`: Local method if using global optimizer (default: :LD_TNEWTON_PRECOND)
- `xtol_rel`: Relative tolerance in parameters (default: 1e-9)
- `ftol_rel`: Relative tolerance in function value (default: 1e-9)
- `optmaxtime`: Maximum time per optimization in seconds (default: 60.0)
- `popsize`: Population size for global methods (default: 50)
- `track_convergence`: Whether to track NLopt return codes (default: false)

# Returns
- `θ_opt`: Optimal full parameter vector
- `ω_opt`: Optimal nuisance parameter values
- `lnlike_opt`: Optimized log-likelihood value
- `converged_to`: NLopt return code (if track_convergence=true, else :NOT_TRACKED)

# Notes
- All starting points are tried sequentially (NLopt is not thread-safe)
- Returns best result across all starting points
- If no nuisance parameters (dim_ω=0), returns ψ_fixed with its likelihood
"""
function profile_point(lnlike_θ, ψ_fixed::Vector{Float64}, ψ_indices::Vector{Int},
                       θ_bounds_lower, θ_bounds_upper, ω_initial::Vector{Float64};
                       ω_initial_extras::Union{Nothing, Vector{Vector{Float64}}}=nothing,
                       method=:LN_BOBYQA, local_method=:LD_TNEWTON_PRECOND,
                       xtol_rel=1e-9, ftol_rel=1e-9, optmaxtime=60.0, popsize=50,
                       track_convergence=false)

    # Get dimensions and indices
    dim_all = length(θ_bounds_lower)
    ω_indices = setdiff(1:dim_all, ψ_indices)
    dim_ω = length(ω_indices)

    if length(ψ_fixed) != length(ψ_indices)
        error("profile_point: length(ψ_fixed)=$(length(ψ_fixed)) must match length(ψ_indices)=$(length(ψ_indices)).")
    end

    # Build index mapping for reconstructing full θ vector
    ψω_to_θ_indices = construct_ψω_to_θ_indices(dim_all, ψ_indices, ω_indices)
    ψω_to_θ = ψω -> ψω[ψω_to_θ_indices]

    # Pure gridding case (no nuisance parameters)
    if dim_ω == 0
        θ_opt = ψω_to_θ(ψ_fixed)
        lnlike_opt = lnlike_θ(θ_opt)
        converged_to = :NO_OPTIMIZATION
        return θ_opt, Float64[], lnlike_opt, converged_to
    end

    if length(ω_initial) != dim_ω
        error("profile_point: length(ω_initial)=$(length(ω_initial)) must match nuisance dimension $dim_ω.")
    end
    if !isnothing(ω_initial_extras)
        for (k, ω₀) in enumerate(ω_initial_extras)
            if length(ω₀) != dim_ω
                error("profile_point: ω_initial_extras[$k] has length $(length(ω₀)); expected $dim_ω.")
            end
        end
    end

    # Extract bounds for nuisance parameters
    ω_bounds_lower = θ_bounds_lower[ω_indices]
    ω_bounds_upper = θ_bounds_upper[ω_indices]

    # Setup NLopt optimizer
    opt = if method in [:G_MLSL_LDS, :G_MLSL]
        # Global optimization with local refinement
        opt = Opt(method, dim_ω)
        local_opt = Opt(local_method, dim_ω)
        local_opt.maxtime = optmaxtime
        local_opt.lower_bounds = ω_bounds_lower
        local_opt.upper_bounds = ω_bounds_upper
        local_opt.xtol_rel = xtol_rel
        local_opt.ftol_rel = ftol_rel
        local_optimizer!(opt, local_opt)
        opt.population = popsize
        opt
    else
        # Direct optimization methods
        opt = Opt(method, dim_ω)
        if method in (:GN_DIRECT, :GN_DIRECT_L, :GN_DIRECT_L_RAND)
            opt.population = popsize
        end
        opt
    end

    # Set common optimizer options
    opt.maxtime = optmaxtime
    opt.lower_bounds = ω_bounds_lower
    opt.upper_bounds = ω_bounds_upper
    opt.xtol_rel = xtol_rel
    opt.ftol_rel = ftol_rel

    # Prepare starting points (ω_initial + extras)
    starting_points = [ω_initial]
    if !isnothing(ω_initial_extras)
        append!(starting_points, ω_initial_extras)
    end

    # Try multiple starting points sequentially (NLopt is not thread-safe)
    best_lnlike = -Inf
    best_ω = similar(ω_initial)
    converged_to = :NOT_TRACKED

    for ω₀ in starting_points
        # Define objective: maximize likelihood with ψ fixed at ψ_fixed
        opt.max_objective = construct_lnlike_to_max(ω -> lnlike_θ(ψω_to_θ([ψ_fixed..., ω...])))

        # Optimize
        (lnlike_opt, ω_opt, return_code) = optimize(opt, ω₀)

        # Keep best result
        if lnlike_opt > best_lnlike
            best_lnlike = lnlike_opt
            best_ω = ω_opt
            if track_convergence
                converged_to = return_code
            end
        end
    end

    # Reconstruct full parameter vector
    θ_opt = ψω_to_θ([ψ_fixed..., best_ω...])

    return θ_opt, best_ω, best_lnlike, converged_to
end


"""
    profile_grid_sequential(lnlike_θ, ψ_grid, ψ_indices, θ_bounds_lower, θ_bounds_upper, ω_initial;
                           ω_initial_extras=nothing, method=:LN_BOBYQA, local_method=:LD_TNEWTON_PRECOND,
                           xtol_rel=1e-9, ftol_rel=1e-9, optmaxtime=60.0, popsize=50, track_convergence=false)

Execute profile likelihood over a pre-defined grid with adaptive continuation.

Iterates through grid points in the order provided, using the optimized nuisance
parameters from each point as the starting guess for the next. This "adaptive
continuation" significantly speeds up optimization in smooth regions of parameter space.

# Arguments
- `lnlike_θ`: Log-likelihood function taking full parameter vector θ
- `ψ_grid`: Vector of interest parameter vectors (each element is a Vector{Float64})
- `ψ_indices`: Indices of interest parameters in full θ vector
- `θ_bounds_lower`: Lower bounds for all parameters
- `θ_bounds_upper`: Upper bounds for all parameters
- `ω_initial`: Initial guess for nuisance parameters (used for first grid point)
- `ω_initial_extras`: Additional starting guesses for nuisance parameters
- `method`: NLopt method for optimization (default: :LN_BOBYQA)
- `local_method`: Local method if using global optimizer
- `xtol_rel`: Relative tolerance in parameters (default: 1e-9)
- `ftol_rel`: Relative tolerance in function value (default: 1e-9)
- `optmaxtime`: Maximum time per optimization in seconds (default: 60.0)
- `popsize`: Population size for global methods (default: 50)
- `track_convergence`: Whether to track NLopt return codes (default: false)

# Returns
- `θ_values`: Vector of optimal parameter vectors (length = length(ψ_grid))
- `lnlike_values`: Vector of log-likelihood values (unnormalized)
- `convergence_info`: Vector of convergence outcomes (if track_convergence=true)

# Notes
- Grid points are evaluated in the order provided
- ω_initial_extras (if provided) are regenerated once continuation has started
  (current behavior: regenerate after the second point, then use for subsequent points)
- Results are NOT normalized (caller should normalize if desired)
"""
function profile_grid_sequential(lnlike_θ, ψ_grid::Vector{Vector{Float64}}, ψ_indices::Vector{Int},
                                 θ_bounds_lower, θ_bounds_upper, ω_initial::Vector{Float64};
                                 ω_initial_extras::Union{Nothing, Vector{Vector{Float64}}}=nothing,
                                 method=:LN_BOBYQA, local_method=:LD_TNEWTON_PRECOND,
                                 xtol_rel=1e-9, ftol_rel=1e-9, optmaxtime=60.0, popsize=50,
                                 track_convergence=false)

    # Get dimensions
    dim_all = length(θ_bounds_lower)
    ω_indices = setdiff(1:dim_all, ψ_indices)
    dim_ω = length(ω_indices)
    n_grid = length(ψ_grid)

    # Extract bounds for nuisance parameters (for regenerating extras)
    ω_bounds_lower = θ_bounds_lower[ω_indices]
    ω_bounds_upper = θ_bounds_upper[ω_indices]

    # Pre-allocate result arrays
    θ_values = Vector{Vector{Float64}}(undef, n_grid)
    lnlike_values = Vector{Float64}(undef, n_grid)
    if track_convergence
        convergence_info = Vector{Symbol}(undef, n_grid)
    end

    # Track current nuisance initial guess for adaptive continuation
    ω_current = ω_initial
    ω_extras_current = ω_initial_extras

    # Loop over grid points in order
    for (i, ψᵢ) in enumerate(ψ_grid)
        # Optimize at this grid point
        θ_opt, ω_opt, lnlike_opt, conv = profile_point(
            lnlike_θ, ψᵢ, ψ_indices,
            θ_bounds_lower, θ_bounds_upper, ω_current;
            ω_initial_extras=ω_extras_current,
            method=method, local_method=local_method,
            xtol_rel=xtol_rel, ftol_rel=ftol_rel,
            optmaxtime=optmaxtime, popsize=popsize,
            track_convergence=track_convergence
        )

        # Store results
        θ_values[i] = θ_opt
        lnlike_values[i] = lnlike_opt
        if track_convergence
            convergence_info[i] = conv
        end

        # Adaptive continuation: use optimized ω as next starting point
        # Clamp to be strictly inside bounds to avoid NLopt errors
        eps_bound = 1e-6
        ω_current = clamp.(ω_opt, ω_bounds_lower .+ eps_bound, ω_bounds_upper .- eps_bound)

        # Once continuation is established, regenerate extras around current point
        # Current behavior: regenerate when i > 1 (used from the next iteration onward)
        if i > 1 && !isnothing(ω_initial_extras) && dim_ω > 0
            ω_extras_current = generate_initial_guesses(
                ω_bounds_lower, ω_bounds_upper, length(ω_initial_extras);
                reference_point=ω_current
            )
        end
    end

    # Return results (unnormalized)
    if track_convergence
        return θ_values, lnlike_values, convergence_info
    else
        return θ_values, lnlike_values
    end
end


"""
    profile_grid_distributed(lnlike_θ, ψ_grid, ψ_indices, θ_bounds_lower, θ_bounds_upper, ω_initial;
                            ω_initial_extras=nothing, method=:LN_BOBYQA, local_method=:LD_TNEWTON_PRECOND,
                            xtol_rel=1e-9, ftol_rel=1e-9, optmaxtime=60.0, popsize=50,
                            n_chunks=nothing, worker_pool=nothing, track_convergence=false)

Execute profile likelihood over a grid using distributed process-level parallel execution.

Splits the grid into contiguous chunks, runs chunks in parallel across workers via `pmap`,
and concatenates results. Inside each chunk, execution is sequential via
`profile_grid_sequential()`. Adaptive continuation works within each chunk but not between chunks.

# Arguments
- `lnlike_θ`: Log-likelihood function (must be serializable)
- `ψ_grid`: Vector of interest parameter vectors (from any dimensionality: 1D, 2D, 3D, ...)
- `ψ_indices`: Indices of interest parameters
- `θ_bounds_lower`, `θ_bounds_upper`: Parameter bounds (must be serializable)
- `ω_initial`: Initial guess for nuisance parameters
- `ω_initial_extras`: Additional starting guesses
- `method`, `local_method`: NLopt methods
- `xtol_rel`, `ftol_rel`: Tolerances
- `optmaxtime`: Time limit per optimization (seconds)
- `popsize`: Population size for global methods
- `n_chunks`: Number of chunks (default: worker count in pool)
- `worker_pool`: Optional `WorkerPool` specifying which workers to use (default: all workers)
- `track_convergence`: Whether to track convergence (default: false)

# Returns
- `θ_values`: Vector of optimal parameter vectors
- `lnlike_values`: Vector of log-likelihood values (unnormalized)
- `convergence_info`: Convergence outcomes (if track_convergence=true)

# Notes
- Requires workers: `addprocs(4); @everywhere using ReparamTools`
- Chunks are contiguous slices of ψ_grid
- Chunks run in parallel across workers; each chunk runs sequentially internally
- Continuation preserved WITHIN chunks, lost BETWEEN chunks
- For 21×21 grid with 4 workers: ~14.7 hrs sequential → ~4 hrs distributed

# Example
```julia
using Distributed
addprocs(4)
@everywhere using ReparamTools

θ_vals, ll_vals = profile_grid_distributed(
    lnlike_θ, ψ_grid, ψ_indices,
    θ_lower, θ_upper, ω_init; n_chunks=4
)
```
"""
function profile_grid_distributed(lnlike_θ, ψ_grid::Vector{Vector{Float64}}, ψ_indices::Vector{Int},
                                  θ_bounds_lower, θ_bounds_upper, ω_initial::Vector{Float64};
                                  ω_initial_extras::Union{Nothing, Vector{Vector{Float64}}}=nothing,
                                  method=:LN_BOBYQA, local_method=:LD_TNEWTON_PRECOND,
                                  xtol_rel=1e-9, ftol_rel=1e-9, optmaxtime=60.0, popsize=50,
                                  n_chunks::Union{Nothing, Int}=nothing,
                                  worker_pool::Union{Nothing, WorkerPool}=nothing,
                                  track_convergence=false)

    # Check workers available
    pool = isnothing(worker_pool) ? WorkerPool(workers()) : worker_pool
    n_pool_workers = length(pool.workers)
    if n_pool_workers == 0
        @warn "Worker pool is empty; using sequential execution instead."
        return profile_grid_sequential(
            lnlike_θ, ψ_grid, ψ_indices,
            θ_bounds_lower, θ_bounds_upper, ω_initial;
            ω_initial_extras=ω_initial_extras,
            method=method, local_method=local_method,
            xtol_rel=xtol_rel, ftol_rel=ftol_rel,
            optmaxtime=optmaxtime, popsize=popsize,
            track_convergence=track_convergence
        )
    end

    # Determine number of chunks (don't exceed grid size)
    n_grid = length(ψ_grid)
    n_chunks_actual = isnothing(n_chunks) ? n_pool_workers : n_chunks
    n_chunks_actual = max(1, min(n_chunks_actual, n_grid))  # Clamp to valid range

    # Split grid into contiguous chunks
    chunk_size = div(n_grid, n_chunks_actual)
    remainder = n_grid % n_chunks_actual

    chunks = Vector{Vector{Vector{Float64}}}(undef, n_chunks_actual)
    start_idx = 1
    for i in 1:n_chunks_actual
        this_size = chunk_size + (i <= remainder ? 1 : 0)
        chunks[i] = ψ_grid[start_idx:start_idx+this_size-1]
        start_idx += this_size
    end

    println("Distributed profiling: $n_grid points → $n_chunks_actual chunks on $n_pool_workers workers")
    println("Chunk sizes: ", [length(c) for c in chunks])

    # Evaluate each chunk in parallel using pmap
    # Each chunk runs sequentially with adaptive continuation
    results = pmap(pool, chunks) do chunk_grid
        profile_grid_sequential(
            lnlike_θ, chunk_grid, ψ_indices,
            θ_bounds_lower, θ_bounds_upper, ω_initial;
            ω_initial_extras=ω_initial_extras,
            method=method, local_method=local_method,
            xtol_rel=xtol_rel, ftol_rel=ftol_rel,
            optmaxtime=optmaxtime, popsize=popsize,
            track_convergence=track_convergence
        )
    end

    # Concatenate results (chunks were contiguous slices, so just join them)
    if track_convergence
        θ_values = vcat([r[1] for r in results]...)
        lnlike_values = vcat([r[2] for r in results]...)
        convergence_info = vcat([r[3] for r in results]...)
        return θ_values, lnlike_values, convergence_info
    else
        θ_values = vcat([r[1] for r in results]...)
        lnlike_values = vcat([r[2] for r in results]...)
        return θ_values, lnlike_values
    end
end


function _profile_strict_clamp(ω::Vector{Float64}, ω_bounds_lower::Vector{Float64}, ω_bounds_upper::Vector{Float64}; eps_bound=1e-6)
    isempty(ω) && return Float64[]
    return clamp.(ω, ω_bounds_lower .+ eps_bound, ω_bounds_upper .- eps_bound)
end

function _profile_dedup_start_points(start_candidates)
    starts = Vector{Vector{Float64}}()
    seen = Set{String}()
    for candidate in start_candidates
        candidate === nothing && continue
        start = Vector{Float64}(candidate)
        key = join(string.(round.(start, digits=8)), ",")
        if !(key in seen)
            push!(starts, start)
            push!(seen, key)
        end
    end
    return starts
end

function _profile_partition_ranges(n::Int, n_parts::Int)
    n_parts = max(1, min(n_parts, n))
    base = div(n, n_parts)
    remainder = n % n_parts

    ranges = Vector{UnitRange{Int}}(undef, n_parts)
    start_idx = 1
    for i in 1:n_parts
        this_size = base + (i <= remainder ? 1 : 0)
        ranges[i] = start_idx:start_idx + this_size - 1
        start_idx += this_size
    end
    return ranges
end

function _profile_row_minor_indices(i::Int, n2::Int, sweep_direction::Symbol)
    if sweep_direction == :forward
        return isodd(i) ? (1:n2) : (n2:-1:1)
    else
        return isodd(i) ? (n2:-1:1) : (1:n2)
    end
end

function _profile_col_minor_indices(j::Int, n1::Int, sweep_direction::Symbol)
    if sweep_direction == :forward
        return isodd(j) ? (1:n1) : (n1:-1:1)
    else
        return isodd(j) ? (n1:-1:1) : (1:n1)
    end
end

function _profile_grid_2d_structured_block(
    lnlike_θ, ψ_grids::Vector{Vector{Float64}}, ψ_indices::Vector{Int},
    θ_bounds_lower, θ_bounds_upper, ω_initial::Vector{Float64},
    ω_global_backstops::Vector{Vector{Float64}}, ω_bootstrap_extras::Vector{Vector{Float64}},
    θ_prev::Matrix{Vector{Float64}}, ω_prev::Matrix{Vector{Float64}}, ll_prev::Matrix{Float64}, conv_prev,
    major_block::UnitRange{Int};
    snake_direction=:column, sweep_direction=:forward,
    method=:LN_BOBYQA, local_method=:LD_TNEWTON_PRECOND,
    xtol_rel=1e-9, ftol_rel=1e-9, optmaxtime=60.0, popsize=50,
    track_convergence=false)

    n1, n2 = length(ψ_grids[1]), length(ψ_grids[2])

    dim_all = length(θ_bounds_lower)
    ω_indices = setdiff(1:dim_all, ψ_indices)
    ω_bounds_lower = θ_bounds_lower[ω_indices]
    ω_bounds_upper = θ_bounds_upper[ω_indices]

    if snake_direction == :row
        block_len = length(major_block)
        θ_block = Matrix{Vector{Float64}}(undef, block_len, n2)
        ω_block = Matrix{Vector{Float64}}(undef, block_len, n2)
        ll_block = Matrix{Float64}(undef, block_len, n2)
        conv_block = track_convergence ? Matrix{Symbol}(undef, block_len, n2) : nothing

        for (li, i) in enumerate(major_block), j in 1:n2
            θ_block[li, j] = θ_prev[i, j]
            ω_block[li, j] = ω_prev[i, j]
            ll_block[li, j] = ll_prev[i, j]
            if track_convergence
                conv_block[li, j] = conv_prev[i, j]
            end
        end

        row_iter = sweep_direction == :forward ? major_block : reverse(major_block)
        prev_point = nothing

        for i in row_iter
            li = i - first(major_block) + 1
            for j in _profile_row_minor_indices(i, n2, sweep_direction)
                start_candidates = Any[]
                has_local_info = false

                if isfinite(ll_prev[i, j])
                    push!(start_candidates, ω_prev[i, j])
                    has_local_info = true
                end

                if !(prev_point === nothing)
                    pi, pj = prev_point
                    lpi = pi - first(major_block) + 1
                    if 1 <= lpi <= block_len && isfinite(ll_block[lpi, pj])
                        push!(start_candidates, ω_block[lpi, pj])
                        has_local_info = true
                    end
                end

                ortho_i = sweep_direction == :forward ? i - 1 : i + 1
                if 1 <= ortho_i <= n1
                    if ortho_i in major_block
                        lortho = ortho_i - first(major_block) + 1
                        if isfinite(ll_block[lortho, j])
                            push!(start_candidates, ω_block[lortho, j])
                            has_local_info = true
                        end
                    elseif isfinite(ll_prev[ortho_i, j])
                        push!(start_candidates, ω_prev[ortho_i, j])
                        has_local_info = true
                    end
                end

                push!(start_candidates, ω_initial)
                append!(start_candidates, has_local_info ? ω_global_backstops : ω_bootstrap_extras)

                starts = _profile_dedup_start_points(start_candidates)
                isempty(starts) && push!(starts, ω_initial)

                θ_opt, ω_opt, ll_opt, conv = profile_point(
                    lnlike_θ, [ψ_grids[1][i], ψ_grids[2][j]], ψ_indices,
                    θ_bounds_lower, θ_bounds_upper, starts[1];
                    ω_initial_extras=length(starts) > 1 ? starts[2:end] : nothing,
                    method=method, local_method=local_method,
                    xtol_rel=xtol_rel, ftol_rel=ftol_rel,
                    optmaxtime=optmaxtime, popsize=popsize,
                    track_convergence=track_convergence
                )

                ω_opt_clamped = _profile_strict_clamp(ω_opt, ω_bounds_lower, ω_bounds_upper)
                existing_ll = ll_block[li, j]
                if ll_opt > existing_ll || (!isfinite(existing_ll) && !isfinite(ll_opt))
                    θ_block[li, j] = θ_opt
                    ω_block[li, j] = ω_opt_clamped
                    ll_block[li, j] = ll_opt
                    if track_convergence
                        conv_block[li, j] = conv
                    end
                end

                prev_point = (i, j)
            end
        end

        return major_block, θ_block, ω_block, ll_block, conv_block
    elseif snake_direction == :column
        block_len = length(major_block)
        θ_block = Matrix{Vector{Float64}}(undef, n1, block_len)
        ω_block = Matrix{Vector{Float64}}(undef, n1, block_len)
        ll_block = Matrix{Float64}(undef, n1, block_len)
        conv_block = track_convergence ? Matrix{Symbol}(undef, n1, block_len) : nothing

        for i in 1:n1, (lj, j) in enumerate(major_block)
            θ_block[i, lj] = θ_prev[i, j]
            ω_block[i, lj] = ω_prev[i, j]
            ll_block[i, lj] = ll_prev[i, j]
            if track_convergence
                conv_block[i, lj] = conv_prev[i, j]
            end
        end

        col_iter = sweep_direction == :forward ? major_block : reverse(major_block)
        prev_point = nothing

        for j in col_iter
            lj = j - first(major_block) + 1
            for i in _profile_col_minor_indices(j, n1, sweep_direction)
                start_candidates = Any[]
                has_local_info = false

                if isfinite(ll_prev[i, j])
                    push!(start_candidates, ω_prev[i, j])
                    has_local_info = true
                end

                if !(prev_point === nothing)
                    pi, pj = prev_point
                    lpj = pj - first(major_block) + 1
                    if 1 <= lpj <= block_len && isfinite(ll_block[pi, lpj])
                        push!(start_candidates, ω_block[pi, lpj])
                        has_local_info = true
                    end
                end

                ortho_j = sweep_direction == :forward ? j - 1 : j + 1
                if 1 <= ortho_j <= n2
                    if ortho_j in major_block
                        lortho = ortho_j - first(major_block) + 1
                        if isfinite(ll_block[i, lortho])
                            push!(start_candidates, ω_block[i, lortho])
                            has_local_info = true
                        end
                    elseif isfinite(ll_prev[i, ortho_j])
                        push!(start_candidates, ω_prev[i, ortho_j])
                        has_local_info = true
                    end
                end

                push!(start_candidates, ω_initial)
                append!(start_candidates, has_local_info ? ω_global_backstops : ω_bootstrap_extras)

                starts = _profile_dedup_start_points(start_candidates)
                isempty(starts) && push!(starts, ω_initial)

                θ_opt, ω_opt, ll_opt, conv = profile_point(
                    lnlike_θ, [ψ_grids[1][i], ψ_grids[2][j]], ψ_indices,
                    θ_bounds_lower, θ_bounds_upper, starts[1];
                    ω_initial_extras=length(starts) > 1 ? starts[2:end] : nothing,
                    method=method, local_method=local_method,
                    xtol_rel=xtol_rel, ftol_rel=ftol_rel,
                    optmaxtime=optmaxtime, popsize=popsize,
                    track_convergence=track_convergence
                )

                ω_opt_clamped = _profile_strict_clamp(ω_opt, ω_bounds_lower, ω_bounds_upper)
                existing_ll = ll_block[i, lj]
                if ll_opt > existing_ll || (!isfinite(existing_ll) && !isfinite(ll_opt))
                    θ_block[i, lj] = θ_opt
                    ω_block[i, lj] = ω_opt_clamped
                    ll_block[i, lj] = ll_opt
                    if track_convergence
                        conv_block[i, lj] = conv
                    end
                end

                prev_point = (i, j)
            end
        end

        return major_block, θ_block, ω_block, ll_block, conv_block
    else
        error("snake_direction must be :column or :row, got $snake_direction")
    end
end

function _profile_grid_2d_structured(
    lnlike_θ, ψ_grids::Vector{Vector{Float64}}, ψ_indices::Vector{Int},
    θ_bounds_lower, θ_bounds_upper, ω_initial::Vector{Float64};
    ω_initial_extras::Union{Nothing, Vector{Vector{Float64}}}=nothing,
    method=:LN_BOBYQA, local_method=:LD_TNEWTON_PRECOND,
    xtol_rel=1e-9, ftol_rel=1e-9, optmaxtime=60.0, popsize=50,
    use_distributed=false, n_chunks=nothing, worker_pool=nothing,
    track_convergence=false, snake_direction=:column)

    n1, n2 = length(ψ_grids[1]), length(ψ_grids[2])
    dim_all = length(θ_bounds_lower)
    ω_indices = setdiff(1:dim_all, ψ_indices)
    ω_bounds_lower = θ_bounds_lower[ω_indices]
    ω_bounds_upper = θ_bounds_upper[ω_indices]

    ω_initial_clamped = _profile_strict_clamp(Vector{Float64}(ω_initial), ω_bounds_lower, ω_bounds_upper)
    ω_bootstrap_extras = isnothing(ω_initial_extras) ? Vector{Vector{Float64}}() :
        [_profile_strict_clamp(Vector{Float64}(ω), ω_bounds_lower, ω_bounds_upper) for ω in ω_initial_extras]
    n_backstops = min(3, length(ω_bootstrap_extras))
    ω_global_backstops = n_backstops == 0 ? Vector{Vector{Float64}}() : ω_bootstrap_extras[1:n_backstops]

    θ_best = [fill(NaN, dim_all) for _ in 1:n1, _ in 1:n2]
    ω_best = [copy(ω_initial_clamped) for _ in 1:n1, _ in 1:n2]
    ll_best = fill(-Inf, n1, n2)
    conv_best = track_convergence ? fill(:NOT_TRACKED, n1, n2) : nothing

    pool = worker_pool
    n_pool_workers = 0
    if use_distributed
        pool = isnothing(worker_pool) ? WorkerPool(workers()) : worker_pool
        n_pool_workers = length(pool.workers)
        if n_pool_workers == 0
            @warn "Worker pool is empty; using sequential structured 2D execution instead."
            use_distributed = false
        end
    end

    primary_n = snake_direction == :row ? n1 : n2
    n_blocks_actual = if use_distributed
        requested = isnothing(n_chunks) ? n_pool_workers : n_chunks
        max(1, min(requested, n_pool_workers, primary_n))
    else
        1
    end
    blocks = _profile_partition_ranges(primary_n, n_blocks_actual)
    block_axis_label = snake_direction == :row ? "row" : "column"
    sweep_directions = isempty(ω_indices) ? (:forward,) : (:forward, :reverse)
    println("Structured 2D profiling: $(n1)×$(n2) grid → $(length(blocks)) $(block_axis_label) blocks × $(length(sweep_directions)) sweep(s)")
    if use_distributed
        println("Workers: $n_pool_workers")
    end

    for (sweep_idx, sweep_direction) in enumerate(sweep_directions)
        println("  Sweep $(sweep_idx)/$(length(sweep_directions)): $(sweep_direction)")
        θ_prev = copy(θ_best)
        ω_prev = copy(ω_best)
        ll_prev = copy(ll_best)
        conv_prev = track_convergence ? copy(conv_best) : nothing

        if use_distributed
            block_results = pmap(pool, blocks) do major_block
                _profile_grid_2d_structured_block(
                    lnlike_θ, ψ_grids, ψ_indices,
                    θ_bounds_lower, θ_bounds_upper, ω_initial_clamped,
                    ω_global_backstops, ω_bootstrap_extras,
                    θ_prev, ω_prev, ll_prev, conv_prev,
                    major_block;
                    snake_direction=snake_direction, sweep_direction=sweep_direction,
                    method=method, local_method=local_method,
                    xtol_rel=xtol_rel, ftol_rel=ftol_rel,
                    optmaxtime=optmaxtime, popsize=popsize,
                    track_convergence=track_convergence
                )
            end
        else
            block_results = map(blocks) do major_block
                _profile_grid_2d_structured_block(
                    lnlike_θ, ψ_grids, ψ_indices,
                    θ_bounds_lower, θ_bounds_upper, ω_initial_clamped,
                    ω_global_backstops, ω_bootstrap_extras,
                    θ_prev, ω_prev, ll_prev, conv_prev,
                    major_block;
                    snake_direction=snake_direction, sweep_direction=sweep_direction,
                    method=method, local_method=local_method,
                    xtol_rel=xtol_rel, ftol_rel=ftol_rel,
                    optmaxtime=optmaxtime, popsize=popsize,
                    track_convergence=track_convergence
                )
            end
        end

        for (major_block, θ_block, ω_block, ll_block, conv_block) in block_results
            if snake_direction == :row
                for (li, i) in enumerate(major_block), j in 1:n2
                    θ_best[i, j] = θ_block[li, j]
                    ω_best[i, j] = ω_block[li, j]
                    ll_best[i, j] = ll_block[li, j]
                    if track_convergence
                        conv_best[i, j] = conv_block[li, j]
                    end
                end
            else
                for i in 1:n1, (lj, j) in enumerate(major_block)
                    θ_best[i, j] = θ_block[i, lj]
                    ω_best[i, j] = ω_block[i, lj]
                    ll_best[i, j] = ll_block[i, lj]
                    if track_convergence
                        conv_best[i, j] = conv_block[i, lj]
                    end
                end
            end
        end
    end

    θ_values = vec(θ_best)
    lnlike_values = vec(ll_best)
    if track_convergence
        return θ_values, lnlike_values, vec(conv_best)
    else
        return θ_values, lnlike_values
    end
end


function profile_target(lnlike_θ, ψ_indices, θ_bounds_lower, θ_bounds_upper, ω_initial;
    grid_steps=100, ω_initial_extras::Union{Nothing, Vector{Vector{Float64}}}=nothing,
    method=:LD_TNEWTON_PRECOND, local_method=:LD_TNEWTON_PRECOND, xtol_rel=1e-9, ftol_rel=1e-9,
    optmaxtime=120, popsize=50, track_convergence=false,
    use_distributed=false, n_chunks=nothing, worker_pool=nothing,
    snake_direction=:column)
    """
    Construct profile likelihood by maximizing over nuisance parameters.

    Profile likelihood splits parameters into interest (ψ) and nuisance (ω) parameters,
    then maximizes over nuisance parameters for each value of interest parameters.

    Parameters:
    - lnlike_θ: Log-likelihood function taking full parameter vector θ
    - ψ_indices: Indices of parameters of interest (empty for MLE)
    - θ_bounds_lower: Lower bounds for all parameters
    - θ_bounds_upper: Upper bounds for all parameters
    - ω_initial: Initial guess for nuisance parameters
    - grid_steps: Number of grid points for interest parameters (default: 100)
    - ω_initial_extras: Vector of additional initial guesses for nuisance parameters,
        where each guess is a vector of the same dimension as ω_initial (default: nothing)
    - method: Overall optimization method for nuisance parameters (default: :LD_TNEWTON_PRECOND)
    - local_method: Local optimization method if using a global method which requires it (default: :LD_TNEWTON_PRECOND)
    - xtol_rel: Relative tolerance in parameter values (default: 1e-9)
    - ftol_rel: Relative tolerance in function value (default: 1e-9)
    - optmaxtime: Maximum optimization time in seconds (default: 120)
    - popsize: Population size for global optimization methods (default: 50)
    - use_distributed: Use distributed parallel execution (default: false)
    - n_chunks: Number of chunks for distributed execution (default: worker count)
    - worker_pool: Optional Distributed.WorkerPool to target specific workers (default: all)
    - snake_direction: For 2D grids, :column (default) traverses ψ₁ within columns,
        :row traverses ψ₂ within rows. Use :row when the nuisance landscape is smoother
        along ψ₂ (e.g., when ψ₂ is non-identifiable).

    Returns:
    - θ_values: Array of parameter vectors in original ordering
    - lnlike_ψ_values: Profile log-likelihood values (normalized to max of 0)

    Notes:
    - This function delegates execution to profile_grid_sequential() or profile_grid_distributed()
    - For 2D targets, snake_direction defines the 1D traversal order of grid points
    - In distributed mode, that ordered list is chunked; continuation is preserved within chunks only
    """
    # Ensure ψ_indices is Vector{Int} (handles empty [], scalar Int, and Vector{Any} cases)
    ψ_indices_int = ψ_indices isa AbstractVector ? convert(Vector{Int}, ψ_indices) : [Int(ψ_indices)]

    # Get dimensions
    dim_ψ = length(ψ_indices_int)

    # Special case: Point estimation (no interest parameters, just find MLE)
    if dim_ψ == 0
        dim_all = length(θ_bounds_lower)
        ω_indices = setdiff(1:dim_all, ψ_indices_int)
        dim_ω = length(ω_indices)

        # No parameters at all
        if dim_ω == 0
            return Float64[], lnlike_θ([])
        end

        # Use profile_point for MLE (empty ψ_indices_int)
        θ_opt, _, lnlike_opt, _ = profile_point(
            lnlike_θ, Float64[], ψ_indices_int,
            θ_bounds_lower, θ_bounds_upper, ω_initial;
            ω_initial_extras=ω_initial_extras,
            method=method, local_method=local_method,
            xtol_rel=xtol_rel, ftol_rel=ftol_rel,
            optmaxtime=optmaxtime, popsize=popsize,
            track_convergence=false
        )
        return θ_opt, lnlike_opt
    end

    # Build Cartesian grid for interest parameters
    ψ_bounds_lower = θ_bounds_lower[ψ_indices_int]
    ψ_bounds_upper = θ_bounds_upper[ψ_indices_int]

    ψ_grids = Vector{Vector{Float64}}(undef, dim_ψ)
    for i in 1:dim_ψ
        if length(grid_steps) == 1
            ψ_grids[i] = collect(LinRange(ψ_bounds_lower[i], ψ_bounds_upper[i], grid_steps[1]))
        else
            ψ_grids[i] = collect(LinRange(ψ_bounds_lower[i], ψ_bounds_upper[i], grid_steps[i]))
        end
    end

    if dim_ψ == 2
        if track_convergence
            θ_values, lnlike_values, convergence_outcomes = _profile_grid_2d_structured(
                lnlike_θ, ψ_grids, ψ_indices_int,
                θ_bounds_lower, θ_bounds_upper, ω_initial;
                ω_initial_extras=ω_initial_extras,
                method=method, local_method=local_method,
                xtol_rel=xtol_rel, ftol_rel=ftol_rel,
                optmaxtime=optmaxtime, popsize=popsize,
                use_distributed=use_distributed,
                n_chunks=n_chunks, worker_pool=worker_pool,
                track_convergence=true,
                snake_direction=snake_direction
            )
        else
            θ_values, lnlike_values = _profile_grid_2d_structured(
                lnlike_θ, ψ_grids, ψ_indices_int,
                θ_bounds_lower, θ_bounds_upper, ω_initial;
                ω_initial_extras=ω_initial_extras,
                method=method, local_method=local_method,
                xtol_rel=xtol_rel, ftol_rel=ftol_rel,
                optmaxtime=optmaxtime, popsize=popsize,
                use_distributed=use_distributed,
                n_chunks=n_chunks, worker_pool=worker_pool,
                track_convergence=false,
                snake_direction=snake_direction
            )
        end

        lnlike_values = lnlike_values .- maximum(lnlike_values)
        if track_convergence
            return θ_values, lnlike_values, convergence_outcomes
        else
            return θ_values, lnlike_values
        end
    end

    # Convert Cartesian product to vector of vectors for 1D / higher-dimensional grids
    ψ_combinations = Base.product(ψ_grids...)
    ψ_grid_raw = [collect(ψᵢ) for ψᵢ in ψ_combinations]
    ψ_grid = vec(ψ_grid_raw)

    # Choose sequential or distributed execution
    if use_distributed
        # Distributed execution (track_convergence not supported)
        if track_convergence
            @warn "track_convergence=true not supported with use_distributed=true, ignoring"
        end
        θ_values, lnlike_values = profile_grid_distributed(
            lnlike_θ, ψ_grid, ψ_indices_int,
            θ_bounds_lower, θ_bounds_upper, ω_initial;
            ω_initial_extras=ω_initial_extras,
            method=method, local_method=local_method,
            xtol_rel=xtol_rel, ftol_rel=ftol_rel,
            optmaxtime=optmaxtime, popsize=popsize,
            n_chunks=n_chunks,
            worker_pool=worker_pool
        )
    else
        # Sequential execution
        if track_convergence
            θ_values, lnlike_values, convergence_outcomes = profile_grid_sequential(
                lnlike_θ, ψ_grid, ψ_indices_int,
                θ_bounds_lower, θ_bounds_upper, ω_initial;
                ω_initial_extras=ω_initial_extras,
                method=method, local_method=local_method,
                xtol_rel=xtol_rel, ftol_rel=ftol_rel,
                optmaxtime=optmaxtime, popsize=popsize,
                track_convergence=true
            )
        else
            θ_values, lnlike_values = profile_grid_sequential(
                lnlike_θ, ψ_grid, ψ_indices_int,
                θ_bounds_lower, θ_bounds_upper, ω_initial;
                ω_initial_extras=ω_initial_extras,
                method=method, local_method=local_method,
                xtol_rel=xtol_rel, ftol_rel=ftol_rel,
                optmaxtime=optmaxtime, popsize=popsize,
                track_convergence=false
            )
        end
    end

    # Unshuffle results back to column-major order for callers expecting reshape
    if dim_ψ == 2
        n1, n2 = length(ψ_grids[1]), length(ψ_grids[2])
        # Build unshuffle indices: map snake order back to column-major
        snake_to_colmajor = Vector{Int}(undef, n1 * n2)

        if snake_direction == :column
            # Column-wise snake: was vec(matrix) with reversed even columns
            for j in 1:n2
                for i in 1:n1
                    snake_idx = (j - 1) * n1 + (iseven(j) ? (n1 - i + 1) : i)
                    colmajor_idx = (j - 1) * n1 + i
                    snake_to_colmajor[snake_idx] = colmajor_idx
                end
            end
        else  # :row
            # Row-wise snake: was vec(matrix') with reversed even rows
            for i in 1:n1
                for j in 1:n2
                    snake_idx = (i - 1) * n2 + (iseven(i) ? (n2 - j + 1) : j)
                    colmajor_idx = (j - 1) * n1 + i
                    snake_to_colmajor[snake_idx] = colmajor_idx
                end
            end
        end

        # Reorder results
        θ_values_reordered = similar(θ_values)
        lnlike_values_reordered = similar(lnlike_values)
        for (snake_idx, colmajor_idx) in enumerate(snake_to_colmajor)
            θ_values_reordered[colmajor_idx] = θ_values[snake_idx]
            lnlike_values_reordered[colmajor_idx] = lnlike_values[snake_idx]
        end
        θ_values = θ_values_reordered
        lnlike_values = lnlike_values_reordered
        if track_convergence
            convergence_outcomes_reordered = similar(convergence_outcomes)
            for (snake_idx, colmajor_idx) in enumerate(snake_to_colmajor)
                convergence_outcomes_reordered[colmajor_idx] = convergence_outcomes[snake_idx]
            end
            convergence_outcomes = convergence_outcomes_reordered
        end
    end

    # Normalize likelihood values (as before)
    lnlike_values = lnlike_values .- maximum(lnlike_values)

    # Return in original format
    if track_convergence
        return θ_values, lnlike_values, convergence_outcomes
    else
        return θ_values, lnlike_values
    end
end

function construct_upper_lower_profile_wise_CIs_for_mean(
    distrib_ψω, ψω_values, lnlike_ψ_values; l_level=95, df=nothing)
    """
    Compute confidence intervals for mean predictions using profile likelihood.
    Works with arbitrary dimensional interest parameter ψ.

    Parameters:
    - distrib_ψω: Distribution function taking full parameter vector (ψ,ω)
    - ψω_values: Array of parameter vectors (each combining interest and nuisance parameters)
    - lnlike_ψ_values: Log-likelihood values for each parameter combination
    - l_level: Confidence level, e.g. 95 for 95% CI (default: 95)
    - df: Degrees of freedom (default: dimension of full parameter ψω)

    Returns:
    - lower: Lower bounds of confidence interval
    - upper: Upper bounds of confidence interval
    - pred_matrix: Matrix of predictions at each parameter value
    """
    if isnothing(df)
        print("assuming df:")
        print(length(ψω_values[1]))
        df = length(ψω_values[1])
    end
    threshold = -quantile(Chisq(df), l_level/100)/2

    # Filter by likelihood threshold
    ψω_filtered = ψω_values[lnlike_ψ_values .> threshold]

    # One predicted mean vector per column
    pred_matrix = stack(mean.(distrib_ψω.(ψω_filtered)))
    lower = minimum(pred_matrix, dims=2)
    upper = maximum(pred_matrix, dims=2)

    return lower, upper, pred_matrix
end

# --------------------------------------------------------
# Specialized Methods (1D/2D)
# --------------------------------------------------------
function get_1D_profiles_from_2D(ψ_values, lnlike_ψ_values)
    """
    Extract 1D profile likelihoods from 2D grid by maximizing over each parameter.

    Parameters:
    - ψ_values: Array of 2D parameter vectors from grid evaluation
    - lnlike_ψ_values: Log-likelihood values at each grid point

    Returns:
    - ψ1_values: Unique values of first parameter
    - ψ2_values: Unique values of second parameter
    - like_ψ1_values: Profile likelihood for first parameter
    - like_ψ2_values: Profile likelihood for second parameter

    Note: Profile likelihoods obtained by maximizing over the other parameter
    """
    # Split into grid components. Need unique to undo Cartesian product
    ψ1_values = unique([ψ1 for (ψ1, _) in ψ_values])
    ψ2_values = unique([ψ2 for (_, ψ2) in ψ_values])

    # Reshape to grid format
    lnlike_ψ_values = reshape(lnlike_ψ_values, length(ψ1_values), length(ψ2_values))

    # Convert to likelihood scale. Note: input assumed normalized
    like_ψ_values = exp.(lnlike_ψ_values)

    # Get profile likelihoods by maximizing over other parameter
    like_ψ1_values = maximum(like_ψ_values, dims=2)
    like_ψ2_values = maximum(like_ψ_values, dims=1)

    # Ensure profiles are 1D vectors
    like_ψ1_values = vec(like_ψ1_values)
    like_ψ2_values = vec(like_ψ2_values)

    return ψ1_values, ψ2_values, like_ψ1_values, like_ψ2_values
end
