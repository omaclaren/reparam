# ----------------------------------------------------------------
# Note: NLopt is not thread-safe
# ----------------------------------------------------------------
# Multi-start optimization runs sequentially to avoid NLopt threading issues.
# For parallelization, use Distributed.jl with separate processes instead.

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
    - method_type: :auto for automatic differentiation (default), otherwise finite differences
    - compute_svd: Whether to compute and return SVD (default: false)
    
    Returns:
    - If compute_svd=false: Just the Jacobian matrix
    - If compute_svd=true: Tuple of (Jacobian, SVD factorization)
    """
    if method_type === :auto
        J = ForwardDiff.jacobian(ϕ_func, θ)
    else
        println("warning finite difference not implemented, no Jacobian")
        # todo finite diff with checks
        #J = finite_diff_gradient(ϕ_func, θ)
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
    - method_type: :auto for automatic Hessian computation (default)
    - return_h: Whether to return Hessian matrix (default: true)

    Returns: 
    - If return_h=true: Tuple of (quadratic approximation function, Hessian matrix)
    - If return_h=false: Quadratic approximation function only
    """
    if method_type === :auto
        H = -ForwardDiff.hessian(lnlike, θ_est)
    else
        println("warning finite difference not implemented, no Hessian")
        # todo finite diff 
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
- After the first point, ω_initial_extras (if provided) are regenerated around
  the adaptive continuation point for better local exploration
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

        # After first grid point, regenerate extras around continuation point
        # This provides local exploration while maintaining continuation benefit
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

Execute profile likelihood over a grid using distributed parallel execution.

Splits grid into contiguous chunks, runs each chunk via `pmap` using `profile_grid_sequential()`,
and concatenates results. Adaptive continuation works within each chunk but not between chunks.

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


function profile_target(lnlike_θ, ψ_indices, θ_bounds_lower, θ_bounds_upper, ω_initial;
    grid_steps=100, ω_initial_extras::Union{Nothing, Vector{Vector{Float64}}}=nothing,
    method=:LD_TNEWTON_PRECOND, local_method=:LD_TNEWTON_PRECOND, xtol_rel=1e-9, ftol_rel=1e-9,
    optmaxtime=120, popsize=50, track_convergence=false,
    use_distributed=false, n_chunks=nothing, worker_pool=nothing)
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
    - ω_initial_extras: : Vector of additional initial guesses for nuisance parameters, 
        where each guess is a vector of the same dimension as ω_initial (default: nothing)
    - method: Overall optimization method for nuisance parameters (default: :LD_TNEWTON_PRECOND)
    - local_method: Local optimization method if using a global method which requires it (default: :LD_TNEWTON_PRECOND)
    - xtol_rel: Relative tolerance in parameter values (default: 1e-9)
    - ftol_rel: Relative tolerance in function value (default: 1e-9)
    - optmaxtime: Maximum optimization time in seconds (default: 120)
    - popsize: Population size for global optimization methods (default: 10)
    - use_distributed: Use distributed parallel execution (default: false)
    - n_chunks: Number of chunks for distributed execution (default: worker count)
    - worker_pool: Optional Distributed.WorkerPool to target specific workers (default: all)

    Returns:
    - θ_values: Array of parameter vectors in original ordering
    - lnlike_ψ_values: Profile log-likelihood values (normalized to max of 0)

    Notes:
    - This function now delegates to profile_grid_sequential() for the actual work
    - The layered architecture enables future distributed execution (see profile_grid_distributed)
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

    # Convert Cartesian product to vector of vectors with snake ordering
    # Snake ordering alternates direction for each increment of second dimension
    # to maintain spatial continuity for warm-starting
    ψ_combinations = Base.product(ψ_grids...)
    ψ_grid_raw = [collect(ψᵢ) for ψᵢ in ψ_combinations]

    if dim_ψ == 2
        # For 2D: reshape, apply snake ordering, flatten
        n1, n2 = length(ψ_grids[1]), length(ψ_grids[2])
        ψ_grid_matrix = reshape(ψ_grid_raw, n1, n2)
        # Reverse every other column for snake pattern
        for j in 2:2:n2
            ψ_grid_matrix[:, j] = reverse(ψ_grid_matrix[:, j])
        end
        ψ_grid = vec(ψ_grid_matrix)
    else
        # For 1D or higher dimensions, use standard ordering
        ψ_grid = vec(ψ_grid_raw)
    end

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
        for j in 1:n2
            for i in 1:n1
                snake_idx = (j - 1) * n1 + (iseven(j) ? (n1 - i + 1) : i)
                colmajor_idx = (j - 1) * n1 + i
                snake_to_colmajor[snake_idx] = colmajor_idx
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
