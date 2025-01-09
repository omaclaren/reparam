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
    Compute Jacobian of φ mapping at given parameters, optionally with SVD.
    Works in any coordinate system.
    
    Parameters:
    - φ_func: Function implementing the φ mapping
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

function profile_target(lnlike_θ, ψ_indices, θ_bounds_lower, θ_bounds_upper, ω_initial;
    grid_steps=100, ω_initial_extras::Union{Nothing, Vector{Vector{Float64}}}=nothing,
    method=:LD_TNEWTON_PRECOND, local_method=:LD_TNEWTON_PRECOND, xtol_rel=1e-9, ftol_rel=1e-9, 
    optmaxtime=120, popsize=50)
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

    Returns:
    - θ_values: Array of parameter vectors in original ordering
    - lnlike_ψ_values: Profile log-likelihood values (normalized to max of 0)
    """
    # Get dimensions and indices
    dim_all = length(θ_bounds_lower)
    ω_indices = setdiff(1:dim_all, ψ_indices)
    dim_ψ = length(ψ_indices)
    dim_ω = length(ω_indices)

    # Extract bounds for nuisance parameters
    ω_bounds_lower = θ_bounds_lower[ω_indices]
    ω_bounds_upper = θ_bounds_upper[ω_indices]

    # Optimizer setup for nuisance parameters
    if dim_ω > 0 
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
    end

    # Check for point estimation case (no interest parameters)
    if dim_ψ == 0
        if dim_ω == 0
            return Float64[], lnlike_θ([])
        end
        
        # Try multiple starting points if provided
        best_lnlike = -Inf
        best_ω = similar(ω_initial)

        starting_points = [ω_initial]
        if !isnothing(ω_initial_extras)
            println("Type of starting_points: ", typeof(starting_points))
            println("Type of ω_initial_extras: ", typeof(ω_initial_extras))
            println("Element type of starting_points: ", eltype(starting_points))
            println("Element type of ω_initial_extras: ", eltype(ω_initial_extras))
            append!(starting_points, ω_initial_extras)
        end

        for ω₀ in starting_points
            opt.max_objective = construct_lnlike_to_max(lnlike_θ)
            (lnlike_opt, ω_opt) = optimize(opt, ω₀)
            if lnlike_opt > best_lnlike
                best_lnlike = lnlike_opt
                best_ω = ω_opt
            end
        end

        return best_ω, best_lnlike

    end

    # Set up grids for parameters of interest
    ψ_bounds_lower = θ_bounds_lower[ψ_indices]
    ψ_bounds_upper = θ_bounds_upper[ψ_indices]
    ψ_grids = Vector{Vector{Float64}}(undef, dim_ψ)
    for i in 1:dim_ψ
        if length(grid_steps) == 1
            ψ_grids[i] = LinRange(ψ_bounds_lower[i], ψ_bounds_upper[i], grid_steps[1])
        else
            ψ_grids[i] = LinRange(ψ_bounds_lower[i], ψ_bounds_upper[i], grid_steps[i])
        end
    end

    # Get Cartesian product of interest parameter grid
    ψ_combinations = Base.product(ψ_grids...)

    # Setup storage for results
    θ_values = Vector{Vector{Float64}}(undef, length(ψ_combinations))
    lnlike_ψ_values = Vector{Float64}(undef, length(ψ_combinations))

    # Get indices for reconstructing full parameter vector
    ψω_to_θ_indices = construct_ψω_to_θ_indices(dim_all, ψ_indices, ω_indices)

    # Profile over grid
    for (i, ψᵢ) in enumerate(ψ_combinations)
        ψω_to_θ = ψω -> ψω[ψω_to_θ_indices]
        if dim_ω > 0
            # Optimize nuisance parameters
            best_lnlike = -Inf
            best_ω = similar(ω_initial)

            starting_points = [ω_initial]
            if !isnothing(ω_initial_extras)
                append!(starting_points, ω_initial_extras)
            end

            for ω₀ in starting_points
                # Construct log-likelihood function for fixed current interest parameter value
                opt.max_objective = construct_lnlike_to_max(ω -> lnlike_θ(ψω_to_θ([ψᵢ..., ω...])))
                (lnlike_opt, ωᵢ_opt) = optimize(opt, ω₀)
                if lnlike_opt > best_lnlike
                    best_lnlike = lnlike_opt
                    best_ω = ωᵢ_opt
                end
            end

            θ_values[i] = ψω_to_θ([ψᵢ..., best_ω...])
            lnlike_ψ_values[i] = best_lnlike

            # Update initial guess for next iteration
            ω_initial = best_ω

        else
            # Pure gridding case
            θ_values[i] = ψω_to_θ([ψᵢ...])
            lnlike_ψ_values[i] = lnlike_θ(θ_values[i])
        end
    end

    # Normalize likelihood values
    lnlike_ψ_values = lnlike_ψ_values .- maximum(lnlike_ψ_values)

    return θ_values, lnlike_ψ_values
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
    - df: Degrees of freedom (default: dimension of interest parameter ψ)

    Returns: 
    - lower: Lower bounds of confidence interval
    - upper: Upper bounds of confidence interval
    - pred_matrix: Matrix of predictions at each parameter value
    """
    if isnothing(df)
        print("df")
        print(length(ψω_values))
        df = length(ψω_values)
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