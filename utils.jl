function construct_observation_matrix(grid_obs, grid_fine; distance_func=nothing)
    """
    Construct sparse observation matrix mapping from fine grid to observation points.
    
    Parameters:
    - grid_obs: Vector of observation indices or points
    - grid_fine: Vector representing the fine grid (either indices or points)
    - distance_func: Optional function to compute distances between points.
                    If nothing, assumes grid_obs contains indices into grid_fine.
    
    Returns:
    - Sparse matrix mapping from fine grid to observations
    """
    n_obs = length(grid_obs)
    n_fine = length(grid_fine)
    
    if distance_func === nothing  # Assume grid_obs contains indices
        return sparse(1:n_obs, grid_obs, ones(n_obs), n_obs, n_fine)
    else  # Use the provided distance function for spatial points
        # Precompute all distances
        distances = [distance_func(x_fine, x_obs) for x_fine in grid_fine, x_obs in grid_obs]
        
        # Find closest fine grid point for each observation
        rows = Int[]
        cols = Int[]
        vals = Float64[]
        
        for i in 1:n_obs
            _, j = findmin(distances[:, i])
            push!(rows, i)
            push!(cols, j)
            push!(vals, 1.0)
        end
        
        return sparse(rows, cols, vals, n_obs, n_fine)
    end
end

function finite_diff_gradient(f, θ; h=1e-8)
    """
    Compute gradient using finite differences.

    Not typically used -- use ForwardDiff.jl instead -- but included for completeness.

    Parameters:
    - f: Function to differentiate
    - θ: Parameter vector at which to evaluate gradient
    - h: Step size for finite difference (default: 1e-8)

    Returns:
    - Vector containing numerical gradient
    """
    dim = length(θ)
    numerical_gradient = similar(θ)
    
    for i in 1:dim
        θ_plus = copy(θ)
        θ_minus = copy(θ)
        θ_plus[i] += h
        θ_minus[i] -= h
        numerical_gradient[i] = (f(θ_plus) - f(θ_minus))/(2h)
    end
    
    return numerical_gradient
end

function generate_initial_guesses(bounds_lower, bounds_upper, n_guesses)
    """
    Generate a collection of initial guesses for optimization.

    Parameters:
    - bounds_lower: Lower bounds for parameters
    - bounds_upper: Upper bounds for parameters
    - n_guesses: Number of starting guesses to generate

    Returns: Vector of parameter vectors, including:
    - n=1: Center point
    - n=2: Center point + lower corner
    - n=3: Center point + both corners
    - n≥4: Center point + both corners + (n-3) random guesses
    """
    dims = length(bounds_lower)
    guesses = Vector{Vector{Float64}}(undef, n_guesses)
    
    if n_guesses == 1
        guesses[1] = 0.5 * (bounds_lower + bounds_upper)
    elseif n_guesses == 2
        guesses[1] = 0.5 * (bounds_lower + bounds_upper)
        guesses[2] = bounds_lower
    elseif n_guesses == 3
        guesses[1] = 0.5 * (bounds_lower + bounds_upper)
        guesses[2] = bounds_lower
        guesses[3] = bounds_upper
    else
        guesses[1] = 0.5 * (bounds_lower + bounds_upper)
        guesses[2] = bounds_lower
        guesses[3] = bounds_upper
        for i in 4:n_guesses
            guesses[i] = bounds_lower + rand(dims) .* (bounds_upper - bounds_lower)
        end
    end
    
    return guesses
end