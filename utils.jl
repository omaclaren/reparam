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

function generate_initial_guesses(bounds_lower, bounds_upper, n_guesses;
                                   reference_point=nothing, perturbation_scale=0.15)
    """
    Generate a collection of initial guesses for optimization.

    Parameters:
    - bounds_lower: Lower bounds for parameters
    - bounds_upper: Upper bounds for parameters
    - n_guesses: Number of starting guesses to generate
    - reference_point: Optional reference point for adaptive continuation (default: nothing)
    - perturbation_scale: Scale for perturbations around reference point (default: 0.15 = 15% of range)

    Returns: Vector of parameter vectors, including:

    If reference_point provided (adaptive continuation mode):
    - n=1: Small random perturbation around reference
    - n=2: Perturbation toward lower bounds
    - n=3: Perturbation toward upper bounds
    - n≥4: Additional random perturbations

    If no reference_point (original mode):
    - n=1: Center point
    - n=2: Center point + lower corner
    - n=3: Center point + both corners
    - n≥4: Center point + both corners + (n-3) random guesses
    """
    dims = length(bounds_lower)
    guesses = Vector{Vector{Float64}}(undef, n_guesses)
    param_range = bounds_upper - bounds_lower

    if !isnothing(reference_point)
        # Adaptive continuation mode: perturb around reference point
        for i in 1:n_guesses
            if i == 1
                # Small random perturbation
                perturbation = perturbation_scale * param_range .* (rand(dims) .- 0.5)
                guesses[i] = clamp.(reference_point + perturbation, bounds_lower, bounds_upper)
            elseif i == 2
                # Perturbation toward lower bounds
                direction = bounds_lower - reference_point
                dir_norm = norm(direction)
                if dir_norm > 1e-10  # Guard against zero vector
                    step = perturbation_scale * dir_norm * normalize(direction)
                    guesses[i] = clamp.(reference_point + step, bounds_lower, bounds_upper)
                else
                    # At lower bound, use random perturbation instead
                    perturbation = perturbation_scale * param_range .* (rand(dims) .- 0.5)
                    guesses[i] = clamp.(reference_point + perturbation, bounds_lower, bounds_upper)
                end
            elseif i == 3
                # Perturbation toward upper bounds
                direction = bounds_upper - reference_point
                dir_norm = norm(direction)
                if dir_norm > 1e-10  # Guard against zero vector
                    step = perturbation_scale * dir_norm * normalize(direction)
                    guesses[i] = clamp.(reference_point + step, bounds_lower, bounds_upper)
                else
                    # At upper bound, use random perturbation instead
                    perturbation = perturbation_scale * param_range .* (rand(dims) .- 0.5)
                    guesses[i] = clamp.(reference_point + perturbation, bounds_lower, bounds_upper)
                end
            else
                # Additional random perturbations
                perturbation = perturbation_scale * param_range .* (rand(dims) .- 0.5)
                guesses[i] = clamp.(reference_point + perturbation, bounds_lower, bounds_upper)
            end
        end
    else
        # Original mode: use fixed points based on bounds
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
    end

    return guesses
end