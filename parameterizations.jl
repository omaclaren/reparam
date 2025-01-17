using LinearAlgebra

# --------------------------------------------------------
# Parameter Scaling Methods
# --------------------------------------------------------
function scale_and_round(evecs; round_within=0.5, column_scales=nothing)
    """
    Scale and round eigenvectors for more interpretable parameter transformations.

    Note mainly used when using coarser grid for parameter estimation. 
 
    Parameters:
    - evecs: Matrix of eigenvectors
    - round_within: Threshold for rounding (default: 0.5)
    - column_scales: Vector of scaling factors for each column (optional)
 
    Returns:
    - Matrix of scaled and rounded eigenvectors
    """
    if column_scales === nothing
        column_scales = ones(size(evecs,2))
    end
 
    # Scale first
    evecs_scaled = similar(evecs)
    evecs_scaled_rounded = similar(evecs)
    
    # Rescale each column so smallest non-zero is one
    for (i, col) in enumerate(eachcol(evecs))
        min_nonzero = col[argmin(abs.(col[abs.(col) .> round_within]))]
        evecs_scaled[:, i] = col / min_nonzero
    end
    
    # Round last
    evecs_scaled_rounded = round.(evecs_scaled/round_within)*round_within*diagm(column_scales)
    
    return evecs_scaled_rounded
 end

# --------------------------------------------------------
# Coordinate Transformation Methods
# --------------------------------------------------------
function reparam(evecs_scaled; a_func=x->log.(x), a_func_inv=x->exp.(x))
    """
    Construct parameter transformation based on scaled eigenvectors.
    
    Parameters:
    - evecs_scaled: Matrix of scaled eigenvectors
    - a_func: Component-wise transformation (default: log)
    - a_func_inv: Inverse of component-wise transformation (default: exp)
    
    Returns:
    - (xytoXY, XYtoxy): Tuple of forward and inverse transformation functions
    """
    # Forward and inverse transformations
    xytoXY(xy) = a_func_inv(evecs_scaled'*a_func(xy))
    XYtoxy(XY) = a_func_inv(inv(evecs_scaled')*a_func(XY))
    
    return xytoXY, XYtoxy
end

function construct_ϕ_XY(ϕ_xy, XYtoxy)
    """
    Construct auxiliary mapping in transformed coordinates.

    Note XY is of arbitrary dimension (not necessarily two).

    Auxiliary mapping maps mechanistic parameters to data distribution parameters.

    Parameters:
    - ϕ_xy: Auxiliary mapping in original coordinates (mechanistic → distribution parameters)
    - XYtoxy: Transformation from new (XY) to original (xy) coordinates

    Returns: Function computing auxiliary mapping in XY coordinates, i.e. XY -> distribution parameters.
    """
    return XY -> ϕ_xy(XYtoxy(XY))
end

function construct_lnlike_XY(lnlike_xy, XYtoxy)
    """
    Construct log-likelihood function in transformed coordinates.

    Note XY is of arbitrary dimension (not necessarily two).

    Parameters:
    - lnlike_xy: Original coordinate log-likelihood function
    - XYtoxy: Transformation from new (XY) to original (xy) coordinates

    Returns: Log-likelihood function taking XY coordinates
    """
    return XY -> lnlike_xy(XYtoxy(XY))
end

function construct_distrib_XY(distrib_xy, XYtoxy)
    """
    Construct distribution function in transformed coordinates.

    Note XY is of arbitrary dimension (not necessarily two).

    Parameters:
    - distrib_xy: Original coordinate distribution function
    - XYtoxy: Transformation from new (XY) to original (xy) coordinates

    Returns: Distribution function taking XY coordinates
    """
    return XY -> distrib_xy(XYtoxy(XY))
end 

# --------------------------------------------------------
# Parameter Re-ordering Methods
# --------------------------------------------------------

function construct_ψω_to_θ_indices(dim_all, ψ_indices, ω_indices)
    """
    Create mapping between parameter orderings when split into interest (ψ)
    and nuisance (ω) parameters.

    Parameters:
    - dim_all: Total number of parameters
    - ψ_indices: Indices of parameters of interest (can be multi-dimensional)
    - ω_indices: Indices of nuisance parameters

    Returns: Vector defining the mapping from (ψ,ω) ordering to θ ordering
    """
    rearrange_indices = zeros(Int, dim_all)
    
    # Map interest parameters
    for (i, ti) in enumerate(ψ_indices)
        rearrange_indices[ti] = i
    end

    # Map nuisance parameters
    n_target = length(ψ_indices)
    for (i, ti) in enumerate(ω_indices)
        rearrange_indices[ti] = n_target + i
    end

    return rearrange_indices
end

# --------------------------------------------------------
# Two-dimensional Constraint and Grid Handling Methods
# --------------------------------------------------------

function construct_2D_internal_constraint_box(lbs, ubs, lb_funcs, ub_funcs;
    grid_steps=[100], safety_factors=[0.0, 0.0])
    """
    Construct box constraints allowing a simple Cartesian product domain inside more complex 
    feasible parameter region.

    Parameters:
    - lbs: Lower bounds vector
    - ubs: Upper bounds vector
    - lb_funcs: Vector of functions giving lower bounds
    - ub_funcs: Vector of functions giving upper bounds
    - grid_steps: Number of grid points for evaluating bounds (default: [100])
    - safety_factors: Additional buffer for bounds (default: [0.0, 0.0])

    Returns:
    - (new_lbs, new_ubs): Tuple of adjusted lower and upper bounds vectors that define
    a rectangular region guaranteed to be within feasible space
    """
    # Validate dimensions
    @assert length(lbs) == 2 "Lower bounds vector 'lbs' must be of length 2."
    @assert length(ubs) == 2 "Upper bounds vector 'ubs' must be of length 2."
    @assert length(lb_funcs) == 2 "Lower bound functions 'lb_funcs' must be of length 2."
    @assert length(ub_funcs) == 2 "Upper bound functions 'ub_funcs' must be of length 2."

    # Set up grids based on input info
    grids = Vector{Vector{Float64}}(undef, 2)
    for i in 1:2
        if length(grid_steps) == 1
            grids[i] = LinRange(lbs[i], ubs[i], grid_steps[1])
        else
            grids[i] = LinRange(lbs[i], ubs[i], grid_steps[i])
        end
    end

    # Lower and upper of first across second
    lb1 = max(maximum(lb_funcs[1].(grids[2])), lbs[1]) + safety_factors[1]
    ub1 = min(minimum(ub_funcs[1].(grids[2])), ubs[1]) - safety_factors[1]

    # Lower and upper of second across first
    lb2 = max(maximum(lb_funcs[2].(grids[1])), lbs[2]) + safety_factors[2]
    ub2 = min(minimum(ub_funcs[2].(grids[1])), ubs[2]) - safety_factors[2]

    # Validate that new bounds are feasible
    if lb1 >= ub1
        throw(DomainError("No feasible region in first dimension (x₁): lb1 >= ub1 ($lb1 >= $ub1)."))
    end
    if lb2 >= ub2
        throw(DomainError("No feasible region in second dimension (x₂): lb2 >= ub2 ($lb2 >= $ub2)."))
    end

    new_lbs = [lb1, lb2]
    new_ubs = [ub1, ub2]

    return new_lbs, new_ubs
end

