# --------------------------------------------------------
# Parameter Scaling Methods
# --------------------------------------------------------

"""
    varimax_rotation(N_perp; n_restarts=200, threshold=1e-2, gamma=1.0)

Apply varimax rotation to N_perp basis to encourage sparse, interpretable loadings.
Uses multiple random restarts to escape local optima.

Essential for sequential IIR: ensures each stage produces parameter combinations
with local structure (e.g., products like n₁p₁, n₂p₂) rather than global mixtures.

# Arguments
- `N_perp`: n×k matrix of potentially identifiable basis vectors (columns)
- `n_restarts`: Number of random restarts (default: 200). More restarts help for
                symmetric problems with flat optimization surfaces.
- `threshold`: Threshold for zeroing small entries after rotation (default: 1e-2)
- `gamma`: Varimax parameter (1.0 for varimax, 0.0 for quartimax)

# Returns
Rotated N_perp with sparse structure

# Notes
- Requires FactorLoadingMatrices.jl package
- Multiple restarts essential: single random start often gives poor local optimum
- Tentative observation: SVD output from symmetric problems may need more restarts
  than typical factor analysis applications. Further investigation needed.

# Reference
Kaiser, H. F. (1958). The varimax criterion for analytic rotation in factor analysis.
"""
function varimax_rotation(N_perp; n_restarts=200, threshold=1e-2, gamma=1.0)
    # Note: Requires FactorLoadingMatrices to be loaded
    # Save original column norms
    col_norms = [norm(N_perp[:, i]) for i in 1:size(N_perp, 2)]
    N_norm = N_perp ./ col_norms'

    # Varimax objective function
    function varimax_objective(L)
        n, p = size(L)
        sum(sum(L.^4, dims=1) .- (sum(L.^2, dims=1).^2) ./ n)
    end

    # Multiple random restarts to escape local optima
    best_obj = -Inf
    best_rotated = N_norm

    for trial in 1:n_restarts
        # Random orthogonal rotation as starting point
        Q_rand = Matrix(qr(randn(size(N_perp, 2), size(N_perp, 2))).Q)
        candidate = N_norm * Q_rand

        # Apply varimax
        rotated = varimax(candidate; gamma=gamma)

        # Compute objective
        obj = varimax_objective(rotated)

        if obj > best_obj + 1e-6
            best_obj = obj
            best_rotated = rotated
        end
    end

    # Re-orthonormalize via QR
    Q_final = Matrix(qr(best_rotated).Q)
    rotated = Q_final .* col_norms'

    # Threshold small entries
    rotated[abs.(rotated) .< threshold] .= 0.0

    # Renormalize non-zero columns
    for j in 1:size(rotated, 2)
        col_norm = norm(rotated[:, j])
        if col_norm > 0
            rotated[:, j] ./= col_norm
        end
    end

    return rotated
end

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
        # Find values above threshold
        above_threshold = abs.(col) .> round_within
        if any(above_threshold)
            # Scale by smallest value above threshold
            col_above = col[above_threshold]
            min_nonzero = col_above[argmin(abs.(col_above))]
            evecs_scaled[:, i] = col / min_nonzero
        else
            # All values below threshold - scale by maximum absolute value
            max_val = maximum(abs.(col))
            if max_val > eps()
                evecs_scaled[:, i] = col / col[argmax(abs.(col))]
            else
                # Column is essentially zero
                evecs_scaled[:, i] = col
            end
        end
    end
    
    # Round last
    evecs_scaled_rounded = round.(evecs_scaled/round_within)*round_within
    # Apply column scales
    for i in 1:length(column_scales)
        evecs_scaled_rounded[:, i] *= column_scales[i]
    end

    return evecs_scaled_rounded
 end

# --------------------------------------------------------
# Coordinate Transformation Methods
# --------------------------------------------------------
function reparam(evecs_scaled; a_func=x->log.(x), a_func_inv=x->exp.(x))
    """
    Construct log-linear forward and inverse transformations from a matrix of
    parameter combinations.

    This helper assumes **columns** of `evecs_scaled` correspond to the desired
    combinations in the transformed coordinates. Internally it forms
    ``A = evecs_scaled'`` so that the forward map is
    ``ψ = f^{-1}(A f(θ))`` with `f = a_func` (log by default).

    Parameters:
    - evecs_scaled: Matrix whose **columns** are the parameter combinations
      (typically scaled and rounded output from `find_invariant_subspace`)
    - a_func: Component-wise transform to enforce positivity (default: `log`)
    - a_func_inv: Inverse of `a_func` (default: `exp`)

    Returns:
    - `(θ_to_ψ, ψ_to_θ)`: Tuple of forward and inverse transformation functions

    Notes:
    - If you prefer to work with a matrix `A` whose **rows** are combinations,
      you can bypass this helper and define the transform explicitly as:
      `θ_to_ψ(θ) = a_func_inv(A * a_func(θ))` and
      `ψ_to_θ(ψ) = a_func_inv(inv(A) * a_func(ψ))`.
    """
    # Forward and inverse transformations
    xytoXY(xy) = a_func_inv(evecs_scaled' * a_func(xy))
    XYtoxy(XY) = a_func_inv(inv(evecs_scaled') * a_func(XY))

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
