# --------------------------------------------------------
# Parameter Scaling Methods
# --------------------------------------------------------

"""
    varimax_rotation(N_perp; n_restarts=200, threshold=1e-2, gamma=1.0)

Apply varimax rotation to N_perp basis to encourage sparse, interpretable loadings.
Uses multiple random restarts to escape local optima.

Optional interpretability enhancement: often helps produce sparse, human-readable
combinations within span(N_perp) (e.g., ratio/product structure) rather than dense mixtures.
For this paper's single-stage IIR focus, use as a presentation aid rather than a required step.

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
- Thresholding (`threshold > 0`) can break strict orthogonality; use `threshold=0`
  if strict orthogonality is required
- Empty basis (`k=0`) is treated as a no-op and returned unchanged
- Zero-norm columns are treated as invalid input and raise an error
- Tentative observation: SVD output from symmetric problems may need more restarts
  than typical factor analysis applications. Further investigation needed.

# Reference
Kaiser, H. F. (1958). The varimax criterion for analytic rotation in factor analysis.
"""
function varimax_rotation(N_perp; n_restarts=200, threshold=1e-2, gamma=1.0)
    # Note: Requires FactorLoadingMatrices to be loaded
    n, k = size(N_perp)

    if k == 0
        return copy(N_perp)
    end

    if n_restarts < 1
        throw(ArgumentError("n_restarts must be >= 1"))
    end

    # Validate input columns
    tol = eps(Float64)
    col_norms = [norm(N_perp[:, i]) for i in 1:k]
    zero_cols = findall(c -> c <= tol, col_norms)
    if !isempty(zero_cols)
        throw(ArgumentError("varimax_rotation received zero-norm columns at indices $(collect(zero_cols)). This usually indicates a malformed basis matrix."))
    end

    N_norm = N_perp ./ col_norms'

    # Rotation objective consistent with gamma
    function varimax_objective(L)
        n_rows, _ = size(L)
        sum(sum(L.^4, dims=1) .- gamma * (sum(L.^2, dims=1).^2) ./ n_rows)
    end

    # Multiple random restarts to escape local optima
    best_obj = -Inf
    best_rotated = N_norm

    for trial in 1:n_restarts
        # Random orthogonal rotation as starting point
        Q_rand = Matrix(qr(randn(k, k)).Q)
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

    # Re-orthonormalize via QR (before thresholding)
    Q_final = Matrix(qr(best_rotated).Q)
    rotated = Q_final .* col_norms'

    # Threshold small entries (can break strict orthogonality)
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
      `ψ_to_θ(ψ) = a_func_inv(A \\ a_func(ψ))`.
    """
    # Forward and inverse transformations
    A = evecs_scaled'
    xytoXY(xy) = a_func_inv(A * a_func(xy))
    XYtoxy(XY) = a_func_inv(A \ a_func(XY))

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


