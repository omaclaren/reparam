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
    Scale and round eigenvectors for interpretable integer-exponent transformations.

    Parameters:
    - evecs: Matrix of eigenvectors (columns = basis vectors)
    - round_within: Threshold for (i) choosing nonzero entries during scaling and
      (ii) tolerated distance from an integer exponent after scaling (default: 0.5)
    - column_scales: Vector of scaling factors for each column (optional)

    Returns:
    - Matrix of scaled and integer-rounded eigenvectors
    """
    if column_scales === nothing
        column_scales = ones(size(evecs, 2))
    end

    # Scale first
    scaled_columns = similar(evecs)

    # Rescale each column so smallest nonzero entry (above threshold) is one
    for (i, col) in enumerate(eachcol(evecs))
        above_threshold = abs.(col) .> round_within
        if any(above_threshold)
            col_above = col[above_threshold]
            min_nonzero = col_above[argmin(abs.(col_above))]
            scaled_columns[:, i] = col / min_nonzero
        else
            max_val = maximum(abs.(col))
            if max_val > eps()
                scaled_columns[:, i] = col / col[argmax(abs.(col))]
            else
                scaled_columns[:, i] = col
            end
        end
    end

    # Round to nearest integer exponents
    rounded_columns = round.(scaled_columns)

    # Hard check: significant entries must be close to integers
    for i in 1:size(scaled_columns, 2)
        sig = abs.(scaled_columns[:, i]) .> round_within
        if any(sig)
            max_dev = maximum(abs.(scaled_columns[sig, i] .- rounded_columns[sig, i]))
            if max_dev > round_within
                error("scale_and_round: column $i is not close enough to integer exponents (max deviation = $max_dev, tolerance = $round_within)")
            end
        end
    end

    # Apply optional column scales
    for i in 1:length(column_scales)
        rounded_columns[:, i] *= column_scales[i]
    end

    return rounded_columns
end

# --------------------------------------------------------
# Sparse monomial basis search helpers
# --------------------------------------------------------

function orthonormalize_columns(A::AbstractMatrix{<:Real})
    size(A, 2) == 0 && return zeros(Float64, size(A, 1), 0)
    Q = qr(Matrix{Float64}(A)).Q
    return Matrix(Q[:, 1:size(A, 2)])
end

function numerical_rank(A::AbstractMatrix{<:Real}; rtol=1e-8)
    if size(A, 1) == 0 || size(A, 2) == 0
        return 0
    end
    s = svdvals(Matrix{Float64}(A))
    isempty(s) && return 0
    τ = rtol * maximum(s)
    return count(>(τ), s)
end

function subspace_residual(Q::AbstractMatrix{<:Real}, v::AbstractVector{<:Real})
    vn = Float64.(v)
    vn ./= norm(vn)
    return norm(vn - Q * (Q' * vn))
end

function lexless(u::Vector{Int}, v::Vector{Int})
    for i in eachindex(u)
        if u[i] != v[i]
            return u[i] < v[i]
        end
    end
    return false
end

function combinations_indices(n::Int, k::Int)
    results = Vector{Vector{Int}}()
    current = Int[]

    function backtrack(start::Int, left::Int)
        if left == 0
            push!(results, copy(current))
            return
        end
        for i in start:(n - left + 1)
            push!(current, i)
            backtrack(i + 1, left - 1)
            pop!(current)
        end
    end

    backtrack(1, k)
    return results
end

function canonicalize_exponent(v::Vector{Int})
    all(==(0), v) && return nothing
    g = 0
    for x in v
        g = gcd(g, abs(x))
    end
    g = g == 0 ? 1 : g
    w = div.(v, g)
    nz = findfirst(!=(0), w)
    if nz !== nothing && w[nz] < 0
        w .*= -1
    end
    return w
end

function _check_label_names_length(v::AbstractVector, coord_names::AbstractVector)
    length(v) == length(coord_names) ||
        error("label construction: coefficient vector has length $(length(v)) but coord_names has length $(length(coord_names))")
end

function monomial_label(v::AbstractVector{<:Integer}, param_names::AbstractVector)
    _check_label_names_length(v, param_names)
    num = String[]
    den = String[]
    for (name, exp) in zip(param_names, v)
        if exp > 0
            push!(num, exp == 1 ? string(name) : string(name, "^", exp))
        elseif exp < 0
            nexp = -exp
            push!(den, nexp == 1 ? string(name) : string(name, "^", nexp))
        end
    end
    num_str = isempty(num) ? "1" : join(num, "*")
    den_str = isempty(den) ? "" : join(den, "*")
    return isempty(den_str) ? num_str : string(num_str, "/(", den_str, ")")
end

function linear_combination_label(v::AbstractVector{<:Integer}, coord_names::AbstractVector)
    _check_label_names_length(v, coord_names)
    terms = String[]

    for (name, coeff) in zip(coord_names, v)
        coeff == 0 && continue

        sign = coeff < 0 ? "-" : "+"
        abs_coeff = abs(coeff)
        body = abs_coeff == 1 ? string(name) : string(abs_coeff, "*", name)

        if isempty(terms)
            push!(terms, sign == "-" ? string("-", body) : body)
        else
            push!(terms, string(" ", sign, " ", body))
        end
    end

    return isempty(terms) ? "0" : join(terms, "")
end

function generate_candidate_dictionary(p::Int; s_max::Int=2, c_max::Int=1)
    seen = Set{String}()
    out = Vector{Vector{Int}}()
    coeff_choices = collect(filter(!=(0), -c_max:c_max))

    for k in 1:min(s_max, p)
        for idxs in combinations_indices(p, k)
            for coeffs in Iterators.product(ntuple(_ -> coeff_choices, k)...)
                v = zeros(Int, p)
                for (idx, coeff) in zip(idxs, coeffs)
                    v[idx] = coeff
                end
                vc = canonicalize_exponent(v)
                vc === nothing && continue
                key = join(vc, ",")
                if !(key in seen)
                    push!(seen, key)
                    push!(out, vc)
                end
            end
        end
    end

    sort!(out, lt=lexless)
    return out
end

Base.@kwdef struct MonomialBasisCandidate
    v::Vector{Int}
    label::String
    support::Int
    l1::Int
    linf::Int
    residual::Float64
end

candidate_key(c::MonomialBasisCandidate) = (c.support, c.l1, c.linf, c.residual)

function candidate_lt(a::MonomialBasisCandidate, b::MonomialBasisCandidate)
    ka = candidate_key(a)
    kb = candidate_key(b)
    if ka != kb
        return ka < kb
    end
    return lexless(a.v, b.v)
end

function projected_coordinates(Q::AbstractMatrix{<:Real}, v::AbstractVector{<:Real})
    vn = Float64.(v)
    vn ./= norm(vn)
    return Q' * vn
end

function projected_coordinate_matrix(Q::AbstractMatrix{<:Real}, candidates::Vector{MonomialBasisCandidate})
    isempty(candidates) && return zeros(Float64, size(Q, 2), 0)
    return hcat([projected_coordinates(Q, cand.v) for cand in candidates]...)
end

function normalized_direction(v::AbstractVector{<:Real})
    u = Float64.(v)
    u ./= norm(u)
    return u
end

function directional_information(M::AbstractMatrix{<:Real}, v::AbstractVector{<:Real})
    u = normalized_direction(v)
    return dot(u, M * u)
end

function conditional_information(M::AbstractMatrix{<:Real}, v::AbstractVector{<:Real},
                                 selected::Vector{MonomialBasisCandidate})
    info = directional_information(M, v)
    isempty(selected) && return info

    B = hcat([normalized_direction(cand.v) for cand in selected]...)
    u = normalized_direction(v)
    G = Matrix(B' * M * B)
    c = B' * M * u
    correction = dot(c, pinv(G) * c)
    return max(info - correction, 0.0)
end

function build_monomial_candidates(Q::AbstractMatrix{<:Real}, param_names::Vector{String};
                                    s_max::Int=2, c_max::Int=1)
    dict = generate_candidate_dictionary(length(param_names); s_max=s_max, c_max=c_max)
    candidates = MonomialBasisCandidate[]

    for v in dict
        resid = subspace_residual(Q, v)
        push!(candidates, MonomialBasisCandidate(
            v=v,
            label=monomial_label(v, param_names),
            support=count(!=(0), v),
            l1=sum(abs.(v)),
            linf=maximum(abs.(v)),
            residual=resid,
        ))
    end

    sort!(candidates, lt=candidate_lt)
    return dict, candidates
end

function greedy_simple_basis(Q::AbstractMatrix{<:Real}, accepted::Vector{MonomialBasisCandidate};
                             gain_rtol::Float64=1e-8)
    target_dim = size(Q, 2)
    selected = MonomialBasisCandidate[]
    coords = zeros(Float64, target_dim, 0)
    current_rank = 0
    support_levels = sort(unique(cand.support for cand in accepted))

    for support_level in support_levels
        for cand in accepted
            cand.support == support_level || continue
            a = projected_coordinates(Q, cand.v)
            new_rank = numerical_rank(hcat(coords, reshape(a, :, 1)); rtol=gain_rtol)
            if new_rank > current_rank
                push!(selected, cand)
                coords = hcat(coords, reshape(a, :, 1))
                current_rank = new_rank
                current_rank == target_dim && break
            end
        end
        current_rank == target_dim && break
    end

    return selected, current_rank
end

function greedy_informed_basis(Q::AbstractMatrix{<:Real}, accepted::Vector{MonomialBasisCandidate},
                               M::AbstractMatrix{<:Real}, σ1_sq::Float64;
                               gain_rtol::Float64=1e-8, score_rtol::Float64=1e-10)
    target_dim = size(Q, 2)
    selected = MonomialBasisCandidate[]
    coords = zeros(Float64, target_dim, 0)
    current_rank = 0
    remaining = copy(accepted)

    while current_rank < target_dim && !isempty(remaining)
        best_idx = 0
        best_score = -Inf
        best_tiebreak = nothing
        best_cand = nothing
        best_rank = current_rank
        best_a = nothing

        for (idx, cand) in enumerate(remaining)
            a = projected_coordinates(Q, cand.v)
            new_rank = numerical_rank(hcat(coords, reshape(a, :, 1)); rtol=gain_rtol)
            new_rank > current_rank || continue

            score = isempty(selected) ? directional_information(M, cand.v) / σ1_sq : conditional_information(M, cand.v, selected) / σ1_sq
            tiebreak = (cand.support, cand.l1, cand.linf, cand.residual)
            tol = score_rtol * max(1.0, abs(score), abs(best_score))

            if best_cand === nothing || score > best_score + tol || (abs(score - best_score) <= tol && (tiebreak < best_tiebreak || (tiebreak == best_tiebreak && lexless(cand.v, best_cand.v))))
                best_idx = idx
                best_score = score
                best_tiebreak = tiebreak
                best_cand = cand
                best_rank = new_rank
                best_a = a
            end
        end

        best_cand === nothing && break
        push!(selected, best_cand)
        coords = hcat(coords, reshape(best_a, :, 1))
        current_rank = best_rank
        deleteat!(remaining, best_idx)
    end

    return selected, current_rank
end

function monomial_basis_matrix(candidates::Vector{MonomialBasisCandidate}, p::Int)
    isempty(candidates) && return zeros(Float64, p, 0)
    return hcat([Float64.(cand.v) for cand in candidates]...)
end

function basis_labels(candidates::Vector{MonomialBasisCandidate})
    return [cand.label for cand in candidates]
end

function monomial_basis_labels(candidates::Vector{MonomialBasisCandidate}, coord_names::AbstractVector)
    return [monomial_label(cand.v, coord_names) for cand in candidates]
end

function linear_basis_labels(candidates::Vector{MonomialBasisCandidate}, coord_names::AbstractVector)
    return [linear_combination_label(cand.v, coord_names) for cand in candidates]
end

function basis_labels(candidates::Vector{MonomialBasisCandidate}, coord_names::AbstractVector; representation::Symbol=:monomial)
    if representation == :stored
        return basis_labels(candidates)
    elseif representation == :monomial
        return monomial_basis_labels(candidates, coord_names)
    elseif representation == :linear
        return linear_basis_labels(candidates, coord_names)
    else
        error("basis_labels: representation must be :stored, :monomial, or :linear; got $representation")
    end
end

function _simple_monomial_basis_search_fixed_support(U_basis::AbstractMatrix{<:Real}, param_names::Vector{String};
                                                      s_max::Int=2, c_max::Int=1,
                                                      residual_cap::Float64=1e-2,
                                                      gain_rtol::Float64=1e-8)
    Q = orthonormalize_columns(U_basis)
    target_dim = size(Q, 2)
    dict, all_candidates = build_monomial_candidates(Q, param_names; s_max=s_max, c_max=c_max)

    accepted = [cand for cand in all_candidates if cand.residual <= residual_cap]
    accepted_rank = numerical_rank(projected_coordinate_matrix(Q, accepted); rtol=gain_rtol)
    selected, selected_rank = greedy_simple_basis(Q, accepted; gain_rtol=gain_rtol)
    basis_ok = selected_rank == target_dim
    selected_residual_max = isempty(selected) ? nothing : maximum(cand.residual for cand in selected)

    return (
        dictionary=dict,
        all_candidates=all_candidates,
        accepted=accepted,
        accepted_rank=accepted_rank,
        selected=selected,
        selected_rank=selected_rank,
        target_dim=target_dim,
        basis_ok=basis_ok,
        selected_residual_max=selected_residual_max,
        # Backwards-compatible field name: now the maximum selected residual,
        # not an adaptively scanned acceptance threshold.
        successful_threshold=basis_ok ? selected_residual_max : nothing,
        residual_cap=residual_cap,
    )
end

function simple_monomial_basis_search(U_basis::AbstractMatrix{<:Real}, param_names::Vector{String};
                                      s_max::Int=2, c_max::Int=1,
                                      residual_cap::Float64=1e-2,
                                      gain_rtol::Float64=1e-8,
                                      retry_support::Bool=false)
    p = length(param_names)
    initial_s_max = min(s_max, p)
    result = _simple_monomial_basis_search_fixed_support(U_basis, param_names;
        s_max=initial_s_max,
        c_max=c_max,
        residual_cap=residual_cap,
        gain_rtol=gain_rtol)

    effective_s_max = initial_s_max
    if retry_support
        while !result.basis_ok && effective_s_max < p
            effective_s_max += 1
            result = _simple_monomial_basis_search_fixed_support(U_basis, param_names;
                s_max=effective_s_max,
                c_max=c_max,
                residual_cap=residual_cap,
                gain_rtol=gain_rtol)
        end
    end

    return (
        result...,
        initial_s_max=initial_s_max,
        effective_s_max=effective_s_max,
        retry_support=retry_support,
    )
end

function informed_monomial_basis_search(U_basis::AbstractMatrix{<:Real}, M::AbstractMatrix{<:Real},
                                        σ1_sq::Float64, param_names::Vector{String};
                                        s_max::Int=2, c_max::Int=1,
                                        residual_cap::Float64=1e-2,
                                        gain_rtol::Float64=1e-8)
    Q = orthonormalize_columns(U_basis)
    dict, all_candidates = build_monomial_candidates(Q, param_names; s_max=s_max, c_max=c_max)
    accepted = [cand for cand in all_candidates if cand.residual <= residual_cap]
    accepted_rank = numerical_rank(projected_coordinate_matrix(Q, accepted); rtol=gain_rtol)
    selected, selected_rank = greedy_informed_basis(Q, accepted, Matrix{Float64}(M), σ1_sq; gain_rtol=gain_rtol)

    return (
        dictionary=dict,
        all_candidates=all_candidates,
        accepted=accepted,
        accepted_rank=accepted_rank,
        selected=selected,
        selected_rank=selected_rank,
        target_dim=size(Q, 2),
        basis_ok=selected_rank == size(Q, 2),
        residual_cap=residual_cap,
        σ1_sq=σ1_sq,
    )
end

# --------------------------------------------------------
# Coordinate Transformation Methods
# --------------------------------------------------------
function reparam(basis_columns; a_func=x->log.(x), a_func_inv=x->exp.(x))
    """
    Construct log-linear forward and inverse transformations from a matrix of
    parameter combinations.

    This helper assumes **columns** of `basis_columns` correspond to the desired
    combinations in the transformed coordinates. Internally it forms
    ``A = basis_columns'`` so that the forward map is
    ``ψ = f^{-1}(A f(θ))`` with `f = a_func` (log by default).

    Parameters:
    - basis_columns: Matrix whose **columns** are the parameter combinations
      (for example the output of `monomial_basis_matrix(...)`, or any other
      column-stacked log-linear basis)
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
    A = basis_columns'
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
