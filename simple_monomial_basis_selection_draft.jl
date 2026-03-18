# Draft implementation: simple fixed-dictionary monomial basis search.
#
# Purpose
# -------
# Given a numerically determined IIR target subspace U, build a pool of simple
# primitive monomial candidates and select a basis of size dim(U).
#
# Two modes are available:
# - singleton_first (alias: simple): first-success threshold relaxation with
#   singleton/support-first greedy rank-increasing sparse basis selection;
# - stepwise_informed (alias: informed): broader residual-cap pool, then greedy
#   identified-side basis selection using local J'J directional information /
#   conditional gain, with simplicity only as a tie-break.
#
# This is a deliberately simple draft implementation, not a general sparse-basis
# solver.
#
# Usage
# -----
#   julia --project=. simple_monomial_basis_selection_draft.jl transport --mode=singleton_first
#   julia --project=. simple_monomial_basis_selection_draft.jl transport --mode=stepwise_informed
#   julia --project=. simple_monomial_basis_selection_draft.jl both --mode=stepwise_informed --smax=2 --cmax=1 --residual-cap=1e-2
#
# This draft script currently implements basis views B and C:
# - B = singleton-first sparse basis
# - C = stepwise most-informative simple basis
# The orthogonal SVD basis (A) remains a separate numerical reference rather than
# a search mode in this script.
#
# For each example, the script now reports searches on both:
# - the identified side `N_perp`
# - the invariant null side `N` (if non-empty)
# If an informed search is requested on a space where the local `J'J` metric is
# degenerate, the script reports that and falls back explicitly to B.
# If B then fails at the requested support cap `s_max`, the script automatically
# retries once at support `s_max + 1` (up to the ambient dimension) and reports
# that support expansion explicitly.

if !@isdefined(ReparamTools)
    include("ReparamTools.jl")
    println("✓ ReparamTools module included")
else
    println("✓ ReparamTools module already included")
end

include(joinpath(@__DIR__, "examples", "RepressilatorModel.jl"))

using .ReparamTools
using .RepressilatorModel
using DifferentialEquations
using Distributions
using LinearAlgebra
using Printf
using Random
using Serialization

# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

function orthonormalize_columns(A::AbstractMatrix{<:Real})
    size(A, 2) == 0 && return zeros(Float64, size(A, 1), 0)
    Q = qr(Matrix{Float64}(A)).Q
    return Matrix(Q[:, 1:size(A, 2)])
end

function numerical_rank(A::AbstractMatrix{<:Real}; rtol=1e-8)
    if size(A, 2) == 0 || size(A, 1) == 0
        return 0
    end
    s = svdvals(Matrix{Float64}(A))
    isempty(s) && return 0
    τ = rtol * maximum(s)
    return count(>(τ), s)
end

function subspace_projector(Q::AbstractMatrix{<:Real})
    Qm = Matrix{Float64}(Q)
    return Qm * Qm'
end

function subspace_residual(Q::AbstractMatrix{<:Real}, v::AbstractVector{<:Real})
    vn = Float64.(v)
    vn ./= norm(vn)
    return norm(vn - Q * (Q' * vn))
end

function subspace_distance(A::AbstractMatrix{<:Real}, B::AbstractMatrix{<:Real})
    QA = orthonormalize_columns(A)
    QB = orthonormalize_columns(B)
    PA = subspace_projector(QA)
    PB = subspace_projector(QB)
    d1 = norm((I - PB) * QA)
    d2 = norm((I - PA) * QB)
    return max(d1, d2)
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

function monomial_label(v::AbstractVector{<:Integer}, param_names::Vector{String})
    num = String[]
    den = String[]
    for (name, exp) in zip(param_names, v)
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

function generate_candidate_dictionary(p::Int; s_max::Int=2, c_max::Int=1)
    s_max < 1 && error("s_max must be at least 1")
    c_max < 1 && error("c_max must be at least 1")

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

Base.@kwdef struct BasisCandidate
    v::Vector{Int}
    label::String
    support::Int
    l1::Int
    linf::Int
    residual::Float64
end

candidate_key(c::BasisCandidate) = (c.support, c.l1, c.linf, c.residual)

function candidate_lt(a::BasisCandidate, b::BasisCandidate)
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

function projected_coordinate_matrix(Q::AbstractMatrix{<:Real}, candidates::Vector{BasisCandidate})
    if isempty(candidates)
        return zeros(Float64, size(Q, 2), 0)
    end
    return hcat([projected_coordinates(Q, cand.v) for cand in candidates]...)
end

function projected_rank_gain(coords::AbstractMatrix{<:Real}, a::AbstractVector{<:Real})
    if size(coords, 2) == 0
        return norm(a)
    end
    QS = orthonormalize_columns(coords)
    return norm(a - QS * (QS' * a))
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
                                 selected::Vector{BasisCandidate})
    info = directional_information(M, v)
    isempty(selected) && return info

    B = hcat([normalized_direction(cand.v) for cand in selected]...)
    u = normalized_direction(v)
    G = Matrix(B' * M * B)
    c = B' * M * u
    correction = dot(c, pinv(G) * c)
    return max(info - correction, 0.0)
end

function informed_item_better(a, b; score_rtol::Float64=1e-10)
    tol = score_rtol * max(1.0, abs(a.score), abs(b.score))
    if a.score > b.score + tol
        return true
    elseif b.score > a.score + tol
        return false
    end
    return candidate_lt(a.cand, b.cand)
end

function greedy_simple_basis(Q::AbstractMatrix{<:Real}, accepted::Vector{BasisCandidate};
                             gain_rtol::Float64=1e-8)
    target_dim = size(Q, 2)
    selected = BasisCandidate[]
    coords = zeros(Float64, target_dim, 0)
    current_rank = 0
    trace = NamedTuple[]
    support_levels = sort(unique(cand.support for cand in accepted))

    for support_level in support_levels
        for cand in accepted
            cand.support == support_level || continue

            a = projected_coordinates(Q, cand.v)
            gain = projected_rank_gain(coords, a)
            new_rank = numerical_rank(hcat(coords, reshape(a, :, 1)); rtol=gain_rtol)
            keep = new_rank > current_rank

            push!(trace, (
                label=cand.label,
                v=copy(cand.v),
                support=cand.support,
                l1=cand.l1,
                linf=cand.linf,
                residual=cand.residual,
                projected_gain=gain,
                rank_before=current_rank,
                rank_after=new_rank,
                support_layer=support_level,
                selected=keep,
            ))

            if keep
                push!(selected, cand)
                coords = hcat(coords, reshape(a, :, 1))
                current_rank = new_rank
                current_rank == target_dim && break
            end
        end

        current_rank == target_dim && break
    end

    return selected, current_rank, trace
end

function greedy_informed_basis(Q::AbstractMatrix{<:Real}, accepted::Vector{BasisCandidate},
                               M::AbstractMatrix{<:Real};
                               σ1_sq::Float64,
                               gain_rtol::Float64=1e-8,
                               score_rtol::Float64=1e-10)
    target_dim = size(Q, 2)
    selected = BasisCandidate[]
    coords = zeros(Float64, target_dim, 0)
    current_rank = 0
    trace = NamedTuple[]
    remaining = copy(accepted)
    Mmat = Matrix{Float64}(M)

    while current_rank < target_dim && !isempty(remaining)
        best = nothing
        best_idx = 0

        for (idx, cand) in enumerate(remaining)
            a = projected_coordinates(Q, cand.v)
            gain = projected_rank_gain(coords, a)
            new_rank = numerical_rank(hcat(coords, reshape(a, :, 1)); rtol=gain_rtol)
            new_rank > current_rank || continue

            first_step = isempty(selected)
            info_rel = directional_information(Mmat, cand.v) / σ1_sq
            conditional_rel = conditional_information(Mmat, cand.v, selected) / σ1_sq
            score = first_step ? info_rel : conditional_rel

            item = (
                cand=cand,
                idx=idx,
                a=a,
                gain=gain,
                new_rank=new_rank,
                info_rel=info_rel,
                conditional_rel=conditional_rel,
                score=score,
                score_type=first_step ? "I_rel" : "Δ_rel",
            )

            if best === nothing || informed_item_better(item, best; score_rtol=score_rtol)
                best = item
                best_idx = idx
            end
        end

        best === nothing && break

        push!(selected, best.cand)
        coords = hcat(coords, reshape(best.a, :, 1))

        push!(trace, (
            label=best.cand.label,
            v=copy(best.cand.v),
            support=best.cand.support,
            l1=best.cand.l1,
            linf=best.cand.linf,
            residual=best.cand.residual,
            projected_gain=best.gain,
            rank_before=current_rank,
            rank_after=best.new_rank,
            score_type=best.score_type,
            score=best.score,
            info_rel=best.info_rel,
            conditional_rel=best.conditional_rel,
            selected=true,
        ))

        current_rank = best.new_rank
        deleteat!(remaining, best_idx)
    end

    return selected, current_rank, trace
end

function build_all_candidates(Q::AbstractMatrix{<:Real}, param_names::Vector{String};
                              s_max::Int=2, c_max::Int=1)
    dict = generate_candidate_dictionary(length(param_names); s_max=s_max, c_max=c_max)
    candidates = BasisCandidate[]

    for v in dict
        resid = subspace_residual(Q, v)
        push!(candidates, BasisCandidate(
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

function compute_local_metric(ϕ_func, θ0)
    J = compute_ϕ_Jacobian(ϕ_func, θ0)
    σ = svdvals(J)
    isempty(σ) && error("Jacobian has no singular values")
    return (
        J=J,
        M=J' * J,
        σ1=σ[1],
        σ1_sq=σ[1]^2,
    )
end

function metric_is_degenerate_on_space(U_basis::AbstractMatrix{<:Real},
                                       M::AbstractMatrix{<:Real},
                                       σ1_sq::Float64;
                                       metric_rtol::Float64=1e-12)
    Q = orthonormalize_columns(U_basis)
    size(Q, 2) == 0 && return true
    projected_metric = Matrix(Q' * M * Q)
    s = svdvals(projected_metric)
    isempty(s) && return true
    return maximum(s) <= metric_rtol * max(1.0, σ1_sq)
end

function evaluate_selected_basis(U_basis::AbstractMatrix{<:Real}, selected::Vector{BasisCandidate},
                                 target_dim::Int; basis_tol::Float64=1e-8)
    selected_basis = if isempty(selected)
        zeros(Float64, size(U_basis, 1), 0)
    else
        hcat([Float64.(cand.v) for cand in selected]...)
    end

    basis_distance = size(selected_basis, 2) == target_dim ? subspace_distance(U_basis, selected_basis) : Inf
    basis_ok = size(selected_basis, 2) == target_dim && basis_distance <= basis_tol
    τ_eff = isempty(selected) ? Inf : maximum(cand.residual for cand in selected)

    return selected_basis, basis_distance, basis_ok, τ_eff
end

function simple_monomial_basis_search(U_basis::AbstractMatrix{<:Real}, param_names::Vector{String};
                                      s_max::Int=2,
                                      c_max::Int=1,
                                      residual_cap::Float64=1e-2,
                                      gain_rtol::Float64=1e-8,
                                      basis_tol::Float64=1e-8)
    Q = orthonormalize_columns(U_basis)
    target_dim = size(Q, 2)
    dict, all_candidates = build_all_candidates(Q, param_names; s_max=s_max, c_max=c_max)

    candidate_thresholds = sort(unique(cand.residual for cand in all_candidates if cand.residual <= residual_cap))
    attempts = NamedTuple[]

    last_accepted = BasisCandidate[]
    last_accepted_rank = 0
    last_selected = BasisCandidate[]
    last_trace = NamedTuple[]
    last_selected_rank = 0
    last_basis_distance = Inf
    last_τ_eff = Inf

    if isempty(candidate_thresholds)
        return (
            selection_mode=:simple,
            pool_strategy=:first_success_threshold,
            Q=Q,
            dictionary=dict,
            all_candidates=all_candidates,
            accepted=last_accepted,
            accepted_rank=last_accepted_rank,
            selected=last_selected,
            trace=last_trace,
            attempts=attempts,
            candidate_thresholds=Float64[],
            target_dim=target_dim,
            selected_rank=last_selected_rank,
            basis_distance=last_basis_distance,
            basis_ok=false,
            successful_threshold=nothing,
            effective_threshold=last_τ_eff,
            residual_cap=residual_cap,
        )
    end

    for τ in candidate_thresholds
        accepted = [cand for cand in all_candidates if cand.residual <= τ]
        accepted_rank = numerical_rank(projected_coordinate_matrix(Q, accepted); rtol=gain_rtol)
        selected, selected_rank, trace = greedy_simple_basis(Q, accepted; gain_rtol=gain_rtol)
        _, basis_distance, basis_ok, τ_eff = evaluate_selected_basis(U_basis, selected, target_dim; basis_tol=basis_tol)

        push!(attempts, (
            threshold=τ,
            accepted_count=length(accepted),
            accepted_rank=accepted_rank,
            selected_rank=selected_rank,
            basis_distance=basis_distance,
            success=basis_ok,
        ))

        last_accepted = accepted
        last_accepted_rank = accepted_rank
        last_selected = selected
        last_trace = trace
        last_selected_rank = selected_rank
        last_basis_distance = basis_distance
        last_τ_eff = τ_eff

        if basis_ok
            return (
                selection_mode=:simple,
                pool_strategy=:first_success_threshold,
                Q=Q,
                dictionary=dict,
                all_candidates=all_candidates,
                accepted=accepted,
                accepted_rank=accepted_rank,
                selected=selected,
                trace=trace,
                attempts=attempts,
                candidate_thresholds=candidate_thresholds,
                target_dim=target_dim,
                selected_rank=selected_rank,
                basis_distance=basis_distance,
                basis_ok=true,
                successful_threshold=τ,
                effective_threshold=τ_eff,
                residual_cap=residual_cap,
            )
        end
    end

    return (
        selection_mode=:simple,
        pool_strategy=:first_success_threshold,
        Q=Q,
        dictionary=dict,
        all_candidates=all_candidates,
        accepted=last_accepted,
        accepted_rank=last_accepted_rank,
        selected=last_selected,
        trace=last_trace,
        attempts=attempts,
        candidate_thresholds=candidate_thresholds,
        target_dim=target_dim,
        selected_rank=last_selected_rank,
        basis_distance=last_basis_distance,
        basis_ok=false,
        successful_threshold=nothing,
        effective_threshold=last_τ_eff,
        residual_cap=residual_cap,
    )
end

function informed_monomial_basis_search(U_basis::AbstractMatrix{<:Real},
                                        M::AbstractMatrix{<:Real},
                                        σ1_sq::Float64,
                                        param_names::Vector{String};
                                        s_max::Int=2,
                                        c_max::Int=1,
                                        residual_cap::Float64=1e-2,
                                        gain_rtol::Float64=1e-8,
                                        basis_tol::Float64=1e-8,
                                        score_rtol::Float64=1e-10)
    Q = orthonormalize_columns(U_basis)
    target_dim = size(Q, 2)
    dict, all_candidates = build_all_candidates(Q, param_names; s_max=s_max, c_max=c_max)

    accepted = [cand for cand in all_candidates if cand.residual <= residual_cap]
    accepted_rank = numerical_rank(projected_coordinate_matrix(Q, accepted); rtol=gain_rtol)
    selected, selected_rank, trace = greedy_informed_basis(Q, accepted, M;
        σ1_sq=σ1_sq,
        gain_rtol=gain_rtol,
        score_rtol=score_rtol)

    _, basis_distance, basis_ok, τ_eff = evaluate_selected_basis(U_basis, selected, target_dim; basis_tol=basis_tol)

    return (
        selection_mode=:informed,
        pool_strategy=:residual_cap,
        Q=Q,
        dictionary=dict,
        all_candidates=all_candidates,
        accepted=accepted,
        accepted_rank=accepted_rank,
        selected=selected,
        trace=trace,
        attempts=NamedTuple[],
        candidate_thresholds=Float64[],
        target_dim=target_dim,
        selected_rank=selected_rank,
        basis_distance=basis_distance,
        basis_ok=basis_ok,
        successful_threshold=nothing,
        effective_threshold=τ_eff,
        residual_cap=residual_cap,
        σ1_sq=σ1_sq,
    )
end

# -----------------------------------------------------------------------------
# Example-specific target-subspace loaders
# -----------------------------------------------------------------------------

function load_repressilator_target_subspace(; input_file="nesi/repressilator_16nuisance_50x50_results.jls")
    isfile(input_file) || error("Repressilator results file not found: $input_file")
    results = deserialize(input_file)

    required = ["A_T_final", "n_ident", "param_names", "θ_MLE"]
    missing = filter(k -> !haskey(results, k), required)
    isempty(missing) || error("Missing keys in repressilator results: $(missing)")

    A_T_final = Matrix{Float64}(results["A_T_final"])
    n_ident = Int(results["n_ident"])
    param_names = Vector{String}(results["param_names"])
    θ_MLE = Vector{Float64}(results["θ_MLE"])

    X0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    T_end = 10000.0
    t_iir = LinRange(0, T_end, 501)

    function ϕ_iir_highprec(θ)
        sol_matrix = RepressilatorModel.solve_repressilator(t_iir, θ, X0; abstol=1e-10, reltol=1e-8)
        mRNA = sol_matrix[1:3, :]
        return vec(mRNA)
    end
    ϕ_iir_log(θ_log) = ϕ_iir_highprec(exp.(θ_log))
    θ_log_MLE = log.(θ_MLE)

    return (
        label="repressilator target subspace from saved profile basis",
        U_basis=A_T_final[:, 1:n_ident],
        N_perp_basis=A_T_final[:, 1:n_ident],
        N_basis=A_T_final[:, n_ident+1:end],
        param_names=param_names,
        ϕ_func=ϕ_iir_log,
        θ0=θ_log_MLE,
    )
end

function solve_transport_model(θ, x, L)
    y = Vector{eltype(θ)}(undef, length(x))
    mid_index = Int((length(x) - 1) / 2)

    β = L^2 * θ[3] / 8 * (1 / θ[2] - 1 / θ[1])
    α = θ[3] * L / (2 * θ[2]) - β / L

    H_1(x) = -θ[3] / (2 * θ[1]) * x^2 + α * x
    H_2(x) = -θ[3] / (2 * θ[2]) * x^2 + α * x + β

    for i in 1:mid_index
        y[i] = H_1(x[i])
    end
    for i in mid_index:length(x)
        y[i] = H_2(x[i])
    end
    return y
end

function load_transport_target_subspace()
    XY_log_MLE = [-1.620844, -2.274112, -2.276164]

    L = 100
    x = LinRange(0, L, 201)
    ϕ_func_xy = θ -> solve_transport_model(θ, x, L)
    XYtoxy_log(XY) = exp.(XY)
    ϕ_func_XY_log = construct_ϕ_XY(ϕ_func_xy, XYtoxy_log)

    _, N_inv, N_perp_inv, _ = find_invariant_subspace(
        ϕ_func_XY_log, XY_log_MLE; verbose=false)

    return (
        label="transport target subspace",
        U_basis=N_perp_inv,
        N_perp_basis=N_perp_inv,
        N_basis=N_inv,
        param_names=["T1", "T2", "R"],
        ϕ_func=ϕ_func_XY_log,
        θ0=XY_log_MLE,
    )
end

function load_stat_target_subspace(; poisson_limit::Bool=true)
    Random.seed!(12)

    if poisson_limit
        ϕ_xy = xy -> [xy[1] * xy[2], xy[1] * xy[2]]
        label = "stat_model target subspace (Poisson limit)"
    else
        ϕ_xy = xy -> [xy[1] * xy[2], xy[1] * xy[2] * (1 - xy[2])]
        label = "stat_model target subspace (non-limit)"
    end

    distrib_xy = xy -> Normal(ϕ_xy(xy)[1], sqrt(ϕ_xy(xy)[2]))

    xy_lower_bounds = [0.1, 0.0001]
    xy_upper_bounds = [500.0, 1.0]
    xy_initial = [50.0, 0.3]
    data = [21.9, 22.3, 12.8, 16.4, 16.4, 20.3, 16.2, 20.0, 19.7, 24.4]

    lnlike_xy = construct_lnlike_xy(distrib_xy, data)
    xytoXY_log(xy) = log.(xy)
    XYtoxy_log(XY) = exp.(XY)

    XY_log_lower_bounds = log.(xy_lower_bounds)
    XY_log_upper_bounds = log.(xy_upper_bounds)
    XY_log_initial = xytoXY_log(xy_initial)

    lnlike_XY_log = construct_lnlike_XY(lnlike_xy, XYtoxy_log)
    ϕ_XY_log = construct_ϕ_XY(ϕ_xy, XYtoxy_log)

    XY_log_MLE, _ = profile_target(
        lnlike_XY_log,
        Int[],
        XY_log_lower_bounds,
        XY_log_upper_bounds,
        XY_log_initial;
        grid_steps=[500],
    )

    _, N_inv, N_perp_inv, _ = find_invariant_subspace(ϕ_XY_log, XY_log_MLE; verbose=false)

    return (
        label=label,
        U_basis=N_perp_inv,
        N_perp_basis=N_perp_inv,
        N_basis=N_inv,
        param_names=["n", "p"],
        ϕ_func=ϕ_XY_log,
        θ0=XY_log_MLE,
    )
end

function mm_DE!(dS, S, θ, t)
    dS[1] = -θ[1] * S[1] / (θ[2] + S[1])
end

function mm_DE_limit!(dS, S, θ, t)
    dS[1] = -θ[1] * S[1] / θ[2]
end

function solve_mm_ode(t_save, θ, S0; solver=Rodas4(), limit=false)
    tspan = (0.0, maximum(t_save))
    ODE = limit ? mm_DE_limit! : mm_DE!
    prob = ODEProblem(ODE, [S0], tspan, θ)
    sol = solve(prob, solver, saveat=t_save, abstol=1e-12, reltol=1e-9)
    return sol[1, :]
end

function create_mm_ϕ_mapping(t, S0; limit=false)
    return θ -> solve_mm_ode(t, θ, S0; limit=limit)
end

function load_mm_target_subspace(; limit::Bool=false)
    Random.seed!(4321)

    T = 20
    NT = 201
    t = LinRange(0, T, NT)
    indices_fine = 1:NT
    NT_obs = 11
    indices_obs = 1:Int((NT - 1) / (NT_obs - 1)):NT
    obs_matrix = construct_observation_matrix(indices_obs, indices_fine)
    t_obs = t[indices_obs]

    S0 = 1.0
    σ = 0.05
    solver = Rodas4()
    ϕ_func_xy = create_mm_ϕ_mapping(t, S0; limit=limit)
    distrib_xy = xy -> MvNormal(solve_mm_ode(t_obs, xy, S0; solver=solver, limit=limit), σ^2 * I(NT_obs))

    xy_lower_bounds = [0.1, 0.1]
    xy_upper_bounds = [10.0, 50.0]
    xy_initial = 0.5 .* (xy_lower_bounds + xy_upper_bounds)
    data = [1.04, 0.66, 0.50, 0.36, 0.28, 0.18, 0.01, 0.08, 0.02, 0.07, 0.05]

    lnlike_xy = construct_lnlike_xy(distrib_xy, data; dist_type=:multi)
    xytoXY_log(xy) = log.(xy)
    XYtoxy_log(XY) = exp.(XY)

    XY_log_lower_bounds = log.(xy_lower_bounds)
    XY_log_upper_bounds = log.(xy_upper_bounds)
    XY_log_initial = xytoXY_log(xy_initial)

    lnlike_XY_log = construct_lnlike_XY(lnlike_xy, XYtoxy_log)
    ϕ_func_XY_log = construct_ϕ_XY(ϕ_func_xy, XYtoxy_log)

    XY_log_mle_initial_guesses = generate_initial_guesses(XY_log_lower_bounds, XY_log_upper_bounds, 3)
    XY_log_MLE, _ = profile_target(
        lnlike_XY_log,
        Int[],
        XY_log_lower_bounds,
        XY_log_upper_bounds,
        XY_log_initial;
        grid_steps=[500],
        ω_initial_extras=XY_log_mle_initial_guesses,
        method=:LN_BOBYQA,
    )

    _, N_inv, N_perp_inv, _ = find_invariant_subspace(ϕ_func_XY_log, XY_log_MLE; verbose=false)

    return (
        label=limit ? "mm_model target subspace (limit)" : "mm_model target subspace (non-limit)",
        U_basis=N_perp_inv,
        N_perp_basis=N_perp_inv,
        N_basis=N_inv,
        param_names=["nu", "K"],
        ϕ_func=ϕ_func_XY_log,
        θ0=XY_log_MLE,
    )
end

# -----------------------------------------------------------------------------
# Reporting helpers
# -----------------------------------------------------------------------------

function print_candidate_table(title::String, candidates::Vector{BasisCandidate})
    println("\n", repeat("=", 100))
    println(title)
    println(repeat("=", 100))
    if isempty(candidates)
        println("(none)")
        return
    end
    for (i, cand) in enumerate(candidates)
        @printf("%3d  %-20s  support=%d  l1=%d  linf=%d  residual=%.3e  v=%s\n",
            i, cand.label, cand.support, cand.l1, cand.linf, cand.residual, string(cand.v))
    end
end

function print_selection_trace(trace)
    println("\n", repeat("-", 100))
    println("Greedy singleton/support-layered selection trace")
    println(repeat("-", 100))
    isempty(trace) && (println("(none)"); return)
    for (i, step) in enumerate(trace)
        flag = step.selected ? "SELECT" : "skip"
        support_layer = hasproperty(step, :support_layer) ? step.support_layer : step.support
        @printf("%3d  %-6s  s=%d  %-20s  gain=%.3e  rank %d -> %d  residual=%.3e\n",
            i, flag, support_layer, step.label, step.projected_gain, step.rank_before, step.rank_after, step.residual)
    end
end

function print_informed_selection_trace(trace)
    println("\n", repeat("-", 100))
    println("Informed greedy selection trace")
    println(repeat("-", 100))
    isempty(trace) && (println("(none)"); return)
    for (i, step) in enumerate(trace)
        @printf("%3d  %-20s  %s=%.6e  I_rel=%.6e  Delta_rel=%.6e  rank %d -> %d  residual=%.3e\n",
            i,
            step.label,
            step.score_type,
            step.score,
            step.info_rel,
            step.conditional_rel,
            step.rank_before,
            step.rank_after,
            step.residual)
    end
end

function print_attempt_table(attempts)
    println("\n", repeat("-", 100))
    println("Threshold relaxation attempts")
    println(repeat("-", 100))
    isempty(attempts) && (println("(none)"); return)
    for (i, attempt) in enumerate(attempts)
        flag = attempt.success ? "SUCCESS" : "retry"
        @printf("%3d  %-7s  threshold=%.3e  accepted=%3d  accepted_rank=%2d  selected_rank=%2d  basis_distance=%s\n",
            i,
            flag,
            attempt.threshold,
            attempt.accepted_count,
            attempt.accepted_rank,
            attempt.selected_rank,
            isfinite(attempt.basis_distance) ? @sprintf("%.3e", attempt.basis_distance) : "Inf")
    end
end

function canonical_selection_mode(mode::Symbol)
    if mode in (:simple, :singleton_first)
        return :simple
    elseif mode in (:informed, :stepwise_informed)
        return :informed
    else
        error("Unsupported selection mode: $mode")
    end
end

function selection_mode_view_label(mode::Symbol)
    mode_c = canonical_selection_mode(mode)
    if mode_c == :simple
        return "B. Singleton-first sparse basis"
    elseif mode_c == :informed
        return "C. Stepwise most-informative simple basis"
    end
end

function print_basis_report(example_name::String, result)
    println("\n", repeat("#", 100))
    println("Monomial basis search: ", example_name)
    println(repeat("#", 100))
    if hasproperty(result, :requested_selection_mode)
        println("Requested selection mode: ", result.requested_selection_mode)
    end
    println("Selection mode: ", result.selection_mode)
    println("Basis view: ", selection_mode_view_label(result.selection_mode))
    if hasproperty(result, :fallback_reason) && !isnothing(result.fallback_reason)
        println("Selection note: ", result.fallback_reason)
    end
    if hasproperty(result, :support_retry_note) && !isnothing(result.support_retry_note)
        println("Support note: ", result.support_retry_note)
    end
    println("Pool strategy: ", result.pool_strategy)
    if hasproperty(result, :initial_s_max) && hasproperty(result, :effective_s_max)
        println("Support cap requested/used: ", result.initial_s_max, " -> ", result.effective_s_max)
    end
    println("Dictionary size: ", length(result.dictionary))
    println("Residual cap: ", @sprintf("%.3e", result.residual_cap))
    if result.selection_mode == :simple
        println("Thresholds tried: ", length(result.attempts))
        println("Successful threshold: ", result.successful_threshold === nothing ? "none" : @sprintf("%.3e", result.successful_threshold))
    else
        println("Thresholds tried: n/a (broader residual-cap pool)")
        println("Successful threshold: n/a")
        if hasproperty(result, :σ1_sq)
            println("Relative information normalization σ1^2: ", @sprintf("%.6e", result.σ1_sq))
        end
    end
    println("Effective threshold (max selected residual): ", isfinite(result.effective_threshold) ? @sprintf("%.3e", result.effective_threshold) : "Inf")
    println("Accepted candidates in pool: ", length(result.accepted))
    println("Accepted pool projected rank: ", result.accepted_rank)
    println("Target dimension d: ", result.target_dim)
    println("Selected basis size: ", length(result.selected))
    println("Selected basis rank: ", result.selected_rank)
    println("Selected basis span distance to target: ", @sprintf("%.3e", result.basis_distance))
    println("Basis validated? ", result.basis_ok)

    !isempty(result.attempts) && print_attempt_table(result.attempts)
    if result.selection_mode == :simple
        print_candidate_table("Accepted candidates at successful/final threshold (simplicity order)", result.accepted)
        print_selection_trace(result.trace)
    else
        print_candidate_table("Accepted candidates within residual cap (simplicity order)", result.accepted)
        print_informed_selection_trace(result.trace)
    end
    print_candidate_table("Selected basis", result.selected)
end

# -----------------------------------------------------------------------------
# Simple command-line parsing
# -----------------------------------------------------------------------------

function parse_args(args)
    example = "both"
    selection_mode = :singleton_first
    s_max = 2
    c_max = 1
    residual_cap = 1e-2
    gain_rtol = 1e-8
    basis_tol = 1e-8

    valid_examples = (
        "transport", "repressilator", "both",
        "stat", "stat_limit", "stat_nonlimit",
        "mm", "mm_limit", "mm_nonlimit",
        "all",
    )
    valid_modes = ("simple", "singleton_first", "informed", "stepwise_informed")

    for arg in args
        if arg in valid_examples
            example = arg
        elseif startswith(arg, "--mode=")
            mode_str = split(arg, "=", limit=2)[2]
            mode_str in valid_modes || error("Unknown mode: $mode_str")
            selection_mode = canonical_selection_mode(Symbol(mode_str))
        elseif startswith(arg, "--smax=")
            s_max = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--cmax=")
            c_max = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--residual-cap=")
            residual_cap = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--vec-tol=")
            residual_cap = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--gain-rtol=")
            gain_rtol = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--basis-tol=")
            basis_tol = parse(Float64, split(arg, "=", limit=2)[2])
        else
            error("Unknown argument: $arg")
        end
    end

    return (
        example=example,
        selection_mode=selection_mode,
        s_max=s_max,
        c_max=c_max,
        residual_cap=residual_cap,
        gain_rtol=gain_rtol,
        basis_tol=basis_tol,
    )
end

function simple_search_with_support_retry(U_basis::AbstractMatrix{<:Real}, param_names::Vector{String};
                                          s_max::Int=2,
                                          c_max::Int=1,
                                          residual_cap::Float64=1e-2,
                                          gain_rtol::Float64=1e-8,
                                          basis_tol::Float64=1e-8)
    p = length(param_names)
    initial_s_max = s_max
    effective_s_max = min(p, s_max)
    result = simple_monomial_basis_search(
        U_basis,
        param_names;
        s_max=effective_s_max,
        c_max=c_max,
        residual_cap=residual_cap,
        gain_rtol=gain_rtol,
        basis_tol=basis_tol,
    )

    retry_history = [(s_max=effective_s_max, basis_ok=result.basis_ok, selected_rank=result.selected_rank, target_dim=result.target_dim)]

    if !result.basis_ok && effective_s_max < p
        retry_s_max = min(p, effective_s_max + 1)
        retry_result = simple_monomial_basis_search(
            U_basis,
            param_names;
            s_max=retry_s_max,
            c_max=c_max,
            residual_cap=residual_cap,
            gain_rtol=gain_rtol,
            basis_tol=basis_tol,
        )
        push!(retry_history, (s_max=retry_s_max, basis_ok=retry_result.basis_ok, selected_rank=retry_result.selected_rank, target_dim=retry_result.target_dim))
        result = retry_result
        effective_s_max = retry_s_max
    end

    support_retry_note = if effective_s_max > initial_s_max
        if result.basis_ok
            "Singleton-first search failed at s_max=$(initial_s_max); automatically retried at s_max=$(effective_s_max) and succeeded."
        else
            "Singleton-first search failed at s_max=$(initial_s_max); automatically retried at s_max=$(effective_s_max) but still did not find a basis."
        end
    else
        nothing
    end

    return (
        result...,
        initial_s_max=initial_s_max,
        effective_s_max=effective_s_max,
        support_retry_history=retry_history,
        support_retry_note=support_retry_note,
    )
end

function search_basis_for_space(U_basis::AbstractMatrix{<:Real}, param_names::Vector{String};
                                selection_mode::Symbol,
                                metric=nothing,
                                s_max::Int=2,
                                c_max::Int=1,
                                residual_cap::Float64=1e-2,
                                gain_rtol::Float64=1e-8,
                                basis_tol::Float64=1e-8,
                                metric_rtol::Float64=1e-12)
    requested_selection_mode = canonical_selection_mode(selection_mode)
    actual_selection_mode = requested_selection_mode
    fallback_reason = nothing

    if requested_selection_mode == :informed
        if metric === nothing
            fallback_reason = "No local metric provided for this space; falling back to singleton-first sparse basis."
            actual_selection_mode = :simple
        elseif metric_is_degenerate_on_space(U_basis, metric.M, metric.σ1_sq; metric_rtol=metric_rtol)
            fallback_reason = "Local J'J metric is degenerate on this space; falling back to singleton-first sparse basis."
            actual_selection_mode = :simple
        end
    end

    result = if actual_selection_mode == :simple
        simple_search_with_support_retry(
            U_basis,
            param_names;
            s_max=s_max,
            c_max=c_max,
            residual_cap=residual_cap,
            gain_rtol=gain_rtol,
            basis_tol=basis_tol,
        )
    elseif actual_selection_mode == :informed
        informed_monomial_basis_search(
            U_basis,
            metric.M,
            metric.σ1_sq,
            param_names;
            s_max=s_max,
            c_max=c_max,
            residual_cap=residual_cap,
            gain_rtol=gain_rtol,
            basis_tol=basis_tol,
        )
    else
        error("Unsupported selection mode: $actual_selection_mode")
    end

    return (
        result...,
        requested_selection_mode=requested_selection_mode,
        fallback_reason=fallback_reason,
    )
end

function run_example(example_data; selection_mode::Symbol, s_max::Int, c_max::Int, residual_cap::Float64, gain_rtol::Float64, basis_tol::Float64)
    requested_selection_mode = canonical_selection_mode(selection_mode)
    println("\nRunning basis searches with requested mode=$requested_selection_mode ($(selection_mode_view_label(requested_selection_mode))), s_max=$s_max, c_max=$c_max, residual_cap=$residual_cap, gain_rtol=$gain_rtol, basis_tol=$basis_tol")

    metric = requested_selection_mode == :informed ? compute_local_metric(example_data.ϕ_func, example_data.θ0) : nothing

    identified_label = string(example_data.label, " — identified side N_perp")
    identified_result = search_basis_for_space(
        example_data.N_perp_basis,
        example_data.param_names;
        selection_mode=requested_selection_mode,
        metric=metric,
        s_max=s_max,
        c_max=c_max,
        residual_cap=residual_cap,
        gain_rtol=gain_rtol,
        basis_tol=basis_tol,
    )
    print_basis_report(identified_label, identified_result)

    if size(example_data.N_basis, 2) > 0
        null_label = string(example_data.label, " — invariant null side N")
        null_result = search_basis_for_space(
            example_data.N_basis,
            example_data.param_names;
            selection_mode=requested_selection_mode,
            metric=metric,
            s_max=s_max,
            c_max=c_max,
            residual_cap=residual_cap,
            gain_rtol=gain_rtol,
            basis_tol=basis_tol,
        )
        print_basis_report(null_label, null_result)
    else
        println("\n", repeat("#", 100))
        println("Monomial basis search: ", example_data.label, " — invariant null side N")
        println(repeat("#", 100))
        println("No invariant null space detected; null-side search skipped.")
    end
end

function main(args)
    opts = parse_args(args)

    if opts.example in ("repressilator", "both", "all")
        repressilator_data = load_repressilator_target_subspace()
        run_example(repressilator_data;
            selection_mode=opts.selection_mode,
            s_max=opts.s_max,
            c_max=opts.c_max,
            residual_cap=opts.residual_cap,
            gain_rtol=opts.gain_rtol,
            basis_tol=opts.basis_tol)
    end

    if opts.example in ("transport", "both", "all")
        transport_data = load_transport_target_subspace()
        run_example(transport_data;
            selection_mode=opts.selection_mode,
            s_max=opts.s_max,
            c_max=opts.c_max,
            residual_cap=opts.residual_cap,
            gain_rtol=opts.gain_rtol,
            basis_tol=opts.basis_tol)
    end

    if opts.example in ("stat", "stat_limit", "all")
        stat_limit_data = load_stat_target_subspace(; poisson_limit=true)
        run_example(stat_limit_data;
            selection_mode=opts.selection_mode,
            s_max=opts.s_max,
            c_max=opts.c_max,
            residual_cap=opts.residual_cap,
            gain_rtol=opts.gain_rtol,
            basis_tol=opts.basis_tol)
    end

    if opts.example in ("stat", "stat_nonlimit", "all")
        stat_nonlimit_data = load_stat_target_subspace(; poisson_limit=false)
        run_example(stat_nonlimit_data;
            selection_mode=opts.selection_mode,
            s_max=opts.s_max,
            c_max=opts.c_max,
            residual_cap=opts.residual_cap,
            gain_rtol=opts.gain_rtol,
            basis_tol=opts.basis_tol)
    end

    if opts.example in ("mm", "mm_limit", "all")
        mm_limit_data = load_mm_target_subspace(; limit=true)
        run_example(mm_limit_data;
            selection_mode=opts.selection_mode,
            s_max=opts.s_max,
            c_max=opts.c_max,
            residual_cap=opts.residual_cap,
            gain_rtol=opts.gain_rtol,
            basis_tol=opts.basis_tol)
    end

    if opts.example in ("mm", "mm_nonlimit", "all")
        mm_nonlimit_data = load_mm_target_subspace(; limit=false)
        run_example(mm_nonlimit_data;
            selection_mode=opts.selection_mode,
            s_max=opts.s_max,
            c_max=opts.c_max,
            residual_cap=opts.residual_cap,
            gain_rtol=opts.gain_rtol,
            basis_tol=opts.basis_tol)
    end
end

main(ARGS)
