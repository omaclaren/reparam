# Standalone diagnostic: fixed bounded monomial dictionary scan in an identified subspace.
#
# Purpose
# -------
# Given a numerically discovered identified subspace U, test whether U contains
# simple monomial directions from a fixed bounded dictionary of primitive integer
# exponent vectors. This is an interpretability probe, not a general sparse-basis
# solver.
#
# Usage
# -----
#   julia --project=. sparse_monomial_scan_diagnostic.jl repressilator
#   julia --project=. sparse_monomial_scan_diagnostic.jl transport
#   julia --project=. sparse_monomial_scan_diagnostic.jl both --smax=2 --cmax=1 --tol=1e-6

if !@isdefined(ReparamTools)
    include("ReparamTools.jl")
    println("✓ ReparamTools module included")
else
    println("✓ ReparamTools module already included")
end

include(joinpath(@__DIR__, "examples", "RepressilatorModel.jl"))

using .ReparamTools
using .RepressilatorModel
using LinearAlgebra
using Printf
using Serialization

# -----------------------------------------------------------------------------
# Small generic helpers
# -----------------------------------------------------------------------------

function orthonormalize_columns(A::AbstractMatrix{<:Real})
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

Base.@kwdef struct ScanCandidate
    v::Vector{Int}
    label::String
    support::Int
    l1::Int
    residual::Float64
end

function candidate_lt(a::ScanCandidate, b::ScanCandidate)
    ka = (a.support, a.l1, a.residual)
    kb = (b.support, b.l1, b.residual)
    if ka != kb
        return ka < kb
    end
    return lexless(a.v, b.v)
end

function projected_coordinate_matrix(Q::AbstractMatrix{<:Real}, candidates::Vector{ScanCandidate})
    if isempty(candidates)
        return zeros(Float64, size(Q, 2), 0)
    end
    return hcat([Q' * Float64.(cand.v) for cand in candidates]...)
end

function greedy_spanning_subset(Q::AbstractMatrix{<:Real}, accepted::Vector{ScanCandidate}; rank_rtol=1e-8)
    target_dim = size(Q, 2)
    selected = ScanCandidate[]
    coords = zeros(Float64, target_dim, 0)
    current_rank = 0

    for cand in accepted
        col = reshape(Q' * Float64.(cand.v), :, 1)
        new_rank = numerical_rank(hcat(coords, col); rtol=rank_rtol)
        if new_rank > current_rank
            push!(selected, cand)
            coords = hcat(coords, col)
            current_rank = new_rank
            current_rank == target_dim && break
        end
    end

    return selected, current_rank
end

function scan_identified_subspace(U_basis::AbstractMatrix{<:Real}, param_names::Vector{String};
                                  s_max::Int=2, c_max::Int=1, tol::Float64=1e-6,
                                  rank_rtol::Float64=1e-8)
    Q = orthonormalize_columns(U_basis)
    dict = generate_candidate_dictionary(length(param_names); s_max=s_max, c_max=c_max)

    accepted = ScanCandidate[]
    for v in dict
        resid = subspace_residual(Q, v)
        if resid <= tol
            push!(accepted, ScanCandidate(
                v=v,
                label=monomial_label(v, param_names),
                support=count(!=(0), v),
                l1=sum(abs.(v)),
                residual=resid,
            ))
        end
    end

    sort!(accepted, lt=candidate_lt)

    accepted_coords = projected_coordinate_matrix(Q, accepted)
    accepted_rank = numerical_rank(accepted_coords; rtol=rank_rtol)

    selected, selected_rank = greedy_spanning_subset(Q, accepted; rank_rtol=rank_rtol)
    selected_span_distance = isempty(selected) ? Inf : subspace_distance(U_basis, hcat([Float64.(cand.v) for cand in selected]...))

    return (
        Q=Q,
        dictionary=dict,
        accepted=accepted,
        accepted_rank=accepted_rank,
        selected=selected,
        selected_rank=selected_rank,
        target_dim=size(Q, 2),
        selected_span_distance=selected_span_distance,
    )
end

# -----------------------------------------------------------------------------
# Example-specific identified subspace loaders
# -----------------------------------------------------------------------------

function load_repressilator_identified_subspace(; input_file="nesi/repressilator_16nuisance_50x50_results.jls")
    isfile(input_file) || error("Repressilator results file not found: $input_file")
    results = deserialize(input_file)

    required = ["A_T_final", "n_ident", "param_names"]
    missing = filter(k -> !haskey(results, k), required)
    isempty(missing) || error("Missing keys in repressilator results: $(missing)")

    A_T_final = Matrix{Float64}(results["A_T_final"])
    n_ident = Int(results["n_ident"])
    param_names = Vector{String}(results["param_names"])

    return (
        label="repressilator identified subspace from saved profile basis",
        U_basis=A_T_final[:, 1:n_ident],
        param_names=param_names,
        reference_basis=A_T_final[:, 1:n_ident],
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

function load_transport_identified_subspace()
    # Fixed MLE from the previously verified fixed-data transport diagnostic.
    XY_log_MLE = [-1.620844, -2.274112, -2.276164]

    L = 100
    x = LinRange(0, L, 201)
    ϕ_func_xy = θ -> solve_transport_model(θ, x, L)
    XYtoxy_log(XY) = exp.(XY)
    ϕ_func_XY_log = construct_ϕ_XY(ϕ_func_xy, XYtoxy_log)

    _, _, N_perp_inv, _ = find_invariant_subspace(
        ϕ_func_XY_log, XY_log_MLE; verbose=false)

    return (
        label="transport identified subspace",
        U_basis=N_perp_inv,
        param_names=["T1", "T2", "R"],
        reference_basis=N_perp_inv,
    )
end

# -----------------------------------------------------------------------------
# Reporting helpers
# -----------------------------------------------------------------------------

function print_candidate_table(title::String, candidates::Vector{ScanCandidate})
    println("\n", repeat("=", 90))
    println(title)
    println(repeat("=", 90))
    if isempty(candidates)
        println("(none)")
        return
    end
    for (i, cand) in enumerate(candidates)
        @printf("%3d  %-20s  support=%d  l1=%d  residual=%.3e  v=%s\n",
            i, cand.label, cand.support, cand.l1, cand.residual, string(cand.v))
    end
end

function print_scan_report(example_name::String, result; reference_basis=nothing)
    println("\n", repeat("#", 90))
    println("Sparse monomial scan: ", example_name)
    println(repeat("#", 90))
    println("Dictionary size: ", length(result.dictionary))
    println("Accepted candidates: ", length(result.accepted))
    println("Target dimension: ", result.target_dim)
    println("Accepted candidate span rank: ", result.accepted_rank)
    println("Selected basis size: ", length(result.selected))
    println("Selected basis rank: ", result.selected_rank)
    println("Selected span distance to target subspace: ", @sprintf("%.3e", result.selected_span_distance))

    if reference_basis !== nothing && !isempty(result.selected)
        selected_basis = hcat([Float64.(cand.v) for cand in result.selected]...)
        ref_dist = subspace_distance(reference_basis, selected_basis)
        println("Selected span distance to reference basis: ", @sprintf("%.3e", ref_dist))
    end

    println("Full span recovered from accepted candidates? ", result.accepted_rank == result.target_dim)
    println("Full span recovered from selected basis? ", result.selected_rank == result.target_dim)

    print_candidate_table("Accepted candidates", result.accepted)
    print_candidate_table("Greedy selected basis", result.selected)
end

# -----------------------------------------------------------------------------
# Simple command-line parsing
# -----------------------------------------------------------------------------

function parse_args(args)
    example = "both"
    s_max = 2
    c_max = 1
    tol = 1e-6
    rank_rtol = 1e-8

    for arg in args
        if arg in ("transport", "repressilator", "both")
            example = arg
        elseif startswith(arg, "--smax=")
            s_max = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--cmax=")
            c_max = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--tol=")
            tol = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--rank-rtol=")
            rank_rtol = parse(Float64, split(arg, "=", limit=2)[2])
        else
            error("Unknown argument: $arg")
        end
    end

    return (example=example, s_max=s_max, c_max=c_max, tol=tol, rank_rtol=rank_rtol)
end

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

function run_example(example_data; s_max::Int, c_max::Int, tol::Float64, rank_rtol::Float64)
    println("\nRunning scan with s_max=$s_max, c_max=$c_max, tol=$tol, rank_rtol=$rank_rtol")
    result = scan_identified_subspace(
        example_data.U_basis,
        example_data.param_names;
        s_max=s_max,
        c_max=c_max,
        tol=tol,
        rank_rtol=rank_rtol,
    )
    print_scan_report(example_data.label, result; reference_basis=example_data.reference_basis)
end

function main(args)
    opts = parse_args(args)

    if opts.example in ("repressilator", "both")
        repressilator_data = load_repressilator_identified_subspace()
        run_example(repressilator_data;
            s_max=opts.s_max, c_max=opts.c_max, tol=opts.tol, rank_rtol=opts.rank_rtol)
    end

    if opts.example in ("transport", "both")
        transport_data = load_transport_identified_subspace()
        run_example(transport_data;
            s_max=opts.s_max, c_max=opts.c_max, tol=opts.tol, rank_rtol=opts.rank_rtol)
    end
end

main(ARGS)
