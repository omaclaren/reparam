# transport_basis_diagnostic.jl
# Read-only diagnostic comparing candidate transport-model bases.

if !@isdefined(ReparamTools)
    include("ReparamTools.jl")
    println("✓ ReparamTools module included")
else
    println("✓ ReparamTools module already included")
end

using .ReparamTools
using Distributions
using LinearAlgebra
using ForwardDiff

# --------------------------------------------------------
# Model definition (copied from examples/transport_model.jl)
# --------------------------------------------------------
function solve_model(θ, x, L)
    y = Vector{eltype(θ)}(undef, length(x))
    mid_index = Int((length(x)-1)/2)

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

create_ϕ_mapping(x, L) = θ -> solve_model(θ, x, L)

# --------------------------------------------------------
# Small helpers
# --------------------------------------------------------
function normalize_columns(A)
    B = Matrix{Float64}(A)
    C = similar(B)
    for j in 1:size(B, 2)
        nj = norm(B[:, j])
        C[:, j] = nj > 0 ? B[:, j] / nj : B[:, j]
    end
    return C
end

function print_basis_summary(name, A; raw_full=nothing, raw_identifiable=nothing, null_basis=nothing)
    A_mat = Matrix{Float64}(A)
    A_norm = normalize_columns(A_mat)

    println("\n", repeat("=", 80))
    println(name)
    println(repeat("=", 80))

    println("Columns (log-space exponent vectors):")
    display(A_mat)

    println("Normalized Gram matrix:")
    display(round.(A_norm' * A_norm, digits=4))

    if null_basis !== nothing
        N_norm = normalize_columns(null_basis)
        println("Dot products with null basis (should be 0 for identifiable columns):")
        display(round.(A_norm' * N_norm, digits=4))
    end

    if raw_identifiable !== nothing && size(A_mat, 2) >= size(raw_identifiable, 2)
        B_id = A_norm[:, 1:size(raw_identifiable, 2)]
        println("Coordinates of candidate identifiable columns in raw identified basis:")
        display(round.(raw_identifiable' * B_id, digits=4))
        proj_resid = norm((I - raw_identifiable * raw_identifiable') * B_id)
        println("Projection residual onto raw identified plane: ", round(proj_resid, digits=8))
    end

    if raw_full !== nothing
        println("Absolute dot products against raw full orthonormal basis:")
        display(round.(abs.(raw_full' * A_norm), digits=4))
    end

    if size(A_mat, 1) == size(A_mat, 2)
        println("det(A) = ", round(det(A_mat), digits=6))
        println("cond(A) = ", round(cond(A_mat), digits=6))
    end
end

function try_integer_rounding(name, A; round_within)
    println("\n", repeat("-", 80))
    println(name, " with round_within = ", round_within)
    println(repeat("-", 80))
    try
        A_round = ReparamTools.scale_and_round(A; round_within=round_within,
            column_scales=ones(Int, size(A, 2)))
        display(A_round)
        return A_round
    catch err
        println("scale_and_round failed: ", sprint(showerror, err))
        return nothing
    end
end

function varimax_objective(L; gamma=1.0)
    n_rows, _ = size(L)
    return sum(sum(L.^4, dims=1) .- gamma .* (sum(L.^2, dims=1).^2) ./ n_rows)
end

function brute_force_2d_varimax(U; n_angles=2001, gamma=1.0)
    best_obj = -Inf
    best_angle = 0.0
    best_basis = copy(U)

    for θ in range(0.0, stop=π, length=n_angles)
        R = [cos(θ) -sin(θ); sin(θ) cos(θ)]
        B = U * R
        obj = varimax_objective(B; gamma=gamma)
        if obj > best_obj
            best_obj = obj
            best_angle = θ
            best_basis = B
        end
    end

    return best_angle, best_basis, best_obj
end

function canonicalize_integer_vector(v::Vector{Int})
    w = copy(v)
    nz = findfirst(!=(0), w)
    if nz !== nothing && w[nz] < 0
        w .*= -1
    end
    return w
end

function primitive_integer_vectors_sum_zero(max_abs::Int)
    seen = Set{NTuple{3, Int}}()
    out = Vector{Vector{Int}}()

    for a in -max_abs:max_abs, b in -max_abs:max_abs, c in -max_abs:max_abs
        v = [a, b, c]
        if all(==(0), v)
            continue
        end
        if sum(v) != 0
            continue
        end
        g = gcd(gcd(abs(a), abs(b)), abs(c))
        if g == 0 || g != 1
            continue
        end
        w = canonicalize_integer_vector(v)
        key = (w[1], w[2], w[3])
        if !(key in seen)
            push!(seen, key)
            push!(out, w)
        end
    end

    return out
end

function basis_alignment_score(U, B)
    U_norm = normalize_columns(U)
    B_norm = normalize_columns(B)
    C = abs.(U_norm' * B_norm)
    score = max(C[1, 1] + C[2, 2], C[1, 2] + C[2, 1]) / 2
    return score, C
end

function monomial_label(v)
    names = ["T1", "T2", "R"]
    num = String[]
    den = String[]
    for (name, exp) in zip(names, v)
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

function vector_complexity(v::Vector{Int})
    return (
        count(x -> x != 0, v),
        sum(abs.(v)),
        maximum(abs.(v)),
    )
end

function direction_alignment(raw_identifiable, v::Vector{Int})
    vn = normalize(Float64.(v))
    overlaps = abs.(raw_identifiable' * vn)
    best_idx = argmax(overlaps)
    return overlaps[best_idx], overlaps, best_idx
end

function transformed_metric(A, M)
    A_mat = Matrix{Float64}(A)
    M_mat = Matrix{Float64}(M)
    return Matrix(Symmetric(A_mat' * M_mat * A_mat))
end

function identifiable_correlation(M; identifiable_cols=1:(size(M, 1)-1))
    M_id = Matrix{Float64}(M[identifiable_cols, identifiable_cols])
    if size(M_id, 1) != 2
        return NaN, M_id
    end
    d1 = M_id[1, 1]
    d2 = M_id[2, 2]
    if d1 <= 0 || d2 <= 0
        return NaN, M_id
    end
    ρ = M_id[1, 2] / sqrt(d1 * d2)
    return clamp(ρ, -1.0, 1.0), M_id
end

function print_metric_separation_summary(name, A, metric, metric_label; null_col=size(A, 2))
    M_trans = transformed_metric(A, metric)
    ρ, M_id = identifiable_correlation(M_trans; identifiable_cols=1:(null_col-1))
    cross_null = M_trans[1:(null_col-1), null_col]
    rel_cross = norm(cross_null) / max(opnorm(M_id), eps())

    println("  ", metric_label, " in candidate coordinates:")
    display(round.(M_trans, digits=4))
    println("  identifiable 2×2 block:")
    display(round.(M_id, digits=4))
    println("  standardized identifiable correlation ρ = ", round(ρ, digits=4))
    println("  separation score 1 - |ρ| = ", round(1 - abs(ρ), digits=4))
    println("  relative coupling to null coordinate = ", round(rel_cross, digits=8))
end

function print_separation_bundle(name, A; sensitivity_metric=nothing, observed_info=nothing)
    println("\nInformation-separation diagnostics for ", name)
    if sensitivity_metric !== nothing
        print_metric_separation_summary(name, A, sensitivity_metric, "Jacobian sensitivity metric J'J")
    end
    if observed_info !== nothing
        print_metric_separation_summary(name, A, observed_info, "Observed information -∇²ℓ")
    end
end

function print_sparse_direction_search(title, vectors, raw_identifiable; topk=10)
    candidates = NamedTuple[]
    for v in vectors
        best_align, overlaps, best_idx = direction_alignment(raw_identifiable, v)
        support, l1, maxabs = vector_complexity(v)
        push!(candidates, (
            v=v,
            label=monomial_label(v),
            support=support,
            l1=l1,
            maxabs=maxabs,
            best_align=best_align,
            overlaps=overlaps,
            best_idx=best_idx,
        ))
    end

    sort!(candidates, by=x -> (x.support, x.l1, x.maxabs, -x.best_align, x.label))

    println("\n", repeat("=", 80))
    println(title)
    println(repeat("=", 80))
    println("Primitive = gcd-1 representative of an integer direction in the identified plane.")
    println("Sparse = few nonzero exponents and small integer magnitudes.")
    println("Showing top ", min(topk, length(candidates)), " primitive identified directions")

    for (k, cand) in enumerate(candidates[1:min(topk, length(candidates))])
        println("\nDirection ", k)
        println("  v = ", cand.v, "   -> ", cand.label)
        println("  support = ", cand.support, ", l1 = ", cand.l1, ", max |exp| = ", cand.maxabs)
        println("  abs overlaps with raw identified basis = ", round.(cand.overlaps, digits=4))
        println("  best alignment = ", round(cand.best_align, digits=4), " (raw identified direction ", cand.best_idx, ")")
    end
end

function print_sparse_oblique_pair_search(title, vectors, raw_identifiable, N_inv;
                                          sensitivity_metric=nothing, observed_info=nothing, topk=8)
    candidates = NamedTuple[]
    null_vec = vec(N_inv[:, 1])

    for i in 1:length(vectors)-1
        for j in i+1:length(vectors)
            v1_int = vectors[i]
            v2_int = vectors[j]
            B = hcat(Float64.(v1_int), Float64.(v2_int))
            if rank(B) < 2
                continue
            end
            A = hcat(B, null_vec)
            score, C = basis_alignment_score(raw_identifiable, B)
            s1, l1_1, m1 = vector_complexity(v1_int)
            s2, l1_2, m2 = vector_complexity(v2_int)

            ρ_sens = sensitivity_metric === nothing ? NaN : identifiable_correlation(transformed_metric(A, sensitivity_metric))[1]
            ρ_info = observed_info === nothing ? NaN : identifiable_correlation(transformed_metric(A, observed_info))[1]

            push!(candidates, (
                v1=v1_int,
                v2=v2_int,
                score=score,
                overlap=C,
                support_total=s1 + s2,
                l1_total=l1_1 + l1_2,
                maxabs=max(m1, m2),
                offdiag=abs(dot(normalize(Float64.(v1_int)), normalize(Float64.(v2_int)))),
                rho_sens=ρ_sens,
                rho_info=ρ_info,
                condA=cond(A),
                detA=det(A),
            ))
        end
    end

    sort!(candidates, by=x -> (x.support_total, x.l1_total, x.maxabs,
        isnan(x.rho_info) ? 1.0 : abs(x.rho_info), -x.score, x.condA))

    println("\n", repeat("=", 80))
    println(title)
    println(repeat("=", 80))
    println("Oblique = no orthogonality constraint within the identified plane; only exact null/complement separation is kept.")
    println("Showing top ", min(topk, length(candidates)), " sparse oblique basis pairs")

    for (k, cand) in enumerate(candidates[1:min(topk, length(candidates))])
        println("\nCandidate ", k)
        println("  v1 = ", cand.v1, "   -> ", monomial_label(cand.v1))
        println("  v2 = ", cand.v2, "   -> ", monomial_label(cand.v2))
        println("  total support = ", cand.support_total, ", total l1 = ", cand.l1_total,
            ", max |exp| = ", cand.maxabs)
        println("  alignment score = ", round(cand.score, digits=4))
        println("  abs overlaps with raw identified basis:")
        display(round.(cand.overlap, digits=4))
        println("  within-plane Euclidean correlation = ", round(cand.offdiag, digits=4))
        if !isnan(cand.rho_sens)
            println("  sensitivity-metric correlation ρ(J'J) = ", round(cand.rho_sens, digits=4),
                "   => separation ", round(1 - abs(cand.rho_sens), digits=4))
        end
        if !isnan(cand.rho_info)
            println("  observed-information correlation ρ(-∇²ℓ) = ", round(cand.rho_info, digits=4),
                "   => separation ", round(1 - abs(cand.rho_info), digits=4))
        end
        println("  det([v1 v2 n]) = ", round(cand.detA, digits=4), ", cond = ", round(cand.condA, digits=4))
    end
end

# --------------------------------------------------------
# Setup matching the manuscript transport example
# --------------------------------------------------------
L = 100
x = LinRange(0, L, 201)
obs_step_size = 10
indices_obs = obs_step_size:obs_step_size:length(x)-obs_step_size
x_obs = x[indices_obs]
σ = 0.2

ϕ_func_xy = create_ϕ_mapping(x, L)
distrib_xy = xy -> MvLogNormal(log.(abs.(ϕ_func_xy(xy)[indices_obs])), σ^2 * I(length(x_obs)))

xy_lower_bounds = [0.1, 0.1, 0.1]
xy_upper_bounds = [5.0, 5.0, 5.0]
xy_initial = 0.5 * (xy_lower_bounds + xy_upper_bounds)
xy_true = [3.0, 1.0, 1.0]

# Fixed full-precision realization matching the manuscript's rounded data vector.
data = [
    186.41985649628793,
    402.5960016186346,
    505.1903104103731,
    756.1484631238714,
    1144.1217734222512,
    790.9347742347392,
    1283.5035454196181,
    1647.646222948307,
    872.2879198900218,
    1144.358910322539,
    1691.1523621174656,
    1352.7415676126952,
    1519.8895992925186,
    1315.9842212425283,
    1437.075610344087,
    726.694229390357,
    952.2875562522823,
    759.3936211738769,
    271.9972290000625,
]

lnlike_xy = construct_lnlike_xy(distrib_xy, data; dist_type=:multi)
xytoXY_log(xy) = log.(xy)
XYtoxy_log(XY) = exp.(XY)

XY_log_lower_bounds = log.(xy_lower_bounds)
XY_log_upper_bounds = log.(xy_upper_bounds)
XY_log_initial = xytoXY_log(xy_initial)
lnlike_XY_log = construct_lnlike_XY(lnlike_xy, XYtoxy_log)
ϕ_func_XY_log = construct_ϕ_XY(ϕ_func_xy, XYtoxy_log)

println("Finding log-space MLE...")
XY_log_MLE, lnlike_XY_log_MLE = profile_target(
    lnlike_XY_log, [],
    XY_log_lower_bounds, XY_log_upper_bounds,
    XY_log_initial;
    grid_steps=[500],
    ω_initial_extras=generate_initial_guesses(XY_log_lower_bounds, XY_log_upper_bounds, 3),
    method=:LN_BOBYQA,
)
println("XY_log_MLE = ", round.(XY_log_MLE, digits=6))
println("xy_MLE = ", round.(exp.(XY_log_MLE), digits=6))
println("lnlike_MLE = ", round(lnlike_XY_log_MLE, digits=6))

J_ϕ_XY_log, U_XY_log, S_XY_log, Vt_XY_log = compute_ϕ_Jacobian(
    ϕ_func_XY_log, XY_log_MLE; method_type=:auto, compute_svd=true)
println("\nRaw log-space singular values:")
println(S_XY_log)

S_inv, N_inv, N_perp_inv, rank_inv = find_invariant_subspace(
    ϕ_func_XY_log, XY_log_MLE; verbose=true)
println("\nrank_inv = ", rank_inv)
println("dim(N_inv) = ", size(N_inv, 2))
println("dim(N_perp_inv) = ", size(N_perp_inv, 2))

A_raw = hcat(N_perp_inv, N_inv)
G_sens = Matrix(Symmetric(J_ϕ_XY_log' * J_ϕ_XY_log))
H_obs = Matrix(Symmetric(-ForwardDiff.hessian(lnlike_XY_log, XY_log_MLE)))

println("\nTwo comparison objectives:")
println("  1. Orthogonal / SVD-derived basis: use raw SVD directions and round them directly.")
println("  2. Interpretable / sparse basis: search for simple integer monomials in the exact identified plane.")

print_basis_summary(
    "Raw orthonormal basis from find_invariant_subspace",
    A_raw;
    raw_full=A_raw,
    raw_identifiable=N_perp_inv,
    null_basis=N_inv,
)
print_separation_bundle(
    "raw orthonormal basis",
    A_raw;
    sensitivity_metric=G_sens,
    observed_info=H_obs,
)

A_ratio = hcat(
    [1.0, 0.0, -1.0],
    [0.0, 1.0, -1.0],
    [1.0, 1.0, 1.0],
)
print_basis_summary(
    "Sparsest exact oblique ratio basis + null direction",
    A_ratio;
    raw_full=A_raw,
    raw_identifiable=N_perp_inv,
    null_basis=N_inv,
)
print_separation_bundle(
    "sparsest exact oblique ratio basis",
    A_ratio;
    sensitivity_metric=G_sens,
    observed_info=H_obs,
)

println("\nCurrent transport_model.jl path uses integer rounding only:")
println("  rounded_columns = ReparamTools.scale_and_round(Vt_XY_log; round_within=0.5, column_scales=[1,1,1])")
println("The helper rounds to nearest integers, not half-integers.")

A_current_050 = try_integer_rounding(
    "Current script rounding applied to Vt_XY_log (legacy path)",
    Vt_XY_log;
    round_within=0.5,
)
if A_current_050 !== nothing
    print_basis_summary(
        "Current script rounded basis (legacy path)",
        A_current_050;
        raw_full=A_raw,
        raw_identifiable=N_perp_inv,
        null_basis=N_inv,
    )
    print_separation_bundle(
        "current script rounded basis (legacy path)",
        A_current_050;
        sensitivity_metric=G_sens,
        observed_info=H_obs,
    )
end

for rw in (0.45, 0.4, 0.35, 0.3, 0.25)
    A_try = try_integer_rounding(
        "Integer rounding applied to raw orthonormal basis hcat(N_perp_inv, N_inv)",
        A_raw;
        round_within=rw,
    )
    if A_try !== nothing
        print_basis_summary(
            "Rounded raw basis with round_within=$(rw)",
            A_try;
            raw_full=A_raw,
            raw_identifiable=N_perp_inv,
            null_basis=N_inv,
        )
        print_separation_bundle(
            "rounded raw basis with round_within=$(rw)",
            A_try;
            sensitivity_metric=G_sens,
            observed_info=H_obs,
        )
    end
end

println("\nAttempting package Varimax on the identified subspace (inspection only)...")
try
    N_perp_varimax = ReparamTools.varimax_rotation(N_perp_inv; n_restarts=200, threshold=1e-2)
    A_varimax = hcat(N_perp_varimax, N_inv)
    print_basis_summary(
        "Package Varimax-rotated identified basis + null",
        A_varimax;
        raw_full=A_raw,
        raw_identifiable=N_perp_inv,
        null_basis=N_inv,
    )
catch err
    println("Package Varimax failed: ", sprint(showerror, err))
end

println("\nBrute-force 2D orthogonal rotation search with Varimax objective...")
θ_varimax, N_perp_varimax_bruteforce, obj_varimax = brute_force_2d_varimax(N_perp_inv)
println("Best angle θ = ", round(θ_varimax, digits=6), " radians")
println("Best Varimax objective = ", round(obj_varimax, digits=8))
A_varimax_bruteforce = hcat(N_perp_varimax_bruteforce, N_inv)
print_basis_summary(
    "Brute-force orthogonal rotation (Varimax objective) + null",
    A_varimax_bruteforce;
    raw_full=A_raw,
    raw_identifiable=N_perp_inv,
    null_basis=N_inv,
)

for rw in (0.5, 0.45, 0.4, 0.35)
    A_try = try_integer_rounding(
        "Integer rounding applied to brute-force Varimax basis",
        A_varimax_bruteforce;
        round_within=rw,
    )
    if A_try !== nothing
        print_basis_summary(
            "Rounded brute-force Varimax basis with round_within=$(rw)",
            A_try;
            raw_full=A_raw,
            raw_identifiable=N_perp_inv,
            null_basis=N_inv,
        )
    end
end

small_vectors = primitive_integer_vectors_sum_zero(3)
print_sparse_direction_search(
    "Sparse primitive directions in the identified plane (simplicity first)",
    small_vectors,
    N_perp_inv;
    topk=10,
)

print_sparse_oblique_pair_search(
    "Sparse oblique basis pairs in the identified plane (simplicity first)",
    small_vectors,
    N_perp_inv,
    N_inv;
    sensitivity_metric=G_sens,
    observed_info=H_obs,
    topk=8,
)
