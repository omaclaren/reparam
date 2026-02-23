# Recover IIR diagnostics from saved repressilator profile results (.jls)
#
# Purpose:
# - Recompute Jacobian singular values and IIR rank diagnostics at saved θ_MLE
# - Avoid expensive profile-likelihood reruns on NeSI
#
# Usage:
#   julia --project=. extract_iir_diagnostics_from_results.jl
#   julia --project=. extract_iir_diagnostics_from_results.jl nesi/repressilator_16nuisance_50x50_results.jls
#   julia --project=. extract_iir_diagnostics_from_results.jl <results1.jls> <results2.jls> ...
#
# Default behavior (no args):
#   auto-discover canonical files:
#   nesi/repressilator_16nuisance_<NxN>_results.jls
#
# Outputs:
#   nesi/repressilator_iir_diagnostics_from_saved_results.csv
#   nesi/repressilator_iir_diagnostics_from_saved_results.md

using Pkg
Pkg.activate(@__DIR__)

using Serialization
using LinearAlgebra
using Printf
using Dates

include(joinpath(@__DIR__, "ReparamTools.jl"))
include(joinpath(@__DIR__, "examples", "RepressilatorModel.jl"))
using .ReparamTools
using .RepressilatorModel

const X0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
const T_END = 10000.0
const T_IIR = LinRange(0, T_END, 501)

const PARAM_NAMES = [
    "α₀₁", "α₀₂", "α₀₃", "α₁", "α₂", "α₃",
    "β₁", "β₂", "β₃", "K₁", "K₂", "K₃",
    "k_degm₁", "k_degm₂", "k_degm₃", "k_degp₁", "k_degp₂", "k_degp₃"
]

function discover_default_files()
    nesi_dir = joinpath(@__DIR__, "nesi")
    if !isdir(nesi_dir)
        error("Could not find nesi/ directory at $nesi_dir")
    end

    rx = r"^repressilator_16nuisance_[0-9]+x[0-9]+_results\.jls$"
    names = filter(name -> occursin(rx, name), readdir(nesi_dir))

    if isempty(names)
        error("No canonical repressilator_16nuisance_*_results.jls files found in nesi/")
    end

    # sort by grid size (e.g., 20x20, 50x50, 100x100)
    function grid_n(name)
        m = match(r"_([0-9]+)x([0-9]+)_results\.jls$", name)
        return m === nothing ? typemax(Int) : parse(Int, m.captures[1])
    end

    sort!(names, by=grid_n)
    return [joinpath("nesi", n) for n in names]
end

function monomial_label(v::AbstractVector{<:Real}; tol::Real=0.25)
    coeffs = round.(Int, v)
    if any(abs.(v .- coeffs) .> tol)
        # fallback for non-nearly-integer vectors
        nz = findall(i -> abs(v[i]) > 1e-8, eachindex(v))
        if isempty(nz)
            return "1"
        end
        parts = String[]
        for i in nz
            c = round(v[i], digits=3)
            push!(parts, "$(PARAM_NAMES[i])^$(c)")
        end
        return join(parts, "·")
    end

    nz = findall(!=(0), coeffs)
    if isempty(nz)
        return "1"
    end

    parts = String[]
    for i in nz
        c = coeffs[i]
        p = PARAM_NAMES[i]
        if c == 1
            push!(parts, p)
        elseif c == -1
            push!(parts, "$(p)⁻¹")
        else
            push!(parts, "$(p)^$(c)")
        end
    end
    return join(parts, "·")
end

function classify_gene_betaK(v::AbstractVector{<:Real}; tol::Real=0.25)
    coeffs = round.(Int, v)
    if any(abs.(v .- coeffs) .> tol)
        return "mixed"
    end

    nz = findall(!=(0), coeffs)
    if length(nz) != 2
        return "mixed"
    end

    β_idx = nz[1] in 7:9 ? nz[1] : (nz[2] in 7:9 ? nz[2] : 0)
    K_idx = nz[1] in 10:12 ? nz[1] : (nz[2] in 10:12 ? nz[2] : 0)
    if β_idx == 0 || K_idx == 0
        return "mixed"
    end

    geneβ = β_idx - 6
    geneK = K_idx - 9
    if geneβ != geneK
        return "mixed"
    end

    b = coeffs[β_idx]
    k = coeffs[K_idx]

    if abs(b) == 1 && abs(k) == 1 && b == -k
        return b == -1 ? "K$(geneβ)/β$(geneβ)" : "β$(geneβ)/K$(geneβ)"
    elseif abs(b) == abs(k) && abs(b) >= 1 && sign(b) == sign(k)
        return b > 0 ? "β$(geneβ)·K$(geneβ)" : "(β$(geneβ)·K$(geneβ))⁻¹"
    else
        return "mixed"
    end
end

function compute_iir_diagnostics_at_mle(θ_MLE::Vector{Float64};
                                        rtol_rank=1e-7,
                                        rtol_invariance=1e-6)
    ϕ_iir_highprec(θ) = begin
        sol_matrix = RepressilatorModel.solve_repressilator(T_IIR, θ, X0; abstol=1e-10, reltol=1e-8)
        mRNA = sol_matrix[1:3, :]
        vec(mRNA)
    end
    ϕ_iir_log(θ_log) = ϕ_iir_highprec(exp.(θ_log))

    θ_log_MLE = log.(θ_MLE)
    S, N, N_perp, rank_J = ReparamTools.find_invariant_subspace(
        ϕ_iir_log, θ_log_MLE;
        rtol_rank=rtol_rank,
        rtol_invariance=rtol_invariance,
        verbose=false,
    )

    n_ident_total = size(N_perp, 2)
    n_nonident = size(N, 2)
    n_noninvariant_null = n_ident_total - rank_J

    σ1 = S[1]
    σr = S[rank_J]
    σnext = rank_J < length(S) ? S[rank_J + 1] : 0.0

    cond_ident = σ1 / max(σr, eps())
    gap_rank = σr / max(σnext, eps())

    return (
        S=S,
        rank_J=rank_J,
        n_ident_total=n_ident_total,
        n_nonident=n_nonident,
        n_noninvariant_null=n_noninvariant_null,
        σ1=σ1,
        σr=σr,
        σnext=σnext,
        cond_ident=cond_ident,
        gap_rank=gap_rank,
    )
end

function csv_escape(s)
    str = string(s)
    if occursin(',', str) || occursin('"', str) || occursin('\n', str)
        return "\"" * replace(str, '"' => "\"\"") * "\""
    end
    return str
end

function main(files::Vector{String})
    println("Recovering IIR diagnostics from saved results...")
    println("Files to process: $(length(files))")
    for f in files
        println("  - $f")
    end

    rows = NamedTuple[]

    for file in files
        println("\n---")
        println("Processing: $file")

        if !isfile(file)
            println("  ERROR: file not found, skipping")
            continue
        end

        d = deserialize(file)

        required_keys = ["θ_MLE", "A_T_final", "target_2d", "GRID", "rank_J", "n_ident", "n_nonident"]
        missing = filter(k -> !haskey(d, k), required_keys)
        if !isempty(missing)
            println("  ERROR: missing keys $(missing), skipping")
            continue
        end

        θ_MLE = Float64.(d["θ_MLE"])
        A_T_final = Matrix{Float64}(d["A_T_final"])
        target_2d = Int.(d["target_2d"])

        rank_saved = Int(d["rank_J"])
        n_ident_saved = Int(d["n_ident"])
        n_nonident_saved = Int(d["n_nonident"])
        grid = Int(d["GRID"])
        mode = get(d, "mode", "unknown")

        t_start = time()
        diag = compute_iir_diagnostics_at_mle(θ_MLE)
        elapsed = time() - t_start

        target1_label = monomial_label(A_T_final[:, target_2d[1]])
        target2_label = monomial_label(A_T_final[:, target_2d[2]])
        target1_class = classify_gene_betaK(A_T_final[:, target_2d[1]])
        target2_class = classify_gene_betaK(A_T_final[:, target_2d[2]])

        singular_values_str = join([@sprintf("%.6g", s) for s in diag.S], ";")

        println(@sprintf("  IIR recompute: rank %d, σ1=%.4g, σr=%.4g, gap=%.3g× (%.1fs)",
            diag.rank_J, diag.σ1, diag.σr, diag.gap_rank, elapsed))
        println("  Saved rank/ident/non-ident: $rank_saved / $n_ident_saved / $n_nonident_saved")
        println("  Recomputed ident/non-ident/non-invariant-null: $(diag.n_ident_total) / $(diag.n_nonident) / $(diag.n_noninvariant_null)")
        println("  target_2d=$(target_2d): [$(target1_class), $(target2_class)]")

        push!(rows, (
            file=file,
            grid=grid,
            mode=mode,
            rank_saved=rank_saved,
            rank_recomputed=diag.rank_J,
            n_ident_saved=n_ident_saved,
            n_nonident_saved=n_nonident_saved,
            n_ident_recomputed=diag.n_ident_total,
            n_nonident_recomputed=diag.n_nonident,
            n_noninvariant_null_recomputed=diag.n_noninvariant_null,
            sigma1=diag.σ1,
            sigma_rank=diag.σr,
            sigma_next=diag.σnext,
            cond_ident=diag.cond_ident,
            gap_rank=diag.gap_rank,
            target_idx_1=target_2d[1],
            target_idx_2=target_2d[2],
            target_1_class=target1_class,
            target_2_class=target2_class,
            target_1_label=target1_label,
            target_2_label=target2_label,
            singular_values=singular_values_str,
        ))
    end

    if isempty(rows)
        error("No diagnostics produced (all files failed?)")
    end

    out_csv = joinpath(@__DIR__, "nesi", "repressilator_iir_diagnostics_from_saved_results.csv")
    out_md = joinpath(@__DIR__, "nesi", "repressilator_iir_diagnostics_from_saved_results.md")

    # CSV output
    headers = [
        :file, :grid, :mode,
        :rank_saved, :rank_recomputed,
        :n_ident_saved, :n_nonident_saved,
        :n_ident_recomputed, :n_nonident_recomputed, :n_noninvariant_null_recomputed,
        :sigma1, :sigma_rank, :sigma_next, :cond_ident, :gap_rank,
        :target_idx_1, :target_idx_2, :target_1_class, :target_2_class,
        :target_1_label, :target_2_label,
        :singular_values,
    ]

    open(out_csv, "w") do io
        println(io, join(string.(headers), ","))
        for r in rows
            vals = [getfield(r, h) for h in headers]
            println(io, join(csv_escape.(vals), ","))
        end
    end

    # Markdown summary output
    open(out_md, "w") do io
        println(io, "# IIR Diagnostics Recovered from Saved Repressilator Results")
        println(io)
        println(io, "Generated: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
        println(io)
        println(io, "No profile reruns were performed. Diagnostics were recomputed at saved θ_MLE.")
        println(io)

        println(io, "| file | rank(saved/recomputed) | n_ident(saved/recomp) | n_nonident(saved/recomp) | σ₁ | σ_r | gap σ_r/σ_{r+1} | target classes |")
        println(io, "|---|---:|---:|---:|---:|---:|---:|---|")
        for r in rows
            println(io,
                "| $(basename(r.file)) | $(r.rank_saved)/$(r.rank_recomputed) | $(r.n_ident_saved)/$(r.n_ident_recomputed) | $(r.n_nonident_saved)/$(r.n_nonident_recomputed) | ",
                "$(round(r.sigma1, sigdigits=4)) | $(round(r.sigma_rank, sigdigits=4)) | $(round(r.gap_rank, sigdigits=4)) | ",
                "$(r.target_1_class), $(r.target_2_class) |"
            )
        end

        println(io)
        println(io, "## Notes")
        println(io, "- `gap σ_r/σ_{r+1}` is the rank-separation diagnostic at the recomputed point.")
        println(io, "- `n_noninvariant_null_recomputed = n_ident_recomputed - rank_recomputed`.")
        println(io, "- Target classes are inferred from saved `A_T_final` columns (basis can be permuted/sign-flipped across runs).")
    end

    println("\nSaved:")
    println("  $out_csv")
    println("  $out_md")
end

files = if isempty(ARGS)
    discover_default_files()
else
    ARGS
end

main(files)
