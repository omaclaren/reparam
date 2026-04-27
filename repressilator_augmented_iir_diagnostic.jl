# Compare repressilator invariant-image structure under augmented auxiliary maps.
#
# Usage:
#   julia --project=. repressilator_augmented_iir_diagnostic.jl <results.jls>
#
# This is a cheap local diagnostic built from a saved repressilator profiling
# result. It does not rerun any profile likelihood. Instead it recomputes the
# local invariant-image split at the saved θ_MLE for three auxiliary maps:
#   1) mRNA only:        (m₁, m₂, m₃)
#   2) mRNA + protein 1: (m₁, m₂, m₃, p₁)
#   3) mRNA + all proteins: (m₁, m₂, m₃, p₁, p₂, p₃)
#
# For manuscript support this is useful for testing whether direct protein
# observations remove particular invariant-null combinations that are invisible
# to mRNA-only data.

using Pkg
Pkg.activate(@__DIR__)

using Serialization
using LinearAlgebra
using Dates

include(joinpath(@__DIR__, "ReparamTools.jl"))
include(joinpath(@__DIR__, "examples", "RepressilatorModel.jl"))
using .ReparamTools
using .RepressilatorModel

isempty(ARGS) && error("Usage: julia --project=. repressilator_augmented_iir_diagnostic.jl <results.jls>")
input_file = ARGS[1]
isfile(input_file) || error("Input file not found: $input_file")

results = deserialize(input_file)
required = ["θ_MLE", "X0", "T_end", "param_names"]
missing_keys = filter(k -> !haskey(results, k), required)
isempty(missing_keys) || error("Missing keys in results file: $(missing_keys)")

θ_MLE = Float64.(results["θ_MLE"])
X0 = Float64.(results["X0"])
T_end = Float64(results["T_end"])
param_names = String.(results["param_names"])

t_iir = collect(LinRange(0, T_end, 501))

println("Loading saved repressilator result: $input_file")
println("θ_MLE length: $(length(θ_MLE))")
println("T_end: $T_end")
println("Fine diagnostic grid length: $(length(t_iir))")

function solve_matrix(θ)
    RepressilatorModel.solve_repressilator(t_iir, θ, X0; abstol=1e-10, reltol=1e-8)
end

function make_log_aux_map(row_selector)
    function ϕ_log(θ_log)
        θ = exp.(θ_log)
        sol = solve_matrix(θ)
        return row_selector(sol)
    end
    return ϕ_log
end

function null_basis_labels_from_N(N, param_names)
    if size(N, 2) == 0
        return String[], nothing
    end
    result = ReparamTools.simple_monomial_basis_search(
        N, param_names;
        s_max=2,
        c_max=1,
        residual_cap=1e-2,
        retry_support=true,
    )
    labels = result.basis_ok ? ReparamTools.basis_labels(result.selected) : String[]
    return labels, result
end

function report_case(label, row_selector)
    println("\n", "="^80)
    println(label)
    println("="^80)

    ϕ_log = make_log_aux_map(row_selector)
    S, N, N_perp, rank_J = ReparamTools.find_invariant_subspace(
        ϕ_log, log.(θ_MLE); rtol_rank=1e-7, rtol_invariance=1e-6, verbose=false)

    null_dim = size(N, 2)
    labels, null_basis_result = null_basis_labels_from_N(N, param_names)

    println("  rank_J = ", rank_J)
    println("  invariant-null dimension = ", null_dim)
    println("  singular values (first few) = ", round.(S[1:min(end, 6)], sigdigits=5))
    println("  null basis search success = ", isnothing(null_basis_result) ? "not needed (null dim = 0)" : null_basis_result.basis_ok)
    println("  null basis labels = ", labels)

    label_set = Set(labels)
    println("  contains β₁*K₁? ", "β₁*K₁" in label_set)
    println("  contains β₂*K₂? ", "β₂*K₂" in label_set)
    println("  contains β₃*K₃? ", "β₃*K₃" in label_set)

    return (label=label, S=S, N=N, N_perp=N_perp, rank_J=rank_J,
            null_dim=null_dim, null_labels=labels, null_basis_result=null_basis_result)
end

mrna_only = report_case(
    "mRNA only auxiliary map (m₁,m₂,m₃)",
    sol -> vec(sol[1:3, :]),
)

mrna_plus_p1 = report_case(
    "mRNA + protein 1 auxiliary map (m₁,m₂,m₃,p₁)",
    sol -> vcat(vec(sol[1:3, :]), vec(sol[4:4, :])),
)

mrna_plus_all_proteins = report_case(
    "mRNA + all proteins auxiliary map (m₁,m₂,m₃,p₁,p₂,p₃)",
    sol -> vec(sol[1:6, :]),
)

println("\n", "#"^80)
println("Augmented-IIR comparison summary")
println("#"^80)
println("mRNA only: rank=$(mrna_only.rank_J), null_dim=$(mrna_only.null_dim), labels=$(mrna_only.null_labels)")
println("mRNA + p₁: rank=$(mrna_plus_p1.rank_J), null_dim=$(mrna_plus_p1.null_dim), labels=$(mrna_plus_p1.null_labels)")
println("mRNA + all proteins: rank=$(mrna_plus_all_proteins.rank_J), null_dim=$(mrna_plus_all_proteins.null_dim), labels=$(mrna_plus_all_proteins.null_labels)")
println("\nKey manuscript-facing question:")
println("  β₁*K₁ removed by adding p₁? ", !("β₁*K₁" in Set(mrna_plus_p1.null_labels)))
println("  β₂*K₂ retained with p₁? ", "β₂*K₂" in Set(mrna_plus_p1.null_labels))
println("  β₃*K₃ retained with p₁? ", "β₃*K₃" in Set(mrna_plus_p1.null_labels))

beta1_removed = !in("β₁*K₁", Set(mrna_plus_p1.null_labels))
beta2_retained = in("β₂*K₂", Set(mrna_plus_p1.null_labels))
beta3_retained = in("β₃*K₃", Set(mrna_plus_p1.null_labels))

out_md = joinpath(dirname(input_file), "repressilator_augmented_iir_diagnostic.md")
open(out_md, "w") do io
    println(io, "# Repressilator augmented-IIR diagnostic")
    println(io)
    println(io, "Generated: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    println(io)
    println(io, "Input file: `$(input_file)`")
    println(io)
    println(io, "| auxiliary map | rank | invariant-null dim | null labels |")
    println(io, "|---|---:|---:|---|")
    for case in (mrna_only, mrna_plus_p1, mrna_plus_all_proteins)
        labels = isempty(case.null_labels) ? "(none)" : join(case.null_labels, ", ")
        println(io, "| $(case.label) | $(case.rank_J) | $(case.null_dim) | $(labels) |")
    end
    println(io)
    println(io, "## Key question")
    println(io)
    println(io, "- β₁*K₁ removed by adding p₁? **$(beta1_removed)**")
    println(io, "- β₂*K₂ retained with p₁? **$(beta2_retained)**")
    println(io, "- β₃*K₃ retained with p₁? **$(beta3_retained)**")
end
println("\nWrote: $out_md")
