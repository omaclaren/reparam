# Extract IIR sensitivity ranking from a saved repressilator profile result (.jls)
#
# Purpose:
# - Recompute Jacobian at saved θ_MLE
# - Rank identifiable directions by
#     sigma_eff(v) = ||J * v|| / ||v||
# - Output a machine-readable CSV + human-readable markdown table
#
# Usage:
#   julia --project=. extract_iir_diagnostics_from_results.jl <results.jls>

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

# ---------------------------------------------------------------
# 1) Input
# ---------------------------------------------------------------
isempty(ARGS) && error("Usage: julia --project=. extract_iir_diagnostics_from_results.jl <results.jls>")
input_file = ARGS[1]
isfile(input_file) || error("Input file not found: $input_file")

results = deserialize(input_file)

required = ["θ_MLE", "A_T_final", "rank_J", "n_ident"]
missing_keys = filter(k -> !haskey(results, k), required)
isempty(missing_keys) || error("Missing keys in results file: $(missing_keys)")

θ_MLE = Vector{Float64}(results["θ_MLE"])
A_T_final = Matrix{Float64}(results["A_T_final"])
rank_saved = Int(results["rank_J"])
n_ident_saved = Int(results["n_ident"])

grid = get(results, "GRID", "unknown")
mode = get(results, "mode", "unknown")

# ---------------------------------------------------------------
# 2) Recompute J at saved θ_MLE (same setup as profiling script)
# ---------------------------------------------------------------
X0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
T_end = 10000.0
t_iir = LinRange(0, T_end, 501)

# highprec: tighter ODE tolerances for most accurate Jacobian/rank diagnostics.
# Profiling uses looser defaults for speed.
function ϕ_iir_highprec(θ)
    sol_matrix = RepressilatorModel.solve_repressilator(t_iir, θ, X0; abstol=1e-10, reltol=1e-8)
    mRNA = sol_matrix[1:3, :]
    return vec(mRNA)
end
ϕ_iir_log(θ_log) = ϕ_iir_highprec(exp.(θ_log))

θ_log_MLE = log.(θ_MLE)
J = ReparamTools.compute_ϕ_Jacobian(ϕ_iir_log, θ_log_MLE)

# Also recover singular values/rank from the same point using library call
S, _, _, rank_recomputed = ReparamTools.find_invariant_subspace(
    ϕ_iir_log, θ_log_MLE;
    rtol_rank=1e-7,
    rtol_invariance=1e-6,
    verbose=false,
)

# ---------------------------------------------------------------
# 3) Rank identifiable directions by sigma_eff
# ---------------------------------------------------------------
# Assumption from run_repressilator_profile.jl:
# A_T_final columns are ordered as [identifiable..., non-identifiable...].
# This script ranks only the first n_ident columns as identifiable directions.
n_cols = size(A_T_final, 2)
n_ident_saved <= n_cols || error("Inconsistent saved dimensions: n_ident=$n_ident_saved but A_T_final has $n_cols columns")
n_ident = n_ident_saved

param_names = get(results, "param_names",
    ["α₀₁", "α₀₂", "α₀₃", "α₁", "α₂", "α₃",
     "β₁", "β₂", "β₃", "K₁", "K₂", "K₃",
     "k_degm₁", "k_degm₂", "k_degm₃", "k_degp₁", "k_degp₂", "k_degp₃"])

function monomial_string(v::AbstractVector{<:Real}, names::Vector{String}; atol=1e-10)
    terms = String[]
    for i in eachindex(v)
        c = v[i]
        if abs(c) > atol
            push!(terms, "$(names[i])^$(round(c, digits=6))")
        end
    end
    return isempty(terms) ? "1" : join(terms, " * ")
end

rows = NamedTuple[]
for j in 1:n_ident
    v = Vector{Float64}(A_T_final[:, j])
    # Directional sensitivity gain in observable space.
    # For unit-norm v, sigma_eff reduces to ||J * v||.
    sigma_eff = norm(J * v) / max(norm(v), eps())
    coeff_vector = join(string.(round.(v, digits=6)), ";")
    monomial = monomial_string(v, param_names)
    push!(rows, (psi_index=j, sigma_eff=sigma_eff, coeff_vector=coeff_vector, monomial=monomial))
end

sort!(rows, by = r -> -r.sigma_eff)

# ---------------------------------------------------------------
# 4) Write outputs (same filenames as current tracked diagnostics)
# ---------------------------------------------------------------
out_dir = dirname(input_file)
out_csv = joinpath(out_dir, "repressilator_iir_diagnostics_from_saved_results.csv")
out_md = joinpath(out_dir, "repressilator_iir_diagnostics_from_saved_results.md")

open(out_csv, "w") do io
    println(io, "rank_order,psi_index,sigma_eff,coeff_vector,monomial")
    for (k, r) in enumerate(rows)
        monomial_csv = replace(r.monomial, '"' => "\"\"")
        println(io, "$(k),$(r.psi_index),$(r.sigma_eff),\"$(r.coeff_vector)\",\"$(monomial_csv)\"")
    end
end

open(out_md, "w") do io
    println(io, "# IIR sensitivity ranking from saved repressilator result")
    println(io)
    println(io, "Generated: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    println(io)
    println(io, "Input file: `$(input_file)`")
    println(io, "Grid: $(grid)")
    println(io, "Mode: $(mode)")
    println(io)
    println(io, "Saved rank: $(rank_saved)")
    println(io, "Recomputed rank: $(rank_recomputed)")
    println(io, "σ₁ = $(S[1])")
    println(io, "σ_r = $(S[rank_recomputed])")
    if rank_recomputed < length(S)
        println(io, "σ_{r+1} = $(S[rank_recomputed+1])")
        println(io, "gap σ_r/σ_{r+1} = $(S[rank_recomputed] / max(S[rank_recomputed+1], eps()))")
    end
    println(io)

    println(io, "## Identifiable-direction ranking")
    println(io)
    println(io, "| rank | ψ index | sigma_eff | monomial |")
    println(io, "|---:|---:|---:|---|")
    for (k, r) in enumerate(rows)
        println(io, "| $(k) | $(r.psi_index) | $(round(r.sigma_eff, digits=6)) | `$(r.monomial)` |")
    end
end

println("Wrote:")
println("  $out_csv")
println("  $out_md")
