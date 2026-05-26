# Run with:
#   julia --project=. "examples/repressilator_unknown_n_iir.jl"
#
# Supplementary/basic IIR diagnostic for the repressilator with the Hill
# coefficient n treated as an unknown parameter.
#
# Why this example is included:
#   With n fixed, the maintained repressilator workflow finds the familiar
#   mRNA-only scaling non-identifiabilities: βᵢ and Kᵢ are not separately
#   identifiable, while ratios such as βᵢ/Kᵢ are retained image coordinates.
#   If n is also unknown, the repression term (pᵢ/Kᵢ)^n is no longer a monomial
#   or rational function of the original parameter vector. This example shows
#   that this is generally fine for IIR.
#
# In particular, the protein equation still depends on βᵢ and Kᵢ only through
# βᵢ/Kᵢ, which is monomial, while n is identifiable and treated as a separate
# parameter. In log coordinates this is the statement that the mRNA auxiliary
# map factors through coordinates such as
#
#     log βᵢ - log Kᵢ,    log n,
#
# followed by a nonlinear/numerical ODE-solution map. The expected result is
# therefore rank 16 in 19 parameters, with the same three log-monomial
# null/free coordinates βᵢKᵢ and with n on the image/complement side.
#
# This script deliberately does not run likelihood profiling or PWA. It only
# builds the mRNA-only auxiliary map, runs the local invariant-image split in log
# coordinates, searches for sparse log-monomial bases, and validates the selected
# null/free directions by direct perturbation.

if !@isdefined(ReparamTools)
    include("../ReparamTools.jl")
    println("✓ ReparamTools module included")
else
    println("✓ ReparamTools module already included")
end

using .ReparamTools
using DifferentialEquations
using LinearAlgebra
using Printf

# --------------------------------------------------------
# Repressilator with unknown Hill coefficient
# --------------------------------------------------------

"""
    repressilator_unknown_n!(dX, X, θ, t)

Eisenberg--Hayashi-style repressilator with parameter vector

    θ = [α₀₁, α₀₂, α₀₃, α₁, α₂, α₃,
         β₁, β₂, β₃, K₁, K₂, K₃,
         k_degm₁, k_degm₂, k_degm₃, k_degp₁, k_degp₂, k_degp₃, n]

The usual fixed value n = 2.5 is replaced by the unknown positive parameter
`θ[19]`.
"""
function repressilator_unknown_n!(dX, X, θ, t)
    m₁, m₂, m₃, p₁, p₂, p₃ = X

    α₀₁, α₀₂, α₀₃ = θ[1], θ[2], θ[3]
    α₁, α₂, α₃ = θ[4], θ[5], θ[6]
    β₁, β₂, β₃ = θ[7], θ[8], θ[9]
    K₁, K₂, K₃ = θ[10], θ[11], θ[12]
    k_degm₁, k_degm₂, k_degm₃ = θ[13], θ[14], θ[15]
    k_degp₁, k_degp₂, k_degp₃ = θ[16], θ[17], θ[18]
    n = θ[19]

    dX[1] = α₀₁ + α₁ / (1 + (p₃ / K₃)^n) - k_degm₁ * m₁
    dX[2] = α₀₂ + α₂ / (1 + (p₁ / K₁)^n) - k_degm₂ * m₂
    dX[3] = α₀₃ + α₃ / (1 + (p₂ / K₂)^n) - k_degm₃ * m₃

    dX[4] = β₁ * m₁ - k_degp₁ * p₁
    dX[5] = β₂ * m₂ - k_degp₂ * p₂
    dX[6] = β₃ * m₃ - k_degp₃ * p₃

    return nothing
end

function solve_repressilator_unknown_n(t_save, θ, X0; abstol=1e-9, reltol=1e-7)
    T = promote_type(eltype(θ), eltype(X0), eltype(t_save))
    u0 = T.(X0)
    ts = T.(collect(t_save))
    prob = ODEProblem(repressilator_unknown_n!, u0, (zero(T), maximum(ts)), θ)

    # Tsit5 is used here because the IIR invariance test differentiates through
    # the Jacobian computation (nested ForwardDiff dual numbers). In this setup
    # Tsit5 is more robust than the stiff Rosenbrock-family solvers for the
    # nested-dual preflight.
    sol = solve(prob, Tsit5(); saveat=ts, abstol=abstol, reltol=reltol)
    return Array(sol)
end

# --------------------------------------------------------
# Reference point and auxiliary mapping
# --------------------------------------------------------

param_names = [
    "α₀₁", "α₀₂", "α₀₃", "α₁", "α₂", "α₃",
    "β₁", "β₂", "β₃", "K₁", "K₂", "K₃",
    "k_degm₁", "k_degm₂", "k_degm₃", "k_degp₁", "k_degp₂", "k_degp₃", "n",
]

θ_ref = [
    0.008, 0.009, 0.010,      # α₀
    1.0, 1.2, 1.5,            # α
    0.02, 0.025, 0.015,       # β
    30.0, 28.0, 32.0,         # K
    0.006, 0.0055, 0.0065,    # k_degm
    0.0012, 0.0011, 0.0013,   # k_degp
    2.5,                      # n, now unknown
]

X0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
T_end = 10000.0
# Fine enough to recover the same structural split, but cheaper than the full
# profile/PWA workflow.
t_iir = collect(LinRange(0.0, T_end, 101))

function ϕ_mrna(θ)
    sol_matrix = solve_repressilator_unknown_n(t_iir, θ, X0)
    return vec(sol_matrix[1:3, :])
end

# Log-coordinate IIR for positive parameters.
θlog_ref = log.(θ_ref)
ϕ_log(θlog) = ϕ_mrna(exp.(θlog))

println("\n" * "="^72)
println("REPRESSILATOR UNKNOWN-n BASIC IIR TEST")
println("="^72)
println("Parameters: $(length(θ_ref)) (18 original + unknown n)")
println("Auxiliary map: mRNA time course only, length $(length(ϕ_log(θlog_ref)))")
println("Reference n = $(θ_ref[end])")

# --------------------------------------------------------
# ForwardDiff preflight and IIR
# --------------------------------------------------------

println("\nForwardDiff preflight...")
J_preflight = ReparamTools.compute_ϕ_Jacobian(ϕ_log, θlog_ref)
println("  Jacobian size: $(size(J_preflight, 1)) × $(size(J_preflight, 2))")
println("  finite entries: ", all(isfinite, J_preflight))

println("\nRunning IIR split...")
@time S, N, N_perp, rank_J = ReparamTools.find_invariant_subspace(
    ϕ_log, θlog_ref;
    rtol_rank=1e-7,
    rtol_invariance=1e-6,
    verbose=true,
)

p = length(θ_ref)
local_null_dim = p - rank_J
println("\nSummary:")
println("  rank_J = $rank_J / $p")
println("  local null dimension = $local_null_dim")
println("  invariant-null dimension = $(size(N, 2))")
println("  image/complement dimension = $(size(N_perp, 2))")
println("  smallest singular values = ", round.(S[max(1, end-5):end], sigdigits=5))

# --------------------------------------------------------
# Sparse log-monomial coordinates for a non-monomial auxiliary map
# --------------------------------------------------------

println("\nSparse basis search in log coordinates...")
residual_cap = 1e-2

# Use the simple sparse search for the image side here: it gives a readable
# coordinate set with n appearing as its own retained coordinate. The informed
# search can choose equally valid mixed coordinates such as k_degp₂/n and
# k_degp₂*n, which are less convenient for this diagnostic example.
image_basis = ReparamTools.simple_monomial_basis_search(
    N_perp,
    param_names;
    s_max=2,
    c_max=1,
    residual_cap=residual_cap,
    retry_support=true,
)

null_basis = ReparamTools.simple_monomial_basis_search(
    N,
    param_names;
    s_max=2,
    c_max=1,
    residual_cap=residual_cap,
    retry_support=true,
)

println("  image basis ok: ", image_basis.basis_ok,
        " (selected rank $(image_basis.selected_rank) / $(image_basis.target_dim), effective s_max=$(image_basis.effective_s_max))")
println("  null basis ok:  ", null_basis.basis_ok,
        " (selected rank $(null_basis.selected_rank) / $(null_basis.target_dim), effective s_max=$(null_basis.effective_s_max))")

image_labels = ReparamTools.basis_labels(image_basis.selected)
null_labels = ReparamTools.basis_labels(null_basis.selected)

println("\nSelected image/complement log-monomial coordinates:")
for (j, label) in enumerate(image_labels)
    println("  ψ_$j = ", label)
end

println("\nSelected null/free log-monomial coordinates:")
if isempty(null_labels)
    println("  (none)")
else
    for (j, label) in enumerate(null_labels)
        println("  ζ_$j = ", label)
    end
end

println("\nChecks:")
println("  n appears in null/free labels? ", any(contains("n"), null_labels))
println("  n appears in image/complement labels? ", any(contains("n"), image_labels))
println("  expected βᵢ*Kᵢ null labels present? ", all(x -> x in Set(null_labels), ["β₁*K₁", "β₂*K₂", "β₃*K₃"]))

# Directly perturb the selected null/free monomial directions in log space.
println("\nPerturbation check along selected null/free directions:")
y_ref = ϕ_log(θlog_ref)
for cand in null_basis.selected
    v = Float64.(cand.v)
    v ./= norm(v)
    for step in (0.1, 0.5)
        y_step = ϕ_log(θlog_ref .+ step .* v)
        rel_change = norm(y_step - y_ref) / norm(y_ref)
        @printf("  %-10s step=%0.2f  relative output change = %.3e\n", cand.label, step, rel_change)
    end
end

println("\nInterpretation:")
println("  With n treated as an unknown positive parameter, the mRNA-only auxiliary")
println("  map is no longer monomial in the original parameters, but it still has")
println("  the three βᵢ/Kᵢ scaling symmetries: simultaneous scaling of βᵢ and Kᵢ")
println("  leaves pᵢ/Kᵢ unchanged, regardless of n. The Hill coefficient n is")
println("  retained on the image/complement side at this reference point, rather")
println("  than adding a new null/free direction.")
