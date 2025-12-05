# Test to examine actual invariance scores for repressilator in both coordinate systems

if !@isdefined(ReparamTools)
    include("ReparamTools.jl")
end

using .ReparamTools
using LinearAlgebra
using ForwardDiff

# Load repressilator model components (just the essentials)
include("examples/repressilator.jl")

println("\n" * "="^70)
println("DETAILED INVARIANCE SCORE ANALYSIS: Repressilator")
println("="^70)

# Use θ_true as test point (we know MLE is close to this)
θ_test = θ_true
θ_log_test = log.(θ_test)

# Short observation grid for speed
t_test = [0.0, 1000.0, 2000.0, 3000.0, 4000.0]

println("\nTest point: θ_true")
println("Using $length(t_test) time points for speed")

# Define auxiliary maps
ϕ_log = θ_log -> begin
    θ = exp.(θ_log)
    sol_matrix = solve_repressilator(t_test, θ, X0)
    mRNA = sol_matrix[1:3, :]  # mRNA concentrations
    return vec(mRNA)
end

ϕ_original = θ -> begin
    sol_matrix = solve_repressilator(t_test, θ, X0)
    mRNA = sol_matrix[1:3, :]  # mRNA concentrations
    return vec(mRNA)
end

# Custom function to compute invariance with detailed output
function analyze_invariance_detailed(ϕ_func, θ0, coord_name)
    println("\n" * "="^70)
    println("$coord_name COORDINATE SYSTEM")
    println("="^70)

    # Compute Jacobian
    J = ForwardDiff.jacobian(ϕ_func, θ0)
    svd_result = svd(J)

    rtolJ = sqrt(eps())
    rank_J = count(svd_result.S .> rtolJ * svd_result.S[1])
    r0 = length(θ0) - rank_J

    println("Jacobian dimensions: ", size(J))
    println("Max singular value: ", round(svd_result.S[1], sigdigits=4))
    println("Rank: $rank_J/18")
    println("Null space dimension: $r0")

    if r0 == 0
        println("No null space - fully identified!")
        return
    end

    # Get null space basis
    V_0 = svd_result.V[:, rank_J+1:end]

    # Finite-difference invariance test
    fd_epsilon = 1e-5
    n_probes = 5
    m = size(J, 1)

    M_test = zeros(m * n_probes, r0)

    println("\nFinite-difference invariance test:")
    println("  ε = $fd_epsilon, n_probes = $n_probes")

    # Compute J(θ+s*α)·α for each null vector α and each probe s
    for j in 1:r0
        α = V_0[:, j]
        for i in 1:n_probes
            s = (i - (n_probes+1)/2) * fd_epsilon
            if abs(s) < 1e-12
                s = fd_epsilon
            end
            θ_pert = θ0 + s * α
            J_pert = ForwardDiff.jacobian(ϕ_func, θ_pert)
            row_start = (i-1)*m + 1
            M_test[row_start:row_start+m-1, j] = J_pert * α
        end
    end

    # Compute invariance scores
    invariance_scores = [norm(M_test[:, j]) for j in 1:r0]

    # Compute threshold
    rtolM = sqrt(eps())
    σ_max = svd_result.S[1]
    τM = rtolM * σ_max

    println("  Tolerance: τM = rtolM × σ_max = $(round(rtolM, sigdigits=3)) × $(round(σ_max, sigdigits=4)) = $(round(τM, sigdigits=4))")

    println("\nInvariance scores (||J(θ+δ)·α|| over all perturbations):")
    for j in 1:r0
        score = invariance_scores[j]
        ratio = score / τM
        status = score < τM ? "✓ PASS" : "✗ FAIL"
        println("  Null vector $j: score = $(round(score, sigdigits=4)), score/τM = $(round(ratio, digits=2)) → $status")
    end

    n_pass = count(invariance_scores .< τM)
    println("\nSummary: $n_pass/$r0 null vectors pass invariance test")

    # If some fail, show how much they fail by
    if n_pass < r0
        failing_indices = findall(invariance_scores .>= τM)
        println("\nFailing vectors (score > τM):")
        for j in failing_indices
            excess = (invariance_scores[j] / τM - 1) * 100
            println("  Vector $j: exceeds threshold by $(round(excess, digits=1))%")
        end
    end

    return invariance_scores, τM
end

# Analyze both coordinate systems
scores_log, τM_log = analyze_invariance_detailed(ϕ_log, θ_log_test, "LOG-SPACE")
scores_orig, τM_orig = analyze_invariance_detailed(ϕ_original, θ_test, "ORIGINAL-SPACE")

println("\n" * "="^70)
println("COMPARISON")
println("="^70)

println("\nScore comparison (smaller = more invariant):")
println("  Null Vec | Log-space score | Original-space score | Ratio (orig/log)")
println("  " * "-"^68)
for j in 1:min(length(scores_log), length(scores_orig))
    ratio = scores_orig[j] / scores_log[j]
    println("  $j        | $(round(scores_log[j], sigdigits=4))            | $(round(scores_orig[j], sigdigits=4))                  | $(round(ratio, digits=2))")
end

println("\nInterpretation:")
println("  - If ratios >> 1: original-space is much less invariant (expected)")
println("  - If ratios ≈ 1: both coordinate systems similarly invariant (surprising!)")
