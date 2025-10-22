# Test stat_model with BOTH Hessian and Finite-Difference methods

if !@isdefined(ReparamTools)
    include("ReparamTools.jl")
end

using .ReparamTools
using LinearAlgebra

println("="^70)
println("STAT_MODEL: Comparing Hessian vs Finite-Difference Methods")
println("="^70)

# Simple auxiliary map: ϕ(n,p) = [np, np]
ϕ_xy = xy -> [xy[1]*xy[2], xy[1]*xy[2]]
ϕ_log = xy_log -> ϕ_xy(exp.(xy_log))

# Test point
xy_test = [100.0, 0.1]
xy_log_test = log.(xy_test)

println("\nTest point: θ = $xy_test (n, p)")

# Test in LOG-SPACE with both methods
println("\n" * "="^70)
println("LOG-SPACE ANALYSIS")
println("="^70)

println("\n1. HESSIAN-BASED METHOD:")
S_hess, N_hess, N_perp_hess, rank_hess = find_invariant_subspace(
    ϕ_log, xy_log_test;
    # No invariance_method specified = uses Hessian (default)
    verbose=true
)
n_inv_hess = size(N_hess, 2)
println("  Result: $n_inv_hess/$(2-rank_hess) null vectors invariant")

println("\n2. FINITE-DIFFERENCE METHOD:")
S_fd, N_fd, N_perp_fd, rank_fd = find_invariant_subspace(
    ϕ_log, xy_log_test;
    invariance_method=:finite_difference,
    verbose=true
)
n_inv_fd = size(N_fd, 2)
println("  Result: $n_inv_fd/$(2-rank_fd) null vectors invariant")

println("\n" * "="^70)
println("COMPARISON:")
println("="^70)
println("  Method          | Rank | Null Dim | Invariant | Match")
println("  " * "-"^60)
hess_status = n_inv_hess == (2-rank_hess) ? "✓" : "✗"
fd_status = n_inv_fd == (2-rank_fd) ? "✓" : "✗"
match = (rank_hess == rank_fd && n_inv_hess == n_inv_fd) ? "✓" : "✗"
println("  Hessian         | $rank_hess/2  |    $(2-rank_hess)     |    $n_inv_hess      | $hess_status")
println("  Finite-Diff     | $rank_fd/2  |    $(2-rank_fd)     |    $n_inv_fd      | $fd_status")
println("  Agreement: $match")

if match == "✓"
    println("\n✓ Both methods agree! Finite-difference implementation is correct.")
else
    println("\n✗ Methods disagree - there's a problem!")
end

# Test in ORIGINAL-SPACE with both methods
println("\n" * "="^70)
println("ORIGINAL-SPACE ANALYSIS")
println("="^70)

println("\n1. HESSIAN-BASED METHOD:")
S_hess_orig, N_hess_orig, N_perp_hess_orig, rank_hess_orig = find_invariant_subspace(
    ϕ_xy, xy_test;
    verbose=true
)
n_inv_hess_orig = size(N_hess_orig, 2)
println("  Result: $n_inv_hess_orig/$(2-rank_hess_orig) null vectors invariant")

println("\n2. FINITE-DIFFERENCE METHOD:")
S_fd_orig, N_fd_orig, N_perp_fd_orig, rank_fd_orig = find_invariant_subspace(
    ϕ_xy, xy_test;
    invariance_method=:finite_difference,
    verbose=true
)
n_inv_fd_orig = size(N_fd_orig, 2)
println("  Result: $n_inv_fd_orig/$(2-rank_fd_orig) null vectors invariant")

println("\n" * "="^70)
println("COMPARISON:")
println("="^70)
println("  Method          | Rank | Null Dim | Invariant | Match")
println("  " * "-"^60)
hess_status_orig = n_inv_hess_orig == (2-rank_hess_orig) ? "✓" : "✗"
fd_status_orig = n_inv_fd_orig == (2-rank_fd_orig) ? "✓" : "✗"
match_orig = (rank_hess_orig == rank_fd_orig && n_inv_hess_orig == n_inv_fd_orig) ? "✓" : "✗"
println("  Hessian         | $rank_hess_orig/2  |    $(2-rank_hess_orig)     |    $n_inv_hess_orig      | $hess_status_orig")
println("  Finite-Diff     | $rank_fd_orig/2  |    $(2-rank_fd_orig)     |    $n_inv_fd_orig      | $fd_status_orig")
println("  Agreement: $match_orig")

if match_orig == "✓"
    println("\n✓ Both methods agree! Finite-difference implementation is correct.")
else
    println("\n✗ Methods disagree - there's a problem!")
end
