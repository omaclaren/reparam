"""
Simple profiling timing test - loads only what's needed
"""

# This is a simplified test to check how long a single profile optimization takes
# Run from repo root: julia examples/test_profile_timing_simple.jl

using Printf

println("\n" * repeat("=", 70))
println("SIMPLIFIED PROFILING TIMING TEST")
println(repeat("=", 70))

# Estimate based on typical ODE optimization with 18 parameters
# These are conservative estimates based on similar problems

println("\n1. Baseline assumptions:")
println("   - 18 parameters (17 nuisance)")
println("   - ODE solve time: ~0.1s per solve")
println("   - Optimization evaluations: ~100-500 per point")
println("   - With optmaxtime=60s: typically hits timeout or converges earlier")

# Estimate time per optimization
println("\n2. Time estimates per profile point:")
t_fast_converge = 10.0  # Fast convergence
t_typical = 30.0        # Typical case
t_timeout = 60.0        # Hits timeout

println("   Best case (fast convergence): $(t_fast_converge)s")
println("   Typical case: $(t_typical)s")
println("   Worst case (timeout): $(t_timeout)s")

# Current configuration
n_1d_points = 10
n_2d_points = 7 * 7
n_guesses = 2
n_profiles_1d = 2  # K₁, β₁
n_profiles_2d = 1  # K₁/β₁ joint

total_optimizations = (n_profiles_1d * n_1d_points + n_profiles_2d * n_2d_points) * n_guesses

println("\n3. Current configuration:")
println("   1D profiles: $(n_profiles_1d) × $(n_1d_points) points × $(n_guesses) guesses = $(n_profiles_1d * n_1d_points * n_guesses) optimizations")
println("   2D profiles: $(n_profiles_2d) × $(n_2d_points) points × $(n_guesses) guesses = $(n_profiles_2d * n_2d_points * n_guesses) optimizations")
println("   Total: $(total_optimizations) optimizations")

println("\n4. Total runtime estimates (optmaxtime=60s):")
@printf("   Best case: %.1f minutes\n", total_optimizations * t_fast_converge / 60)
@printf("   Typical:   %.1f minutes\n", total_optimizations * t_typical / 60)
@printf("   Worst:     %.1f minutes\n", total_optimizations * t_timeout / 60)

# Recommendation 1: Reduce timeout
t_timeout_10s = 10.0
println("\n5. With reduced timeout (optmaxtime=10s):")
@printf("   Typical:   %.1f minutes\n", total_optimizations * t_timeout_10s / 60)
saving = (total_optimizations * t_typical - total_optimizations * t_timeout_10s) / 60
@printf("   Saves:     %.1f minutes vs current\n", saving)

# Recommendation 2: Smaller grid
n_1d_small = 5
n_2d_small = 5 * 5
total_opt_small = (n_profiles_1d * n_1d_small + n_profiles_2d * n_2d_small) * n_guesses

println("\n6. With smaller grid (5 for 1D, 5×5 for 2D, optmaxtime=10s):")
println("   Total optimizations: $(total_opt_small)")
@printf("   Typical runtime: %.1f minutes\n", total_opt_small * t_timeout_10s / 60)

# Recommendation 3: Skip 2D or reduce guesses
println("\n7. Additional options:")
println("   - Skip 2D profile (only 1D): ~$(Int(round((n_profiles_1d * n_1d_points * n_guesses * t_timeout_10s) / 60))) minutes")
println("   - Reduce to 1 guess: ~$(Int(round((total_optimizations * t_timeout_10s / 2) / 60))) minutes")
println("   - Both (1D only, 1 guess): ~$(Int(round((n_profiles_1d * n_1d_points * 1 * t_timeout_10s) / 60))) minutes")

println("\n8. Recommended configuration for paper:")
println("   ✓ grid_steps=[5] for 1D")
println("   ✓ grid_steps=[5,5] for 2D")
println("   ✓ optmaxtime=10s")
println("   ✓ n_guesses=1 (MLE vicinity usually good enough)")
println("   → Expected: ~4 minutes total")

println("\n9. Minimal configuration for testing:")
println("   ✓ grid_steps=[3] for 1D")
println("   ✓ Skip 2D profile")
println("   ✓ optmaxtime=10s")
println("   ✓ n_guesses=1")
println("   → Expected: ~1 minute total")

println("\n" * repeat("=", 70))
println("NOTE: These are estimates. Actual times may vary depending on:")
println("  - ODE solver convergence (stiff problems take longer)")
println("  - Optimizer convergence rate")
println("  - Parameter bounds (tighter = faster)")
println(repeat("=", 70))
