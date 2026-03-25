"""
Quick timing test for repressilator profiling
Tests a single profile point to estimate full runtime
"""

using NLopt

# Load the repressilator setup
include("../repressilator/repressilator.jl")

println("\n" * repeat("=", 70))
println("PROFILING TIMING TEST")
println(repeat("=", 70))

# Test 1: Single 1D profile point optimization
println("\n1. Testing single 1D profile point (K₁)...")
println("   Current settings: grid_steps=[10], optmaxtime=60s, n_guesses=2")
println("   Expected per 1D profile: 10 points × 2 guesses × ~time per optimization")

# Profile K₁ at one point
K1_index = 10
nuisance_indices_K1 = setdiff(1:18, K1_index)
nuisance_guess_K1 = θ_log_MLE[nuisance_indices_K1]

# Single point: fix K₁ at MLE and optimize over nuisance
θ_log_lower_test = θ_log_lower
θ_log_upper_test = θ_log_upper

function test_single_point()
    # Create optimization problem
    opt = Opt(:LN_BOBYQA, length(nuisance_indices_K1))
    opt.lower_bounds = θ_log_lower_test[nuisance_indices_K1]
    opt.upper_bounds = θ_log_upper_test[nuisance_indices_K1]
    opt.maxtime = 60.0

    # Objective: maximize likelihood with K₁ fixed at MLE
    function obj(ω, grad)
        θ_full = copy(θ_log_MLE)
        θ_full[nuisance_indices_K1] = ω
        return lnlike_θ_log(θ_full)
    end

    opt.max_objective = obj

    # Run optimization
    (maxf, maxω, ret) = optimize(opt, nuisance_guess_K1)
    return (maxf, ret)
end

# Time it
println("\n   Timing single optimization...")
t_single = @elapsed begin
    result = test_single_point()
end
println("   Result: $(result[2]) after $(round(t_single, digits=2))s")

# Estimates
println("\n2. Runtime estimates:")
println("   Single point: $(round(t_single, digits=1))s")
println("   Full 1D profile (10 points × 2 guesses): ~$(round(10 * 2 * t_single / 60, digits=1)) minutes")
println("   Full 2D profile (7×7 points × 2 guesses): ~$(round(7 * 7 * 2 * t_single / 60, digits=1)) minutes")
println("   Total for 3 profiles (K₁, β₁, K₁/β₁ joint): ~$(round((10*2 + 10*2 + 7*7*2) * t_single / 60, digits=1)) minutes")

# Test with shorter timeout
println("\n3. Testing with reduced timeout (10s)...")
function test_single_point_fast()
    opt = Opt(:LN_BOBYQA, length(nuisance_indices_K1))
    opt.lower_bounds = θ_log_lower_test[nuisance_indices_K1]
    opt.upper_bounds = θ_log_upper_test[nuisance_indices_K1]
    opt.maxtime = 10.0  # Reduced

    function obj(ω, grad)
        θ_full = copy(θ_log_MLE)
        θ_full[nuisance_indices_K1] = ω
        return lnlike_θ_log(θ_full)
    end

    opt.max_objective = obj
    (maxf, maxω, ret) = optimize(opt, nuisance_guess_K1)
    return (maxf, ret)
end

t_fast = @elapsed begin
    result_fast = test_single_point_fast()
end
println("   Result: $(result_fast[2]) after $(round(t_fast, digits=2))s")
println("   Likelihood difference from 60s: $(round(result_fast[1] - result[1], digits=6))")

println("\n4. Runtime estimates with 10s timeout:")
println("   Full 1D profile (10 points × 2 guesses): ~$(round(10 * 2 * t_fast / 60, digits=1)) minutes")
println("   Full 2D profile (7×7 points × 2 guesses): ~$(round(7 * 7 * 2 * t_fast / 60, digits=1)) minutes")
println("   Total for 3 profiles: ~$(round((10*2 + 10*2 + 7*7*2) * t_fast / 60, digits=1)) minutes")

# Test with even smaller grid
println("\n5. Recommendations:")
if t_single > 30
    println("   ⚠️  Current optmaxtime=60s is quite slow")
    println("   Consider: optmaxtime=10s (saves $(round((t_single - t_fast) * (10*2 + 10*2 + 7*7*2) / 60, digits=1)) min)")
end
if (10*2 + 10*2 + 7*7*2) * t_fast / 60 > 10
    println("   ⚠️  Even with 10s timeout, total > 10 minutes")
    println("   Consider: grid_steps=[5] for 1D, [5,5] for 2D")
    println("   Estimated time: ~$(round((5*2 + 5*2 + 5*5*2) * t_fast / 60, digits=1)) minutes")
end

println("\n" * repeat("=", 70))
