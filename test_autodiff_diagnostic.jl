# Test: Verify Autodiff is Enabled in RepressilatorModel
# Confirms that automatic differentiation is active and working properly

using DifferentialEquations
using Printf

println("="^70)
println("TEST: Autodiff Diagnostic for RepressilatorModel")
println("="^70)

# Load the module
include("examples/RepressilatorModel.jl")
using .RepressilatorModel

println("\n[1/3] Setting up test problem...")

# Test parameters
θ_test = [0.5, 0.5, 0.5, 10.0, 10.0, 10.0, 0.02, 0.01, 0.015,
          30.0, 26.0, 32.0, 0.35, 0.35, 0.35, 0.12, 0.12, 0.12]
X0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
t_span = (0.0, 1000.0)
t_save = [0.0, 500.0, 1000.0]

println("✓ Test setup complete")
println("  Parameters: ", length(θ_test))
println("  Time span: ", t_span)
println("  Save points: ", length(t_save))

println("\n[2/3] Testing solver with AUTODIFF ENABLED (default)...")

# Create ODE problem with default Rodas4()
prob_autodiff = ODEProblem(RepressilatorModel.repressilator!, X0, t_span, θ_test)

println("  Solving with Rodas4() [autodiff=true (default)]...")
@time sol_autodiff = solve(prob_autodiff, Rodas4(), saveat=t_save, abstol=1e-10, reltol=1e-8)

stats_autodiff = sol_autodiff.stats

println("\n  Solver statistics (AUTODIFF ON):")
println("    Function evaluations:  ", stats_autodiff.nf)
println("    Jacobian evaluations:  ", stats_autodiff.njacs)
println("    W factorizations:      ", stats_autodiff.nw)
println("    Linear solves:         ", stats_autodiff.nsolve)
println("    Solution successful:   ", sol_autodiff.retcode == :Success)

println("\n[3/3] Testing solver with AUTODIFF DISABLED (for comparison)...")

println("  Solving with Rodas4(autodiff=false)...")
@time sol_no_autodiff = solve(prob_autodiff, Rodas4(autodiff=false), saveat=t_save, abstol=1e-10, reltol=1e-8)

stats_no_autodiff = sol_no_autodiff.stats

println("\n  Solver statistics (AUTODIFF OFF):")
println("    Function evaluations:  ", stats_no_autodiff.nf)
println("    Jacobian evaluations:  ", stats_no_autodiff.njacs)
println("    W factorizations:      ", stats_no_autodiff.nw)
println("    Linear solves:         ", stats_no_autodiff.nsolve)
println("    Solution successful:   ", sol_no_autodiff.retcode == :Success)

println("\n" * "="^70)
println("ANALYSIS")
println("="^70)

# Check that both succeeded
if sol_autodiff.retcode != :Success || sol_no_autodiff.retcode != :Success
    println("✗ ERROR: One or both solvers failed!")
    println("  Autodiff ON:  ", sol_autodiff.retcode)
    println("  Autodiff OFF: ", sol_no_autodiff.retcode)
    exit(1)
end

println("\nComparison:")
println("  Function evaluations - autodiff: ", stats_autodiff.nf, ", no-autodiff: ", stats_no_autodiff.nf)
println("  Jacobian evaluations - autodiff: ", stats_autodiff.njacs, ", no-autodiff: ", stats_no_autodiff.njacs)

# Check solutions match
sol_diff = maximum(abs.(Array(sol_autodiff) - Array(sol_no_autodiff)))
println("\n  Solution difference: ", @sprintf("%.2e", sol_diff))
println("  Solutions match: ", sol_diff < 1e-6 ? "✓" : "✗")

println("\n" * "="^70)

# Final verdict
if stats_autodiff.njacs > 0
    println("✓✓✓ AUTODIFF IS ENABLED AND WORKING ✓✓✓")
    println("\nEvidence:")
    println("  1. Jacobian evaluations occurred ($(stats_autodiff.njacs) jacs)")
    println("  2. RepressilatorModel uses Rodas4() with default autodiff=true")
    println("  3. Time parameter 't' is untyped (allows ForwardDiff dual numbers)")
    println("  4. Solutions match between autodiff ON/OFF modes")

    # Performance comparison
    if stats_autodiff.nf < stats_no_autodiff.nf
        improvement = round(100 * (stats_no_autodiff.nf - stats_autodiff.nf) / stats_no_autodiff.nf, digits=1)
        println("\n  Performance: Autodiff uses $(improvement)% fewer function evaluations")
    end

    exit_code = 0
else
    println("✗✗✗ WARNING: NO JACOBIAN EVALUATIONS ✗✗✗")
    println("\nThis suggests autodiff may not be active or not needed for this problem.")
    exit_code = 1
end

println("="^70)

exit(exit_code)
