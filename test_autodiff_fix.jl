# Quick test to verify autodiff works with untyped t parameter
using DifferentialEquations

include("examples/RepressilatorModel.jl")
using .RepressilatorModel

# Test parameters
θ_test = [0.5, 0.5, 0.5, 10.0, 10.0, 10.0, 0.02, 0.01, 0.015,
          30.0, 26.0, 32.0, 0.35, 0.35, 0.35, 0.12, 0.12, 0.12]
X0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
t_obs = [0.0, 1000.0, 2000.0]

println("Testing RepressilatorModel with autodiff enabled (default Rodas4)...")
try
    result = RepressilatorModel.solve_repressilator(t_obs, θ_test, X0)
    println("✓ SUCCESS - autodiff working!")
    println("  Solution size: ", size(result))
    println("  mRNA at t=0: ", result[1:3, 1])
    println("  mRNA at t=2000: ", result[1:3, end])
catch e
    println("✗ FAILED - autodiff error:")
    println(e)
end
