using Plots
using DifferentialEquations
using LinearAlgebra

# Load the repressilator code to get the functions
include("examples/repressilator.jl")

# Test with TRUE parameters
t_test = LinRange(0, 100, 500)
y_sol = predict_mRNA(θ_true, t_test)

println("Length of y_sol: ", length(y_sol))
println("Should be 3 * 500 = ", 3 * 500)
println("\nFirst 10 elements: ", y_sol[1:10])

# Plot the raw vector
p1 = plot(y_sol, title="Raw predict_mRNA output (STACKED VECTOR)",
          xlabel="Index", ylabel="Value", lw=2, legend=false)

# Now extract properly
y_reshaped = reshape(y_sol, 3, length(t_test))
p2 = plot(t_test, y_reshaped[1, :], title="mRNA 1 (reshaped)",
          xlabel="Time", ylabel="Concentration", lw=2, label="m1")
p3 = plot(t_test, y_reshaped[2, :], title="mRNA 2 (reshaped)",
          xlabel="Time", ylabel="Concentration", lw=2, label="m2")
p4 = plot(t_test, y_reshaped[3, :], title="mRNA 3 (reshaped)",
          xlabel="Time", ylabel="Concentration", lw=2, label="m3")

combined = plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 900))
savefig(combined, "stacked_vector_proof.png")
println("\nPlot saved to stacked_vector_proof.png")
println("The SHARP TRANSITIONS in p1 are between different mRNA species!")
