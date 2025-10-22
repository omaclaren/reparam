using Plots

# Simulate what the code does
println("Testing repressilator plotting logic...")

# Simulate ODE solution: 3 species × 10 time points
NT = 10
t = 1:NT

# Create fake trajectories
mrna_direct = zeros(3, NT)
mrna_direct[1, :] = sin.(t)      # mRNA 1: sine wave
mrna_direct[2, :] = cos.(t)      # mRNA 2: cosine wave
mrna_direct[3, :] = sin.(t .+ 1) # mRNA 3: shifted sine

println("\nDirect matrix (3×NT):")
println("  Row 1 (mRNA 1): ", round.(mrna_direct[1, :], digits=2))
println("  Row 2 (mRNA 2): ", round.(mrna_direct[2, :], digits=2))
println("  Row 3 (mRNA 3): ", round.(mrna_direct[3, :], digits=2))

# Simulate what predict_mRNA does: flatten in time-major order
flattened = vec(mrna_direct')  # Transpose then vectorize
println("\nFlattened vector (time-major):")
println("  First 9 elements: ", round.(flattened[1:9], digits=2))
println("  (Should be: m1(t1), m2(t1), m3(t1), m1(t2), m2(t2), m3(t2), m1(t3), m2(t3), m3(t3))")

# Reshape back (WRONG way - what the old code did)
reshaped_wrong = reshape(flattened, 3, NT)

# Reshape back (CORRECT way - what the new code does)
reshaped = reshape(flattened, 3, :)'  # NT×3, then transpose to 3×NT
println("\nReshaped back to 3×NT:")
println("  Row 1: ", round.(reshaped[1, :], digits=2))
println("  Row 2: ", round.(reshaped[2, :], digits=2))
println("  Row 3: ", round.(reshaped[3, :], digits=2))

# Check if they match
println("\nDo they match?")
println("  Row 1 matches: ", mrna_direct[1, :] ≈ reshaped[1, :])
println("  Row 2 matches: ", mrna_direct[2, :] ≈ reshaped[2, :])
println("  Row 3 matches: ", mrna_direct[3, :] ≈ reshaped[3, :])

# Plot comparison
p1 = plot(t, mrna_direct[1, :], title="Direct mRNA 1", label="Direct", lw=2)
p2 = plot(t, reshaped[1, :], title="Reshaped mRNA 1", label="Reshaped", lw=2)
p3 = plot(t, mrna_direct[2, :], title="Direct mRNA 2", label="Direct", lw=2)
p4 = plot(t, reshaped[2, :], title="Reshaped mRNA 2", label="Reshaped", lw=2)

combined = plot(p1, p2, p3, p4, layout=(2, 2), size=(1000, 800))
savefig(combined, "test_reshape_consistency.png")
println("\nPlot saved to test_reshape_consistency.png")
