#!/usr/bin/env julia

# Quick test of the reshape issue
using LinearAlgebra

# Simulate the flattened vector from predict_mRNA
# [m1(t1), m2(t1), m3(t1), m1(t2), m2(t2), m3(t2), ...]
NT = 5
flattened = Float64[]
for t in 1:NT
    push!(flattened, 1.0 * t)        # m1(t) = 1*t
    push!(flattened, 2.0 * t)        # m2(t) = 2*t
    push!(flattened, 3.0 * t)        # m3(t) = 3*t
end

println("Flattened vector (time-major):")
println(flattened)
println("\nExpected structure:")
println("  [m1(1), m2(1), m3(1), m1(2), m2(2), m3(2), ..., m1(5), m2(5), m3(5)]")
println("  [1, 2, 3, 2, 4, 6, ..., 5, 10, 15]")

# Reshape like the code does
reshaped = reshape(flattened, 3, NT)
println("\nReshaped to (3, NT):")
println(reshaped)
println("\nRow 1 (should be m1 = [1, 2, 3, 4, 5]): ", reshaped[1, :])
println("Row 2 (should be m2 = [2, 4, 6, 8, 10]): ", reshaped[2, :])
println("Row 3 (should be m3 = [3, 6, 9, 12, 15]): ", reshaped[3, :])

# Check if correct
correct_m1 = [1.0, 2.0, 3.0, 4.0, 5.0]
correct_m2 = [2.0, 4.0, 6.0, 8.0, 10.0]
correct_m3 = [3.0, 6.0, 9.0, 12.0, 15.0]

println("\n✓ Checks:")
println("  m1 correct: ", reshaped[1, :] == correct_m1)
println("  m2 correct: ", reshaped[2, :] == correct_m2)
println("  m3 correct: ", reshaped[3, :] == correct_m3)
