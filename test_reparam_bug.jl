#!/usr/bin/env julia

include("ReparamTools.jl")
using .ReparamTools
using LinearAlgebra

println("="^70)
println("TESTING REPRESSILATOR'S REPARAM BUG")
println("="^70)

# Mimic repressilator pattern exactly
println("\n1. Build A_varimax_T with columns = exponent vectors")
A_varimax_T = [1.0  0.0    # Column 1: ψ₁ = θ₁^1.0 * θ₂^0.0
               0.5 -1.0]   # Column 2: ψ₂ = θ₁^0.5 * θ₂^(-1.0)
display(A_varimax_T)

println("\n2. Transpose to A_varimax (rows = exponent vectors)")
A_varimax = A_varimax_T'
display(A_varimax)

θ = [2.0, 4.0]
println("\n3. Test with θ = $θ")

# Correct answer
println("\n" * "="^70)
println("CORRECT ANSWER (manual: A_varimax * log(θ))")
println("="^70)
ψ_correct = exp.(A_varimax * log.(θ))
println("ψ = $ψ_correct")
println("  Row 1 of A_varimax: [1.0, 0.5] → ψ[1] = θ₁^1.0 * θ₂^0.5 = 2^1.0 * 4^0.5 = 2*2 = 4.0")
println("  Row 2 of A_varimax: [0.0, -1.0] → ψ[2] = θ₁^0.0 * θ₂^(-1.0) = 1 * (1/4) = 0.25")
println("  Expected: [4.0, 0.25]")

# What repressilator does
println("\n" * "="^70)
println("REPRESSILATOR CODE: reparam(A_varimax)")
println("="^70)
θ_to_ψ, ψ_to_θ = reparam(A_varimax)
ψ_repressilator = θ_to_ψ(θ)
println("ψ = $ψ_repressilator")

# Check
println("\n" * "="^70)
println("VERDICT")
println("="^70)
if isapprox(ψ_repressilator, ψ_correct, atol=1e-10)
    println("✓ CORRECT: reparam(A_varimax) matches manual calculation")
else
    println("❌ BUG CONFIRMED: reparam(A_varimax) gives wrong answer!")
    println("   Correct:  $ψ_correct")
    println("   Got:      $ψ_repressilator")
    println("   Diff:     $(ψ_correct - ψ_repressilator)")

    println("\n" * "="^70)
    println("TESTING THE FIX")
    println("="^70)

    # Fix 1: Pass A_varimax_T instead
    println("\nFix 1: reparam(A_varimax_T)")
    θ_to_ψ_fix1, _ = reparam(A_varimax_T)
    ψ_fix1 = θ_to_ψ_fix1(θ)
    println("  Result: $ψ_fix1")
    println("  Correct? $(isapprox(ψ_fix1, ψ_correct, atol=1e-10))")

    # Fix 2: Manual definition
    println("\nFix 2: Manual exp.(A_varimax * log.(θ))")
    ψ_fix2 = exp.(A_varimax * log.(θ))
    println("  Result: $ψ_fix2")
    println("  Correct? $(isapprox(ψ_fix2, ψ_correct, atol=1e-10))")
end
