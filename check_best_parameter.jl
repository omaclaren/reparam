# Quick script to identify best individual parameter for profiling

# After IIR analysis, we have:
# - S: singular values (largest = most identifiable)
# - N_perp: basis for identifiable space (columns are directions)

# The SVD tells us:
# - First few singular values are largest (most identifiable combinations)
# - Last few are smallest (least identifiable combinations)

# For individual parameters in ORIGINAL coordinates:
# We need to check which θ[i] has the strongest contribution to the top singular vectors

# From the IIR analysis output around line 470-480:
# σ[1] = largest (most identifiable combination)
# σ[rank_J] = smallest identifiable (but still above threshold)

# Strategy:
# 1. Look at the first few columns of N_perp (best identified directions)
# 2. See which INDIVIDUAL parameters have largest absolute coefficients
# 3. Profile that parameter

# For repressilator with rank_J=15:
# - Top singular values correspond to best identified combinations
# - But these are COMBINATIONS, not individual parameters
# - Individual parameters might all be poorly identified

# BETTER APPROACH: Look at Varimax-rotated basis
# Varimax tries to make combinations sparse
# If a Varimax direction has only ONE large coefficient, that's a well-identified parameter!

println("""
DIAGNOSIS: Why is the 'identified parameter' profile triangular?
================================================================

The β₁/K₁ RATIO is identified, but we're profiling it as a COMBINATION
in ψ-space, not as an individual parameter.

Issue: The ratio ψ[k] is a TRANSFORMED parameter, and profiling it
correctly requires good coverage of the nuisance parameter space.

If the profile looks triangular, possible causes:
1. Grid resolution too coarse (CONFIG.grid_1d too small)
2. Initial guesses for nuisance parameters not diverse enough
3. Optimization hitting bounds
4. The parameter space is genuinely restricted (curved manifold)

SUGGESTION: Profile a well-identified INDIVIDUAL parameter instead
===================================================================

Best candidates:
1. Degradation rates (k_degm₁, k_degm₂, k_degm₃, k_degp₁, k_degp₂, k_degp₃)
   - These typically have strong signal in data
   - Not involved in β·K products (structurally identifiable)

2. Look for Varimax directions with only 1 significant coefficient
   - These represent individual well-identified parameters

To fix: Add profiling of k_degm₁ (or similar) for comparison
""")
