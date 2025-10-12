# PK Model: Dictionary Approach Findings

## Model: Nonlinear 2-Compartment PK (Meshkat et al. 2011)

**Parameters**: [b₁, c₁, k₀₁, k₀₂, k₁₂, k₂₁, V_M, K_M] (8 parameters)

**Known identifiable combinations** (from Meshkat 2011):
1. q₁ = b₁c₁
2. q₂ = c₁K_M
3. q₃ = k₀₂ + k₁₂
4. q₄ = c₁V_Mk₁₂k₂₁
5. q₅ = c₁V_M(k₀₁ + k₂₁)

Note: Combinations include both **products** (q₁, q₂, q₄) and **sums** (q₃, and the sum within q₅).

## Numerical Results

### Stage 1 (f=log + Varimax)

- **Rank**: 6/8
- **Dim(N_perp)**: 7 (potentially identifiable)
- **Dim(N)**: 1 (structurally non-identifiable)

**Varimax-rotated coordinates** (after sign correction):
- θ¹[1] = k₂₁
- θ¹[2] = k₁₂
- θ¹[3] = c₁ * K_M^0.5
- θ¹[4] = (some complex combination involving V_M, K_M)
- θ¹[5] = k₀₁
- θ¹[6] = (combination)
- θ¹[7] = k₀₂
- θ¹[8] = K_M

**Observation**: Varimax successfully separated individual rate constants (k₁₂, k₂₁, k₀₁, k₀₂) but did NOT produce Meshkat's combinations.

### Stage 2 (f=identity on Stage 1 coordinates)

- **Rank**: 6 (should be 5 according to Meshkat)
- **Dim(N)**: 1 (when using orthonormal basis) or 0 (when using scaled basis)
- **Dim(N_perp)**: 7 or 8

**Found combinations**:
- Column 4: θ¹[1] + θ¹[7] = k₂₁ + k₀₂ (a sum!)

**Expected**: k₀₂ + k₁₂ (Meshkat's q₃)

**Discrepancy**: Stage 2 found a sum (good!) but the wrong sum (k₂₁ + k₀₂ instead of k₀₂ + k₁₂).

### Rank Discrepancy Investigation

**Expected rank** (from Meshkat): 5 identifiable combinations

**Observed rank**: 6 consistently across multiple tests:
- Stage 2 with scaled basis: rank 6
- Stage 2 with orthonormal basis: rank 6
- Stage 2 with f=log: rank 6
- Stage 2 with f=identity: rank 6

**Singular values** (typical):
```
[1]: 0.765
[2]: 0.238
[3]: 0.116
[4]: 0.020
[5]: 0.000212
[6]: 1.08e-6   ← borderline (67× machine epsilon)
[7]: 0.0
[8]: 0.0
```

The 6th singular value is consistently ~1e-6, which is **barely above numerical noise** but fails the `rtol = sqrt(eps()) ≈ 1.5e-8` threshold.

**Hypothesis**: The Varimax transformation is not producing a clean separation of identifiable combinations. The extra 6th "direction" is likely a numerical artifact from the fact that our Stage 1 coordinates don't align with Meshkat's true identifiable combinations.

## Fundamental Limitation Discovered

**Varimax in log-space can only find multiplicative (monomial) combinations**, not additive ones.

- **Can find**: b₁c₁, c₁K_M, k₁₂/k₂₁ (products/ratios)
- **Cannot find**: k₀₂ + k₁₂, k₀₁ + k₂₁ (sums)

**Meshkat's combinations** require:
- 3 pure products: q₁ = b₁c₁, q₂ = c₁K_M, q₄ = c₁V_Mk₁₂k₂₁
- 2 involving sums: q₃ = k₀₂ + k₁₂, q₅ = c₁V_M(k₀₁ + k₂₁)

**Sequential IIR strategy**:
- **Stage 1** (f=log + Varimax): Find multiplicative combinations
- **Stage 2** (f=identity): Find additive combinations among Stage 1 coordinates

**This should work in principle**, but we're seeing:
1. Varimax isolated individual parameters (k₀₁, k₀₂, k₁₂, k₂₁) rather than products
2. Stage 2 found a sum, but the wrong one (k₂₁ + k₀₂ vs k₀₂ + k₁₂)
3. Rank remains 6 instead of reducing to 5

## Why Dictionary Approach Struggles Here

The dictionary approach (providing multiple bases including Varimax) works well when:
- Identifiable combinations are purely **multiplicative** (products/ratios)
- OR the Stage 1 transformation cleanly separates identifiable from non-identifiable directions

For the PK model:
- Identifiable combinations include **both multiplicative and additive** structure
- Varimax doesn't align with the true identifiable combinations
- Stage 2 has to "fix" the misalignment, but this creates numerical issues
- Result: borderline 6th singular value that shouldn't be there

## Comparison with Working Example (stat_sum_model)

**stat_sum_model** (sum of independent Poisson limits):
- **Parameters**: [n₁, p₁, n₂, p₂]
- **Identifiable**: θ¹ = n₁p₁ (product), θ² = n₂p₂ (product)
- **Stage 1** (f=log + Varimax): Finds products cleanly
- **Stage 2** (f=identity): Finds θ¹ + θ² (sum of products)
- **Result**: ✅ Works perfectly, clean rank reduction

**Why it works**: Stage 1 gets the "right" coordinates (the products n₁p₁, n₂p₂) that align with what's identifiable. Stage 2 just needs to find their sum.

**PK model**: Stage 1 does NOT get the "right" coordinates. Varimax finds individual parameters instead of the products that are identifiable, leaving Stage 2 to clean up a more complex mess.

## Sign Correction Strategy

Implemented logic to flip Varimax columns with negative dominant exponents:
```julia
for col in 1:size(N_perp_varimax, 2)
    v = N_perp_varimax[:, col]
    max_idx = argmax(abs.(v))
    max_val = v[max_idx]

    if max_val < -0.5
        # Flip to convert reciprocal (1/k) to direct parameter (k)
        N_perp_corrected[:, col] = -N_perp_corrected[:, col]
    end
end
```

**Result**: Successfully converted reciprocals to direct parameters (e.g., 1/k₀₁ → k₀₁), but this doesn't solve the deeper issue that Varimax isn't finding Meshkat's combinations.

## Conclusions

1. **Dictionary approach with Varimax works for models where identifiable combinations are purely multiplicative** (like stat_sum_model)

2. **For models with mixed multiplicative + additive structure** (like PK model), Varimax may not align with the true identifiable combinations, leading to:
   - Wrong sum found at Stage 2
   - Borderline singular values causing rank ambiguity
   - Numerical artifacts

3. **The "sign lottery" problem** (Varimax choosing reciprocals) can be addressed with sign correction, but this is a superficial fix

4. **Fundamental limitation**: Without prior knowledge of which combinations are identifiable, Varimax is a heuristic that may or may not produce useful coordinates for Stage 2

## Next Steps / Open Questions

1. Is there a different rotation (not Varimax) that would work better for this model?

2. Should we accept that the dictionary approach is **best suited for models with purely multiplicative combinations**?

3. Would using Meshkat's COMBOS tool to pre-compute identifiable combinations be more reliable than the dictionary approach?

4. Is rank = 6 vs 5 a practical problem, or acceptable given the 6th value is ~1e-6?

5. Should the paper position the dictionary approach as:
   - A general solution? (Optimistic, not supported by PK model)
   - A solution for multiplicative-combination models? (Honest, supported by stat_sum)
   - An optional enhancement that may help in some cases? (Conservative)
