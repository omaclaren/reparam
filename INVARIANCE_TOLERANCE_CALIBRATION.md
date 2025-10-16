# Invariance Tolerance Calibration for Stiff ODE Systems

## Summary

Updated `find_invariant_subspace` to use **rtolM = 32√eps ≈ 4.8e-7** as the default relative tolerance for the invariance test. This value is calibrated for stiff ODE systems like the repressilator and provides ~2× safety margin over the minimum required value.

## Problem

The previous default (`rtolM = √eps ≈ 1.5e-8`) was too tight for stiff ODE systems:
- ODE solver introduces numerical noise (tolerances ~1e-6 to 1e-10)
- Hessian-based invariance test produces MS values in range 1e-6 to 1e-4
- With tight tolerance, some invariant directions were misclassified as non-invariant

## Solution: Calibrated Default

### Sweep Results (Repressilator Example)

With σ_max ≈ 600 and MS values [1.32e-4, 1.76e-5, 6.58e-6]:

| rtolM       | τM        | N_inv | Status |
|-------------|-----------|-------|--------|
| 1.5e-8 (old)| 8.9e-6    | 1/3   | ✗ FAIL |
| 2.0e-7      | 1.2e-4    | 2/3   | ✗ FAIL |
| **3.0e-7**  | **1.8e-4**| **3/3**| **✓ PASS** (minimum) |
| 4.0e-7      | 2.4e-4    | 3/3   | ✓ PASS |
| **4.8e-7 (new)** | **2.9e-4** | **3/3** | **✓ PASS** (default) |
| 5.0e-7      | 3.0e-4    | 3/3   | ✓ PASS |
| 1.0e-6      | 6.0e-4    | 3/3   | ✓ PASS |

**Minimum value:** rtolM ≈ 3e-7 (just works)
**Safe value:** rtolM ≈ 3.3e-7 (1.5× margin)
**Chosen default:** rtolM = 32√eps ≈ 4.8e-7 (~2× margin)

### Derivation

For MS values to be classified as invariant, need:
```
MS[i] < τM
MS[i] < rtolM * σ_max
rtolM > MS[i] / σ_max
```

For repressilator:
```
rtolM_min > max(MS) / σ_max
          > 1.32e-4 / 600
          > 2.2e-7
```

With 1.5× safety margin:
```
rtolM_safe ≈ 3.3e-7
```

Chosen value `32√eps ≈ 4.8e-7` provides ~2.2× margin over minimum.

### Heuristic Formula

```julia
rtolM ≳ 1.5 * max(MS_invariant) / σ_max
```

To find appropriate value:
1. Run with `verbose=true` to see MS values
2. Calculate: `rtolM_min = max(MS) / σ_max`
3. Apply safety margin: `rtolM = 1.5 * rtolM_min`

## Implementation Changes

### invariance.jl

**Old default:**
```julia
rtolM=sqrt(eps(real(eltype(θ0))))  # ≈ 1.5e-8
```

**New default:**
```julia
rtolM=32*sqrt(eps(real(eltype(θ0))))  # ≈ 4.8e-7
```

**Updated docstring:**
```
- `rtolM`: Relative tolerance for invariance test (default: 32√eps ≈ 4.8e-7).
           Effective tolerance is τM = rtolM * σ_max, which scales with Jacobian magnitude.
           The default value is calibrated for stiff ODE systems and provides ~2× safety margin.
           For smooth problems, can tighten to √eps; for very stiff systems, may need up to 1e-6.
           Heuristic: rtolM ≳ 1.5 * max(MS_invariant) / σ_max where MS are singular values of M_test.
```

###examples/repressilator.jl

**Before:** Explicit `rtolM=1e-4` override (too loose, risked false positives)

**After:** Uses default (no override needed)
```julia
S, N, N_perp, rank_J = find_invariant_subspace(
    ϕ_log, θ_log_MLE
    # Uses default rtolM = 32√eps ≈ 4.8e-7, calibrated for stiff ODEs
)
```

### Tests

**test_repressilator_rtolM.jl:** Added assertions
```julia
@assert size(N, 2) == 3 "Sanity check: Should find exactly 3 invariant directions"
@assert rank_J == 15 "Sanity check: Jacobian rank should be 15"
```

**sweep_rtolM.jl:** New diagnostic tool
- Sweeps rtolM values from √eps to 1e-4
- Reports classification results
- Provides recommendations

## Usage Guidelines

### Default behavior (most cases)
```julia
S, N, N_perp, rank = find_invariant_subspace(ϕ_log, θ_log_MLE)
# Uses rtolM = 32√eps ≈ 4.8e-7
```

### Smooth problems (optional tightening)
```julia
S, N, N_perp, rank = find_invariant_subspace(ϕ_log, θ_log_MLE; rtolM=sqrt(eps()))
# rtolM ≈ 1.5e-8 for very smooth problems
```

### Very stiff systems (if default fails)
```julia
S, N, N_perp, rank = find_invariant_subspace(ϕ_log, θ_log_MLE; rtolM=1e-6, verbose=true)
# Check MS values to diagnose
```

### Diagnostic mode
```julia
S, N, N_perp, rank = find_invariant_subspace(ϕ_log, θ_log_MLE; verbose=true)
# Prints MS values and classification
```

Example output:
```
Hessian-based invariance test diagnostics:
  τM (threshold): 0.000286
  M_test singular values (should be ~0 for invariant):
    MS[1] = 0.000132 (0.46×τM) ✓ invariant
    MS[2] = 1.757e-5 (0.06×τM) ✓ invariant
    MS[3] = 6.58e-6 (0.02×τM) ✓ invariant
  Classification: 0 non-invariant, 3 invariant
```

## Benefits

1. **No manual tuning:** Default works for stiff ODEs out of the box
2. **Safe margin:** 2× headroom prevents false negatives
3. **Not too loose:** Still <1e-6, avoiding false positives
4. **Diagnostic tools:** Verbose mode and sweep script help troubleshoot edge cases
5. **Backward compatible:** Can still specify `atolM` or custom `rtolM` if needed

## Validation

**Repressilator (18 parameters, rank 15):**
- ✓ Correctly identifies 3 invariant null vectors (βK products)
- ✓ Default tolerance works without manual override
- ✓ MS values: [1.32e-4, 1.76e-5, 6.58e-6] all < τM ≈ 2.9e-4
- ✓ Assertions pass in test suite

**stat_model (2 parameters, rank 1):**
- ✓ Correctly identifies 1 invariant null vector
- ✓ Works with both default and tightened tolerances
- ✓ Smooth problem benefits from flexibility

## Files Modified

1. `invariance.jl`: Updated default rtolM and documentation
2. `examples/repressilator.jl`: Removed explicit rtolM override
3. `test_repressilator_rtolM.jl`: Added assertions, updated for new default
4. `minimal_invariance_test.jl`: Removed finite-difference, uses Hessian+rtolM
5. `sweep_rtolM.jl`: New diagnostic tool for tolerance calibration
6. `INVARIANCE_TOLERANCE_CALIBRATION.md`: This document

## References

- Original issue: Default rtolM = √eps too tight for stiff ODEs
- User review: Identified minimum rtolM ≈ 3e-7 for repressilator
- Recommendation: Use 32√eps ≈ 4.8e-7 for good balance
- Formula: rtolM ≳ 1.5 * max(MS_invariant) / σ_max
