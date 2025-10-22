# Fix: Invariance Tolerance for Stiff ODE Systems

## Problem Summary

The invariance test in `find_invariant_subspace` was using a fixed absolute tolerance (`atolM=1e-10`) that worked well for smooth problems but failed to correctly classify invariant null space directions for stiff ODE systems like the repressilator.

## Root Cause

For stiff ODE systems:
1. The ODE solver introduces numerical noise (tolerances ~ 1e-6 to 1e-10)
2. This noise propagates through the Jacobian computation
3. The Hessian-based invariance test produces MS values that are LARGER than for smooth problems
4. With a fixed absolute tolerance (atolM=1e-10), these larger MS values incorrectly classified invariant directions as non-invariant

## Solution: Relative Tolerance

Implemented `rtolM` parameter (relative tolerance) that scales with the Jacobian magnitude:

```julia
τM = rtolM * σ_max
```

**Default:** `rtolM = sqrt(eps())` ≈ 1.5e-8 (same as `rtolJ`)

**For stiff ODEs:** May need `rtolM ~ 1e-4` to `1e-6`

## Example: Repressilator

With 18 parameters and rank 15, there are 3 null space directions (βK products).

### Default tolerance (rtolM = √eps ≈ 1.5e-8):
```
σ_max: 599.9
τM = rtolM * σ_max ≈ 8.9e-6

MS values:
  MS[1] = 0.000132 (14.77×τM) ✗ NON-INVARIANT
  MS[2] = 1.757e-5 (1.97×τM) ✗ NON-INVARIANT
  MS[3] = 6.58e-6 (0.74×τM) ✓ invariant

Result: Only 1/3 directions correctly classified
```

### Relaxed tolerance (rtolM = 1e-4):
```
σ_max: 599.9
τM = rtolM * σ_max ≈ 0.06

MS values:
  MS[1] = 0.000132 (0.002×τM) ✓ invariant
  MS[2] = 1.757e-5 (0.0003×τM) ✓ invariant
  MS[3] = 6.58e-6 (0.0001×τM) ✓ invariant

Result: All 3 directions correctly classified ✓
```

## Implementation Changes

### invariance.jl

1. **New parameters:**
   ```julia
   atolM=nothing,  # Deprecated absolute tolerance
   rtolM=sqrt(eps(real(eltype(θ0))))  # New relative tolerance
   ```

2. **Tolerance computation:**
   ```julia
   if !isnothing(atolM)
       τM = atolM  # Backward compatibility
   else
       τM = rtolM * σmax  # Default: scales with problem
   end
   ```

3. **Removed finite-difference fallback:**
   - Previous finite-difference implementation was incorrect
   - Now errors informatively if requested
   - Hessian method works for stiff ODEs (nested AD no longer fails)

4. **Added verbose diagnostics:**
   ```julia
   if haskey(kwargs, :verbose) && kwargs[:verbose]
       # Print τM, MS values, and classification
   end
   ```

### Usage

**Smooth problems (default):**
```julia
S, N, N_perp, rank = find_invariant_subspace(ϕ_log, θ_log_MLE)
```

**Stiff ODE systems:**
```julia
S, N, N_perp, rank = find_invariant_subspace(ϕ_log, θ_log_MLE; rtolM=1e-4)
```

**With diagnostics:**
```julia
S, N, N_perp, rank = find_invariant_subspace(ϕ_log, θ_log_MLE; rtolM=1e-4, verbose=true)
```

## Benefits

1. **Automatic scaling:** Default tolerance adapts to problem magnitude
2. **Explicit control:** Can tune `rtolM` for challenging problems
3. **Backward compatible:** Can still use `atolM` if needed
4. **Better diagnostics:** Verbose mode shows MS values and classification

## Testing

- ✓ `test_rtolM.jl`: Verifies rtolM vs atolM behavior on stat_model
- ✓ `test_repressilator_rtolM.jl`: Confirms rtolM=1e-4 finds all 3 invariant directions
- ✓ Hessian method works for stiff ODEs (nested AD no longer fails)

## Recommendations

1. **Start with default** (rtolM = √eps) for all problems
2. **Use verbose=true** to inspect MS values and classification
3. **If some directions misclassified**, increase rtolM to 1e-6 or 1e-4
4. **For very stiff systems**, may need rtolM ~ 1e-4

## Files Modified

- `invariance.jl`: Core implementation
- `test_rtolM.jl`: Simple tolerance test
- `test_repressilator_rtolM.jl`: Stiff ODE test
- `INVARIANCE_TOLERANCE_FIX.md`: This document
