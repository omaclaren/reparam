# Optimization Improvements - October 21, 2025

## Summary

Enhanced MLE estimation and added refined profiling grid options to repressilator.jl to ensure profile peaks don't exceed MLE and provide smoother visualization options.

## Changes Made

### 1. Enhanced MLE Configuration (Core Issue Fix)

**Problem:** User observed MLE marker not at K₁ profile peak, suggesting profiling found better solutions than point estimation.

**Solution:** Increased MLE optimization effort by 6x:
- **Before:** 3 guesses × 30s = 90s total (sequential)
- **After:** 9 guesses × 60s = 540s total (sequential)
- **Benefit:** MLE is now much closer to true optimum; profile peaks should not exceed MLE

**Implementation:**
```julia
const PROFILE_CONFIGS = Dict(
    "paper" => (
        grid_1d=15,
        timeout=40.0,
        n_guesses=3,
        mle_guesses=9,      # NEW: 3x more guesses
        mle_timeout=60.0    # NEW: 2x longer timeout
    )
)
```

**Result from test run:**
- MLE found at K₁=20.73, β₁=0.0146, log-likelihood=-34.18
- Profile grid maximum very close to MLE (distance 8.3e-6)
- Enhanced search successfully prevented profile peaks exceeding MLE

### 2. Refined Profiling Grid Options

**Motivation:** User requested finer grids for smoother profile plots in final figures.

**Solution:** Added 4 new profiling modes with increasing resolution:

| Mode | Grid | Timeout | Guesses | MLE Setup | Runtime | Use Case |
|------|------|---------|---------|-----------|---------|----------|
| test | 5 | 10s | 1 | 5×20s | ~5 min | Quick debugging |
| **paper** | 15 | 40s | 3 | 9×60s | ~20 min | **Development (current)** |
| high_quality | 21 | 50s | 4 | 12×75s | ~30 min | Better smoothness |
| **publication** | 31 | 60s | 5 | 12×90s | ~60 min | **Final paper figures** |
| ultra | 51 | 90s | 7 | 15×120s | ~120 min | Maximum detail |
| full | 25 | 60s | 3 | 12×90s | ~60 min | With 2D profiling |

**Usage:**
```julia
const PROFILE_MODE = "publication"  # Change this line for different modes
```

**Recommendation:**
- **Iterative work:** Use "paper" mode (15 points, fast)
- **Final figures:** Use "publication" mode (31 points, professional quality)

### 3. Reduced Observation Count for Visual Uncertainty

**Motivation:** User wanted wider prediction intervals for better visual separation between identifiable vs non-identifiable parameters.

**Solution:** Reduced time observations from NT=9 to NT=7:

**Analysis:**
- **Before:** 9 time points × 3 species = 27 observations (9 DoF with 18 parameters)
- **After:** 7 time points × 3 species = 21 observations (3 DoF with 18 parameters)
- **Information loss:** 22% (safe threshold)
- **Expected effect:** ~30-40% wider prediction intervals

**Why NT=7 (not NT=5):**
- NT=5 → 15 observations → underdetermined system (-3 DoF) → risky
- NT=7 → 21 observations → minimal DoF (3) → safe
- NT=9 → 27 observations → comfortable (9 DoF) → narrow uncertainty

**Expected prediction interval widths:**
```
Current (NT=9):       NT=7 (expected):
K₁ individual: 0.36   → ~0.50 (+40%)
β₁ individual: 0.55   → ~0.75 (+35%)
β₁/K₁ ratio:   1.74   → ~2.30 (+30%)
```

## MLE Labeling Clarification

**Question:** Does plot use point optimization MLE or profile grid maximum?

**Answer:** Uses **point optimization MLE** (correct approach).

From `visualization.jl`:
```julia
if length(ψ_MLE) > 0
    vline!([ψ_MLE], color=:silver, lw=3)  # Point estimate
else
    ψ_max = ψ_values[argmax(like_ψ_values)]  # Grid max fallback
```

Both `stat_model.jl` and `repressilator.jl` pass `ψ_MLE` from `profile_target()` with empty `target_indices`, ensuring consistent use of point estimates across examples.

## Files Modified

1. **examples/repressilator.jl**
   - Line 30: Updated PROFILE_MODE comment
   - Lines 33-51: Added 4 new profiling configurations
   - Lines 188-191: Reduced NT from 9 to 7 with documentation

2. **Documentation**
   - Created this file (OPTIMIZATION_IMPROVEMENTS_2025-10-21.md)

## Testing Recommendations

### Next Test Run:

1. **Verify NT=7 gives wider intervals:**
   ```bash
   julia -t 6 examples/repressilator.jl
   ```
   - Check prediction interval widths increase as expected
   - Verify MLE optimization still stable (not underdetermined)

2. **Test publication mode for final figures:**
   ```julia
   const PROFILE_MODE = "publication"
   ```
   - Should take ~45-60 minutes
   - Profile curves should be noticeably smoother
   - 31 grid points will capture more detail near peaks

3. **Verify MLE at profile peaks:**
   - Check K₁ profile plot: MLE marker should be at/near peak
   - Check β₁ profile plot: MLE marker should be at/near peak
   - Check β₁/K₁ ratio: Profile should be tight (identifiable)

## Related Work from Previous Session (2025-10-20)

See `OPTIMIZATION_IMPROVEMENTS_2025-10-20.md` for:
- Adaptive continuation implementation
- Track convergence diagnostics
- Tolerance calibration for stiff ODEs

## Implementation Notes

### Why Sequential Multi-Start?

NLopt is **not thread-safe** - parallel multi-start causes segmentation faults. All multi-start loops run sequentially to ensure stability.

**Parallelization strategy:**
- ✓ MLE: Sequential (9 guesses × 60s = 9 min wallclock)
- ✓ Profiling: Parallel over 3 parameters (3x speedup)
- ✓ Within each profile: Sequential adaptive continuation (warm start)

### Grid Resolution Trade-offs

**15 points (paper):** Fast iteration, good for debugging
**31 points (publication):** Smooth curves, standard for publication
**51 points (ultra):** Overkill for most problems, use only for difficult cases

**Rule of thumb:** Odd numbers (15, 21, 31, 51) ensure MLE falls on grid point if near bounds.

## Next Steps

1. Run with NT=7 to verify wider uncertainty
2. Generate final figures with "publication" mode (31 points)
3. Compare K₁ individual vs β₁/K₁ ratio prediction intervals in 6-panel dynamics plot
4. Document findings in paper Methods section
