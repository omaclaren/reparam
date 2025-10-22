# MLE Estimation Enhancement - Session Summary
**Date:** 2025-10-21
**Status:** ✅ IMPLEMENTED (needs testing)

## Problem Identified

**Observation**: MLE on K₁ profile plot is not at the profile maximum.

**Root cause analysis**:
- MLE estimation used 3 guesses × 30s timeout = limited search effort
- Profiling used 3 guesses × 40s timeout per grid point
- Profiling can potentially find better solutions than initial MLE!
- This violates fundamental property: profile peaks ≤ MLE

## Solution: Invest More in MLE Estimation

### Key Insight
**MLE estimation can be parallelized** but profiling cannot:
- MLE: Not in threaded region yet → `profile_target()` parallelizes over initial guesses
- Profiling: Already uses `Threads.@threads` for 3 parameters → nested threading avoided

**Strategy**: Use MORE guesses with LONGER timeout for MLE, leveraging parallelization

### Implementation

**Configuration Changes** ([repressilator.jl:33-37](examples/repressilator.jl#L33-L37)):
```julia
const PROFILE_CONFIGS = Dict(
    "test"  => (..., mle_guesses=5,  mle_timeout=20.0),
    "paper" => (..., mle_guesses=9,  mle_timeout=60.0),
    "full"  => (..., mle_guesses=12, mle_timeout=90.0)
)
```

**Usage** ([repressilator.jl:394-415](examples/repressilator.jl#L394-L415)):
```julia
n_guesses_mle = CONFIG.mle_guesses  # 9 for "paper" mode
θ_log_MLE, lnlike_MLE = profile_target(
    lnlike_θ_log, target_indices,
    θ_log_lower, θ_log_upper,
    θ_log_initial;
    grid_steps=grid_steps_mle,
    ω_initial_extras=nuisance_guesses_mle,
    method=:LN_BOBYQA,
    optmaxtime=CONFIG.mle_timeout)  # 60s for "paper" mode
```

## Comparison: Before vs After

### "paper" Mode Settings

| Aspect | Before | After | Notes |
|--------|--------|-------|-------|
| **MLE guesses** | 3 | 9 | **+200%** search diversity |
| **MLE timeout** | 30s | 60s | **+100%** optimization time |
| **MLE wallclock** | ~30s | ~60s | Parallelized over 6 threads |
| **Profile guesses** | 3 | 3 | (unchanged) |
| **Profile timeout** | 40s | 40s | (unchanged) |
| **Total MLE effort** | 90s | 540s | **6x more optimization** |
| **Actual wallclock** | 30s | 60s | **2x** (due to parallelization) |

### Expected Outcomes

1. **MLE should be at profile peak** ✓
   - More initial guesses explore parameter space better
   - Longer timeout allows convergence to true optimum

2. **Profile peaks ≤ MLE** (guaranteed by definition)
   - If profiling finds better solution → indicates MLE search was insufficient
   - Our enhanced MLE search should prevent this

3. **Modest wallclock time increase** (30s → 60s)
   - Acceptable trade-off for proper MLE estimation
   - Essential for valid uncertainty quantification

## Parallelization Strategy

### MLE Estimation (Point Estimation)
```julia
if Threads.in_threaded_region()
    # Sequential fallback
else
    # PARALLEL: each guess runs in separate thread ✓
    Threads.@threads for j in 1:length(starting_points)
        # Independent optimization
    end
end
```
- **Status**: NOT in threaded region
- **Result**: PARALLELIZES over 9 guesses
- **Benefit**: 6x more effort for ~2x wallclock time

### Profiling (Conditional Optimization)
```julia
Threads.@threads for i in 1:3  # K₁, β₁, β₁/K₁ ratio
    profile_target(...)  # Sequential within each profile
end
```
- **Status**: IN threaded region (outer loop)
- **Result**: Sequential guess evaluation (adaptive continuation)
- **Benefit**: Warm-start improves convergence

## Code Changes

### Files Modified
1. **examples/repressilator.jl** (~30 lines)
   - Added `mle_guesses` and `mle_timeout` to CONFIG
   - Updated MLE optimization call to use CONFIG values
   - Enhanced console output to explain parallelization
   - Improved documentation of parallelization strategy

### No Changes to Core Library
- [core.jl](core.jl) already supports parallelization via `Threads.in_threaded_region()`
- Implementation from [OPTIMIZATION_IMPROVEMENTS_2025-10-20.md](OPTIMIZATION_IMPROVEMENTS_2025-10-20.md)
- No modifications needed!

## Testing Plan

### Quick Verification (5 minutes)
```bash
cd /Users/omac010/Git-Working/reparam
julia -t 6 -e 'include("examples/repressilator.jl")' 2>&1 | tee mle_test.log &
# Let MLE complete (~60s), then Ctrl+C to check MLE quality
```

**Check**:
- [ ] MLE completes in ~60s (not 9×60s)
- [ ] Console shows "parallelized over 6 threads"
- [ ] Log shows different guesses converging to similar MLE

### Full Test (35 minutes)
```bash
julia -t 6 examples/repressilator.jl 2>&1 | tee repressilator_enhanced_mle.log
```

**Verify**:
- [ ] MLE at peak of K₁ profile
- [ ] MLE at peak of β₁ profile
- [ ] MLE at peak of β₁/K₁ ratio profile
- [ ] No profile points exceed MLE likelihood

## Design Rationale

### Why More Guesses for MLE?

**Problem**: 18-dimensional parameter space is enormous
- 3 guesses barely scratch the surface
- Profiling found better solutions → MLE search was insufficient

**Solution**: 9 guesses (3²) provides much better coverage
- Still parallelizes efficiently (9 guesses / 6 threads ≈ 2 batches)
- Modest wallclock increase (30s → 60s)
- Essential for establishing proper reference point

### Why Longer Timeout for MLE?

**Problem**: 30s timeout may cut off convergence
- Some optimizations were still improving when hit timeout
- MLE is critical - deserves more time

**Solution**: 60s timeout matches profiling
- Consistent with profile optimization effort
- Allows convergence to true optimum
- Parallelize over guesses amortizes cost

### Why Not More Profile Guesses?

**Current**: 3 guesses with adaptive continuation
- Sequential warm-start improves convergence
- Previous solution → perturbations around it

**Alternative**: 9 guesses from MLE (no warm-start)
- Would allow parallelization within each profile
- But: slower convergence without warm-start
- Net: probably worse (see OPTIMIZATION_IMPROVEMENTS doc)

**Decision**: Keep profile at 3 guesses, improve MLE instead

## Expected Impact

### Numerical Quality
- ✅ MLE should be at profile peaks (proper reference point)
- ✅ Valid confidence intervals (profile peaks ≤ MLE)
- ✅ Better exploration of parameter space

### Runtime
- **Before**: MLE 30s, Profiling 32min = **32.5 min total**
- **After**: MLE 60s, Profiling 32min = **33 min total**
- **Increase**: +30s (+1.5%)

### Scientific Validity
- **Critical**: MLE must be best we can find
- **Previous**: Profiling sometimes found better solutions (red flag!)
- **Enhanced**: More rigorous MLE search ensures proper reference

## Known Limitations

### Still May Not Find Global Optimum
- 18-dimensional space is genuinely difficult
- Local optimization methods (BOBYQA) can get stuck
- 9 guesses helps but doesn't guarantee global optimum

**Mitigation**:
- Could use global optimization for MLE (G_MLSL)
- Trade-off: much slower (10-20x)
- Current approach: balance between thoroughness and practicality

### Profile May Still Timeout
- Profile optimizations still challenging
- 40s timeout with 3 guesses
- Some grid points still hit timeout

**Note**: This is profiling problem, not MLE problem
- If profile finds better solution despite timeout → fixed by enhanced MLE
- If profile can't converge → separate issue (addressed in OPTIMIZATION_IMPROVEMENTS)

## References

**Related improvements**:
- [OPTIMIZATION_IMPROVEMENTS_2025-10-20.md](OPTIMIZATION_IMPROVEMENTS_2025-10-20.md) - Adaptive continuation, threading
- [core.jl:209-240](core.jl#L209-L240) - Smart threading detection for MLE

**Core insight**: MLE is reference point - must be best we can find!

## Next Steps

### Immediate (testing)
- [ ] Run full test with enhanced MLE settings
- [ ] Verify MLE at profile peaks in all plots
- [ ] Check wallclock time is acceptable (~60s for MLE)

### Optional (if issues persist)
- [ ] Try global optimization for MLE (G_MLSL_LDS)
- [ ] Increase MLE guesses to 12 or 15
- [ ] Consider Latin hypercube sampling for initial guesses

### Documentation
- [ ] Update OPTIMIZATION_IMPROVEMENTS to reference this enhancement
- [ ] Note in paper Methods that MLE uses multiple initial guesses
- [ ] Explain why profile peaks might not exactly equal MLE (convergence tolerances)

---

## Commit Message (Suggested)

```
feat: enhance MLE estimation with more guesses and longer timeout

Major improvement to MLE estimation quality:

**Problem:**
- Profiling sometimes found better solutions than initial MLE
- This violates fundamental property: profile peaks ≤ MLE
- Indicates insufficient search effort during MLE estimation

**Solution:**
- Increase MLE guesses: 3 → 9 (paper mode)
- Increase MLE timeout: 30s → 60s
- Leverage parallelization over initial guesses (6 threads)

**Key Insight:**
- MLE estimation CAN be parallelized (not in threaded region)
- Profiling CANNOT (already uses Threads.@threads for parameters)
- Therefore: invest more computational effort in MLE

**Results:**
- 6x more optimization effort (90s → 540s total)
- Only 2x wallclock increase (30s → 60s due to parallelization)
- MLE should now be at profile peaks (proper reference point)

**Rationale:**
MLE is the reference point for all uncertainty quantification.
If profiling finds better solutions, we didn't try hard enough!
Enhanced MLE search ensures valid confidence intervals.

Files modified: examples/repressilator.jl
```

---

## End of Session Summary

**Status:** Implementation complete, needs testing ✅
**Runtime impact:** +30s (+1.5%) total runtime
**Scientific benefit:** Essential for valid uncertainty quantification
**Next session:** Run full test, verify MLE at profile peaks
