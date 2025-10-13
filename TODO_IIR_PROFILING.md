# TODO: Complete IIR-Based Profiling for Repressilator

## Current Status

The repressilator example currently has a fundamental flaw: it performs IIR analysis but then **does not use the IIR reparameterization for profiling**. Instead, it profiles K₁ and β₁ as individual parameters in the original θ coordinates.

## What Needs to Be Done

### 1. Replace Profiling Section (lines 981-1080)

**Current approach (WRONG):**
- Profile K₁ individually in θ space
- Profile β₁ individually in θ space
- Extract K₁/β₁ ratio from 2D joint profile of (β₁, K₁)

**Correct approach:**
- After IIR, K₁/β₁ ratio IS a single parameter ψ[k] in the reparameterized model
- Profile ψ[k] directly as a 1D parameter (not 2D in θ space)
- This demonstrates the actual value of IIR

### 2. Implementation Steps

The IIR transformation setup is already in place (commit 4c3112f):
- ✅ Forward/inverse transformations created: `θ_to_ψ`, `ψ_to_θ`
- ✅ Likelihood in ψ space: `lnlike_ψ`, `lnlike_ψ_log`
- ✅ Distribution in ψ space: `distrib_fine_ψ`, `distrib_fine_ψ_log`
- ✅ K₁/β₁ ratio index identified: `ψ_K1_β1_index`

**Still TODO:**

a) **Set bounds for ψ space profiling:**
```julia
# Need to determine appropriate bounds for ψ parameters
# Can transform bounds from θ space or set manually based on expected ranges
ψ_lower_bounds = ... # 18-dimensional
ψ_upper_bounds = ... # 18-dimensional
```

b) **Profile K₁/β₁ ratio as single 1D parameter:**
```julia
# Profile the ratio parameter ψ[ψ_K1_β1_index] in ψ space
ψ_log_MLE = log.(ψ_MLE)
nuisance_indices_ratio = setdiff(1:18, ψ_K1_β1_index)
nuisance_guess_ratio = ψ_log_MLE[nuisance_indices_ratio]

ψ_ratio_values, lnlike_ratio_values = profile_target(
    lnlike_ψ_log, ψ_K1_β1_index,
    log.(ψ_lower_bounds), log.(ψ_upper_bounds),
    nuisance_guess_ratio;
    grid_steps=[CONFIG.grid_1d],
    method=:LN_BOBYQA,
    optmaxtime=CONFIG.timeout)
```

c) **Compute prediction intervals:**
```julia
lower_ratio, upper_ratio, _ = construct_upper_lower_profile_wise_CIs_for_mean(
    distrib_fine_ψ_log, ψ_ratio_values, lnlike_ratio_values;
    l_level=95, df=18)
```

d) **Optional: Also profile K₁ and β₁ individually in θ space for comparison**
This would show that the ratio has tighter confidence intervals than the individual parameters (demonstrating identifiability).

### 3. Expected Outcome

After this refactor:
- K₁/β₁ ratio will be profiled as a **single scalar parameter** in ψ space
- Plot labels will correctly say "K₁/β₁ ratio" (not "K₁/β₁ ratio (joint)")
- The example will properly demonstrate IIR reparameterization
- MLE will be guaranteed to be in the profile (consistent optimization method)

### 4. Files to Modify

- `examples/repressilator.jl` (lines 981-1080): Replace entire profiling section

### 5. Reference Implementation

See `examples/transport_model.jl` for correct pattern:
- Lines 604-610: Create transformations with `reparam()`
- Lines 607-610: Create likelihood/distributions in IIR coordinates
- Line 671: Profile in IIR coordinates using `lnlike_XY_iir`

## Why This Matters

The current implementation **completely defeats the purpose of IIR**:
1. IIR identifies that K₁/β₁ is an identifiable parameter combination
2. IIR provides a transformation where this ratio becomes a first-class parameter
3. **But then we ignore this and profile in the original coordinates anyway!**

This is like discovering a shortcut and then not taking it. The whole point of IIR is to make identifiable combinations into simple parameters that can be profiled directly.
