# Question: Parallelizing 2D Joint Profile Likelihood for β₁-K₁

## Context

We have a repressilator model with 18 parameters. We want to compute a 2D joint profile likelihood for (β₁, K₁) to visualize their non-identifiability (ridge along β₁K₁ = constant line).

## Current Implementation

### Existing Code Structure

The `profile_target()` function in `core.jl` has this nested loop structure:

```julia
# Line 255 - OUTER LOOP over grid points
for (i, ψᵢ) in enumerate(ψ_combinations)
    # For 2D: ψᵢ = (β₁_val, K₁_val) from 7×7 grid = 49 points

    # Line 274-285 - INNER LOOP: Multi-start optimization
    for ω₀ in starting_points  # e.g., 3 initial guesses
        opt.max_objective = construct_lnlike_to_max(...)
        (lnlike_opt, ωᵢ_opt, return_code) = optimize(opt, ω₀)  # NLopt call
        # Keep best result
    end

    # Line 296: Adaptive continuation
    ω_initial = best_ω  # Use best result as initial guess for next grid point
end
```

### Current Constraint: NLopt is NOT Thread-Safe

From line 274 comment: "Try multiple starting points sequentially (NLopt is not thread-safe)"

We already discovered through trial and error that:
- Calling NLopt `optimize()` from multiple threads simultaneously causes segfaults
- This is why the INNER loop (multi-start) must be sequential

### Current 1D Profiling Parallelization

In `repressilator.jl` (lines ~1430-1480), we successfully parallelize THREE separate 1D profiles:

```julia
Threads.@threads for i in 1:3
    if i == 1
        # Profile K₁ (calls profile_target for 15 grid points)
    elseif i == 2
        # Profile β₁ (calls profile_target for 15 grid points)
    elseif i == 3
        # Profile ratio (calls profile_target for 15 grid points)
    end
end
```

**This works because:**
- Each thread calls `profile_target()` independently
- Within each call, NLopt is used sequentially (no concurrent NLopt calls)
- The three profiles don't share any state

## The Question

### Computational Cost for 2D Profiling

For a 7×7 grid of (β₁, K₁):
- 49 grid points
- 3 initial guesses per point (multi-start optimization)
- ~40 seconds per optimization
- **Sequential time**: 49 × 3 × 40s = **~98 minutes**
- **Parallel time** (6 threads): 49/6 × 3 × 40s = **~16 minutes**

### Proposed Parallelization Approach

Instead of modifying `profile_target()`, manually parallelize in `repressilator.jl`:

```julia
# Generate 2D grid
β1_grid = LinRange(θ_log_lower[β1_index], θ_log_upper[β1_index], 7)
K1_grid = LinRange(θ_log_lower[K1_index], θ_log_upper[K1_index], 7)
grid_2d = [(β, K) for β in β1_grid for K in K1_grid]  # 49 points

results_2d = Vector{Any}(undef, 49)

# OUTER LOOP: Parallelize over grid points
Threads.@threads for i in 1:49
    β1_val, K1_val = grid_2d[i]

    # Call profile_target for THIS SINGLE POINT
    # (internally, profile_target will run multi-start sequentially)
    θ_vals, ll_vals = profile_target(
        lnlike_θ_log,
        [β1_index, K1_index],
        θ_log_lower,
        θ_log_upper,
        nuisance_guess;
        grid_steps=[1, 1],  # Single point evaluation
        ω_initial_extras=extra_guesses,
        method=:LN_BOBYQA,
        optmaxtime=40.0)

    results_2d[i] = (β1_val, K1_val, ll_vals[1])
end
```

### Key Questions

1. **Thread Safety**: Is this safe given NLopt constraints?
   - Each thread calls `profile_target()` independently
   - Within each `profile_target()` call, NLopt is used sequentially
   - No two threads call NLopt simultaneously
   - **Should this work?**

2. **NLopt Optimizer State**: Does NLopt create per-call state?
   - In `profile_target()`, `opt = Opt(method, dim_ω)` is created fresh each call
   - Each thread would create its own `opt` object
   - **Is this safe, or does NLopt have global state?**

3. **Adaptive Continuation Trade-off**:
   - Sequential: Line 296 uses `ω_initial = best_ω` (use previous solution)
   - Parallel: Each grid point starts fresh (no cross-thread communication)
   - **Is losing adaptive continuation a major performance hit?**
   - Note: Current 1D parallel implementation (3 profiles) also doesn't use adaptive continuation across profiles

4. **Alternative Approach**: Modify `core.jl` to add threading?
   - Could add `parallelize=true` parameter to `profile_target()`
   - Use `Threads.@threads for (i, ψᵢ) in enumerate(ψ_combinations)` on line 255
   - Handle thread-safety of pre-allocated arrays and adaptive continuation
   - **Worth the complexity vs. localized solution in repressilator.jl?**

## Summary

We want to parallelize 2D profiling (49 grid points → 16 min instead of 98 min). The proposed approach:
- Manually loop over 2D grid with `Threads.@threads` in `repressilator.jl`
- Call `profile_target()` for each single grid point independently
- Relies on NLopt being safe when called from separate threads (no concurrent calls within same thread)

**Is this thread-safe given NLopt's constraints? Are there hidden gotchas we're missing?**
