# Refactoring Plan: 2D Distributed Profiling Infrastructure

## Executive Summary

We need to compute 2D joint profile likelihood for (β₁, K₁) at publication quality (21×21 or 31×31 grid). Sequential execution would take 15-32 hours, which is impractical. The current `profile_target()` function is monolithic and cannot be parallelized safely due to NLopt thread-safety constraints.

**Solution:** Refactor `profile_target()` into layered, reusable components that enable distributed execution via process-based parallelism (`pmap`).

## Current State

### What Works
- **NT=8 repressilator analysis complete** (repressilator_NT8.log, Oct 21)
- 1D profiling for K₁, β₁, and ratio working correctly
- Correct ratio labeling (β₁/K₁ vs K₁/β₁) after scope bug fix
- Simple plotting script at `examples/plot_repressilator_profiles.jl`

### The Problem
- Need 2D joint profile of (β₁, K₁) to show non-identifiability ridge
- Minimum acceptable resolution: 15×15 = 225 points = 7.5 hours sequential
- Publication quality: 21×21 = 441 points = 14.7 hours sequential
- High quality: 31×31 = 961 points = 32 hours sequential

### Why Threading Doesn't Work
NLopt is **not thread-safe**:
- Has global state in C/Fortran code
- Static workspaces shared across calls
- NLopt.jl reuses global callback tables
- Using `Threads.@threads` causes segfaults even with separate `Opt()` objects per thread

### Why Distributed Works
- Each worker process has isolated address space
- No shared NLopt global state
- Can use `pmap` or `@distributed` safely
- 4 workers → 4× speedup (21×21 in ~4 hours)

## Proposed Architecture

### Current Monolithic Structure

```julia
function profile_target(...)
    # 1. Build Cartesian grid from grid_steps
    # 2. Loop over all grid points sequentially
    #    - For each point:
    #      - Multi-start optimization (sequential, NLopt calls)
    #      - Adaptive continuation (ω_initial = best_ω)
    # 3. Normalize results
    # 4. Return
end
```

**Problems:**
- Everything tangled together
- Cannot parallelize (would need concurrent NLopt calls)
- Cannot reuse for custom grids or single evaluations
- Hard to test individual pieces

### New Layered Architecture

```
profile_target()                    # High-level API (unchanged signature)
  └─> profile_grid_sequential()     # Sequential grid execution
        └─> profile_point()          # Single point optimization (multi-start)
              └─> NLopt optimize()   # Actual NLopt call

profile_grid_distributed()          # New: parallel execution
  └─> partition grid into chunks
  └─> pmap over chunks
        └─> profile_grid_sequential() # Reuse sequential logic per chunk
              └─> profile_point()      # Same primitive
```

### Layer 1: `profile_point()` - Single Point Optimizer

**Purpose:** Optimize at ONE fixed grid point with multi-start sequential NLopt calls.

**Signature:**
```julia
function profile_point(
    lnlike_θ,
    ψ_fixed::Vector{Float64},      # Fixed interest parameter values
    ψ_indices::Vector{Int},          # Which parameters are fixed
    θ_bounds_lower, θ_bounds_upper,
    ω_initial::Vector{Float64},      # Starting guess for nuisance params
    ω_initial_extras::Union{Nothing, Vector{Vector{Float64}}}=nothing;
    method=:LN_BOBYQA,
    optmaxtime=60.0,
    xtol_rel=1e-9,
    ftol_rel=1e-9,
    track_convergence=false
)
    # Returns: (θ_opt, ω_opt, lnlike_opt, convergence_info)
end
```

**What it does:**
1. Extract nuisance indices and bounds
2. Create NLopt optimizer
3. Prepare starting points: `[ω_initial, ω_initial_extras...]`
4. Loop sequentially through starting points (NLopt not thread-safe)
5. Return best result

**Key:** This is where NLopt is called. Must be sequential.

### Layer 2: `profile_grid_sequential()` - Sequential Grid Iterator

**Purpose:** Execute profiling over an **ordered** grid with adaptive continuation.

**Signature:**
```julia
function profile_grid_sequential(
    lnlike_θ,
    ψ_grid::Vector{Vector{Float64}},  # Pre-ordered grid points
    ψ_indices::Vector{Int},
    θ_bounds_lower, θ_bounds_upper,
    ω_initial::Vector{Float64};
    ω_initial_extras::Union{Nothing, Vector{Vector{Float64}}}=nothing,
    method=:LN_BOBYQA,
    optmaxtime=60.0,
    xtol_rel=1e-9,
    ftol_rel=1e-9,
    track_convergence=false
)
    # Returns: (θ_values, lnlike_values, convergence_info)
end
```

**What it does:**
1. Pre-allocate result arrays
2. Loop over `ψ_grid` in order:
   ```julia
   for (i, ψᵢ) in enumerate(ψ_grid)
       θ_opt, ω_opt, lnlike_opt, conv = profile_point(
           lnlike_θ, ψᵢ, ψ_indices, ..., ω_initial, ω_initial_extras; ...)

       θ_values[i] = θ_opt
       lnlike_values[i] = lnlike_opt
       convergence_info[i] = conv

       # Adaptive continuation
       ω_initial = ω_opt  # Use result as next starting point
   end
   ```
3. Return arrays (no normalization at this level)

**Key:** Maintains adaptive continuation within a sequence of grid points.

### Layer 3: `profile_target()` - High-Level API (Refactored)

**Purpose:** Maintain existing API, delegate to new layers.

**Signature:** Unchanged (backwards compatibility)

**New implementation:**
```julia
function profile_target(lnlike_θ, ψ_indices, θ_bounds_lower, θ_bounds_upper, ω_initial;
    grid_steps=100, ω_initial_extras=nothing, method=:LN_BOBYQA, ...)

    # 1. Build Cartesian grid (as before)
    ψ_grids = [LinRange(lower[i], upper[i], steps[i]) for i in 1:dim_ψ]
    ψ_combinations = collect(Base.product(ψ_grids...))
    ψ_grid = [collect(ψᵢ) for ψᵢ in ψ_combinations]

    # 2. Delegate to sequential grid runner
    θ_values, lnlike_values, conv_info = profile_grid_sequential(
        lnlike_θ, ψ_grid, ψ_indices, θ_bounds_lower, θ_bounds_upper, ω_initial;
        ω_initial_extras=ω_initial_extras, method=method, ...)

    # 3. Normalize (as before)
    lnlike_values = lnlike_values .- maximum(lnlike_values)

    # 4. Return (same format as before)
    if track_convergence
        return θ_values, lnlike_values, conv_info
    else
        return θ_values, lnlike_values
    end
end
```

**Key:** Existing callers work without changes.

### Layer 4: `profile_grid_distributed()` - Parallel Execution

**Purpose:** Distribute grid evaluation across worker processes.

**Signature:**
```julia
function profile_grid_distributed(
    lnlike_θ,
    ψ_grid::Vector{Vector{Float64}},
    ψ_indices::Vector{Int},
    θ_bounds_lower, θ_bounds_upper,
    ω_initial::Vector{Float64};
    ω_initial_extras::Union{Nothing, Vector{Vector{Float64}}}=nothing,
    method=:LN_BOBYQA,
    optmaxtime=60.0,
    n_chunks::Union{Nothing, Int}=nothing,  # Default: nworkers()
    chunk_strategy=:stripes,  # :stripes, :blocks, :random
    kwargs...
)
    # Returns: (θ_values, lnlike_values, convergence_info)
end
```

**What it does:**
1. Partition grid into chunks (preserving some ordering for continuation)
   ```julia
   # Example for 2D β₁-K₁ with :stripes strategy
   # Worker 1: β₁=val1, K₁ varies (sequential strip)
   # Worker 2: β₁=val2, K₁ varies (sequential strip)
   # etc.
   ```

2. Use `pmap` to distribute chunks:
   ```julia
   results = pmap(chunks) do chunk_grid
       # Each chunk runs sequentially with continuation
       profile_grid_sequential(
           lnlike_θ, chunk_grid, ψ_indices,
           θ_bounds_lower, θ_bounds_upper, ω_initial;
           ω_initial_extras=ω_initial_extras, ...)
   end
   ```

3. Merge results back into full grid order
4. Return (no normalization - caller does that)

**Chunking Strategies:**

- **`:stripes`** (default for 2D): Preserve one dimension sequential
  - For β₁-K₁: Each worker gets fixed β₁, varies K₁ sequentially
  - Good continuation within strips, loses it between strips

- **`:blocks`**: Divide grid into spatial blocks
  - Good for truly independent evaluations
  - Loses most continuation benefits

- **`:random`**: Random assignment
  - Load balancing if some regions harder than others
  - No continuation benefit

**Key considerations:**
- Must ensure serializable data (likelihood functions, bounds)
- No `addprocs`/`rmprocs` inside this function (caller manages workers)
- Each chunk is small enough to fit in memory
- Return results in consistent order with input `ψ_grid`

## Implementation Plan

### Phase 1: Extract Core Primitives (core.jl)

**Time estimate: 2-3 hours**

1. **Create `profile_point()`**
   - Extract lines 258-296 from current `profile_target()` (the inner optimization loop)
   - Add function signature with all necessary parameters
   - Return `(θ_opt, ω_opt, lnlike_opt, convergence_info)`
   - Test in isolation with single point

2. **Create `profile_grid_sequential()`**
   - Wrap the outer loop from current `profile_target()`
   - Accept pre-built `ψ_grid` instead of `grid_steps`
   - Call `profile_point()` for each grid point
   - Maintain adaptive continuation logic
   - Return unnormalized results

3. **Refactor `profile_target()`**
   - Keep signature unchanged
   - Build Cartesian grid (existing logic)
   - Delegate to `profile_grid_sequential()`
   - Normalize results (existing logic)
   - Add comment explaining the delegation

4. **Validation tests**
   - Run stat_model example before and after
   - Compare outputs (should be identical)
   - Check that 1D profiles still work

### Phase 2: Add Distributed Support (core.jl)

**Time estimate: 1-2 hours**

1. **Add `Distributed` to dependencies**
   ```julia
   # At top of core.jl
   using Distributed
   ```

2. **Implement chunking helper**
   ```julia
   function partition_grid_for_continuation(
       ψ_grid::Vector{Vector{Float64}},
       n_chunks::Int;
       strategy=:stripes,
       dim_indices=nothing  # Which dimension to preserve for stripes
   )
       # Returns: Vector{Vector{Vector{Float64}}} - array of chunks
   end
   ```

3. **Implement `profile_grid_distributed()`**
   - Check workers available: `nworkers() > 1` or error
   - Partition grid using chunking helper
   - Use `pmap` to distribute chunks
   - Merge results maintaining order
   - Return in same format as `profile_grid_sequential()`

4. **Serialization checks**
   - Ensure likelihood functions serialize (use closures carefully)
   - Test with 2 workers on small grid (5×5)
   - Compare against sequential baseline

### Phase 3: Update Driver Script (repressilator.jl)

**Time estimate: 30-60 minutes**

1. **Add worker management section**
   ```julia
   if CONFIG.do_2d
       println("\n2D Profiling with distributed execution...")

       # Start workers
       n_workers = 4
       addprocs(n_workers)
       @everywhere using ReparamTools

       try
           # Run 2D profiling
           # ... (see below)
       finally
           # Always clean up workers
           rmprocs(workers())
       end
   end
   ```

2. **Call distributed profiling**
   ```julia
   # Inside try block
   β1_grid = LinRange(θ_log_lower[β1_index], θ_log_upper[β1_index], CONFIG.grid_2d[1])
   K1_grid = LinRange(θ_log_lower[K1_index], θ_log_upper[K1_index], CONFIG.grid_2d[2])
   ψ_grid_2d = [collect(p) for p in Base.product(β1_grid, K1_grid)]

   θ_vals_2d, ll_vals_2d, conv_2d = profile_grid_distributed(
       lnlike_θ_log,
       ψ_grid_2d,
       [β1_index, K1_index],
       θ_log_lower, θ_log_upper,
       nuisance_guess_2d;
       ω_initial_extras=nuisance_extras_2d,
       method=:LN_BOBYQA,
       optmaxtime=CONFIG.timeout,
       n_chunks=n_workers,
       chunk_strategy=:stripes,
       track_convergence=true
   )

   # Normalize
   ll_vals_2d = ll_vals_2d .- maximum(ll_vals_2d)

   # Save and plot results
   # ... (similar to existing 1D plotting)
   ```

3. **Add 2D contour plotting**
   ```julia
   # Reshape for contour plot
   β1_vals = unique([ψ[1] for ψ in ψ_grid_2d])
   K1_vals = unique([ψ[2] for ψ in ψ_grid_2d])
   ll_matrix = reshape(ll_vals_2d, length(β1_vals), length(K1_vals))

   # Create contour plot
   contourf(exp.(β1_vals), exp.(K1_vals), ll_matrix',
            xlabel="β₁", ylabel="K₁",
            title="Joint Profile Likelihood (β₁, K₁)",
            levels=20, colorbar=true)

   # Add ridge line: β₁K₁ = constant
   K1_theory = β1K1_mle ./ exp.(β1_vals)
   plot!(exp.(β1_vals), K1_theory,
         linewidth=3, color=:red, label="β₁K₁ = $(β1K1_mle)")
   ```

### Phase 4: Testing and Validation

**Time estimate: 1-2 hours**

1. **Sequential validation (5×5 grid)**
   ```julia
   # Run with do_2d=false (sequential)
   # Run with do_2d=true, n_workers=1 (sequential via distributed)
   # Compare outputs - should be identical
   ```

2. **Parallel smoke test (5×5 grid, 2 workers)**
   ```julia
   # Check no errors
   # Check deterministic results
   # Visual inspection of contour plot
   ```

3. **Production test (15×15 grid, 4 workers)**
   ```julia
   # Time: ~2 hours
   # Generate publication-quality figure
   # Check contour shows expected ridge
   ```

4. **Final production run (21×21 or 31×31)**
   ```julia
   # 21×21: ~4 hours with 4 workers
   # 31×31: ~8 hours with 4 workers
   # Smooth contours, clear ridge
   ```

## Configuration Updates

Add to `PROFILE_CONFIGS` in repressilator.jl:

```julia
const PROFILE_CONFIGS = Dict(
    # ... existing configs ...

    # 2D profiling configurations
    "2d_test" => (
        grid_1d=15, grid_2d=[5,5], timeout=40.0, n_guesses=3,
        do_2d=true, mle_guesses=9, mle_timeout=60.0
    ),

    "2d_dev" => (
        grid_1d=15, grid_2d=[15,15], timeout=40.0, n_guesses=3,
        do_2d=true, mle_guesses=9, mle_timeout=60.0
    ),

    "2d_paper" => (
        grid_1d=15, grid_2d=[21,21], timeout=40.0, n_guesses=3,
        do_2d=true, mle_guesses=9, mle_timeout=60.0
    ),

    "2d_high_quality" => (
        grid_1d=15, grid_2d=[31,31], timeout=50.0, n_guesses=3,
        do_2d=true, mle_guesses=9, mle_timeout=60.0
    )
)
```

## Expected Runtime Improvements

| Grid Size | Points | Sequential | 4 Workers | Speedup |
|-----------|--------|------------|-----------|---------|
| 5×5       | 25     | ~30 min    | ~8 min    | 3.75×   |
| 15×15     | 225    | ~450 min   | ~120 min  | 3.75×   |
| 21×21     | 441    | ~882 min   | ~235 min  | 3.75×   |
| 31×31     | 961    | ~1920 min  | ~512 min  | 3.75×   |

(Speedup slightly less than 4× due to chunk boundaries losing continuation)

## File Changes Summary

### Modified Files
1. **core.jl**
   - Add `profile_point()` function (~50 lines)
   - Add `profile_grid_sequential()` function (~60 lines)
   - Refactor `profile_target()` to delegate (~30 lines changed)
   - Add `partition_grid_for_continuation()` helper (~40 lines)
   - Add `profile_grid_distributed()` function (~80 lines)
   - Add `using Distributed` at top
   - **Total addition: ~260 lines, ~30 lines modified**

2. **repressilator.jl**
   - Add 2D profiling section with worker management (~100 lines)
   - Add 2D plotting code (~50 lines)
   - Add new PROFILE_CONFIGS entries (~20 lines)
   - **Total addition: ~170 lines**

### New Files (Optional)
3. **test/test_profile_distributed.jl**
   - Unit tests for new functions
   - Validation tests comparing sequential vs distributed
   - **~100 lines**

## Key Design Decisions

### 1. Why Layer the Architecture?

**Benefits:**
- **Testability**: Each layer can be tested independently
- **Reusability**: `profile_point()` useful for ad-hoc evaluations
- **Maintainability**: Clear separation of concerns
- **Flexibility**: Can mix and match (sequential for 1D, distributed for 2D)

**Trade-offs:**
- More code (~260 lines vs ~50 line hack)
- More abstraction (3 levels instead of 1)
- **Worth it:** Clean, general solution vs technical debt

### 2. Why Process-Based (Distributed) Not Thread-Based?

**NLopt is not thread-safe:**
- C/Fortran global state
- Concurrent calls cause segfaults
- No workaround with locks (eliminates speedup)

**Process-based parallelism:**
- Isolated address spaces
- No shared NLopt state
- Clean, safe solution
- Standard approach for this problem

### 3. Why Chunk with Continuation?

**Alternative 1: Fully parallel (no continuation)**
- Each grid point independent
- Maximum parallelism
- **Problem:** Wastes warm start information, slower per-point

**Alternative 2: Chunked with continuation (chosen)**
- Each worker gets sequential strip
- Preserves continuation within chunks
- Loses continuation between chunks
- **Best balance:** Near-maximum parallelism with most continuation benefit

**Alternative 3: Fully sequential**
- Maximum continuation benefit
- **Problem:** Too slow (15-32 hours)

### 4. Why Not Modify profile_target() Directly?

**Could add parallel mode inside profile_target():**
```julia
function profile_target(...; parallel=false, n_workers=4)
    if parallel
        # distributed logic here
    else
        # sequential logic here
    end
end
```

**Problems:**
- Monolithic function gets even bigger
- Harder to test (two paths in one function)
- Mixing concerns (API + execution strategy)

**Chosen approach:**
- Separate functions for different strategies
- `profile_target()` stays simple (delegates)
- Clear interfaces between layers

## Testing Strategy

### Unit Tests
```julia
@testset "profile_point" begin
    # Test single point optimization
    # Test multi-start finds global optimum
    # Test convergence tracking
end

@testset "profile_grid_sequential" begin
    # Test grid iteration
    # Test adaptive continuation improves results
    # Test matches old profile_target() output
end

@testset "partition_grid_for_continuation" begin
    # Test stripe partitioning
    # Test chunk sizes balanced
    # Test order preservation
end

@testset "profile_grid_distributed" begin
    # Test 2 workers vs sequential (deterministic)
    # Test 4 workers speedup
    # Test serialization works
end
```

### Integration Tests
```julia
@testset "repressilator 2D profiling" begin
    # Test 5×5 grid completes
    # Test output format correct
    # Test contour plot generates
end
```

### Validation Tests
```julia
@testset "distributed matches sequential" begin
    # Run 5×5 both ways
    # Compare likelihood values (should match within tolerance)
    # Compare convergence statistics
end
```

## Success Criteria

### Must Have (Phase 1-3)
- [ ] `profile_target()` refactored, existing code still works
- [ ] stat_model example produces identical results before/after
- [ ] `profile_grid_distributed()` implemented and tested
- [ ] 15×15 2D profile completes in ~2 hours with 4 workers
- [ ] Contour plot shows expected ridge along β₁K₁ = constant

### Nice to Have (Phase 4)
- [ ] Unit tests for all new functions
- [ ] Documentation in docstrings
- [ ] 21×21 or 31×31 publication-quality figure
- [ ] Example usage in documentation

### Future Work (Post-Paper)
- [ ] 3D profiling support (extend chunking strategy)
- [ ] Better load balancing (dynamic work stealing)
- [ ] Progress reporting for long runs
- [ ] Checkpoint/resume for crashed runs

## Timeline Estimate

### Optimistic (focused work, no interruptions)
- Phase 1 (refactor): 2 hours
- Phase 2 (distributed): 1 hour
- Phase 3 (driver): 0.5 hours
- Phase 4 (testing): 1 hour
- **Total: 4.5 hours + 2-4 hours runtime for tests**

### Realistic (with debugging, testing)
- Phase 1: 3 hours
- Phase 2: 2 hours
- Phase 3: 1 hour
- Phase 4: 2 hours
- **Total: 8 hours + runtime**

### Conservative (Murphy's law)
- Phase 1: 4 hours
- Phase 2: 3 hours
- Phase 3: 1.5 hours
- Phase 4: 3 hours
- **Total: 11.5 hours + runtime**

**Recommendation:** Block out 1-2 work days for implementation and testing.

## Rollback Plan

If implementation hits major issues:

### Fallback Option 1: Sequential 15×15
- Don't implement distributed
- Just enable `do_2d=true` with 15×15 grid
- Run overnight (~7.5 hours)
- Good enough for paper

### Fallback Option 2: Quick Hack
- Implement simple pmap in repressilator.jl only
- No refactoring of core.jl
- 2-3 hours work
- Technical debt but gets results

### Fallback Option 3: Lower Resolution
- Accept 10×10 or 12×12 grid
- ~3-4 hours sequential
- Contours less smooth but shows the ridge

## Next Steps

1. **Save context** for fresh session (use context-manager agent)
2. **Review this plan** with fresh eyes
3. **Block out time** (8-12 hours work + runtime)
4. **Start with Phase 1** (refactoring) - can validate before committing
5. **Test thoroughly** at each phase before proceeding
6. **Generate 15×15 figure** first, then decide on 21×21 vs 31×31

## Questions to Resolve Before Starting

1. **Timeline**: When do you need the 2D profile figure for the paper?
2. **Workers**: How many cores available on your machine?
3. **Resolution**: Is 15×15 acceptable, or must have 21×21/31×21?
4. **Validation**: How much testing before production run?
5. **Fallback**: If distributed fails, is sequential 15×15 overnight acceptable?

## References

- NLopt thread safety: https://github.com/JuliaOpt/NLopt.jl/issues/103
- Julia Distributed docs: https://docs.julialang.org/en/v1/stdlib/Distributed/
- Profile likelihood theory: Raue et al. (2009) Bioinformatics
- Expert feedback document: `2D_PROFILING_PARALLELIZATION_QUESTION.md`
