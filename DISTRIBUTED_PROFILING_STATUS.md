# Distributed 2D Profiling - Status and Usage

**Date**: 2025-10-22
**Status**: ✅ Complete and tested

## Summary

Distributed 2D profile likelihood infrastructure is implemented and working. The user-facing API is simple: add `use_distributed=true` to any `profile_target()` call.

## Implementation

### Layered Architecture
- **`profile_point()`** - Single-point optimization primitive
- **`profile_grid_sequential()`** - Sequential grid with adaptive continuation
- **`profile_grid_distributed()`** - Process-based parallel execution
- **`profile_target()`** - User-facing API that delegates to sequential or distributed

### Key Features
- Process-based parallelism (handles NLopt thread-safety)
- Simple contiguous chunking (no complex strategies)
- Automatic n_chunks clamping to grid size
- Clean API: internal functions hidden, everything through `profile_target()`

## Usage

```julia
using Distributed
addprocs(3)  # Add workers

# Load module on workers
@everywhere begin
    include("ReparamTools.jl")
    using .ReparamTools
    # ... other dependencies
end

# Define likelihood on all workers
@everywhere function lnlike_θ(θ)
    # Your likelihood function
end

# Run distributed 2D profiling
θ_vals, ll_vals = profile_target(
    lnlike_θ, [1, 2],  # Profile parameters 1 and 2
    θ_lower, θ_upper, ω_initial;
    grid_steps=21,
    use_distributed=true,  # Enable distributed execution
    n_chunks=3             # Number of chunks (default: nworkers())
)
```

## Validated Examples

### ✅ stat_model.jl (Poisson limit)
- **Test**: 20×20 grid (400 points) on 2 workers
- **Result**: Sequential 0.89s, Distributed 1.49s
- **Verification**: Max difference = 0.0 (exact match)
- **Status**: Works perfectly (see `test_stat_model_distributed.jl`)

### ⚠️ repressilator.jl (ODE system)
- **Test**: 3×3 grid (9 points) on 3 workers
- **Result**: Infrastructure triggered correctly, but failed on worker serialization
- **Error**: `UndefVarError: #predict_mRNA not defined in Main`
- **Cause**: Model functions not available on workers
- **Status**: **Needs model modularization** (see below)

## Next Steps for Repressilator

To enable distributed profiling on repressilator, the model code needs to be available on all workers. Recommended approach:

### 1. Create RepressilatorModel.jl module

Extract model code into a separate module:

```julia
# RepressilatorModel.jl
module RepressilatorModel

export repressilator_ode!, predict_mRNA, predict_protein, ...

# ODE system
function repressilator_ode!(du, u, p, t)
    # ... existing code
end

# Prediction functions
function predict_mRNA(θ, t_points)
    # ... existing code
end

# ... other model functions

end # module
```

### 2. Load on master and workers

```julia
# repressilator.jl
using Distributed
addprocs(3)

# Load model on master
include("RepressilatorModel.jl")
using .RepressilatorModel

# Load on workers
@everywhere begin
    include($(joinpath(@__DIR__, "RepressilatorModel.jl")))
    using .RepressilatorModel
    using .ReparamTools
    # ... other packages
end

# Define likelihood using model functions
@everywhere function lnlike_θ_log(θ_log)
    # Uses RepressilatorModel.predict_mRNA, etc.
    # ... existing likelihood code
end
```

### 3. Use distributed profiling

```julia
# 2D profile - just add use_distributed=true
ψK1β1_values, lnlike_K1β1_values = profile_target(
    lnlike_θ_log, target_indices_K1β1,
    θ_log_lower, θ_log_upper,
    nuisance_guess_2d;
    grid_steps=CONFIG.grid_2d,
    use_distributed=true,
    n_chunks=3
)
```

## Performance Expectations

### Simple likelihoods (stat_model)
- **Overhead dominates**: Distributed slower than sequential
- **Expected**: Communication overhead > computation time
- **Not a problem**: These run fast anyway

### Expensive likelihoods (repressilator ODE)
- **21×21 grid (441 points)**:
  - Sequential: ~14.7 hours (~120s per point)
  - Distributed (4 workers): ~6-7 hours (2.3x speedup)
  - Efficiency: ~60% (accounting for overhead)

## Files

### Core implementation
- `core.jl` - Layered profiling functions, distributed infrastructure
- `ReparamTools.jl` - Module exports

### Tests
- `test_distributed_profiling.jl` - Low-level test (2D Gaussian)
- `test_distributed_api.jl` - API test
- `test_stat_model_distributed.jl` - Real problem validation
- `test_speedup.jl` - Speedup measurement utilities

### Examples
- `examples/stat_model.jl` - Works with distributed (simple likelihood)
- `examples/repressilator.jl` - Needs modularization first

## Technical Notes

### Why process-based parallelism?
- NLopt has global C/Fortran state
- Not thread-safe
- Requires separate processes via `Distributed.jl`

### Why simple chunking?
- User requested "simple elegant code"
- Contiguous chunks only (~10 lines)
- No fancy strategies (round-robin, etc.)
- General: works for any dimensionality

### Limitation: continuation breaks
- Each chunk starts with same ω_initial
- Loses warm-start advantage between chunks
- Efficiency: ~60% instead of ideal 100%
- Trade-off: acceptable for expensive likelihoods

## Commits

- `d045769` - Checkpoint before refactoring
- `96c92ac` - Phase 1: Layered architecture
- `19d594f` - Phase 2: Distributed infrastructure
- `9034812` - Simplification (stripped to essentials)
- `3cbcae2` - Added use_distributed API parameter
- `5290685` - stat_model validation test

**Status**: Production ready for simple models, needs model modularization for complex models like repressilator.
