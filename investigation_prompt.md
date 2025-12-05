# Investigation Request: Repressilator 2D Profile Discrepancy

## Problem Statement

We have a mystery with repressilator 2D profile likelihood plots. An Oct 29, 2024 plot shows a clean diagonal ridge structure, but recent attempts to reproduce it with 10×10 grids show complex multi-modal structure instead.

## Key Files to Examine

1. **Oct 29 "good" plot**: `repressilator_2D_distributed_beta1_K1.png` (created Oct 27-28, modified Oct 28)
2. **Recent 10×10 plot**: `repressilator_2D_10x10_beta1_K1.png` (shows complex structure)
3. **Recent 3×3 plot**: `repressilator_2D_3x3_beta1_K1.png` (shows tiny region in corner)
4. **Oct 29 context**: `prompt_20251029_002945.md` (describes what was working at that time)

## What We Know

### Git History
- **Commit 7a7d8bb** (Oct 24): Had GRID_2D=3, NT=6, T_end=8000.0, "OLD" parameters
- **Commit 00b9143** (Oct 28-29): Upgraded to GRID_2D=100, same OLD parameters
- **Commit 8658bfd** (Oct 21): "EXACT" parameters from examples/repressilator.jl

### Parameter Sets

**"OLD" parameters** (from commits 7a7d8bb, 00b9143):
```julia
NT = 6
T_end = 8000.0
θ_true = [0.5, 0.5, 0.5,  # α₀
          10.0, 10.0, 10.0,  # α
          0.02, 0.01, 0.015,  # β
          30.0, 26.0, 32.0,  # K
          0.35, 0.35, 0.35,  # k_degm
          0.12, 0.12, 0.12]  # k_degp
```

**"EXACT" parameters** (from prompt_20251029_002945.md, commit 8658bfd):
```julia
NT = 8
T_end = 10000.0
θ_true = [0.008, 0.009, 0.010,  # α₀
          1.0, 1.2, 1.5,         # α
          0.02, 0.025, 0.015,    # β
          30.0, 28.0, 32.0,      # K
          0.006, 0.0055, 0.0065, # k_degm
          0.0012, 0.0011, 0.0013]# k_degp
```

### Current Confusion

The Oct 29 prompt file (`prompt_20251029_002945.md`) says they were using the "EXACT" parameters and it was working well. But:

1. Commit 00b9143 (Oct 28-29) has the "OLD" parameters
2. The Oct 29 plot looks smooth and clean (diagonal ridge)
3. Recent 10×10 run with "OLD" parameters shows complex structure
4. Recent 10×10 run with "EXACT" parameters also shows complex structure

## Questions to Investigate

1. **What actually generated `repressilator_2D_distributed_beta1_K1.png`?**
   - Which commit/code version?
   - What grid resolution? (prompt mentions 7×7 initially, commit says 100×100)
   - What parameter values?

2. **Why do different parameter sets produce different likelihood surfaces?**
   - The "OLD" vs "EXACT" parameters are very different
   - But the Oct 29 prompt suggests "EXACT" was working
   - Is the Oct 29 plot actually from "OLD" or "EXACT" parameters?

3. **Is the complex structure in 10×10 plots real or an artifact?**
   - Could it be numerical noise in the optimization?
   - Could it be sparse grid sampling revealing true complexity?
   - Or is there a bug in the current implementation?

4. **What changed between Oct 29 and now?**
   - Same random seed (42) is used
   - Same model code (RepressilatorModel)
   - But different results

## Investigation Tasks

1. Check `git show 00b9143:test_repressilator_full_workflow.jl` to see exact settings
2. Check examples/repressilator.jl at commit 8658bfd to verify "EXACT" parameters
3. Try to find which exact command/settings generated the Oct 29 PNG
4. Look for any other changes in the codebase that might affect likelihood computation
5. Examine the actual likelihood values from different runs (not just plots)

## Files Available for Review

- `test_repressilator_10x10.jl` - current 10×10 test (has "OLD" parameters now)
- `test_repressilator_2D_distributed.jl` - current 3×3 test (has "EXACT" parameters)
- `test_repressilator_full_workflow.jl` - not in current directory but in git history
- `examples/RepressilatorModel.jl` - the actual ODE model
- `ReparamTools.jl` - contains profile_target function

## Expected Outcome

Identify:
1. Exactly what generated the "good" Oct 29 plot
2. Why current attempts don't match it
3. What settings/parameters we should use to reproduce it
4. Whether the complex structure is real or an artifact

## Context

This is for a paper revision (SIAM/ASA JUQ). The repressilator example demonstrates IIR on a realistic ODE model with 18 parameters. The 2D profile of (β₁, K₁) should show that their ratio K₁/β₁ is identifiable (ridge structure) while individual parameters are not.
