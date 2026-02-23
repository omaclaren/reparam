# Phase 6: Prediction Intervals — BROKEN, Needs Fix

**Saved:** 2025-02-23 15:30
**Branch:** revision1
**Last commits:** `0da6e67` (Tasks 2+3 OK), `08580cc` (Task 1 — incorrect), `d9f3f35` (marked incorrect)

## What Was Done (Tasks 2+3 — OK)
- Fixed Hill coefficient n=3 → n=2.5 in `examples/repressilator.jl` and `CLAUDE.md`
- Added SUPERSEDED headers to `iir_guided_profiling_18param.jl`, `nesi/run_nesi_50x50.jl`, `nesi/run_nesi_100x100.jl`
- Committed as `0da6e67` — these changes are fine

## What Is Broken (Task 1 — `compute_prediction_intervals.jl`)

### Problem 1: Wrong parameterisation (likely)
- `ψ_vals` from .jls appear to be in **log(ψ) space** (values like `[1.97, -4.13, ...]`)
- `ψ_MLE` is in **natural ψ space** (values like `[0.001, 0.006, ...]`)
- The script treats `ψ_vals` entries as log(ψ) and does `exp.()` then `ψ_to_θ()`, but this needs verification
- `ψ_to_θ` from `ReparamTools.reparam()` is `exp.(A \ log.(ψ))` where `A = A_T_final'`
- The prediction CIs do NOT follow the MLE trend line, confirming something is wrong with the transform

### Problem 2: Wrong method for 1D extraction
- Current approach: fix at MLE row/column, vary the other direction (slice)
- Correct approach: for each value of target, **maximize likelihood over the other direction** (proper profile extraction)
- Two-stage profiling = simultaneous: max_a(max_b|a) = max_{a,b}
- Evidence of method failure: m₂ shows REVERSED pattern (non-identifiable width 0.628 > identifiable width 0.014) — impossible if method were correct

### Problem 3: Premature success declaration
- Results committed and "success" declared without user verification
- Should have shown results and asked for verification first

## Key Files
- `compute_prediction_intervals.jl` — marked KNOWN INCORRECT at top
- `nesi/repressilator_16nuisance_50x50_results_predictions.png` — incorrect output (do not use)
- `nesi/repressilator_16nuisance_50x50_results_predictions.jls` — incorrect output (do not use)
- `run_repressilator_profile.jl` — the correct profiling script (reference for parameterisation)
- `parameterizations.jl:161` — `reparam()` function defines `ψ_to_θ`
- `core.jl:731` — `construct_upper_lower_profile_wise_CIs_for_mean` (reference API)

## What Needs to Happen Next
1. **Fix parameterisation**: Verify what space `ψ_vals` are actually in. Check by converting the MLE entry and confirming it matches `θ_MLE`. The `run_repressilator_profile.jl` operates in log(ψ) space for optimization — trace exactly what gets saved.
2. **Fix method**: Replace MLE-slice approach with proper 1D profile extraction from 2D grid. For each row (K₁/β₁ value), take the column that maximises likelihood. For each column (β₁·K₁ value), take the row that maximises likelihood. Use those profile points for prediction bands.
3. **Verify before declaring success**: Show results to user, get confirmation.
4. **df = rank_J = 15**: This was agreed as correct and doesn't need changing.

## Agreed Design (Still Valid)
- Post-process saved .jls (no re-optimization)
- Solve ODE at surviving parameter vectors on fine time grid
- Prediction bands = pointwise min/max envelope
- df = rank_J for threshold (joint confidence region over identifiable directions)
- Expected result: identifiable direction (K₁/β₁) → non-zero prediction width; non-identifiable (β₁·K₁) → ~zero width
- Separate script from `replot_profile_results.jl` (needs ODE solves)

## Continuation Prompt
"Resume fixing `compute_prediction_intervals.jl`. Two bugs: (1) parameterisation — verify ψ_vals space by round-tripping MLE, (2) method — extract proper 1D profiles from 2D grid instead of MLE slices. See `context/context_20260223_phase6_prediction_intervals_broken.md`."
