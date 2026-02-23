# Phase 6: Prediction Intervals — BROKEN, Needs Fix

**Saved:** 2025-02-23 15:30
**Branch:** revision1
**Last commits:** `0da6e67` (Tasks 2+3 OK), `08580cc` (Task 1 — incorrect), `d9f3f35` (marked incorrect)

## What Was Done (Tasks 2+3 — OK)
- Fixed Hill coefficient n=3 → n=2.5 in `examples/repressilator.jl` and `CLAUDE.md`
- Added SUPERSEDED headers to `iir_guided_profiling_18param.jl`, `nesi/run_nesi_50x50.jl`, `nesi/run_nesi_100x100.jl`
- Committed as `0da6e67` — these changes are fine

## What Is Broken (Task 1 — `compute_prediction_intervals.jl`)

### Problem 1: Parameterisation likely wrong (NOT YET DIAGNOSED)
- `ψ_vals` from .jls are in **log space** (values like -4.13); `ψ_MLE` is in **natural space** (values like 0.006)
- To predict, need to go ψ-log → ψ (natural) → **θ (original model params: α₀, α, β, K, k_degm, k_degp)**
- `RepressilatorModel.predict_mRNA` takes **θ** (original params), NOT ψ
- The transform chain is: `θ = ψ_to_θ(exp.(ψ_log))` where `ψ_to_θ = exp.(A \ log.(ψ))` — formula matches `parameterizations.jl` but **not verified by round-tripping MLE**
- User observed CIs don't follow MLE trend line and suspects wrong parameterisation
- Key distinction: β and K are original model parameters; β·K and K/β are IIR coordinates. The model takes β and K, not their products/ratios
- **First debugging step**: round-trip `ψ_MLE` through the transform chain, compare against `θ_MLE`

### Problem 2: Method likely wrong (NOT YET CONFIRMED as cause of symptoms)
- Current approach: fix at MLE row/column, vary the other direction (slice)
- Correct approach: for each value of target, **maximize likelihood over the other direction** (proper profile extraction)
- Two-stage profiling = simultaneous: max_a(max_b|a) = max_{a,b}
- m₂ shows reversed pattern (non-identifiable width 0.628 > identifiable width 0.014) — could be caused by method bug OR parameterisation bug OR both. Not diagnosed.

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

## Instructions for Next Session
- **Do not trust the previous agent's assertions.** Multiple confident claims were made without verification and turned out wrong or unverified. Treat everything in this file as hypotheses unless you verify it yourself.
- **Diagnose before fixing.** Don't assume you know the root cause. Start with the round-trip test (step 1 below), look at what the data actually says, and build understanding incrementally.
- **Do not declare success.** Show results to the user and ask them to verify before committing or claiming anything works.
- **Be honest about uncertainty.** If you don't know something, say so. Don't paper over gaps with confident language.

## What Needs to Happen Next
1. **Diagnose parameterisation**: Round-trip `ψ_MLE` through the transform chain and compare against `θ_MLE`. If they don't match, trace exactly where the chain breaks. Don't assume you know the answer — check it.
2. **Diagnose method**: Once parameterisation is fixed, check whether MLE-slice vs proper profile extraction actually matters. It's theoretically wrong but the practical impact is unknown.
3. **Verify before declaring success**: Show results to user, get confirmation.
4. **df = rank_J = 15**: This was agreed as correct and doesn't need changing — but verify the reasoning still holds once the bugs are fixed.

## Agreed Design (Still Valid)
- Post-process saved .jls (no re-optimization)
- Solve ODE at surviving parameter vectors on fine time grid
- Prediction bands = pointwise min/max envelope
- df = rank_J for threshold (joint confidence region over identifiable directions)
- Expected result: identifiable direction (K₁/β₁) → non-zero prediction width; non-identifiable (β₁·K₁) → ~zero width
- Separate script from `replot_profile_results.jl` (needs ODE solves)

## Continuation Prompt
"Resume fixing `compute_prediction_intervals.jl`. The previous attempt produced incorrect results — see `context/context_20260223_phase6_prediction_intervals_broken.md` for what's known and what's hypothesised. Start by diagnosing, not fixing. Do not declare success without user verification."
