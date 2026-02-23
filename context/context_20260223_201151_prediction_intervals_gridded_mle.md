# Context Save — Prediction Intervals (gridded-MLE aligned)

## Goal
Fix `compute_prediction_intervals.jl` so prediction bands are computed/visualized correctly from saved 2D profile results, then leave a resumable state.

## Current State
- `compute_prediction_intervals.jl` now fixes both core issues:
  1. **ψ ordering reconstruction** from saved `ψ_vals` (optimization order → canonical ψ order before `ψ_to_θ`).
  2. **Proper 1D profile extraction** (maximize over the other target coordinate), not MLE row/column slices.
- Additional alignment update:
  - Figure now uses **gridded MLE** as the plotted MLE trajectory.
  - **Original (continuous) MLE curve removed from plot** (still computed/saved for diagnostics).
- Diagnostics printed in script:
  - original MLE nearest grid point vs gridded MLE location
  - inclusion counts of original/gridded MLE in each band

## Decisions
- Keep `df = rank_J` thresholding (`df=15`) unchanged.
- Keep both MLE trajectories in saved `.jls` metadata, but only plot gridded MLE.
- Keep explicit diagnostics in console + saved fields for auditability.

## Key Files
- Updated script: `compute_prediction_intervals.jl`
- Latest outputs (regenerated):
  - `nesi/repressilator_16nuisance_50x50_results_predictions.png`
  - `nesi/repressilator_16nuisance_50x50_results_predictions.jls`

## Validation Command
```bash
julia --project=. compute_prediction_intervals.jl nesi/repressilator_16nuisance_50x50_results.jls
```

## Notable Validation Results
- Gridded MLE at row 27, col 27 (`k=1327`), `ll=0.0`.
- Original MLE nearest grid point row 29, col 25, `ll=-0.0594`.
- Gridded MLE inclusion in both bands: `[0,0,0]` outside points for all species.

## Next Steps
1. User visual/scientific verification of regenerated figure.
2. If accepted, use this figure/data for manuscript text and captioning.
3. Optional cleanup: reduce console diagnostic verbosity once final.

## Next-Session Reminder (from user)
- For PWA-style presentation, we usually show the influence of **each parameter combination** on **each output**.
- Also include the **union envelope over all individual profile-wise intervals** (as in Simpson & Maclaren, PLOS CB framing), not only the two-direction comparison panel.
- This is a planning note for next session; no implementation done yet.

## Continuation Prompt
"Continue from the gridded-MLE aligned `compute_prediction_intervals.jl`. Next, plan/implement PWA-style per-combination influence plots per output plus the union over all individual profile-wise intervals (Simpson & Maclaren PLOS CB style), while keeping ψ-order reconstruction and profile-path extraction logic unchanged."
