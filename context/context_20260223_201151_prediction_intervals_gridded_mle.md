# Context Save — Prediction Intervals (gridded-MLE aligned)

## Goal
Fix `compute_prediction_intervals.jl` so prediction bands are computed/visualized correctly from saved 2D profile results, then leave a resumable state.

## Current State
- `compute_prediction_intervals.jl` now fixes both core issues:
  1. **ψ ordering reconstruction** from saved `ψ_vals` (optimization order → canonical ψ order before `ψ_to_θ`).
  2. **Proper 1D profile extraction** (maximize over the other target coordinate), not MLE row/column slices.
- MLE alignment update:
  - Figure uses **gridded MLE** as the plotted MLE trajectory.
  - Original (continuous) MLE is kept in saved outputs for diagnostics but not plotted.
- PWA-style extension added:
  - New figure overlays available individual profile-wise intervals and their union:
    - `nesi/repressilator_16nuisance_50x50_results_predictions_pwa_union.png`
  - Union currently means **union over available profile-wise intervals in this file** (here, identifiable and non-identifiable target directions).
  - Recommended write-up wording: this is a **profile-wise skeleton approximation to the pushforward of the full likelihood acceptance set** (not claimed as exact/full pushforward).
- Diagnostics printed in script:
  - original MLE nearest grid point vs gridded MLE location
  - inclusion counts of original/gridded MLE in identifiable/non-identifiable/union bands

## Decisions
- Keep `df = rank_J` thresholding (`df=15`) unchanged.
- Keep both MLE trajectories in saved `.jls` metadata, but only plot gridded MLE.
- Keep explicit diagnostics in console + saved fields for auditability.

## Key Files
- Updated script: `compute_prediction_intervals.jl`
- Latest outputs (regenerated):
  - `nesi/repressilator_16nuisance_50x50_results_predictions.png`
  - `nesi/repressilator_16nuisance_50x50_results_predictions_pwa_union.png`
  - `nesi/repressilator_16nuisance_50x50_results_predictions.jls`

## Validation Command
```bash
julia --project=. compute_prediction_intervals.jl nesi/repressilator_16nuisance_50x50_results.jls
```

## Notable Validation Results
- Gridded MLE at row 27, col 27 (`k=1327`), `ll=0.0`.
- Original MLE nearest grid point row 29, col 25, `ll=-0.0594`.
- Gridded MLE inclusion in identifiable/non-identifiable/union bands: `[0,0,0]` outside points for all species.

## Next Steps
1. User visual/scientific verification of the two regenerated figures.
2. Manuscript/caption wording: make explicit that union is over available profile-wise directions and is a profile-wise skeleton approximation.
3. If needed later: extend from available-direction union to broader per-combination unions by running additional profile directions.

## Next-Session Reminder (from user)
- In PWA presentation we usually show influence of each parameter combination on each output.
- The union envelope should be interpreted as union over **available** individual profile-wise intervals unless a broader set of directions is explicitly profiled.

## Continuation Prompt
"Continue from the gridded-MLE aligned `compute_prediction_intervals.jl` with PWA union figure now implemented. Focus on manuscript-ready wording/annotation and any small presentation cleanup; keep ψ-order reconstruction and profile-path extraction logic unchanged."
