# Context Save — repressilator 5163478 selected as canonical candidate

## Goal
Finalize a clean repressilator 2D profile/prediction package with:
- broad ratio/product window
- no missing corners in the displayed 2D interest-space rectangle
- no optimiser striping
- clear identified vs non-identified directions

## Final Selected Run
Selected NeSI run:
- `5163478`

Primary local artifacts from that run:
- `nesi/repressilator_16nuisance_50x50_results_5163478.jls`
- `nesi/iir_50x50_5163478.out`
- `nesi/iir_50x50_5163478.err`

Log confirms:
- exact-interest + original-coordinate-complement chart
- structured 2D continuation
- broad clean full rectangle

Key lines from `.out`:
```text
Target window: clean full-rectangle range [K₁/β₁ ∈ (10, 40000), β₁·K₁ ∈ (0.02, 20)]
Structured 2D profiling: 50×50 grid → 36 row blocks × 2 sweep(s)
```

## Chosen Policy
Current canonical profiling/display policy:
- interest-space window:
  - `K₁/β₁ ∈ [10, 40000]`
  - `β₁·K₁ ∈ [0.02, 20]`
- profiling bounds:
  - `β₁, β₂, β₃ ∈ [0.0005, 2.0]`
  - `K₁, K₂, K₃ ∈ [0.4, 1000]`
- other parameter classes unchanged from prior broader profiling box

## Why 5163478 Was Chosen
Compared with the earlier clean-but-narrow and structured-legacy-window runs, `5163478` is the first one that cleanly combines:
- no missing corners in the 2D ratio/product rectangle
- no suspicious interior optimisation dips
- full target-pair feasibility (`2500 / 2500`)
- identifiable direction still visibly peaked
- non-identifiable direction still flat
- prediction bands with the scientifically desired behavior

Read-only summary for `5163478`:
- finite points: `2500 / 2500`
- accepted (`df = 2`): `1350 / 2500`
- accepted (`df = rank_J = 15`): `1800 / 2500`
- 1D identifiable rows (`df = 1`): `24 / 50`
- 1D non-identifiable columns (`df = 1`): `50 / 50`
- suspicious interior dips: `0`
- analytic target-pair feasibility: `2500 / 2500`
- finite xor feasible: `0`

Accepted identifiable ratio span remained close to the previous good story:
- approximately `90` to `4430`

## Generated Outputs
Job-numbered outputs generated locally:
- `nesi/repressilator_16nuisance_50x50_results_5163478_replot.png`
- `nesi/repressilator_16nuisance_50x50_results_5163478_scatter.png`
- `nesi/repressilator_16nuisance_50x50_results_5163478_predictions_full2d_vs_profiles.png`
- `nesi/repressilator_16nuisance_50x50_results_5163478_predictions.jls`

Canonical outputs then regenerated after copying `5163478` to the unnumbered canonical result path:
- `nesi/repressilator_16nuisance_50x50_results.jls`
- `nesi/repressilator_16nuisance_50x50_results_replot.png`
- `nesi/repressilator_16nuisance_50x50_results_scatter.png`
- `nesi/repressilator_16nuisance_50x50_results_predictions_full2d_vs_profiles.png`
- `nesi/repressilator_16nuisance_50x50_results_predictions.jls`

## θ-space Display Decision
A display-only θ-space crop was adopted to focus on the well-resolved region rather than the full admissible wedge:
```text
β₁ ∈ [0, 0.6]
K₁ ∈ [0, 400]
```

Rationale:
- this captures the practically relevant evaluated/accepted region cleanly
- avoids a huge mostly empty θ-space panel
- remains a **display crop only**, not a new inferential bound

Important caveat:
- the interpolated θ-space panel in `replot_profile_results.jl` still uses RBF interpolation over transformed points, so it can visually extrapolate across the cropped box
- the scatter plot (`replot_scatter.jl`) is therefore the more truthful diagnostic of where actual evaluated points lie in θ-space

## Files Expected to Be in the Final Selective Commit
Code:
- `core.jl`
- `run_repressilator_profile.jl`
- `replot_profile_results.jl`
- `replot_scatter.jl`

Canonical artifacts:
- `nesi/repressilator_16nuisance_50x50_results.jls`
- `nesi/repressilator_16nuisance_50x50_results_replot.png`
- `nesi/repressilator_16nuisance_50x50_results_predictions.jls`
- optionally/likely also:
  - `nesi/repressilator_16nuisance_50x50_results_scatter.png`
  - `nesi/repressilator_16nuisance_50x50_results_predictions_full2d_vs_profiles.png`

## Next Steps
1. Make a **selective commit only** of the repressilator profiling/plotting files and canonical artifacts (avoid the noisy unrelated working-tree files).
2. If desired later, do a small plotting cleanup so the interpolated θ-space panel masks or avoids extrapolated fill outside the actual transformed-point cloud.
3. Then continue with manuscript-facing figure/caption/results text decisions using `5163478` as the leading canonical result.

## Continuation Prompt
Resume from the canonicalized `5163478` repressilator result. The run is scientifically and technically the current best candidate. Treat the scatter θ-space panel as the truthful check on actual evaluated coverage; the interpolated θ-space panel is useful but can still visually extrapolate within the display crop. If further cleanup is desired, focus on presentation/manuscript alignment or on masking the interpolated θ-space display outside the actual transformed-point cloud.
