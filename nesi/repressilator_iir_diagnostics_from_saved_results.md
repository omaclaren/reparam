# IIR Diagnostics Recovered from Saved Repressilator Results

Generated: 2026-02-24 15:08:08

No profile reruns were performed. Diagnostics were recomputed at saved θ_MLE.

| file | rank(saved/recomputed) | n_ident(saved/recomp) | n_nonident(saved/recomp) | σ₁ | σ_r | gap σ_r/σ_{r+1} | target classes |
|---|---:|---:|---:|---:|---:|---:|---|
| repressilator_16nuisance_50x50_results.jls | 15/15 | 15/15 | 3/3 | 8262.0 | 14.61 | 687500.0 | K1/β1, β1·K1 |

## Notes
- `gap σ_r/σ_{r+1}` is the rank-separation diagnostic at the recomputed point.
- `n_noninvariant_null_recomputed = n_ident_recomputed - rank_recomputed`.
- Target classes are inferred from saved `A_T_final` columns.
- For manuscript claims, use the chosen publication result file (currently 50x50) and its own saved `θ_MLE`.
- If two files have identical `θ_MLE` and identical `A_T_final`, target indices should match.
- If `θ_MLE` and/or `A_T_final` differ across files, target index numbers may differ; compare monomial class (e.g., K₁/β₁) rather than raw ψ index.
