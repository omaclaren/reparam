# IIR Diagnostics Recovered from Saved Repressilator Results

Generated: 2026-02-24 11:23:23

No profile reruns were performed. Diagnostics were recomputed at saved θ_MLE.

| file | rank(saved/recomputed) | n_ident(saved/recomp) | n_nonident(saved/recomp) | σ₁ | σ_r | gap σ_r/σ_{r+1} | target classes |
|---|---:|---:|---:|---:|---:|---:|---|
| repressilator_16nuisance_20x20_results.jls | 15/15 | 15/15 | 3/3 | 8642.0 | 14.75 | 605500.0 | K1/β1, β1·K1 |
| repressilator_16nuisance_50x50_results.jls | 15/15 | 15/15 | 3/3 | 8262.0 | 14.61 | 687500.0 | K1/β1, β1·K1 |
| repressilator_16nuisance_100x100_results.jls | 15/15 | 15/15 | 3/3 | 8642.0 | 14.75 | 605500.0 | K1/β1, β1·K1 |

## Notes
- `gap σ_r/σ_{r+1}` is the rank-separation diagnostic at the recomputed point.
- `n_noninvariant_null_recomputed = n_ident_recomputed - rank_recomputed`.
- Target classes are inferred from saved `A_T_final` columns (basis can be permuted/sign-flipped across runs).
