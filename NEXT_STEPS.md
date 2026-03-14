# NEXT STEPS — IIR Revision Backlog

**Last Updated:** 2026-03-13  
**Branch:** `revision1`  
**Canonical plan:** `AGENTS.md`

---

## PHASE 1 — Repressilator closeout (code/repo side complete; manuscript items remain)

> For current code/repo work, this phase is no longer blocking. Remaining unchecked items here are manuscript-side.

### 1. Figure/package finalization
- [x] Regenerate/verify final full2D vs 1D-id vs 1D-nonid prediction comparison figure
- [x] Confirm figure is generated from canonical `nesi/repressilator_16nuisance_50x50_results.jls`
- [x] Confirm final filename/path used by manuscript workflow
  - 2026-03-10: regenerated canonical artifacts from `nesi/repressilator_16nuisance_50x50_results.jls` using current `repressilator_prediction_intervals_from_2d_profile.jl`.
  - Backed up prior predictions data: `nesi/repressilator_16nuisance_50x50_results_predictions_pre_rename_keys.jls`.
  - Current outputs: `nesi/repressilator_16nuisance_50x50_results_predictions_full2d_vs_profiles.png` and `nesi/repressilator_16nuisance_50x50_results_predictions.jls`.
  - Prediction-envelope key names in `..._predictions.jls` now use accepted-set naming (`lower_pred_from_accepted_ψ1ψ2`, etc.); old keys (`lower_full2d`, etc.) remain in the backup file above.

### 2. Manuscript integration (repressilator)
- [ ] Finalize caption for 3-way prediction comparison figure
- [ ] Add concise Results paragraph interpreting the three comparison bands
- [ ] Ensure repressilator Methods/Results wording matches current implementation

### 3. Repressilator utility ownership pass
- [x] Final walkthrough: `run_repressilator_profile.jl`
- [x] Final walkthrough: `replot_profile_results.jl`
- [x] Final walkthrough: `extract_iir_diagnostics_from_results.jl`
- [x] Final walkthrough: `repressilator_prediction_intervals_from_2d_profile.jl`

### 4. Repressilator consistency checks
- [x] Verify required stored-data keys contract is explicit and stable
  - `run_repressilator_profile.jl` writes `data`, `t_obs`, `X0`, `σ`, `NT`, `T_end` into results.
  - `repressilator_prediction_intervals_from_2d_profile.jl` prefers stored data keys and validates dimensions; legacy files fall back to seeded regeneration with warning.
- [x] Verify postprocessor behavior is clear for old `.jls` files (fallback/warning path)
  - 2026-03-05 verification (legacy canonical artifact): `nesi/repressilator_16nuisance_50x50_results.jls` lacks stored data keys, so fallback seed-42 regeneration path is expected.
  - Full-grid check (2500/2500 points): recomputed likelihoods match saved `ll_vals` up to constant offset `-83.18443557844786`; max residual after offset `1.66e-10` (mean `6.47e-13`).
- [ ] Decide whether to replace the historical repressilator interpretability/Varimax presentation with a simple fixed monomial scan
  - 2026-03-13 read-only identified-space scan from `nesi/repressilator_16nuisance_50x50_results.jls` using `support ≤ 2`, coefficients in `{-1,0,1}`:
    - dictionary size `324`, accepted candidates `147`, accepted span rank `15/15`;
    - greedy selected scan basis recovers exactly the expected interpretable identified basis structure: 12 singleton directions + `β₁/K₁`, `β₂/K₂`, `β₃/K₃`;
    - selected scan-basis span distance to the saved identified profile basis is ~`1e-15`.
  - Earlier manual null-space check also showed `βᵢKᵢ` products span the saved null space to numerical precision.
  - If the basis actually used for profiling is changed materially, plan one clean NeSI rerun for consistency.

---

## PHASE 2 — stat_model paper example refresh (code work substantially complete)

- [x] Run `examples/stat_model.jl` with current codebase
- [x] Confirm rank / identifiable / non-identifiable outputs remain as expected
- [x] Update script comments/output text if drift exists
- [x] Add directional practical near-invariance probe for non-limit case (±δ along weakest singular direction in log-space)
- [x] Report one-sided asymmetry diagnostics for practical weakness (e.g., ε₊(δ), ε₋(δ), optional drift d₊(δ), d₋(δ))
  - Verified both `poisson_limit=true` (on-disk script) and `poisson_limit=false` (in-memory check) run successfully with current code.
  - Non-limit case remains structurally identifiable (full rank) but shows one-sided weakening toward the Poisson limit.
  - Current output compares two weak-coordinate choices: exact local SVD basis and rounded interpretable basis.
- [x] Decide whether to promote this practical probe into a reusable library helper (general vector-direction perturbations) and call it from `examples/stat_model.jl`
  - 2026-03-10 decision: keep the practical directional probe example-local for now.
- [ ] Ensure manuscript wording matches current stat_model behavior

---

## PHASE 3 — Legacy example compatibility (public repo, non-paper) (CURRENT CODE/REPO FOCUS)

Public-facing keeper examples:
- [ ] `examples/mm_model.jl`
  - [x] 2026-03-11 guided technical verification with current code.
  - [x] Explicit invariance-test path added via `find_invariant_subspace`.
  - [x] Limit case verified as the clean exact-IIR case: `K/ν` identifiable, `νK` invariant (in log coordinates, up to basis sign).
  - [x] Non-limit case verified as full-rank / non-invariant, with `νK` remaining a practically weak one-sided direction near the MLE.
  - [x] Script kept in top-level example style; prior soft-scope warnings removed by renaming reused top-level variables.
  - [ ] Add/confirm minimal run instructions for the final public-facing keeper version.
- [ ] `examples/transport_model.jl`
  - 2026-03-10 audit: runs successfully with current code and current seeded randomness.
  - 2026-03-13 read-only transport diagnostics now show:
    - exact null direction `(1,1,1)` and identified plane `a + b + c = 0`;
    - direct rounding of the raw orthogonal identified/null basis gives a stable orthogonal rounded SVD basis (`round_within = 0.45, 0.4, 0.35`);
    - a standalone minimal identified-space monomial scan (`sparse_monomial_scan_diagnostic.jl`, `support ≤ 2`, coefficients in `{-1,0,1}`) accepts `T₂/R`, `T₁/T₂`, and `T₁/R`, which span the identified plane.
  - Decide how to present orthogonal SVD coordinates versus sparse interpretable oblique coordinates in the final example.
  - Decide whether to replace `data = rand(...)` with a fixed saved realization for public-facing reproducibility.
  - Add explicit invariance-test path (`find_invariant_subspace`) for the public-facing version.
  - Do not center Varimax unless later evidence gives a concrete reason.

Exploratory / non-public examples for now:
- [ ] `examples/pk_model.jl` (keep on `revision1`, do not prioritize for `main` public example set)
- [ ] `examples/stat_sum_model.jl` (keep on `revision1`, do not prioritize for `main` public example set)

Working mode for public-facing keeper cleanup:
- [ ] Follow guided/incremental cleanup: inspect, discuss, then edit one example at a time

Per-example checklist for public-facing keepers:
- [ ] Runs with current APIs
- [ ] Has minimal run instructions that are current
- [ ] Keeps keeper/legacy/non-paper metadata in docs rather than script headers
- [ ] Includes explicit invariance-test path
- [ ] Has explicit interpretable-basis decision where relevant (notably `transport_model.jl`: orthogonal SVD basis vs sparse monomial scan)
- [ ] Does not over-claim reliability

---

## PHASE 4 — Merge and docs hygiene

### Documentation consistency
- [ ] Keep `README.md`, `AGENTS.md`, and `NEXT_STEPS.md` aligned with actual state
- [ ] Keep `CLAUDE.md` as compatibility pointer to `AGENTS.md`
- [ ] Keep `context/` to timestamped `context_*.md` notes only
- [ ] Archive any additional discovered legacy writeups to `archive/legacy-sequential/`

### Selective merge plan (`revision1` -> `main`)
- [ ] Define minimal include set (files + rationale)
- [ ] Explicitly exclude local-noise/stale artifacts
- [ ] Prepare merge narrative:
  1. repressilator workflow stabilization,
  2. postprocessing refactor,
  3. reproducibility/data-contract improvements,
  4. hybrid deprecation + archival,
  5. stat_model refresh + legacy-example compatibility updates
- [ ] Dry-run checklist before merge

---

## Recently completed (orientation)

- [x] Canonical context moved to `AGENTS.md`
- [x] `CLAUDE.md` converted to compatibility shim
- [x] Legacy/sequential writeups moved under `archive/legacy-sequential/`
- [x] `context/CURRENT_FOCUS.md` removed
- [x] Repressilator postprocessing focused on full2D vs 1D-id vs 1D-nonid
- [x] Hybrid mode removed from active runner; artifacts archived under `archive/hybrid/`
