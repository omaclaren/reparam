# NEXT STEPS — IIR Revision Backlog

**Last Updated:** 2026-03-10  
**Branch:** `revision1`  
**Canonical plan:** `AGENTS.md`

---

## PHASE 1 — Repressilator closeout (ACTIVE, BLOCKING)

> Do not start Phase 2+ until all Phase 1 items are complete.

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

---

## PHASE 2 — stat_model paper example refresh (after Phase 1)

- [ ] Run `examples/stat_model.jl` with current codebase
- [ ] Confirm rank / identifiable / non-identifiable outputs remain as expected
- [ ] Update script comments/output text if drift exists
- [ ] Add directional practical near-invariance probe for non-limit case (±δ along weakest singular direction in log-space)
- [ ] Report one-sided asymmetry diagnostics for practical weakness (e.g., ε₊(δ), ε₋(δ), optional drift d₊(δ), d₋(δ))
- [ ] Decide whether to promote this practical probe into a reusable library helper (general vector-direction perturbations) and call it from `examples/stat_model.jl`
- [ ] Ensure manuscript wording matches current stat_model behavior

---

## PHASE 3 — Legacy example compatibility (public repo, non-paper)

Target examples:
- [ ] `examples/mm_model.jl`
- [ ] `examples/transport_model.jl`
- [ ] `examples/pk_model.jl`
- [ ] `examples/stat_sum_model.jl`

Per-example checklist:
- [ ] Runs with current APIs (or mark archival with explicit rationale)
- [ ] Has clear status header: legacy / exploratory / non-paper
- [ ] Has minimal run instructions that are current
- [ ] Does not over-claim reliability if exploratory

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
