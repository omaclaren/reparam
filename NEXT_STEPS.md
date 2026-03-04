# NEXT STEPS — IIR Revision (Action Backlog)

**Last Updated:** 2026-03-04
**Branch:** `revision1`

This file is the actionable backlog. For project context/strategy, see `AGENTS.md`.

---

## A) Immediate (active workflow)

### A1. Repressilator figure/text integration
- [ ] Finalize caption for full2D vs 1D-id vs 1D-nonid prediction comparison
- [ ] Add short Results paragraph interpreting the three bands
- [ ] Confirm figure filename/placement in manuscript workflow

### A2. Ownership walkthrough completion
- [ ] Walk through `run_repressilator_profile.jl` end-to-end (post-hybrid)
- [ ] Walk through `replot_profile_results.jl`
- [ ] Walk through `extract_iir_diagnostics_from_results.jl`

### A3. Active script consistency checks
- [ ] Quick contract check: required keys produced by `run_repressilator_profile.jl`
- [ ] Quick legacy check: postprocessor fallback still clear for pre-data-contract `.jls`

---

## B) Manuscript-facing tasks

- [ ] Methods section pass: align with current `invariance.jl` + active workflow scripts
- [ ] Results section pass: stat model + repressilator narrative coherence
- [ ] Add note on invariance threshold scaling rationale (`τ_inv` interpretation)
- [ ] Draft reviewer response items for single-stage focus

---

## C) Legacy/example governance

- [ ] Audit non-active examples (`mm_model`, `transport_model`, `pk_model`, `stat_sum_model`)
  - classify each as: keep active / keep archival / remove
- [ ] Add short status note in each legacy example header (if retained)
- [ ] Archive any remaining legacy/sequential writeups to `archive/legacy-sequential/` (if identified)

---

## D) Merge-back plan (`revision1` → `main`)

- [ ] Define minimal merge set (files + commits)
- [ ] Exclude archived/legacy and local-noise artifacts
- [ ] Prepare clear merge narrative:
  1. workflow stabilization,
  2. postprocessing refactor,
  3. reproducibility/data-contract improvements,
  4. hybrid deprecation + archival cleanup
- [ ] Dry-run checklist before merge

---

## E) Optional quality improvements (later)

- [ ] Consider promoting reusable envelope/aggregation helper from repressilator script into library utilities
- [ ] Revisit docstring-style consistency in core module files (non-urgent)
- [ ] Optional dependency simplification (e.g., varimax implementation strategy)

---

## Done recently (for orientation)

- [x] Prediction postprocessing refocused on full2D vs 1D-id vs 1D-nonid
- [x] `compute_prediction_intervals.jl` replaced by `repressilator_prediction_intervals_from_2d_profile.jl`
- [x] Repressilator runner stores observation data/metadata in results
- [x] Hybrid mode removed from active runner
- [x] Hybrid artifacts archived under `archive/hybrid/`
- [x] Canonical context recentered to `AGENTS.md`
