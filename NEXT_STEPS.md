# NEXT STEPS — IIR Revision Backlog

**Last Updated:** 2026-03-04  
**Branch:** `revision1`  
**Canonical overview:** see `AGENTS.md`

---

## A) Paper readiness (must finish)

### A1. stat_model (paper example) refresh
- [ ] Run `examples/stat_model.jl` with current codebase
- [ ] Confirm rank / identifiable / non-identifiable outputs remain as expected
- [ ] Update script comments/output text if any drift exists
- [ ] Ensure manuscript wording matches current stat_model behavior

### A2. Repressilator paper package finalization
- [ ] Regenerate/verify final full2D vs 1D-id vs 1D-nonid prediction comparison figure
- [ ] Finalize caption text for manuscript
- [ ] Add concise Results paragraph interpreting three-band comparison
- [ ] Confirm figure filename/path used in manuscript pipeline

### A3. Methods/results consistency
- [ ] Methods pass: ensure text reflects current `invariance.jl` + profiling workflow
- [ ] Add/update note on invariance threshold scaling rationale (`τ_inv`)
- [ ] Results pass: stat_model + repressilator narrative coherence
- [ ] Draft reviewer-response bullet points for single-stage focus

---

## B) Public repo example compatibility (non-paper but important)

Target examples:
- [ ] `examples/mm_model.jl`
- [ ] `examples/transport_model.jl`
- [ ] `examples/pk_model.jl`
- [ ] `examples/stat_sum_model.jl`

Per-example checklist:
- [ ] Runs with current APIs (or explicitly mark archival if not retained)
- [ ] Add clear header/status note: legacy/exploratory/non-paper
- [ ] Keep minimal run instructions current
- [ ] Avoid over-claiming reliability if example is exploratory

---

## C) Ownership closure on active scripts

- [ ] Final walkthrough: `run_repressilator_profile.jl`
- [ ] Final walkthrough: `replot_profile_results.jl`
- [ ] Final walkthrough: `extract_iir_diagnostics_from_results.jl`
- [ ] Final walkthrough: `repressilator_prediction_intervals_from_2d_profile.jl`

Checks during walkthrough:
- [ ] contracts (inputs/outputs/required keys) are explicit
- [ ] no dead or hidden behavior remains
- [ ] stored-data workflow behavior remains clear

---

## D) Documentation and archival hygiene

- [ ] Keep `README.md`, `AGENTS.md`, and `NEXT_STEPS.md` consistent
- [ ] Archive any additional discovered legacy writeups to `archive/legacy-sequential/`
- [ ] Keep `CLAUDE.md` as compatibility pointer to `AGENTS.md`
- [ ] Keep `context/` to timestamped `context_*.md` notes only

---

## E) Selective merge plan (`revision1` -> `main`)

- [ ] Define minimal include set (files + rationale)
- [ ] Explicitly exclude local-noise/stale artifacts
- [ ] Prepare merge narrative:
  1. paper-workflow stabilization,
  2. postprocessing refactor,
  3. reproducibility/data-contract improvements,
  4. hybrid deprecation + archival,
  5. legacy example compatibility updates
- [ ] Dry-run checklist before merge

---

## Recently completed (orientation)

- [x] Canonical context moved to `AGENTS.md`
- [x] `CLAUDE.md` converted to compatibility shim
- [x] Legacy/sequential writeups moved under `archive/legacy-sequential/`
- [x] `context/CURRENT_FOCUS.md` removed
- [x] Repressilator postprocessing focused on full2D vs 1D-id vs 1D-nonid
- [x] Hybrid mode removed from active runner; artifacts archived under `archive/hybrid/`
