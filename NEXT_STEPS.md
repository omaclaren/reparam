# NEXT STEPS — IIR Revision Backlog

**Last Updated:** 2026-03-04  
**Branch:** `revision1`  
**Canonical plan:** `AGENTS.md`

---

## PHASE 1 — Repressilator closeout (ACTIVE, BLOCKING)

> Do not start Phase 2+ until all Phase 1 items are complete.

### 1. Figure/package finalization
- [ ] Regenerate/verify final full2D vs 1D-id vs 1D-nonid prediction comparison figure
- [ ] Confirm figure is generated from canonical `nesi/repressilator_16nuisance_50x50_results.jls`
- [ ] Confirm final filename/path used by manuscript workflow

### 2. Manuscript integration (repressilator)
- [ ] Finalize caption for 3-way prediction comparison figure
- [ ] Add concise Results paragraph interpreting the three comparison bands
- [ ] Ensure repressilator Methods/Results wording matches current implementation

### 3. Repressilator utility ownership pass
- [ ] Final walkthrough: `run_repressilator_profile.jl`
- [ ] Final walkthrough: `replot_profile_results.jl`
- [ ] Final walkthrough: `extract_iir_diagnostics_from_results.jl`
- [ ] Final walkthrough: `repressilator_prediction_intervals_from_2d_profile.jl`

### 4. Repressilator consistency checks
- [ ] Verify required stored-data keys contract is explicit and stable
- [ ] Verify postprocessor behavior is clear for old `.jls` files (fallback/warning path)

---

## PHASE 2 — stat_model paper example refresh (after Phase 1)

- [ ] Run `examples/stat_model.jl` with current codebase
- [ ] Confirm rank / identifiable / non-identifiable outputs remain as expected
- [ ] Update script comments/output text if drift exists
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
