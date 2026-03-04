# Project Status (Current) — IIR Revision

**Last updated:** 2026-03-04  
**Branch:** `revision1` (ahead work already pushed to `origin/revision1`)

This file is the concise, current source-of-truth for:
1) paper/revision status,  
2) active vs non-active examples,  
3) code-ownership progress.

---

## 1) Paper / article direction (bigger picture)

### Main contribution (locked)
- Single-stage IIR with monomial transformation.
- Clear separation of identifiable vs non-identifiable parameter combinations.
- Numerical invariance test + interpretable basis.

### Main paper examples (locked)
- `examples/stat_model.jl` (pedagogical) ✅
- Repressilator (ambitious ODE example) ✅ computationally established

### Repressilator profile-likelihood demonstration (current status)
- Canonical full-profile result: `nesi/repressilator_16nuisance_50x50_results.jls` ✅
- Prediction comparison now centered on:
  1. full accepted 2D pushforward,
  2. 1D profile over identifiable target (`K₁/β₁`),
  3. 1D profile over non-identifiable target (`β₁K₁`).
- Active postprocessor script:
  - `repressilator_prediction_intervals_from_2d_profile.jl`

### Manuscript-facing work still open
- Final figure caption/text integrating full2D vs 1D-id vs 1D-nonid result.
- Methods/Results wording updates to match current implementation.
- Reviewer-response drafting.

---

## 2) Codebase scope: active vs non-active

### Active workflow scripts (current)
- `run_repressilator_profile.jl` (slice/profile modes)
- `replot_profile_results.jl`
- `extract_iir_diagnostics_from_results.jl`
- `repressilator_prediction_intervals_from_2d_profile.jl`

### Hybrid status
- Hybrid mode removed from active runner (`--mode=hybrid` no longer supported).
- Deprecated hybrid artifacts are archived under:
  - `archive/hybrid/`

### Non-repressilator / legacy examples
- `examples/stat_model.jl`: active and paper-relevant.
- `examples/stat_sum_model.jl`, `examples/pk_model.jl`: sequential-IIR exploration; not main-text workflow.
- `examples/mm_model.jl`, `examples/transport_model.jl`: legacy; not currently part of revision-critical path.

---

## 3) Ownership/review status

### Completed ownership work
- Module review phases (utils, parameterizations, invariance, visualization, core).
- Prediction postprocessing ownership pass (including layout/data-contract cleanup).
- Hybrid removal/archival decisions implemented.

### Remaining ownership tasks (short list)
1. Final walkthrough pass on `run_repressilator_profile.jl` (post-hybrid version).
2. Walkthrough pass on:
   - `replot_profile_results.jl`
   - `extract_iir_diagnostics_from_results.jl`
3. Decide whether to keep envelope helper logic local to repressilator script or promote to library helper.

---

## 4) Recent high-impact commits

- `807d836` — refocus prediction postprocessing to full2D vs 1D profiles.
- `bfe27a8` — remove hybrid mode; make postprocessing data-driven from saved results.
- `1a5d9a4` — archive deprecated hybrid artifacts.

---

## 5) Immediate next steps

1. Lock final paper figure style/caption from the new 3-case comparison.
2. Complete final ownership walkthrough of remaining active scripts.
3. Update manuscript Methods/Results text with current script/data-flow reality.
