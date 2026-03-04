# AGENTS.md — Canonical Project Context (reparam)

**Last Updated:** 2026-03-04  
**Primary Branch:** `revision1`

This is the canonical, cross-harness project context file.

---

## 1) Current Article Direction

The paper revision is focused on **single-stage IIR with monomial transformations**.

Main examples for manuscript:
1. `examples/stat_model.jl` (pedagogical)
2. Repressilator (18-parameter ODE model, ambitious case)

Sequential/multi-stage findings remain useful background, but are treated as legacy/future-work context, not the active core workflow.

---

## 2) Active Repressilator Workflow (authoritative)

### Canonical compute artifact
- `nesi/repressilator_16nuisance_50x50_results.jls` (publication baseline)

### Active scripts
- `run_repressilator_profile.jl` (slice/profile only)
- `replot_profile_results.jl`
- `extract_iir_diagnostics_from_results.jl`
- `repressilator_prediction_intervals_from_2d_profile.jl`

### Prediction comparison now used
- full accepted 2D pushforward
- 1D profile over identifiable coordinate (`K₁/β₁`)
- 1D profile over non-identifiable coordinate (`β₁K₁`)

### Data reproducibility contract
- `run_repressilator_profile.jl` stores observation data/metadata in saved results
  (`data`, `t_obs`, `X0`, `σ`, `θ_true`, etc.)
- post-processing scripts should prefer stored data over synthetic regeneration

---

## 3) Branch Strategy and Ownership Plan

`revision1` is currently a **work-through / ownership / cleanup** branch after a large experimental coding phase.

Goals on `revision1`:
1. Gain explicit ownership over active code paths
2. Simplify and clarify script contracts
3. Archive legacy/deprecated exploration artifacts
4. Keep only understandable, useful, reproducible workflows
5. Plan a selective, understandable merge back to `main`

### Merge-to-main principle
Do **not** merge everything from `revision1` blindly.
Merge only:
- active runner + post-processing scripts,
- validated module improvements,
- minimal documentation needed for maintenance.

---

## 4) Legacy / Exploratory Material Policy

Legacy exploratory content is kept for provenance but not treated as active workflow.

- Hybrid repressilator artifacts are archived in `archive/hybrid/`.
- Sequential/legacy writeups should be archived under `archive/legacy-sequential/`.
- Legacy examples (`mm_model`, `transport_model`, `pk_model`, `stat_sum_model`) are repo-only references unless explicitly reactivated.

---

## 5) Documentation Roles (keep these distinct)

- `README.md` — repo-facing overview and active usage entrypoint
- `AGENTS.md` — canonical internal project context and strategy (this file)
- `NEXT_STEPS.md` — actionable backlog (ownership + manuscript + merge tasks)
- `context/context_*.md` — timestamped session notes only (ephemeral)

Do not create additional “status” files unless explicitly requested.

---

## 6) Ownership Work Remaining (code)

1. Final walkthrough pass on `run_repressilator_profile.jl` (post-hybrid removal)
2. Walkthrough pass on:
   - `replot_profile_results.jl`
   - `extract_iir_diagnostics_from_results.jl`
3. Decide whether helper logic in repressilator postprocessing (e.g., envelope aggregation) stays local or is promoted to library utilities

---

## 7) Manuscript Work Remaining

1. Final caption/text integration for repressilator prediction comparison figure
2. Methods/Results consistency pass vs current implementation
3. Reviewer response draft aligned with single-stage focus and repressilator evidence

---

## 8) Compatibility Note

Some tools look for `CLAUDE.md`. In this repo, `CLAUDE.md` should remain a lightweight pointer to `AGENTS.md`.
