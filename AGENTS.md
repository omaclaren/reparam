# AGENTS.md — Master Overview & Execution Plan (reparam)

**Last Updated:** 2026-03-04  
**Primary working branch:** `revision1`  
**Goal:** move from stabilized revision branch to (a) manuscript-ready results and (b) clean public-repo update on `main`.

---

## 0) What this project is currently doing

This revision cycle has two coupled outputs:

1. **Paper output (primary):**
   - Finalize the **single-stage IIR** narrative and evidence.
   - Ensure both paper examples are fully up to date:
     - `examples/stat_model.jl` (pedagogical)
     - repressilator workflow (ambitious ODE case)

2. **Public repo output (parallel):**
   - Finish code ownership/cleanup on `revision1`.
   - Modernize legacy examples so they run with current code (even if not used in paper).
   - Merge only clear, maintainable, reproducible pieces back to `main`.

This is intentionally not a repressilator-only cleanup pass.

---

## 1) Scientific scope and positioning

### Core contribution (paper)
- **Single-stage numerical IIR** with monomial transforms.
- Separation of identifiable/non-identifiable combinations via invariant null-space testing.

### Paper examples (must be current and reproducible)
1. **`examples/stat_model.jl`** (pedagogical)
   - Keep mathematically clean and synchronized with current API/outputs.
2. **Repressilator** (ambitious)
   - Use established profiling/post-processing workflow and canonical results.

### Legacy examples (public repo, non-paper)
- `examples/mm_model.jl`
- `examples/transport_model.jl`
- `examples/pk_model.jl`
- `examples/stat_sum_model.jl`

These should be kept compatible and understandable for public users, but clearly labeled as **legacy / exploratory / non-paper**.

---

## 2) Current locked baseline (already done)

### Repressilator workflow state
- Canonical result file: `nesi/repressilator_16nuisance_50x50_results.jls`
- Active runner: `run_repressilator_profile.jl` (**slice/profile only**)
- Hybrid mode removed from active path (`--mode=hybrid` errors by design)
- Historical hybrid artifacts archived under `archive/hybrid/`

### Repressilator prediction comparison framing
Current postprocessing compares:
1. full accepted 2D pushforward
2. 1D profile over identifiable target (`K₁/β₁`)
3. 1D profile over non-identifiable target (`β₁K₁`)

Threshold policy remains consistent with accepted-set interpretation (`df = rank_J`).

### Data reproducibility contract
`run_repressilator_profile.jl` now stores data/metadata in results (`data`, `t_obs`, `X0`, `σ`, etc.), and postprocessing prefers stored data.

---

## 3) Workstreams from here to completion

## Workstream A — Paper readiness

### A1) `stat_model` refresh (required)
- Re-run and verify `examples/stat_model.jl` against current core code.
- Confirm reported rank/coordinates/interpretation are unchanged or intentionally updated.
- Ensure script comments/output still match manuscript wording.

**Exit criteria:** stat_model is fully current, reproducible, and paper-text consistent.

### A2) Repressilator final integration (required)
- Confirm final figure generation path for full2D vs 1D-id vs 1D-nonid comparison.
- Finalize caption + concise results interpretation text.
- Ensure manuscript methods/results language aligns with current implementation.

**Exit criteria:** figure + text are publication-ready and directly insertable.

---

## Workstream B — Public repo example modernization (required for repo quality)

Modernize legacy examples for compatibility with current APIs and conventions.

For each legacy example (`mm_model`, `transport_model`, `pk_model`, `stat_sum_model`):
1. Make code runnable with current module interfaces.
2. Add short status header noting non-paper role.
3. Keep behavior honest (do not over-claim stability if exploratory).
4. Add minimal run instructions/comments.

**Exit criteria:** each retained legacy example is runnable or explicitly marked archival with reason.

---

## Workstream C — Ownership closure on active utilities

Perform final ownership walkthroughs:
- `run_repressilator_profile.jl`
- `replot_profile_results.jl`
- `extract_iir_diagnostics_from_results.jl`
- `repressilator_prediction_intervals_from_2d_profile.jl`

Focus:
- explicit contracts (inputs/outputs/assumptions)
- removal of dead/implicit behavior
- consistency with stored-data pipeline

**Exit criteria:** active scripts are understandable, stable, and maintainable by future-you.

---

## Workstream D — Selective merge & release hygiene

`revision1` is a work-through branch. Merge back to `main` must be selective.

### Include in merge
- Active, validated workflow scripts
- Necessary core module improvements
- Minimal, current docs (`README.md`, `AGENTS.md`, `NEXT_STEPS.md`)
- Archives needed for provenance (`archive/hybrid/`, `archive/legacy-sequential/`)

### Exclude from merge
- temporary artifacts / local noise
- stale exploratory outputs not useful for users
- accidental branch-era clutter

**Exit criteria:** merge diff is understandable and justifiable file-by-file.

---

## 4) Documentation policy (active)

- `README.md` → repo-facing overview and user entrypoint
- `AGENTS.md` → canonical master overview and strategy (this file)
- `NEXT_STEPS.md` → actionable backlog/checklist
- `context/context_*.md` → timestamped session notes only

`CLAUDE.md` is a compatibility shim that points to `AGENTS.md`.

Do not reintroduce evergreen status docs in `context/`.

---

## 5) Definition of Done (project-level)

All items below must be true:

1. **Paper examples complete:**
   - `stat_model` current and manuscript-aligned
   - repressilator figure/text finalized

2. **Public repo examples complete:**
   - legacy examples compatible or explicitly archived with clear status

3. **Ownership complete:**
   - active utility scripts fully walked through and cleaned

4. **Merge complete:**
   - selective `revision1 -> main` update done with clean narrative and no major noise

5. **Docs coherent:**
   - README/AGENTS/NEXT_STEPS consistent with actual code state

---

## 6) Immediate practical priority order

1. Refresh `stat_model` paper example
2. Finalize repressilator figure/caption/results text
3. Modernize legacy public-repo examples
4. Finish active-script ownership pass
5. Execute selective merge plan to `main`

This order keeps manuscript progress and public-repo quality moving together.
