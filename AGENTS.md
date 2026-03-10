# AGENTS.md — Master Overview & Execution Plan (reparam)

**Last Updated:** 2026-03-10  
**Primary working branch:** `revision1`  
**Current sprint:** **Phase 3 keeper-example cleanup** (`examples/mm_model.jl`, `examples/transport_model.jl`).

---

## 0) Working agreement for this phase

1. **Current code/repo focus: Phase 3 keeper examples.**
   - Work in guided/incremental mode on `examples/mm_model.jl` and `examples/transport_model.jl`.
2. **Repressilator remaining items are manuscript-side.**
   - Do not reopen repressilator code work unless a concrete repo/code issue appears.
3. **stat_model code refresh is sufficiently complete to move on.**
   - Keep the practical directional probe example-local for now.
4. **Then selective merge (`revision1` -> `main`).**

This keeps the docs aligned with the actual current repo focus.

---

## 1) What this revision cycle must produce

Two coupled outputs:

1. **Manuscript output (primary):**
   - Finalize single-stage IIR evidence package.
   - Paper examples up to date and manuscript-consistent.
2. **Public repo output (parallel):**
   - Clean, maintainable, understandable code/docs in `main`.
   - Legacy examples compatible (or explicitly archived) for public users.

---

## 2) Locked baseline (already done)

### Repressilator workflow state
- Canonical result artifact: `nesi/repressilator_16nuisance_50x50_results.jls`
- Active runner: `run_repressilator_profile.jl` (**slice/profile only**)
- Hybrid mode removed from active path (`--mode=hybrid` errors by design)
- Hybrid-era artifacts archived: `archive/hybrid/`

### Repressilator prediction framing (paper)
Postprocessing compares:
1. full accepted 2D pushforward
2. 1D profile over identifiable target (`K₁/β₁`)
3. 1D profile over non-identifiable target (`β₁K₁`)

Threshold policy: accepted-set comparisons use `df = rank_J`.

### Reproducibility contract
`run_repressilator_profile.jl` stores data/metadata in results (`data`, `t_obs`, `X0`, `σ`, etc.); postprocessing should prefer stored data.

---

## 3) Phase plan from here

## Phase 1 (code/repo side complete; manuscript items remain): Repressilator closeout

### Required outputs
- Final/verified prediction comparison figure (full2D vs 1D-id vs 1D-nonid)
- Final manuscript caption for that figure
- Short Results paragraph interpreting the three bands
- Final ownership walkthrough on active repressilator utilities:
  - `run_repressilator_profile.jl`
  - `replot_profile_results.jl`
  - `extract_iir_diagnostics_from_results.jl`
  - `repressilator_prediction_intervals_from_2d_profile.jl`

### Exit criteria
- Repressilator figure/text are insertion-ready.
- Repressilator utility contracts are explicit and stable.

---

## Phase 2 (core code work done; manuscript wording remains): stat_model refresh (paper)

- `examples/stat_model.jl` has been rerun against current code.
- Outputs have been checked against current expectations.
- Directional practical near-invariance check has been added for the non-limit case.
- Current decision: keep the practical directional probe example-local for now (do not promote it to a reusable library helper yet).
- Any remaining work here is mainly manuscript wording / commentary alignment.

### Exit criteria
- stat_model is current, reproducible, and manuscript-consistent.

---

## Phase 3 (CURRENT CODE/REPO FOCUS): legacy examples for public repo (non-paper)

Public-facing keepers to refresh:
- `examples/mm_model.jl`
- `examples/transport_model.jl`

Exploratory examples to keep on `revision1` but not treat as public `main` examples:
- `examples/pk_model.jl`
- `examples/stat_sum_model.jl`

Working style for this phase:
- Do cleanup in guided/incremental mode: inspect, discuss, then edit one example at a time.
- Treat this as moderate cleanup, not just status-header polish.

Per-example policy for public-facing keepers:
1. Make runnable with current APIs where practical.
2. Add clear status header (legacy/supplementary/non-paper).
3. Keep minimal run instructions current.
4. Add explicit invariance-test logic (`find_invariant_subspace`) for both `examples/mm_model.jl` and `examples/transport_model.jl`, since current keeper scripts appear to rely mainly on SVD/rounding.
5. For `examples/transport_model.jl`, assess whether a Varimax-based interpretable basis should be added for the public-facing version.

### Exit criteria
- `examples/mm_model.jl` and `examples/transport_model.jl` are compatible-and-kept.
- Both public-facing keeper examples include explicit invariance testing.
- `examples/transport_model.jl` has an explicit, documented decision on interpretable-basis / Varimax construction.
- `examples/pk_model.jl` and `examples/stat_sum_model.jl` are explicitly treated as exploratory/non-public for merge planning.

---

## Phase 4 (after Phase 3): selective merge to `main`

### Merge framing
- Decide deliberately what to include in `main`; do not assume any file or directory is automatically in-scope.
- Build the merge as an explicit include/exclude decision with file-by-file rationale.
- Likely candidates to evaluate include:
  - validated workflow scripts,
  - necessary core-module improvements,
  - current docs where they genuinely help the public repo,
  - historical/provenance material only if it is worth carrying into `main`.

### Default exclusions unless there is a clear reason otherwise
- local noise / temp artifacts
- stale experimental clutter
- anything whose role in `main` is unclear or not yet justified

### Exit criteria
- Merge diff is explainable file-by-file, with explicit rationale for both what is included and what is left out.

---

## 4) Documentation roles

- `README.md` — repo-facing overview / user entrypoint
- `AGENTS.md` — canonical master plan and execution order (this file)
- `NEXT_STEPS.md` — actionable phase checklist
- `context/context_*.md` — timestamped session notes only

`CLAUDE.md` remains a compatibility shim pointing to `AGENTS.md`.

---

## 5) Definition of Done (project-level)

All of the following must hold:

1. Repressilator closeout complete (figure + caption + Results text + utility ownership pass).
2. stat_model paper example refreshed and manuscript-aligned.
3. Legacy public-repo examples compatible or explicitly archived.
4. Selective `revision1 -> main` merge complete with clean rationale.
5. Docs are consistent with actual code/repro state.
