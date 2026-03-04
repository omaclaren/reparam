# CURRENT FOCUS (Session Guardrail)

**Last updated:** 2026-03-04
**Project:** `reparam`

## Active Mode (authoritative)
Repressilator post-processing clarity pass:
- script renamed to `repressilator_prediction_intervals_from_2d_profile.jl`
- prediction comparison now emphasizes:
  1. full accepted 2D pushforward,
  2. 1D identifiable-profile band,
  3. 1D non-identifiable-profile band.

Primary goal: keep the script simple, explicit, and paper-aligned.


## Non-negotiable constraints
1. Keep this as a **specialized repressilator utility**, not a fake-general tool.
2. No unnecessary abstraction/overengineering.
3. Keep likelihood-threshold assumptions explicit (`df = rank_J`).
4. If docs conflict, prioritize latest `context/context_*.md` notes over older planning docs.

## Session startup checklist (fresh sessions)
Before doing edits/commands beyond reads:
1. Read latest `context/context_*.md`.
2. Read `context/CURRENT_FOCUS.md`.
3. Restate in 2–4 bullets:
   - current goal,
   - constraints,
   - immediate next step,
   - whether edits are allowed.
4. Ask for confirmation before modifying files.

## Current immediate next step
1. Confirm figure/story choice for paper:
   - show full 2D vs identifiable vs non-identifiable,
   - leave union out of main text.
2. Decide whether any extra convenience wrappers are needed (likely no).

## Deferred cleanup task (recorded)
- Hybrid mode (`--mode=hybrid`) was private experimental work.
- ✅ Completed now:
  1. removed hybrid branch from `run_repressilator_profile.jl`,
  2. updated active usage/docs to slice/profile workflow.
- ✅ Completed now:
  1. archived `wald_profile_comparison.jl`, `hybrid_profile_likelihood_summary.md`,
     `repressilator_hybrid_*.jls`, and related hybrid PNGs to `archive/hybrid/`.
- Remaining low-priority cleanup:
  1. prune stale hybrid notes in legacy context/docs if desired.
