# CURRENT FOCUS (Session Guardrail)

**Last updated:** 2026-03-03
**Project:** `reparam`

## Active Mode (authoritative)
Guided **ownership walkthrough** of:
- `compute_prediction_intervals.jl`

Primary goal: help Oliver gain full understanding/ownership.

## Non-negotiable constraints
1. **No behavior/code changes** unless explicitly requested.
2. Walk through script **section-by-section**, with purpose + assumptions + outputs.
3. Keep discussion tied to current paper/revision context (repressilator 50×50 canonical result).
4. If docs conflict, prioritize latest `context/context_*.md` notes over older planning docs.

## Session startup checklist (fresh sessions)
Before doing any edits/commands beyond reads:
1. Read latest `context/context_*.md`.
2. Read `context/CURRENT_FOCUS.md`.
3. Restate in 2–4 bullets:
   - current goal,
   - constraints,
   - immediate next step,
   - whether edits are allowed.
4. Ask for confirmation before modifying files.

## Current immediate next step
Start walkthrough at top of `compute_prediction_intervals.jl`:
- script contract,
- required keys in results file,
- ψ layout reconstruction logic and why it matters.

## Deferred cleanup task (recorded)
- Hybrid mode (`--mode=hybrid`) was private experimental work and is **low-priority to remove directly** (not a public deprecation workflow).
- Planned cleanup (when time allows):
  1. remove hybrid mode branch from `run_repressilator_profile.jl`,
  2. remove hybrid mentions from usage/comments/docs,
  3. mark `wald_profile_comparison.jl` + `repressilator_hybrid_*.jls` as archival/delete,
  4. simplify post-processing assumptions/loaders accordingly.
