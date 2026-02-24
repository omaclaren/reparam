# Session Context - 2026-02-24 15:30

## Goal
Preserve the **actual current state** before pause: key work exists on `revision1`, but merge scope and packaging for `main` are still **explicitly undecided**.

## Current State (verified)
- Branch: `revision1`
- Recent commits include:
  - `9df14d7` Docs update for current NeSI runner workflow
  - `aeea69b` Added diagnostics extractor script + outputs
  - `d1ac42f` Clarified diagnostics notes and 50x50 paper-focus wording
- Paper profile run currently treated as the publication candidate: `nesi/repressilator_16nuisance_50x50_results.jls`

## Explicitly NOT Decided Yet (important)
1. What exactly should be merged/cherry-picked into `main`.
2. Whether `nesi/` content should be in `main` (and if so, how much).
3. Whether paper artifacts (`.jls`, generated `.csv/.md/.png`) live in-repo vs release assets vs both.
4. Which scripts are `main`-ready vs `revision`-only.

## Style / Ownership Requirements (user-stated)
- Code for eventual `main` should be:
  - simple, human-readable, in the user's own style
  - reusing existing library functionality where possible
  - reproducible without over-engineered scaffolding
- Avoid AI-style one-off complexity that increases ownership burden.

## Working Principle Going Forward
- Do **not** make unilateral packaging/merge decisions before discussion.
- Treat `revision1` as active workbench until merge scope is agreed.
- Before further edits, first settle merge/share strategy at high level.

## Next Step When Resuming
1. Decide merge policy (what belongs in `main` vs stays on `revision1`).
2. Then apply targeted simplifications/curation to match ownership criteria.
3. Use file-targeted commits only.

## Continuation Prompt
"Resume from `context/context_20260224_153041_merge_scope_pending.md`. First decide merge/share scope (main vs revision artifacts/scripts) before any code changes. Keep decisions explicit and avoid introducing additional complexity."
