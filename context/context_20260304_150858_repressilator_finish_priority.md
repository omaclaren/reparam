# Context Checkpoint — Repressilator Finish Priority

## Goal
Finish the in-progress repressilator closeout first, then proceed to broader repo tasks.

## Current State
- Branch: `revision1`
- Repressilator workflow already stabilized in prior commits (slice/profile only; hybrid removed; 3-way prediction comparison established).
- User corrected priority: do **not** shift active effort away from repressilator until repressilator closeout is complete.
- Docs were updated locally to reflect this priority and phase gating:
  - `AGENTS.md`
  - `NEXT_STEPS.md`
  - `README.md`

## Decisions (explicit)
1. **Phase 1 (blocking):** Repressilator closeout only.
2. **Phase 2:** stat_model refresh (paper example).
3. **Phase 3:** legacy example compatibility for public repo (non-paper).
4. **Phase 4:** `revision1 -> main` merge decided jointly, file-by-file; no automatic inclusion assumptions.
5. Keep summaries in chat when asked; do not assume file edits unless requested.

## Next Steps
1. Commit and push the current doc alignment changes + this context file.
2. Execute Phase 1 checklist only:
   - verify final repressilator 3-way figure path/output,
   - finalize caption,
   - finalize short Results paragraph,
   - complete utility ownership walkthrough for:
     - `run_repressilator_profile.jl`
     - `replot_profile_results.jl`
     - `extract_iir_diagnostics_from_results.jl`
     - `repressilator_prediction_intervals_from_2d_profile.jl`

## Key Files
- `AGENTS.md`
- `NEXT_STEPS.md`
- `README.md`
- `nesi/repressilator_16nuisance_50x50_results.jls`
- `run_repressilator_profile.jl`
- `replot_profile_results.jl`
- `extract_iir_diagnostics_from_results.jl`
- `repressilator_prediction_intervals_from_2d_profile.jl`

## Useful Commands
- Check current doc/context status:
  - `git status --short -- "AGENTS.md" "NEXT_STEPS.md" "README.md" "context"`
- Commit/push doc alignment:
  - `git add "AGENTS.md" "NEXT_STEPS.md" "README.md" "context/context_20260304_150858_repressilator_finish_priority.md"`
  - `git commit -m "Refocus docs on repressilator-first closeout and save context checkpoint"`
  - `git push`

## Continuation Prompt
"Continue from Phase 1 only: finish repressilator closeout deliverables (final 3-way figure verification, caption, short Results paragraph, and active utility ownership pass). Do not start stat_model or legacy-example work yet."