# Context Checkpoint — Phase 3 keeper examples guided cleanup

## Goal
Start Phase 3 public-repo example cleanup in a guided/incremental way, focusing only on the keeper examples `examples/mm_model.jl` and `examples/transport_model.jl`.

## Current State
- Branch: `revision1`
- Remote sync before current doc edits: clean (`ahead/behind = 0/0`).
- Latest pushed work already includes the stat_model directional practical probe updates.
- Current uncommitted local changes are documentation only:
  - `AGENTS.md`
  - `NEXT_STEPS.md`
- No uncommitted code changes in the keeper examples themselves.

## Decisions (explicit)
1. Phase 3 scope is narrowed to the two public-facing keepers:
   - `examples/mm_model.jl`
   - `examples/transport_model.jl`
2. These are the exploratory/non-public examples for now (keep on `revision1`, do not prioritize for `main` public example set):
   - `examples/pk_model.jl`
   - `examples/stat_sum_model.jl`
3. Cleanup for Phase 3 should follow guided/incremental mode:
   - inspect,
   - discuss,
   - then edit one example at a time.
4. Treat Phase 3 as moderate cleanup, not just status-header polish.
5. Key technical questions to document/decide for keeper examples:
   - add explicit invariance-test logic (`find_invariant_subspace`) for both `examples/mm_model.jl` and `examples/transport_model.jl`, since current keeper scripts appear to rely mainly on SVD/rounding;
   - whether `examples/transport_model.jl` should use a Varimax-based interpretable basis in the public-facing version.

## Audit Findings
### `examples/mm_model.jl`
- Runs successfully with current code.
- Generates fresh figures under `figures/`.
- Uses saved reproducible data (good for public-facing example).
- Current mode in script is `limit = true`.
- Emits soft-scope warnings during execution, including:
  - `nuisance_guesses`
  - `all_inside`
- Current script appears to use SVD/rounding (`scale_and_round`) but not explicit `find_invariant_subspace`.

### `examples/transport_model.jl`
- Runs successfully with current code.
- Generates fresh figures under `figures/`.
- Uses seeded randomness (`Random.seed!(1)`) and currently does `data = rand(distrib_xy(xy_true), Nrep)`.
- Current script appears to use SVD/rounding (`scale_and_round`) but not explicit `find_invariant_subspace`.
- Likely candidate for Varimax-based interpretable basis in the public-facing version.
- Needs an explicit decision on whether to keep seeded random data or replace it with a fixed saved realization for public-facing reproducibility.

## Existing File Updates Made (not yet committed)
### `AGENTS.md`
Phase 3 now states:
- keeper examples = `mm_model`, `transport_model`
- `pk_model` and `stat_sum_model` are exploratory/non-public for merge planning
- cleanup should be guided/incremental
- explicit invariance-test decision should be made
- `transport_model` should be assessed for Varimax/interpretable basis use

### `NEXT_STEPS.md`
Phase 3 now includes:
- the narrowed keeper scope
- audit notes for `mm_model` and `transport_model`
- guided cleanup working mode
- explicit checklist item for invariance-test / interpretable-basis decision

## Next Steps
1. Commit the current doc-only updates (`AGENTS.md`, `NEXT_STEPS.md`, this context file if desired).
2. Start guided cleanup on `examples/mm_model.jl` first:
   - fix soft-scope warnings,
   - decide/add explicit invariance-test path if appropriate,
   - add clear status header / run instructions.
3. Then do guided cleanup on `examples/transport_model.jl`:
   - decide fixed saved realization vs seeded random data,
   - assess/add explicit invariance-test path,
   - assess Varimax-based interpretable basis,
   - add clear status header / run instructions.

## Key Files
- `AGENTS.md`
- `NEXT_STEPS.md`
- `examples/mm_model.jl`
- `examples/transport_model.jl`
- `context/context_20260310_233902_phase3_keeper_examples_guided_cleanup.md`

## Useful Commands
- Run keeper examples:
  - `julia --project=. "examples/mm_model.jl"`
  - `julia --project=. "examples/transport_model.jl"`
- Check current targeted changes:
  - `git status --short -- "AGENTS.md" "NEXT_STEPS.md" "examples/mm_model.jl" "examples/transport_model.jl" "context/context_20260310_233902_phase3_keeper_examples_guided_cleanup.md"`

## Continuation Prompt
"Continue Phase 3 in guided/incremental mode. Start with `examples/mm_model.jl`, using the documented audit findings: it runs but has soft-scope warnings and likely needs an explicit invariance-test decision/path. Then move to `examples/transport_model.jl`, including the reproducibility decision and possible Varimax/interpretable-basis update."