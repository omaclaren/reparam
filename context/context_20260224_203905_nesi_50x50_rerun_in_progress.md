# Session Context - 2026-02-24 20:39

## Goal
Run canonical paper rerun on NeSI after integer-exponent `scale_and_round` fix, then regenerate diagnostics/predictions from the new 50x50 artifact.

## Current State
- Branch: `revision1`
- Latest pushed commit: `b3db4ba` (submit_50x50 walltime increased to 6h)
- Key fix already pushed earlier: `0260880` (`scale_and_round` integer exponent behavior + tolerance checks)
- NeSI layout confirmed: `/home/omac010/reparam` (with `examples/` subdir)

## NeSI Run In Progress
- Host/session: `login03` (Mahuika shell)
- Submitted command:
  - `sbatch submit_50x50.sl`
- Job ID: `4725448`
- Current queue status at submit check:
  - `PENDING (Priority)`
  - request: 72 CPUs, 64G, 6:00:00 walltime

## Files synced/uploaded for run
- `/home/omac010/reparam/run_repressilator_profile.jl`
- `/home/omac010/reparam/core.jl`
- `/home/omac010/reparam/invariance.jl`
- `/home/omac010/reparam/parameterizations.jl`
- `/home/omac010/reparam/ReparamTools.jl` (safety)
- `/home/omac010/reparam/submit_50x50.sl`
- `/home/omac010/reparam/examples/RepressilatorModel.jl`

## Expected Outputs
- `repressilator_16nuisance_50x50_results.jls`
- `iir_50x50_4725448.out`
- `iir_50x50_4725448.err`

## Next Steps (after completion)
1. Download new 50x50 `.jls` and logs.
2. Regenerate:
   - `extract_iir_diagnostics_from_results.jl`
   - `replot_profile_results.jl`
   - `compute_prediction_intervals.jl`
3. Compare old vs new 50x50 diagnostics/profile summaries (rank, singular values/gap, identifiable ranking, target combos, profile behavior).

## Continuation Prompt
"Resume from `context/context_20260224_203905_nesi_50x50_rerun_in_progress.md`. Check NeSI job 4725448 completion, then run local postprocessing and old-vs-new 50x50 comparison for paper-facing outputs."