# Session Context - 2026-02-24 20:48

## Goal
Complete the canonical paper rerun (repressilator, 16 nuisance, 50x50) after the integer-exponent `scale_and_round` fix, then regenerate and compare paper-facing outputs.

## Current State (authoritative)
- Branch: `revision1`
- Remote: `origin/revision1` up to date
- Latest commits:
  - `b3db4ba` Increase NeSI 50x50 walltime to 6 hours
  - `94ab698` Update NeSI 50x50 submit settings and minor script cleanups
  - `0260880` Fix `scale_and_round` to integer exponents + tolerance checks

## Key Technical Fix Already Landed
- `parameterizations.jl` `scale_and_round` now rounds significant coefficients to nearest **integer** and errors if outside tolerance.
- Local validation at saved 50x50 MLE passed:
  - nonzero coefficients in `A_T_final` are `±1`
  - no residual `±1.05` issue at that checkpoint.

## NeSI Canonical Run Status
- Job submitted on Mahuika from `~/reparam`:
  - `sbatch submit_50x50.sl`
- Job ID: `4725448`
- Last seen status at submission check: `PENDING (Priority)`
- Requested resources: 72 CPUs, 64G, 6:00:00

## NeSI Directory/Workflow Facts (important)
- NeSI project root is `/home/omac010/reparam` (not local mac path).
- Submit script is used from root: `submit_50x50.sl`.
- `examples/RepressilatorModel.jl` is under `/home/omac010/reparam/examples/`.
- User runs NeSI manually via OnDemand website + shell.

## Files intended for this rerun (already synced)
- `/home/omac010/reparam/run_repressilator_profile.jl`
- `/home/omac010/reparam/ReparamTools.jl`
- `/home/omac010/reparam/core.jl`
- `/home/omac010/reparam/invariance.jl`
- `/home/omac010/reparam/parameterizations.jl`
- `/home/omac010/reparam/submit_50x50.sl`
- `/home/omac010/reparam/examples/RepressilatorModel.jl`

## Expected Output Files
- `repressilator_16nuisance_50x50_results.jls`
- `iir_50x50_4725448.out`
- `iir_50x50_4725448.err`

## Immediate Next Steps (when resuming)
1. Check job state:
   - `squeue --me`
   - `sacct -j 4725448 --format=JobID,State,Elapsed,MaxRSS`
2. If running:
   - `tail -f iir_50x50_4725448.out`
3. When complete, download new `.jls` + logs to local `nesi/`.
4. Local postprocessing:
   - `julia --project=. replot_profile_results.jl nesi/repressilator_16nuisance_50x50_results.jls`
   - `julia --project=. compute_prediction_intervals.jl nesi/repressilator_16nuisance_50x50_results.jls`
   - `julia --project=. extract_iir_diagnostics_from_results.jl nesi/repressilator_16nuisance_50x50_results.jls`
5. Compare old vs new 50x50 on:
   - rank
   - singular values + gap
   - identifiable ranking (`sigma_eff`)
   - target combo identification (`K1/β1`, `β1*K1`)
   - 1D profile summary behavior.

## Guardrails / Constraints
- No method changes unless explicitly agreed.
- File-targeted commits only (avoid broad staging due local `.julia` noise).
- `revision1` remains workbench; merge-to-main scope still undecided.
- Keep code simple/readable/owner-friendly.

## Broader Deferred Work (keep in view)
- Additional 1D identifiable-combo profiling (beyond current pair) and updated union-over-available intervals.
- Manuscript/reviewer integration updates after canonical rerun artifacts are confirmed.

## Continuation Prompt
Resume from `context/context_20260224_204800_fresh_session_resume_master.md` on `revision1`.
First check NeSI job `4725448`. If complete, run local postprocessing + old-vs-new 50x50 comparison and report only verified differences.