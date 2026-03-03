# NeSI Setup for IIR Profiling (Current Workflow)

**Updated:** 2026-02-24  
**Project Code:** `uoa04634`

This document reflects the **current** repressilator profiling workflow.

## Active scripts (use these)

- Main runner: `run_repressilator_profile.jl`
- NeSI submit scripts:
  - `nesi/submit_50x50.sl`
  - `nesi/submit_100x100.sl`
  - `nesi/submit_test_20x20.sl` (quick check)

## Superseded scripts (do not use)

- `iir_guided_profiling_18param.jl`
- `nesi/run_nesi_50x50.jl`
- `nesi/run_nesi_100x100.jl`

---

## 1) Local preparation

```bash
julia setup_project.jl
```

This ensures `Project.toml` / `Manifest.toml` are ready.

## 2) Upload to NeSI

Via OnDemand Files, upload/update:

- `Project.toml`, `Manifest.toml`
- `ReparamTools.jl`
- `core.jl`, `invariance.jl`, `parameterizations.jl`, `visualization.jl` (if changed)
- `run_repressilator_profile.jl`
- `examples/RepressilatorModel.jl`
- `nesi/submit_50x50.sl`, `nesi/submit_100x100.sl` (if changed)

Also upload any plotting/post-processing scripts you plan to run locally (e.g., `replot_profile_results.jl`, `repressilator_prediction_intervals_from_2d_profile.jl`) to keep versions synced.

## 3) Optional test job

```bash
cd /path/to/reparam
sbatch submit_nesi.sl
```

Check:

```bash
squeue --me
```

## 4) Run profiling jobs

```bash
cd /path/to/reparam/nesi
sbatch submit_test_20x20.sl   # quick smoke test
sbatch submit_50x50.sl        # publication-quality baseline
sbatch submit_100x100.sl      # higher resolution (longer runtime)
```

## 5) Outputs

Main result files are written in `nesi/`, e.g.

- `repressilator_16nuisance_50x50_results.jls`
- `repressilator_16nuisance_100x100_results.jls`

Download `.jls` outputs and plot **locally** (standard workflow).

---

## Typical local post-processing

```bash
julia --project=. replot_profile_results.jl nesi/repressilator_16nuisance_50x50_results.jls
julia --project=. repressilator_prediction_intervals_from_2d_profile.jl nesi/repressilator_16nuisance_50x50_results.jls
```

---

## Notes

- NeSI runs are for profiling compute; plotting is usually done locally.
- Keep commits/file uploads targeted to changed files only.
- Prefer the `nesi/submit_*.sl` scripts over older root-level submit/runner files.
