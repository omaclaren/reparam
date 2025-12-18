# NeSI Setup for IIR Profiling

**Date:** 2025-12-18
**Project Code:** uoa04634

## Status

Local 50x50 profiling completed successfully. Results confirm IIR analysis:
- K1/B1 (identifiable): 37/50 peaked profile
- B1*K1 (non-identifiable): 50/50 flat profile

Current results have some artifacts - need finer grid and more optimizer restarts for publication quality.

## NeSI Setup Steps

### 1. Local Preparation (DONE)
```bash
julia setup_project.jl
```
This adds all dependencies to Project.toml and generates Manifest.toml.

### 2. Upload to NeSI
Via ondemand.nesi.org.nz Files dashboard, upload:
- `Project.toml`
- `Manifest.toml`
- `nesi_test.jl`
- `submit_nesi.sl`
- `iir_guided_profiling_18param.jl`
- `ReparamTools.jl`
- `examples/RepressilatorModel.jl`
- (and any other needed files)

### 3. Submit Test Job
```bash
cd /path/to/reparam
sbatch submit_nesi.sl
```

### 4. Check Status
```bash
squeue --me
```

### 5. View Output
Look for `nesi_test_XXXX.out` file.

## Files for NeSI

Key files to upload:
- `Project.toml`, `Manifest.toml` - dependencies
- `nesi_test.jl` - test script
- `submit_nesi.sl` - SLURM submission script
- `iir_guided_profiling_18param.jl` - main profiling script
- `ReparamTools.jl` - core module
- `examples/RepressilatorModel.jl` - model definition

## Next Steps for Publication-Quality Results

1. Run on NeSI with more cores (e.g., 32+)
2. Increase grid to 75x75 or 100x100
3. Increase `n_extra_guesses` (more optimizer restarts)
4. Consider longer `optmaxtime` per point

## Current Local Results

50x50 grid, 7 workers, ~2.5h compute time (excluding laptop suspend):
- Confirms IIR-identified structure
- K1/B1 peaked (identifiable)
- B1*K1 flat (non-identifiable)
- Plot: `iir_guided_profiling_18param_result.png`
