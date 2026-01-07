# Profile Likelihood Runs Summary

Summary of profile likelihood demonstrations for IIR paper revision.

## Nuisance Parameter Progression

The goal is to show that IIR-identified structure (peaked vs flat profiles) is robust as we add nuisance parameters.

### 1. No Nuisance (2D grid evaluation)

| Script | Result | Description |
|--------|--------|-------------|
| `minimal_2D_IIR_coords.jl` | `minimal_2D_IIR_coords_result.png` | Pure 2D grid in IIR coords (K/β, β·K), no optimization |
| `minimal_2D_reparam.jl` | `minimal_2D_reparam_result.png` | Similar, with 1D profiles |

**Key result**: Perfectly smooth contours. Clear vertical band (flat in β·K, peaked in K/β).

### 2. Minimal Nuisance (1 parameter)

| Script | Result | Description |
|--------|--------|-------------|
| `minimal_2D_IIR_nuisance.jl` | `minimal_2D_IIR_nuisance_result.png` | 2D profile with β₂ optimized as nuisance |
| `minimal_2D_1nuisance.jl` | `minimal_2D_1nuisance_result.png` | Similar approach |

**Key result**: Still smooth. Structure preserved with optimization.

### 3. Small Nuisance (4-6 parameters)

| Script | Result | Description |
|--------|--------|-------------|
| `iir_guided_profiling.jl` | `iir_guided_profiling_result.png` | 6-param subset: β₁,K₁ + 4 nuisance (ψ_2,3,4,6) |
| `minimal_2D_3param.jl` | `minimal_2D_3param_result.png` | 3 param exploration |
| `minimal_2D_6param.jl` | `minimal_2D_6param_result.png` | 6 param exploration |

**Key result**: Smooth profiles. Demonstrates IIR works with moderate nuisance parameters.

### 4. Full Model (16 nuisance parameters)

| Script | Result | Description |
|--------|--------|-------------|
| `iir_guided_profiling_18param.jl` | `iir_guided_profiling_18param_result.png` | Local 50×50, 16 nuisance |
| `run_18param_profile.jl` | (configurable) | Unified local/NeSI script |
| NeSI 50×50 | `nesi/iir_50x50_result.png` | 50×50, improved optimizer |
| NeSI 100×100 | `nesi/iir_100x100_result.png` | 100×100, original optimizer |

**Key result**: Jagged profiles due to optimizer noise in 16D, but structure (peaked vs flat) preserved.

**Replotted with θ-space**:
- `nesi/iir_50x50_replot.png` - 4-panel with θ-space diagonal ridge
- `nesi/iir_100x100_replot.png` - 4-panel with θ-space diagonal ridge

## Scripts Reference

### Production Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `run_18param_profile.jl` | Unified 18-param profiling | `julia run_18param_profile.jl --grid=50 --workers=7` |
| `replot_profile_results.jl` | Regenerate 4-panel plots from .jls | `julia replot_profile_results.jl results.jls` |

### Development/Calibration Scripts

| Script | Purpose |
|--------|---------|
| `minimal_2D_IIR_coords.jl` | Calibrate 2D grid evaluation (ideal case) |
| `minimal_2D_IIR_nuisance.jl` | Test nuisance parameter handling |
| `iir_guided_profiling.jl` | 6-param subset profiling |
| `iir_guided_profiling_18param.jl` | Original 18-param (local version) |

### NeSI Scripts (in `nesi/` folder)

| Script | Purpose |
|--------|---------|
| `run_nesi_50x50.jl` | 50×50 test run with improved optimizer |
| `run_nesi_100x100.jl` | 100×100 production run |
| `submit_50x50.sl` | SLURM submission for 50×50 |
| `submit_100x100.sl` | SLURM submission for 100×100 |

## Key Findings

### Profile Structure
- **Identifiable (K₁/β₁)**: U-shaped profile, bounded confidence interval
- **Non-identifiable (β₁·K₁)**: Flat profile, unbounded confidence interval

### Computational Notes
- No nuisance: ~seconds (pure grid evaluation)
- 1-4 nuisance: ~minutes (quick optimization)
- 16 nuisance: ~hours (complex optimization, parallelization needed)
- Jaggedness in full model is optimizer noise, not structural

### θ-space vs ψ-space
- ψ-space (IIR): Vertical band (flat in β·K direction)
- θ-space (original): Diagonal ridge along K/β = constant

## Optimizer Settings

| Setting | Default | Improved (NeSI) | Notes |
|---------|---------|-----------------|-------|
| n_extra_guesses | 7 | 15 | More restarts → smoother |
| optmaxtime | 90s | 150s | More time per point |
| grid | 50-100 | 100 | Higher resolution |

## File Outputs

All profile runs save:
- `iir_NxN_results.jls` - Serialized results (can replot later)
- `iir_NxN_result.png` - Basic 3-panel plot
- Use `replot_profile_results.jl` for full 4-panel with θ-space
