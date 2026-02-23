# Project Status: Invariant Image Reparameterisation (IIR)

**Last Updated:** 2025-12-10
**Phase:** Paper Revision - Profile Likelihood Demonstration In Progress

## Overview

The IIR paper has been through peer review at SIAM/ASA Journal on Uncertainty Quantification. We are implementing changes to address reviewer feedback. The core contribution is now clearly defined: **a single-stage numerical method for discovering image reparameterizations** that separates identifiable from non-identifiable parameter combinations using monomial transformations.

## Strategic Focus: Single-Stage IIR with Monomials

### Core Method
- **Transformation**: ψ(θ) = exp(A log(θ)) where A comes from Algorithm 1
- **Key innovation**: Numerical invariance test (Hessian-based) identifies which directions in parameter space have globally invariant null spaces
- **Output**: Clean separation between identifiable (N_perp) and non-identifiable (N) combinations
- **Enhancement**: Varimax rotation provides interpretable basis within span(N_perp)

### Why Single-Stage Focus?

**Single-stage IIR works well in tested cases:**
- ✅ Algorithm 1 identifies correct rank/invariant subspace reliably
- ✅ Works across diverse model types (statistical, ODE systems)
- ✅ Varimax rotation produces interpretable monomial combinations
- ✅ Clear theoretical foundation
- ✅ No basis-dependence issues

**Sequential/multi-stage IIR is fragile:**
- ⚠️ Success depends on intermediate basis "aligning" with final target combinations
- ⚠️ Varimax optimizes sparsity, not compositional reducibility
- ⚠️ Opens complicated questions about optimal basis selection (open research problem)

**Decision**: Focus paper on single-stage method. Multi-stage briefly mentioned in future work with honest assessment of limitations.

## Examples for Paper

### 1. stat_model.jl ✅ (Pedagogical)
**Purpose**: Simple, clear demonstration of IIR basics

**Model**: Poisson limit distribution
- Parameters: θ = [n, p]
- Auxiliary mapping: ϕ(n,p) = [np, np]
- Data: Y ~ Poisson(np)

**Results**:
- Rank: 1/2 (one identifiable combination)
- Identifiable: ψ₁ = np
- Non-identifiable: ψ₂ = n/p
- Clean 2×2 transformation: [1,1; 1,-1] in log space

**Status**: Complete and verified

### 2. Repressilator (Ambitious) - In Progress
**Purpose**: Demonstrate IIR on realistic mechanistic ODE model (reviewer request)

**Model**: Eisenberg & Hayashi (2010) 3-gene repressilator
- Parameters: 18 (fixing n=2.5 from original 19)
- System: 6 coupled nonlinear ODEs (stiff)
- Observables: All 3 mRNA time series

**IIR Analysis Results** ✅:
- Rank: 15/18 (3 non-identifiable directions)
- Invariant null space: βK products (3-dimensional)
- Identifiable combinations: K/β ratios (up to sign/reciprocal ambiguity)
- Hessian-based invariance test works for stiff ODEs (tested on repressilator)

**Profile Likelihood Demonstration** ✅ Complete:
Shows that profiling IIR-identified combinations (K₁/β₁ identifiable, β₁K₁ non-identifiable) reveals the true identifiability structure. **50×50 grid selected for publication** (cleaner than 100×100 which had optimizer artifacts in low K₁/β₁ region).

**Development Strategy** (incremental complexity):
1. ✅ 2D ideal case (`minimal_2D_IIR_coords.jl`) - calibrate optimizer, understand expected behavior
2. ✅ 2D + 1 nuisance (`minimal_2D_IIR_nuisance.jl`) - verify nuisance handling
3. ✅ 6 params: β, K subset (`iir_guided_profiling.jl`) - working
4. ✅ 18 params: full model IIR (`iir_guided_profiling_18param.jl`) - **IIR complete, profiling next**

**18-Parameter IIR Results** (2025-12-10):
- Rank: 15/18, gap 723,000× (unambiguous)
- Identifiable (15): K₁/β₁, K₂/β₂, K₃/β₃, plus 12 individual params (α₀, α, k_degm, k_degp)
- Non-identifiable (3): β₁K₁, β₂K₂, β₃K₃
- Gene 1 coordinates found: ψ₆ = K₁/β₁ (identifiable), ψ₁₇ = β₁K₁ (non-identifiable)
- Key settings: fine time grid (501 pts) + tight ODE tolerances (abstol=1e-10, reltol=1e-8) for IIR; observation grid (8 pts) for likelihood

**Next step**: Add 2D profiling over gene 1 coordinates (K₁/β₁ vs β₁K₁) with 16 nuisance parameters

**Key files**:
- `examples/repressilator.jl` - full IIR analysis (reference)
- `examples/RepressilatorModel.jl` - model definition
- `iir_guided_profiling.jl` - 6-param subset profiling
- `iir_guided_profiling_18param.jl` - **18-param IIR complete, profiling to be added**
- `minimal_2D_IIR_coords.jl`, `minimal_2D_IIR_nuisance.jl` - calibration scripts

**Narrative** (target for paper):
1. **Discovery**: IIR automatically identifies K/β structure without symbolic computation
2. **Problem**: Profiling individual parameters (K₁, β₁) gives misleading narrow uncertainty
3. **Solution**: Profiling identifiable ratio (K₁/β₁) provides honest prediction intervals

## Key Documents

- **Revised paper**: `/Users/omac010/Git-Working/overleaf_projects/invariant-image-reparameterisation/arXiv/Maclaren_IIR2025.tex`
- **Original submission**: `/Users/omac010/Dropbox/research/manuscripts/01-submitted/iir-arxiv/arXiv/Maclaren_IIR2025.tex`
- **Reviewer comments**: `/Users/omac010/Dropbox/research/manuscripts/03-revising/iir/iir-reviews.pdf`

## Core Implementation

### invariance.jl
**Function**: `find_invariant_subspace(ϕ_func, θ0; compute_J, rtol_rank, rtol_invariance)`

**Algorithm**:
1. Compute Jacobian J at θ0 using automatic differentiation
2. Determine rank via SVD with relative tolerance rtol_rank
3. Extract null space candidates from right singular vectors
4. Test each null vector for invariance using Hessian-based criterion
5. Separate N (invariant) from N_perp (potentially identifiable)

**Parameters**:
- `rtol_rank = 1e-8`: Relative tolerance for Jacobian rank
- `rtol_invariance = 1e-6`: Relative tolerance for invariance test
  - Effective threshold: τ_inv = rtol_invariance * σ_max (scales with Jacobian magnitude)
  - May need adjustment for different problem types; use verbose=true to check classification

**Returns**: `(S, N, N_perp, rank_J)` where
- S: Singular values
- N: Invariant null space (non-identifiable directions)
- N_perp: Complement (potentially identifiable directions)
- rank_J: Numerical rank of Jacobian

### Transformation Convention

**Full reparameterization**: ψ(θ) = f⁻¹(A f(θ))

For monomials (f = log):
```julia
# Build transformation matrix (rows = parameter combinations)
A_full_T = hcat(N_perp, N)           # Stack as columns
A_full_T_scaled = scale_and_round(A_full_T)  # Integer coefficients
A_full = A_full_T_scaled'            # Transpose to get row combinations

# Apply transformation
ψ(θ) = exp.(A_full * log.(θ))
```

**Critical**: Scale/round on COLUMNS (basis vectors) before transposing. Scaling rows corrupts the parameter combinations.

### Varimax Rotation (Optional Enhancement)

**Purpose**: Improve interpretability of identifiable combinations within span(N_perp)

For the current single-stage paper focus, use this mainly as an interpretation/presentation aid rather than a required algorithmic step.

**Implementation** (parameterizations.jl):
```julia
N_perp_rotated = varimax_rotation(N_perp; n_restarts=200)
```

**When to use**:
- High-dimensional N_perp where interpretation is difficult
- Use rotated basis for display/interpretation
- Keep orthonormal for transformation (numerical stability)

**When to skip**:
- Low-dimensional problems (2-3 parameters)
- Already sparse/interpretable structure
- Final stage of analysis (raw SVD sufficient)

## Repository Structure

```
reparam/
├── ReparamTools.jl          # Main module file
├── invariance.jl             # Algorithm 1 implementation ✅
├── core.jl                   # Profile likelihood, optimization
├── utils.jl                  # Helper functions
├── parameterizations.jl      # Transformations, Varimax rotation
├── visualization.jl          # Plotting functions
├── iir_guided_profiling.jl   # 🔄 Active: 6-param profiling development
├── minimal_2D_IIR_coords.jl  # ✅ 2D calibration (ideal case)
├── minimal_2D_IIR_nuisance.jl # ✅ 2D + nuisance calibration
└── examples/
    ├── stat_model.jl         # ✅ Pedagogical example (complete)
    ├── repressilator.jl      # IIR analysis complete, profiling in progress
    ├── RepressilatorModel.jl # Model definition
    ├── stat_sum_model.jl     # Sequential IIR (works, but not in paper)
    ├── pk_model.jl           # Sequential IIR (reveals limitations)
    ├── mm_model.jl           # Legacy (needs update)
    └── transport_model.jl    # Legacy (needs update)
```

## Technical Implementation Notes

### 1. Tolerance Selection

**Jacobian rank (rtol_rank)**: Relative tolerance
- Default: 1e-8
- Scales with problem magnitude
- Threshold: `τ = rtol_rank * σ_max`

**Invariance test (rtol_invariance)**: Relative tolerance
- Default: 1e-6
- Effective threshold: τ_inv = rtol_invariance * σ_max (scales with Jacobian magnitude)
- May need adjustment for different problem types; use verbose=true to check classification

### 2. Invariance Method

**Hessian-based** (only method):
- Uses nested automatic differentiation (ForwardDiff)
- Efficiently computes Hessian-vector products: differentiates J(θ)*V_0 instead of full J(θ)
- Tested on smooth (stat_model) and stiff ODE (repressilator) systems

### 3. Matrix Construction

**Always follow this order**:
1. Column-stack: `A_T = hcat(N_perp, N)`
2. Scale columns: `A_T_scaled = scale_and_round(A_T)`
3. Transpose: `A = A_T_scaled'`

**Never** scale after transposing - it corrupts the parameter combinations.

### 4. Practical Non-Identifiability

Even when rank = p (full rank), **always compute and report**:
- Condition number: σ_max/σ_min
- Ranking of combinations by singular values
- IIR reparameterization still valuable for separating well-identified from poorly-identified combinations

### 5. Coordinate Spaces and Bounds

**Computation vs Visualization Spaces**

1. **Computation is done in ψ-space (IIR coordinates)**
   - Grid is log-uniform over target coordinates (e.g., ψ₇=K₁/β₁, ψ₁₈=β₁·K₁)
   - Nuisance optimization also in ψ-space
   - Natural space: rectangular grid, separates identifiable/non-identifiable

2. **θ-space visualization is a transform**
   - Transform back via: K₁ = √(ψ₇·ψ₁₈), β₁ = √(ψ₁₈/ψ₇)
   - Regular ψ grid → hyperbolic pattern in θ-space
   - Shows likelihood surface in original parameters

**Bounds Hierarchy**

```
θ profile bounds (user-specified, e.g., K≤100, β≤0.5)
    ↓ Monte Carlo sampling
ψ bounds (derived, define 2D target grid)
    ↓ inverse transform
θ plotting bounds (should match θ profile bounds)
```

**Key insight**: Wide ψ bounds needed to cover θ-space adequately due to nonlinear transform. But some ψ grid points map to θ values outside intended region.

**Example** (repressilator gene 1):
- θ profile bounds: K≤100, β≤0.5
- ψ bounds derived: ψ₇∈[7, 63500], ψ₁₈∈[0.02, 60]
- But max K from ψ grid: √(63500×60) ≈ 1950 (far exceeds K≤100)

**Practical approach**:
- Tighter θ profile bounds → tighter ψ bounds → less wasted computation
- Clip θ-space scatter plots to profile bounds for visualization
- RBF interpolation fills gaps for contour plots

**Plotting scripts**:
- `replot_profile_results.jl` - contourf with RBF interpolation (publication quality)
- `replot_scatter.jl` - scatter showing actual computed points (diagnostic)

## Investigation History (For Reference)

### Sequential IIR Exploration (Not in Paper)

**Successes**:
- `stat_sum_model.jl`: Sum of independent Poisson limits
  - Stage 1 (f=log): Finds products n₁p₁, n₂p₂
  - Stage 2 (f=identity): Finds sum n₁p₁ + n₂p₂
  - Works because structure naturally aligns

**Limitations discovered**:
- `pk_model.jl`: 2-compartment pharmacokinetic model
  - Stage 1 correctly identifies rank 6/8
  - Varimax produces interpretable monomials
  - BUT: Target identifiable combinations (k₀₂+k₁₂, etc.) become nonlinear functions of Stage 1 coordinates
  - Stage 2 with f=identity cannot recover them
  - **Root cause**: Varimax optimizes sparsity, not compositional reducibility

**Conclusion**: Multi-stage IIR reveals an open research problem (basis selection for compositional reduction). Single-stage method works well for tested cases.

## Paper Positioning

### Main Text: Two Examples

1. **stat_model.jl**: Clear pedagogical introduction
   - Shows basic IIR workflow
   - Demonstrates minimal image reparameterization
   - Easy to understand and verify

2. **repressilator.jl**: Ambitious mechanistic demonstration
   - Addresses reviewer request for "more compelling examples"
   - 18 parameters, realistic ODE system
   - Validates against established results (Eisenberg 2010)
   - Shows practical importance via prediction uncertainty

### Future Work Section

Brief mention of multi-stage possibilities with honest assessment:
> "The single-stage method can be extended to sequential application for discovering compositional structure (e.g., products followed by sums). This works when intermediate bases naturally align with target combinations, but basis selection for optimal compositional reduction remains an open problem. Our investigation of a pharmacokinetic model revealed that standard rotation criteria (Varimax) optimize interpretability but not reducibility across stages."

## Immediate Tasks

### Profile Likelihood Demonstration ✅ Complete
- [x] 2D ideal case working (`minimal_2D_IIR_coords.jl`)
- [x] 2D + nuisance working (`minimal_2D_IIR_nuisance.jl`)
- [x] 6-param subset profiling working (`iir_guided_profiling.jl`)
- [x] 18-param full model profiling (`nesi/repressilator_16nuisance_100x100_results.jls`)
- [x] snake_direction optimization validated (54× smoothness improvement)
- [ ] Generate comparison figure: original params vs IIR coords (for paper)

### Practical Challenge: Basis Ambiguity
IIR identifies the correct *subspace* but the basis vectors have ambiguity:
- Sign flips (K/β vs β/K)
- Reciprocals (K/β vs K·β could be confused without careful interpretation)

For paper figures, need to manually select which IIR coordinates correspond to gene 1's β and K to make an interpretable comparison. This is a presentation issue, not an algorithm issue.

### For Manuscript
- [x] stat_model complete ✅
- [x] Repressilator profile computation complete (100×100 grid)
- [ ] Note in paper: τ_inv = rtol_invariance × σ₁ scales the invariance test relative to J's first-order signal. This is motivated by perturbation theory — we're checking if second-order leakage out of the null space is negligible compared to the range of J. Scaling by σ₁(M_test) instead would fail when the entire null space is invariant (noise vs noise).
- [ ] Repressilator profile demonstration figure (format for paper)
- [ ] Update Methods section to match invariance.jl implementation
- [ ] Write Results section highlighting both examples

### For Reviewer Response
- [ ] Draft reviewer response for single-stage focus
- [ ] Highlight repressilator example
- [ ] Explain focus on single-stage over multi-stage

## Questions Resolved

1. ~~Which ambitious example?~~ → Repressilator
2. ~~Single vs multi-stage focus?~~ → Single-stage
3. ~~How to handle sequential IIR findings?~~ → Brief mention in future work
4. ~~Development strategy for profiling?~~ → Incremental: 2D → 2D+nuisance → 6-param → 18-param

## Computational Resources (for Paper)

### Profile Likelihood Computation

**Hardware**: NeSI (New Zealand eScience Infrastructure) HPC cluster. See `NESI_CHEATSHEET.md` for workflow.
- Node: Milan compute nodes
- CPUs per job: 72 (71 workers + 1 coordinator)
- Julia parallelization: Distributed.jl with `pmap`

**Run Times** (18-parameter repressilator, 16 nuisance profiled):

| Grid | Points | Profiling Time | Wall Time | Workers | Bounds | Notes |
|------|--------|----------------|-----------|---------|--------|-------|
| 10×10 | 100 | 22.1 min | ~26 min | 7 (local) | K≤100 | |
| 20×20 | 400 | 20.4 min | ~25 min | 71 (NeSI) | K≤100 | |
| 50×50 | 2,500 | ~2 hr | ~2 hr | 71 (NeSI) | K≤100 | **Publication quality** |
| 100×100 | 10,000 | ~8 hr | ~8 hr | 71 (NeSI) | K≤100 | More artifacts than 50×50 |

**Per-point cost**: ~2.9 seconds per grid point on NeSI (including 15 multi-start restarts for nuisance optimization)

**Bounds**: K≤100 (tighter than original K≤200) reduces wasted computation on out-of-bounds θ regions.

**Optimization settings**:
- Nuisance optimization: NLopt L-BFGS with 15 random restarts
- ODE solver: DifferentialEquations.jl with Rodas5P (stiff solver)
- ODE tolerances: abstol=1e-8, reltol=1e-6

### IIR Analysis
- Single-threaded on standard workstation
- Runtime: ~60 seconds for 18-parameter model
- Memory: <1GB

### Result Files
- `nesi/repressilator_16nuisance_50x50_results.jls` - **50×50 profile (publication quality)**
- `nesi/repressilator_16nuisance_50x50_results_replot.png` - **publication figure**
- `nesi/repressilator_16nuisance_100x100_results.jls` - 100×100 profile (more artifacts)
- `nesi/repressilator_16nuisance_20x20_results.jls` - 20×20 profile (low resolution)

### MLE Point Estimate Note
The MLE used for IIR analysis may differ slightly from the gridded profile maximum (expected due to different optimization strategies). Options:
1. Re-run MLE with more restarts/adaptive methods for better convergence
2. Show both on plot: original MLE (star) + gridded max (circle) for transparency
3. Note in caption that IIR was computed at nearby high-likelihood point

### Snake Direction Optimization

The `snake_direction` parameter in `compute_profile_grid` controls grid traversal order:

```julia
function compute_profile_grid(
    neg_log_likelihood, θ_fixed, nuisance_idx, bounds_fixed;
    ...
    snake_direction::Symbol = :row  # :row or :col
)
```

**Why it works**: The original column-major snake order caused distant initialization when traversing along the non-identifiable direction (rows). Row-major snake order ensures:
1. Each point initializes from geometrically adjacent neighbor
2. Optimization converges to consistent local minimum along flat ridges
3. No artificial "dips" from far-away initialization

**Quality metrics** showing the improvement:

| Metric | Old | New (snake fix) | Improvement |
|--------|-----|-----------------|-------------|
| Row std mean | 2.124 | 0.039 | 54× |
| Jump mean | 5.0 | 0.28 | 18× |
| Local dips | varied | 0 | eliminated |
| Profile violations | varied | 0 | eliminated |

## Last Updated

**2026-01-17**: 50×50 grid selected as publication quality - cleaner than 100×100 (which had optimizer artifacts in rows 27-28). Good region extremely smooth (mean row std 0.004). Minor θ-space interpolation artifacts at extreme corners acceptable. Documented MLE offset issue and options for handling.

**2026-01-14**: Profile likelihood demonstration complete. 100×100 grid validated with snake_direction fix showing 54× improvement in smoothness. Added documentation on coordinate spaces and bounds hierarchy. Testing tighter bounds (K≤100) to reduce computation.
