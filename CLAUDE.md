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

**Single-stage IIR is robust:**
- ✅ Algorithm 1 identifies correct rank/invariant subspace reliably
- ✅ Works across diverse model types (statistical, ODE systems)
- ✅ Varimax rotation produces interpretable monomial combinations
- ✅ Clear theoretical foundation
- ✅ No basis-dependence issues

**Sequential/multi-stage IIR is fragile:**
- ⚠️ Success depends on intermediate basis "aligning" with final target combinations
- ⚠️ Varimax optimizes sparsity, not compositional reducibility
- ⚠️ Opens complicated questions about optimal basis selection (open research problem)

**Decision**: Focus paper on single-stage method as solid, practical contribution. Multi-stage briefly mentioned in future work with honest assessment of limitations.

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
- Parameters: 18 (fixing n=2 from original 19)
- System: 6 coupled nonlinear ODEs (stiff)
- Observables: All 3 mRNA time series

**IIR Analysis Results** ✅:
- Rank: 15/18 (3 non-identifiable directions)
- Invariant null space: βK products (3-dimensional)
- Identifiable combinations: K/β ratios (up to sign/reciprocal ambiguity)
- Finite-difference invariance test works for stiff ODEs (atolM=1e-6)

**Profile Likelihood Demonstration** 🔄 In Progress:
The goal is to show that profiling original parameters (β₁, K₁) gives misleading narrow uncertainty, while profiling IIR-identified combinations (K₁/β₁ identifiable, β₁K₁ non-identifiable) reveals the true structure.

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
**Function**: `find_invariant_subspace(ϕ_func, θ0; compute_J, rtolJ, atolM, invariance_method)`

**Algorithm**:
1. Compute Jacobian J at θ0 using automatic differentiation
2. Determine rank via SVD with relative tolerance rtolJ
3. Extract null space candidates from right singular vectors
4. Test each null vector for invariance using Hessian-based criterion
5. Separate N (invariant) from N_perp (potentially identifiable)

**Parameters**:
- `rtolJ = sqrt(eps())` ≈ 1.5e-8: Relative tolerance for Jacobian rank
- `atolM = 1e-10`: Absolute tolerance for invariance test (stricter for smooth problems)
- `atolM = 1e-6`: Relaxed tolerance for stiff ODE systems
- `invariance_method = :hessian_based` (default) or `:finite_difference` (for stiff ODEs)

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

**Jacobian rank (rtolJ)**: Relative tolerance
- Default: `sqrt(eps())` ≈ 1.5e-8
- Scales with problem magnitude
- Threshold: `τ = rtolJ * σ_max`

**Invariance test (atolM)**: Absolute tolerance
- Default: 1e-10 (smooth problems)
- Relaxed: 1e-6 (stiff ODEs with numerical noise)
- Does NOT scale (testing if Hessian products ≈ 0)

### 2. Invariance Methods

**:hessian_based** (default):
- Uses nested automatic differentiation
- Most accurate for smooth problems
- May fail for stiff ODEs

**:finite_difference** (for stiff systems):
- Only uses first-order AD
- Numerically probes parameter perturbations
- Essential for repressilator and similar ODE models

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

**Conclusion**: Multi-stage IIR reveals important open research problem (basis selection for compositional reduction) but is not ready for production use. Single-stage method is robust and reliable.

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

### Profile Likelihood Demonstration (Current Focus)
- [x] 2D ideal case working (`minimal_2D_IIR_coords.jl`)
- [x] 2D + nuisance working (`minimal_2D_IIR_nuisance.jl`)
- [x] 6-param subset profiling working (`iir_guided_profiling.jl`)
- [ ] **Next**: Extend to full 18-param model using learnings from 6-param
- [ ] Generate comparison figure: original params vs IIR coords

### Practical Challenge: Basis Ambiguity
IIR identifies the correct *subspace* but the basis vectors have ambiguity:
- Sign flips (K/β vs β/K)
- Reciprocals (K/β vs K·β could be confused without careful interpretation)

For paper figures, need to manually select which IIR coordinates correspond to gene 1's β and K to make an interpretable comparison. This is a presentation issue, not an algorithm issue.

### For Manuscript
- [x] stat_model complete ✅
- [ ] Repressilator profile demonstration figure
- [ ] Update Methods section to match invariance.jl implementation
- [ ] Write Results section highlighting both examples

### For Reviewer Response
- [ ] Draft response emphasizing single-stage robustness
- [ ] Highlight repressilator as ambitious mechanistic example
- [ ] Explain strategic focus on reliable method over speculative extensions

## Questions Resolved

1. ~~Which ambitious example?~~ → Repressilator
2. ~~Single vs multi-stage focus?~~ → Single-stage (robust and reliable)
3. ~~How to handle sequential IIR findings?~~ → Brief mention in future work
4. ~~Development strategy for profiling?~~ → Incremental: 2D → 2D+nuisance → 6-param → 18-param

## Last Updated

**2025-12-10**: Updated to reflect actual status. IIR analysis for repressilator works; profile likelihood demonstration in progress using incremental development strategy (2D → 6-param → 18-param). Documented basis ambiguity challenge for figure generation.
