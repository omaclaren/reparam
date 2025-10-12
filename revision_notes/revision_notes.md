# Revision Notes: IIR Paper

**Last Updated:** 2025-10-13
**Status:** Examples complete, manuscript finalization in progress

## Strategic Decision: Single-Stage IIR Focus

After extensive investigation of multi-stage/sequential IIR approaches, we have decided to focus the paper on **single-stage IIR with monomial transformations** as the main contribution.

### Rationale

**Single-stage IIR is robust and reliable:**
- Algorithm 1 correctly identifies invariant subspaces across diverse model types
- Varimax rotation provides interpretable basis within span(N_perp)
- Works for statistical models (stat_model) and mechanistic ODEs (repressilator)
- Clear theoretical foundation with no basis-dependence issues

**Multi-stage IIR reveals open research problems:**
- Success depends critically on intermediate basis "aligning" with target combinations
- Works perfectly for stat_sum_model (products → sums naturally align)
- Fails for pk_model (Varimax produces basis misaligned with target sums)
- Varimax optimizes sparsity, NOT compositional reducibility
- Basis selection for optimal compositional reduction is an unsolved problem

### Paper Structure

**Main Text:**
1. **stat_model.jl** - Pedagogical example (Poisson limit, 2 parameters)
2. **repressilator.jl** - Ambitious mechanistic example (18-parameter ODE system)
3. Algorithm 1 with both invariance methods (:hessian_based, :finite_difference)
4. Varimax rotation as optional interpretability enhancement

**Future Work (brief mention):**
- Sequential application possible when bases naturally align
- Basis selection for compositional reduction remains open problem
- Reference to stat_sum_model (success) and pk_model (limitation)

**Supplement (optional):**
- Detailed multi-stage investigation
- PK model analysis showing basis dependence
- When sequential IIR works vs when it doesn't

## Completed Examples

### 1. stat_model.jl ✅
**Model:** Poisson limit distribution (n, p parameters)
**Results:** Identifies np (identifiable) and n/p (non-identifiable)
**Status:** Complete and verified
**Role:** Clear pedagogical introduction to IIR

### 2. repressilator.jl ✅
**Model:** Eisenberg & Hayashi (2010) 3-gene repressilator
**Parameters:** 18 (with n=2 fixed)
**Results:**
- Rank 15/18 (3-dimensional invariant null space)
- Identifies K₁/β₁, K₂/β₂, K₃/β₃ ratios
- Matches Eisenberg's profile likelihood results
**Technical Advances:**
- Finite-difference invariance test for stiff ODEs (atolM=1e-6)
- Profile-wise prediction uncertainty (Discovery→Problem→Solution narrative)
**Status:** Complete (path bug fixed 2025-10-13)
**Role:** Demonstrates IIR on realistic mechanistic model

## Multi-Stage Investigation (Reference Only)

### What We Learned

**stat_sum_model.jl** - Sequential IIR Success:
- Stage 1 (f=log): Identifies products n₁p₁, n₂p₂
- Stage 2 (f=identity): Identifies sum n₁p₁ + n₂p₂
- Works because structure naturally aligns

**pk_model.jl** - Sequential IIR Limitation Discovery:
- Stage 1 correctly identifies rank 6/8
- Varimax produces interpretable monomials
- BUT: Target sums (k₀₂+k₁₂, etc.) become nonlinear functions of Stage 1 coordinates
- Stage 2 with f=identity cannot recover them
- Root cause: Varimax optimizes wrong objective (sparsity, not reducibility)

**Key Insight:**
Sequential IIR is basis-dependent. Any rotation within span(N_perp) is mathematically equivalent at Stage 1 but produces different Stage 2 results. This is an important open research problem, not a production-ready method.

## Varimax Role Clarified

**Original idea:** Use Varimax during sequential stages to improve alignment

**Current understanding:**
- Varimax is an **interpretability enhancement**, not part of core algorithm
- Apply AFTER Algorithm 1 completes (single or multi-stage)
- Rotates within span(N_perp) to produce sparse, human-readable combinations
- For single-stage IIR: Works reliably (repressilator K/β ratios)
- For multi-stage IIR: May help or hurt depending on problem structure

**Implementation:**
```julia
# After find_invariant_subspace returns N_perp
N_perp_rotated = varimax_rotation(N_perp; n_restarts=200)
# Use rotated basis for display/interpretation
# Keep orthonormal for transformation (numerical stability)
```

## Addressing Reviewer Comments

### "More ambitious/compelling computed examples"
✅ **Addressed:** repressilator.jl
- 18 parameters, realistic ODE system
- Published benchmark (Eisenberg & Hayashi 2010)
- Demonstrates IIR on stiff mechanistic model
- Validates against established profile likelihood results

### "More systematic algorithmic description"
✅ **Addressed:** Algorithm 1 in invariance.jl
- Clear step-by-step procedure
- Two invariance test methods documented
- Tolerance choices explained (rtolJ, atolM)
- Handles both minimal and non-minimal image cases

### "Clearer distinction between minimal image and image"
✅ **Addressed:** Terminology consistent throughout
- "Minimal image": entire null space is invariant
- "Image (not minimal)": only part of null space is invariant
- Examples demonstrate both cases

## Implementation Status

**Core algorithm (invariance.jl):** ✅ Complete
- `find_invariant_subspace()` implements Algorithm 1
- Both invariance methods working (:hessian_based, :finite_difference)
- Proper tolerance handling (rtolJ, atolM)

**Examples for paper:** ✅ Complete
- stat_model.jl: pedagogical
- repressilator.jl: ambitious mechanistic

**Exploratory code (not in paper):** ✅ Complete but deferred
- stat_sum_model.jl: multi-stage success case
- pk_model.jl: multi-stage limitation discovery
- Various test files documenting investigation

## Immediate Next Steps

### High Priority
1. **Verify repressilator runs end-to-end**
   ```julia
   include("examples/repressilator.jl")
   ```
   Expected: Rank 15/18, K/β ratios, prediction comparison figure

2. **Update manuscript Methods section**
   - Algorithm 1 matches invariance.jl
   - Document both invariance methods
   - Explain tolerance choices

3. **Write manuscript Results section**
   - stat_model narrative (pedagogical)
   - repressilator narrative (Discovery→Problem→Solution)
   - Comparison with traditional methods

### Medium Priority
4. **Draft reviewer response**
   - Emphasize single-stage robustness
   - Highlight repressilator as ambitious example
   - Explain strategic focus on reliable method

5. **Prepare figures**
   - stat_model profiles (should exist)
   - repressilator prediction comparison (verify generation)

## Key Messages for Paper

1. **IIR is a practical, numerically robust method** for discovering identifiable parameter combinations without symbolic computation

2. **Single-stage monomial transformations** (ψ = exp(A log(θ))) work reliably across diverse model types

3. **Numerical invariance test** enables automatic discovery purely from Jacobian (no symbolic algebra required)

4. **Varimax rotation** enhances interpretability by producing sparse combinations within identifiable subspace

5. **Profile-wise prediction** demonstrates practical importance - honest uncertainty quantification via correct reparameterization

## Timeline to Submission

- Repressilator verification: 2-4 hours
- Manuscript Methods/Results: 12-20 hours
- Reviewer response: 4-6 hours
- **Total: 20-30 hours**

## Documentation

**For quick overview:** [PROJECT_SUMMARY.md](../PROJECT_SUMMARY.md)
**For user introduction:** [README.md](../README.md)
**For complete details:** [CLAUDE.md](../CLAUDE.md)
**For next actions:** [NEXT_STEPS.md](../NEXT_STEPS.md)
