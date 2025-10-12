# IIR Project Summary

**Status:** Ready for manuscript finalization (2025-10-13)

## What is IIR?

**Invariant Image Reparameterisation (IIR)** is a numerical method for automatically discovering which parameter combinations in a mathematical model are identifiable from data, without requiring symbolic computation.

**Key innovation**: Uses a numerical invariance test to identify which directions in parameter space have invariant null spaces, enabling separation of identifiable from non-identifiable combinations.

## The Method (Single-Stage Focus)

**Transformation**: ψ(θ) = exp(A log(θ))

Where A is determined by Algorithm 1 (`find_invariant_subspace`):
1. Compute Jacobian at reference point
2. Determine numerical rank via SVD
3. Test null space vectors for invariance using Hessian criterion
4. Separate invariant (N) from potentially identifiable (N_perp) directions
5. Optional: Apply Varimax rotation to N_perp for interpretability

**Result**: Monomial parameter combinations with integer exponents that cleanly separate identifiable from non-identifiable structure.

## Two Examples for Paper

### 1. stat_model.jl (Pedagogical)
- **Model**: Poisson limit distribution
- **Parameters**: n, p (2 total)
- **Identifiable**: np (mean)
- **Non-identifiable**: n/p (ratio)
- **Purpose**: Clear introduction to IIR workflow

### 2. repressilator.jl (Ambitious)
- **Model**: 3-gene repressilator ODE system (Eisenberg & Hayashi 2010)
- **Parameters**: 18 (with n=2 fixed)
- **Identifiable**: K₁/β₁, K₂/β₂, K₃/β₃ ratios
- **Validation**: Matches profile likelihood results from Eisenberg
- **Demonstrates**:
  - IIR on realistic mechanistic model
  - Finite-difference invariance test for stiff ODEs
  - Profile-wise prediction uncertainty (honest vs misleading intervals)

## Strategic Decisions

### ✅ Single-Stage IIR (Main Contribution)
**Why**: Robust, reliable, clear theoretical foundation, works across model types

**What goes in paper**:
- Algorithm 1 (numerical invariance test)
- Two examples (pedagogical + ambitious)
- Varimax rotation as interpretability enhancement
- Profile-wise prediction uncertainty

### ⚠️ Multi-Stage IIR (Future Work)
**Why not in main text**: Fragile, basis-dependent, reveals open research problems

**What we learned**:
- Works when bases align (stat_sum_model: products → sums ✓)
- Fails when Varimax misaligns (pk_model: basis choice problem ✗)
- Basis selection for compositional reduction is an open problem

**How to position**: Brief mention in future work as interesting direction requiring further research

## Key Files

**Documentation**:
- [CLAUDE.md](CLAUDE.md) - Complete project documentation
- [NEXT_STEPS.md](NEXT_STEPS.md) - Actionable next steps for manuscript
- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - This file

**Core Code**:
- [invariance.jl](invariance.jl) - Algorithm 1 implementation
- [parameterizations.jl](parameterizations.jl) - Transformations, Varimax
- [core.jl](core.jl) - Profile likelihood
- [ReparamTools.jl](ReparamTools.jl) - Main module

**Examples for Paper**:
- [examples/stat_model.jl](examples/stat_model.jl) - Pedagogical ✅
- [examples/repressilator.jl](examples/repressilator.jl) - Ambitious ✅

**Exploratory Code** (not in paper):
- [examples/stat_sum_model.jl](examples/stat_sum_model.jl) - Multi-stage success case
- [examples/pk_model.jl](examples/pk_model.jl) - Multi-stage limitation discovery

## Immediate Next Step

**Verify repressilator runs end-to-end**:
```julia
cd("/Users/omac010/Git-Working/reparam")
include("examples/repressilator.jl")
```

Expected: Rank 15/18, K/β ratios identified, prediction comparison figure generated

## Paper Message

**IIR is a practical, numerically robust method for discovering identifiable parameter combinations without symbolic computation. It works reliably across diverse model types (statistical, ODE systems) when applied as single-stage monomial transformation.**

## Timeline to Submission

- Repressilator verification: 2-4 hours
- Manuscript Methods/Results: 12-20 hours
- Reviewer response: 4-6 hours
- **Total: 20-30 hours**

## Success Metrics

Ready when:
- [ ] Repressilator verified working
- [ ] Two publication-quality figures
- [ ] Methods section matches Algorithm 1
- [ ] Results section tells compelling story
- [ ] Reviewer response addresses all concerns constructively
