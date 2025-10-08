# Project Status: Invariant Image Reparameterisation (IIR)

## Current Phase: Paper Revision - Code Updates

### Context

The IIR paper has been through peer review at SIAM/ASA Journal on Uncertainty Quantification. We are implementing changes to address reviewer feedback, which includes:

1. **More systematic algorithmic description** (Algorithm 1 in revised paper)
2. **More compelling/ambitious computed examples** (as requested by Associate Editor and reviewers)
3. **Clearer distinction** between minimal image and image reparameterizations
4. **Better integration** of symbolic and numerical approaches

### Relevant Documents

- **New paper**: `/Users/omac010/Git-Working/overleaf_projects/invariant-image-reparameterisation/arXiv/Maclaren_IIR2025.tex`
- **Old paper**: `/Users/omac010/Dropbox/research/manuscripts/01-submitted/iir-arxiv/arXiv/Maclaren_IIR2025.tex`
- **Reviewer comments**: `/Users/omac010/Dropbox/research/manuscripts/03-revising/iir/iir-reviews.pdf`

### What We've Done

#### 1. Implemented `invariance.jl` (NEW)

Created new module implementing Algorithm 1 from the revised paper:
- Function: `find_invariant_subspace(ϕ_func, θ0; compute_J, rtolJ, atolM)`
- **Key innovation**: Uses higher-order Hessian test to separate invariant from non-invariant null space
- **Distinguishes**:
  - **Minimal image**: entire null space is invariant → maximum model reduction
  - **Image (not minimal)**: only part of null space is invariant → partial reduction
- Returns: `S` (singular values), `N` (invariant null space), `N_perp` (identifiable space), `rankJ`

#### 2. Updated `stat_model.jl` Example

Successfully integrated new `invariance.jl` functionality:
- ✅ Uses `find_invariant_subspace()` instead of simple SVD
- ✅ Correctly identifies minimal image reparameterization for Poisson limit
- ✅ Uses full transformation matrix `vcat(N_perp', N')` for code compatibility
- ✅ Fixed transformation to properly implement ψ(θ) = exp(A log(θ))
- ✅ Results now match previous code output (profiles look correct)
- ✅ Correctly identifies `np` as identifiable, `n/p` as non-identifiable

#### 3. Key Fixes Made

1. **Docstring interpolation** (invariance.jl:60): Escaped `\$` in example code
2. **Tolerance naming** (invariance.jl:8,33,117): `rtolM` → `atolM` with practical default `1e-10`
3. **Full transformation matrix** (stat_model.jl:458): Include both identifiable and non-identifiable components
4. **Minimal vs Image logic** (stat_model.jl:372-392, 470-481): Fixed backwards interpretation
5. **Missing exponential** (stat_model.jl:498-499): Added `exp.()` to transformation functions

### Next Steps

#### Completed Examples

1. ✅ **Sum of Independent Poisson Limit Models** - Implemented in `stat_sum_model.jl`
   - Demonstrates sequential IIR application (Stage 1: products, Stage 2: sum)
   - Shows handling of nested non-identifiability structure
   - Verified working for both stages

#### Critical Issue for Manuscript

The revised manuscript assumes sequential IIR "just works" - that Stage 1 naturally produces clean local monomials for Stage 2 to combine. **We've now proven this assumption is false.** Without the interpretability pass:
- Stage 1 produces generic global monomials (mixing all parameters)
- Stage 2 fails to identify correct structure (finds 2 dimensions instead of 1)
- Sequential composition breaks down

**Manuscript needs:**
1. Acknowledge that Algorithm 1 returns arbitrary rotation within span(N_perp)
2. Add procedure/algorithm for interpretability rotation (essential for sequential IIR)
3. Frame interpretability as enforcing compositional sparsity (connection to Poggio et al.)
4. Make explicit in stat_sum_model example that interpretability pass is necessary, not cosmetic
5. Clarify when interpretability matters: crucial for sequential composition, less critical for single-stage

This is not a minor detail - it's a gap between what the paper claims and what actually works.

#### Next: More Sophisticated Mechanistic Model

As requested by reviewers, implement a more complex/realistic mechanistic example:

**Candidate: Systems Biology Model**
   - Multi-parameter ODE system (e.g., extended Michaelis-Menten, gene regulation, signaling cascade)
   - Realistic parameter dimensionality (5-10 parameters)
   - Shows practical identifiability issues in real applications
   - Could use model from literature to anchor credibility
   - May require multiple stages of IIR or more complex parameter combinations

**Alternative: Engineering/Physics Application**
   - Multi-layer transport problem (e.g., heat/diffusion with multiple regions)
   - Parameter-dependent geometry or boundary conditions
   - Demonstrates breadth beyond biology

#### Implementation Plan

For each new example:
1. Define model and auxiliary mapping ϕ(θ)
2. Apply `find_invariant_subspace()`
3. Compare with traditional approaches (symbolic, profile likelihood, sloppiness)
4. Show how IIR identifies parameter combinations without symbolic computation
5. Demonstrate uncertainty quantification via Profile-Wise Analysis
6. Include predictive uncertainty analysis

### Technical Notes

#### Transformation Convention

The reparameterization follows: **ψ(θ) = f⁻¹(A f(θ))**

Where:
- `f = log` (componentwise transformation)
- `f⁻¹ = exp` (componentwise inverse)
- `A = [N_perp'; N']` (full transformation matrix)

This gives both identifiable (`N_perp'`) and non-identifiable (`N'`) combinations.

#### Practical Reparameterization

Even for minimal image case (full null space invariant), we use full square transformation:
```julia
A_full = vcat(N_perp', N')
ψ(θ) = exp.(A_full * log.(θ))
```

This maintains compatibility with existing simulation code while clearly separating identifiable from non-identifiable parameters.

#### Tolerance Choices

- `rtolJ = sqrt(eps())` ≈ 1.5e-8 for Jacobian rank (relative to max singular value)
- `atolM = 1e-10` for Hessian test matrix (absolute, since expecting ~0 for invariant null space)

The absolute tolerance is critical because Hessian-based test produces very small values (~1e-14) for invariant directions, which need practical threshold to distinguish from numerical noise.

#### Transformation Matrix Construction (CRITICAL)

**Correct procedure for IIR transformation:**
1. Build column-stacked matrix: `A_full_T = hcat(N_perp, N)`
2. Scale/round the columns: `A_full_T_scaled = scale_and_round(A_full_T; column_scales=[1,1])`
3. Transpose to get transformation matrix: `A_full = A_full_T_scaled'`

**Why this matters:**
- Rows of A_full define the parameter combinations ψ(θ) = exp(A_full * log(θ))
- Scaling must happen on columns (the basis vectors) BEFORE transposing
- Scaling after transposing corrupts the row combinations (parameter combinations)
- This was the root cause of the Binomial case failure (fixed 2025-01-07)

### Repository Structure

```
reparam/
├── ReparamTools.jl         # Main module
├── invariance.jl            # Algorithm 1 implementation
├── core.jl                  # Profile likelihood, optimization
├── utils.jl                 # Helper functions
├── parameterizations.jl     # Coordinate transformations (fixed scale_and_round)
├── visualization.jl         # Plotting functions
└── examples/
    ├── stat_model.jl        # ✅ Single Poisson limit model
    ├── stat_sum_model.jl    # ✅ NEW: Sum of two Poisson limits (sequential IIR)
    ├── mm_model.jl          # TODO: Update to use invariance.jl
    └── transport_model.jl   # TODO: Update to use invariance.jl
```

### Questions/Decisions Needed

1. Which ambitious example to prioritize first?
2. Should we update existing examples (mm_model.jl, transport_model.jl) before or after new examples?
3. Target parameter dimension for "ambitious" example? (Reviewers want "larger, more realistic")
4. Include comparison with Stigter/Molenaar method mentioned by Reviewer #1?

### Recent Progress (2025-01-07)

#### 1. Tolerance Consistency Verification ✅

Thoroughly verified that tolerances are correctly implemented:
- **rtolJ (relative, ~1.5e-8)**: For Jacobian rank determination, scales with problem magnitude
- **atolM (absolute, 1e-10)**: For invariance test, detects if Hessian products are ~0
- Both tolerances serve different purposes and are correctly designed
- Tested across Poisson and Binomial cases - works correctly

#### 2. Critical Fix: Transformation Matrix Construction ✅

**Problem Identified:**
- `scale_and_round()` was applied to `A_full = vcat(N_perp', N')` where rows are parameter combinations
- Scaling columns of A_full corrupted the row combinations
- In Binomial case: column sign flip changed ψ₁ from `np` to `p/n`

**Root Cause:**
- When smallest column entry is negative (e.g., -0.655), dividing by it flips the column sign
- This sheared the transformation matrix rows, breaking the parameter combinations

**Solution Applied (stat_model.jl:500-504):**
```julia
# OLD (broken):
A_full = vcat(N_perp_inv', N_inv')
evecs_scaled = scale_and_round(A_full; column_scales=[1,1])

# NEW (fixed):
A_full_T = hcat(N_perp_inv, N_inv)  # Columns are vectors
A_full_T_scaled = scale_and_round(A_full_T; column_scales=[1,1])  # Scale columns
evecs_scaled = A_full_T_scaled'  # Then transpose
```

**Result:**
- Poisson: Transformation `[1,1; 1,-1]` → MLE `[np, n/p] = [19.05, 110.6]` ✓
- Binomial: Transformation `[1,1; 1,-1]` → MLE `[np, n/p] = [19.05, 110.6]` ✓
- Both cases now produce identical, correct transformations
- Bounds `[13, 25]` to `[25, 1000]` work for both cases

#### 3. Fixed 2D Profile Plot Bug ✅

**Problem:** Lines 194, 347, 672 used `xy_MLE[i]` for both parameters in 2D-derived 1D plots

**Fix:** Changed second parameter to use `xy_MLE[j]`
```julia
# Line 194, 347, 672:
ψ_MLE=xy_MLE[j]  # Was: xy_MLE[i]
```

**Result:** MLE markers now appear at correct positions on plots

#### 4. Enhanced Practical Non-Identifiability Detection ✅

**Added to stat_model.jl:**
- Always show parameter combination ranking by singular values (lines 411-423)
- Report condition number even when structurally identifiable
- Distinguish structural vs practical non-identifiability
- Show which combinations are better/worse identified

**Key Insight:**
The reparameterization should ALWAYS be done to optimally separate well-identified from poorly-identified combinations, regardless of structural identifiability status.

#### 5. Toned Down Interpretative Output ✅

**Problem:** Output was too enthusiastic and misleading (e.g., declaring parameters "identifiable" when upper confidence limits hit bounds)

**Fix:** Changed output to be factual and objective:
- Removed subjective assessments ("✓ Results match expectation!", "well-conditioned")
- Use factual language ("Appears structurally identifiable", "Condition number: 5.5")
- Let numerical results speak for themselves without interpretation
- Removed emoji and arrows for cleaner, more professional output

**Rationale:** The code should report facts; interpretation belongs in the paper/analysis, not the output.

### Recent Progress (2025-10-07)

#### Implemented stat_sum_model.jl - Sequential IIR Application ✅

Created new example demonstrating sequential application of IIR to sum of two independent Poisson limit models:

**Model Structure:**
- Parameters: θ = [n₁, p₁, n₂, p₂]
- Auxiliary mapping: ϕ(θ) = [n₁p₁ + n₂p₂, n₁p₁ + n₂p₂]
- Distribution: Y ~ N(μ, σ²) where μ = σ² = n₁p₁ + n₂p₂
- Only the sum n₁p₁ + n₂p₂ is identifiable

**Sequential IIR Application:**
1. **Stage 1 (f=log)**: Identifies monomial combinations
   - Applied at true parameters for clean structure
   - Interpretability pass: rotates N_perp to align with sparse [1,1,0,0] and [0,0,1,1] patterns
   - Uses `scale_and_round` on N_perp to get integer coefficients
   - Keeps N (invariant null space) orthonormal for numerical stability
   - Result: θ¹[1] = n₂p₂, θ¹[2] = n₁p₁ (clean products)

2. **Stage 2 (f=id)**: Identifies linear combinations
   - Applied to Stage 1 coordinates
   - **No interpretability pass needed**—Stage 1's clean output means Stage 2's SVD naturally produces [1,1,0,0] pattern
   - Uses `scale_and_round` on N_perp to get integer coefficients
   - Keeps N orthonormal
   - Result: ψ[1] = n₁p₁ + n₂p₂ (the identifiable sum)

**Final Transformation:**
- ψ(θ) = A2 * exp(A1 * log(θ))
- ψ[1] = n₁p₁ + n₂p₂ (identifiable, integer coefficients)
- ψ[2] = 0.71(n₁p₁ - n₂p₂) (non-identifiable, orthonormal)
- ψ[3] = (n₁/p₁)^0.71 (non-identifiable, orthonormal)
- ψ[4] = (p₂/n₂)^0.71 (non-identifiable, orthonormal)

**Key Implementation Details:**

1. **Stage 1 Interpretability Pass** (lines 332-376):
   - Needed at Stage 1 to identify clean monomial products
   - Clusters parameters based on N_perp loadings
   - Projects cluster indicators onto span(N_perp)
   - Orthonormalizes to get sparse basis vectors
   - Produces clean [1,1,0,0] and [0,0,1,1] patterns for products
   - In this example, Stage 2 does not need this—the clean Stage 1 output naturally leads to clean Stage 2 patterns via scale_and_round alone

2. **Selective Scaling** (critical for correctness):
   - Apply `scale_and_round` to N_perp (potentially identifiable) for integer coefficients
   - Keep N (invariant null space) orthonormal for numerical stability
   - This separates "interpretability" (identifiable) from "stability" (non-identifiable)

3. **Bounds Management** (lines 649-709):
   - Use Fisher eigenvalues to determine truly identifiable parameters
   - Tight bounds for identifiable parameters
   - Wide positive bounds for non-identifiable parameters with positive MLEs
   - Symmetric bounds for ψ[2] which can be negative (difference of products)
   - Threshold of 0.1 to distinguish "near-zero" from "positive" MLEs

4. **Bug Fixes:**
   - Fixed `scale_and_round` in parameterizations.jl (line 32): was indexing into original column instead of filtered subset
   - Fixed bounds logic: ψ[2] gets special treatment as it can be negative
   - Used true parameters (not MLE) for invariance analysis to get clean asymmetric structure

**Plot Labels:**
- Use symbolic expressions: $n_1p_1 + n_2p_2$, $0.71(n_1p_1 - n_2p_2)$, etc.
- Makes profiles immediately interpretable
- Fractional exponents (0.71) remain for orthonormal non-identifiable directions

**Design Philosophy:**
- **Identifiable directions**: Maximize interpretability (integer coefficients, sparse patterns)
- **Non-identifiable directions**: Maximize numerical stability (orthonormal basis)
- The fractional coefficients in non-identifiable directions don't matter statistically (likelihood is flat)

**Additional Files:**
- `examples/test_stat_sum.jl`: Lightweight test script (~130 lines) for debugging transformation logic without profiling
- Figures save to main `figures/` directory via `save_dir="../figures/"` parameter

**Verification (2025-10-07):**
- Confirmed Stage 2 does NOT need interpretability pass **in this example**—A2 matrix correctly produces [1.0, 1.0, 0.0, 0.0] pattern
- For this specific case (sum of two products), Stage 1's clean output naturally leads to clean Stage 2 patterns via `scale_and_round` alone
- Other examples with more complex linear combinations at Stage 2 may still benefit from an interpretability pass

**Critical Finding: Interpretability Pass is Essential for Sequential IIR (2025-10-07):**

The Stage 1 interpretability pass (lines 332-377 in stat_sum_model.jl) is **not just cosmetic** - it's essential for sequential IIR to work correctly:

**Without the pass:**
- Stage 1 produces generic monomials: θ¹[1] = n₁^0.5·p₁^0.5·n₂·p₂ (mixing all parameters)
- Stage 2 incorrectly identifies 2 "potentially identifiable" dimensions instead of 1
- Result: defeats the purpose of sequential application - you get what single-stage IIR would produce

**With the pass:**
- Stage 1 produces local products: θ¹[1] = n₁p₁, θ¹[2] = n₂p₂ (respecting parameter structure)
- Stage 2 correctly identifies 1 potentially identifiable dimension (the sum n₁p₁ + n₂p₂)
- Result: true sequential decomposition - products first, then linear combinations

**Conclusion:** The interpretability pass is what makes sequential IIR actually *sequential* rather than just a complicated way to get generic monomials. It enforces local structure at each stage.

**Key principle:** Interpretability is crucial for sequential composition. Without it, `find_invariant_subspace` returns an arbitrary rotation within span(N_perp), which mixes parameters globally. This prevents subsequent stages from building on the structure established by earlier stages. The interpretability pass bridges stages by ensuring each stage's output has the structure the next stage expects.

**Connection to Compositional Sparsity (Poggio et al.):**
The interpretability pass enforces what Poggio calls "compositional sparsity" - the property that functions decompose as compositions of constituent functions, each depending only on low-dimensional subsets of inputs. In our setting:
- Stage 1 products (n₁p₁, n₂p₂) are constituent functions each depending on local parameter subsets
- Stage 2 sum combines these constituents
- Without the interpretability pass, we get global functions mixing all parameters, losing compositional structure
- This suggests interpretability isn't just "making results pretty" - it's **enforcing the compositional decomposition** of the parameter-to-data mapping

**For the manuscript:** This connection could strengthen the paper by framing interpretability as fundamental to discovering compositional structure, not as optional post-processing. The stat_sum_model example should explicitly discuss this.

**Varimax Rotation Approach (2025-10-07):**
Based on varimax_rotation.md, the proper solution is to use **Varimax rotation** (Kaiser 1958) instead of the ad-hoc clustering approach:
- Apply varimax to N_perp with multiple random restarts (e.g., 200)
- Maximizes variance of squared loadings → encourages sparse structure
- Stays within the invariant subspace (orthogonal rotation)
- Use for intermediate stages only (final stage can use raw SVD)

**Implementation:**
- Julia package FactorLoadingMatrices.jl provides `varimax()`
- Added `varimax_rotation()` function to parameterizations.jl (wraps varimax with random restarts)
- Testing shows multiple random restarts are essential - single starts get stuck in poor local optima (objective 0.03-0.49 vs. optimal ~0.5)
- Test case in factor_test.jl successfully reproduces Python results

**Tentative observation:** Our application (varimax on SVD output from small, symmetric problems) may require more restarts (~200) than traditional factor analysis. Possible reasons: perfect symmetry, small dimensions (4×2), balanced loadings from SVD create nearly-flat optimization surface. Further investigation needed.

**Integration (2025-10-08):**
- ✅ Replaced ad-hoc clustering approach in stat_sum_model.jl with `varimax_rotation()`
- ✅ Added FactorLoadingMatrices to package dependencies (ReparamTools.jl line 11)
- ✅ Exported `varimax_rotation` from ReparamTools.jl (line 41)
- ✅ Verified: Stage 1 produces clean [1,1,0,0] / [0,0,1,1] sparse patterns
- ✅ Verified: Stage 2 correctly identifies 1 potentially identifiable dimension (the sum)

**Implementation Status:**
- `varimax_rotation()` is now a reusable utility in parameterizations.jl
- Takes N_perp matrix, returns rotated basis with sparse structure
- Parameters: n_restarts (default 200), threshold (default 1e-2), gamma (default 1.0)
- Use for intermediate stages of sequential IIR (not needed for final stage)
- Single-stage examples (stat_model.jl) don't need it
- Some Stage 2 examples may not need it if Stage 1 produces sufficiently clean output

### Last Updated

2025-10-08 (varimax rotation integration complete and verified)
