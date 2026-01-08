# Hybrid Profile Likelihood: Quadratic Approximation for Nuisance Parameters

## Context

We are working on profile likelihood for a high-dimensional parameter space (18 parameters) where:
- 2 parameters are of **interest** (profiled on a 2D grid)
- 16 parameters are **nuisance** (must be handled somehow at each grid point)

Additionally, some parameter directions are **structurally non-identifiable** (flat likelihood), complicating standard approaches.

## Current Approaches: Two Extremes

### 1. Slice (Fixed at MLE)
- Fix all 16 nuisance parameters at their MLE values
- Evaluate likelihood on 2D grid of interest parameters only
- **Fast**: Just one likelihood evaluation per grid point
- **Problem**: Underestimates uncertainty by ignoring nuisance parameter effects

### 2. Full Profile
- At each grid point, optimize over all 16 nuisance parameters
- Uses multi-start optimization with warm-starting (snake ordering)
- **Accurate**: Proper profile likelihood accounting for all nuisance
- **Slow**: 16-dimensional optimization at each of ~10,000 grid points

## Proposed Hybrid: Quadratic Approximation for Nuisance

### Basic Idea
Instead of full optimization over nuisance at each grid point, use a quadratic (Laplace) approximation to the log-likelihood in nuisance directions.

### Mathematical Framework: The "Linear Path" Approach

**IMPORTANT**: We do NOT approximate the profile likelihood as quadratic. That would force a Gaussian shape and defeat the purpose of profiling non-identifiable parameters.

Instead, we approximate the *optimization path* of nuisance parameters as linear, but evaluate the *true* likelihood along that path.

**Step 1: Compute Hessian at MLE**
```
H = -∂²log L / ∂ψ²  (evaluated at ψ_MLE)
```

Partition into interest (I) and nuisance (N):
```
H = | H_II  H_IN |
    | H_NI  H_NN |
```

**Step 2: Approximate the Optimal Nuisance Path**

The true optimal nuisance satisfies: `ψ_N*(ψ_I) = argmax_ψ_N log L(ψ_I, ψ_N)`

We approximate this path as linear (tangent at MLE):
```
ψ_N*(ψ_I) ≈ ψ_N_MLE - H_NN⁺ H_NI (ψ_I - ψ_I_MLE)
```

where H_NN⁺ is the pseudo-inverse (handles singular/flat directions).

**Step 3: Evaluate TRUE Likelihood Along Path**
```
log L_hybrid(ψ_I) = log L(ψ_I, ψ_N*(ψ_I))
```

This preserves non-Gaussian features (ridges, plateaus, asymmetry) in the interest dimensions.

### Computational Advantage
- Compute H once at MLE (via autodiff)
- At each grid point: one matrix-vector multiply instead of 16-dim optimization
- Speed-up: potentially 100-1000x faster than full profile

## Handling Non-Identifiable Directions

### The Challenge
Some nuisance parameters lie in non-identifiable directions (zero curvature in likelihood). For these:
- H_NN has zero/near-zero eigenvalues
- H_NN⁻¹ doesn't exist or is numerically unstable

### Proposed Solutions

**Option A: Pseudo-inverse**
- Use Moore-Penrose pseudo-inverse H_NN⁺ instead of H_NN⁻¹
- Flat directions contribute zero to the shift
- Nuisance parameters in flat directions stay at MLE

**Option B: Partition by Identifiability**
- Separate nuisance into identifiable (curved) and non-identifiable (flat)
- Apply quadratic approximation only to identifiable nuisance
- Fix non-identifiable nuisance at MLE (they're flat anyway)

**Option C: Regularization**
- Add small ridge: H_NN + εI
- Flat directions get small curvature, stay near MLE

Option B seems most principled given our IIR analysis already identifies which directions are non-identifiable.

## Comparison of Approaches

| Method | Nuisance Handling | Speed | CI Bias |
|--------|------------------|-------|---------|
| Slice | Fixed at MLE | Very fast | Too narrow (ignores nuisance) |
| Full Profile | Full optimization | Slow | Correct |
| Hybrid (linear path) | Linear shift from MLE | Fast | Too narrow (anti-conservative) |

### Critical: The "Anti-Conservative" Property

Because hybrid uses an *approximate* nuisance path, we are strictly suboptimal:
```
log L_hybrid(ψ_I) = log L(ψ_I, ψ_N*(ψ_I)) ≤ log L(ψ_I, ψ_N_optimal(ψ_I)) = log L_true_profile(ψ_I)
```

**Consequences:**
- Hybrid profile drops off *faster* than true profile
- Hybrid crosses significance threshold *sooner*
- Hybrid confidence intervals are **subsets** of true intervals (narrower)

**Conclusion:** The method is **anti-conservative** (over-confident). If the linear path approximation fails, we underestimate uncertainty. This makes validation and diagnostics crucial.

## When Hybrid Should Work Well

1. **Near MLE**: Quadratic approximation is local; accurate near the mode
2. **Identifiable nuisance**: Curved directions are well-approximated by quadratic
3. **Small interest region**: If profiling close to MLE, linearization of nuisance shift is valid

## When Hybrid May Fail

1. **Far from MLE**: Quadratic approximation breaks down
2. **Non-quadratic likelihood**: Strong nonlinearity, multimodality
3. **Correlated flat directions**: Complex interactions between non-identifiable parameters

## Relevant Literature

1. **Profile Likelihood with Laplace Approximation**:
   - [PMC: Profile Likelihood for Hierarchical Models](https://pmc.ncbi.nlm.nih.gov/articles/PMC10530212/) discusses combining quadratic approximation with profile likelihood

2. **Laplace Approximation Basics**:
   - [Wikipedia: Laplace's Approximation](https://en.wikipedia.org/wiki/Laplace's_approximation)
   - [Duke Stats Notes](https://www2.stat.duke.edu/~st118/sta250/laplace.pdf)

3. **Profile Likelihood and Nuisance Parameters**:
   - [Iowa Notes on Profile Likelihood](https://myweb.uiowa.edu/pbreheny/7110/f20/notes/11-04.pdf)
   - [Reid: Aspects of Likelihood Inference](https://utstat.toronto.edu/reid/research/published.pdf)

4. **Limitations of Laplace**:
   - [MDPI: Bias in Laplace Approximation](https://www.mdpi.com/1099-4300/27/3/289) - discusses when approximation fails

## Implementation Notes

### What We Have
- `construct_quadratic_approximation()` in `core.jl` — computes Hessian via ForwardDiff
- Likelihood defined in ψ-space (IIR coordinates)
- IIR analysis identifying which directions are non-identifiable

### Algorithm

**Setup (once):**
1. Compute H = -∂²log L / ∂ψ² at MLE via autodiff
2. Partition indices: interest I = {12, 17}, nuisance N = {rest}
3. Extract blocks H_NN, H_NI
4. Compute eigendecomposition of H_NN: identify identifiable subspace U_r (eigenvalues above threshold) and compute H_NN⁺

**Per grid point ψ_I:**
1. Compute δψ_I = ψ_I - ψ_I_MLE
2. Compute δψ_N = -H_NN⁺ H_NI δψ_I
3. Evaluate log L_hybrid(ψ_I) = log L(ψ_I, ψ_N_MLE + δψ_N)
4. (Diagnostic) Compute projected gradient: g_ident = U_rᵀ ∇_N log L

### Handling Non-Identifiable Nuisance Directions

With structurally non-identifiable nuisance directions, the conditional maximiser is set-valued (a manifold of equivalent optima). Restricting to the identifiable nuisance subspace yields a well-defined tangent via the implicit function theorem; the Moore–Penrose pseudoinverse selects the canonical minimum-norm representative among equivalent paths.

**Practical implementation:**
- Symmetrise H_NN before eigendecomposition
- Use explicit eigenvalue threshold (e.g., rtol = 1e-8 relative to largest eigenvalue) aligned with IIR predictions
- Verify spectral gap matches expected number of non-identifiable directions

## Key Subtlety: Singular/Non-Identifiable Nuisance Directions

The main technical challenge is that H_NN (nuisance Hessian) will be singular when some nuisance directions are structurally non-identifiable.

In our repressilator example:
- Interest: ψ_12 (K₁/β₁, identifiable), ψ_17 (β₁·K₁, non-identifiable)
- Nuisance includes: β₂·K₂, β₃·K₃ (also non-identifiable)

So H_NN has ~2 zero eigenvalues corresponding to these flat directions.

**Our approach**: Use pseudo-inverse H_NN⁺ which:
- Inverts along curved (identifiable) directions normally
- Projects out flat (non-identifiable) directions (they stay at MLE)

This is mathematically equivalent to: "profile over identifiable nuisance, fix non-identifiable nuisance at MLE" - which is sensible since flat directions don't affect the likelihood anyway.

## Diagnostics

### 1. Structural Check (at MLE)

Report eigenvalue spectrum of H_NN. Expect clear spectral gap between curved (identifiable) and flat (non-identifiable) directions. Verify count of near-zero eigenvalues matches IIR predictions.

### 2. Validity Check (per grid point)

Compute projected gradient onto identifiable nuisance subspace:
```
g_ident(ψ_I) = U_rᵀ ∇_N log L |_{(ψ_I, ψ_N*(ψ_I))}
```

Interpretation:
- ||g_ident|| ≈ 0: linear approximation is near the valley floor (good)
- ||g_ident|| large: walking along the side of the valley (approximation failing)

For a dimensionless version, scale by curvature:
```
g_scaled = Λ_r^{-1/2} U_rᵀ ∇_N log L
```
where Λ_r contains the non-zero eigenvalues. Then ||g_scaled|| < 0.1 indicates the approximation is adequate (gradient is small relative to local curvature).

### 3. Alternative Diagnostic — Newton Step Size
```
δ_Newton = -H_{NN,r}⁻¹ (U_rᵀ ∇_N log L)
```
This directly measures "how far would one correction step move us?" in parameter units. Large ||δ_Newton|| indicates we're far from the valley floor.

## Validation: Three-Way Comparison

Since we compute slice, hybrid, and full profile on the same grid:

1. **Slice vs Full**: Quantifies uncertainty lost by ignoring nuisance coupling
2. **Hybrid vs Full**: Tests whether profile path is approximately linear
3. **Diagnostic overlay**: Show ||g_scaled|| or ||δ_Newton|| as heatmap; verify divergence occurs where diagnostics spike

Contours shown at χ²₂ threshold (Δℓ = -3.0) as reference for visual comparison. Given non-identifiable structure in ψ_17, this is a nominal threshold rather than a calibrated coverage claim.

## Validation Figure: 3-Panel Layout

1. **Profile Comparison:** Overlay Hybrid (dashed) vs Full (solid). Show that Hybrid ≤ Full (lower bound).
2. **Path Deviation:** Plot nuisance trajectory (ψ_N* vs ψ_I). Show linear tangent vs true optimized path.
3. **Gradient Diagnostic:** Plot ||g_scaled|| across grid as heatmap. Demonstrate profiles diverge exactly where diagnostics spike.

This framing turns the method's limitation (linearity) into a rigorous, self-diagnosing tool.

## Questions for Discussion

1. Which diagnostic to prioritize: ||g_scaled|| (dimensionless) or ||δ_Newton|| (parameter units)?
2. Should we implement adaptive hybrid that falls back to optimization when diagnostic exceeds threshold?
3. For the paper: include hybrid as methodological contribution, or just use as computational tool?

---
*Summary prepared for discussion, 2026-01-09*
