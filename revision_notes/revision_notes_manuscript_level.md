# Revision Notes: Sequential IIR for Manuscript

## Framing for Manuscript

### Main Concept

Sequential Invariant Image Reparameterization applies Algorithm 1 iteratively with different coordinate transformations. Each stage follows a conjugacy-type pattern:

1. **Transform** to new coordinates via `f` (e.g., log-transformed space)
2. **Analyze** using Algorithm 1 (linear Jacobian/Hessian analysis in that space)
3. **Transform back** via `f⁻¹` to express results in original parameter space

Multiple stages compose these transformations:

1. **First stage**: Use transformation `f₁` (e.g., `f = log`) to identify one type of structure (e.g., multiplicative combinations)
2. **Second stage**: Use transformation `f₂` (e.g., `f = identity`) on the first-stage coordinates to identify another type (e.g., additive combinations)
3. **Further stages** (if needed): Additional transformations with other choices of `f`

The choice of transformation functions depends on the anticipated structure (e.g.):
- **f = log**: Maps products to sums, exposing multiplicative/log-monomial combinations
- **f = identity**: No transformation, exposing additive/linear combinations directly

This compositional approach allows discovery of **mixed structures** like linear combinations of monomials (e.g., n₁p₁ + n₂p₂) without symbolic computation.

**Example workflow** (not the only option): Apply `f = log` (first stage) then `f = identity` (second stage) to find linear combinations of monomials.

### Example: Sum of Independent Poisson Limits

**Model**: Y ~ N(μ, σ²) where μ = σ² = n₁p₁ + n₂p₂

**First stage** (f=log):
- Finds 2D subspace spanned by log-monomial combinations
- Numerical analysis identifies this span contains log(n₁p₁) and log(n₂p₂)
- Creates coordinates where these products are separated

**Second stage** (f=identity):
- Applies Algorithm 1 to first-stage coordinates
- Discovers that only the *difference* log(n₁p₁) - log(n₂p₂) affects output
- The *sum* log(n₁p₁) + log(n₂p₂) ≈ log(n₁p₁ + n₂p₂) is invariant

**Result**: Identifies compositional non-identifiability without symbolic tools

### Basis Selection (Optional Enhancement)

Within the potentially identifiable subspace identified by Algorithm 1, **any orthonormal basis is mathematically equivalent**. However, for interpretability, we can apply rotations such as:

- **Varimax rotation**: Maximizes sparsity of coefficients
- **Canonical alignment**: Aligns with original parameter directions where possible

These rotations are applied **after** invariance detection to aid interpretation, but do not affect which subspace is identified.

For the sum-of-Poisson example, Varimax rotation yields the interpretable basis:
- θ¹₁ = n₁p₁
- θ¹₂ = n₂p₂

making it immediate to see that the sum n₁p₁ + n₂p₂ is the non-identifiable combination.

### Key Points for Paper

1. **Sequential application** of Algorithm 1 with different `f` functions enables compositional discovery

2. **Extensible to multiple stages** - framework supports arbitrary sequences of transformations, though two stages suffice for most cases

3. **No symbolic computation required** - purely numerical analysis of Jacobian/Hessian structure

4. **Basis rotations** (like Varimax) are optional enhancements for interpretation, not part of the core algorithm

5. **Works for models where traditional methods struggle** - e.g., sums of products, ratios of sums, etc.

### Suggested Manuscript Language

> While Algorithm 1 identifies the invariant subspace span(N), any orthonormal basis for this subspace is mathematically equivalent. However, for cases where interpretable parameter combinations are desired, sparse rotations such as Varimax [cite] can be applied to the identified basis vectors. This yields representations like θ₁ = n₁p₁, θ₂ = n₂p₂ that make the compositional structure (e.g., the sum n₁p₁ + n₂p₂) immediately apparent.

> For sequential IIR, this approach is particularly useful: after the first stage identifies the span of log-monomials, a sparse rotation can separate individual products or ratios. Subsequent stages then identify which linear combinations of these products are invariant. This workflow—identify subspace numerically at each stage, rotate for interpretability—separates the algorithmic core from presentational choices. While two stages (log followed by identity) suffice for most applications, the framework naturally extends to additional stages for more complex compositional structures.

### What NOT to include in manuscript

- Implementation details about SVD vs Varimax bases in the code
- How to handle non-orthogonal transformations
- Numerical stability considerations of scale_and_round
- Specific Julia code structure

### What TO include

- High-level description of sequential composition
- Conceptual role of basis rotations (interpretation, not detection)
- Worked example showing how it discovers compositional structure
- Comparison with symbolic approaches (when available)

## Bottom Line

**For the manuscript**: Focus on the mathematical/statistical concept that sequential IIR enables compositional discovery through staged transformations.

**For implementation docs**: The detailed workflow about SVD bases, Jacobian filtering, and post-processing dictionary goes in code documentation and examples.
