# Basis rotation in the sequential IIR algorithm

We consider a smooth mapping `ϕ : Θ → Φ`. In Stage 1 of the sequential IIR procedure we apply the elementwise log transform `θ ↦ x = log θ`, compute the SVD of `Dϕ(x)` at a reference point, and build the square matrix

```
A₁ = [ V_{	ext{r}}ᵀ ; V_{	ext{0}}ᵀ ]
```
where the rows span the orthogonal complements of the invariant and null subspaces. The Stage 1 change of variables is `x ↦ x¹ = A₁ x`, with inverse `x = A₁ᵀ x¹`, and hence

```
θ¹ = exp(x¹) = exp(A₁ log θ)
```
where `exp` and `log` are applied componentwise. Setting `ψ(θ) = exp(A₁ log θ)` yields the Stage 1 “superordinate” parameterisation. Because `A₁` is square with an orthonormal block structure, we may multiply it on the left by any orthogonal matrix `Q`: this simply rotates the Stage 1 coordinates, without altering the invariant subspace or its complement.

In Stage 2 we repeat the same conjugacy, now using `A₂` built from the SVD of `θ¹ ↦ ϕ(θ¹)` at the reference point. The Stage 2 change of variables is `x¹ ↦ x² = A₂ x¹` (with inverse `x¹ = A₂ᵀ x²`), so

```
θ² = exp(A₂ log θ¹) = exp(A₂ A₁ log θ).
```
If `A₂` is also square and orthonormal, then any post-processing rotation (e.g. Varimax) applied to the columns of `A₁` or `A₂` only affects the final display of the identifiable combination(s); the product `A₂ A₁` continues to encode the same subspaces. In other words, the nonlinear map is always a conjugation

```
θ ↦ log θ ↦ A₁ log θ ↦ exp ↦ … ↦ exp(A₂ A₁ log θ)
```
so any additional rotation can safely be postponed until after Stage 2. The identifiable direction is unique up to such an orthogonal change of basis.

