# Sequential IIR: Clarification on Dictionary Approach

## Key Insight from Recent Discussion

The **dictionary approach does NOT require pre-computing nice bases** for the sequential algorithm to work!

## What Sequential IIR Actually Does

### Stage 1: Coordinate transformation with f=log

```julia
# find_invariant_subspace returns orthonormal SVD bases
S_s1, N_s1, N_perp_s1, rank_s1 = find_invariant_subspace(ϕ_log, log(θ_true))

# Build SQUARE transformation matrix (4×4 for 4 parameters)
A1 = [N_perp_s1; N_s1]'

# Define coordinate transformations
θ¹ = exp(A1 * log(θ))           # Forward
θ = exp(inv(A1) * log(θ¹))      # Inverse (or A1' if orthonormal)
```

**Result**: New 4D coordinate system where multiplicative structure is exposed

### Stage 2: Find invariance in transformed coordinates with f=identity

```julia
# Auxiliary mapping in Stage 1 coordinates
ϕ_stage2(θ1) = ϕ(θ(θ1))

# Apply find_invariant_subspace again (f=identity this time)
S_s2, N_s2, N_perp_s2, rank_s2 = find_invariant_subspace(ϕ_stage2, θ¹_MLE)
```

**Result**: Identifies additive structure among the Stage 1 coordinates

### Post-Processing: Dictionary rotation for interpretation (OPTIONAL)

```julia
# Compute Varimax rotation of Stage 1 basis
N_perp_varimax = varimax_rotation(N_perp_s1)
R_s1 = N_perp_s1' * N_perp_varimax

# Rotate Stage 2 results to Varimax basis for interpretation
for col in 1:size(N_perp_s2, 2)
    v_svd = N_perp_s2[:, col]
    v_varimax = R_s1' * v_svd
    # Display in terms of Varimax combinations
end
```

**Result**: Human-readable combinations like "n₁p₁ + n₂p₂"

## Dimension Reduction

**Dimension reduction happens WITHIN each `find_invariant_subspace` call**, NOT by explicitly working in reduced coordinates.

- `find_invariant_subspace` returns `rank` (numerical rank of Jacobian)
- Dimension of identifiable space = `rank`
- Dimension of invariant space = `n_params - rank`

The returned `N_perp` may have more columns than the rank (it spans the full null space complement), but **Jacobian projection** reveals which directions actually affect output.

## Why This Works Without Pre-Computing Dictionary

1. **Stage 1** transforms to coordinates where multiplicative combinations are separated
   - Works with ANY orthonormal basis (SVD is fine)
   - Varimax just makes it more interpretable afterward

2. **Stage 2** finds additive combinations in those coordinates
   - Again, works with ANY orthonormal basis
   - Varimax makes final result more interpretable

3. **Compositional invariance** (like sum of products) emerges automatically
   - Stage 1 creates coordinates for products (n₁p₁, n₂p₂)
   - Stage 2 finds their sum
   - Together: identifies invariance in n₁p₁ + n₂p₂

4. **Dictionary is pure display layer**
   - Doesn't affect which invariant subspace is found
   - Only affects how we interpret/communicate the results

## Comparison: Old vs New Understanding

### Old Understanding (INCORRECT)
- Stage 1: 4D → 2D reduction (explicitly work in 2D)
- Need to pre-compute Varimax to get "nice" 2D coordinates
- Stage 2: 2D → 1D reduction
- Dictionary built into the transformation

### New Understanding (CORRECT)
- Stage 1: 4D → 4D transformation (square matrix)
- Stage 2: 4D → 4D, but identifies smaller invariant subspace
- Dictionary applied AFTER both stages for interpretation
- Both stages work with SVD orthonormal bases

## Practical Implications

### Simplified Workflow

```julia
# ============================================================
# STAGE 1: f=log
# ============================================================
S1, N1, Nperp1, r1 = find_invariant_subspace(θ_log -> ϕ(exp(θ_log)), log(θ_true))

A1 = [Nperp1; N1]'
θ_to_θ1(θ) = exp(A1 * log(θ))
θ1_to_θ(θ1) = exp(inv(A1) * log(θ1))

# ============================================================
# STAGE 2: f=identity
# ============================================================
S2, N2, Nperp2, r2 = find_invariant_subspace(θ1 -> ϕ(θ1_to_θ(θ1)), θ1_MLE)

# ============================================================
# POST-PROCESS: Varimax for interpretation
# ============================================================
# Stage 1 Varimax
Nperp1_var = varimax_rotation(Nperp1)
R1 = Nperp1' * Nperp1_var

# Interpret Stage 2 results in Varimax basis
# (multiply Stage 2 basis vectors by R1')
```

### Benefits

1. **Numerical stability**: Transformation matrices stay orthonormal
2. **Algorithmic clarity**: Two stages are identical in structure
3. **Flexibility**: Can try different dictionaries without re-running stages
4. **Modularity**: Dictionary is separate from invariance detection

## What About "Extra Columns"?

When `find_invariant_subspace` returns `N_perp` with multiple columns:

1. **Check Jacobian projection**: `J * N_perp`
2. **Compute SVD**: See which columns have non-zero effect
3. **Keep active columns**: Only those with large singular values
4. **The rest are artifacts**: Orthogonal directions in the span that don't affect output

This is **expected behavior**, not a bug. The Hessian test returns an orthonormal basis for the full potentially identifiable span. The Jacobian tells you which directions within that span actually matter.

## Example: stat_sum_model

**Model**: Sum of independent Poisson limits
- Parameters: [n₁, p₁, n₂, p₂]
- Identifiable: n₁p₁ + n₂p₂ (sum of products)

**Stage 1 (f=log)**:
- Finds 2D span: [log(n₁p₁), log(n₂p₂)]
- SVD basis works fine
- Varimax makes it sparse: [1,1,0,0] and [0,0,1,1]
- Transform: θ¹[1] = n₁p₁, θ¹[2] = n₂p₂, θ¹[3], θ¹[4] (invariant)

**Stage 2 (f=identity)**:
- Finds that θ¹[1] + θ¹[2] is invariant (the sum!)
- Returns 2D basis, but only 1D affects output
- Varimax shows: combination is [1,1,0,0] in θ¹ space = n₁p₁ + n₂p₂

**Result**: Compositional invariance identified without pre-computing dictionary!

## Conclusion

The **dictionary approach is purely post-processing**. The sequential algorithm works with standard SVD orthonormal bases throughout. Varimax or other sparse rotations are applied **after** both stages complete, solely for human interpretation.

This dramatically simplifies the implementation and removes the numerical issues we encountered when trying to build the dictionary into the transformation matrices.
