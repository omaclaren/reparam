# Sequential IIR with Profile Likelihood Plots

## Overview

This document describes how to generate profile likelihood plots after applying the clean sequential IIR workflow where **dictionary (Varimax) is post-processing only**.

## Workflow

### 1. Find Reparameterization (SVD bases)

```julia
# Stage 1: f=log
S1, N1, N_perp1, r1 = find_invariant_subspace(ϕ_log, log(θ_MLE))

# Stage 2: f=identity
A1 = [N_perp1'; N1']
θ1_to_θ(θ1) = exp(A1' * log(θ1))
ϕ_stage2(θ1) = ϕ(θ1_to_θ(θ1))

S2, N2, N_perp2, r2 = find_invariant_subspace(ϕ_stage2, θ1_MLE)
```

### 2. Filter Active Directions (Jacobian)

```julia
J = compute_ϕ_Jacobian(ϕ_stage2, θ1_MLE)

active_cols = []
for col in 1:size(N_perp2, 2)
    if norm(J * N_perp2[:, col]) > 1e-6
        push!(active_cols, col)
    end
end
```

### 3. Post-Process (Varimax for interpretation)

```julia
# Compute Varimax rotation of Stage 1 basis
N_perp_varimax = varimax_rotation(N_perp1)
R = N_perp1' * N_perp_varimax

# Interpret active directions
for col in active_cols
    v_svd = N_perp2[:, col]
    v_varimax = R' * v_svd[1:size(N_perp1,2)]
    # Display v_varimax to see combination like [1,1] (sum) or [1,-1] (difference)
end
```

### 4. Implement Final Transformation

Two options:

#### Option A: Use Varimax basis for transformation

```julia
# Use Varimax basis for N_perp, SVD for N
A_final = [N_perp_varimax'; N1']

θ_to_ψ(θ) = exp(A_final * log(θ))
ψ_to_θ(ψ) = exp(inv(A_final) * log(ψ))
```

**Pros**: Transformed parameters have interpretable names (e.g., ψ₁ = n₁p₁)
**Cons**: A_final is no longer orthogonal, so inv(A_final) ≠ A_final'

#### Option B: Use SVD basis, rotate only for display

```julia
# Use SVD for transformation
A_final = [N_perp1'; N1']

θ_to_ψ(θ) = exp(A_final * log(θ))
ψ_to_θ(ψ) = exp(A_final' * log(ψ))  # Orthogonal!

# Rotate ψ to Varimax coordinates only for axis labels
ψ_varimax = R' * ψ[1:size(N_perp1,2)]
```

**Pros**: Transformation is orthogonal, numerically stable
**Cons**: Need to handle Varimax rotation when labeling plots

### 5. Generate Profiles

```julia
# Likelihood in transformed coordinates
lnlike_ψ = ψ -> lnlike_θ(ψ_to_θ(ψ))

# Profile each transformed parameter
for i in 1:n_params
    # Use profile_target from ReparamTools
    ψ_grid, ll_grid = profile_target(lnlike_ψ, i, ψ_lower, ψ_upper, ψ_MLE)

    # Plot
    plot(ψ_grid, ll_grid, xlabel="ψ[$i]", ylabel="Log-likelihood")
end
```

## Key Differences from stat_model.jl and stat_sum_model.jl

### Old Approach (stat_sum_model.jl)

- **Builds dictionary into transformation**: Uses `scale_and_round` on both N_perp and N before building A
- **Stage 2 sees scaled basis**: Runs on coordinates with integer exponents
- **More complex**: Dictionary choice affects numerical behavior

### New Approach (Clean Sequential)

- **Dictionary is post-processing**: Both stages use pure SVD bases
- **Stage 2 sees SVD basis**: Runs on orthonormal coordinates
- **Cleaner**: Dictionary choice only affects interpretation, not algorithm

### Backward Compatibility

To match stat_sum_model.jl behavior exactly:
1. After Stage 1, apply `scale_and_round` to N_perp and N
2. Build transformation from scaled bases
3. Run Stage 2 on those coordinates
4. Final rotation is implicit in the scaled bases

For new code, prefer the clean approach with post-processing dictionary.

## Example Output

For sum-of-Poisson model:

```
STAGE 1 (f=log):
  Rank: 1/4
  Dim(N_perp): 2 → span of [n₁p₁, n₂p₂]

STAGE 2 (f=identity):
  Rank: 1/4
  Dim(N_perp): 2 (raw)
  After Jacobian filter: 1 active

VARIMAX INTERPRETATION:
  Active: [1, -1] → ratio n₁p₁/n₂p₂ (identifiable)
  Invariant: [1, 1] → sum n₁p₁ + n₂p₂ (non-identifiable)

TRANSFORMED COORDINATES:
  ψ₁ = n₁p₁  (using Varimax basis)
  ψ₂ = n₂p₂
  ψ₃, ψ₄ (invariant combinations)

PROFILES:
  - ψ₁ vs ψ₂: Shows ridge along ψ₁ + ψ₂ = constant
  - Individual profiles show sum is flat, ratio is curved
```

## Implementation Notes

1. **profile_target** expects integer index, not Pair - call as `profile_target(lnlike, i, ...)`

2. **Bounds** in transformed space need care - use crude transformation of original bounds or compute via optimization

3. **2D profiles** can show ridge structure along invariant direction

4. **Plotting** with Varimax: Either
   - Transform to Varimax coordinates for clean axis labels
   - OR use SVD coordinates with custom axis labels from Varimax interpretation

## Summary

The clean sequential IIR workflow separates:
- **Algorithm** (invariance detection): Uses SVD bases throughout
- **Interpretation** (human readability): Applies Varimax/dictionary after filtering
- **Implementation** (for profiling): Can use either SVD or Varimax basis

This modularity makes the method more flexible and the code cleaner.
