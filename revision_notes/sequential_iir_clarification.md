# Sequential IIR: Clarification on Dictionary Approach

## Key Insight
- **Stage 1** and **Stage 2** should be run entirely in the SVD bases produced by `find_invariant_subspace`.
- Any "dictionary" (e.g. Varimax rotation) is a post-processing step for interpretation only.

## Updated Recommendations for the Sum-of-Poisson Example
1. **Use the same Normal approximation** as the original paper and `stat_sum_model.jl`. This keeps the data model identical to the published example so the log-likelihood depends only on \(n_1p_1\) and \(n_2p_2\).
2. **After Stage 2**, compute the Jacobian `J = compute_ϕ_Jacobian(ϕ_stage2, θ1_MLE)` and identify the column(s) of `N_perp_stage2` whose projections `J * N_perp_stage2` have significant norm. Keep those columns and discard the rest (they lie in the output nullspace).
3. **Only then** rotate the surviving column (or columns) with the Stage-1 rotation matrix (Varimax) to display the combination in a sparse form.

The dictionary does not affect the subspaces found by Stage 1 or Stage 2; it only transforms the final basis for readability (e.g. “n₁p₁ + n₂p₂”).

