# Repressilator Revisions To-Do

Paper (Maclaren_IIR2025.tex)
- Algorithm 1: document fallback invariance method (finite-difference), remind column ordering (A = [N_perp'; N']), and clarify N_perp vs N roles with repressilator reference.
- Sequential stages: add caveat about orthogonal rotations (e.g. varimax) before Stage 2 to obtain compositionally simple basis; note rotations are orthogonal and maintain spans.
- Examples section: insert repressilator example narrative (full three-mRNA data, finite-difference invariance, β·K products vs K/β ratios, interpretability rotation), with numeric considerations.
- Mechanistic monomial example: ensure Michaelis–Menten limit discussed as pure monomial case running through find_invariant_subspace.
- Minimal vs image: mention repressilator as minimal-image (N non-empty), monomial limit as image-only when basis choice differs.

Code/Repo
- Update examples/repressilator_eisenberg.jl to rotate N_perp and N separately, re-orthonormalize after scale_and_round, and fix SVD diagnostic indices (β at positions 7:9, K at 10:12).
- Add FactorLoadingMatrices to Project.toml/Manifest; note requirement in README.
- Sprinkle comments clarifying ratio/product interpretation and finite-difference rationale.

