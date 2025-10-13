ReparamTools.jl
======================

Question: do different variants of Stage 1 give the same Stage 2 (up to rotation)?
- Stage 1 always applies log → linear → exp at some reference point, using the SVD bases of Dϕ.
- Stage 2 repeats the same structure, but using the Stage 1 coordinates.
- If Stage 1 uses a different square basis (e.g., A₁ = Q·Vᵀ instead of Vᵀ), Stage 2 discovers A₂′ = A₂·Q⁻¹ so that the composite A₂·A₁ stays the same. Hence the subspace found is invariant even though the coordinate representation differs by rotation. Varimax should therefore be regarded as a post-processing (display) rotation on the final composite; it need not be applied during the sequential steps.
