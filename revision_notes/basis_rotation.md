# Basis Rotation Between Sequential Stages

- After each call to `find_invariant_subspace` we may apply a local, one-to-one change of coordinates before running the next stage. This does not affect the likelihood value or the invariant/potentially-identifiable split, because the Jacobian at the reference point transforms accordingly.
- Between sequential stages we **prefer** a rotation that yields sparse, parameter-local loadings (mirroring Varimax/simple-structure rotations in factor analysis). Sparse loadings make the next stage of IIR much more informative: the resulting monomials line up with the intended parameter groups.
- For the final stage, where no further composition is performed, we can leave the basis as the raw SVD directions; these already maximise the variance explained (or, in our setting, align with the largest singular values) and provide the identifiability ranking directly.

## Varimax and Compositional Sparsity

- The Varimax objective is a standard way to encourage sparse, interpretable columns. In our context, applying Varimax (with column normalisation and a few random restarts) rotates `N_perp` to a “simple structure” basis before passing it to the next stage.
- Sequential composition benefits directly from this: the rotated monomials are the natural building blocks for the linear stage that follows.
- Detecting whether a rotation is adequate is easy: if the next stage still fails to produce a minimal/image reparameterisation, the intermediate basis was not sparse enough; we can try a different rotation or perturbation.

## References
- I. T. Jolliffe, *Principal Component Analysis*, Springer, 2002.
- H. F. Kaiser, “The varimax criterion for analytic rotation in factor analysis,” *Psychometrika* 23 (1958), 187–200.
- E. Trendafilov, “Sparse loadings extraction in component/factor analysis,” *Computational Statistics & Data Analysis* 61 (2013).
- T. Poggio et al., “Compositional Sparsity in Deep Learning,” arXiv:2403.16963 (2024) – motivation for compositional sparsity.

## Practical Note on the Final Stage
- After the last stage of sequential IIR, there is no further composition, so the original SVD ordering already provides the identifiability ranking. Applying Varimax (or any other rotation) at this point is optional—it may help with presentation, but it is not required for the algorithm to work. In contrast, when a stage feeds into another stage, a sparsity-oriented rotation (Varimax or similar) helps the downstream stage interpret the combinations more effectively.
