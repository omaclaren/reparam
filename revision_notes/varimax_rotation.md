# Varimax-based Rotation for Sequential IIR

## Motivation
- The raw right-singular vectors returned by `find_invariant_subspace` form an orthonormal basis of the potentially identifiable subspace, but they need not be sparse. Dense “balanced” columns (e.g. `[1,1,1,1]` / `[1,1,-1,-1]`) make the next sequential stage hard to interpret.
- We previously tried clustering `N_perp` row loadings, projecting templates into the span, and orthonormalising via QR. This proved brittle—perfectly symmetric cases collapsed to a single cluster and QR then manufactured a second column. We no longer rely on that heuristic.
- Instead we adopt a classic **Varimax rotation** (with multiple random restarts) before moving to the next stage. Varimax encourages “simple structure,” i.e. sparse parameter-local loadings, while staying inside the same subspace.

## How it works
1. Normalise the columns of `N_perp` to unit length.
2. Sample random orthogonal rotations (via QR) and apply Varimax (`factor_analyzer.rotator.Rotator` or a custom implementation) to each candidate.
3. Keep the loading matrix with the highest Varimax objective. Re-orthonormalise it (QR) and rescale by the original column norms to stay exactly in the invariant subspace.
4. Optionally threshold/round tiny entries provided the Varimax objective does not drop.
5. Use the rotated basis only when the output will feed into another stage. On the final stage, the raw SVD ordering already yields the identifiability ranking; Varimax is optional (“cosmetic”) there.

## Example (from `factor_test.py`)
```python
import numpy as np
from numpy.linalg import norm, qr
from factor_analyzer.rotator import Rotator

N = np.array([[1, 1], [1, 1], [1, -1], [1, -1]], dtype=float)
col_norms = norm(N, axis=0)
N_norm = N / col_norms

rotator = Rotator(method="varimax")

best_obj = 0.0
best = N_norm
for _ in range(200):
    Q_rand, _ = qr(np.random.standard_normal((N.shape[1], N.shape[1])))
    candidate = N_norm @ Q_rand
    rotated = rotator.fit_transform(candidate)
    obj = float(np.sum(rotated**4, axis=0) - (np.sum(rotated**2, axis=0)**2)/N.shape[0])
    if obj > best_obj + 1e-6:
        best_obj = obj
        best = rotated

Q_final, _ = qr(best)          # re-orthonormalise
rotated = Q_final * col_norms  # rescale columns

threshold = 1e-3
mask = np.abs(rotated) < threshold
rotated[mask] = 0.0
col_norms_thresh = norm(rotated, axis=0)
valid = col_norms_thresh > 0
rotated[:, valid] /= col_norms_thresh[valid]

print("Final rotated columns:\n", np.round(rotated, 6))
```
This simple procedure turns `[1,1,1,1]` / `[1,1,-1,-1]` into sparse columns `[1,1,0,0]` / `[0,0,1,1]` (up to sign), and the next stage can then form `n₁p₁ + n₂p₂` naturally.

## Notes
- With symmetric matrices the Varimax objective has flat ridges. Random restarts help escape the stationary point. If no restart improves the objective, we leave `N_perp` untouched.
- References: Kaiser (1958) for Varimax; Trendafilov (2013) for sparse rotations; Golub & Van Loan for QR projection.
- This interpretability step is optional in the final stage. For sequential composition, however, simple-structure rotations make each stage far more meaningful.
