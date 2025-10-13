import numpy as np
from numpy.linalg import norm, qr
from factor_analyzer.rotator import Rotator

# example matrix with columns (1,1,1,1) and (1,1,-1,-1)
N = np.array([[1, 1],
              [1, 1],
              [1,-1],
              [1,-1]], dtype=float)


def varimax_objective(M):
    p, _ = M.shape
    term = np.sum(M**4, axis=0)
    correction = (np.sum(M**2, axis=0)**2) / p
    return float(np.sum(term - correction))


def rotate_varimax(N, n_restarts=100, tol=1e-6, random_state=None):
    rng = np.random.default_rng(random_state)
    col_norms = norm(N, axis=0)
    M_norm = N / col_norms
    rotator = Rotator(method="varimax")
    best_obj = varimax_objective(M_norm)
    best = M_norm.copy()

    for _ in range(n_restarts):
        Q_rand, _ = qr(rng.standard_normal((N.shape[1], N.shape[1])))
        candidate = M_norm @ Q_rand
        rotated = rotator.fit_transform(candidate)
        obj = varimax_objective(rotated)
        if obj > best_obj + tol:
            best_obj = obj
            best = rotated

    Q_final, _ = qr(best)
    return Q_final * col_norms, best_obj


rotated, obj_best = rotate_varimax(N, n_restarts=200, random_state=0)
print("Rotated columns:\n", np.round(rotated, 6))
print("Varimax objective (best):", obj_best)

# Optional thresholding/rounding stage
threshold = 1e-3
Q_thresh = rotated.copy()
mask = np.abs(Q_thresh) < threshold
Q_thresh[mask] = 0.0

# re-normalise columns after thresholding
col_norms = norm(Q_thresh, axis=0)
# avoid division by zero if a column became zero
valid = col_norms > 0
Q_thresh[:, valid] = Q_thresh[:, valid] / col_norms[valid]
if np.any(~valid):
    print("Warning: a column became zero after thresholding; keeping untrimmed column(s).")
    Q_thresh[:, ~valid] = rotated[:, ~valid]

obj_thresh = varimax_objective(Q_thresh)
if obj_thresh >= obj_best - 1e-6:
    print("Using thresholded columns (objective maintained).")
    rotated = Q_thresh
else:
    print("Thresholding degraded objective; reverting to unthresholded columns.")

print("Final rotated columns:\n", np.round(rotated, 6))
print("Final objective:", varimax_objective(rotated))
