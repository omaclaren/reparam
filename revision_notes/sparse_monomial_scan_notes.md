# Sparse monomial scans for interpretable bases of identified subspaces

## Purpose of this note

This note is for **ownership and method clarification**, not for direct insertion into the manuscript. The goal is to explain, in a responsible way:

1. what mathematical problem we are actually trying to solve after the SVD / invariance analysis,
2. how that problem relates to nearby literature,
3. what simple algorithm we are currently using,
4. what the current transport and repressilator results do and do not justify.

The main practical conclusion at present is:

- the **SVD / invariance analysis** gives the numerically discovered target subspace for IIR,
- a **simple fixed monomial dictionary scan** can then be used to build a pool of interpretable basis candidates,
- a secondary basis-selection step is then needed to obtain a **minimal interpretable reparameterisation**,
- on the identified side `N_perp`, that secondary step should likely be **informed** by local `J'J` geometry rather than by simplicity alone,
- on the invariant null side `N`, the search should remain simplicity-based,
- this now looks sufficient for the current transport / repressilator / low-dimensional examples,
- and this is probably a cleaner main story than the historical Varimax-based presentation.

---

## 1. What problem are we trying to solve?

After the invariance test and local linear analysis, we have a chosen numerical target subspace
\[
U \subset \mathbb{R}^p,
\]
typically represented in log-parameter coordinates. In the current examples this target is usually `N_perp`, the orthogonal complement of the invariant null space, though in benign cases this coincides with the purely row-space identifiable directions.

### 1.1 What exactly is the target subspace?

At a reference point `θ*`, the local Jacobian `J = Dϕ(θ*)` gives the usual SVD split
\[
\mathbb{R}^p = V_r \oplus V_0,
\]
where

- `V_r` is the span of the right singular vectors associated with nonzero singular values of `J` (the local row space / locally identifiable directions), and
- `V_0 = \ker J` is the local null space.

The higher-order invariance test used in IIR then splits the local null space into

- `N`, the **invariant null space**, and
- `N_{\mathrm{ni}}`, the **null but non-invariant directions**.

Thus
\[
\mathbb{R}^p = V_r \oplus N_{\mathrm{ni}} \oplus N,
\]
and
\[
N_\perp = V_r \oplus N_{\mathrm{ni}}.
\]

This distinction matters. In general, the target space for an image reparameterisation is `N_perp`, not necessarily just the local row space `V_r`. Only when `N_{\mathrm{ni}} = 0` do we have `N_perp = V_r`. In the current examples the geometry is relatively benign, but the general IIR setting is broader: a scan of `N_perp` may include directions that are not purely row-space identifiable, but are still needed because they lie outside the invariant null space.

The raw SVD basis is useful because it is orthogonal and aligned with local sensitivity geometry. However, the individual SVD basis vectors need not be easy to interpret.

So the interpretability question is:

> Can we find a **simple monomial basis** for the discovered target subspace, so that it yields a minimal interpretable reparameterisation?

If
\[
v = (v_1,\dots,v_p) \in \mathbb{Z}^p,
\]
then in log coordinates this corresponds to the monomial
\[
\psi(\theta) = \prod_{i=1}^p \theta_i^{v_i}.
\]

So the search problem is:

> Given a numerical subspace `U`, find `dim(U)` low-complexity integer exponent vectors `v` whose span matches `U` (approximately or exactly), and use them to define a minimal interpretable reparameterisation.

This is **not** the same as asking for an orthogonal basis, and it is **not** the same as the original SVD problem.

---

## 2. Why the SVD basis is not the whole story

The SVD basis answers a local numerical question:

- which directions are orthogonal,
- which directions correspond to large/small singular values,
- and which coordinates are best separated in the `J'J` sense.

But it does **not** necessarily answer the interpretability question.

This is now very clear in the transport example:

- the raw identified basis is orthogonal and aligned with local sensitivity,
- but the simple interpretable coordinates are sparse ratios like `T1/R`, `T2/R`, `T1/T2`,
- and these are naturally **oblique** rather than orthogonal.

So we need to keep distinct:

1. **orthogonal / local-sensitivity coordinates** (SVD-side), and
2. **simple monomial coordinates** (interpretability-side).

This is the main conceptual split.

---

## 3. Literature map: what is this problem close to?

There is no single standard named method that exactly matches “find a simple monomial basis for a discovered numerical subspace.” The closest literature falls into several nearby strands.

### 3.1 Sparse null-space basis computation

This is the most classical nearby literature.

- **Coleman and Pothen (1986)**, *The Null Space Problem I. Complexity*.
  - Establishes that the null-space sparsity problem is difficult in general.
- **Coleman and Pothen (1987)**, *The Null Space Problem II. Algorithms*.
  - Gives early combinatorial/algorithmic approaches.
- **Gilbert and Heath (1987)**, *Computing a Sparse Basis for the Null Space*.
  - A classic sparse-basis computation paper in numerical linear algebra.
- **Brualdi, Friedland and Pothen (1995)**, *The Sparse Basis Problem and Multilinear Algebra*.
  - A more abstract basis-level treatment of sparse-basis questions, useful because our present concern is an interpretable **basis** for the target subspace rather than only a single sparse direction.

This strand is closest when the target subspace is literally a row space or null space of a matrix. That is already quite relevant here, because the local SVD stage does produce row/null-space geometry from the Jacobian, even if the final IIR target may be the larger complement `N_perp = V_r \oplus N_{\mathrm{ni}}`. It is also relevant more broadly because the current problem is often about sparse **spanning sets** or **bases**, not just isolated directions.

### 3.2 Finding sparse vectors in a subspace

This is even closer to the current use case.

- **Demanet and Hand (2014)**, *Scaling law for recovering the sparsest element in a subspace*.
- **Qu, Sun and Wright (2016)**, *Finding a Sparse Vector in a Subspace: Linear Sparsity Using Alternating Directions*.

These papers are directly about recovering **sparse directions** lying inside a known subspace. This is close in spirit to the present problem, but not identical. In particular, our present goal is not merely to recover a single sparsest vector, but to obtain a **simple basis** of monomial directions for the discovered target subspace. Moreover, our notion of simplicity is not arbitrary sparsity alone, but rather **small-support monomials with small integer exponents**. The current implementation therefore uses a fixed bounded monomial dictionary scan plus a secondary basis-selection step rather than a general sparse optimization procedure.

### 3.3 Simple-structure rotations / factor rotations

This is the literature strand historical Varimax came from.

- **Kaiser (1958)**, *The Varimax Criterion for Analytic Rotation in Factor Analysis*.

This strand is relevant because it is explicitly about rotating a basis inside a fixed subspace to get a more interpretable basis. However, the typical outputs are continuous rotated loadings, not exact integer monomial directions. So it is adjacent, but not the same problem.

### 3.4 Sparse PCA / sparse components

- **Zou, Hastie and Tibshirani (2006)**, *Sparse Principal Component Analysis*.

This is nearby because it also tries to combine low-dimensional structure with sparse interpretable loadings. But sparse PCA typically balances variance explanation and sparsity, whereas here the subspace is already fixed by the invariance/SVD stage, and the problem is only how to describe that subspace interpretably.

### 3.5 Fixed-library sparse model discovery

- **Brunton, Proctor and Kutz (2016)**, *Discovering governing equations from data by sparse identification of nonlinear dynamical systems*.

This strand is not a basis-of-a-subspace literature, but it is conceptually useful because it searches a **fixed library** of simple candidate terms (often monomials) for a sparse interpretable description. In that sense it is a better analogy for the present implementation than dictionary learning proper: we are not learning a dictionary from data, but scanning a pre-specified monomial library for useful structure inside a discovered subspace.

### Working conclusion on the literature

The present problem is best described as **related to sparse-basis and sparse-vector-in-a-subspace problems**, with some conceptual overlap with simple-structure rotations, sparse PCA, and fixed-library sparse model discovery. The mathematically closest strands are still sparse-basis / sparse-subspace problems, especially because the final object here must be a **basis** for a minimal reparameterisation. However, the current procedure should not be presented as a general solver for any of those problems. Rather, it is a **simple bounded dictionary scan plus a secondary basis-selection step for low-complexity monomial bases inside a discovered numerical subspace**.

---

## 4. What should we *not* claim?

We should **not** claim that the current procedure solves the general problem of finding the sparsest basis of a numerical subspace, or even the sparsest vector in a subspace. Those are broader optimization problems studied in the literature. We should also not claim that the current procedure produces a unique interpretable basis, the globally sparsest basis, or a general replacement for the broader sparse-subspace literature.

That would open a can of worms.

Instead, the defensible claim is much more modest:

> If the discovered subspace admits a useful low-complexity monomial basis within a small fixed dictionary, then a simple scan-plus-selection procedure can expose that structure.

This is strong enough for the present examples and does not overreach.

---

## 5. The current algorithmic idea

### 5.1 Target of the scan

At present, the scan is applied to a chosen numerical target subspace `U`, usually `N_perp`.

This is deliberate:

- it keeps the method simple,
- it focuses on the coordinates of interpretive interest,
- and it avoids dragging in additional null-space logic too early.

However, it is important not to collapse `N_perp` automatically to the purely row-space identifiable directions. In general,
\[
N_\perp = V_r \oplus N_{\mathrm{ni}},
\]
so the scan may include directions that are locally null but not invariant. In the present transport/repressilator/mm examples this distinction is relatively mild, but the general target of the scan is the complement of the invariant null space, not merely `V_r`.

A dedicated null-space scan may still be useful later, but it is not needed for the current phase-1 method.

### 5.2 Candidate dictionary and simplicity criterion

A candidate monomial direction is a primitive integer exponent vector
\[
v \in \mathbb{Z}^p.
\]

The current scan uses a **fixed bounded dictionary**:

- support bound `s_max`,
- coefficient bound `c_max`,
- primitive reduction,
- identify `v` and `-v`.

For the current first cut we tested mainly:

- `s_max = 2`,
- `c_max = 1`.

This means the scan checks:

- singleton directions,
- pairwise products,
- pairwise ratios.

The objective is therefore **not** to find the globally sparsest element of the subspace. Rather, it is to test whether the subspace contains **simple monomial basis candidates from a small pre-specified dictionary**.

### 5.3 Subspace score

Given an orthonormal basis `Q` for the identified subspace `U`, and a candidate `v`, we compute the residual
\[
r_U(v) = \left\|(I - QQ^T) \frac{v}{\|v\|_2}\right\|_2.
\]

A small residual means the monomial direction lies in the identified subspace.

### 5.4 Output of the scan and the final object of interest

The **first-stage output** of the scan is the set of accepted simple directions.

This accepted set is important, but it is **not** the final IIR object of interest. For reparameterisation, the final object must be a **basis** of the target subspace `U`, of size `dim(U)`, because that is what gives a **minimal reparameterisation**. A redundant spanning set may be diagnostically useful, but it is not sufficient as the final answer.

So the present procedure should be viewed in two stages:

1. build a feasible pool of simple monomial candidates that are approximately contained in `U`;
2. choose from that pool a simple basis whose span matches `U`.

### 5.5 Search vs optimisation: what is the current method actually doing?

The current procedure is best described as a **finite dictionary scan followed by a basis-selection problem**.

The first stage is not a general continuous optimisation over all vectors in the subspace. Instead, it fixes a finite dictionary `𝒟` of primitive integer exponent vectors and checks each one for approximate containment in the target subspace. In that sense it is closer to a feasibility scan or fixed-library search than to a full sparse-basis solver.

The second stage is the real minimal-reparameterisation step: among the accepted candidates, choose `d = dim(U)` linearly independent vectors whose span matches `U`. That part **does** have a combinatorial optimisation flavour. The current implementation should still be regarded as modest and heuristic, but this basis-selection stage is essential rather than optional.

### 5.6 What should the basis-selection step optimise?

The clearest current view is:

- use residual-to-subspace mainly for **accept/reject**, and at most as a tie-breaker once candidates are already accepted;
- use the accepted set as the feasible pool for a genuine **basis** problem;
- treat the real objective as finding a **minimum-complexity basis among accepted candidates** whose span matches the target subspace.

Concretely, once candidates have passed the containment test, the main remaining issue is not whether one accepted vector is infinitesimally closer to the target subspace than another, but whether it contributes genuinely new direction beyond what has already been selected. So the right version of “diversity” is really **incremental span gain / linear independence after projection into the target subspace**.

A natural abstract formulation would therefore be: choose a subset `B = {v_1, ..., v_d}` of accepted candidates, with `d = dim(U)`, such that `span(B)` matches the target subspace while minimizing a complexity score such as support, `ℓ₁` size, or a lexicographic combination of small-support / small-coefficient criteria. A basis-level subspace distance should then be used to verify that `span(B)` is a good numerical match to `U`.

For the present phase, the clean narrow implementation is intentionally simple:

1. enumerate the bounded dictionary;
2. compute each candidate residual to `U`;
3. relax the admissible residual threshold only as needed, by scanning the sorted distinct residual levels (up to a loose residual cap);
4. at each threshold, sort accepted candidates lexicographically by simplicity, e.g. `(support, ℓ₁, ℓ_∞, residual)`;
5. sweep through this ordered list and retain a candidate if its projection into `U` increases the rank of the already selected projected candidates;
6. stop at the **first** threshold for which `d = dim(U)` basis vectors are obtained and the final span matches `U`;
7. report the effective selected residual level together with the final basis-level subspace match.

This keeps the role of each ingredient clear:

- **residual** is mainly a feasibility filter, relaxed only as needed rather than tuned to target a candidate count;
- **simplicity ordering** expresses interpretability preference;
- **rank increase after projection into `U`** is the right notion of “diversity” / new information;
- **first successful threshold** avoids over-relaxing the feasible pool;
- **final subspace match** confirms that the selected basis really yields the desired minimal reparameterisation.

The current greedy basis extraction should therefore be understood as a deliberately simple first practical implementation of the basis problem, not yet the final word.

### 5.7 Informed basis ordering on the identified side

The exact-invariance cases suggest that simplicity-first basis extraction can already work well, but the full-rank non-limit examples show that **basis selection on `N_perp` should not be based on simplicity alone**.

Once a primitive candidate exponent vector `v` has been accepted into the identified-side feasible pool, score its **normalized direction**
\[
\hat u = \frac{v}{\|v\|_2}
\]
by its local information content
\[
I(\hat u) = \hat u^T M \hat u = \|J\hat u\|_2^2,
\]
where `M = J^T J`.

Because the absolute magnitude of `I(\hat u)` depends strongly on the overall scale of the model and the Jacobian, cross-example comparisons should be made using the **relative** quantity
\[
I_{\mathrm{rel}}(\hat u) = \frac{\hat u^T M \hat u}{\sigma_1^2},
\]
where `σ₁` is the leading singular value of `J`.

Given an already selected identified-side basis block `B` (with columns interpreted as normalized selected directions), the new information contributed by `\hat u` can be measured by the conditional gain
\[
\Delta(\hat u \mid B)
= \hat u^T M \hat u - \hat u^T M B\,(B^T M B)^{-1} B^T M \hat u.
\]
This is the Schur-complement amount of `J'J` information in `\hat u` that is not already explained by the selected directions in `B`.

This suggests the following refinement.

- On the identified side `N_perp`, first use the monomial scan only to build a feasible pool of simple candidates.
- Then choose an **ordered basis** greedily by local information: first maximize `I(\hat u)`, then maximize `Δ(\hat u \mid B)` at later steps, using simplicity/residual only as tie-breaks.
- On the invariant null side `N`, do **not** use this metric: there `J\hat u = 0`, so informedness is meaningless and the search should remain simplicity-based.

This is a narrow refinement, not a grand optimization claim. It keeps the simple bounded-dictionary search, but uses `J'J` geometry where it is actually informative.

### 5.8 Numerical tolerances should not be conflated

There are several distinct numerical thresholds in play:

- a rank tolerance for splitting `V_r` and `V_0` at the Jacobian stage,
- an invariance tolerance for splitting `V_0` into `N` and `N_{\mathrm{ni}}`,
- a monomial-scan tolerance for deciding approximate containment in the chosen target subspace.

These are conceptually different and should be kept separate. Approximate containment in the scan is not an ad hoc rounding trick: the candidates themselves are exact primitive integer vectors, and only the membership test is approximate because the target subspace is known numerically.

---

## 6. Why the accepted pool still matters even though the final object is a basis

This became clear in the transport example.

The scan accepted the three support-2 ratio directions:

- `T2/R`
- `T1/T2`
- `T1/R`

These satisfy a simple algebraic relation, so any two form a basis of the identified plane. The final reparameterisation still requires choosing a **basis** of size 2, but the accepted pool tells us something important first: the plane admits a very simple ratio description inside the chosen dictionary.

So the robust interpretive statement is:

> the identified plane admits a minimal interpretable basis drawn from a simple sparse ratio family.

The family itself is useful as the feasible pool; the final output is then a basis chosen from that pool.

By contrast, in repressilator the accepted directions more or less force the expected interpretable basis structure, so the subsequent basis-selection problem is much less ambiguous.

---

## 7. Current empirical findings

### 7.1 Transport

The transport diagnostics established:

- exact null direction `(1,1,1)`;
- identified plane `a + b + c = 0`;
- a stable orthogonal rounded SVD basis from direct rounding of the raw SVD-derived coordinates;
- but also a simple sparse ratio family discovered by the fixed monomial scan:
  - `T2/R`
  - `T1/T2`
  - `T1/R`.

Interpretation:

- the SVD basis is useful for orthogonality/local sensitivity;
- the sparse monomial scan is useful for constructing a simple minimal basis;
- these are **different goals**;
- transport shows clearly that interpretability can naturally be **oblique**;
- on the simple ratio family, the relative directional-information scores are strongly uneven (`T2/R` is much stronger than `T1/T2` or `T1/R`), but after selecting `T2/R` the remaining simple directions `T1/T2` and `T1/R` have essentially the same conditional information gain because `T1/R = T1/T2 + T2/R` in log-exponent coordinates;
- raw `u^T J^T J u` or `Δ(u \mid B)` values can look numerically large in transport because the Jacobian scale itself is large, so these quantities should be interpreted relative to `σ₁²` rather than by absolute magnitude alone;
- in this case, the scan first reveals the ratio-family candidate pool, and then any two independent members give a minimal 2-dimensional reparameterisation.

### 7.2 Repressilator

From the saved canonical result `nesi/repressilator_16nuisance_50x50_results.jls`, the fixed scan with

- `support ≤ 2`,
- coefficients in `{-1,0,1}`

already recovers a full interpretable identified basis structure:

- 12 singleton directions,
- `β1/K1`, `β2/K2`, `β3/K3`.

The selected scan basis spans the same identified subspace as the saved profile basis to numerical precision.

Interpretation:

- a simple sparse monomial scan already recovers the identified repressilator basis structure,
- the accepted pool is rich enough that the resulting minimal basis is essentially forced,
- a further read-only informedness diagnostic can be run cheaply from the saved `θ_MLE` plus the standard high-precision repressilator IIR map, without a fresh NeSI rerun,
- that informedness diagnostic suggests that the sparse identified family itself remains the right one, while `J'J` geometry mainly supplies a **nontrivial ordering** inside that family rather than a different sparse basis,
- in the current calculation the strongest relative identified-side candidates begin with `β3/K3`, `β1/K1`, then `α3`, `k_degp2`, ... and the greedy conditional-information order begins `β3/K3`, `β1/K1`, `k_degp2`, `k_degp1`, `k_degp3`, ...
- so Varimax is not needed to justify interpretability here.

This does **not** by itself settle whether the profiling run should be rerun under a search-based basis construction; that depends on whether the actual profiled basis/targets are changed materially. But it strongly weakens the need to keep Varimax as the main conceptual story.

### 7.3 Low-dimensional sanity checks: `stat_model` and `mm_model`

The simple basis-search draft was also tested on the low-dimensional examples.

- **stat_model, Poisson limit**: the target space is 1-dimensional and the search recovers the expected monomial basis `np`.
- **mm_model, limit case**: the target space is 1-dimensional and the search recovers the expected monomial basis `ν/K` (equivalently `K/ν` up to sign convention in log-exponent space).

These are encouraging because they show the method behaves naturally in exact invariant low-dimensional cases.

The full-rank non-limit cases are more nuanced.

- **stat_model, non-limit**: the target space is the full 2-dimensional space, and the simplicity-first search returns a simple basis of that full space (`n`, `p`), with `n/p` also appearing as an accepted alternative. But the informed calculation is more revealing: `np` carries almost all of the strongest identified-side information, and after selecting `np`, the largest conditional gain comes from `n/p`. So an informed identified-side basis would naturally be ordered as `np`, then `n/p`.
- **mm_model, non-limit**: the target space is again full 2-dimensional, and the first successful threshold returns a valid simple basis (`K`, `νK`). But the informed calculation again separates strong and weak roles more cleanly: `ν/K` carries essentially all of the strongest identified-side information, while the weak complementary direction is close to `νK`. So an informed identified-side basis would naturally be ordered as `ν/K`, then `νK`.

These full-rank checks are useful because they show both the strength and the limitation of the current narrow method. A simplicity-first search does find simple valid bases of the target space, but in practical-identifiability cases that is not necessarily the same thing as finding coordinates aligned with the strong/weak local information structure. This is the main reason the identified-side search should now probably be refined to an **informed** basis search on `N_perp`, while a raw SVD basis, or a rounded orthogonal presentation of it, still remains useful when the goal is to display local weak-direction structure rather than only to produce a simple minimal basis.

---

## 8. Why Varimax is no longer the best main story

Historically, Varimax made sense as a way to rotate dense SVD directions into something more interpretable.

However:

- in **transport**, Varimax is not conceptually well-matched because the natural interpretable coordinates are oblique rather than orthogonal;
- in **repressilator**, a simple monomial scan already seems sufficient to recover the interpretable structure.

So the current cleaner methodological split is:

- **SVD / invariance analysis** for orthogonal/local-sensitivity coordinates,
- **fixed monomial dictionary scan plus basis selection** for interpretable minimal coordinates.

Varimax can remain historical background or an exploratory heuristic, but it no longer looks like the clean main method.

---

## 9. How the current method should be described

A good internal summary is:

> We use a simple fixed bounded scan over primitive monomial candidates to build a feasible pool of interpretable directions for a chosen numerically discovered target subspace; on the identified side we then select an informed ordered basis from that pool using local `J'J` information, while on the invariant null side we keep the basis selection simplicity-based.

Important features of this wording:

- **simple**: not a full optimization framework,
- **fixed bounded scan**: finite, pre-specified dictionary,
- **primitive monomial candidates**: integer exponent vectors up to sign/scaling,
- **feasible pool**: the scan is a screening stage rather than the final answer,
- **identified-side informed ordering**: use local information only where it is meaningful,
- **null-side simplicity**: do not force an information metric where `Ju = 0`,
- **basis-level final object**: IIR still needs a minimal reparameterisation rather than a redundant family,
- **minimal reparameterisation**: interpretability-first, without claiming global optimality.

### 9.1 Subspaces are primary; bases are different views

The primary IIR objects are the subspaces themselves:

- the identified-side target space `N_perp`,
- and, when relevant, the invariant null space `N`.

Different bases are best understood as different **views** of the same subspace rather than rival claims about what the subspace is.

For the identified side `N_perp`, the current best three-way split is:

- **A. SVD basis**
  - orthogonal, local-sensitivity / strong-weak geometry,
  - canonical up to the usual repeated-singular-value rotations,
  - often dense and not especially interpretable.

- **B. Singleton-first sparse basis**
  - structural / interpretability view,
  - search the bounded monomial dictionary in increasing support, keeping singleton directions where possible and then adding the smallest-support combinations needed to complete the span,
  - this is also the natural null-side strategy, because on `N` there is no informative `J'J` preference.

- **C. Stepwise most-informative simple basis**
  - identified-side only,
  - dictionary-constrained analogue of a sequential SVD construction,
  - first maximize `I_rel`, then at later steps maximize the conditional gain `Δ(\hat u \mid B)` / `Δ_rel(\hat u \mid B)`, i.e. the best next simple direction after accounting for what the already selected simple directions explain,
  - this should not be described too literally as “largest projection onto the next SVD vector”, because once the earlier selected simple directions fail to align exactly with the earlier singular vectors, the deflated problem is no longer the same.

In this language:

- **A** shows orthogonal geometry,
- **B** shows structural simplicity / individual recoverability,
- **C** shows the best SVD-like basis available inside the chosen simple dictionary.

On the invariant null side `N`, only **A** and **B** are meaningful. A strict `J'J`-based analogue of **C** does not exist there, because every direction in `N` satisfies `J\hat u = 0`. In benign one-dimensional null cases this is still easy to recover from **B** alone; for example, in the Poisson-limit `stat_model`, the null line is spanned by `(1,-1)` in log-exponent coordinates, so the null-side simple search recovers `n/p` rather than `n` simply because `[1,-1]` lies in `N` while `[1,0]` does not.

---

## 10. Open questions

### 10.1 How much of this enters the main manuscript?

Likely main-manuscript level claim:

- the identified/null subspaces are the primary IIR objects;
- these subspaces can be understood through different basis views rather than one compulsory coordinate system;
- SVD gives orthogonal/local-sensitivity coordinates;
- a singleton-first sparse basis gives a structural/interpretable basis, keeping individual parameters where possible and introducing simple combinations only when needed;
- on the identified side, a stepwise most-informative simple basis gives a dictionary-constrained SVD-like basis;
- on the invariant null side, a simple monomial basis search remains sufficient;
- in general sparse-basis recovery in numerical subspaces is nontrivial;
- here a simple bounded scan-plus-selection procedure is enough for the examples.

### 10.2 Should repressilator be rerun?

If the final basis used for the profiling workflow is changed materially, a clean NeSI rerun is probably the conservative thing to do.

If the recovered basis is effectively the same at the level of actual profiled interest coordinates, then a rerun may not be necessary. But if the interpretability story is rewritten, it is probably better to decide that explicitly rather than leave it ambiguous.

### 10.3 Should the null space also be scanned later?

Yes, probably — but as a **second pass** rather than as part of the identified-side informed search itself.

The current best picture is:

- search `N_perp` first with an informed monomial basis selection, because that is where local `J'J` geometry is meaningful;
- then search `N` with a simplicity-based monomial basis selection, because there `Ju = 0` and informedness is not meaningful;
- concatenate the two bases only when a full square reparameterisation is wanted.

For the present interpretability diagnostics, the identified-side search is still the primary object.

---

## 11. Practical next step

The current implementations to understand and inspect are:

- `sparse_monomial_scan_diagnostic.jl`
- `simple_monomial_basis_selection_draft.jl`

The current conceptual next step is:

1. implement the identified-side informed basis selection in `simple_monomial_basis_selection_draft.jl` using `I(u)` / `Δ(u \mid B)` with simplicity and residual only as tie-breaks,
2. implement the null-side simplicity-based search as a second pass when a full square reparameterisation is wanted,
3. report both the broader accepted family and the final selected basis, with information scores given in relative form where cross-example comparison is intended,
4. then settle the wording/scope of the scan-plus-basis-selection story and decide whether to demote/remove Varimax from the main narrative,
5. only then decide whether a clean repressilator rerun is needed for consistency.

---

## References mentioned in this note

- Coleman, T. F. and Pothen, A. (1986). *The Null Space Problem I. Complexity*. SIAM Journal on Algebraic and Discrete Methods 7(4), 527–537.
- Coleman, T. F. and Pothen, A. (1987). *The Null Space Problem II. Algorithms*. SIAM Journal on Algebraic and Discrete Methods 8(4), 544–563.
- Gilbert, J. R. and Heath, M. T. (1987). *Computing a Sparse Basis for the Null Space*. SIAM Journal on Algebraic and Discrete Methods 8(3), 446–459.
- Brualdi, R. A., Friedland, S. and Pothen, A. (1995). *The Sparse Basis Problem and Multilinear Algebra*. SIAM Journal on Matrix Analysis and Applications 16(1), 1–20.
- Demanet, L. and Hand, P. (2014). *Scaling law for recovering the sparsest element in a subspace*. Information and Inference 3(4), 295–329.
- Qu, Q., Sun, J. and Wright, J. (2016). *Finding a Sparse Vector in a Subspace: Linear Sparsity Using Alternating Directions*. IEEE Transactions on Information Theory 62(10), 5855–5880.
- Kaiser, H. F. (1958). *The Varimax Criterion for Analytic Rotation in Factor Analysis*. Psychometrika 23, 187–200.
- Zou, H., Hastie, T. and Tibshirani, R. (2006). *Sparse Principal Component Analysis*. Journal of Computational and Graphical Statistics 15(2), 265–286.
- Brunton, S. L., Proctor, J. L. and Kutz, J. N. (2016). *Discovering governing equations from data by sparse identification of nonlinear dynamical systems*. Proceedings of the National Academy of Sciences 113(15), 3932–3937.
