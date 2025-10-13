Great brief, Oliver. You want **compact, instructive models (≈5–20 parameters)** where *only combinations of parameters are identifiable*, and where those combinations are **explicitly extracted/reparameterised**. Below is a curated review with concrete candidates you can drop straight into figures or notebooks to showcase your new method. I’ve prioritised examples where the paper itself spells out the combinations (or derives them from profiles) and where reparameterisation is natural.

---

## A. Compact “show‑piece” models with explicit identifiable combinations

> **Legend**: *What’s measured* → outputs; **Combos** → functions of parameters the authors show are identifiable; **Why useful** → what your method can illustrate quickly.

1. **Linear 2‑compartment PK model (5 params)**

   * **Equations:** two compartments (x_1,x_2), exchange (k_{12},k_{21}), losses (k_{01},k_{02}); output (y=x_1/V).
   * **What’s measured:** (y(t)) only.
   * **Combos (explicit from profiles):** (k_{12}k_{21}), (k_{12}+k_{02}), (k_{21}+k_{01}); (V) identifiable alone. The paper fits rational relations on profile plateaus and then factors them to read off the combinations.
   * **Why useful:** tiny model; profiles cleanly expose flat directions; combinations are sums/products (easy to plot / algebraically verify); perfect “first example”. 

2. **2‑compartment variant with a rank‑deficient pair (6–7 params)**

   * **What’s measured:** (y=x_1/V) as above.
   * **Combos (explicit):** (k_2+k_3+k_4), (k_1+k_5), and (k_1k_4) (the paper shows why naive profiling fails when there are “loose” degrees of freedom and how to fix it with subset‑FIM preconditioning before profiling).
   * **Why useful:** lets you demonstrate your method’s handling of **overlapping combinations** and the “loose parameter” pitfall (too many free params in a connected component). 

3. **Thyroid hormone mini‑model (7 params; non‑rational)**

   * **Equations include:** sinusoidal drive and an exponential; outputs include TSH.
   * **Combos (explicit):** (c,k_{34}) is identifiable; (c) and (k_{34}) are not separately.
   * **Why useful:** showcases you can extract combos **beyond rational ODEs** (where some algebraic tools don’t apply). Very compact, visually convincing profile (c) vs (k_{34}). 

4. **Repressilator (synthetic gene circuit; 19 params)**

   * **What’s measured:** three mRNA time series.
   * **Combos (explicit from fits):** (K_1/\beta_1), (K_2/\beta_2), (K_3/\beta_3) (Hill repression constants over translation rates).
   * **Why useful:** “biggish” nonlinear model still under 20 parameters; combinations have a **clear biological interpretation (effective regulation strength)** and the paper plots the β–K relations directly. 

5. **Nonlinear 2‑compartment model with explicit canonical reparameterisation (≈8 params)**

   * **Source/method:** differential‑algebra + Gröbner bases; authors *construct* identifiable combinations and then **rewrite the model in those variables**.
   * **Combos (explicit, denoted (q_i)):**
     (q_1=b_1c_1), (q_2=c_1K_M), (q_3=k_{02}+k_{12}), (q_4=c_1V_Mk_{12}k_{21}), (q_5=c_1V_M(k_{01}+k_{21})).
     They show the input–output coefficients rewritten uniquely in terms of (q), and then give a canonical ODE in (q).
   * **Why useful:** a **textbook reparameterisation example**—you can reproduce the algebra and then compare your method’s output (q(\theta)) to theirs. 

6. **“Classical” one‑compartment oral PK (illustrative micro‑case)**

   * **What’s measured:** concentration (C(t)); unknown bioavailability (F) and volume (V).
   * **Combo (explicit statement):** only (F/V) is identifiable from concentration–time data; separate (F) and (V) are not without extra info.
   * **Why useful:** tiny vignette to motivate *why* combinations matter (simple figure: identical fits for different ((F,V)) lying on a hyperbola with fixed (F/V)). ([Frontiers][1])

---

## B. Medium models & families where combinations are provably extractable

* **Linear compartment models (LCMs): identifiable scaling reparameterisations**
  Meshkat & Sullivant give graph‑theoretic conditions and prove that when an identifiable scaling reparameterisation exists, you can take it to be **monomial scalings** (i.e., explicit products of parameters). Great for making a clean “before/after” demo on a 4–6 compartment mammillary/catenary model. ([arXiv][2])

* **Profile‑likelihood “groups” that imply scale‑type combinations (IL‑13/JAK‑STAT benchmark)**
  Raue et al. show how flat profiles trace **functional relationships** among parameters and cluster into groups that amount to *scale × production* combinations for mRNAs. While the full model has >20 parameters, you can **lift a 5–10 parameter sub‑module** to stay within your target size and still reproduce the “profile‑derived combination” story (and contrast with your method if you extract the formula directly). ([OUP Academic][3])

* **Pharmacodynamic building blocks (effect compartment, receptor binding, turnover)**
  Janzén et al. analyse 16 fundamental PD structures and explicitly discuss when reparameterisation (or fixing one parameter) resolves non‑identifiability; the paper **recommends and documents** reparameterisations (e.g., for receptor binding + effect compartment). This gives you a **menu of 6–12‑parameter models** with known combo structure. ([Frontiers][1])

* **Symbolic tools that return combinations (for triangulation/validation):**

  * **COMBOS** (Meshkat et al.) outputs identifiable combinations using differential algebra; use it as a cross‑check against your method on the same models. ([PLOS][4])
  * **STRIKE‑GOLDD** can search for **identifiable reparameterisations** (via differential geometry and symmetries); the benchmarking review summarises this capability. ([PMC][5])
  * **SIAN / Ovchinnikov–Pogudin–Scanlon**: algorithms to compute **all identifiable functions** (not just single parameters), useful when you want a “ground‑truth” set of combos on rational models. ([arXiv][6])

---

## C. Ready‑to‑use reparameterisations (lift straight into your paper/notebook)

Below I restate the combinations in a notation that’s easy to drop into code. For each model, define the θ’s and then **fit/plot in the combo space**.

1. **PK 2‑compartment (Example 1 above):**
   [
   \theta_1=k_{12}k_{21},\quad \theta_2=k_{12}+k_{02},\quad \theta_3=k_{21}+k_{01},\quad \theta_4=V.
   ]
   Fit in (\theta)-space; show that many ((k_{12},k_{21},k_{01},k_{02})) map to the same (\theta). 

2. **PK 2‑compartment with rank deficiency (Example 2):**
   [
   \theta_1=k_2+k_3+k_4,\quad \theta_2=k_1+k_5,\quad \theta_3=k_1k_4.
   ]
   Demonstrate why **subset‑FIM preconditioning** (or your method’s analogue) is required before profiling to avoid loose directions. 

3. **Thyroid model (non‑rational):** (\ \theta_1=c,k_{34}). Flat profile in (c) with reciprocal in (k_{34}); a single simple figure conveys the point. 

4. **Repressilator:** (\ \theta_{i}=K_i/\beta_i,\ i=1,2,3). Show the three β–K curves from profiles collapsing to constants in (\theta)-space. 

5. **Nonlinear 2‑compartment (Meshkat 2011):**
   [
   q_1=b_1c_1,\quad q_2=c_1K_M,\quad q_3=k_{02}+k_{12},\quad q_4=c_1V_Mk_{12}k_{21},\quad q_5=c_1V_M(k_{01}+k_{21}).
   ]
   Reproduce the authors’ “canonical‑form” model written **entirely** in (q). (It makes a striking side‑by‑side diagram: original vs. reparameterised.) 

6. **One‑compartment oral PK:** (\ \theta_1=F/V). Put two parameter sets with different (F) and (V) on the same (\theta_1) and show identical (C(t)). (Short didactic inset.) ([Frontiers][1])

---

## D. How I’d stage your demonstration (to highlight your method’s value)

1. **Start small, then escalate.**

   * **“Hello world”**: PK 2‑compartment (Ex. 1) — extract (\theta_{1..3}) numerically and (if your method is symbolic/analytic) output the exact forms; compare to Eisenberg’s profile‑fitted relations. 
   * **“Loose pair” stress‑test**: modified PK (Ex. 2) — show your method *still* returns the right combos without manual subset selection. 
   * **Non‑rational**: thyroid mini‑model — demonstrate you don’t require polynomial/rational structure. 
   * **“Bigger nonlinear”**: repressilator — scale to ~20 parameters, recover (K_i/\beta_i). 

2. **Side‑by‑side with established tools** (for credibility, not to knock them):

   * Run **COMBOS** / **STRIKE‑GOLDD** / **SIAN** where applicable; show agreement on the set of combos. Position your method as *simpler*, *faster*, or *wider‑applicability* (e.g., handles non‑rational terms), depending on your actual strengths. ([PLOS][4])

3. **Make the “combo” the parameter.**

   * Refit each model directly in (\theta)-space (reparameterised ODE)—you should see tight posteriors, no flat directions. This is literally the point of the exercise and makes a powerful visual (e.g., narrow likelihoods for (K_i/\beta_i) where (K_i,\beta_i) profiles were flat). 

4. **Optional “family result”.**

   * Include one **linear compartment** example (e.g., 4‑compartment mammillary) and cite the **monomial scaling reparameterisation** existence result—helps generalise your method beyond single exemplars. ([arXiv][2])

---

## E. Pitfalls worth anticipating (and how your method can show it handles them)

* **Excess degrees of freedom in a combination component** → flat profiles that *don’t* reveal a unique relation unless you restrict the subset (Eisenberg’s Example 2). If your method resolves this automatically, it’s a clear advantage. 
* **Scale ambiguities** (measurement scaling × production rate): show you return the *product/ratio* as the identifiable function without needing extra experiments, then discuss what experiment would disambiguate if individual parameters are desired (Raue 2014). ([OUP Academic][3])
* **Non‑rational dynamics**: algebraic tools may stumble; your method extracting (ck_{34}) in the thyroid model makes the case crisp. 

---

## F. Pointers to methodology papers you can cite alongside your method

* **Profile‑likelihood as a combo detector** (profiles trace functional relationships; structural vs practical non‑identifiability): Raue et al. 2009. ([PubMed][7])
* **FIM‑guided subset profiling to *compute* combos** (and fit rational functions to read them off): Eisenberg & Hayashi; includes the 2‑compartment, thyroid, and repressilator case studies above. 
* **Differential‑algebraic extraction and canonical reparameterisation** (nonlinear ODEs): Meshkat, Anderson & DiStefano 2011. 
* **Monomial scaling reparameterisations for LCMs** (graph conditions; existence and construction): Meshkat & Sullivant 2013. ([arXiv][2])
* **Computing *all* identifiable functions** (symbolic; single/multi‑experiment): Ovchinnikov, Pillay, Pogudin & Scanlon 2020–2021. ([arXiv][6])
* **Recent benchmark & tools overview** (GenSSI, STRIKE‑GOLDD/AutoRepar, SIAN, etc.): Barreiro et al. 2023. ([PMC][5])

---

## G. Which three to five would I actually use in your paper?

* **Tier 1 (fast, visually clear):** PK 2‑compartment (Ex. 1), thyroid combo (ck_{34}), repressilator (K_i/\beta_i). 
* **Tier 2 (method edge‑case):** PK variant with a loose pair (Ex. 2). 
* **Tier 3 (algebraic gold standard):** Meshkat canonical reparameterisation (nonlinear 2‑compartment), to show your output matches a known (q)-set. 

That set spans **sums, products, and ratios**; **rational and non‑rational** dynamics; **small to moderate** dimensionality; and includes a **case where naive profiling fails**—all ideal to highlight the distinctive strengths of your new method.

---

### Closing thought (constructive):

If your method (i) **automates** the preconditioning step that Eisenberg needed before profiling, (ii) **handles non‑rational** terms like the thyroid example, or (iii) **returns closed‑form combinations** rather than just pairwise traces, lean into those in your narrative and figure design. Consider one figure per model: **(a)** original parameter profiles (or a sketch), **(b)** your recovered combinations, **(c)** the reparameterised fit with tight posteriors—readers “get it” immediately.

If you want, I can draft a minimal, self‑contained notebook for any two of the above cases (simulated noiseless data + recovery in combo space) so you have plug‑and‑play figures.

[1]: https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2016.00590/pdf "Parameter Identifiability of Fundamental Pharmacodynamic Models"
[2]: https://arxiv.org/abs/1305.5768 "[1305.5768] Identifiable reparametrizations of linear compartment models"
[3]: https://academic.oup.com/bioinformatics/article/30/10/1440/266188 "Comparison of approaches for parameter identifiability analysis of biological systems | Bioinformatics | Oxford Academic"
[4]: https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0110261&utm_source=chatgpt.com "On Finding and Using Identifiable Parameter Combinations in ..."
[5]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9913045/ "
            Benchmarking tools for a priori identifiability analysis - PMC
        "
[6]: https://arxiv.org/abs/2004.07774?utm_source=chatgpt.com "Computing all identifiable functions of parameters for ODE models"
[7]: https://pubmed.ncbi.nlm.nih.gov/19505944/ "Structural and practical identifiability analysis of partially observed dynamical models by exploiting the profile likelihood - PubMed"
