# Context Checkpoint — Phase 2 stat_model directional near-invariance

## Goal
Start Phase 2 code work (stat_model refresh) and add a practical-identifiability check that can capture one-sided near-non-identifiability near limit regimes.

## Current State
- Phase 1 technical/code checklist is effectively closed (repressilator artifacts regenerated; utility ownership pass done; docs updated).
- `examples/stat_model.jl` runs successfully with current in-repo `ReparamTools.jl`.
- Poisson and non-Poisson branches both run (non-Poisson tested via in-memory toggle, no permanent source toggle required).
- Figures are written to `./figures/`.
- Added a practical rank heuristic block in `examples/stat_model.jl` (relative singular values + cutoff-based practical rank suggestion).
- Planning docs updated for this new direction:
  - `NEXT_STEPS.md` Phase 2 now includes directional practical near-invariance TODOs.
  - `AGENTS.md` Phase 2 now includes the directional/one-sided practical check decision point.

## Decisions (explicit)
1. Keep structural and practical analyses separate:
   - Structural: `find_invariant_subspace` with strict/default tolerances.
   - Practical: additional directional probe diagnostics.
2. Agreed direction for practical check:
   - Primary metric: directional weakness along weakest singular vector.
   - Secondary metric: directional drift/stability of that weakness.
3. Proposed practical diagnostics (non-limit case):
   - Let `v_weak` = weakest right singular vector at `θ0` (MLE, log-space).
   - Use symmetric steps `±δ`.
   - Report
     - `ε±(δ) = ||J(θ0 ± δ v_weak) v_weak|| / σ1(θ0 ± δ v_weak)`
     - optional `d±(δ) = ||J(θ0 ± δ v_weak)v_weak - J(θ0)v_weak|| / ||J(θ0)v_weak||`
   - Emphasize one-sided asymmetry `ε-(δ)/ε+(δ)` (or vice versa).
4. Possible library step (agreed as desirable to evaluate next):
   - Add reusable helper for directional practical probe (general vector perturbations; 1D is special case), then call from `examples/stat_model.jl`.

## Key exploratory results supporting the suggestion
Non-Poisson run (`poisson_limit=false`) at log-space MLE:
- `θ0 ≈ [3.8265, -0.8800]`, `xy ≈ [45.90, 0.4148]`
- singular values: `S ≈ [28.8542, 5.2114]`
- local weakness ratio: `σ2/σ1 ≈ 0.1806`
- weakest direction `v_weak ≈ [0.6553, -0.7553]` (so `+δ` reduces `p`, moving toward Poisson limit)

Directional probe snapshot:
- `δ=0.8`: `ε+(δ)≈0.0808`, `ε-(δ)≈0.3904` (strong one-sided behavior)
- trend: one side gets closer to near-invariance; opposite side gets less invariant
- this matches observed one-sided profile flattening and the limit-based interpretation

## Next Steps
1. Implement directional practical probe in `examples/stat_model.jl` output (readable table for `δ ∈ {0.05,0.1,0.2,0.4,0.8}`).
2. Decide whether to lift that probe into library code (`invariance.jl` helper) and call it from example.
3. Keep manuscript text editing outside this session (user will handle separately).
4. After implementation, rerun `examples/stat_model.jl` and verify output/figures remain clean.

## Key Files
- `examples/stat_model.jl`
- `invariance.jl`
- `ReparamTools.jl`
- `NEXT_STEPS.md`
- `AGENTS.md`
- `figures/`

## Useful Commands
- Run canonical stat_model script:
  - `julia --project=. "examples/stat_model.jl"`
- Non-Poisson sanity run without editing file on disk:
  - `julia --project=. -e 'src = read("examples/stat_model.jl", String); src = replace(src, "include(\"../ReparamTools.jl\")" => "include(\"ReparamTools.jl\")"); src = replace(src, "poisson_limit = true" => "poisson_limit = false"); include_string(Main, src, "stat_model_nonpoisson_check.jl")'`

## Continuation Prompt
"Continue Phase 2 stat_model code work: implement the directional practical near-invariance probe (ε± and optional drift) in a scientist-readable way, verify one-sided behavior in non-limit mode, and decide whether to promote it into a reusable library helper."