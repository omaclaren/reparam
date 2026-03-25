# Invariant Image Reparameterisation (IIR)

**Status:** Under revision at SIAM/ASA Journal on Uncertainty Quantification  
**Current active sprint:** Repressilator closeout (figure/text integration + workflow finalization)

This repository contains a Julia implementation of methods for automatically discovering identifiable parameter combinations in mathematical models using numerical invariance testing.

## Overview

**Invariant Image Reparameterisation (IIR)** provides methods for:

- **Structural identifiability analysis**: Determine which parameter combinations are theoretically identifiable from data
- **Practical identifiability**: Separate well-identified from poorly-identified combinations
- **Automatic discovery**: Find identifiable monomial combinations without symbolic computation
- **Model reparameterisation**: Transform to coordinates that cleanly separate identifiable/non-identifiable structure
- **Uncertainty quantification**: Profile-wise analysis for parameters and predictions

### Key Innovation

IIR uses a **numerical invariance test** (Hessian-based criterion) to identify which directions in parameter space have globally invariant null spaces. This enables:
- Discovery of parameter combinations purely from numerical Jacobian
- No symbolic computation required
- Works for complex models (ODEs, PDEs, stochastic systems)

## Quick Start

### Installation

Requires Julia 1.6 or higher. Install dependencies:

```julia
using Pkg
Pkg.add([
    "Distributions",
    "ForwardDiff",
    "LaTeXStrings",
    "Measures",
    "NLopt",
    "Plots",
    "DifferentialEquations",  # For ODE examples
    "FactorLoadingMatrices"   # Legacy Varimax helper
])
```

### Basic Usage

```julia
include("ReparamTools.jl")
using .ReparamTools

# Define auxiliary mapping (parameters → data distribution parameters)
ϕ(θ) = [θ[1]*θ[2], θ[1]*θ[2]]  # Example: Poisson limit
θ0 = [100.0, 0.2]

# Find invariant subspace at reference parameters
S, N, N_perp, rank_J = find_invariant_subspace(ϕ, θ0)
J = compute_ϕ_Jacobian(ϕ, θ0)

# Build a shared monomial basis:
# - informed search on the identified side
# - simplicity-only search on the invariant/null side
identified = informed_monomial_basis_search(
    N_perp, J' * J, S[1]^2, ["n", "p"]; s_max=2, c_max=1, residual_cap=1e-2)
null = simple_search_with_support_retry(
    N, ["n", "p"]; s_max=2, c_max=1, residual_cap=1e-2)

A_cols = hcat(
    basis_candidate_matrix(identified.selected, 2),
    basis_candidate_matrix(null.selected, 2),
)
θ_to_ψ, ψ_to_θ = reparam(A_cols)
```

See [examples/stat_model.jl](examples/stat_model.jl), [examples/mm_model.jl](examples/mm_model.jl), and [examples/transport_model.jl](examples/transport_model.jl) for complete maintained example workflows.

## Examples

### Maintained simple examples
- `examples/stat_model.jl` - Pedagogical two-parameter example with `np` / `n/p`
- `examples/mm_model.jl` - Michaelis-Menten/Monod example with exact-limit and practical non-limit views
- `examples/transport_model.jl` - Transport example showing orthogonal subspaces and sparse ratio coordinates

### Maintained workflow / HPC example
- `run_repressilator_profile.jl` - Canonical repressilator profiling runner
- `examples/RepressilatorModel.jl` - Repressilator model definition used by the runner
- `replot_profile_results.jl` - Replot saved 2D profile likelihood surfaces from `.jls` results
- `repressilator_prediction_intervals_from_2d_profile.jl` - Compare prediction envelopes from:
  1. full accepted 2D pushforward,
  2. 1D profile over identifiable target (`K₁/β₁`),
  3. 1D profile over non-identifiable target (`β₁K₁`).

Notes:
- `run_repressilator_profile.jl` supports **slice/profile** modes (hybrid removed).
- Repressilator result files store observation data/metadata (`data`, `t_obs`, `X0`, `σ`, etc.) so post-processing does not need to regenerate data from RNG state.
- Historical hybrid artifacts are archived under `archive/hybrid/`.

Example:
```bash
julia --project=. repressilator_prediction_intervals_from_2d_profile.jl nesi/repressilator_16nuisance_50x50_results.jls
```

### Archived exploratory / legacy material
- `archive/exploratory-examples/` - exploratory non-paper examples kept on the revision branch
- `archive/legacy-examples/` - historical repressilator scripts and old test/example entrypoints
- `archive/legacy-sequential/` - sequential-IIR legacy scripts and notes

## Repository Structure

```
reparam/
├── ReparamTools.jl          # Main module
├── invariance.jl            # Algorithm 1: find_invariant_subspace()
├── core.jl                  # Profile likelihood, optimization
├── utils.jl                 # Helper functions
├── parameterizations.jl     # Transformations, monomial basis search, legacy helpers
├── visualization.jl         # Plotting utilities
├── run_repressilator_profile.jl                      # Canonical repressilator profiling runner
├── repressilator_prediction_intervals_from_2d_profile.jl  # Repressilator prediction-band post-processing
├── examples/
│   ├── stat_model.jl        # Pedagogical maintained example
│   ├── mm_model.jl          # Maintained small nonlinear example
│   ├── transport_model.jl   # Maintained transport example
│   └── RepressilatorModel.jl # Maintained repressilator model definition
├── AGENTS.md                # Canonical internal project context/strategy
├── CLAUDE.md                # Compatibility shim pointing to AGENTS.md
├── NEXT_STEPS.md            # Actionable backlog
└── archive/                 # Archived exploratory and legacy artifacts
```

## Documentation

**For developers/contributors**:
- [AGENTS.md](AGENTS.md) - Canonical project context, strategy, and workflow status
- [NEXT_STEPS.md](NEXT_STEPS.md) - Actionable backlog (ownership + manuscript + merge)
- [CLAUDE.md](CLAUDE.md) - Compatibility shim (redirects to AGENTS.md)

## Method Details

### Algorithm 1: `find_invariant_subspace()`

**Inputs**:
- `ϕ_func`: Auxiliary mapping θ → ϕ(θ)
- `θ0`: Reference parameter values
- `rtolJ`: Relative tolerance for Jacobian rank (default: √eps ≈ 1.5e-8)
- `atolM`: Absolute tolerance for invariance test (default: 1e-10)
- `invariance_method`: `:hessian_based` (default) or `:finite_difference` (stiff ODEs)

**Outputs**:
- `S`: Singular values of Jacobian
- `N`: Invariant null space (non-identifiable directions)
- `N_perp`: Complement (potentially identifiable directions)
- `rank_J`: Numerical rank

**Key features**:
- Uses nested AD for Hessian-based invariance test
- Finite-difference option for stiff ODE systems
- Separates structural from practical non-identifiability

### Monomial Basis Search (Current Default)

For maintained examples, the public/default path is:

```julia
identified = informed_monomial_basis_search(
    N_perp, J' * J, S[1]^2, param_names; s_max=2, c_max=1, residual_cap=1e-2)
null = simple_search_with_support_retry(
    N, param_names; s_max=2, c_max=1, residual_cap=1e-2)
```

This separates two goals cleanly:
- **identified side**: choose a simple basis, but order it using local information
- **null side**: choose a simple sparse invariant basis

### Legacy Varimax Helpers

`scale_and_round` and `varimax_rotation` remain in the codebase for archived diagnostics and historical scripts, but they are no longer the public default path for maintained examples.

## Paper Strategy

### Main Contribution: Single-Stage IIR
Focus on robust, reliable monomial transformations (ψ = exp(A log(θ)))

**Why single-stage?**
- ✅ Works reliably across model types
- ✅ Clear theoretical foundation
- ✅ No basis-dependence issues
- ✅ Produces interpretable sparse monomial results with a shared basis-selection path

### Multi-Stage Extensions (Future Work)
Sequential application (e.g., products → sums) mentioned briefly as open research direction. Investigation revealed:
- Success depends on basis alignment (open problem)
- Simplicity and informedness should be separated explicitly rather than forced through a single rotation heuristic

## Citation

If you use this code, please cite:

```
@article{maclaren2025iir,
  title={Invariant Image Reparameterisation: A Unified Approach to Structural and Practical Identifiability and Model Reduction},
  author={Maclaren, Oliver J.},
  journal={arXiv preprint arXiv:2502.04867},
  year={2025}
}
```

Preprint: [arxiv.org/abs/2502.04867](https://arxiv.org/abs/2502.04867)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions about the method or implementation, please open an issue on GitHub or contact the author.

## Version History

- **v1.0** (2025-01): Initial submission to SIAM/ASA JUQ
- **v2.0-dev** (2025-10): Revision with repressilator example, finite-difference invariance test, strategic focus on single-stage IIR
