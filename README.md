# Invariant Image Reparameterisation (IIR)

**Status:** Under revision at SIAM/ASA Journal on Uncertainty Quantification

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
    "FactorLoadingMatrices"   # For Varimax rotation
])
```

### Basic Usage

```julia
include("ReparamTools.jl")
using .ReparamTools

# Define auxiliary mapping (parameters → data distribution parameters)
ϕ(θ) = [θ[1]*θ[2], θ[1]*θ[2]]  # Example: Poisson limit

# Find invariant subspace at reference parameters
θ0 = [100.0, 0.2]
S, N, N_perp, rank_J = find_invariant_subspace(ϕ, θ0)

# N = invariant null space (non-identifiable directions)
# N_perp = complement (potentially identifiable directions)

# Build transformation matrix (monomial reparameterization)
A_full = vcat(N_perp', N')
ψ(θ) = exp.(A_full * log.(θ))  # New coordinates
```

See [examples/stat_model.jl](examples/stat_model.jl) for complete workflow.

## Examples

### 1. stat_model.jl (Pedagogical)
**Model**: Poisson limit distribution
**Parameters**: n (sample size), p (probability)
**Identifiable**: np (mean)
**Non-identifiable**: n/p

**Purpose**: Clear introduction to IIR workflow

### 2. repressilator.jl (Ambitious)
**Model**: 3-gene repressilator (Eisenberg & Hayashi 2010)
**Parameters**: 18 (nonlinear ODE system)
**Identifiable**: K₁/β₁, K₂/β₂, K₃/β₃ ratios
**Demonstrates**:
- IIR on realistic mechanistic model
- Finite-difference invariance test for stiff ODEs
- Profile-wise prediction uncertainty
- Validation against profile likelihood (Eisenberg 2010)

### Repressilator Post-processing Utilities
- `replot_profile_results.jl` - Replot saved 2D profile likelihood surfaces from `.jls` results
- `repressilator_prediction_intervals_from_2d_profile.jl` - Specialized utility for repressilator runs that compares prediction envelopes from:
  1. full accepted 2D pushforward,
  2. 1D profile over identifiable target (`K₁/β₁`),
  3. 1D profile over non-identifiable target (`β₁K₁`).

Notes:
- `run_repressilator_profile.jl` now supports **slice/profile** modes (hybrid removed).
- Repressilator result files now store observation data/metadata (`data`, `t_obs`, `X0`, `σ`, etc.) so post-processing does not need to regenerate data from RNG state.
- Historical hybrid artifacts are archived under `archive/hybrid/`.
- Legacy/sequential writeups are archived under `archive/legacy-sequential/` (as available).

Example:
```bash
julia --project=. repressilator_prediction_intervals_from_2d_profile.jl nesi/repressilator_16nuisance_50x50_results.jls
```

### Legacy Examples
- `transport_model.jl` - Diffusive transport in composite medium
- `mm_model.jl` - Michaelis-Menten/Monod kinetics
- `stat_sum_model.jl` - Multi-stage IIR exploration (not in paper)
- `pk_model.jl` - Pharmacokinetic model revealing multi-stage limitations

## Repository Structure

```
reparam/
├── ReparamTools.jl          # Main module
├── invariance.jl             # Algorithm 1: find_invariant_subspace()
├── core.jl                   # Profile likelihood, optimization
├── utils.jl                  # Helper functions
├── parameterizations.jl      # Transformations, Varimax rotation
├── visualization.jl          # Plotting utilities
├── run_repressilator_profile.jl                      # Canonical repressilator profiling runner
├── repressilator_prediction_intervals_from_2d_profile.jl  # Repressilator prediction-band post-processing
├── examples/
│   ├── stat_model.jl         # Pedagogical example
│   ├── repressilator.jl      # Ambitious ODE example
│   └── [other examples]
├── AGENTS.md                 # Canonical internal project context/strategy
├── CLAUDE.md                 # Compatibility shim pointing to AGENTS.md
├── NEXT_STEPS.md             # Actionable backlog
└── archive/                  # Archived legacy artifacts and notes
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

### Varimax Rotation (Optional)

Improve interpretability of identifiable combinations:

```julia
N_perp_rotated = varimax_rotation(N_perp; n_restarts=200)
```

Maximizes sparsity within span(N_perp) while preserving invariant subspace structure.
For the current single-stage paper focus, treat this as an interpretability/presentation aid rather than a required algorithmic step.

## Paper Strategy

### Main Contribution: Single-Stage IIR
Focus on robust, reliable monomial transformations (ψ = exp(A log(θ)))

**Why single-stage?**
- ✅ Works reliably across model types
- ✅ Clear theoretical foundation
- ✅ No basis-dependence issues
- ✅ Produces interpretable results (Varimax available as optional enhancement)

### Multi-Stage Extensions (Future Work)
Sequential application (e.g., products → sums) mentioned briefly as open research direction. Investigation revealed:
- Success depends on basis alignment (open problem)
- Varimax optimizes sparsity, not compositional reducibility

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
