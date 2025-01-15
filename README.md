# Invariant Image Reparameterisation

This repository contains a Julia implementation of methods described in "Invariant Image Reparameterisation: A Unified Approach to Structural and Practical Identifiability and Model Reduction" (Preprint Coming Soon, 2025).

## Overview

The code implements methods for

- Analysing structural and practical parameter identifiability in mathematical models in a unified way
- Finding identifiable and nonidentifiable nonlinear (monomial) parameter combinations
- Model reparameterisation techniques based on identifiable/nonidentifiable parameter combinations
- Profile likelihood, in the form of Profile-Wise Analysis (PWA), for uncertainty quantification for both parameters and predictions

## Installation

This package requires Julia 1.0 or higher. Install the required packages:

```julia
using Pkg
Pkg.add([
    "Distributions", 
    "LinearAlgebra",
    "ForwardDiff",
    "NLopt",
    "Plots"
])
```

## Structure

The codebase is organized as:

### Main Module
- `ReparamTools.jl` - The primary module that users interact with, providing the interface for parameter transformations, identifiability analysis, and likelihood-based inference

### Examples
- `diffusion_model.jl` - Demonstration of parameter identifiability analysis for a diffusion model
- `monod_model.jl` - Example using the Monod/Michaelis-Menten model
- `stat_model.jl` - Simple statistical model example 

### Implementation Files
- `core.jl` - Internal implementation of likelihood and identifiability methods
- `parameterizations.jl` - Implementation of parameter transformation methods
- `utils.jl` - Common utility functions and helpers
- `visualization.jl` - Internal plotting and visualization tools

## Usage

Basic model setup (using the simple stat_model.jl as an example):

```julia
include("../ReparamTools.jl") # Assuming working in examples directory
using .ReparamTools 

# Define model through auxiliary mapping (maps parameters to data distribution parameters)
ϕ_xy = xy -> [xy[1]*xy[2], xy[2]]

# Create distribution mapping (specifies how distribution depends on parameters)
distrib_xy = xy -> Normal(ϕ_xy(xy)[1], sqrt(ϕ_xy(xy)[1]*(1-ϕ_xy(xy)[2])))

# Construct likelihood using data
lnlike_xy = construct_lnlike_xy(distrib_xy, data)

```

See the example files for complete analyses including identifiability analysis, model reduction, parameter combinations, and inference.

## Examples

The repository includes several examples demonstrating the methods:

1. Statistical Model Example (`stat_model.jl`)
   - Demonstrates basic structural and practical parameter identifiability analysis for model with scalar output

2. Diffusion Model (`diffusion_model.jl`)
   - Demonstrates basic structural and practical parameter identifiability analysis for model with vector output on a fine grid and coarser observation grid
   - Includes predictive uncertainty quantification

3. Monod Model (`monod_model.jl`)
   - Similar to diffusion example but includes the use of an ODE solver as part of model definition

Each example includes:
- Model definition
- Identifiability analysis
- Parameter transformations
- Profile likelihood calculations
- Visualisation of results

## References

[Citation of Invariant Image Reparameterisation paper, 2025]

## Contributing

Please feel free to submit issues or pull requests.

## License

[Add appropriate license information]
