# Invariant Image Reparameterisation

This repository contains a Julia implementation of methods described in "Invariant Image Reparameterisation: A Unified Approach to Structural and Practical Identifiability and Model Reduction". A preprint is available from arxiv.org/abs/2502.04867, and a working version of the paper is available in the repository under docs. The docs also contain some presentation slides from The Joint Meeting of the New Zealand, Australian and American Mathematical Societies (2024). 

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
    "ForwardDiff",
    "LaTeXStrings",
    "Measures",
    "NLopt",
    "Plots"
])
```

This package also uses the following standard libraries:
* `LinearAlgebra`
* `SparseArrays`

## Structure

The codebase is organized as:

### Main Module
- `ReparamTools.jl` - The primary module that users interact with, providing the interface for parameter transformations, identifiability analysis, and likelihood-based inference

### Examples
- `transport_model.jl` - Demonstration of parameter identifiability analysis for a transport model (diffusive flow in composite medium)
- `mm_model.jl` - Example using the Michaelis-Menten/Monod model
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

2. Transport Model (`transport_model.jl`)
   - Demonstrates basic structural and practical parameter identifiability analysis for model with vector output on a fine grid and coarser observation grid
   - Includes predictive uncertainty quantification

3. Michaelis-Menten Model (`mm_model.jl`)
   - Similar to diffusion example but includes the use of an ODE solver as part of model definition

Each example includes:
- Model definition
- Identifiability analysis
- Parameter transformations
- Profile likelihood calculations
- Visualisation of results

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details


