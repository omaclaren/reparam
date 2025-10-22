"""
RepressilatorModel.jl

Lightweight module containing the repressilator ODE system and helper functions
for distributed profiling.

This module can be loaded on worker processes to enable distributed 2D profiling
in repressilator.jl.
"""
module RepressilatorModel

using DifferentialEquations
using LinearAlgebra

export repressilator!, solve_repressilator, extract_mrna, extract_proteins,
       create_ϕ_mapping, predict_mRNA

"""
Repressilator ODE system - Eisenberg & Hayashi formulation with n=2.5 fixed.

State vector X = [m₁, m₂, m₃, p₁, p₂, p₃]
Parameter vector θ = [α₀₁, α₀₂, α₀₃, α₁, α₂, α₃,
                      β₁, β₂, β₃, K₁, K₂, K₃,
                      k_degm₁, k_degm₂, k_degm₃, k_degp₁, k_degp₂, k_degp₃]

18 parameters (Hill coefficient n fixed at 2.5)
"""
function repressilator!(dX, X, θ, t)
    # Unpack state variables
    m₁, m₂, m₃, p₁, p₂, p₃ = X

    # Unpack parameters (18 total)
    α₀₁, α₀₂, α₀₃ = θ[1:3]      # Basal transcription
    α₁, α₂, α₃ = θ[4:6]          # Regulated transcription
    β₁, β₂, β₃ = θ[7:9]          # Translation
    K₁, K₂, K₃ = θ[10:12]        # Inhibition constants
    k_degm₁, k_degm₂, k_degm₃ = θ[13:15]  # mRNA degradation
    k_degp₁, k_degp₂, k_degp₃ = θ[16:18]  # Protein degradation

    # Hill coefficient fixed at 2.5
    n = 2.5

    # mRNA dynamics: basal + regulated transcription - degradation
    dX[1] = α₀₁ + α₁ / (1 + (p₃/K₃)^n) - k_degm₁ * m₁
    dX[2] = α₀₂ + α₂ / (1 + (p₁/K₁)^n) - k_degm₂ * m₂
    dX[3] = α₀₃ + α₃ / (1 + (p₂/K₂)^n) - k_degm₃ * m₃

    # Protein dynamics: translation - degradation
    dX[4] = β₁ * m₁ - k_degp₁ * p₁
    dX[5] = β₂ * m₂ - k_degp₂ * p₂
    dX[6] = β₃ * m₃ - k_degp₃ * p₃
end

"""
Solve the repressilator ODE system.
"""
function solve_repressilator(t_save, θ, X0; solver=Rodas4())
    tspan = (0.0, maximum(t_save))
    prob = ODEProblem(repressilator!, X0, tspan, θ)
    sol = solve(prob, solver, saveat=t_save, abstol=1e-10, reltol=1e-8)
    return Array(sol)
end

"""
Extract all three mRNA concentrations.
"""
function extract_mrna(solution_matrix)
    return solution_matrix[1:3, :]
end

"""
Extract all three protein concentrations.
"""
function extract_proteins(solution_matrix)
    return solution_matrix[4:6, :]
end

"""
Create ϕ mapping from parameters to mRNA observations.
"""
function create_ϕ_mapping(t, X0)
    function ϕ(θ)
        sol_matrix = solve_repressilator(t, θ, X0)
        mrna_matrix = extract_mrna(sol_matrix)
        # Flatten: [m₁(t₁), m₂(t₁), m₃(t₁), m₁(t₂), ...]
        return vec(mrna_matrix')
    end
    return ϕ
end

"""
Predict mRNA concentrations for given parameters and time grid.
Returns flattened vector in column-major order.
"""
function predict_mRNA(θ, t_grid, X0)
    sol_matrix = solve_repressilator(t_grid, θ, X0)
    mRNA = extract_mrna(sol_matrix)  # 3×NT matrix (rows=species, cols=time)
    return vec(mRNA)  # Flatten → [m1(t1), m2(t1), m3(t1), m1(t2), ...]
end

end # module
