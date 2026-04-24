"""
RepressilatorModel.jl

Module containing the repressilator ODE system (Eisenberg & Hayashi formulation)
and helper functions for simulation and observation mapping.
"""
module RepressilatorModel

using DifferentialEquations
using LinearAlgebra

export repressilator!, solve_repressilator, predict_mRNA, create_ϕ_mapping

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
function solve_repressilator(t_save, θ, X0; solver=Rodas4(), abstol=1e-8, reltol=1e-6)
    tspan = (0.0, maximum(t_save))
    prob = ODEProblem(repressilator!, X0, tspan, θ)
    sol = solve(prob, solver, saveat=t_save, abstol=abstol, reltol=reltol)
    return Array(sol)
end

"""
Predict mRNA concentrations for given parameters and time grid.
Returns flattened vector in column-major order: [m1(t1), m2(t1), m3(t1), m1(t2), ...]
"""
function predict_mRNA(θ, t_grid, X0)
    sol_matrix = solve_repressilator(t_grid, θ, X0)
    mRNA = sol_matrix[1:3, :]  # 3×NT matrix (rows=species, cols=time)
    return vec(mRNA)
end

"""
Create ϕ mapping from parameters to mRNA observations.
"""
function create_ϕ_mapping(t, X0)
    function ϕ(θ)
        return predict_mRNA(θ, t, X0)
    end
    return ϕ
end

end # module
