include("../ReparamTools.jl")
using .ReparamTools, DifferentialEquations

function repressilator_eisenberg!(dX, X, θ, t)
    m₁, m₂, m₃, p₁, p₂, p₃ = X
    α₀₁, α₀₂, α₀₃ = θ[1:3]
    α₁, α₂, α₃ = θ[4:6]
    β₁, β₂, β₃ = θ[7:9]
    K₁, K₂, K₃ = θ[10:12]
    k_degm₁, k_degm₂, k_degm₃ = θ[13:15]
    k_degp₁, k_degp₂, k_degp₃ = θ[16:18]
    n = θ[19]

    dX[1] = α₀₁ + α₁ / (1 + (p₃/K₃)^n) - k_degm₁ * m₁
    dX[2] = α₀₂ + α₂ / (1 + (p₁/K₁)^n) - k_degm₂ * m₂
    dX[3] = α₀₃ + α₃ / (1 + (p₂/K₂)^n) - k_degm₃ * m₃
    dX[4] = β₁ * m₁ - k_degp₁ * p₁
    dX[5] = β₂ * m₂ - k_degp₂ * p₂
    dX[6] = β₃ * m₃ - k_degp₃ * p₃
end

function solve_repressilator(t_save, θ, X0; solver=Rodas4())
    tspan = (0.0, maximum(t_save))
    prob = ODEProblem(repressilator_eisenberg!, X0, tspan, θ)
    sol = solve(prob, solver, saveat=t_save, abstol=1e-10, reltol=1e-8)
    return Array(sol)
end

function extract_mrna(solution_matrix)
    return solution_matrix[1:3, :]
end

# True params
θ_true = [5e-4, 1e-4, 9e-4, 0.5, 0.7, 0.3, 0.002, 0.003, 0.001, 40.0, 30.0, 50.0, 0.005776, 0.00987, 0.00345, 0.001155, 0.00059, 0.004982, 2.0]
X0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
t = LinRange(0, 100.0, 21)

println("Solving...")
sol_matrix = solve_repressilator(t, θ_true, X0)
println("Solution matrix size: ", size(sol_matrix))
mRNA = extract_mrna(sol_matrix)
println("mRNA size: ", size(mRNA))
data_obs = vec(mRNA')
println("Flattened data length: ", length(data_obs))
