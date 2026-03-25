using LinearAlgebra
using DifferentialEquations
using ForwardDiff

include("../../../ReparamTools.jl")
using .ReparamTools

# True parameters
b₁_true, c₁_true, k₀₁_true, k₀₂_true = 2.0, 1.5, 0.2, 0.1
k₁₂_true, k₂₁_true, V_M_true, K_M_true = 0.3, 0.25, 1.0, 3.0
θ_true = [b₁_true, c₁_true, k₀₁_true, k₀₂_true, k₁₂_true, k₂₁_true, V_M_true, K_M_true]

# ODE model
function pk_ode!(dx, x, θ, t)
    b₁, c₁, k₀₁, k₀₂, k₁₂, k₂₁, V_M, K_M = θ
    x₁, x₂ = x
    mm_clearance = (V_M * x₁) / (K_M + x₁)
    u_input = t < 0.1 ? 1.0 : 0.0
    dx[1] = -(k₀₁ + k₁₂)*x₁ + k₂₁*x₂ - mm_clearance + b₁*u_input
    dx[2] = k₁₂*x₁ - (k₀₂ + k₂₁)*x₂
end

t_obs = collect(range(0.1, 5.0, length=20))
x0 = [0.0, 0.0]

function ϕ_func_θ(θ)
    prob = ODEProblem(pk_ode!, x0, (0.0, maximum(t_obs)), θ)
    sol = solve(prob, Tsit5(), saveat=t_obs, abstol=1e-10, reltol=1e-10)
    c₁ = θ[2]
    return c₁ * sol[1, :]
end

# Stage 1: f = log
ϕ_log = θ_log -> ϕ_func_θ(exp.(θ_log))
S_stage1, N_stage1, N_perp_stage1, rankJ_stage1 = find_invariant_subspace(
    ϕ_log, log.(θ_true);
    rtolJ=sqrt(eps()),
    atolM=1e-10
)

println("="^60)
println("N_perp BEFORE Varimax rotation:")
println("="^60)
display(N_perp_stage1)

println("\n\nParameter mapping:")
println("  Param 1: b₁")
println("  Param 2: c₁")
println("  Param 3: k₀₁")
println("  Param 4: k₀₂")
println("  Param 5: k₁₂")
println("  Param 6: k₂₁")
println("  Param 7: V_M")
println("  Param 8: K_M")

# Look for columns that might relate to k₀₂ (row 4) and k₁₂ (row 5)
println("\n\nAnalyzing which columns involve k₀₂ (param 4) and k₁₂ (param 5):")
for col in 1:size(N_perp_stage1, 2)
    v = N_perp_stage1[:, col]
    k02_coef = v[4]
    k12_coef = v[5]
    if abs(k02_coef) > 0.1 || abs(k12_coef) > 0.1
        println("\n  Column $col:")
        println("    k₀₂ coefficient: ", round(k02_coef, digits=4))
        println("    k₁₂ coefficient: ", round(k12_coef, digits=4))
        println("    Full column: ", round.(v, digits=4))
    end
end

println("\n\nNow apply Varimax:")
N_perp_varimax = varimax_rotation(N_perp_stage1; n_restarts=200, threshold=1e-2)

println("\n" * "="^60)
println("N_perp AFTER Varimax rotation:")
println("="^60)
display(N_perp_varimax)

# Build transformation
A1_full = hcat(N_perp_varimax, N_stage1)'

println("\n\nTransformation matrix A1 (rows are exponents in log-space):")
display(A1_full)

println("\n\nWhich rows involve k₀₂ or k₁₂?")
for row in 1:size(A1_full, 1)
    r = A1_full[row, :]
    k02_exp = r[4]
    k12_exp = r[5]

    # Check if this row is JUST k₀₂ or JUST k₁₂
    other_params = [r[1:3]; r[6:8]]
    if abs(k02_exp) > 0.9 && all(abs.(other_params) .< 0.1)
        println("  Row $row: k₀₂^(", round(k02_exp, digits=2), ") → θ¹[$row] = ",
                k02_exp > 0 ? "k₀₂" : "1/k₀₂")
    end
    if abs(k12_exp) > 0.9 && all(abs.(other_params) .< 0.1)
        println("  Row $row: k₁₂^(", round(k12_exp, digits=2), ") → θ¹[$row] = ",
                k12_exp > 0 ? "k₁₂" : "1/k₁₂")
    end
end

println("\n\nConclusion:")
println("If k₀₂ appears with NEGATIVE exponent, we get θ¹ = 1/k₀₂")
println("This is arbitrary - Varimax could have chosen positive exponent")
println("But then q₃ = k₀₂ + k₁₂ would be expressible as θ¹[i] + θ¹[j]")
