function finite_diff_gradient(f, θ; h=1e-8)
    """
    Compute gradient using finite differences.

    Not typically used -- use ForwardDiff.jl instead -- but included for completeness.

    Parameters:
    - f: Function to differentiate
    - θ: Parameter vector at which to evaluate gradient
    - h: Step size for finite difference (default: 1e-8)

    Returns:
    - Vector containing numerical gradient
    """
    dim = length(θ)
    numerical_gradient = similar(θ)
    
    for i in 1:dim
        θ_plus = copy(θ)
        θ_minus = copy(θ)
        θ_plus[i] += h
        θ_minus[i] -= h
        numerical_gradient[i] = (f(θ_plus) - f(θ_minus))/(2h)
    end
    
    return numerical_gradient
end