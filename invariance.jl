# ----------------------------------------------------------------
#   Functions for finding invariant null space and complement
# ----------------------------------------------------------------

function find_invariant_subspace(ϕ_func, θ0; 
                                 compute_J=compute_ϕ_Jacobian,
                                 rtolJ=sqrt(eps(real(eltype(θ0)))), 
                                 rtolM=sqrt(eps(real(eltype(θ0)))))

    """
    Finds invariant null subspace and its orthogonal complement for the auxiliary mapping ϕ_func at point θ0.

    Key steps:
    1. Local SVD in current parameterisation to find a candidate basis.
    2. Higher-order invariance test to filter for the invariant structure.
    3. Construction of the final, invariant sufficient reparameterization.

    # Arguments
    - `ϕ_func`: Auxiliary mapping function (mechanistic → distribution parameters)
    - `θ0`: Point in current parameter space to evaluate the Jacobian
    - `compute_J`: Function to compute Jacobian (default: compute_ϕ_Jacobian). 
                   Can pass custom implementation for flexibility.
    - `rtolJ`: Relative tolerance for determining numerical rank of J (default: √eps)
    - `rtolM`: Relative tolerance for determining numerical rank of test matrix (default: √eps)

    # Returns
    - `S`: Singular values from the initial Jacobian SVD
    - `N`: Matrix (p×k0) whose columns form an orthonormal basis for the invariant null space
    - `N_perp`: Matrix (p×k⊥) whose columns form an orthonormal basis for the potentially identifiable space
    - `rankJ`: Numerical rank of the overall model Jacobian
    
    # Examples
    ```julia
    # Standard usage
    S, N, N_perp, rankJ = find_invariant_subspace(ϕ_func, θ0)
    
    # With custom Jacobian computation
    S, N, N_perp, rankJ = find_invariant_subspace(ϕ_func, θ0; 
        compute_J = (f, θ) -> FiniteDiff.finite_difference_jacobian(f, θ))
    ```
    """

    T = real(eltype(θ0))
    
    # --- 1. Initial Transformation (Local SVD) ---
    J = compute_J(ϕ_func, θ0)
    m, p = size(J)  # m = distribution params, p = mechanistic params
    
    # full=true for complete nullspace when p>m
    svd_result = svd(J; full=true)
    S = svd_result.S
    V = svd_result.V  # p×p
    
    # Relative (approximate) rank determination
    σmax = maximum(S)
    τJ = rtolJ * σmax
    rankJ = count(>(τJ), S)
    
    V_r = V[:, 1:rankJ]
    V_0 = V[:, rankJ+1:end]

    # If the local null space is empty, finished
    if size(V_0, 2) == 0
        N = zeros(T, p, 0)  # Correct shape: p×0
        N_perp = V_r
        return S, N, N_perp, rankJ
    end

    # --- 2. Extract Invariant Component (Optimized) ---
    r0 = size(V_0, 2)
    
    # Note; differentiate J(θ)*V_0 instead of forming full J(θ)
    flat_JV_func = θ -> vec(compute_J(ϕ_func, θ) * V_0)
    H_JV = ForwardDiff.jacobian(flat_JV_func, θ0)  # (m*r0) × p

    # Build the stacked test matrix M_test = [H₁V₀; ...; HₚV₀]
    # Preallocated for type stability and performance
    M_test = Matrix{T}(undef, m * p, r0)
    for k in 1:p
        rows = (k-1)*m + 1 : k*m
        M_test[rows, :] = reshape(view(H_JV, :, k), m, r0)
    end

    # reduced SVD of the test matrix. V_M is the matrix of its right singular vectors with non-zero singular values
    M_test_svd = svd(M_test; full=false)
    MS = M_test_svd.S
    τM = rtolM * (isempty(MS) ? one(T) : maximum(MS))
    rankM = count(>(τM), MS)
    
    V_Mr = M_test_svd.V[:, 1:rankM]      # Basis for non-invariant combinations of V₀
    V_M0 = M_test_svd.V[:, rankM+1:end]  # Basis for invariant combinations of V₀

    # --- 3. Construct Final Reparameterization ---
    
    # Construct a basis for the potentially identifiable space (the complement of N)
    N_perp = hcat(V_r, V_0 * V_Mr)
    
    # Construct the basis for the invariant null space
    N = V_0 * V_M0

    return S, N, N_perp, rankJ
end



