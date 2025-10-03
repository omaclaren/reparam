using LinearAlgebra
using ForwardDiff

# ----------------------------------------------------------------
#   Functions for finding invariant null space and complement
# ----------------------------------------------------------------

function find_invariant_subspace(ϕ_func, θ0; tol=1e-8)
    """
    find_invariant_subspace(ϕ_func, θ0; tol=1e-8)

    Find invariant subspace of the auxiliary mapping ϕ_func at point θ0 using the
    "Initial-Extract-Final" algorithm.

    This function:
    1. Performs a local SVD to find a candidate basis.
    2. Uses a higher-order invariance test to filter for the invariant structure.
    3. Constructs the final, guaranteed sufficient reparameterization.

    # Arguments
    - `ϕ_func`: Auxiliary mapping function (mechanistic → distribution parameters)
    - `θ0`: Point in mechanistic parameter space to evaluate the Jacobian
    - `tol`: Tolerance for determining numerical rank (default: 1e-8)

    # Returns
    - `S`: Singular values from the initial Jacobian SVD
    - `N_T`: Matrix whose rows form an orthonormal basis for the invariant null space
    - `N_perp_T`: Matrix whose rows form an orthonormal basis for the potentially identifiable space
    - `rank`: Numerical rank of the Jacobian
    """
    # --- 1. Initial Transformation (Local SVD) ---
    J = compute_ϕ_Jacobian(ϕ_func, θ0)
    p = size(J, 2)  # Number of parameters
    m = size(J, 1)  # Output dimension of ϕ
    
    svd_result = svd(J)
    S = svd_result.S
    V = svd_result.V

    rank_J = sum(S .> tol)
    V_r = V[:, 1:rank_J]
    V_0 = V[:, rank_J+1:end]

    # If the local null space is empty, the analysis is complete
    if size(V_0, 2) == 0
        N_perp_T = Matrix{Float64}(I, p, p)
        N_T = Matrix{Float64}(undef, 0, p)  # Empty invariant null space
        return S, N_T, N_perp_T, rank_J
    end

    # --- 2. Extract Invariant Component ---
    
    # Define a function that returns the *flattened* Jacobian
    flat_J_func = θ -> vec(compute_ϕ_Jacobian(ϕ_func, θ))
    
    # This (m*p) x p matrix contains the flattened Hessian tensor
    H_tensor_flat = ForwardDiff.jacobian(flat_J_func, θ0)

    # Build the stacked test matrix M_test = [H₁V₀; ...; HₚV₀]
    M_test_blocks = []
    for k in 1:p
        H_k = reshape(H_tensor_flat[:, k], m, p)
        push!(M_test_blocks, H_k * V_0)
    end
    M_test = vcat(M_test_blocks...)

    # SVD of the test matrix. V_M is the matrix of its right singular vectors
    M_test_svd = svd(M_test)
    rank_M = sum(M_test_svd.S .> tol)
    V_Mr = M_test_svd.V[:, 1:rank_M]      # Basis for non-invariant combinations of V₀
    V_M0 = M_test_svd.V[:, rank_M+1:end]  # Basis for invariant combinations of V₀

    # --- 3. Construct Final Reparameterization ---
    
    # Construct a basis for the potentially identifiable space (the complement of N)
    N_perp = hcat(V_r, V_0 * V_Mr)
    N_perp_T = N_perp'
    
    # Construct the basis for the invariant null space
    N = V_0 * V_M0
    N_T = N'

    return S, N_T, N_perp_T, rank_J
end



