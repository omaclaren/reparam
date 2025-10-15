# ----------------------------------------------------------------
#   Functions for finding invariant null space and complement
# ----------------------------------------------------------------

function find_invariant_subspace(ϕ_func, θ0;
                                 compute_J=compute_ϕ_Jacobian,
                                 rtolJ=sqrt(eps(real(eltype(θ0)))),
                                 atolM=nothing,  # Absolute tolerance (deprecated, use rtolM instead)
                                 rtolM=sqrt(eps(real(eltype(θ0)))),  # Relative tolerance (default: √eps, same as rtolJ)
                                 kwargs...)

    """
    Finds invariant null subspace and its orthogonal complement for the auxiliary mapping ϕ_func at point θ0.

    Implements Algorithm 1 from the paper: Invariant Image Reparameterisation (IIR).
    
    Key steps:
    1. Local SVD in current parameterisation to find a candidate null space basis.
    2. Higher-order invariance test to separate invariant from non-invariant null space directions.
    3. Construction of the final reparameterization subspaces.

    The function tests whether null space directions remain in the null space under small 
    perturbations using the condition: H_i(θ*)α = 0 for all i, where H_i are Hessian slices.
    
    This determines whether we have:
    - Minimal image reparameterization: if full null space is invariant (N is empty)
    - Image reparameterization: if only part of null space is invariant (N is non-empty)

    # Arguments
    - `ϕ_func`: Auxiliary mapping function (mechanistic → distribution parameters)
    - `θ0`: Point in current parameter space to evaluate the Jacobian (typically MLE in f-transformed space)
    - `compute_J`: Function to compute Jacobian (default: compute_ϕ_Jacobian).
                   Can pass custom implementation for flexibility.
    - `rtolJ`: Relative tolerance for determining numerical rank of J (default: √eps ≈ 1.5e-8)
    - `atolM`: Absolute tolerance for invariance test (default: nothing, deprecated).
               Only use for backward compatibility with old code that requires fixed absolute tolerance.
    - `rtolM`: Relative tolerance for invariance test (default: √eps ≈ 1.5e-8, consistent with rtolJ).
               Effective tolerance is τM = rtolM * σ_max, which scales with Jacobian magnitude.
               This is the recommended approach as it adapts to varying parameter scales automatically.

    # Returns
    - `S`: Singular values from the initial Jacobian SVD
    - `N`: Matrix (p×k₀) whose columns form an orthonormal basis for the invariant null space.
           Empty if full null space is invariant (minimal image case).
    - `N_perp`: Matrix (p×k⊥) whose columns form an orthonormal basis for the identifiable space.
                Orthogonal complement of the invariant null space.
    - `rankJ`: Numerical rank of the Jacobian

    # Reparameterization Construction
    The reparameterization matrix is A = N_perp', giving:
    - Minimal image: ψ(θ) = f⁻¹(A f(θ)) captures all identifiable combinations
    - Image: ψ(θ) = f⁻¹(A f(θ)) captures identifiable + some non-invariant combinations
    
    # Examples
    ```julia
    # Standard usage (e.g., with f = log)
    S, N, N_perp, rankJ = find_invariant_subspace(ϕ_func, log.(θ_mle))
    
    # Construct reparameterization matrix
    A = N_perp'
    
    # Check type of reparameterization
    if size(N, 2) == 0
        println("Minimal image reparameterization - maximum reduction")
    else
        println("Image reparameterization - dimension \$(size(N, 2)) invariant null space remains")
    end
    
    # With custom Jacobian computation
    S, N, N_perp, rankJ = find_invariant_subspace(ϕ_func, θ0; 
        compute_J = (f, θ) -> FiniteDiff.finite_difference_jacobian(f, θ))
    ```
    """

    T = real(eltype(θ0))
    
    # --- 1. Initial Transformation (Local SVD) ---
    # Compute Jacobian and perform full SVD to get complete null space basis
    J = compute_J(ϕ_func, θ0)
    m, p = size(J)  # m = distribution params, p = mechanistic params
    
    # CRITICAL: full=true for complete nullspace when p>m
    svd_result = svd(J; full=true)
    S = svd_result.S
    V = svd_result.V  # p×p
    
    # Relative (approximate) rank determination
    σmax = maximum(S)
    τJ = rtolJ * σmax
    rankJ = count(>(τJ), S)
    
    V_r = V[:, 1:rankJ]        # Right singular vectors for non-zero singular values
    V_0 = V[:, rankJ+1:end]    # Right singular vectors for zero singular values (null space basis)

    # If the local null space is empty, no reparameterization needed
    if size(V_0, 2) == 0
        N = zeros(T, p, 0)  # Empty invariant null space
        N_perp = V_r        # Full space is identifiable
        return S, N, N_perp, rankJ
    end

    # --- 2. Extract Invariant Component via Higher-Order Test ---
    r0 = size(V_0, 2)

    # Compute effective invariance tolerance
    # Prefer rtolM (relative, scales with problem) over atolM (absolute, legacy)
    if !isnothing(atolM)
        # Backward compatibility: use absolute tolerance if explicitly provided
        τM = atolM
    else
        # Default: use relative tolerance (consistent with rtolJ)
        τM = rtolM * σmax
    end

    # Check if we should use finite-difference invariance test (for stiff ODEs)
    use_fd_invariance = haskey(kwargs, :invariance_method) && kwargs[:invariance_method] == :finite_difference

    if use_fd_invariance
        # Finite-difference invariance test (single-level AD only)
        # For each null vector α ∈ V_0, perturb θ along α and check if J(θ+δ)·α ≈ 0

        ε = get(kwargs, :fd_epsilon, 1e-6)  # Perturbation size
        n_probes = get(kwargs, :fd_n_probes, 3)  # Number of perturbation points per direction

        # Build test matrix by evaluating J(θ + ε·s·α)·α for multiple s values
        M_test = zeros(T, m * n_probes, r0)

        for j in 1:r0  # For each null vector
            α = V_0[:, j]

            # Test at multiple perturbations: ±ε, ±2ε, etc.
            for i in 1:n_probes
                s = (i - (n_probes+1)/2) * ε  # e.g., [-2ε, -ε, 0, ε, 2ε] for n_probes=5
                if abs(s) < 1e-12
                    s = ε  # Avoid testing exactly at θ0 (already know J(θ0)·α ≈ 0)
                end

                θ_pert = θ0 + s * α
                J_pert = compute_J(ϕ_func, θ_pert)

                # Store J(θ_pert)·α
                row_start = (i-1)*m + 1
                M_test[row_start:row_start+m-1, j] = J_pert * α
            end
        end

        # α is invariant if ||J(θ+δ)·α|| stays small for all perturbations
        # Use column norms to classify
        invariance_scores = [norm(M_test[:, j]) for j in 1:r0]

        # Use τM as threshold for invariance (scales with Jacobian magnitude if rtolM provided)
        invariant_mask = invariance_scores .< τM
        rankM = count(.!invariant_mask)  # Number of non-invariant directions

        # Separate invariant from non-invariant
        invariant_indices = findall(invariant_mask)
        noninvariant_indices = findall(.!invariant_mask)

        V_Mr = r0 > 0 && rankM > 0 ? V_0[:, noninvariant_indices] : zeros(T, p, 0)
        V_M0 = r0 > 0 && length(invariant_indices) > 0 ? V_0[:, invariant_indices] : zeros(T, p, 0)

        # Need to return as coefficient matrices (like SVD.V)
        if rankM > 0
            V_Mr_coeff = Matrix{T}(I, r0, r0)[:, noninvariant_indices]
        else
            V_Mr_coeff = zeros(T, r0, 0)
        end

        if length(invariant_indices) > 0
            V_M0_coeff = Matrix{T}(I, r0, r0)[:, invariant_indices]
        else
            V_M0_coeff = zeros(T, r0, 0)
        end

        V_Mr = V_Mr_coeff
        V_M0 = V_M0_coeff

    else
        # Original Hessian-based test (requires nested AD)
        # Efficiently compute Hessian-vector products: differentiate J(θ)*V_0 instead of full J(θ)
        # This gives (m*r0)×p instead of (m*p)×p - significant savings when r0 << p
        flat_JV_func = θ -> vec(compute_J(ϕ_func, θ) * V_0)
        H_JV = ForwardDiff.jacobian(flat_JV_func, θ0)  # (m*r0) × p

        # Build the stacked test matrix M_test = [H₁V₀; H₂V₀; ...; HₚV₀]
        # This implements the invariance condition: H_i(θ*)α = 0 for all i
        M_test = Matrix{T}(undef, m * p, r0)
        for k in 1:p
            rows = (k-1)*m + 1 : k*m
            M_test[rows, :] = reshape(view(H_JV, :, k), m, r0)
        end

        # Reduced SVD to separate invariant from non-invariant null space directions
        M_test_svd = svd(M_test; full=false)
        MS = M_test_svd.S
        # Use τM threshold (absolute or relative depending on rtolM parameter)
        # We expect MS ≈ 0 for invariant null space
        rankM = count(>(τM), MS)

        V_Mr = M_test_svd.V[:, 1:rankM]      # Coefficients for non-invariant combinations of V₀
        V_M0 = M_test_svd.V[:, rankM+1:end]  # Coefficients for invariant combinations of V₀
    end

    # --- 3. Construct Final Reparameterization Subspaces ---

    # N_perp spans the orthogonal complement of the invariant null space
    # This is the identifiable space (for minimal image) or identifiable + non-invariant (for image)
    N_perp = hcat(V_r, V_0 * V_Mr)

    # N spans the invariant null space
    # Empty for minimal image, non-empty for image reparameterization
    N = V_0 * V_M0

    return S, N, N_perp, rankJ
end



