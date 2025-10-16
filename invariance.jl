# ----------------------------------------------------------------
#   Functions for finding invariant null space and complement
# ----------------------------------------------------------------

function find_invariant_subspace(ϕ_func, θ0;
                                 compute_J=compute_ϕ_Jacobian,
                                 rtolJ=sqrt(eps(real(eltype(θ0)))),
                                 atolM=nothing,  # Absolute tolerance (deprecated, use rtolM instead)
                                 rtolM=32*sqrt(eps(real(eltype(θ0)))),  # Relative tolerance (default: 32√eps ≈ 4.8e-7 for stiff systems)
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
    - `rtolM`: Relative tolerance for invariance test (default: 32√eps ≈ 4.8e-7).
               Effective tolerance is τM = rtolM * σ_max, which scales with Jacobian magnitude.
               The default value is calibrated for stiff ODE systems and provides ~2× safety margin.
               For smooth problems, can tighten to √eps; for very stiff systems, may need up to 1e-6.
               Heuristic: rtolM ≳ 1.5 * max(MS_invariant) / σ_max where MS are singular values of M_test.

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

    # Check if finite-difference method is requested
    if haskey(kwargs, :invariance_method) && kwargs[:invariance_method] == :finite_difference
        error("Finite-difference invariance test is not currently supported.\n" *
              "The previous implementation was found to be incorrect.\n" *
              "Please use the default Hessian-based method (remove invariance_method kwarg).\n" *
              "If you encounter nested AD errors, this indicates a limitation of the current implementation.")
    end

    # Hessian-based invariance test (default and recommended method)
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
    # Use τM threshold (relative or absolute depending on parameters)
    # We expect MS ≈ 0 for invariant null space
    rankM = count(>(τM), MS)

    # DIAGNOSTIC OUTPUT
    if haskey(kwargs, :verbose) && kwargs[:verbose]
        println("\n  Hessian-based invariance test diagnostics:")
        println("    τM (threshold): $τM")
        println("    M_test singular values (should be ~0 for invariant):")
        for i in 1:min(length(MS), r0)
            ratio = MS[i] / τM
            status = MS[i] > τM ? "✗ NON-INVARIANT" : "✓ invariant"
            println("      MS[$i] = $(round(MS[i], sigdigits=4)) ($(round(ratio, digits=2))×τM) $status")
        end
        println("    Classification: $rankM non-invariant, $(r0-rankM) invariant")
    end

    V_Mr = M_test_svd.V[:, 1:rankM]      # Coefficients for non-invariant combinations of V₀
    V_M0 = M_test_svd.V[:, rankM+1:end]  # Coefficients for invariant combinations of V₀

    # --- 3. Construct Final Reparameterization Subspaces ---

    # N_perp spans the orthogonal complement of the invariant null space
    # This is the identifiable space (for minimal image) or identifiable + non-invariant (for image)
    N_perp = hcat(V_r, V_0 * V_Mr)

    # N spans the invariant null space
    # Empty for minimal image, non-empty for image reparameterization
    N = V_0 * V_M0

    return S, N, N_perp, rankJ
end



