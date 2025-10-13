using LinearAlgebra

"""
    varimax(A; gamma=1.0, maxit=1000, reltol=1e-6)

Perform varimax rotation on loading matrix A.
Maximizes the variance of squared loadings (Kaiser's varimax criterion).

# Arguments
- `A`: n×k loading matrix
- `gamma`: 1.0 for varimax (default), 0.0 for quartimax
- `maxit`: maximum iterations
- `reltol`: relative tolerance for convergence

# Returns
Rotated loading matrix

# Reference
Kaiser, H. F. (1958). The varimax criterion for analytic rotation in factor analysis.
"""
function varimax(A; gamma=1.0, maxit=1000, reltol=1e-6)
    n, k = size(A)

    # Initialize rotation matrix
    T = Matrix{Float64}(I, k, k)

    for iter in 1:maxit
        A_rot = A * T

        # Compute gradient
        u = A_rot.^2
        v = zeros(k, k)

        for i in 1:k
            for j in 1:k
                if i != j
                    v[i,j] = sum(A_rot[:,i] .* A_rot[:,j] .* (u[:,i] - u[:,j] - gamma * (sum(u[:,i]) - sum(u[:,j])) / n))
                end
            end
        end

        # Check convergence
        if norm(v) < reltol
            break
        end

        # Update rotation via SVD
        U, _, V = svd(v)
        T_update = U * V'
        T = T * T_update
    end

    return A * T
end

# Test on the example
N = [1.0 1.0;
     1.0 1.0;
     1.0 -1.0;
     1.0 -1.0]

col_norms = [norm(N[:, i]) for i in 1:size(N, 2)]
N_norm = N ./ col_norms'

println("Original (normalized):")
println(N_norm)
println()

result = varimax(N_norm)
println("After varimax:")
println(round.(result, digits=4))
println()

# Threshold and renormalize
threshold = 1e-2
result_thresh = copy(result .* col_norms')
result_thresh[abs.(result_thresh) .< threshold] .= 0.0
for j in 1:size(result_thresh, 2)
    col_norm = norm(result_thresh[:, j])
    if col_norm > 0
        result_thresh[:, j] ./= col_norm
    end
end

println("After thresholding:")
println(round.(result_thresh, digits=4))
