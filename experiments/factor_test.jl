using LinearAlgebra
using FactorLoadingMatrices

# Test case from factor_test.py
# Input: balanced loadings [1,1,1,1] / [1,1,-1,-1]
# Expected output: sparse loadings [1,1,0,0] / [0,0,1,1] (up to sign/scale)

# NOTE: Multiple random restarts appear necessary for our application.
# Tentative observation: When applying varimax to SVD output from symmetric problems
# (small dimensions, perfectly balanced loadings), the optimization surface can be
# nearly flat with many poor local optima. Preliminary testing suggests ~200 restarts
# help reliably find sparse solutions. This may differ from traditional factor analysis
# where data asymmetries provide more structure. Further investigation needed.

N = [1.0 1.0;
     1.0 1.0;
     1.0 -1.0;
     1.0 -1.0]

println("Original N_perp (balanced loadings):")
println(N)
println()

# Normalize columns
col_norms = [norm(N[:, i]) for i in 1:size(N, 2)]
N_norm = N ./ col_norms'

println("Column-normalized N_norm:")
println(N_norm)
println()

# Apply varimax rotation with multiple random restarts
function varimax_objective(L)
    # Varimax objective: maximize variance of squared loadings
    n, p = size(L)
    sum(sum(L.^4, dims=1) .- (sum(L.^2, dims=1).^2) ./ n)
end

n_restarts = 200

# Use let block or function to avoid global scope issues
function find_best_varimax(N_norm, n_restarts)
    best_obj = -Inf
    best_rotated = N_norm

    for trial in 1:n_restarts
        # Random orthogonal rotation as starting point
        Q_rand = qr(randn(size(N_norm, 2), size(N_norm, 2))).Q
        candidate = N_norm * Matrix(Q_rand)

        # Apply varimax
        rotated = varimax(candidate)

        # Compute objective
        obj = varimax_objective(rotated)

        if obj > best_obj + 1e-6
            best_obj = obj
            best_rotated = rotated
        end
    end

    return best_obj, best_rotated
end

best_obj, best_rotated = find_best_varimax(N_norm, n_restarts)

println("="^60)
println("VARIMAX RESULTS")
println("="^60)
println("Best varimax objective: ", best_obj)
println()

# Re-orthonormalize (QR decomposition)
Q_final, R = qr(best_rotated)
rotated = Matrix(Q_final) .* col_norms'

println("Rotated (before thresholding):")
println(rotated)
println()

# Threshold small entries
threshold = 1e-2
rotated_var = copy(rotated)
rotated_var[abs.(rotated_var) .< threshold] .= 0.0

# Renormalize non-zero columns
for j in 1:size(rotated_var, 2)
    col_norm = norm(rotated_var[:, j])
    if col_norm > 0
        rotated_var[:, j] ./= col_norm
    end
end

println("Final rotated N_perp (after thresholding & renormalizing):")
println(round.(rotated_var, digits=6))
println()

# Check: should be approximately [1,1,0,0] and [0,0,1,1] (up to signs and column order)
println("Expected pattern: sparse parameter-local loadings")
println("Column 1 should load on parameters 1-2 or 3-4")
println("Column 2 should load on the other pair")
println()

# Now test PROMAX
println("="^60)
println("PROMAX RESULTS")
println("="^60)

function find_best_promax(N_norm, n_restarts; power=3)
    best_obj = -Inf
    best_rotated = N_norm

    for trial in 1:n_restarts
        # Random orthogonal rotation as starting point
        Q_rand = qr(randn(size(N_norm, 2), size(N_norm, 2))).Q
        candidate = N_norm * Matrix(Q_rand)

        # Apply promax
        rotated = promax(candidate; power=power)

        # Compute objective (using varimax objective for comparison)
        obj = varimax_objective(rotated)

        if obj > best_obj + 1e-6
            best_obj = obj
            best_rotated = rotated
        end
    end

    return best_obj, best_rotated
end

best_obj_promax, best_rotated_promax = find_best_promax(N_norm, n_restarts)

println("Best promax objective (varimax metric): ", best_obj_promax)
println()

# Promax doesn't preserve orthonormality, so just rescale by original norms
rotated_promax = best_rotated_promax .* col_norms'

println("Rotated (before thresholding):")
println(rotated_promax)
println()

# Check orthogonality
println("Column inner products (should be identity for orthogonal):")
println(rotated_promax' * rotated_promax)
println()

# Threshold small entries
rotated_promax_thresh = copy(rotated_promax)
rotated_promax_thresh[abs.(rotated_promax_thresh) .< threshold] .= 0.0

# Renormalize non-zero columns
for j in 1:size(rotated_promax_thresh, 2)
    col_norm = norm(rotated_promax_thresh[:, j])
    if col_norm > 0
        rotated_promax_thresh[:, j] ./= col_norm
    end
end

println("Final rotated N_perp (after thresholding & renormalizing):")
println(round.(rotated_promax_thresh, digits=6))
println()

println("="^60)
println("COMPARISON")
println("="^60)
println("Varimax sparsity (sum of small values): ", sum(abs.(rotated) .< threshold))
println("Promax sparsity (sum of small values): ", sum(abs.(rotated_promax) .< threshold))
