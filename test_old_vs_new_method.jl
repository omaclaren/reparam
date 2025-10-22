# Compare OLD finite-difference (norm of columns) vs NEW (our "fix" with SVD)

println("OLD method: norm(M_test[:, j]) for each null vector")
println("NEW method: SVD(M_test).S")
println()
println("For repressilator, OLD found 3 invariant (WORKED)")
println("For repressilator, NEW finds 0 invariant (BROKEN)")
println()
println("The methods are testing completely different things!")
