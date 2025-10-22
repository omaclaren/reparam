#!/usr/bin/env julia

# Test vec() ordering in Julia
A = [1 2 3;
     4 5 6]  # 2×3 matrix

println("Matrix A (2×3):")
println(A)
println("\nvec(A):")
println(vec(A))
println("Expected (column-major): [1, 4, 2, 5, 3, 6]")

println("\n" * "="^50)

# Now test what predict_mRNA does
mRNA = [1.0 2.0 3.0 4.0 5.0;   # m1 at times 1,2,3,4,5
        10.0 20.0 30.0 40.0 50.0;   # m2 at times 1,2,3,4,5
        100.0 200.0 300.0 400.0 500.0]  # m3 at times 1,2,3,4,5

println("\nmRNA matrix (3×5, rows=species, cols=time):")
println(mRNA)

println("\nmRNA' (transposed to 5×3, rows=time, cols=species):")
println(mRNA')

println("\nvec(mRNA'):")
result = vec(mRNA')
println(result)

println("\nExpected (column-major from NT×3):")
println("  [1, 10, 100, 2, 20, 200, 3, 30, 300, 4, 40, 400, 5, 50, 500]")
println("  OR time-major: [1, 10, 100, 2, 20, 200, ...]")

println("\nAlternative: vec(mRNA) without transpose:")
println(vec(mRNA))
println("Expected (column-major from 3×NT):")
println("  [1, 10, 100, 2, 20, 200, ...]")
