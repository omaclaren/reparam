# Test reshape logic
flattened = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # [m1(t1), m2(t1), m3(t1), m1(t2), m2(t2), m3(t2), m1(t3), m2(t3), m3(t3), m1(t4), m2(t4), m3(t4)]

println("Flattened: ", flattened)
println("\nWant:")
println("  Row 1 (m1): [1, 4, 7, 10]")
println("  Row 2 (m2): [2, 5, 8, 11]")
println("  Row 3 (m3): [3, 6, 9, 12]")

# Method 1: reshape to (3, NT) - fills column by column
method1 = reshape(flattened, 3, 4)
println("\nMethod 1: reshape(v, 3, NT)")
println(method1)

# Method 2: reshape to (NT, 3) then transpose
method2 = reshape(flattened, 3, :)  # This gives 3×4
println("\nMethod 2: reshape(v, 3, :)")
println(method2)

# Method 3: permutedims
method3 = permutedims(method2, (2, 1))
println("\nMethod 3: permutedims(method2, (2,1))")
println(method3)

# Method 4: transpose
method4 = Matrix(transpose(method2))
println("\nMethod 4: transpose(method2)")
println(method4)
