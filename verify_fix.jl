#!/usr/bin/env julia

# Verify the fix
mRNA = [1.0 2.0 3.0 4.0 5.0;          # m1 at times 1-5
        10.0 20.0 30.0 40.0 50.0;      # m2 at times 1-5
        100.0 200.0 300.0 400.0 500.0] # m3 at times 1-5

println("mRNA matrix (3×5):")
println(mRNA)

# OLD (WRONG) way
old_way = vec(mRNA')
println("\nOLD: vec(mRNA') = ", old_way)
println("This is SPECIES-MAJOR: [m1_all..., m2_all..., m3_all...]")

# NEW (CORRECT) way
new_way = vec(mRNA)
println("\nNEW: vec(mRNA) = ", new_way)
println("This is TIME-MAJOR: [m1(t1), m2(t1), m3(t1), m1(t2), ...]")

# Now test the reshape
println("\n" * "="^60)
println("Testing reshape with CORRECT (new) flattening:")
println("="^60)

reshaped = reshape(new_way, 3, 5)
println("\nreshape(new_way, 3, 5):")
println(reshaped)
println("\nRow 1 (m1): ", reshaped[1, :], " (should be [1, 2, 3, 4, 5])")
println("Row 2 (m2): ", reshaped[2, :], " (should be [10, 20, 30, 40, 50])")
println("Row 3 (m3): ", reshaped[3, :], " (should be [100, 200, 300, 400, 500])")

println("\n✓ Match original?", reshaped == mRNA)
