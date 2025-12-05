using Plots

# Load both images
oct29 = load("repressilator_2D_distributed_beta1_K1.png")
today = load("repressilator_2D_10x10_beta1_K1.png")

# Display side by side
plot(plot(oct29, title="Oct 29 (100×100)"), 
     plot(today, title="Today (10×10 OLD params)"), 
     layout=(1,2), size=(1200, 500))

savefig("comparison.png")
println("Comparison saved")
