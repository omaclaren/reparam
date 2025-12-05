using Plots
using Images

# Side-by-side comparison
oct29 = load("repressilator_2D_distributed_beta1_K1.png")
repro = load("repressilator_2D_REPRODUCTION_oct29.png")

p = plot(
    plot(oct29, title="Oct 29 Original", axis=false, ticks=false),
    plot(repro, title="Today's Reproduction", axis=false, ticks=false),
    layout=(1,2), size=(1400, 600)
)

savefig(p, "comparison_oct29_vs_repro.png")
println("Saved: comparison_oct29_vs_repro.png")
