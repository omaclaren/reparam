using CSV, DataFrames, Plots

# Read the profile data
K1_data = CSV.read("repressilator_results/K1_profile.csv", DataFrame)
beta1_data = CSV.read("repressilator_results/beta1_profile.csv", DataFrame)
ratio_data = CSV.read("repressilator_results/K1_beta1_ratio_profile.csv", DataFrame)

# Convert log-likelihood to relative likelihood (normalized to max = 1)
K1_rel = exp.(K1_data.log_likelihood)
beta1_rel = exp.(beta1_data.log_likelihood)
ratio_rel = exp.(ratio_data.log_likelihood)

# Create plots
p1 = plot(K1_data.parameter_value, K1_rel,
    xlabel="K₁", ylabel="Relative Likelihood",
    title="K₁ Profile (Non-identifiable)",
    legend=false, linewidth=2, marker=:circle, ylims=(0, 1.05))
hline!([0.15], linestyle=:dash, color=:red, label="95% threshold")

p2 = plot(beta1_data.parameter_value, beta1_rel,
    xlabel="β₁", ylabel="Relative Likelihood",
    title="β₁ Profile (Non-identifiable)",
    legend=false, linewidth=2, marker=:circle, ylims=(0, 1.05))
hline!([0.15], linestyle=:dash, color=:red)

p3 = plot(ratio_data.ratio_value, ratio_rel,
    xlabel="K₁/β₁ ratio", ylabel="Relative Likelihood",
    title="K₁/β₁ Ratio Profile (Identifiable)",
    legend=false, linewidth=2, marker=:circle, ylims=(0, 1.05))
hline!([0.15], linestyle=:dash, color=:red)

# Combine into single figure
plot(p1, p2, p3, layout=(1,3), size=(1200, 400))
savefig("profile_comparison.png")

println("Plot saved to profile_comparison.png")
