# Plot 2D Profile from Saved Data
# Handles headless plotting by avoiding display() and setting GR backend options

ENV["GKSwstype"] = "nul"

using Serialization
using Plots
using LaTeXStrings
using Measures
using Distributions
using Printf

println("="^70)
println("PLOTTING FROM SAVED DATA")
println("="^70)

# Default plotting style
default(
    xguidefontsize=24,
    yguidefontsize=24,
    xtickfontsize=16,
    ytickfontsize=16,
    legendfontsize=16,
    bottom_margin=6mm,
    left_margin=6mm,
    top_margin=4.5mm,
    right_margin=4.5mm
)

# Redefine plot_2D_contour locally without display()
function plot_2D_contour_headless(model_name, ψ_values, lnlike_ψ_values, varnames;
    ψ_true=[], ψ_MLE=[], save_dir="./", fmt=:png, dpi=600,
    l_level=95, nshade_levels=20)
    
    println("Plotting 2D contour...")

    # Split into grid components for plotting. Need unique to undo Cartesian product
    ψ1_values = unique([ψ1 for (ψ1, _) in ψ_values])
    ψ2_values = unique([ψ2 for (_, ψ2) in ψ_values])
    lnlike_ψ_values = reshape(lnlike_ψ_values, length(ψ1_values), length(ψ2_values))
    
    # Convert to likelihood scale. Note: returned normalised already
    like_ψ_values = exp.(lnlike_ψ_values)
    
    # Contour plots with chi square calibration
    df = 2
    lstar = exp(-quantile(Chisq(df), l_level/100)/2)
    
    plt = contourf(ψ1_values, ψ2_values, like_ψ_values',
                  color=:dense, levels=nshade_levels, lw=0)

    # Explicitly add colorbar
    plot!(colorbar=true)
    
    contour!(ψ1_values, ψ2_values, like_ψ_values',
            levels=[lstar], color=:black, lw=1, legend=false, fill = false)
    
    xlabel!(latexstring(varnames["ψ1"]))
    ylabel!(latexstring(varnames["ψ2"]))
    xlims!(minimum(ψ1_values), maximum(ψ1_values))
    ylims!(minimum(ψ2_values), maximum(ψ2_values))
    
    # Add best and true
    if length(ψ_MLE) > 0
        scatter!([ψ_MLE[1]], [ψ_MLE[2]], mc=:silver, msc=:match, markersize=8, markershape=:circle, legend=false)
    end
    
    if length(ψ_true) > 0
        scatter!([ψ_true[1]], [ψ_true[2]], mc=:darkgoldenrod, msc=:match, markersize=10, markershape=:star, legend=false)
    end
    
    # NO DISPLAY CALL
    gr(fmt=fmt, dpi=dpi)
    
    # Use absolute path
    filename = joinpath(pwd(), save_dir * model_name * "_" * varnames["ψ1_save"] * "_" * varnames["ψ2_save"] * "." * string(fmt))
    println("Saving to $filename")
    savefig(plt, filename)
    println("Saved successfully.")
end

# Load data
println("Loading data...")
if !isfile("repressilator_2D_production_data.jls")
    println("Error: Data file not found!")
    exit(1)
end

(θ_2d_vals, ll_2d_vals, θ_true, θ_log_MLE) = deserialize("repressilator_2D_production_data.jls")

# Prepare for plotting
varnames = Dict(
    "ψ1" => "\\beta_1", "ψ1_save" => "beta1",
    "ψ2" => "K_1",      "ψ2_save" => "K1"
)

# Convert log parameters to natural scale
θ_2d_natural = [exp.(θ) for θ in θ_2d_vals]
ψ_2d_natural = [[θ[7], θ[10]] for θ in θ_2d_natural]

ψ_MLE_natural = [exp(θ_log_MLE[7]), exp(θ_log_MLE[10])]
ψ_true_natural = [θ_true[7], θ_true[10]]

# Plot
plot_2D_contour_headless(
    "repressilator_production",
    ψ_2d_natural,
    ll_2d_vals,
    varnames;
    ψ_true=ψ_true_natural,
    ψ_MLE=ψ_MLE_natural,
    save_dir="",  # Empty because we use joinpath with pwd() inside
    nshade_levels=30
)

println("Done.")
