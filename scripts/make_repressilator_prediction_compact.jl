# Create a compact 2×3 repressilator prediction figure from cached prediction-band files.
#
# This is plotting-only postprocessing. It does not rerun optimisation, profile
# calculations, or accepted-set construction. It simply reloads cached mRNA and
# protein prediction-band arrays and replots the m₁ and p₁ rows in a compact
# main-text layout.
#
# Usage:
#   julia --project=. scripts/make_repressilator_prediction_compact.jl \
#       <mrna_predictions_pub.jls> <protein_predictions_pub.jls> \
#       [--output-prefix=/path/to/output_base]
#
# Example:
#   julia --project=. scripts/make_repressilator_prediction_compact.jl \
#       /path/to/repressilator_16nuisance_50x50_results_predictions_pub.jls \
#       /path/to/repressilator_16nuisance_50x50_results_protein_predictions_pub.jls
#
# Outputs:
#   <output_prefix>_predictions_m1_p1_compact_pub.png
#   <output_prefix>_predictions_m1_p1_compact_pub.pdf

using Serialization
ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")
using Plots
using Measures
using LaTeXStrings

function parse_args(args)
    length(args) >= 2 || error(
        "Usage: julia --project=. scripts/make_repressilator_prediction_compact.jl <mrna_predictions_pub.jls> <protein_predictions_pub.jls> [--output-prefix=/path/to/output_base]"
    )

    mrna_file = args[1]
    protein_file = args[2]
    output_prefix = nothing

    for arg in args[3:end]
        if startswith(arg, "--output-prefix=")
            output_prefix = split(arg, "=", limit=2)[2]
        else
            error("Unknown argument: $arg")
        end
    end

    if isnothing(output_prefix)
        suffix = "_predictions_pub.jls"
        endswith(mrna_file, suffix) || error("Expected mRNA cache filename ending with '$suffix': $mrna_file")
        output_prefix = mrna_file[1:end-length(suffix)]
    end

    return mrna_file, protein_file, output_prefix
end

function require_keys(d::AbstractDict, keys::Vector{String}, name::String)
    for key in keys
        haskey(d, key) || error("Missing key '$key' in $name")
    end
end

function as_vec(x)
    return collect(Float64.(vec(x)))
end

function as_mat(x)
    return Matrix{Float64}(x)
end

function ci_cases(pred_cache::AbstractDict)
    return [
        ("full2d", latexstring("95\\%\\ \\mathrm{CI}\\ (\\mathrm{full\\ 2D})"), :purple,
         as_mat(pred_cache["lower_pred_from_accepted_ψ1ψ2"]),
         as_mat(pred_cache["upper_pred_from_accepted_ψ1ψ2"])),
        ("ident", latexstring("95\\%\\ \\mathrm{CI}\\ (K_1/\\beta_1)"), :steelblue3,
         as_mat(pred_cache["lower_pred_from_accepted_ψ1_profile"]),
         as_mat(pred_cache["upper_pred_from_accepted_ψ1_profile"])),
        ("nonident", latexstring("95\\%\\ \\mathrm{CI}\\ (\\beta_1 K_1)"), :orange,
         as_mat(pred_cache["lower_pred_from_accepted_ψ2_profile"]),
         as_mat(pred_cache["upper_pred_from_accepted_ψ2_profile"])),
    ]
end

function add_hidden_band_legend!(p, label, color)
    xg = [-1.0, -0.5]
    yg = [-1.0, -1.0]
    plot!(p, xg, yg, lw=0, fillrange=yg, fillalpha=0.20, color=color, label=label)
    return p
end

function add_hidden_line_legend!(p, label; color=:black, lw=2, linestyle=:solid)
    xg = [-1.0, -0.5]
    yg = [-1.0, -1.0]
    plot!(p, xg, yg, color=color, lw=lw, linestyle=linestyle, label=label)
    return p
end

function add_hidden_scatter_legend!(p, label; color=:black, ms=3, markershape=:x)
    scatter!(p, [-1.0], [-1.0], mc=color, msc=:match, ms=ms, markershape=markershape, label=label)
    return p
end

mrna_file, protein_file, output_prefix = parse_args(ARGS)

println("Loading mRNA prediction cache: $mrna_file")
mrna_cache = deserialize(mrna_file)
println("Loading protein prediction cache: $protein_file")
protein_cache = deserialize(protein_file)

required_keys = [
    "states", "t_pred", "t_obs", "data", "pred_gridded_MLE",
    "lower_pred_from_accepted_ψ1ψ2", "upper_pred_from_accepted_ψ1ψ2",
    "lower_pred_from_accepted_ψ1_profile", "upper_pred_from_accepted_ψ1_profile",
    "lower_pred_from_accepted_ψ2_profile", "upper_pred_from_accepted_ψ2_profile",
]
require_keys(mrna_cache, required_keys, "mRNA prediction cache")
require_keys(protein_cache, vcat(required_keys, ["pred_true"]), "protein prediction cache")

mrna_cache["states"] == "mrna" || error("Expected mRNA cache with states='mrna', got $(mrna_cache["states"]) ")
protein_cache["states"] == "protein" || error("Expected protein cache with states='protein', got $(protein_cache["states"]) ")

t_pred_mrna = as_vec(mrna_cache["t_pred"])
t_pred_protein = as_vec(protein_cache["t_pred"])
t_obs_mrna = as_vec(mrna_cache["t_obs"])
t_obs_protein = as_vec(protein_cache["t_obs"])

maximum(abs.(t_pred_mrna .- t_pred_protein)) <= 1e-12 || error("mRNA and protein prediction grids do not match")
maximum(abs.(t_obs_mrna .- t_obs_protein)) <= 1e-12 || error("mRNA and protein observation grids do not match")

t_pred = t_pred_mrna
t_obs = t_obs_mrna
T_end = maximum(t_pred)

mrna_data = as_mat(mrna_cache["data"])
mrna_pred_mle = as_mat(mrna_cache["pred_gridded_MLE"])
protein_pred_mle = as_mat(protein_cache["pred_gridded_MLE"])
protein_pred_true = protein_cache["pred_true"] === nothing ? nothing : as_mat(protein_cache["pred_true"])

mrna_cases = ci_cases(mrna_cache)
protein_cases = ci_cases(protein_cache)

# Match the row-1 mRNA y-limit rule from the existing 3×3 mRNA figure.
mrna_row = 1
mrna_ymax = max(
    maximum(mrna_pred_mle[mrna_row, :]),
    maximum(mrna_cases[1][5][mrna_row, :]),
    maximum(mrna_data[mrna_row, :]),
)
mrna_ylim = (0.0, 1.25 * mrna_ymax)

# Match the row-1 protein y-limit used in the existing row-inset protein figure.
protein_row = 1
protein_ylim = (0.0, 1.5e4)

println("m₁ y-limits: $mrna_ylim")
println("p₁ y-limits: $protein_ylim")
if protein_pred_true !== nothing
    println("p₁ truth max: $(maximum(protein_pred_true[protein_row, :]))")
end

# Publication styling matched to the existing repressilator prediction figures.
gr(size=(2200, 1040), dpi=300)

LINE_W = 2
BAND_ALPHA = 0.20

default(
    xguidefontsize=24,
    yguidefontsize=24,
    xtickfontsize=16,
    ytickfontsize=16,
    legendfontsize=16,
    bottom_margin=6mm,
    left_margin=6mm,
    top_margin=4.5mm,
    right_margin=4.5mm,
    framestyle=:box,
    grid=false,
)

column_titles = ["Full 2D", latexstring("K_1 / \\beta_1"), latexstring("\\beta_1 K_1")]
row_labels = [latexstring("m_1"), latexstring("p_1")]

plots_array = Any[]

for r in 1:2
    row_cases = r == 1 ? mrna_cases : protein_cases
    pred_mle = r == 1 ? mrna_pred_mle : protein_pred_mle
    pred_true = r == 2 ? protein_pred_true : nothing
    ylim = r == 1 ? mrna_ylim : protein_ylim

    for c in 1:3
        _, _, case_color, lower_band, upper_band = row_cases[c]
        row_idx = 1

        left_pad = c == 1 ? 12Plots.mm : 4.5Plots.mm
        right_pad = 4.5Plots.mm
        top_pad = 4.5Plots.mm
        bottom_pad = r == 2 ? 10Plots.mm : 3Plots.mm

        lo = clamp.(lower_band[row_idx, :], ylim[1], ylim[2])
        hi = clamp.(upper_band[row_idx, :], ylim[1], ylim[2])

        p = plot(t_pred, lo, lw=0,
                 fillrange=hi, fillalpha=BAND_ALPHA, color=case_color,
                 label="",
                 title=r == 1 ? column_titles[c] : "",
                 xlabel=r == 2 ? latexstring("t") : "",
                 ylabel=c == 1 ? row_labels[r] : "",
                 legend=(r == 1 && c == 1) ? :outertop : false,
                 legend_column=3,
                 framestyle=:box,
                 grid=false,
                 ylims=ylim,
                 xlims=(0.0, T_end),
                 xticks=0:2500:10000,
                 left_margin=left_pad,
                 right_margin=right_pad,
                 top_margin=top_pad,
                 bottom_margin=bottom_pad)

        plot!(p, t_pred, pred_mle[row_idx, :], lw=LINE_W, color=:black, label="")

        if r == 1
            scatter!(p, t_obs, mrna_data[row_idx, :],
                     mc=:black, msc=:match, ms=3, markershape=:x,
                     label="")
        end

        if pred_true !== nothing
            plot!(p, t_pred, pred_true[row_idx, :], lw=3, color=:darkgoldenrod,
                  linestyle=:dash, label="")
        end

        if r == 1 && c == 1
            add_hidden_band_legend!(p, latexstring("95\\%\\ \\mathrm{CI}\\ (\\mathrm{full\\ 2D})"), :purple)
            add_hidden_band_legend!(p, latexstring("95\\%\\ \\mathrm{CI}\\ (K_1/\\beta_1)"), :steelblue3)
            add_hidden_band_legend!(p, latexstring("95\\%\\ \\mathrm{CI}\\ (\\beta_1 K_1)"), :orange)
            add_hidden_line_legend!(p, "MLE"; color=:black, lw=LINE_W)
            add_hidden_scatter_legend!(p, "data"; color=:black, ms=3, markershape=:x)
            if protein_pred_true !== nothing
                add_hidden_line_legend!(p, "truth"; color=:darkgoldenrod, lw=3, linestyle=:dash)
            end
        end

        push!(plots_array, p)
    end
end

plt = plot(plots_array..., layout=(2, 3), size=(2200, 1040), margin=2.5mm)

output_png = "$(output_prefix)_predictions_m1_p1_compact_pub.png"
output_pdf = "$(output_prefix)_predictions_m1_p1_compact_pub.pdf"
savefig(plt, output_png)
savefig(plt, output_pdf)
println("Saved: $output_png")
println("Saved: $output_pdf")
