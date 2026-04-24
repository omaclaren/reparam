# Protein prediction envelopes from a saved repressilator 2D profile result.
#
# This is a thin wrapper around the mRNA prediction postprocessor. It reuses the
# same accepted sets and plotting code, but extracts protein states p1, p2, p3.
# By default it uses row-consistent linear y-axes with zoom insets for the small
# non-identifiable p2/p3 panels. Pass --protein-yaxis=panel for per-panel y-axis
# scaling, --protein-yaxis=full to render the full envelope range, or
# --protein-yaxis=clipped to reproduce the older hard-clipped diagnostic view.
#
# Usage:
#   julia --project=. repressilator_protein_prediction_intervals_from_2d_profile.jl <results.jls> [--protein-yaxis=row-inset|panel|full|clipped]

ENV["REPRESSILATOR_PREDICTION_STATES"] = "protein"
include(joinpath(@__DIR__, "repressilator_prediction_intervals_from_2d_profile.jl"))
