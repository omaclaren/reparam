# Run with:
#   julia --project=. "examples/stat_model.jl"
#
# This example fits the two-parameter statistical model, computes the
# invariant split in log coordinates, and builds an interpretable
# reparameterisation.
#
# Set `poisson_limit = true` below for the exact-limit case, or
# `poisson_limit = false` for the non-limit practical-identifiability case.

# Include ReparamTools.jl code if not already loaded
if !@isdefined(ReparamTools)
    include("../ReparamTools.jl")
    println("✓ ReparamTools module included")
else
    println("✓ ReparamTools module already included")
end

# Load required packages 
using .ReparamTools
using Plots
using Distributions
using LinearAlgebra
using Random

# Example-local helpers for the practical directional probe.
include("_directional_practical_probe_helpers.jl")

# Set random seed for reproducibility
Random.seed!(12)

# --------------------------------------------------------
# Model Definition
# Define model in xy = θ = [n, p] parameterization
# --------------------------------------------------------
# boolean for whether to use Poisson limit
poisson_limit = true
# Parameter -> data parameter mapping 
if poisson_limit
    ϕ_xy = xy -> [xy[1]*xy[2], xy[1]*xy[2]] # Maps (n,p) to (np,np)
else
    ϕ_xy = xy -> [xy[1]*xy[2], xy[1]*xy[2]*(1-xy[2])]  # Maps (n,p) to (np,np*(1-p))
end

# --------------------------------------------------------
# Setup and Data Generation
# --------------------------------------------------------

# Parameter -> distribution mapping. Use ϕ_xy explicitly 
distrib_xy = xy -> Normal(ϕ_xy(xy)[1], sqrt(ϕ_xy(xy)[2]))

# Variables and bounds
varnames = Dict("ψ1" => "n", "ψ2" => "p")
varnames["ψ1_save"] = "n"
varnames["ψ2_save"] = "p"

# Parameter bounds
n_min, n_max = 0.1, 500.0
p_min, p_max = 0.0001, 1.0

xy_lower_bounds = [n_min, p_min]
xy_upper_bounds = [n_max, p_max]

# Initial guess for optimisation
xy_initial = [50.0, 0.3]

# True parameter values
n_true, p_true = 100.0, 0.2
xy_true = [n_true, p_true]

# Generate/load data
# N_samples = 10
# data = rand(distrib_xy(xy_true),N_samples)
data = [21.9, 22.3, 12.8, 16.4, 16.4, 20.3, 16.2, 20.0, 19.7, 24.4]

# --------------------------------------------------------
# Original Parameterization Analysis
# --------------------------------------------------------
# Construct likelihood
lnlike_xy = construct_lnlike_xy(distrib_xy, data)
if poisson_limit
    model_name = "stat_model_xy_poisson"
else
    model_name = "stat_model_xy"
end
grid_steps = [500]
dim_all = length(xy_initial)
indices_all = 1:dim_all

# Point estimation (MLE)
target_indices = []  # empty for MLE
xy_MLE, lnlike_xy_MLE = profile_target(lnlike_xy, target_indices,
    xy_lower_bounds, xy_upper_bounds, 
    xy_initial; grid_steps=grid_steps)

# Quadratic approximation at MLE
lnlike_xy_ellipse, H_xy_ellipse = construct_ellipse_lnlike_approx(lnlike_xy, xy_MLE)

# Eigenanalysis of Fisher Information
evals, evecs = eigen(H_xy_ellipse; sortby = x -> -real(x))
println("Eigenvectors and eigenvalues for "*model_name)
println("Eigenvalues: ", evals)
println("Eigenvectors: ", evecs)

# Calculate prediction at MLE for reference
pred_mean_MLE = mean(distrib_xy(xy_MLE))
true_mean = mean(distrib_xy(xy_true))

# Determine svd of phi mapping in xy coordinates
J_ϕ_xy = compute_ϕ_Jacobian(ϕ_xy, xy_MLE)
U_xy, S_xy, V_xy = svd(J_ϕ_xy)
println("\nSVD analysis in original coordinates:")
println("Singular values: ", S_xy)
println("Right singular vectors (V): ")
display(V_xy)

# 1D Profiles
for i in 1:dim_all
    target_index = i
    nuisance_indices = setdiff(indices_all, target_index)
    nuisance_guess = xy_MLE[nuisance_indices]

    print("Variable: ", varnames["ψ"*string(i)], "\n")

    # Profile full likelihood
    ψω_values, lnlike_ψ_values = profile_target(lnlike_xy, target_index,
        xy_lower_bounds, xy_upper_bounds,
        nuisance_guess; grid_steps=grid_steps)

    # Profile quadratic approximation
    ψω_ellipse_values, lnlike_ψ_ellipse_values = profile_target(lnlike_xy_ellipse,
        target_index,
        xy_lower_bounds, xy_upper_bounds,
        nuisance_guess; grid_steps=grid_steps)

    # Extract profiled parameter values
    ψ_values = [ψω[target_index] for ψω in ψω_values]
    ψ_ellipse_values = [ψω[target_index] for ψω in ψω_ellipse_values]

    # Plot profiles
    plot_1D_profile(model_name, ψ_values, lnlike_ψ_values,
        varnames["ψ"*string(i)];
        varname_save=varnames["ψ"*string(i)*"_save"],
        ψ_true=xy_true[i], ψ_MLE=xy_MLE[i], save_dir="./figures/")

    plot_1D_profile_comparison(model_name, model_name*"_ellipse",
        ψ_values, ψ_ellipse_values,
        lnlike_ψ_values, lnlike_ψ_ellipse_values,
        varnames["ψ"*string(i)];
        varname_save=varnames["ψ"*string(i)*"_save"],
        ψ_true=xy_true[i], ψ_MLE1=xy_MLE[i], save_dir="./figures/")
end

# 2D Profiles
param_pairs = [(i,j) for i in 1:dim_all for j in (i+1):dim_all]

for (i,j) in param_pairs
    target_indices_ij = [i,j]
    nuisance_indices = setdiff(indices_all, target_indices_ij)
    nuisance_guess = xy_MLE[nuisance_indices]
    ψ_true_pair = xy_true[target_indices_ij]

    # Create a copy of varnames for this iteration
    current_varnames = deepcopy(varnames)
    current_varnames["ψ1"] = varnames["ψ"*string(i)]
    current_varnames["ψ2"] = varnames["ψ"*string(j)]
    current_varnames["ψ1_save"] = varnames["ψ"*string(i)*"_save"]
    current_varnames["ψ2_save"] = varnames["ψ"*string(j)*"_save"]

    # Profile full likelihood
    ψω_values, lnlike_ψ_values = profile_target(lnlike_xy, target_indices_ij,
        xy_lower_bounds, xy_upper_bounds,
        nuisance_guess; grid_steps=grid_steps)

    # Profile quadratic approximation
    ψω_ellipse_values, lnlike_ψ_ellipse_values = profile_target(lnlike_xy_ellipse,
        target_indices_ij,
        xy_lower_bounds, xy_upper_bounds,
        nuisance_guess; grid_steps=grid_steps)

    # Extract profiled parameter values
    ψ_values = [ψω[target_indices_ij] for ψω in ψω_values]
    ψ_ellipse_values = [ψω[target_indices_ij] for ψω in ψω_ellipse_values]

    # Plot contours
    plot_2D_contour(model_name, ψ_values, lnlike_ψ_values,
        current_varnames; ψ_true=ψ_true_pair, ψ_MLE=xy_MLE, save_dir="./figures/")

    # Plot comparison with quadratic approximation
    plot_2D_contour_comparison(model_name, model_name*"_ellipse",
        ψ_values, ψ_ellipse_values,
        lnlike_ψ_values, lnlike_ψ_ellipse_values,
        current_varnames; ψ_true=ψ_true_pair, ψ_MLE1=xy_MLE, save_dir="./figures/")

    # Get and plot 1D profiles from 2D grid
    ψ1_values, ψ2_values, like_ψ1_values, like_ψ2_values = get_1D_profiles_from_2D(
        ψ_values, lnlike_ψ_values)

    plot_1D_profile(model_name, ψ1_values, log.(like_ψ1_values),
        current_varnames["ψ1"];
        varname_save=current_varnames["ψ1_save"]*"_from_2D",
        ψ_true=ψ_true_pair[1], ψ_MLE=xy_MLE[i], save_dir="./figures/")

    plot_1D_profile(model_name, ψ2_values, log.(like_ψ2_values),
        current_varnames["ψ2"];
        varname_save=current_varnames["ψ2_save"]*"_from_2D",
        ψ_true=ψ_true_pair[2], ψ_MLE=xy_MLE[j], save_dir="./figures/")
end

# --------------------------------------------------------
# Log Parameterization Analysis
# --------------------------------------------------------

if poisson_limit
    model_name = "stat_model_log_poisson"
else
    model_name = "stat_model_log"
end

# Coordinate transformation
xytoXY_log(xy) = log.(xy)
XYtoxy_log(XY) = exp.(XY)

# Transform bounds and parameters
XY_log_lower_bounds = log.(xy_lower_bounds)
XY_log_upper_bounds = log.(xy_upper_bounds)
XY_log_initial = xytoXY_log(xy_initial)
XY_log_true = xytoXY_log(xy_true)

# Transform likelihood, distribution, and phi mapping
lnlike_XY_log = construct_lnlike_XY(lnlike_xy, XYtoxy_log)
distrib_XY_log = construct_distrib_XY(distrib_xy, XYtoxy_log)
ϕ_XY_log = construct_ϕ_XY(ϕ_xy, XYtoxy_log)

# Update variable names for log coordinates
varnames["ψ1"] = "\\ln\\ n"
varnames["ψ2"] = "\\ln\\ p"
varnames["ψ1_save"] = "ln_n"
varnames["ψ2_save"] = "ln_p"

# Point estimation in log coordinates
target_indices = []  # empty for MLE
XY_log_MLE, lnlike_XY_log_MLE = profile_target(lnlike_XY_log, target_indices,
    XY_log_lower_bounds, XY_log_upper_bounds, 
    XY_log_initial; grid_steps=grid_steps)

# Quadratic approximation at MLE
lnlike_XY_log_ellipse, H_XY_log_ellipse = construct_ellipse_lnlike_approx(lnlike_XY_log, XY_log_MLE)

# Eigenanalysis in log coordinates
evals_log, evecs_log = eigen(H_XY_log_ellipse; sortby = x -> -real(x))
println("Eigenvectors and eigenvalues for "*model_name)
for (i, eveci) in enumerate(eachcol(evecs_log))
    println("value: ", evals_log[i])
    println("vector: ", evecs_log[:,i])
end

# Determine svd of phi mapping in log coordinates
J_ϕ_XY_log, U_XY_log, S_XY_log, V_XY_log = compute_ϕ_Jacobian(ϕ_XY_log, XY_log_MLE, compute_svd=true)
println("\nSVD analysis in log coordinates:")
println("Singular values: ", S_XY_log)
println("Right singular vectors (V): ")
display(V_XY_log)

# Compare eigenvectors from Fisher Information with singular vectors
println("\nComparison of eigenvectors (1) and singular vectors (2):")
display(evecs_log)
display(V_XY_log)


for i in 1:dim_all
    target_index = i
    nuisance_indices = setdiff(indices_all, target_index)
    nuisance_guess = XY_log_MLE[nuisance_indices]

    print("Variable: ", varnames["ψ"*string(i)], "\n")

    # Profile full likelihood
    ψω_values, lnlike_ψ_values = profile_target(lnlike_XY_log, target_index,
        XY_log_lower_bounds, XY_log_upper_bounds,
        nuisance_guess; grid_steps=grid_steps)

    # Profile quadratic approximation
    ψω_ellipse_values, lnlike_ψ_ellipse_values = profile_target(lnlike_XY_log_ellipse,
        target_index,
        XY_log_lower_bounds, XY_log_upper_bounds,
        nuisance_guess; grid_steps=grid_steps)

    # Extract profiled parameter values
    ψ_values = [ψω[target_index] for ψω in ψω_values]
    ψ_ellipse_values = [ψω[target_index] for ψω in ψω_ellipse_values]

    # Plot profiles
    plot_1D_profile(model_name, ψ_values, lnlike_ψ_values,
        varnames["ψ"*string(i)];
        varname_save=varnames["ψ"*string(i)*"_save"],
        ψ_true=XY_log_true[i], ψ_MLE=XY_log_MLE[i], save_dir="./figures/")

    plot_1D_profile_comparison(model_name, model_name*"_ellipse",
        ψ_values, ψ_ellipse_values,
        lnlike_ψ_values, lnlike_ψ_ellipse_values,
        varnames["ψ"*string(i)];
        varname_save=varnames["ψ"*string(i)*"_save"],
        ψ_true=XY_log_true[i], ψ_MLE1=XY_log_MLE[i], save_dir="./figures/")
end

# 2D Profiles
param_pairs = [(i,j) for i in 1:dim_all for j in (i+1):dim_all]

for (i,j) in param_pairs
    target_indices_ij = [i,j]
    nuisance_indices = setdiff(indices_all, target_indices_ij)
    nuisance_guess = XY_log_MLE[nuisance_indices]
    ψ_true_pair = XY_log_true[target_indices_ij]

    # Create a copy of varnames for this iteration
    current_varnames = deepcopy(varnames)
    current_varnames["ψ1"] = varnames["ψ"*string(i)]
    current_varnames["ψ2"] = varnames["ψ"*string(j)]
    current_varnames["ψ1_save"] = varnames["ψ"*string(i)*"_save"]
    current_varnames["ψ2_save"] = varnames["ψ"*string(j)*"_save"]

    # Profile full likelihood
    ψω_values, lnlike_ψ_values = profile_target(lnlike_XY_log, target_indices_ij,
        XY_log_lower_bounds, XY_log_upper_bounds,
        nuisance_guess; grid_steps=grid_steps)

    # Profile quadratic approximation
    ψω_ellipse_values, lnlike_ψ_ellipse_values = profile_target(lnlike_XY_log_ellipse,
        target_indices_ij,
        XY_log_lower_bounds, XY_log_upper_bounds,
        nuisance_guess; grid_steps=grid_steps)

    # Extract profiled parameter values
    ψ_values = [ψω[target_indices_ij] for ψω in ψω_values]
    ψ_ellipse_values = [ψω[target_indices_ij] for ψω in ψω_ellipse_values]

    # Plot contours
    plot_2D_contour(model_name, ψ_values, lnlike_ψ_values,
        current_varnames; ψ_true=ψ_true_pair, ψ_MLE=XY_log_MLE, save_dir="./figures/")

    # Plot comparison with quadratic approximation
    plot_2D_contour_comparison(model_name, model_name*"_ellipse",
        ψ_values, ψ_ellipse_values,
        lnlike_ψ_values, lnlike_ψ_ellipse_values,
        current_varnames; ψ_true=ψ_true_pair, ψ_MLE1=XY_log_MLE, save_dir="./figures/")

    # Get and plot 1D profiles from 2D grid
    ψ1_values, ψ2_values, like_ψ1_values, like_ψ2_values = get_1D_profiles_from_2D(
        ψ_values, lnlike_ψ_values)

    plot_1D_profile(model_name, ψ1_values, log.(like_ψ1_values),
        current_varnames["ψ1"];
        varname_save=current_varnames["ψ1_save"]*"_from_2D",
        ψ_true=ψ_true_pair[1], ψ_MLE=XY_log_MLE[i], save_dir="./figures/")

    plot_1D_profile(model_name, ψ2_values, log.(like_ψ2_values),
        current_varnames["ψ2"];
        varname_save=current_varnames["ψ2_save"]*"_from_2D",
        ψ_true=ψ_true_pair[2], ψ_MLE=XY_log_MLE[j], save_dir="./figures/")
end

# --------------------------------------------------------
# Invariant Subspace Analysis in Log Coordinates
# --------------------------------------------------------

println("\n" * "="^60)
println("Invariant Subspace Analysis (Algorithm 1 from Paper)")
println("="^60)

# Apply find_invariant_subspace as described in Algorithm 1
# 1. Local SVD to find candidate null space basis V_0
# 2. Higher-order invariance test via Hessian to separate invariant/non-invariant
# 3. Construction of final reparameterization matrix

S_inv, N_inv, N_perp_inv, rank_inv = find_invariant_subspace(ϕ_XY_log, XY_log_MLE)

println("\nJacobian Analysis:")
println("  Singular values: ", S_inv)
println("  Numerical rank: ", rank_inv)
println("  Dimension of invariant null space (N): ", size(N_inv, 2))
println("  Dimension of identifiable space (N_perp): ", size(N_perp_inv, 2))

# Determine type of reparameterization
null_space_dim = length(XY_log_MLE) - rank_inv
if size(N_inv, 2) == null_space_dim && null_space_dim > 0
    println("\nFull null space is invariant")
    println("  Type: Minimal image reparameterization")
    reparam_type = "minimal_image"
elseif size(N_inv, 2) > 0 && size(N_inv, 2) < null_space_dim
    println("\nPartial null space is invariant (dimension ", size(N_inv, 2), " of ", null_space_dim, ")")
    println("  Type: Image (not minimal) reparameterization")
    reparam_type = "image"
elseif size(N_inv, 2) == 0 && null_space_dim > 0
    println("\nNo null space is invariant")
    println("  Type: Image (not minimal) reparameterization")
    reparam_type = "image"
else # null_space_dim == 0
    println("\nNo null space detected")
    println("  Type: Appears structurally identifiable")
    reparam_type = "identifiable"
end

# Display the key subspaces
if size(N_inv, 2) > 0
    println("\nInvariant null space basis N (columns):")
    display(N_inv)
    println("\nThese directions remain in the null space under perturbation")
end

println("\nIdentifiable space basis N_perp (columns):")
display(N_perp_inv)

# Construct reparameterization matrix as in Algorithm 1
A_inv = N_perp_inv'  # A = N_perp^T
println("\nReparameterization matrix A = N_perp^T:")
display(A_inv)

# ALWAYS show ranking by singular values (degree of identifiability)
println("\nParameter Combination Ranking by Identifiability:")
println("-"^60)
for i in 1:size(N_perp_inv, 2)
    println("Combination ", i, " (σ = ", round(S_inv[i], digits=3), "):")
    println("  Direction in log space: ", round.(N_perp_inv[:,i], digits=3))
    if i == 1
        println("  → Best identified combination")
    elseif i == size(N_perp_inv, 2)
        println("  → Least identified combination")
    else
        println("  → Moderately identified")
    end
end

# Practical rank check for near-identifiable settings (heuristic)
println("\nPractical Identifiability Check (heuristic):")
σ_rel = S_inv[1] > 0 ? S_inv ./ S_inv[1] : zeros(length(S_inv))
println("  Relative singular values (σᵢ/σ₁): ", round.(σ_rel, digits=4))

practical_rank_cutoffs = [0.2, 0.1, 0.05]
for cutoff in practical_rank_cutoffs
    practical_rank = count(>=(cutoff), σ_rel)
    println("  cutoff = ", cutoff, "  => practical rank ≈ ", practical_rank)
end

default_practical_cutoff = 0.2
practical_rank_default = count(>=(default_practical_cutoff), σ_rel)

if practical_rank_default < rank_inv
    println("  Suggestion at cutoff ", default_practical_cutoff, ":")
    println("    Keep first ", practical_rank_default, " combination(s) as well-identified.")
    println("    Treat the remaining ", rank_inv - practical_rank_default, " as weakly identified.")
    if practical_rank_default > 0
        println("    Kept direction(s) in log space:")
        display(round.(N_perp_inv[:, 1:practical_rank_default], digits=3))
    end
else
    println("  Suggestion at cutoff ", default_practical_cutoff, ": keep full dimension.")
end

# Shared monomial basis view for the final interpretable coordinates.
# Use informed selection on the identified side and simplicity-only selection on the null side.
param_names_iir = ["n", "p"]
residual_cap_iir = 1e-2
identified_basis_result = informed_monomial_basis_search(
    N_perp_inv, J_ϕ_XY_log' * J_ϕ_XY_log, S_inv[1]^2, param_names_iir;
    s_max=2, c_max=1, residual_cap=residual_cap_iir)
null_basis_result = simple_monomial_basis_search(
    N_inv, param_names_iir; s_max=2, c_max=1, residual_cap=residual_cap_iir, retry_support=true)

if !identified_basis_result.basis_ok
    error("Stepwise informed simple basis search failed on the identified side N_perp")
end
if !null_basis_result.basis_ok
    error("Singleton-first sparse basis search failed on the invariant null side N")
end

identified_basis_columns = monomial_basis_matrix(identified_basis_result.selected, length(param_names_iir))
identified_basis_labels = basis_labels(identified_basis_result.selected)
null_basis_columns = monomial_basis_matrix(null_basis_result.selected, length(param_names_iir))
null_basis_labels = basis_labels(null_basis_result.selected)
final_basis_columns = hcat(identified_basis_columns, null_basis_columns)
final_basis_labels = vcat(identified_basis_labels, null_basis_labels)
final_log_A = final_basis_columns'

# Directional probe for one-sided practical weakness in full-rank settings.
# Compare two weak-coordinate choices in x = log(θ):
# (i) the exact local SVD basis, and (ii) the simple monomial interpretable basis.
if !poisson_limit && rank_inv == length(XY_log_MLE) && length(S_inv) > 1
    println("\nDirectional Practical Check under two weak-coordinate choices:")

    svd_log_probe = svd(J_ϕ_XY_log)
    v_weak_raw = svd_log_probe.V[:, end]
    raw_weak_coord_row = svd_log_probe.V[:, end]
    raw_probe = directional_practical_probe(
        ϕ_XY_log, XY_log_MLE, v_weak_raw,
        XY_log_lower_bounds, XY_log_upper_bounds)

    weak_coord_index = size(N_perp_inv, 2)
    monomial_weak_coord_row = final_log_A[weak_coord_index, :]
    monomial_weak_direction = inv(final_log_A)[:, weak_coord_index]
    monomial_probe = directional_practical_probe(
        ϕ_XY_log, XY_log_MLE, monomial_weak_direction,
        XY_log_lower_bounds, XY_log_upper_bounds)

    print_directional_practical_probe(
        "Exact local weak coordinate (SVD basis)", raw_probe, XYtoxy_log, XY_log_MLE;
        coord_row=raw_weak_coord_row)

    print_directional_practical_probe(
        "Simple monomial weak coordinate (interpretable basis)", monomial_probe, XYtoxy_log, XY_log_MLE;
        coord_row=monomial_weak_coord_row)
end

println("\n" * "="^60)
println("Model-Specific Notes")
println("="^60)

if poisson_limit
    println("\nPoisson limit case: ϕ(n,p) = [np, np]")
    if size(N_inv, 2) == 1
        println("  Invariant null space dimension: 1")
        println("  Identifiable space dimension: 1")
    end
else
    println("\nBinomial case: ϕ(n,p) = [np, np(1-p)]")
    if size(N_inv, 2) == 0 && rank_inv == 2
        println("  Appears structurally identifiable (no invariant null space)")
        println("  Condition number: ", round(S_inv[1] / S_inv[end], digits=1))
    end
end

# --------------------------------------------------------
# IIR Parameterization Analysis
# (Invariant Image Reparameterization - replaces "Sloppy-Informed")
# --------------------------------------------------------
if poisson_limit
    model_name = "stat_model_iir_poisson"
else
    model_name = "stat_model_iir"
end

println("\n" * "="^60)
println("IIR Parameterization: ", model_name)
println("="^60)

println("\nUsing invariant subspace analysis (Algorithm 1 from paper)")
println("Selecting final interpretable coordinates with the shared monomial basis search")

println("\nSelected simple monomial basis labels:")
for (i, label) in enumerate(final_basis_labels)
    println("  ψ_", i, " = ", label)
end
println("\nSelected simple monomial reparameterization matrix:")
display(final_log_A)

if poisson_limit
    println("\nPoisson-limit interpretation:")
    println("  First coordinate is the identifiable combination np.")
    println("  Second coordinate is the invariant combination n/p.")
elseif reparam_type == "identifiable"
    println("\nNon-limit interpretation:")
    println("  No exact invariant null space is present.")
    println("  The informed simple monomial basis is used as a local interpretable basis.")
    println("  np is the stronger local combination; n/p is weaker but non-invariant.")
else
    println("\nNon-limit interpretation:")
    println("  A mixed image reparameterization was detected; keep both coordinates.")
end

println("\nTransformation matrices:")
println("Forward (original → IIR):")
display(final_log_A)
println("\nInverse (IIR → original):")
display(inv(final_log_A))

# Define coordinate transformation using the selected monomial basis.
# reparam expects columns = parameter combinations, so pass final_basis_columns.
xytoXY_iir, XYtoxy_iir = reparam(final_basis_columns)

# Transform likelihood, distribution, and phi mapping
lnlike_XY_iir = construct_lnlike_XY(lnlike_xy, XYtoxy_iir)
distrib_XY_iir = construct_distrib_XY(distrib_xy, XYtoxy_iir)
ϕ_XY_iir = construct_ϕ_XY(ϕ_xy, XYtoxy_iir)

# Set bounds for iir coordinates
XY_iir_lower_bounds = [13.0, 25.0]
XY_iir_upper_bounds = [25.0, 1000.0]
XY_iir_initial = [mean([XY_iir_lower_bounds[1], XY_iir_upper_bounds[1]]),
                  mean([XY_iir_lower_bounds[2], XY_iir_upper_bounds[2]])]

# transform true value
XY_iir_true = xytoXY_iir(xy_true)

# Update variable names for iir coordinates
varnames["ψ1"] = "np"
varnames["ψ2"] = "\\frac{n}{p}"
varnames["ψ1_save"] = "np"
varnames["ψ2_save"] = "n_over_p"

# Point estimation in iir coordinates
target_indices = []  # empty for MLE
XY_iir_MLE, lnlike_XY_iir_MLE = profile_target(lnlike_XY_iir, target_indices,
    XY_iir_lower_bounds, XY_iir_upper_bounds, 
    XY_iir_initial; grid_steps=grid_steps)

# Quadratic approximation at MLE
lnlike_XY_iir_ellipse, H_XY_iir_ellipse = construct_ellipse_lnlike_approx(lnlike_XY_iir, XY_iir_MLE)

# Eigenanalysis in iir coordinates
evals_iir, evecs_iir = eigen(H_XY_iir_ellipse; sortby = x -> -real(x))
println("Eigenvectors and eigenvalues for "*model_name)
println("Eigenvalues: ", evals_iir)
println("Eigenvectors: ", evecs_iir)

# Determine svd of phi mapping in iir coordinates
J_ϕ_XY_iir, U_XY_iir, S_XY_iir, V_XY_iir = compute_ϕ_Jacobian(ϕ_XY_iir, XY_iir_MLE, compute_svd=true)

# Compare eigenvectors from Fisher Information with singular vectors
println("\nComparison of eigenvectors (1) and singular vectors (2):")
display(evecs_iir)
display(V_XY_iir)

# 1D Profiles
for i in 1:dim_all
    target_index = i
    nuisance_indices = setdiff(indices_all, target_index)
    nuisance_guess = XY_iir_MLE[nuisance_indices]

    print("Variable: ", varnames["ψ"*string(i)], "\n")

    # Profile full likelihood
    ψω_values, lnlike_ψ_values = profile_target(lnlike_XY_iir, target_index,
        XY_iir_lower_bounds, XY_iir_upper_bounds,
        nuisance_guess; grid_steps=grid_steps)

    # Profile quadratic approximation
    ψω_ellipse_values, lnlike_ψ_ellipse_values = profile_target(lnlike_XY_iir_ellipse,
        target_index,
        XY_iir_lower_bounds, XY_iir_upper_bounds,
        nuisance_guess; grid_steps=grid_steps)

    # Extract profiled parameter values
    ψ_values = [ψω[target_index] for ψω in ψω_values]
    ψ_ellipse_values = [ψω[target_index] for ψω in ψω_ellipse_values]

    # Plot profiles
    plot_1D_profile(model_name, ψ_values, lnlike_ψ_values,
        varnames["ψ"*string(i)];
        varname_save=varnames["ψ"*string(i)*"_save"],
        ψ_true=XY_iir_true[i], ψ_MLE=XY_iir_MLE[i], save_dir="./figures/")

    plot_1D_profile_comparison(model_name, model_name*"_ellipse",
        ψ_values, ψ_ellipse_values,
        lnlike_ψ_values, lnlike_ψ_ellipse_values,
        varnames["ψ"*string(i)];
        varname_save=varnames["ψ"*string(i)*"_save"],
        ψ_true=XY_iir_true[i], ψ_MLE1=XY_iir_MLE[i], save_dir="./figures/")
end

# 2D Profiles
param_pairs = [(i,j) for i in 1:dim_all for j in (i+1):dim_all]

for (i,j) in param_pairs
    target_indices_ij = [i,j]
    nuisance_indices = setdiff(indices_all, target_indices_ij)
    nuisance_guess = XY_iir_MLE[nuisance_indices]
    ψ_true_pair = XY_iir_true[target_indices_ij]

    # Create a copy of varnames for this iteration
    current_varnames = deepcopy(varnames)
    current_varnames["ψ1"] = varnames["ψ"*string(i)]
    current_varnames["ψ2"] = varnames["ψ"*string(j)]
    current_varnames["ψ1_save"] = varnames["ψ"*string(i)*"_save"]
    current_varnames["ψ2_save"] = varnames["ψ"*string(j)*"_save"]

    # Profile full likelihood
    ψω_values, lnlike_ψ_values = profile_target(lnlike_XY_iir, target_indices_ij,
        XY_iir_lower_bounds, XY_iir_upper_bounds,
        nuisance_guess; grid_steps=grid_steps)

    # Profile quadratic approximation
    ψω_ellipse_values, lnlike_ψ_ellipse_values = profile_target(lnlike_XY_iir_ellipse,
        target_indices_ij,
        XY_iir_lower_bounds, XY_iir_upper_bounds,
        nuisance_guess; grid_steps=grid_steps)

    # Extract profiled parameter values
    ψ_values = [ψω[target_indices_ij] for ψω in ψω_values]
    ψ_ellipse_values = [ψω[target_indices_ij] for ψω in ψω_ellipse_values]

    # Plot contours
    plot_2D_contour(model_name, ψ_values, lnlike_ψ_values,
        current_varnames; ψ_true=ψ_true_pair, ψ_MLE=XY_iir_MLE, save_dir="./figures/")

    # Plot comparison with quadratic approximation
    plot_2D_contour_comparison(model_name, model_name*"_ellipse",
        ψ_values, ψ_ellipse_values,
        lnlike_ψ_values, lnlike_ψ_ellipse_values,
        current_varnames; ψ_true=ψ_true_pair, ψ_MLE1=XY_iir_MLE, save_dir="./figures/")

    # Get and plot 1D profiles from 2D grid
    ψ1_values, ψ2_values, like_ψ1_values, like_ψ2_values = get_1D_profiles_from_2D(
        ψ_values, lnlike_ψ_values)

    plot_1D_profile(model_name, ψ1_values, log.(like_ψ1_values),
        current_varnames["ψ1"];
        varname_save=current_varnames["ψ1_save"]*"_from_2D",
        ψ_true=ψ_true_pair[1], ψ_MLE=XY_iir_MLE[i], save_dir="./figures/")

    plot_1D_profile(model_name, ψ2_values, log.(like_ψ2_values),
        current_varnames["ψ2"];
        varname_save=current_varnames["ψ2_save"]*"_from_2D",
        ψ_true=ψ_true_pair[2], ψ_MLE=XY_iir_MLE[j], save_dir="./figures/")
end