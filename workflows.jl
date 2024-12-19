# --------------------------------------------------------
# Key Analysis Steps
# --------------------------------------------------------
function compute_mle_and_quadratic_approx(lnlike_θ, θ_bounds_lower, θ_bounds_upper, θ_initial;
    grid_steps=100)
    """
    Compute MLE and quadratic approximation to likelihood.

    Returns:
    - θ_MLE: Maximum likelihood parameter estimates
    - evals: Eigenvalues of observed Fisher Information
    - evecs: Eigenvectors of observed Fisher Information
    - lnlike_θ_ellipse: Quadratic approximation to likelihood
    """
    # Point estimation
    target_indices = [] # empty for MLE
    nuisance_guess = θ_initial
    θ_MLE, lnlike_θ_MLE = profile_target(lnlike_θ, target_indices,
                θ_bounds_lower, θ_bounds_upper, 
                nuisance_guess; grid_steps=grid_steps)

    # Quadratic approximation at MLE
    lnlike_θ_ellipse, H_θ_ellipse = construct_ellipse_lnlike_approx(lnlike_θ, θ_MLE)

    # Eigenanalysis of observed Fisher Information
    evals, evecs = eigen(H_θ_ellipse; sortby = x -> -real(x))

    return θ_MLE, evals, evecs, lnlike_θ_ellipse
end

function compute_1D_profiles(model_name, varnames, lnlike_θ, lnlike_θ_ellipse, θ_bounds_lower, θ_bounds_upper, θ_initial;
    grid_steps=100, ψ_true=[], store_profiles=false)
    """
    Compute individual parameter profiles and MLE.

    Returns:
    - None
    
    """
    # --- Setup dimensions, indices and profile storage (if required) ---
    dim_all = length(θ_initial)
    indices_all = 1:dim_all
    profiles = store_profiles ? Dict() : nothing

    # --- Individual parameter profiles ---
    for i in 1:dim_all
        target_index = i
        nuisance_indices = setdiff(indices_all, target_index)
        nuisance_guess = θ_initial[nuisance_indices]

        # Profile full likelihood
        ψω_values, lnlike_ψ_values = profile_target(lnlike_θ, target_index,
            θ_bounds_lower, θ_bounds_upper,
            nuisance_guess; grid_steps=grid_steps)

        # Profile quadratic approximation
        ψω_ellipse_values, lnlike_ψ_ellipse_values = profile_target(lnlike_θ_ellipse,
            target_index,
            θ_bounds_lower,
            θ_bounds_upper,
            nuisance_guess;
            grid_steps=grid_steps)

        # Extract profiled parameter values
        ψ_values = [ψω[target_index] for ψω in ψω_values]
        ψ_ellipse_values = [ψω[target_index] for ψω in ψω_ellipse_values]

        # Plot profiles
        plot_1D_profile(model_name, ψ_values, lnlike_ψ_values,
            varnames["ψ"*string(i)];
            varname_save=varnames["ψ"*string(i)*"_save"],
            ψ_true=ψ_true[i])

        # Plot comparison with quadratic approximation
        plot_1D_profile_comparison(model_name, model_name*"_ellipse",
            ψ_values, ψ_ellipse_values,
            lnlike_ψ_values, lnlike_ψ_ellipse_values,
            varnames["ψ"*string(i)];
            varname_save=varnames["ψ"*string(i)*"_save"],
            ψ_true=ψ_true[i])

        # Store if requested
        if store_profiles
            profiles[varnames["ψ"*string(i)]] = (ψω_values, lnlike_ψ_values)
        end
    end

    return profiles
end

function compute_2D_profiles(model_name, varnames, lnlike_θ, lnlike_θ_ellipse,
    θ_bounds_lower, θ_bounds_upper, θ_initial;
    grid_steps=100, ψ_true=[], store_profiles=false)
    """
    Compute pairwise parameter profiles.
    """
    dim_all = length(θ_initial)
    indices_all = 1:dim_all

    profiles = store_profiles ? Dict() : nothing

    # For all parameter pairs
    param_pairs = [(i,j) for i in 1:dim_all for j in (i+1):dim_all]

    for (i,j) in param_pairs
        target_indices = [i,j]
        nuisance_indices = setdiff(indices_all, target_indices)
        nuisance_guess = θ_initial[nuisance_indices]
        ψ_true_pair = ψ_true[target_indices]

        # Make local copy of varnames for this pair
        current_varnames = copy(varnames)
        current_varnames["ψ1"] = varnames["ψ"*string(i)]
        current_varnames["ψ2"] = varnames["ψ"*string(j)]
        current_varnames["ψ1_save"] = varnames["ψ"*string(i)*"_save"]
        current_varnames["ψ2_save"] = varnames["ψ"*string(j)*"_save"]

        # Profile full likelihood
        ψω_values, lnlike_ψ_values = profile_target(lnlike_θ, target_indices,
            θ_bounds_lower, θ_bounds_upper,
            nuisance_guess; grid_steps=grid_steps)

        # Profile quadratic approximation
        ψω_ellipse_values, lnlike_ψ_ellipse_values = profile_target(lnlike_θ_ellipse,
            target_indices,
            θ_bounds_lower,
            θ_bounds_upper,
            nuisance_guess;
            grid_steps=grid_steps)

        # Extract profiled parameter values
        ψ_values = [ψω[target_indices] for ψω in ψω_values]
        ψ_ellipse_values = [ψω[target_indices] for ψω in ψω_ellipse_values]

        # Plot contours
        plot_2D_contour(model_name, ψ_values, lnlike_ψ_values,
            current_varnames; ψ_true=ψ_true_pair)

        # Plot comparison
        plot_2D_contour_comparison(model_name, model_name*"_ellipse",
            ψ_values, ψ_ellipse_values,
            lnlike_ψ_values, lnlike_ψ_ellipse_values,
            current_varnames; ψ_true=ψ_true_pair)

        # Store if requested
        if store_profiles
            profiles[varnames["ψ"*string(i)]*"_"*varnames["ψ"*string(j)]] = (ψω_values, lnlike_ψ_values)
        end

        # Get and plot 1D profiles from 2D grid
        ψ1_values, ψ2_values, like_ψ1_values, like_ψ2_values = get_1D_profiles_from_2D(ψ_values, lnlike_ψ_values)

        plot_1D_profile(model_name, ψ1_values, log.(like_ψ1_values),
            current_varnames["ψ1"];
            varname_save=current_varnames["ψ1_save"],
            ψ_true=ψ_true_pair[1])

        plot_1D_profile(model_name, ψ2_values, log.(like_ψ2_values),
            current_varnames["ψ2"];
            varname_save=current_varnames["ψ2_save"],
            ψ_true=ψ_true_pair[2])

        # Store if requested
        if store_profiles
            profiles[varnames["ψ"*string(i)]*"_grid"] = (ψ1_values, like_ψ1_values)
            profiles[varnames["ψ"*string(j)]*"_grid"] = (ψ2_values, like_ψ2_values)
        end
    end
    return profiles
end

function execute_model_analysis_workflow_up_to_2D_profiles(model_name, varnames, 
    lnlike_θ, θ_bounds_lower, θ_bounds_upper, θ_initial;
    grid_steps=100, ψ_true=[], return_info=false)
    """
    Execute complete analysis workflow combining 1D and 2D profiling.

    Performs standard likelihood-based analysis pipeline:
    1. Maximum likelihood estimation
    2. Quadratic approximation and eigenanalysis at MLE
    3. Individual parameter profile likelihoods
    4. Pairwise profile likelihoods

    Creates standard visualizations:
    1. 1D profile plots for each parameter
    2. 2D contour plots for parameter pairs
    3. Profile-wise confidence intervals

    Parameters:
    - model_name: Name used in plot saving
    - varnames: Dictionary of parameter names for plotting
    - lnlike_θ: Log-likelihood function
    - θ_bounds_lower, θ_bounds_upper: Parameter bounds
    - θ_initial: Initial parameter guess
    - grid_steps: Number of grid points for profiling
    - ψ_true: True parameter values if known
    - return_info: Whether to return MLE and eigenanalysis results

    Returns:
    - If return_info=true: (θ_MLE, eigenvalues, eigenvectors)
    - If return_info=false: nothing
    """    

    # Get MLE and quadratic approximation first
    θ_MLE, evals, evecs, lnlike_θ_ellipse = compute_mle_and_quadratic_approx(lnlike_θ, θ_bounds_lower, θ_bounds_upper, θ_initial;
        grid_steps=grid_steps)
    
    # Print eigenanalysis results
    println("Eigenvectors and eigenvalues for "*model_name)
    for (i, eveci) in enumerate(eachcol(evecs))
        evecs[:,i] = eveci
        println("value:")
        println(evals[i])
        println("vector:")
        println(evecs[:,i])
    end

    # Run 1D analysis
    compute_1D_profiles(model_name, varnames, lnlike_θ, lnlike_θ_ellipse, θ_bounds_lower, θ_bounds_upper, θ_initial;
        grid_steps=grid_steps, ψ_true=ψ_true)

    # Run 2D analysis using quadratic approximation from 1D analysis
    compute_2D_profiles(
        model_name, varnames, lnlike_θ, lnlike_θ_ellipse,
        θ_bounds_lower, θ_bounds_upper, θ_initial;
        grid_steps=grid_steps, ψ_true=ψ_true)

    # Prediction TODO.

    if return_info
        return θ_MLE, evals, evecs
    end
end