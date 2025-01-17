# --------------------------------------------------------
# 1D Profile Visualization
# --------------------------------------------------------
function plot_1D_profile(model_name, ψ_values, lnlike_ψ_values, varname;
    varname_save="", ψ_true=[], save_dir="./figures/",
    file_extension=".svg", l_level=95)
    """
    Plot 1D profile likelihood.

    Parameters:
    - model_name: Name for saving plot
    - ψ_values: Interest parameter values
    - lnlike_ψ_values: Profile log-likelihood values
    - varname: Parameter name for plotting
    - varname_save: Parameter name for saving (default: varname)
    - ψ_true: True parameter value if known (optional)
    - save_dir: Directory for saving plot (default: "./")
    - file_extension: File type for saving (default: ".svg")
    - l_level: Confidence level for threshold (default: 95)
    """
    if varname_save == ""
        varname_save = varname 
    end

    # Convert to likelihood scale
    like_ψ_values = exp.(lnlike_ψ_values)

    # Get location of max
    max_indices = argmax(like_ψ_values)
    ψ_max = ψ_values[max_indices]
    
    # Create plot
    plt = plot(ψ_values, exp.(lnlike_ψ_values), 
              xlabel=latexstring(varname), 
              ylabel="profile likelihood for "*latexstring(varname),
              xlabelfontsize=14, ylabelfontsize=14,
              color=:black, lw=2, legend=false)
    
    # Add maximum likelihood line
    vline!([ψ_max], color=:black, lw=2)
    
    # Add confidence threshold
    hline!([exp(-quantile(Chisq(1),l_level/100)/2)], color=:black, lw=1)
    
    # Add true value if provided
    if length(ψ_true) > 0
        vline!([ψ_true], color=:black, ls=:dash, lw=2)
    end
    
    display(plt)
    savefig(plt, save_dir*model_name*"_"*varname_save*"_profile"*file_extension)
end

function plot_1D_profile_comparison(model_name1, model_name2, 
              ψ_values1, ψ_values2,
              lnlike_ψ_values1, lnlike_ψ_values2, 
              varname;
              varname_save="", ψ_true=[], 
              save_dir="./figures/", file_extension=".svg",
              l_level=95)
    """
    Plot comparison of 1D profile likelihoods from two models/approximations.
    Typically used to compare ellipse approximation to full profile likelihood.

    Parameters:
    - model_name1: Name of first model
    - model_name2: Name of second model
    - ψ_values1, ψ_values2: Interest parameter values for each model
    - lnlike_ψ_values1, lnlike_ψ_values2: Profile log-likelihood values for each model
    - varname: Parameter name for plotting
    - varname_save: Parameter name for saving (default: varname)
    - ψ_true: True parameter value if known (optional)
    - save_dir: Directory for saving plot (default: "./")
    - file_extension: File type for saving (default: ".svg")
    - l_level: Confidence level for threshold (default: 95)
    """
    if varname_save == ""
        varname_save = varname 
    end
    
    # Convert to likelihood scale. Note: returned normalised already
    like_ψ_values1 = exp.(lnlike_ψ_values1)
    like_ψ_values2 = exp.(lnlike_ψ_values2)
    
    # Get location of max for model 1
    max_indices1 = argmax(like_ψ_values1)
    ψ_max1 = ψ_values1[max_indices1]
    
    plt = plot(ψ_values1, like_ψ_values1, 
              xlabel=latexstring(varname), 
              ylabel="profile likelihood for "*latexstring(varname),
              xlabelfontsize=14, ylabelfontsize=14,
              color=:black, lw=2, legend=false)
    
    plot!(ψ_values2, like_ψ_values2, 
          ls=:dot, color=:black, lw=2, legend=false)
    
    vline!([ψ_max1], color=:black, lw=2)
    hline!([exp(-quantile(Chisq(1),l_level/100)/2)], color=:black, lw=1)
    
    if length(ψ_true) > 0
        vline!([ψ_true], color=:black, ls=:dash, lw=2)
    end
    
    display(plt)
    savefig(plt, save_dir*model_name1*"_"*model_name2*"_"*varname_save*"_profiles"*file_extension)
end

# --------------------------------------------------------
# 2D Profile Visualization
# --------------------------------------------------------
function plot_2D_contour(model_name, ψ_values, lnlike_ψ_values, varnames;
    ψ_true=[], save_dir="./figures/", file_extension=".svg",
    l_level=95, nshade_levels=20)
    """
    Plot 2D profile likelihood contours.

    Parameters:
    - model_name: Name for saving plot
    - ψ_values: 2D interest parameter vector
    - lnlike_ψ_values: 2D Profile log-likelihood values
    - varnames: Dictionary of parameter names for plotting/saving
    - ψ_true: True parameter values if known (optional)
    - save_dir: Directory for saving plot (default: "./")
    - file_extension: File type for saving (default: ".svg")
    - l_level: Confidence level for contours (default: 95)
    - nshade_levels: Number of shading levels (default: 20)
    """
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
    
    contour!(ψ1_values, ψ2_values, like_ψ_values',
            levels=[lstar], color=:black, colorbar=false, lw=1)
    
    xlabel!(latexstring(varnames["ψ1"]), xlabelfontsize=14)
    ylabel!(latexstring(varnames["ψ2"]), ylabelfontsize=14)
    xlims!(minimum(ψ1_values), maximum(ψ1_values))
    ylims!(minimum(ψ2_values), maximum(ψ2_values))
    
    # Add best and true
    max_indices = argmax(like_ψ_values)
    ψ1_max = ψ1_values[max_indices[1]]
    ψ2_max = ψ2_values[max_indices[2]]
    # vline!([ψ1_max], color=:black, lw=2, legend=false)
    # hline!([ψ2_max], color=:black, lw=2, legend=false)
    # marker
    scatter!([ψ1_max], [ψ2_max], mc=:silver, msc=:match, markersize=6, markershape=:circle, legend=false)
    
    if length(ψ_true) > 0
        #vline!([ψ_true[1]], color=:black, ls=:dash, lw=2, legend=false)
        #hline!([ψ_true[2]], color=:black, ls=:dash, lw=2, legend=false)
        # add marker
        scatter!([ψ_true[1]], [ψ_true[2]], mc=:gold, msc=:match, markersize=8, markershape=:star, legend=false)
    end
    
    display(plt)
    savefig(plt, save_dir*model_name*"_"*varnames["ψ1_save"]*"_"*varnames["ψ2_save"]*file_extension)

end

function plot_2D_contour_comparison(model_name1, model_name2, 
              ψ_values1, ψ_values2,
              lnlike_ψ_values1, lnlike_ψ_values2, 
              varnames;
              ψ_true=[], save_dir="./figures/", file_extension=".svg",
              nshade_levels=20, l_level=95, 
              add_model2_MLE=false)
    """
    Plot comparison of 2D profile likelihood contours from two models/approximations.
    Typically used to compare ellipse approximation to full profile likelihood.

    Parameters:
    - model_name1: Name of first model
    - model_name2: Name of second model
    - ψ_values1, ψ_values2: Array of 2D interest parameter vectors for each model
    - lnlike_ψ_values1, lnlike_ψ_values2: Profile log-likelihood values for each model
    - varnames: Dictionary of parameter names for plotting/saving
    - ψ_true: True parameter values if known (optional)
    - save_dir: Directory for saving plot (default: "./")
    - file_extension: File type for saving (default: ".svg")
    - nshade_levels: Number of shading levels (default: 20)
    - l_level: Confidence level for contours (default: 95)
    - add_model2_MLE: Whether to show MLE for second model (default: false)
    """
    # Contour plots with chi square calibration
    df = 2
    lstar = exp(-quantile(Chisq(df), l_level/100)/2)

    # Model 1
    ψ1_values1 = unique([ψ1 for (ψ1, _) in ψ_values1])
    ψ2_values1 = unique([ψ2 for (_, ψ2) in ψ_values1])
    lnlike_ψ_values1 = reshape(lnlike_ψ_values1, length(ψ1_values1), length(ψ2_values1))
    like_ψ_values1 = exp.(lnlike_ψ_values1)

    plt = contourf(ψ1_values1, ψ2_values1, like_ψ_values1',
                  color=:dense, levels=nshade_levels, lw=0)
    
    contour!(ψ1_values1, ψ2_values1, like_ψ_values1',
            levels=[lstar], color=:black, colorbar=false, lw=1)

    # Model 2
    ψ1_values2 = unique([ψ1 for (ψ1, _) in ψ_values2])
    ψ2_values2 = unique([ψ2 for (_, ψ2) in ψ_values2])
    lnlike_ψ_values2 = reshape(lnlike_ψ_values2, length(ψ1_values2), length(ψ2_values2))
    like_ψ_values2 = exp.(lnlike_ψ_values2)

    contour!(ψ1_values2, ψ2_values2, like_ψ_values2',
            levels=[lstar], color=:black, colorbar=false, lw=1.5, ls=:dash)

    # Labels etc
    xlabel!(latexstring(varnames["ψ1"]), xlabelfontsize=14)
    ylabel!(latexstring(varnames["ψ2"]), ylabelfontsize=14)

    # Add model 1 best 
    max_indices1 = argmax(lnlike_ψ_values1)
    ψ1_values1_max = ψ1_values1[max_indices1[1]]
    ψ2_values1_max = ψ2_values1[max_indices1[2]]
    # vline!([ψ1_values1_max], color=:black, lw=2, legend=false)
    # hline!([ψ2_values1_max], color=:black, lw=2, legend=false)
    # marker
    scatter!([ψ1_values1_max], [ψ2_values1_max], mc=:silver, msc=:match, markersize=6, markershape=:circle, legend=false)
    
    # Add model 2 best 
    if add_model2_MLE
        max_indices2 = argmax(lnlike_ψ_values2)
        ψ1_values2_max = ψ1_values2[max_indices2[1]]
        ψ2_values2_max = ψ2_values2[max_indices2[2]]
        # vline!([ψ1_values2_max], color=:black, lw=1, legend=false)
        # hline!([ψ2_values2_max], color=:black, lw=1, legend=false)
        # marker
        scatter!([ψ1_values2_max], [ψ2_values2_max], mc=:silver, msc=:match, markersize=6, markershape=:+, legend=false)
    end

    # Add true
    if length(ψ_true) > 0
        # vline!([ψ_true[1]], color=:black, ls=:dash, lw=2, legend=false)
        # hline!([ψ_true[2]], color=:black, ls=:dash, lw=2, legend=false)
        # marker
        scatter!([ψ_true[1]], [ψ_true[2]], mc=:gold, msc=:match, markersize=8, markershape=:star, legend=false)
    end
    
    display(plt)
    savefig(plt, save_dir*model_name1*"_"*model_name2*"_comparison_"*
            varnames["ψ1_save"]*"_"*varnames["ψ2_save"]*file_extension)
end

# --------------------------------------------------------
# Prediction Confidence Interval Visualization
# --------------------------------------------------------
function plot_profile_wise_CI_for_mean(indep_var, lower, upper, mle, 
                 model_name, indep_varname, indep_varname_save;
                 data_indep=nothing, data_dep=nothing, true_mean=nothing, 
                 target="", save_dir="./figures/", file_extension=".svg", 
                 verbose_labels=false)
    """
    Plot confidence intervals for mean function based on profile likelihood.

    Parameters:
    - indep_var: Independent variable values. Typically fine grid for function.
    - lower, upper: Lower and upper bounds of confidence interval
    - mle: Maximum likelihood estimate of mean
    - model_name: Name for saving plot
    - indep_varname: Name of independent variable for plotting
    - indep_varname_save: Name of independent variable for saving
    - data_indep: Independent variable data points to overlay (optional)
    - data_dep: Dependent variable data points to overlay (optional)
    - true_mean: True mean function if known (optional)
    - target: Description of target parameter (optional)
    - save_dir: Directory for saving plot (default: "./")
    - file_extension: File type for saving (default: ".svg")
    - verbose_labels: Whether to add additional information to legend (default: false)
    """
    plt = plot(indep_var, lower, lw=0, primary=false,
              fillrange=upper, fillalpha=0.20, color=:purple,
              xlabel=latexstring(indep_varname),
              ylabel="profile likelihood CIs for mean",
              xlims=(indep_var[1], indep_var[end]),
              label=latexstring("CI ("*target*")"))
    
    if verbose_labels
        plot!(indep_var, mle, linecolor=:purple4, label="MLE")
    else
        plot!(indep_var, mle, linecolor=:purple4, label="")
    end
    
    # if have both independent and dependent data
    if !isnothing(data_indep) && !isnothing(data_dep)
        if verbose_labels
            scatter!(data_indep, data_dep, 
                    mc=:black, msc=:match, markersize=3, 
                    markershape=:x, label="Data")
        else
            scatter!(data_indep, data_dep, 
                    mc=:black, msc=:match, markersize=3, 
                    markershape=:x, label="")
        end
    end
    
    if !isnothing(true_mean)
        if verbose_labels
            plot!(indep_var, true_mean, linecolor=:black, linestyle=:dash, label="Truth")
        else
            plot!(indep_var, true_mean, linecolor=:black, linestyle=:dash, label="")
        end
    end
    
    display(plt)
    savefig(plt, save_dir*model_name*"_mean_vs_"*indep_varname_save*"_"*target*"_profile"*file_extension)

end