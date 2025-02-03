# --------------------------------------------------------
# Set default plotting style options
# --------------------------------------------------------
default(
    xguidefontsize=14,
    yguidefontsize=14,
    xtickfontsize=12,
    ytickfontsize=12,
    xlabelfontsize=14,
    ylabelfontsize=14,
    legendfontsize=10,
    margin=2.5mm
)

# --------------------------------------------------------
# 1D Profile Visualization
# --------------------------------------------------------
function plot_1D_profile(model_name, ψ_values, lnlike_ψ_values, varname;
    varname_save="", ψ_true=[], ψ_MLE=[], save_dir="./figures/",
    fmt=:png, dpi=600, l_level=95)
    """
    Plot 1D profile likelihood.

    Parameters:
    - model_name: Name for saving plot
    - ψ_values: Interest parameter values
    - lnlike_ψ_values: Profile log-likelihood values
    - varname: Parameter name for plotting
    - varname_save: Parameter name for saving (default: varname)
    - ψ_true: True parameter value if known (optional)
    - ψ_MLE: Maximum likelihood estimate if known (optional). If not provided will use grid max.
    - save_dir: Directory for saving plot (default: "./")
    - fmt: File type for saving (default: :png)
    - l_level: Confidence level for threshold (default: 95)
    """
    if varname_save == ""
        varname_save = varname 
    end

    # Convert to likelihood scale
    like_ψ_values = exp.(lnlike_ψ_values)
    
    # Create plot
    plt = plot(ψ_values, exp.(lnlike_ψ_values), 
              xlabel=latexstring(varname), 
              ylabel="profile likelihood for "*latexstring(varname),
              color=:black, lw=2, legend=false, grid=false)
    
    # Add maximum likelihood line
    if length(ψ_MLE) > 0
        vline!([ψ_MLE], color=:silver, lw=3)
    else
        max_indices = argmax(like_ψ_values)
        ψ_max = ψ_values[max_indices]
        vline!([ψ_max], color=:silver, lw=3)
    end
    
    # Add confidence threshold
    hline!([exp(-quantile(Chisq(1),l_level/100)/2)], color=:black, lw=1)
    
    # Add true value if provided
    if length(ψ_true) > 0
        vline!([ψ_true], color=:darkgoldenrod, ls=:dash, lw=3)
    end
    
    display(plt)
    gr(fmt=fmt, dpi=dpi)
    savefig(plt, save_dir*model_name*"_"*varname_save*"_profile"*".$fmt")
end

function plot_1D_profile_comparison(model_name1, model_name2, 
              ψ_values1, ψ_values2,
              lnlike_ψ_values1, lnlike_ψ_values2, 
              varname;
              varname_save="", ψ_true=[], ψ_MLE1=[], ψ_MLE2=[], add_model2_MLE=false,
              save_dir="./figures/", fmt=:png, dpi=600,
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
    - ψ_MLE1, ψ_MLE2: Maximum likelihood estimates if known (optional). If not provided will use grid max.
    - save_dir: Directory for saving plot (default: "./")
    - fmt: File type for saving (default: :png)
    - l_level: Confidence level for threshold (default: 95)
    """
    if varname_save == ""
        varname_save = varname 
    end
    
    # Convert to likelihood scale. Note: returned normalised already
    like_ψ_values1 = exp.(lnlike_ψ_values1)
    like_ψ_values2 = exp.(lnlike_ψ_values2)
    
    plt = plot(ψ_values1, like_ψ_values1, 
              xlabel=latexstring(varname), 
              ylabel="profile likelihood for "*latexstring(varname),
              color=:black, lw=2, legend=false, grid=false)
    
    plot!(ψ_values2, like_ψ_values2, 
          ls=:dot, color=:black, lw=2, legend=false)

    if length(ψ_MLE1) > 0
        ψ_max1 = ψ_MLE1[1]
    else
        # Get location of max for model 1
        max_indices1 = argmax(like_ψ_values1)
        ψ_max1 = ψ_values1[max_indices1]
    end
    
    vline!([ψ_max1], color=:silver, lw=3)
    hline!([exp(-quantile(Chisq(1),l_level/100)/2)], color=:black, lw=1)
    
    if length(ψ_true) > 0
        vline!([ψ_true], color=:darkgoldenrod, ls=:dash, lw=3)
    end

    if add_model2_MLE
        if length(ψ_MLE2) > 0
            ψ_max2 = ψ_MLE2[1]
        else
            # Get location of max for model 2
            max_indices2 = argmax(like_ψ_values2)
            ψ_max2 = ψ_values2[max_indices2]
        end
        vline!([ψ_max2], color=:silver, lw=3, ls=:dot)
    end
    
    display(plt)
    gr(fmt=fmt, dpi=dpi)
    savefig(plt, save_dir*model_name1*"_"*model_name2*"_"*varname_save*"_profiles"*".$fmt")
end

# --------------------------------------------------------
# 2D Profile Visualization
# --------------------------------------------------------
function plot_2D_contour(model_name, ψ_values, lnlike_ψ_values, varnames;
    ψ_true=[], ψ_MLE=[], save_dir="./figures/", fmt=:png, dpi=600,
    l_level=95, nshade_levels=20)
    """
    Plot 2D profile likelihood contours.

    Parameters:
    - model_name: Name for saving plot
    - ψ_values: 2D interest parameter vector
    - lnlike_ψ_values: 2D Profile log-likelihood values
    - varnames: Dictionary of parameter names for plotting/saving
    - ψ_true: True parameter values if known (optional)
    - ψ_MLE: Maximum likelihood estimate if known (optional). If not provided will use grid max.
    - save_dir: Directory for saving plot (default: "./")
    - fmt: File type for saving (default: :png)
    - l_level: Confidence level for contours (default: 95)
    - nshade_levels: Number of shading levels (default: 20)
    - dpi: Resolution for saving (default: 300)
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
    else
        max_indices = argmax(like_ψ_values)
        ψ1_max = ψ1_values[max_indices[1]]
        ψ2_max = ψ2_values[max_indices[2]]
        scatter!([ψ1_max], [ψ2_max], mc=:silver, msc=:match, markersize=8, markershape=:circle, legend=false)
    end
    
    if length(ψ_true) > 0
        scatter!([ψ_true[1]], [ψ_true[2]], mc=:darkgoldenrod, msc=:match, markersize=10, markershape=:star, legend=false)
    end
    
    display(plt)
    gr(fmt=fmt, dpi=dpi)
    savefig(plt, save_dir*model_name*"_"*varnames["ψ1_save"]*"_"*varnames["ψ2_save"]*".$fmt")

end

function plot_2D_contour_comparison(model_name1, model_name2, 
              ψ_values1, ψ_values2,
              lnlike_ψ_values1, lnlike_ψ_values2, 
              varnames;
              ψ_true=[], ψ_MLE1=[], ψ_MLE2=[], save_dir="./figures/", fmt=:png, dpi=600,
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
    - ψ_MLE1, ψ_MLE2: Maximum likelihood estimates if known (optional). If not provided will use grid max.
    - save_dir: Directory for saving plot (default: "./")
    - fmt: File type for saving (default: :png)
    - nshade_levels: Number of shading levels (default: 20)
    - l_level: Confidence level for contours (default: 95)
    - dpi: Resolution for saving (default: 300)
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
    xlabel!(latexstring(varnames["ψ1"]))
    ylabel!(latexstring(varnames["ψ2"]))

    # Add model 1 best 
    if length(ψ_MLE1) > 0
        scatter!([ψ_MLE1[1]], [ψ_MLE1[2]], mc=:silver, msc=:match, markersize=8, markershape=:circle, legend=false)
    else
        max_indices1 = argmax(like_ψ_values1)
        ψ1_values1_max = ψ1_values1[max_indices1[1]]
        ψ2_values1_max = ψ2_values1[max_indices1[2]]
        scatter!([ψ1_values1_max], [ψ2_values1_max], mc=:silver, msc=:match, markersize=8, markershape=:circle, legend=false)
    end
        
    # Add model 2 best 
    if add_model2_MLE
        if length(ψ_MLE2) > 0
            scatter!([ψ_MLE2[1]], [ψ_MLE2[2]], mc=:silver, msc=:match, markersize=8, markershape=:circle, legend=false)
        else
            max_indices2 = argmax(like_ψ_values2)
            ψ1_values2_max = ψ1_values2[max_indices2[1]]
            ψ2_values2_max = ψ2_values2[max_indices2[2]]
            scatter!([ψ1_values2_max], [ψ2_values2_max], mc=:silver, msc=:match, markersize=8, markershape=:circle, legend=false)
        end
    end

    # Add true
    if length(ψ_true) > 0
        scatter!([ψ_true[1]], [ψ_true[2]], mc=:darkgoldenrod, msc=:match, markersize=10, markershape=:star, legend=false)
    end
    
    display(plt)
    gr(fmt=fmt, dpi=dpi)
    savefig(plt, save_dir*model_name1*"_"*model_name2*"_comparison_"*
            varnames["ψ1_save"]*"_"*varnames["ψ2_save"]*".$fmt")
end

# --------------------------------------------------------
# Prediction Confidence Interval Visualization
# --------------------------------------------------------
function plot_profile_wise_CI_for_mean(indep_var, lower, upper, mle, 
                 model_name, dep_varname,
                 indep_varname, indep_varname_save;
                 data_indep=nothing, data_dep=nothing, true_mean=nothing, 
                 target="", target_save="", save_dir="./figures/", fmt=:png, dpi=600,
                 verbose_labels=false, include_legend=false)
    """
    Plot confidence intervals for mean function based on profile likelihood.

    Parameters:
    - indep_var: Independent variable values. Typically fine grid for function.
    - lower, upper: Lower and upper bounds of confidence interval
    - mle: Maximum likelihood estimate of mean
    - model_name: Name for saving plot
    - dep_varname: Name of dependent variable for plotting
    - indep_varname: Name of independent variable for plotting
    - indep_varname_save: Name of independent variable for saving
    - data_indep: Independent variable data points to overlay (optional)
    - data_dep: Dependent variable data points to overlay (optional)
    - true_mean: True mean function if known (optional)
    - target: Target parameter name for plotting (optional)
    - target_save: Target parameter name for saving (optional)
    - save_dir: Directory for saving plot (default: "./")
    - fmt: File type for saving (default: :png)
    - verbose_labels: Whether to add additional information to legend (default: false)
    """
    # Determine labels based on include_legend and verbose_labels
    ci_label    = include_legend ? latexstring("CI ("*target*")") : ""
    mle_label   = include_legend ? (verbose_labels ? "MLE"   : "") : ""
    data_label  = include_legend ? (verbose_labels ? "Data"  : "") : ""
    truth_label = include_legend ? (verbose_labels ? "Truth" : "") : ""

    plt = plot(indep_var, lower, lw=0,
              fillrange=upper, fillalpha=0.20, color=:purple,
              xlabel=latexstring(indep_varname),
              ylabel=latexstring(dep_varname),
              xlims=(indep_var[1], indep_var[end]),
              label=ci_label, 
              legend=:topright, grid=false)

    
    plot!(indep_var, mle, lw=2, linecolor=:purple4, label=mle_label)
    
    # if have both independent and dependent data
    if !isnothing(data_indep) && !isnothing(data_dep)
        
        scatter!(data_indep, data_dep, 
                    mc=:black, msc=:match, markersize=3, 
                    markershape=:x, label=data_label)
    end
    
    if !isnothing(true_mean)
        plot!(indep_var, true_mean, lw=2, linecolor=:darkgoldenrod, linestyle=:dash, label=truth_label)
    end
    
    display(plt)
    gr(fmt=fmt, dpi=dpi)
    savefig(plt, save_dir*model_name*"_mean_vs_"*indep_varname_save*"_"*target_save*"_profile"*".$fmt")

end