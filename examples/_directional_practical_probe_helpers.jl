# Example-local helpers for directional practical-identifiability diagnostics.
# These support the maintained simple examples without expanding the public
# ReparamTools API surface.

function directional_practical_probe(ϕ_func, θ0, probe_direction,
    lower_bounds, upper_bounds; deltas=[0.05, 0.1, 0.2, 0.4, 0.8])

    v_probe = probe_direction / norm(probe_direction)
    J0 = compute_ϕ_Jacobian(ϕ_func, θ0)
    σ1_0 = svdvals(J0)[1]
    Jv0 = J0 * v_probe
    ε0 = norm(Jv0) / σ1_0

    rows = NamedTuple[]
    for δ in deltas
        θ_plus = θ0 .+ δ .* v_probe
        θ_minus = θ0 .- δ .* v_probe

        plus_in_bounds = all(θ_plus .>= lower_bounds) && all(θ_plus .<= upper_bounds)
        minus_in_bounds = all(θ_minus .>= lower_bounds) && all(θ_minus .<= upper_bounds)

        if !(plus_in_bounds && minus_in_bounds)
            push!(rows, (δ=δ, valid=false, plus_in_bounds=plus_in_bounds,
                minus_in_bounds=minus_in_bounds))
            continue
        end

        J_plus = compute_ϕ_Jacobian(ϕ_func, θ_plus)
        J_minus = compute_ϕ_Jacobian(ϕ_func, θ_minus)
        σ1_plus = svdvals(J_plus)[1]
        σ1_minus = svdvals(J_minus)[1]
        Jv_plus = J_plus * v_probe
        Jv_minus = J_minus * v_probe

        ε_plus = norm(Jv_plus) / σ1_plus
        ε_minus = norm(Jv_minus) / σ1_minus
        d_plus = norm(Jv_plus - Jv0) / max(norm(Jv0), eps())
        d_minus = norm(Jv_minus - Jv0) / max(norm(Jv0), eps())

        push!(rows, (
            δ=δ,
            valid=true,
            θ_plus=θ_plus,
            θ_minus=θ_minus,
            ε_plus=ε_plus,
            ε_minus=ε_minus,
            asymmetry=ε_minus / max(ε_plus, eps()),
            d_plus=d_plus,
            d_minus=d_minus,
        ))
    end

    return (direction=v_probe, baseline_ε=ε0, rows=rows)
end

function print_directional_practical_probe(label, directional_probe, XYtoxy_func, θ0;
    coord_row=nothing, limit_name="Poisson limit")

    xy0 = XYtoxy_func(θ0)
    n0, p0 = xy0
    np0 = n0 * p0
    n_over_p0 = n0 / p0
    p_plus_small = XYtoxy_func(θ0 .+ 0.1 .* directional_probe.direction)[2]

    println("\n  ", label, ":")
    if coord_row !== nothing
        println("    Weak coordinate row in x = log(θ): ", round.(coord_row, digits=4))
    end
    println("    Corresponding perturbation in x = log(θ): ", round.(directional_probe.direction, digits=4))
    println("    (holding the other transformed coordinates fixed)")
    println("    Reference point: n = ", round(n0, digits=4),
        ", p = ", round(p0, digits=4),
        ", np = ", round(np0, digits=4),
        ", n/p = ", round(n_over_p0, digits=4))
    println("    Baseline relative first-order effect ε(0) = ||J(θ₀)v|| / σ₁(θ₀): ", round(directional_probe.baseline_ε, digits=4))
    if p_plus_small < p0
        println("    +δ decreases p (toward the ", limit_name, ")")
    else
        println("    +δ increases p (away from the ", limit_name, ")")
    end
    println("    ε±(δ) = ||J(θ₀ ± δv)v|| / σ₁(θ₀ ± δv)")
    println("    d±(δ) = ||J(θ₀ ± δv)v - J(θ₀)v|| / ||J(θ₀)v||")

    println("\n    Parameter movement:")
    println("    δ      n(+)      n(-)      p(+)      p(-)")
    println("    " * "-"^48)
    for row in directional_probe.rows
        if row.valid
            xy_plus = XYtoxy_func(row.θ_plus)
            xy_minus = XYtoxy_func(row.θ_minus)
            n_plus, p_plus = xy_plus
            n_minus, p_minus = xy_minus
            println("    ",
                lpad(string(round(row.δ, digits=2)), 4), "  ",
                lpad(string(round(n_plus, digits=4)), 9), "  ",
                lpad(string(round(n_minus, digits=4)), 9), "  ",
                lpad(string(round(p_plus, digits=4)), 8), "  ",
                lpad(string(round(p_minus, digits=4)), 8))
        else
            println("    δ = ", row.δ, ": step leaves bounds; skipped")
        end
    end

    println("\n    Derived quantities and weakness:")
    println("    δ     np(+)    np(-)    n/p(+)   n/p(-)    ε(+)     ε(-)   ε(-)/ε(+)    d(+)     d(-)")
    println("    " * "-"^104)
    for row in directional_probe.rows
        if row.valid
            xy_plus = XYtoxy_func(row.θ_plus)
            xy_minus = XYtoxy_func(row.θ_minus)
            np_plus = xy_plus[1] * xy_plus[2]
            np_minus = xy_minus[1] * xy_minus[2]
            n_over_p_plus = xy_plus[1] / xy_plus[2]
            n_over_p_minus = xy_minus[1] / xy_minus[2]
            println("    ",
                lpad(string(round(row.δ, digits=2)), 4), "  ",
                lpad(string(round(np_plus, digits=4)), 8), "  ",
                lpad(string(round(np_minus, digits=4)), 8), "  ",
                lpad(string(round(n_over_p_plus, digits=2)), 8), "  ",
                lpad(string(round(n_over_p_minus, digits=2)), 8), "  ",
                lpad(string(round(row.ε_plus, digits=4)), 8), "  ",
                lpad(string(round(row.ε_minus, digits=4)), 8), "  ",
                lpad(string(round(row.asymmetry, digits=3)), 11), "  ",
                lpad(string(round(row.d_plus, digits=4)), 8), "  ",
                lpad(string(round(row.d_minus, digits=4)), 8))
        else
            println("    δ = ", row.δ, ": step leaves bounds; skipped")
        end
    end
end

function print_mm_directional_practical_probe(label, directional_probe, XYtoxy_func, θ0;
    coord_row=nothing)

    xy0 = XYtoxy_func(θ0)
    ν0, K0 = xy0
    K_over_ν0 = K0 / ν0
    νK0 = ν0 * K0
    νK_plus_small = prod(XYtoxy_func(θ0 .+ 0.1 .* directional_probe.direction))

    println("\n  ", label, ":")
    if coord_row !== nothing
        println("    Weak coordinate row in x = log(θ): ", round.(coord_row, digits=4))
    end
    println("    Corresponding perturbation in x = log(θ): ", round.(directional_probe.direction, digits=4))
    println("    (holding the other transformed coordinates fixed)")
    println("    Reference point: ν = ", round(ν0, digits=4),
        ", K = ", round(K0, digits=4),
        ", K/ν = ", round(K_over_ν0, digits=4),
        ", νK = ", round(νK0, digits=4))
    println("    Baseline relative first-order effect ε(0) = ||J(θ₀)v|| / σ₁(θ₀): ", round(directional_probe.baseline_ε, digits=4))
    if νK_plus_small > νK0
        println("    +δ increases νK")
    else
        println("    +δ decreases νK")
    end
    println("    ε±(δ) = ||J(θ₀ ± δv)v|| / σ₁(θ₀ ± δv)")
    println("      → smaller ε means weaker local sensitivity in this direction")
    println("      → larger ε means the direction is less weak / more identifiable")
    if νK_plus_small > νK0
        println("      → ε(-)/ε(+) > 1 means the +δ side (larger νK) is weaker than the -δ side")
    else
        println("      → ε(-)/ε(+) > 1 means the +δ side (smaller νK) is weaker than the -δ side")
    end
    println("    d±(δ) = ||J(θ₀ ± δv)v - J(θ₀)v|| / ||J(θ₀)v||")
    println("      → d tracks how much the local weak-direction picture changes away from θ₀")

    println("\n    Parameter movement:")
    println("    δ      ν(+)      ν(-)      K(+)      K(-)")
    println("    " * "-"^48)
    for row in directional_probe.rows
        if row.valid
            xy_plus = XYtoxy_func(row.θ_plus)
            xy_minus = XYtoxy_func(row.θ_minus)
            ν_plus, K_plus = xy_plus
            ν_minus, K_minus = xy_minus
            println("    ",
                lpad(string(round(row.δ, digits=2)), 4), "  ",
                lpad(string(round(ν_plus, digits=4)), 9), "  ",
                lpad(string(round(ν_minus, digits=4)), 9), "  ",
                lpad(string(round(K_plus, digits=4)), 8), "  ",
                lpad(string(round(K_minus, digits=4)), 8))
        else
            println("    δ = ", row.δ, ": step leaves bounds; skipped")
        end
    end

    println("\n    Derived quantities and weakness:")
    println("    δ     K/ν(+)   K/ν(-)    νK(+)    νK(-)     ε(+)     ε(-)   ε(-)/ε(+)    d(+)     d(-)")
    println("    " * "-"^104)
    for row in directional_probe.rows
        if row.valid
            xy_plus = XYtoxy_func(row.θ_plus)
            xy_minus = XYtoxy_func(row.θ_minus)
            K_over_ν_plus = xy_plus[2] / xy_plus[1]
            K_over_ν_minus = xy_minus[2] / xy_minus[1]
            νK_plus = xy_plus[1] * xy_plus[2]
            νK_minus = xy_minus[1] * xy_minus[2]
            println("    ",
                lpad(string(round(row.δ, digits=2)), 4), "  ",
                lpad(string(round(K_over_ν_plus, digits=4)), 8), "  ",
                lpad(string(round(K_over_ν_minus, digits=4)), 8), "  ",
                lpad(string(round(νK_plus, digits=4)), 8), "  ",
                lpad(string(round(νK_minus, digits=4)), 8), "  ",
                lpad(string(round(row.ε_plus, digits=4)), 8), "  ",
                lpad(string(round(row.ε_minus, digits=4)), 8), "  ",
                lpad(string(round(row.asymmetry, digits=3)), 11), "  ",
                lpad(string(round(row.d_plus, digits=4)), 8), "  ",
                lpad(string(round(row.d_minus, digits=4)), 8))
        else
            println("    δ = ", row.δ, ": step leaves bounds; skipped")
        end
    end
end
