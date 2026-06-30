"""
    one_shot_ergodic_planner(mdp, gp, n_steps; kwargs...)

Plan a fixed path using a lightweight ergodic-control-style objective.

The target spatial distribution is built from expected information reward under the
initial GP: for each cell, the planner compares the initial GP to the posterior formed
by adding one synthetic measurement at that cell. During path construction the GP is
not updated; the path is chosen to make the empirical visitation distribution match
the information target.
"""
function one_shot_ergodic_planner(
    mdp::MDP,
    gp::Any,
    n_steps::Integer;
    initial_state = nothing,
    rng = GLOBAL_RNG,
    quad_order::Integer = 5,
    fourier_order::Integer = 5,
    revisit_penalty::Real = 0.05,
    observe_fn::Union{Function, Nothing} = nothing,
)
    n_steps < 0 && throw(ArgumentError("n_steps must be nonnegative."))
    quad_order < 1 && throw(ArgumentError("quad_order must be positive."))
    fourier_order < 0 && throw(ArgumentError("fourier_order must be nonnegative."))

    state = isnothing(initial_state) ? rand(rng, initialstate(mdp)) : copy(initial_state)
    sites = collect(cellsites(mdp))
    isempty(sites) && throw(ArgumentError("mdp must provide at least one cellsite."))

    info_rewards = [expected_single_observation_reward(mdp, gp, site, quad_order) for site in sites]
    target_density = normalize_density(info_rewards)

    site_points = [point_tuple(site) for site in sites]
    bounds = coordinate_bounds(site_points)
    modes = fourier_modes(fourier_order)
    target_coeffs = ergodic_coefficients(site_points, target_density, modes, bounds)
    lambda = ergodic_weights(modes)

    states = Any[copy(state)]
    actions_taken = Any[]
    observations = Any[]
    !isnothing(observe_fn) && push!(observations, call_ergodic_observe(observe_fn, mdp, state))

    path_points = Tuple{Float64, Float64}[point_tuple(state)]

    for _ in 1:n_steps
        isterminal(mdp, state) && break

        feasible_actions = collect(actions(mdp, state))
        isempty(feasible_actions) && break

        best_action = nothing
        best_next_state = nothing
        best_score = Inf

        for a in feasible_actions
            candidate_state = next_state(mdp, state, a, rng)
            candidate_points = [path_points; point_tuple(candidate_state)]
            coeffs = trajectory_coefficients(candidate_points, modes, bounds)
            score = ergodic_metric(coeffs, target_coeffs, lambda)
            score += revisit_penalty * revisit_count(candidate_state, states)

            if score < best_score
                best_score = score
                best_action = a
                best_next_state = candidate_state
            end
        end

        isnothing(best_action) && break

        push!(actions_taken, best_action)
        state = best_next_state
        push!(states, copy(state))
        push!(path_points, point_tuple(state))
        !isnothing(observe_fn) && push!(observations, call_ergodic_observe(observe_fn, mdp, state))
    end

    return (
        states = states,
        actions = actions_taken,
        observations = observations,
        sites = sites,
        information_rewards = info_rewards,
        target_density = target_density,
    )
end

function expected_single_observation_reward(mdp::MDP, gp::Any, state, quad_order::Integer)
    μ, σ² = gp_predict(gp, state)
    σ² = max(float(σ²), eps())
    abscissae, weights = gausshermite(quad_order)

    reward = 0.0
    for j in eachindex(abscissae)
        y = μ + sqrt(2 * σ²) * abscissae[j]
        gp_posterior = add_obs_to_gp(state, y, gp)
        reward += (weights[j] / sqrt(π)) * compute_kl_reward(gp, gp_posterior, mdp)
    end
    return max(reward, 0.0)
end

function normalize_density(weights::AbstractVector{<:Real})
    total = sum(weights)
    if !(isfinite(total)) || total <= eps()
        return fill(1.0 / length(weights), length(weights))
    end
    return Float64.(weights ./ total)
end

function point_tuple(state)
    loc = vec(extract_location(state))
    length(loc) < 2 && throw(ArgumentError("ergodic planner requires at least two-dimensional states."))
    return (Float64(loc[1]), Float64(loc[2]))
end

function coordinate_bounds(points::Vector{Tuple{Float64, Float64}})
    xs = first.(points)
    ys = last.(points)
    xmin, xmax = extrema(xs)
    ymin, ymax = extrema(ys)
    xspan = max(xmax - xmin, eps())
    yspan = max(ymax - ymin, eps())
    return (xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, xspan = xspan, yspan = yspan)
end

fourier_modes(order::Integer) = [(kx, ky) for kx in 0:order for ky in 0:order]

ergodic_weights(modes) = [1.0 / (1.0 + kx^2 + ky^2)^1.5 for (kx, ky) in modes]

function ergodic_coefficients(points, density, modes, bounds)
    coeffs = zeros(Float64, length(modes))
    for (point, w) in zip(points, density)
        basis = fourier_basis(point, modes, bounds)
        coeffs .+= w .* basis
    end
    return coeffs
end

function trajectory_coefficients(points, modes, bounds)
    coeffs = zeros(Float64, length(modes))
    for point in points
        coeffs .+= fourier_basis(point, modes, bounds)
    end
    return coeffs ./ length(points)
end

function fourier_basis(point::Tuple{Float64, Float64}, modes, bounds)
    x = clamp((point[1] - bounds.xmin) / bounds.xspan, 0.0, 1.0)
    y = clamp((point[2] - bounds.ymin) / bounds.yspan, 0.0, 1.0)
    return [cos(kx * π * x) * cos(ky * π * y) for (kx, ky) in modes]
end

ergodic_metric(coeffs, target_coeffs, lambda) = sum(lambda .* (coeffs .- target_coeffs) .^ 2)

function revisit_count(state, states)
    p = point_tuple(state)
    return count(s -> point_tuple(s) == p, states)
end

function call_ergodic_observe(observe_fn::Function, mdp::MDP, state)
    if applicable(observe_fn, mdp, state)
        return observe_fn(mdp, state)
    else
        return observe_fn(state)
    end
end
