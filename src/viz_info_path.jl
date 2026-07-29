"""
    simulate_info_path(mdp, solver, observe_fn, n_steps; initial_state=nothing, update_belief=true)

Simulate `n_steps` actions of an agent controlled by a Vulcan policy.

`observe_fn` is called as `observe_fn(mdp, state)` at the initial state and after
each executed action. The returned coordinate vectors include the initial location,
so their length is `n_steps + 1`; the observations vector has the same length.

When `update_model` is true, each real observation is passed to the supplied
environment model's conditioning function before the next planning query.
"""
function simulate_info_path(
    mdp::MDP,
    policy::RiskBoundedInfoPolicy,
    observe_fn::Function,
    n_steps::Integer;
    initial_state = nothing,
    update_model::Bool = true,
)
    state = isnothing(initial_state) ? rand(policy.solver.rng, initialstate(mdp)) : copy(initial_state)
    current_model = initial_environment_model(mdp, state)

    state_vec = Any[]
    observations = Any[]

    obs = record_visit!(state_vec, observations, mdp, observe_fn, state)
    if update_model
        current_model = condition_environment_model(mdp, current_model, state, obs)
    end
    set_environment_model!(policy, state, current_model)

    for _ in 1:n_steps
        a = action(policy, state)
        state = next_state(mdp, state, a, policy.solver.rng)

        obs = record_visit!(state_vec, observations, mdp, observe_fn, state)
        if update_model
            current_model = condition_environment_model(mdp, current_model, state, obs)
        end
        set_environment_model!(policy, state, current_model)
    end

    return state_vec, observations
end

function record_visit!(state_vec::Vector, observations::Vector, mdp::MDP, observe_fn::Function, state)
    push!(state_vec, state)
    obs = observe_fn(mdp, state)
    push!(observations, obs)
    return obs
end

function plot_simulated_path(
    mdp::MDP,
    state_vec::Vector,
    observations::Vector;
    title = "Simulated Explorative Path",
    ground_truth_fn::Union{Function, Nothing} = nothing,
    observation_fn::Union{Function, Nothing} = ground_truth_fn,
    save_path = nothing,
    heatmap_resolution::Integer = 100,
    marker_size::Integer = 4,
    plot_size::Tuple{Int, Int} = (1000, 800),
)
    xs, ys = path_coordinates(state_vec)

    if !isnothing(observation_fn)
        xgrid, ygrid = heatmap_grid(mdp, xs, ys, heatmap_resolution)
        z = [call_observation_fn(observation_fn, mdp, reshape([x, y], 1, 2)) for y in ygrid, x in xgrid]
        fig = Plots.heatmap(
            xgrid,
            ygrid,
            z;
            title = title,
            xlabel = "x",
            ylabel = "y",
            colorbar_title = "Ground Truth",
            aspect_ratio = :equal,
            legend = :topright,
            size = plot_size,
        )
    else
        fig = Plots.plot(;
            title = title,
            xlabel = "x",
            ylabel = "y",
            aspect_ratio = :equal,
            legend = :topright,
            size = plot_size,
        )
    end

    Plots.plot!(
        fig,
        xs,
        ys;
        label = "Simulated Path",
        color = :red,
        linewidth = 2,
        linestyle = :dot,
        marker = :circle,
        markersize = marker_size,
        markerstrokecolor = :white,
        markerstrokewidth = 1,
    )

    if !isnothing(save_path)
        Plots.savefig(fig, save_path)
    end
    return fig
end

"""
    plot_information_reward_path(mdp, state_vec, sites, rewards; kwargs...)

Plot a path over the spatial information reward field used by the ergodic planner.

`sites` and `rewards` should be aligned vectors, such as
`ergodic_result.sites` and `ergodic_result.target_density` or
`ergodic_result.information_rewards`.
"""
function plot_information_reward_path(
    mdp::MDP,
    state_vec::Vector,
    sites::Vector,
    rewards::AbstractVector{<:Real};
    title = "Information Reward Path",
    save_path = nothing,
    marker_size::Integer = 4,
    plot_size::Tuple{Int, Int} = (1000, 800),
    colorbar_title = "Information Reward",
    path_label = "Simulated Path",
    reward_scale::Symbol = :log,
    quantile_clip::Real = 0.98,
)
    length(sites) == length(rewards) ||
        throw(ArgumentError("sites and rewards must have the same length."))
    isempty(sites) && throw(ArgumentError("sites must be nonempty."))

    xs, ys = path_coordinates(state_vec)
    xgrid, ygrid, z = reward_heatmap_grid(sites, rewards)
    z_plot, scaled_colorbar_title = scaled_reward_heatmap(z, colorbar_title, reward_scale, quantile_clip)

    fig = Plots.heatmap(
        xgrid,
        ygrid,
        z_plot;
        title = title,
        xlabel = "x",
        ylabel = "y",
        colorbar_title = scaled_colorbar_title,
        aspect_ratio = :equal,
        legend = :topright,
        size = plot_size,
    )

    Plots.plot!(
        fig,
        xs,
        ys;
        label = path_label,
        color = :red,
        linewidth = 2,
        linestyle = :dot,
        marker = :circle,
        markersize = marker_size,
        markerstrokecolor = :white,
        markerstrokewidth = 1,
    )

    if !isnothing(save_path)
        Plots.savefig(fig, save_path)
    end
    return fig
end

function plot_information_reward_path(
    mdp::MDP,
    ergodic_result;
    use_normalized_density::Bool = true,
    colorbar_title::Union{String, Nothing} = nothing,
    kwargs...,
)
    rewards = use_normalized_density ? ergodic_result.target_density : ergodic_result.information_rewards
    resolved_colorbar_title =
        isnothing(colorbar_title) ? (use_normalized_density ? "Information Density" : "Information Reward") :
        colorbar_title
    return plot_information_reward_path(
        mdp,
        ergodic_result.states,
        ergodic_result.sites,
        rewards;
        colorbar_title = resolved_colorbar_title,
        kwargs...,
    )
end

function path_coordinates(state_vec::Vector)
    xs = Float64[]
    ys = Float64[]
    for state in state_vec
        loc = vec(extract_location(state))
        push!(xs, loc[1])
        push!(ys, loc[2])
    end
    return xs, ys
end

function heatmap_grid(mdp::MDP, xs::Vector{Float64}, ys::Vector{Float64}, resolution::Integer)
    resolution < 2 && throw(ArgumentError("heatmap_resolution must be at least 2."))

    if hasproperty(mdp, :env) && haskey(mdp.env, :field_size)
        nx, ny = mdp.env[:field_size]
        return range(1.0, float(nx); length = resolution), range(1.0, float(ny); length = resolution)
    end

    xmin, xmax = extrema(xs)
    ymin, ymax = extrema(ys)
    xpad = max(1.0, 0.05 * max(xmax - xmin, eps()))
    ypad = max(1.0, 0.05 * max(ymax - ymin, eps()))

    return range(xmin - xpad, xmax + xpad; length = resolution),
    range(ymin - ypad, ymax + ypad; length = resolution)
end

function reward_heatmap_grid(sites::Vector, rewards::AbstractVector{<:Real})
    points = [vec(extract_location(site)) for site in sites]
    xs = sort(unique(Float64(point[1]) for point in points))
    ys = sort(unique(Float64(point[2]) for point in points))
    z = fill(NaN, length(ys), length(xs))

    x_index = Dict(x => i for (i, x) in enumerate(xs))
    y_index = Dict(y => i for (i, y) in enumerate(ys))
    for (point, reward) in zip(points, rewards)
        x = Float64(point[1])
        y = Float64(point[2])
        z[y_index[y], x_index[x]] = Float64(reward)
    end

    return xs, ys, z
end

function scaled_reward_heatmap(
    z::Matrix{Float64},
    colorbar_title,
    reward_scale::Symbol,
    quantile_clip::Real,
)
    if reward_scale == :linear
        return z, colorbar_title
    elseif reward_scale == :log
        finite_positive = [v for v in vec(z) if isfinite(v) && v > 0.0]
        isempty(finite_positive) && return z, colorbar_title
        floor_value = maximum(finite_positive) * 1e-6
        z_scaled = similar(z)
        for i in eachindex(z)
            v = z[i]
            z_scaled[i] = isfinite(v) ? log10(max(v, floor_value)) : v
        end
        return z_scaled, "$colorbar_title (log10)"
    elseif reward_scale == :quantile
        0 < quantile_clip <= 1 ||
            throw(ArgumentError("quantile_clip must satisfy 0 < quantile_clip <= 1."))
        finite_values = sort([v for v in vec(z) if isfinite(v)])
        isempty(finite_values) && return z, colorbar_title
        upper = empirical_quantile(finite_values, Float64(quantile_clip))
        return clamp.(z, -Inf, upper), "$colorbar_title (clipped)"
    else
        throw(ArgumentError("unsupported reward_scale $(repr(reward_scale)); expected :linear, :log, or :quantile."))
    end
end

function empirical_quantile(sorted_values::Vector{Float64}, q::Float64)
    length(sorted_values) == 1 && return only(sorted_values)
    idx = 1 + q * (length(sorted_values) - 1)
    lo = floor(Int, idx)
    hi = ceil(Int, idx)
    lo == hi && return sorted_values[lo]
    α = idx - lo
    return (1 - α) * sorted_values[lo] + α * sorted_values[hi]
end

function call_observation_fn(observation_fn::Function, mdp::MDP, state)
    if applicable(observation_fn, mdp, state)
        return observation_fn(mdp, state)
    else
        return observation_fn(state)
    end
end
