"""
    simulate_info_path(mdp, solver, observe_fn, n_steps; initial_state=nothing, update_belief=true)

Simulate `n_steps` actions of an agent controlled by a Vulcan policy.

`observe_fn` is called as `observe_fn(mdp, state)` at the initial state and after
each executed action. The returned coordinate vectors include the initial location,
so their length is `n_steps + 1`; the observations vector has the same length.

When `update_belief` is true, each real observation is inserted into the GP belief
used to plan the next action.
"""
function simulate_info_path(
    mdp::MDP,
    policy::RiskBoundedInfoPolicy,
    observe_fn::Function,
    n_steps::Integer;
    initial_state = nothing,
    update_belief::Bool = true,
)
    state = isnothing(initial_state) ? rand(solver.rng, initialstate(mdp)) : copy(initial_state)
    current_gp = initialize_gp_belief(mdp, state)

    state_vec = Any[]
    observations = Any[]

    obs = record_visit!(state_vec, observations, mdp, observe_fn, state)
    if update_belief
        ;
        current_gp = add_obs_to_gp(state, obs, current_gp);
    end

    for _ in 1:n_steps
        a = action(policy, state)
        state = next_state(mdp, state, a, policy.solver.rng)

        obs = record_visit!(state_vec, observations, mdp, observe_fn, state)
        if update_belief
            ;
            current_gp = add_obs_to_gp(state, obs, current_gp);
        end
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
    marker_size::Integer = 8,
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
        )
    else
        fig = Plots.plot(;
            title = title,
            xlabel = "x",
            ylabel = "y",
            aspect_ratio = :equal,
            legend = :topright,
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

function call_observation_fn(observation_fn::Function, mdp::MDP, state)
    if applicable(observation_fn, mdp, state)
        return observation_fn(mdp, state)
    else
        return observation_fn(state)
    end
end
