"""
    ergodic_reference_path(mdp, model, n_steps; kwargs...)

Plan a fixed path using a kernel ergodic-control objective.

The target spatial distribution is built by asking the supplied environment
model for expected information reward at each site. During path construction
the model is not updated.
"""
function ergodic_reference_path(
    mdp::Union{MDP,POMDP},
    model::Any,
    n_steps::Integer;
    initial_state = nothing,
    rng = GLOBAL_RNG,
    quad_order::Integer = 5,
    fourier_order::Integer = 5,
    revisit_penalty::Real = 0.05,
    backend::Symbol = :kernel,
    density_bandwidth::Union{Real, Nothing} = nothing,
    kernel_bandwidth::Union{Real, Nothing} = nothing,
    dt::Real = 1.0,
    optimizer_iters::Integer = 150,
    learning_rate::Real = 0.25,
    momentum::Real = 0.85,
    control_weight::Real = 1e-3,
    boundary_weight::Real = 10.0,
    max_speed::Union{Real, Nothing} = nothing,
    line_search_steps::Integer = 8,
    line_search_decay::Real = 0.5,
    observe_fn::Union{Function, Nothing} = nothing,
    objective = LegacyInformation(),
    history = (),
    infer_actions = true,
    initial_controls = nothing,
    dynamics_problem = mdp,
)
    n_steps < 0 && throw(ArgumentError("n_steps must be nonnegative."))
    quad_order < 1 && throw(ArgumentError("quad_order must be positive."))
    fourier_order < 0 && throw(ArgumentError("fourier_order must be nonnegative."))
    dt <= 0 && throw(ArgumentError("dt must be positive."))
    optimizer_iters < 0 && throw(ArgumentError("optimizer_iters must be nonnegative."))
    learning_rate <= 0 && throw(ArgumentError("learning_rate must be positive."))
    !(0 <= momentum < 1) && throw(ArgumentError("momentum must satisfy 0 <= momentum < 1."))
    control_weight < 0 && throw(ArgumentError("control_weight must be nonnegative."))
    boundary_weight < 0 && throw(ArgumentError("boundary_weight must be nonnegative."))
    line_search_steps < 1 && throw(ArgumentError("line_search_steps must be positive."))
    !(0 < line_search_decay < 1) &&
        throw(ArgumentError("line_search_decay must satisfy 0 < line_search_decay < 1."))

    state = isnothing(initial_state) ? rand(rng, initialstate(mdp)) : copy(initial_state)
    sites = collect(cellsites(mdp))
    isempty(sites) && throw(ArgumentError("mdp must provide at least one cellsite."))

    info_rewards =
        [max(expected_information_gain(objective, mdp, model, site, quad_order), 0.0) for site in sites]
    target_density = normalize_density(info_rewards)
    target_stats = density_statistics(target_density)

    site_points = [point_tuple(site) for site in sites]
    bounds = coordinate_bounds(site_points)

    if backend == :kernel
        path_points, controls, loss_history, kernel_metric_history = optimize_ergodic_trajectory(
            point_tuple(state),
            site_points,
            target_density,
            bounds,
            n_steps;
            density_bandwidth = density_bandwidth,
            kernel_bandwidth = kernel_bandwidth,
            dt = Float64(dt),
            optimizer_iters = optimizer_iters,
            learning_rate = Float64(learning_rate),
            momentum = Float64(momentum),
            control_weight = Float64(control_weight),
            boundary_weight = Float64(boundary_weight),
            max_speed = isnothing(max_speed) ? nothing : Float64(max_speed),
            line_search_steps = line_search_steps,
            line_search_decay = Float64(line_search_decay),
            history, initial_controls,
        )
        states = [point_state(p) for p in path_points]
        actions_taken = infer_actions ? infer_actions_from_path(dynamics_problem, states, rng, objective) : Any[]
        observations = Any[]
        !isnothing(observe_fn) &&
            append!(observations, [call_ergodic_observe(observe_fn, mdp, s) for s in states])

        return (
            states = states,
            actions = actions_taken,
            observations = observations,
            sites = sites,
            information_rewards = info_rewards,
            target_density = target_density,
            target_density_stats = target_stats,
            controls = controls,
            loss_history = loss_history,
            kernel_metric_history = kernel_metric_history,
            backend = backend,
        )
    elseif backend != :fourier_greedy
        throw(ArgumentError("unsupported ergodic backend $(repr(backend)); expected :kernel or :fourier_greedy."))
    end

    modes = fourier_modes(fourier_order)
    target_coeffs = ergodic_coefficients(site_points, target_density, modes, bounds)
    lambda = ergodic_weights(modes)

    states = Any[copy(state)]
    actions_taken = Any[]
    observations = Any[]
    !isnothing(observe_fn) && push!(observations, call_ergodic_observe(observe_fn, mdp, state))

    path_points = isempty(history) ? Tuple{Float64, Float64}[point_tuple(state)] : collect(history)

    for _ in 1:n_steps
        isterminal(mdp, state) && break

        feasible_actions = collect(actions(mdp, state))
        isempty(feasible_actions) && break

        best_action = nothing
        best_next_state = nothing
        best_score = Inf

        for a in feasible_actions
            candidate_state = next_state(dynamics_problem, state, a, rng)
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
        target_density_stats = target_stats,
        backend = backend,
    )
end

function optimize_ergodic_trajectory(
    start::Tuple{Float64, Float64},
    sites::Vector{Tuple{Float64, Float64}},
    density::AbstractVector{<:Real},
    bounds,
    n_steps::Integer;
    density_bandwidth::Union{Real, Nothing},
    kernel_bandwidth::Union{Real, Nothing},
    dt::Float64,
    optimizer_iters::Integer,
    learning_rate::Float64,
    momentum::Float64,
    control_weight::Float64,
    boundary_weight::Float64,
    max_speed::Union{Float64, Nothing},
    line_search_steps::Integer,
    line_search_decay::Float64,
    history = (),
    initial_controls = nothing,
)
    n_steps == 0 && return ([start], zeros(Float64, 0, 2), Float64[], Float64[])

    unit_bounds = (xmin = 0.0, xmax = 1.0, ymin = 0.0, ymax = 1.0, xspan = 1.0, yspan = 1.0)
    unit_start = normalize_point(start, bounds)
    unit_sites = [normalize_point(site, bounds) for site in sites]
    site_matrix = points_matrix(unit_sites)
    past = points_matrix(Tuple{Float64,Float64}[normalize_point(p, bounds) for p in history])
    weights = Float64.(density)
    density_sigma = default_normalized_bandwidth(bounds, density_bandwidth, 0.075)
    kernel_sigma = default_normalized_bandwidth(bounds, kernel_bandwidth, 0.050)
    unit_max_speed = isnothing(max_speed) ? nothing : Float64(max_speed) / max(bounds.xspan, bounds.yspan)
    controls = initialize_kernel_controls(unit_start, site_matrix, weights, unit_bounds, n_steps, dt, unit_max_speed)

    if !isnothing(initial_controls)
        controls .= initial_controls
        controls[:, 1] ./= bounds.xspan
        controls[:, 2] ./= bounds.yspan
        project_controls!(controls, unit_max_speed)
    end

    loss_history = Float64[]
    kernel_metric_history = Float64[]
    velocity = zeros(size(controls))

    for _ in 1:optimizer_iters
        traj = rollout_controls(unit_start, controls, dt)
        cost, grad_u, metric = kernel_ergodic_loss_gradient(
            traj,
            controls,
            site_matrix,
            weights,
            unit_bounds,
            density_sigma,
            kernel_sigma,
            dt,
            control_weight,
            boundary_weight,
            past,
        )
        push!(loss_history, cost)
        push!(kernel_metric_history, metric)

        grad_norm = sqrt(sum(abs2, grad_u))
        grad_norm <= sqrt(eps()) && break
        grad_u ./= max(grad_norm / sqrt(length(grad_u)), 1.0)

        velocity .= momentum .* velocity .- learning_rate .* grad_u
        accepted = false
        step_scale = 1.0
        candidate = similar(controls)
        for _ in 1:line_search_steps
            candidate .= controls .+ step_scale .* velocity
            project_controls!(candidate, unit_max_speed)
            candidate_traj = rollout_controls(unit_start, candidate, dt)
            candidate_cost = kernel_ergodic_loss(
                candidate_traj,
                candidate,
                site_matrix,
                weights,
                unit_bounds,
                density_sigma,
                kernel_sigma,
                dt,
                control_weight,
                boundary_weight,
                past,
            )
            if candidate_cost <= cost
                controls .= candidate
                accepted = true
                break
            end
            step_scale *= line_search_decay
        end
        accepted || fill!(velocity, 0.0)
        project_controls!(controls, unit_max_speed)
    end

    unit_path = clamp_path_to_bounds([unit_start; matrix_points(rollout_controls(unit_start, controls, dt))], unit_bounds)
    path = [denormalize_point(point, bounds) for point in unit_path]
    world_controls = denormalize_controls(controls, bounds)
    return path, world_controls, loss_history, kernel_metric_history
end

function kernel_ergodic_loss(
    traj::Matrix{Float64},
    controls::Matrix{Float64},
    sites::Matrix{Float64},
    weights::Vector{Float64},
    bounds,
    density_sigma::Float64,
    kernel_sigma::Float64,
    dt::Float64,
    control_weight::Float64,
    boundary_weight::Float64,
    past::Matrix{Float64} = zeros(0, 2),
)
    n = size(traj, 1)
    density_total = sum(kde_density(view(traj, i, :), sites, weights, density_sigma) for i in axes(traj, 1))
    kernel_mean = trajectory_kernel_mean(traj, kernel_sigma)
    metric = -2.0 * density_total / n + kernel_mean
    if !isempty(past)
        N = n + size(past, 1)
        cross = sum(gaussian_kernel_value(traj[i,1] - past[j,1],
            traj[i,2] - past[j,2], kernel_sigma)
            for i in axes(traj,1), j in axes(past,1))
        metric = -2.0 * density_total / N + (kernel_mean * n^2 + 2cross) / N^2
    end
    effort = control_weight * dt * sum(abs2, controls)
    boundary = boundary_weight * boundary_penalty(traj, bounds)
    return metric + effort + boundary
end

function kernel_ergodic_loss_gradient(
    traj::Matrix{Float64},
    controls::Matrix{Float64},
    sites::Matrix{Float64},
    weights::Vector{Float64},
    bounds,
    density_sigma::Float64,
    kernel_sigma::Float64,
    dt::Float64,
    control_weight::Float64,
    boundary_weight::Float64,
    past::Matrix{Float64} = zeros(0, 2),
)
    n = size(traj, 1)
    N = n + size(past, 1)
    grad_x = zeros(Float64, n, 2)
    density_sum = 0.0

    for i in 1:n
        x = view(traj, i, :)
        density_sum += kde_density(x, sites, weights, density_sigma)
        grad_x[i, :] .-= (2.0 / N) .* kde_density_gradient(x, sites, weights, density_sigma)
    end

    kernel_sum = 0.0
    inv_kernel_var = 1.0 / (kernel_sigma^2)
    for i in 1:n, j in 1:n
        dx = traj[j, 1] - traj[i, 1]
        dy = traj[j, 2] - traj[i, 2]
        kval = gaussian_kernel_value(dx, dy, kernel_sigma)
        kernel_sum += kval
        scale = (2.0 / (N * N)) * kval * inv_kernel_var
        grad_x[i, 1] += scale * dx
        grad_x[i, 2] += scale * dy
    end

    for i in 1:n, j in axes(past, 1)
        dx, dy = past[j, 1] - traj[i, 1], past[j, 2] - traj[i, 2]
        kval = gaussian_kernel_value(dx, dy, kernel_sigma)
        kernel_sum += 2 * kval
        scale = (2.0 / (N * N)) * kval * inv_kernel_var
        grad_x[i, 1] += scale * dx
        grad_x[i, 2] += scale * dy
    end

    if boundary_weight > 0
        grad_x .+= boundary_weight .* boundary_penalty_gradient(traj, bounds)
    end

    grad_u = zeros(size(controls))
    running = zeros(Float64, 2)
    for t in n:-1:1
        running .+= grad_x[t, :]
        grad_u[t, :] .= dt .* running .+ 2.0 * control_weight * dt .* controls[t, :]
    end

    metric = -2.0 * density_sum / N + kernel_sum / (N * N)
    cost = metric + control_weight * dt * sum(abs2, controls) + boundary_weight * boundary_penalty(traj, bounds)
    return cost, grad_u, metric
end

function kde_density(x, sites::Matrix{Float64}, weights::Vector{Float64}, sigma::Float64)
    total = 0.0
    for i in axes(sites, 1)
        total += weights[i] * gaussian_kernel_value(x[1] - sites[i, 1], x[2] - sites[i, 2], sigma)
    end
    return total
end

function kde_density_gradient(x, sites::Matrix{Float64}, weights::Vector{Float64}, sigma::Float64)
    grad = zeros(Float64, 2)
    inv_var = 1.0 / (sigma^2)
    for i in axes(sites, 1)
        dx = sites[i, 1] - x[1]
        dy = sites[i, 2] - x[2]
        kval = gaussian_kernel_value(dx, dy, sigma)
        scale = weights[i] * kval * inv_var
        grad[1] += scale * dx
        grad[2] += scale * dy
    end
    return grad
end

function gaussian_kernel_value(dx::Real, dy::Real, sigma::Float64)
    return exp(-0.5 * (dx^2 + dy^2) / (sigma^2)) / (2π * sigma^2)
end

function trajectory_kernel_mean(traj::Matrix{Float64}, sigma::Float64)
    n = size(traj, 1)
    total = 0.0
    for i in 1:n, j in 1:n
        total += gaussian_kernel_value(traj[i, 1] - traj[j, 1], traj[i, 2] - traj[j, 2], sigma)
    end
    return total / (n * n)
end

function boundary_penalty(traj::Matrix{Float64}, bounds)
    total = 0.0
    for i in axes(traj, 1)
        total += max(bounds.xmin - traj[i, 1], 0.0)^2
        total += max(traj[i, 1] - bounds.xmax, 0.0)^2
        total += max(bounds.ymin - traj[i, 2], 0.0)^2
        total += max(traj[i, 2] - bounds.ymax, 0.0)^2
    end
    return total / max(size(traj, 1), 1)
end

function boundary_penalty_gradient(traj::Matrix{Float64}, bounds)
    grad = zeros(size(traj))
    n = max(size(traj, 1), 1)
    for i in axes(traj, 1)
        if traj[i, 1] < bounds.xmin
            grad[i, 1] += 2.0 * (traj[i, 1] - bounds.xmin) / n
        elseif traj[i, 1] > bounds.xmax
            grad[i, 1] += 2.0 * (traj[i, 1] - bounds.xmax) / n
        end
        if traj[i, 2] < bounds.ymin
            grad[i, 2] += 2.0 * (traj[i, 2] - bounds.ymin) / n
        elseif traj[i, 2] > bounds.ymax
            grad[i, 2] += 2.0 * (traj[i, 2] - bounds.ymax) / n
        end
    end
    return grad
end

function initialize_kernel_controls(
    start::Tuple{Float64, Float64},
    sites::Matrix{Float64},
    weights::Vector{Float64},
    bounds,
    n_steps::Integer,
    dt::Float64,
    max_speed::Union{Float64, Nothing},
)
    center = vec(sum(sites .* weights, dims = 1))
    span = min(bounds.xspan, bounds.yspan)
    radius = 0.35 * span
    turns = max(1.5, sqrt(n_steps) / 2)
    path = zeros(Float64, n_steps + 1, 2)
    path[1, :] .= (start[1], start[2])
    for t in 1:n_steps
        α = t / n_steps
        θ = 2π * turns * α
        path[t + 1, 1] = center[1] + radius * α * cos(θ)
        path[t + 1, 2] = center[2] + radius * α * sin(θ)
    end
    clamp_matrix_to_bounds!(path, bounds)
    controls = diff(path, dims = 1) ./ dt
    project_controls!(controls, max_speed)
    return controls
end

function rollout_controls(start::Tuple{Float64, Float64}, controls::Matrix{Float64}, dt::Float64)
    traj = zeros(Float64, size(controls, 1), 2)
    x = [start[1], start[2]]
    for t in axes(controls, 1)
        x .+= dt .* controls[t, :]
        traj[t, :] .= x
    end
    return traj
end

function project_controls!(controls::Matrix{Float64}, max_speed::Union{Float64, Nothing})
    isnothing(max_speed) && return controls
    for i in axes(controls, 1)
        speed = hypot(controls[i, 1], controls[i, 2])
        if speed > max_speed
            controls[i, :] .*= max_speed / speed
        end
    end
    return controls
end

function clamp_path_to_bounds(points::Vector{Tuple{Float64, Float64}}, bounds)
    return [(clamp(p[1], bounds.xmin, bounds.xmax), clamp(p[2], bounds.ymin, bounds.ymax)) for p in points]
end

function clamp_matrix_to_bounds!(path::Matrix{Float64}, bounds)
    for i in axes(path, 1)
        path[i, 1] = clamp(path[i, 1], bounds.xmin, bounds.xmax)
        path[i, 2] = clamp(path[i, 2], bounds.ymin, bounds.ymax)
    end
    return path
end

function default_normalized_bandwidth(bounds, bandwidth::Union{Real, Nothing}, default_value::Real)
    if isnothing(bandwidth)
        return Float64(default_value)
    end
    bandwidth <= 0 && throw(ArgumentError("kernel bandwidths must be positive."))
    return Float64(bandwidth) / max(bounds.xspan, bounds.yspan)
end

function points_matrix(points::Vector{Tuple{Float64, Float64}})
    mat = zeros(Float64, length(points), 2)
    for (i, p) in enumerate(points)
        mat[i, 1] = p[1]
        mat[i, 2] = p[2]
    end
    return mat
end

matrix_points(mat::Matrix{Float64}) = [(mat[i, 1], mat[i, 2]) for i in axes(mat, 1)]

point_state(point::Tuple{Float64, Float64}) = reshape([point[1], point[2]], 1, 2)

function normalize_point(point::Tuple{Float64, Float64}, bounds)
    return ((point[1] - bounds.xmin) / bounds.xspan, (point[2] - bounds.ymin) / bounds.yspan)
end

function denormalize_point(point::Tuple{Float64, Float64}, bounds)
    return (bounds.xmin + point[1] * bounds.xspan, bounds.ymin + point[2] * bounds.yspan)
end

function denormalize_controls(controls::Matrix{Float64}, bounds)
    world_controls = similar(controls)
    world_controls[:, 1] .= controls[:, 1] .* bounds.xspan
    world_controls[:, 2] .= controls[:, 2] .* bounds.yspan
    return world_controls
end

function density_statistics(density::AbstractVector{<:Real})
    d = Float64.(density)
    total = sum(d)
    positive = filter(>(0.0), d)
    entropy = isempty(positive) ? 0.0 : -sum(p * log(p) for p in positive)
    return (
        total = total,
        min = minimum(d),
        max = maximum(d),
        nonzero = count(>(0.0), d),
        effective_support = sum(abs2, d) <= eps() ? 0.0 : 1.0 / sum(abs2, d),
        entropy = entropy,
        normalized_entropy = isempty(d) ? 0.0 : entropy / log(length(d)),
    )
end

function select_action(problem, state, target, rng::AbstractRNG)
    feasible = collect(actions(problem, state))
    isempty(feasible) && return nothing
    target_point = point_tuple(target)
    best_action, best_dist = first(feasible), Inf
    for a in feasible
        candidate = next_state(problem, state, a, rng)
        cp = point_tuple(candidate)
        dist = hypot(cp[1] - target_point[1], cp[2] - target_point[2])
        if dist < best_dist
            best_dist, best_action = dist, a
        end
    end
    return best_action
end

select_action(objective, problem, state, target, rng::AbstractRNG) =
    select_action(problem, state, target, rng)

# Legacy action inference used gen, including its measurement RNG draws.
# New integration adapters can instead specialize physical-only selection.
select_action(::LegacyInformation, problem, state, target, rng::AbstractRNG) =
    invoke(select_action, Tuple{Any,Any,Any,AbstractRNG}, problem, state, target, rng)

infer_actions_from_path(problem, states, rng::AbstractRNG, objective) =
    Any[select_action(objective, problem, states[i], states[i+1], rng) for i in 1:length(states)-1]

expected_single_observation_reward(mdp::MDP, model, state, quadrature_order::Integer) =
    max(expected_information_gain(LegacyInformation(), mdp, model, state, quadrature_order), 0.0)

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
