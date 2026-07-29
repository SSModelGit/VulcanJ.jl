using Random
using LinearAlgebra
using SpecialFunctions: erf
using GaussianProcesses
using POMDPs
using POMDPTools

using MuKumari
using VulcanJ

"""
    random_volcanoes(bounds, count; rng)

Create a synthetic MuKumari environment characteristic. Each volcano contributes
an additive cone, caldera ring, and a few radial ridge bumps.
"""
function random_volcanoes(bounds::Tuple{<:Real, <:Real}, count::Integer; rng = MersenneTwister(23))
    lo, hi = Float64.(bounds)
    span = hi - lo
    volcanoes = Vector{Dict{Symbol, Any}}()

    for _ in 1:count
        cx = lo + span * rand(rng)
        cy = lo + span * rand(rng)

        cone_radius = 0.05 * span + 0.07 * span * rand(rng)
        caldera_radius = 0.25 * cone_radius + 0.20 * cone_radius * rand(rng)
        caldera_width = 0.08 * cone_radius + 0.08 * cone_radius * rand(rng)
        ridge_count = rand(rng, 1:3)
        ridge_centers = Matrix{Float64}[]
        ridge_heights = Float64[]
        for _ in 1:ridge_count
            theta = 2pi * rand(rng)
            radius = caldera_radius + cone_radius * (0.25 + 0.65 * rand(rng))
            rx = clamp(cx + radius * cos(theta), lo, hi)
            ry = clamp(cy + radius * sin(theta), lo, hi)
            push!(ridge_centers, [rx ry])
            push!(ridge_heights, 1.0 + 3.5 * rand(rng))
        end

        push!(
            volcanoes,
            Dict{Symbol, Any}(
                :center => [cx cy],
                :cone_height => 5.0 + 8.0 * rand(rng),
                :cone_spread => 1.0 / (2.0 * cone_radius^2),
                :caldera_radius => caldera_radius,
                :caldera_width => caldera_width,
                :caldera_bonus => 2.5 + 5.0 * rand(rng),
                :ridge_centers => ridge_centers,
                :ridge_heights => ridge_heights,
                :ridge_spread => 1.0 / (2.0 * (0.02 * span + 0.03 * span * rand(rng))^2),
            ),
        )
    end

    return volcanoes
end

function volcanic_elevation(volcanoes)
    function elevation(X::Matrix)
        x = Float64(X[1])
        y = Float64(X[2])
        total = 0.0

        for volcano in volcanoes
            center = volcano[:center]
            cx = Float64(center[1])
            cy = Float64(center[2])

            total += volcano[:cone_height] *
                     exp(-volcano[:cone_spread] * ((x - cx)^2 + (y - cy)^2))

            caldera_distance = hypot(x - cx, y - cy)
            total += volcano[:caldera_bonus] *
                     exp(-((caldera_distance - volcano[:caldera_radius])^2) /
                         (2.0 * volcano[:caldera_width]^2))

            for (i, ridge) in enumerate(volcano[:ridge_centers])
                rx = Float64(ridge[1])
                ry = Float64(ridge[2])
                total += volcano[:ridge_heights][i] *
                         exp(-volcano[:ridge_spread] * ((x - rx)^2 + (y - ry)^2))
            end
        end

        return total
    end
    return elevation
end

const MUKUMARI_VULCAN_CONFIG = IdDict{KAgentMDP, NamedTuple}()

mukumari_config(mdp::KAgentMDP) = MUKUMARI_VULCAN_CONFIG[mdp]

function build_mukumari_volcano_mdp(;
    rng = MersenneTwister(19),
    dimensions = (0.0, 100.0),
    volcano_count::Integer = 14,
    cellsite_resolution = (32, 32),
    horizon_steps::Integer = 12,
)
    volcanoes = random_volcanoes(dimensions, volcano_count; rng = rng)
    elevation = volcanic_elevation(volcanoes)
    menv = MuEnv(1, [:elevation], Dict(:elevation => elevation))

    # MuKumari objectives are included to show that this is a normal KAgentMDP.
    # VulcanJ's information objective is supplied by the adapter hooks below.
    goal = Dict(:target => [92.0 92.0], :strength => 25.0, :influence => 35.0, :size => 4.0)
    obstacle = Dict(
        :poly => [(47.0, 47.0), (47.0, 53.0), (53.0, 53.0), (53.0, 47.0), (47.0, 47.0)],
        :risk => 150.0,
        :impact => 0.1,
    )
    obj_landscape = AgentObjectiveLandscape(;
        objectives = [(:goal, goal), (:robc, [obstacle]), (:horz, 0.02)],
    )

    mdp = init_standard_KAgentMDP(;
        name = "mukumari_volcano_agent",
        start = [8.0 8.0],
        dimensions = dimensions,
        objl = obj_landscape,
        menv = menv,
        digits = 3,
        agent_width = 0.1,
        agent_speed = 8.0,
        ag_mvt_noise = 0.0,
        obs_noise = 0.0,
        mdp_horizon_discount = 0.97,
    )

    lo, hi = Float64.(dimensions)
    prior_sites = [
        [lo + 0.08 * (hi - lo) lo + 0.08 * (hi - lo)],
        [lo + 0.08 * (hi - lo) lo + 0.92 * (hi - lo)],
        [lo + 0.92 * (hi - lo) lo + 0.08 * (hi - lo)],
        [lo + 0.92 * (hi - lo) lo + 0.92 * (hi - lo)],
        [lo + 0.50 * (hi - lo) lo + 0.50 * (hi - lo)],
    ]

    MUKUMARI_VULCAN_CONFIG[mdp] = (
        bounds = dimensions,
        volcanoes = volcanoes,
        cellsite_resolution = cellsite_resolution,
        horizon_steps = horizon_steps,
        gp_length_scale = 12.0,
        elevation_threshold = 7.0,
        prior_sites = prior_sites,
        risk_budget = 3.0,
        ergodic_max_speed = 12.0,
    )

    return mdp
end

### --------------------------------------------------------------------------
# VulcanJ adapter methods for MuKumari.KAgentMDP
### --------------------------------------------------------------------------

VulcanJ.extract_location(s::KAgentState) = reshape(Float64.(s.x), 1, :)

POMDPs.actions(mdp::KAgentMDP, ::KAgentState) = POMDPs.actions(mdp)
POMDPs.actions(mdp::KAgentMDP, ::Matrix) = POMDPs.actions(mdp)

function POMDPs.gen(mdp::KAgentMDP, s::Matrix, a::Symbol, rng::AbstractRNG)
    return POMDPs.gen(mdp, blindstart_KAgentState(mdp, reshape(Float64.(s), 1, :)), a, rng)
end

function mukumari_elevation(mdp::KAgentMDP, state)
    X = VulcanJ.extract_location(state)
    return Float64(mdp.menv.μf[:elevation](X))
end

function VulcanJ.get_initial_gp(mdp::KAgentMDP, state)
    cfg = mukumari_config(mdp)
    X0 = hcat([site' for site in cfg.prior_sites]...)
    y0 = [mukumari_elevation(mdp, site) for site in cfg.prior_sites]
    log_length_scale = log(Float64(cfg.gp_length_scale))
    return GPE(X0, y0, MeanZero(), SE(fill(log_length_scale, 2), 0.0))
end

function VulcanJ.add_obs_to_gp(state::Union{KAgentState, Matrix}, y::Real, gp::GPE)
    X = VulcanJ.extract_location(state)
    return GPE(hcat(gp.x, X'), vcat(gp.y, Float64(y)), gp.mean, gp.kernel)
end

function VulcanJ.posterior_phenomenon_prob(mdp::KAgentMDP, gp::GPE, state)
    X = VulcanJ.extract_location(state)
    mu, sigma2 = predict_f(gp, X')
    mu_v = first(vec(mu))
    sigma2_v = max(first(vec(sigma2)), eps())
    threshold = mukumari_config(mdp).elevation_threshold
    return 0.5 * (1.0 - erf((threshold - mu_v) / sqrt(2.0 * sigma2_v)))
end

VulcanJ.get_failure_prob(mdp::KAgentMDP, state, action::Symbol) = 0.0
VulcanJ.horizon(mdp::KAgentMDP) = mukumari_config(mdp).horizon_steps

function VulcanJ.cellsites(mdp::KAgentMDP)
    cfg = mukumari_config(mdp)
    lo, hi = Float64.(cfg.bounds)
    nx, ny = cfg.cellsite_resolution
    xs = range(lo, hi; length = nx)
    ys = range(lo, hi; length = ny)
    return [reshape([Float64(x), Float64(y)], 1, 2) for x in xs for y in ys]
end

### --------------------------------------------------------------------------
# Demo entry points
### --------------------------------------------------------------------------

function mukumari_vulcan_solver(; rng = MersenneTwister(7))
    mdp = build_mukumari_volcano_mdp(; rng = rng)
    solver = RiskBoundedInfoMCTS(
        lookahead = 5,
        time_budget = 3.0,
        quad_order = 5,
        risk_budget = mukumari_config(mdp).risk_budget,
        alpha = 0.0,
        reference_reward = 1.0,
        rng = rng,
    )
    return mdp, solver
end

function run_mukumari_standard_vulcan_demo(; rng = MersenneTwister(7), n_steps::Integer = 8)
    mdp, solver = mukumari_vulcan_solver(; rng = rng)
    policy = solve(solver, mdp)
    initial_state = rand(rng, initialstate(mdp))
    states, observations = simulate_info_path(
        mdp,
        policy,
        mukumari_elevation,
        n_steps;
        initial_state = initial_state,
        update_belief = true,
    )
    return (mdp = mdp, solver = solver, policy = policy, states = states, observations = observations)
end

function run_mukumari_ergodic_demo(mdp::KAgentMDP; rng = MersenneTwister(17), n_steps::Integer = 80)
    initial_state = rand(rng, initialstate(mdp))
    gp = get_initial_gp(mdp, initial_state)
    return one_shot_ergodic_planner(
        mdp,
        gp,
        n_steps;
        initial_state = initial_state,
        rng = rng,
        max_speed = mukumari_config(mdp).ergodic_max_speed,
        observe_fn = mukumari_elevation,
        optimizer_iters = 150,
    )
end

function save_mukumari_volcano_plots(standard_result, ergodic_result)
    res_dir = joinpath(@__DIR__, "res")
    mkpath(res_dir)

    mdp = standard_result.mdp
    plot_simulated_path(
        mdp,
        standard_result.states,
        standard_result.observations;
        title = "MuKumari + VulcanJ Search Path",
        ground_truth_fn = mukumari_elevation,
        heatmap_resolution = 180,
        marker_size = 3,
        plot_size = (1300, 1000),
        save_path = joinpath(res_dir, "mukumari_vulcan_search_path.png"),
    )

    plot_information_reward_path(
        mdp,
        standard_result.states,
        ergodic_result.sites,
        ergodic_result.target_density;
        title = "MuKumari + VulcanJ Search Path on Information Density",
        marker_size = 3,
        plot_size = (1300, 1000),
        save_path = joinpath(res_dir, "mukumari_vulcan_information_path.png"),
    )

    plot_simulated_path(
        mdp,
        ergodic_result.states,
        ergodic_result.observations;
        title = "MuKumari + Ergodic Path on Ground Truth",
        ground_truth_fn = mukumari_elevation,
        heatmap_resolution = 180,
        marker_size = 2,
        plot_size = (1300, 1000),
        save_path = joinpath(res_dir, "mukumari_ergodic_search_path.png"),
    )

    plot_information_reward_path(
        mdp,
        ergodic_result;
        title = "MuKumari + Ergodic Path on Information Density",
        marker_size = 2,
        plot_size = (1300, 1000),
        save_path = joinpath(res_dir, "mukumari_ergodic_information_path.png"),
    )
end

standard_result = run_mukumari_standard_vulcan_demo()
ergodic_result = run_mukumari_ergodic_demo(standard_result.mdp)

println("Ergodic target density statistics: ", ergodic_result.target_density_stats)
save_mukumari_volcano_plots(standard_result, ergodic_result)