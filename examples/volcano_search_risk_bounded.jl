using Parameters: @with_kw
using Match

using Random
using LinearAlgebra
using SpecialFunctions: erf
using Distributions: Normal
using GaussianProcesses
using POMDPs
using POMDPTools

using VulcanJ

function random_volcanoes(field_size::Tuple{Int, Int}, count::Integer; rng = MersenneTwister(23))
    nx, ny = field_size
    volcanoes = Vector{Dict{Symbol, Any}}()

    for _ in 1:count
        cx = 1.0 + (nx - 1.0) * rand(rng)
        cy = 1.0 + (ny - 1.0) * rand(rng)

        cone_radius = 45.0 + 55.0 * rand(rng)
        caldera_radius = 15.0 + 25.0 * rand(rng)
        caldera_width = 5.0 + 10.0 * rand(rng)
        ridge_count = rand(rng, 1:3)
        ridge_centers = Matrix{Float64}[]
        ridge_heights = Float64[]
        for _ in 1:ridge_count
            θ = 2π * rand(rng)
            r = caldera_radius + cone_radius * (0.25 + 0.65 * rand(rng))
            rx = clamp(cx + r * cos(θ), 1.0, float(nx))
            ry = clamp(cy + r * sin(θ), 1.0, float(ny))
            push!(ridge_centers, [rx ry])
            push!(ridge_heights, 1.5 + 3.5 * rand(rng))
        end

        push!(
            volcanoes,
            Dict{Symbol, Any}(
                :center => [cx cy],
                :cone_height => 6.0 + 6.0 * rand(rng),
                :cone_spread => 1.0 / (2.0 * cone_radius^2),
                :caldera_radius => caldera_radius,
                :caldera_width => caldera_width,
                :caldera_bonus => 3.0 + 5.0 * rand(rng),
                :ridge_centers => ridge_centers,
                :ridge_heights => ridge_heights,
                :ridge_spread => 1.0 / (2.0 * (20.0 + 35.0 * rand(rng))^2),
            ),
        )
    end

    return volcanoes
end

function random_sites(field_size::Tuple{Int, Int}, count::Integer; rng = MersenneTwister(31))
    nx, ny = field_size
    return [[1.0 + (nx - 1.0) * rand(rng) 1.0 + (ny - 1.0) * rand(rng)] for _ in 1:count]
end

const VOLCANO_FIELD_SIZE = (1000, 1000)

VOLCANO_ENV = Dict{Symbol, Any}(
    :field_size => VOLCANO_FIELD_SIZE,
    :cellsite_resolution => (35, 35),
    :horizon_steps => 12,
    :discount_factor => 0.97,
    :step_size => 150.0,
    :start_state => [50.0 50.0],
    :volcanoes => random_volcanoes(VOLCANO_FIELD_SIZE, 50),
    :gp_length_scale => 140.0,
    :base_risk => 0.02,
    :obstacle_risk_peak => 0.20,
    :obstacle_risk_radius => 90.0,
    :obstacle_risk_sigma => 35.0,
    :obstacle_points => random_sites(VOLCANO_FIELD_SIZE, 20; rng = MersenneTwister(41)),
    :elevation_threshold => 7.0,
    :prior_sites => [
        [50.0 50.0],
        [50.0 950.0],
        [950.0 50.0],
        [950.0 950.0],
        random_sites(VOLCANO_FIELD_SIZE, 8; rng = MersenneTwister(47))...,
    ],
)

function make_volcanic_elevation(env::Dict{Symbol, Any})
    volcanoes = env[:volcanoes]

    function elevation(s)
        x = float(s[1])
        y = float(s[2])

        total = 0.0
        for volcano in volcanoes
            center = volcano[:center]
            cx = float(center[1])
            cy = float(center[2])

            cone_height = float(volcano[:cone_height])
            cone_spread = float(volcano[:cone_spread])
            total += cone_height * exp(-cone_spread * ((x - cx)^2 + (y - cy)^2))

            caldera_distance = hypot(x - cx, y - cy)
            caldera_radius = float(volcano[:caldera_radius])
            caldera_width = float(volcano[:caldera_width])
            caldera_bonus = float(volcano[:caldera_bonus])
            total += caldera_bonus * exp(-((caldera_distance - caldera_radius)^2) / (2 * caldera_width^2))

            ridge_centers = volcano[:ridge_centers]
            ridge_heights = volcano[:ridge_heights]
            ridge_spread = float(volcano[:ridge_spread])
            for (i, ridge) in enumerate(ridge_centers)
                rx = float(ridge[1])
                ry = float(ridge[2])
                total += ridge_heights[i] * exp(-ridge_spread * ((x - rx)^2 + (y - ry)^2))
            end
        end

        return total
    end
    return elevation
end

### ================================================
# POMDP Declaration extending from POMDPs.jl package
### ================================================

@with_kw struct VolcanoSearchMDP <: POMDPs.MDP{Matrix, Symbol}
    env::Dict{Symbol, Any} = VOLCANO_ENV
    elevation_fn::Function = make_volcanic_elevation(VOLCANO_ENV)
end

# Defining all the required interfaces for an object-oriented POMDP approach
# See https://juliapomdp.github.io/POMDPs.jl/stable/def_pomdp/#Object-oriented for details

POMDPs.statetype(::VolcanoSearchMDP) = Matrix
POMDPs.actiontype(::VolcanoSearchMDP) = Symbol
POMDPs.discount(mdp::VolcanoSearchMDP) = mdp.env[:discount_factor]
POMDPs.initialstate(mdp::VolcanoSearchMDP) = Deterministic(copy(mdp.env[:start_state]))

function POMDPs.actions(::VolcanoSearchMDP, ::Matrix)
    [:wait, :north, :south, :east, :west, :northeast, :northwest, :southeast, :southwest]
end

function volcano_step(mdp::VolcanoSearchMDP, s::Matrix, a::Symbol)
    step_size = float(mdp.env[:step_size])
    field_size = mdp.env[:field_size]

    dx, dy = @match a begin
        :wait => (0.0, 0.0)
        :north => (0.0, step_size)
        :south => (0.0, -step_size)
        :east => (step_size, 0.0)
        :west => (-step_size, 0.0)
        :northeast => (step_size, step_size)
        :northwest => (-step_size, step_size)
        :southeast => (step_size, -step_size)
        :southwest => (-step_size, -step_size)
    end

    return [clamp(s[1] + dx, 1.0, float(field_size[1])) clamp(s[2] + dy, 1.0, float(field_size[2]))]
end

function POMDPs.gen(mdp::VolcanoSearchMDP, s::Matrix, a::Symbol, rng::AbstractRNG)
    # Deterministic state update; risk assessment is left to the solver via the collision_probability contract.
    sp = volcano_step(mdp, s, a)
    r = mdp.elevation_fn(sp)
    return (sp = sp, r = r)
end

# no need to terminate exploration in this example
POMDPs.isterminal(mdp::VolcanoSearchMDP, s::Matrix) = false

### ==================================================================
# Defining required interface functions for VulcanJ solver integration
### ==================================================================

# Generate initial Gaussian depending on the prior we have
function VulcanJ.get_initial_gp(mdp::VolcanoSearchMDP, ::Matrix)
    prior_sites = mdp.env[:prior_sites]
    # build X0 as 2×N matrix for GP inputs
    X0 = hcat([site' for site in prior_sites]...)
    y0 = [mdp.elevation_fn(site) for site in prior_sites]
    log_length_scale = log(float(mdp.env[:gp_length_scale]))
    return GPE(X0, y0, MeanZero(), SE(fill(log_length_scale, 2), 0.0))
end

# How do we add observations?
function VulcanJ.add_obs_to_gp(X::Matrix, y::Float64, gp::GPE)
    y_new = vcat(gp.y, y)
    x_new = hcat(gp.x, X')
    return GPE(x_new, y_new, gp.mean, gp.kernel)
end

# Define likelihood of failure / collision / risky behavior

# helper to check out of bounds
function boundary_violation(sp, env)
    let sp1=sp[1], sp2=sp[2], limx=env[:field_size][1], limy=env[:field_size][2]
        sp1 <= 1.0 || sp2 <= 1.0 || sp1 >= float(limx) || sp2 >= float(limy)
    end
end

function obstacle_risk_at(s, obstacles, peak_risk, radius, sigma)
    x = float(s[1])
    y = float(s[2])
    total = 0.0
    for obs in obstacles
        ox = float(obs[1])
        oy = float(obs[2])
        d = hypot(x - ox, y - oy)
        if d ≤ radius
            total += peak_risk * exp(-(d^2) / (2 * sigma^2))
        end
    end
    return total
end

# 
function VulcanJ.get_failure_prob(mdp::VolcanoSearchMDP, s::Matrix, a::Symbol)
    # let sp = volcano_step(mdp, s, a),
    #     base_risk = mdp.env[:base_risk],
    #     obstacle_peak = mdp.env[:obstacle_risk_peak],
    #     obstacle_radius = mdp.env[:obstacle_risk_radius],
    #     obstacle_sigma = mdp.env[:obstacle_risk_sigma],
    #     obstacles = mdp.env[:obstacle_points]

    #     boundary_penalty = boundary_violation(sp, mdp.env) ? 0.10 : 0.0
    #     obstacle_risk = obstacle_risk_at(sp, obstacles, obstacle_peak, obstacle_radius, obstacle_sigma)
    #     return clamp(base_risk + obstacle_risk + boundary_penalty, 0.0, 0.95)
    # end
    return 0.0  # for now, ignore risk in this example
end

# Compute the likelihood of the phenomenon of interest at a given site existing based on the current GP
function VulcanJ.posterior_phenomenon_prob(mdp::VolcanoSearchMDP, gp::GPE, s::Matrix)
    μ, Σ = predict_f(gp, s')
    μv = first(vec(μ))
    σ² = max(first(vec(Σ)), eps())
    threshold = mdp.env[:elevation_threshold]
    return 0.5 * (1 - erf((threshold - μv) / sqrt(2 * σ²)))
end

# Housekeeping items
VulcanJ.horizon(mdp::VolcanoSearchMDP) = mdp.env[:horizon_steps] # MCTS horizon

function VulcanJ.cellsites(mdp::VolcanoSearchMDP)
    nx, ny = mdp.env[:field_size]
    rx, ry = get(mdp.env, :cellsite_resolution, mdp.env[:field_size])
    xs = range(1.0, float(nx); length = rx)
    ys = range(1.0, float(ny); length = ry)
    return [reshape([float(x), float(y)], 1, 2) for x in xs for y in ys]
end

function volcano_search_solver(; rng = MersenneTwister(7))
    mdp = VolcanoSearchMDP()
    solver = RiskBoundedInfoMCTS(
        lookahead = 6,
        time_budget = 25,
        quad_order = 5,
        risk_budget = 5.0,
        alpha = 0.0,
        reference_reward = 1.0,
        rng = rng,
    )
    return mdp, solver
end

function run_volcano_search_demo(; rng = MersenneTwister(7))
    mdp, solver = volcano_search_solver(; rng = rng)
    policy = solve(solver, mdp)
    s0 = copy(mdp.env[:start_state])
    a0 = action(policy, s0)
    return (mdp = mdp, policy = policy, state = s0, action = a0)
end

result = run_volcano_search_demo();

path_states, path_observations = simulate_info_path(
    result[1],
    result[2],
    (m, s) -> m.elevation_fn(s),
    10;
    initial_state = copy(result[1].env[:start_state]),
    update_belief = true,
)

path_plot = plot_simulated_path(result[1],
    path_states,
    path_observations;
    title = "Volcano Search Path",
    ground_truth_fn = (m, s) -> m.elevation_fn(s),
    heatmap_resolution = 180,
    marker_size = 3,
    plot_size = (1200, 900),
    save_path = "/home/shashank/cbase/secondary/jbase/VulcanJ/examples/res/volcano_search_path.png"
)

ergodic_gp = get_initial_gp(result[1], copy(result[1].env[:start_state]))
ergodic_result = one_shot_ergodic_planner(
    result[1],
    ergodic_gp,
    100;
    initial_state = copy(result[1].env[:start_state]),
    rng = result[2].solver.rng,
    max_speed = result[1].env[:step_size],
    observe_fn = (m, s) -> m.elevation_fn(s),
)

println("Ergodic target density statistics: ", ergodic_result.target_density_stats)

ergodic_path_plot = plot_simulated_path(result[1],
    ergodic_result.states,
    ergodic_result.observations;
    title = "Ergodic Search Path on Ground Truth",
    ground_truth_fn = (m, s) -> m.elevation_fn(s),
    heatmap_resolution = 180,
    marker_size = 2,
    plot_size = (1200, 900),
    save_path = "/home/shashank/cbase/secondary/jbase/VulcanJ/examples/res/ergodic_search_path.png"
)

ergodic_information_path_plot = plot_information_reward_path(result[1],
    ergodic_result;
    title = "Ergodic Information Path",
    marker_size = 2,
    plot_size = (1200, 900),
    save_path = "/home/shashank/cbase/secondary/jbase/VulcanJ/examples/res/ergodic_information_path.png"
)
