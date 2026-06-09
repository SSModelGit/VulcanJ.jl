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

VOLCANO_ENV = Dict{Symbol, Any}(
    :field_size => (10, 10),
    :horizon_steps => 12,
    :discount_factor => 0.97,
    :step_size => 1.0,
    :start_state => [2.0 2.0],
    :peak_centers => [
        [5.0 5.0],
        [7.5 3.0],
        [3.0 7.5]
    ],
    :peak_heights => [10.0, 4.0, 3.5],
    :peak_spread => 0.16,
    :caldera_center => [5.0 5.0],
    :caldera_radius => 2.2,
    :caldera_width => 0.75,
    :caldera_bonus => 7.0,
    :base_risk => 0.02,
    :peak_risk => 0.60,
    :risk_scale => 1.25,
    :safe_elevation => 7.5,
    :elevation_threshold => 7.5,
    :prior_sites => [
        [1.5 1.5],
        [1.5 8.5],
        [5.0 5.0],
        [8.5 1.5],
        [8.5 8.5],
    ],
    :phenomenon_sites => [
        [2.0 2.0],
        [2.0 5.0],
        [2.0 8.0],
        [5.0 2.0],
        [5.0 5.0],
        [5.0 8.0],
        [8.0 2.0],
        [8.0 5.0],
        [8.0 8.0],
    ],
)

function make_volcanic_elevation(env::Dict{Symbol, Any})
    peak_centers = env[:peak_centers]
    peak_heights = env[:peak_heights]
    peak_spread = float(env[:peak_spread])
    caldera_center = env[:caldera_center]
    caldera_radius = float(env[:caldera_radius])
    caldera_width = float(env[:caldera_width])
    caldera_bonus = float(env[:caldera_bonus])

    function elevation(s)
        x = float(s[1])
        y = float(s[2])

        total = 0.0
        for (i, c) in enumerate(peak_centers)
            cx = float(c[1])
            cy = float(c[2])
            h = peak_heights[i]
            total += h * exp(-peak_spread * ((x - cx)^2 + (y - cy)^2))
        end

        calx = float(caldera_center[1])
        caly = float(caldera_center[2])
        caldera_distance = hypot(x - calx, y - caly)
        caldera = caldera_bonus * exp(-((caldera_distance - caldera_radius)^2) / (2 * caldera_width^2))
        return total + caldera
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
    return GPE(X0, y0, MeanZero(), SE(zeros(2), 0.0))
end

# How do we add observations?
function VulcanJ.add_obs_to_gp(X::Matrix, y::Float64, gp::GPE)
    y_new = vcat(gp.y, y)
    x_new = hcat(gp.x, X)
    return GPE(x_new, y_new, gp.mean, gp.kernel)
end

# Define likelihood of failure / collision / risky behavior

# helper to check out of bounds
function boundary_violation(sp, env)
    let sp1=sp[1], sp2=sp[2], limx=env[:field_size][1], limy=env[:field_size][2]
        sp1 <= 1.0 || sp2 <= 1.0 || sp1 >= float(limx) || sp2 >= float(limy)
    end
end

# 
function VulcanJ.get_failure_prob(mdp::VolcanoSearchMDP, s::Matrix, a::Symbol)
    let sp = volcano_step(mdp, s, a), elev = mdp.elevation_fn(sp),
        safe_elevation = get(mdp.env, :safe_elevation, mdp.env[:elevation_threshold]),
        risk_scale = mdp.env[:risk_scale],
        base_risk = mdp.env[:base_risk], peak_risk = mdp.env[:peak_risk]

        boundary_penalty = boundary_violation(sp, mdp.env) ? 0.10 : 0.0
        elevation_risk = 1 / (1 + exp(-(elev - safe_elevation) / risk_scale))
        return clamp(base_risk + peak_risk * elevation_risk + boundary_penalty, 0.0, 0.95)
    end
end

# Compute the likelihood of the phenomenon of interest at a given site existing based on the current GP
function VulcanJ.posterior_phenomenon_prob(gp::GPE, idx::Int)
    site = VOLCANO_ENV[:phenomenon_sites][idx]
    μ, Σ = predict_f(gp, Matrix(site'))
    μv = first(vec(μ))
    σ² = max(first(vec(Σ)), eps())
    threshold = VOLCANO_ENV[:elevation_threshold]
    return 0.5 * (1 - erf((threshold - μv) / sqrt(2 * σ²)))
end

# Housekeeping items
VulcanJ.horizon(mdp::VolcanoSearchMDP) = mdp.env[:horizon_steps] # MCTS horizon
VulcanJ.phenomenon_indices(mdp::VolcanoSearchMDP) = eachindex(mdp.env[:phenomenon_sites]) # what?

function volcano_search_solver(; rng = MersenneTwister(7))
    mdp = VolcanoSearchMDP()
    solver = RiskBoundedInfoMCTS(
        lookahead = 6,
        time_budget = 0.25,
        quad_order = 5,
        risk_budget = 0.45,
        alpha = 0.55,
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

if abspath(PROGRAM_FILE) == @__FILE__
    result = run_volcano_search_demo()
    println("Selected action: ", result.action)
end
