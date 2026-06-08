using Random
using LinearAlgebra
using SpecialFunctions: erf
using Distributions: Normal
using GaussianProcesses
using POMDPs
using POMDPTools

using VulcanJ

const VOLCANO_FIELD_SIZE = (10, 10)
const VOLCANO_ELEVATION_THRESHOLD = 7.5

const VOLCANO_PHENOMENON_SITES = [
    reshape([2.0, 2.0], 2, 1),
    reshape([2.0, 5.0], 2, 1),
    reshape([2.0, 8.0], 2, 1),
    reshape([5.0, 2.0], 2, 1),
    reshape([5.0, 5.0], 2, 1),
    reshape([5.0, 8.0], 2, 1),
    reshape([8.0, 2.0], 2, 1),
    reshape([8.0, 5.0], 2, 1),
    reshape([8.0, 8.0], 2, 1),
]

const VOLCANO_PRIOR_SITES = hcat(
    [1.5, 1.5],
    [1.5, 8.5],
    [5.0, 5.0],
    [8.5, 1.5],
    [8.5, 8.5],
)

Base.@kwdef struct VolcanoSearchMDP <: POMDPs.MDP
    field_size::Tuple{Int, Int} = VOLCANO_FIELD_SIZE
    horizon_steps::Int = 12
    discount_factor::Float64 = 0.97
    observation_noise::Float64 = 0.35
    step_size::Float64 = 1.0
    start_state::Vector{Float64} = [2.0, 2.0]
    peak_center::Vector{Float64} = [5.0, 5.0]
    caldera_center::Vector{Float64} = [5.0, 5.0]
    caldera_radius::Float64 = 2.2
    caldera_width::Float64 = 0.75
    peak_height::Float64 = 10.0
    caldera_bonus::Float64 = 7.0
    ridge_center::Vector{Float64} = [7.5, 3.0]
    ridge_height::Float64 = 2.0
    danger_center::Vector{Float64} = [5.0, 5.0]
    danger_radius::Float64 = 2.2
    danger_width::Float64 = 0.60
    base_risk::Float64 = 0.02
    peak_risk::Float64 = 0.60
    phenomenon_sites::Vector{Matrix{Float64}} = VOLCANO_PHENOMENON_SITES
end

POMDPs.statetype(::VolcanoSearchMDP) = Vector{Float64}
POMDPs.actiontype(::VolcanoSearchMDP) = Symbol
POMDPs.horizon(mdp::VolcanoSearchMDP) = mdp.horizon_steps
POMDPs.discount(mdp::VolcanoSearchMDP) = mdp.discount_factor
POMDPs.initialstate(mdp::VolcanoSearchMDP) = Deterministic(copy(mdp.start_state))

function POMDPs.actions(::VolcanoSearchMDP, ::Vector{Float64})
    [:wait, :north, :south, :east, :west, :northeast, :northwest, :southeast, :southwest]
end

function POMDPs.isterminal(mdp::VolcanoSearchMDP, s::Vector{Float64})
    x, y = s
    x < 1 || y < 1 || x > mdp.field_size[1] || y > mdp.field_size[2]
end

function volcano_step(mdp::VolcanoSearchMDP, s::Vector{Float64}, a::Symbol)
    dx, dy = if a === :north
        (0.0, mdp.step_size)
    elseif a === :south
        (0.0, -mdp.step_size)
    elseif a === :east
        (mdp.step_size, 0.0)
    elseif a === :west
        (-mdp.step_size, 0.0)
    elseif a === :northeast
        (mdp.step_size, mdp.step_size)
    elseif a === :northwest
        (-mdp.step_size, mdp.step_size)
    elseif a === :southeast
        (mdp.step_size, -mdp.step_size)
    elseif a === :southwest
        (-mdp.step_size, -mdp.step_size)
    else
        (0.0, 0.0)
    end

    return [
        clamp(s[1] + dx, 1.0, float(mdp.field_size[1])),
        clamp(s[2] + dy, 1.0, float(mdp.field_size[2])),
    ]
end

function volcano_elevation(mdp::VolcanoSearchMDP, s::AbstractVector{<:Real})
    x = float(s[1])
    y = float(s[2])

    peak = mdp.peak_height * exp(-0.16 * ((x - mdp.peak_center[1])^2 + (y - mdp.peak_center[2])^2))
    caldera_distance = hypot(x - mdp.caldera_center[1], y - mdp.caldera_center[2])
    caldera = mdp.caldera_bonus * exp(-((caldera_distance - mdp.caldera_radius)^2) / (2 * mdp.caldera_width^2))
    ridge = mdp.ridge_height * exp(-0.12 * ((x - mdp.ridge_center[1])^2 + (y - mdp.ridge_center[2])^2))

    return peak + caldera + ridge
end

volcano_observation(mdp::VolcanoSearchMDP, s::Vector{Float64}, rng::AbstractRNG) =
    volcano_elevation(mdp, s) + mdp.observation_noise * randn(rng)

function volcano_failure_probability(mdp::VolcanoSearchMDP, s::Vector{Float64}, a::Symbol)
    sp = volcano_step(mdp, s, a)
    dist_to_hazard = hypot(sp[1] - mdp.danger_center[1], sp[2] - mdp.danger_center[2])
    ring_risk = exp(-((dist_to_hazard - mdp.danger_radius)^2) / (2 * mdp.danger_width^2))
    boundary_penalty = sp[1] <= 1.0 || sp[2] <= 1.0 || sp[1] >= float(mdp.field_size[1]) || sp[2] >= float(mdp.field_size[2]) ? 0.10 : 0.0
    return clamp(mdp.base_risk + mdp.peak_risk * ring_risk + boundary_penalty, 0.0, 0.95)
end

POMDPs.observation(mdp::VolcanoSearchMDP, s::Vector{Float64}, a::Symbol, sp::Vector{Float64}) =
    Normal(volcano_elevation(mdp, sp), mdp.observation_noise)

function POMDPs.gen(mdp::VolcanoSearchMDP, s::Vector{Float64}, a::Symbol, rng::AbstractRNG)
    sp = volcano_step(mdp, s, a)
    if rand(rng) < volcano_failure_probability(mdp, s, a)
        sp = copy(s)
    end

    o = volcano_observation(mdp, sp, rng)
    r = volcano_elevation(mdp, sp)
    return (sp = sp, o = o, r = r)
end

VulcanJ.phenomenon_indices(mdp::VolcanoSearchMDP) = eachindex(mdp.phenomenon_sites)

function VulcanJ.posterior_phenomenon_prob(gp::GPE, idx::Int)
    site = VOLCANO_PHENOMENON_SITES[idx]
    μ, Σ = predict_f(gp, site)
    μv = first(vec(μ))
    σ² = max(first(vec(Σ)), eps())
    return 0.5 * (1 - erf((VOLCANO_ELEVATION_THRESHOLD - μv) / sqrt(2 * σ²)))
end

function VulcanJ.add_obs_to_gp(X::Matrix{Float64}, y::Float64, gp::GPE)
    y_new = vcat(gp.y, y)
    x_new = hcat(gp.x, X)
    return GPE(x_new, y_new, gp.mean, gp.kernel)
end

VulcanJ.get_failure_prob(mdp::VolcanoSearchMDP, s::Vector{Float64}, a::Symbol) = volcano_failure_probability(mdp, s, a)

function VulcanJ.get_initial_gp(mdp::VolcanoSearchMDP, ::Vector{Float64})
    X0 = VOLCANO_PRIOR_SITES
    y0 = [volcano_elevation(mdp, X0[:, i]) for i in 1:size(X0, 2)]
    return GPE(X0, y0, MeanZero(), SE(zeros(2), 0.0))
end

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
    s0 = copy(mdp.start_state)
    a0 = action(policy, s0)
    return (mdp = mdp, policy = policy, state = s0, action = a0)
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = run_volcano_search_demo()
    println("Selected action: ", result.action)
end
