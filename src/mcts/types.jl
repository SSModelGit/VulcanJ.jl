"""Extend the MCTS configuration and supply `observation_history` to reuse its policy/search."""
abstract type AbstractInfoMCTS <: Solver end

struct RiskBoundedInfoMCTS{F} <: AbstractInfoMCTS
    lookahead::Int
    time_budget::Float64
    quad_order::Int
    risk_budget::Float64
    alpha_schedule::F
    reference_reward::Float64
    rng::AbstractRNG
end

function RiskBoundedInfoMCTS(; lookahead, time_budget, quad_order=5,
    risk_budget=Inf, alpha_schedule=performance_alpha, reference_reward=1.0,
    rng=GLOBAL_RNG, alpha=nothing)
    if !isnothing(alpha)
        Base.depwarn("The alpha keyword is deprecated; use alpha_schedule=(t,T)->value for a constant schedule.",
                     :RiskBoundedInfoMCTS)
        alpha_schedule = (t, T) -> alpha
    end
    return RiskBoundedInfoMCTS(lookahead, time_budget, quad_order, risk_budget,
                              alpha_schedule, reference_reward, rng)
end

mutable struct ActionNode
    action::Any
    risk::Float64
    visits::Int
    attempts::Int # Includes unsuccessful exploration; visits count successful rollouts.
    value::Float64
    children::Vector{Any}
    admissible::Bool
end

mutable struct TreeNode
    state::Any
    model::Any
    depth::Int
    sequence_risk::Float64
    sequence_information::Float64
    visits::Int
    branches::Vector{ActionNode}
    best::Union{Nothing,Int}
    value::Float64
    terminal::Bool
end

@with_kw mutable struct RiskBoundedInfoPolicy{P,O,S<:AbstractInfoMCTS} <: Policy
    solver::S
    problem::P
    objective::O
    root::Union{Nothing,TreeNode} = nothing
    model::Any = nothing
    root_state::Any = nothing
    history::Any = nothing
    remaining_steps::Int = solver.lookahead
    risk_used::Float64 = 0.0
    reference_reward::Float64 = solver.reference_reward
    alpha::Float64 = 0.0
    mission_start_time::Float64 = 0.0
end

POMDPs.solve(solver::AbstractInfoMCTS, problem::Union{MDP,POMDP};
    objective = Val(:mutual_information)) =
    RiskBoundedInfoPolicy(; solver, problem, objective)

function set_environment_model!(policy::RiskBoundedInfoPolicy, state, model;
    remaining_steps = policy.solver.lookahead, risk_used = 0.0,
    reference_reward = policy.solver.reference_reward, mission_start_time = 0,
    alpha = policy.solver.alpha_schedule(
        state_time(policy.problem, state) - mission_start_time,
        state_time(policy.problem, state) - mission_start_time + remaining_steps))
    policy.root = nothing
    policy.root_state = state
    policy.model = model
    policy.history = observation_history(policy.solver, policy.problem, state)
    policy.remaining_steps = remaining_steps
    policy.risk_used = risk_used
    policy.reference_reward = reference_reward
    policy.alpha = alpha
    policy.mission_start_time = mission_start_time
    return policy
end

function POMDPs.action(policy::RiskBoundedInfoPolicy, state)
    if !isequal(state, policy.root_state) || isnothing(policy.model)
        set_environment_model!(policy, state, initial_environment_model(policy.problem, state))
    end
    isnothing(policy.root) && build_search_tree(policy, state)
    root = policy.root
    return isnothing(root.best) ? nothing : root.branches[root.best].action
end
