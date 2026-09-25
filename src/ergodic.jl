"""Extend the ergodic configuration and supply `observation_history` to reuse its policy."""
abstract type AbstractErgodicSolver <: Solver end

@with_kw struct ErgodicSolver <: AbstractErgodicSolver
    lookahead::Int = 30
    quad_order::Int = 5
    rng::AbstractRNG = GLOBAL_RNG
    backend::Symbol = :kernel
    fourier_order::Int = 5
    revisit_penalty::Float64 = 0.05
    density_bandwidth::Union{Float64,Nothing} = nothing
    kernel_bandwidth::Union{Float64,Nothing} = nothing
    dt::Float64 = 1.0
    optimizer_iters::Int = 150
    learning_rate::Float64 = 0.25
    momentum::Float64 = 0.85
    control_weight::Float64 = 1e-3
    boundary_weight::Float64 = 10.0
    max_speed::Union{Float64,Nothing} = nothing
    line_search_steps::Int = 8
    line_search_decay::Float64 = 0.5
end

@with_kw mutable struct ErgodicPolicy{P,O,S<:AbstractErgodicSolver} <: Policy
    solver::S
    problem::P
    objective::O
    model::Any = nothing
    root_state::Any = nothing
    history::Any = nothing
    remaining_steps::Int = solver.lookahead
    result::Any = nothing
    selected_action::Any = nothing
    planned::Bool = false
end

POMDPs.solve(solver::AbstractErgodicSolver, problem::Union{MDP,POMDP};
    objective = Val(:mutual_information)) = ErgodicPolicy(; solver, problem, objective)

function set_environment_model!(policy::ErgodicPolicy, state, model;
    remaining_steps = policy.solver.lookahead)
    policy.root_state, policy.model = state, model
    policy.history = collect(observation_history(policy.solver, policy.problem, state))
    policy.remaining_steps = remaining_steps
    policy.result = nothing
    policy.selected_action = nothing
    policy.planned = false
    return policy
end

ergodic_options(solver) = (; (key => getproperty(solver, key)
    for key in fieldnames(ErgodicSolver) if key != :lookahead)...)

function POMDPs.action(policy::ErgodicPolicy, state)
    if !isequal(state, policy.root_state) || isnothing(policy.model)
        set_environment_model!(policy, state, initial_environment_model(policy.problem, state))
    end
    if !policy.planned
        n = min(policy.solver.lookahead, policy.remaining_steps)
        if n > 0 && !isterminal(policy.problem, state)
            policy.result = ergodic_reference_path(policy.problem, policy.model, n;
                dynamics_problem=generative_problem(policy.problem, policy.model, policy.solver.rng),
                initial_state = policy.solver.backend == :kernel ? extract_location(state) : state,
                objective = policy.objective,
                history = [point_tuple(record.location) for record in policy.history],
                infer_actions = false, ergodic_options(policy.solver)...)
            if length(policy.result.states) > 1
                policy.selected_action = policy.solver.backend == :kernel ?
                    select_action(generative_problem(policy.problem, policy.model, policy.solver.rng), state, policy.result.states[2],
                                  policy.result.controls[1, :], policy.solver.rng) :
                    first(policy.result.actions)
            end
        end
        policy.planned = true
    end
    return policy.selected_action
end

select_action(problem::Union{MDP,POMDP}, state, target, control, rng::AbstractRNG) =
    select_action(problem, state, target, rng)

select_action(::MDP{S,A}, state, target, control, rng::AbstractRNG) where {S,A<:AbstractVector} =
    convert(A, control)

select_action(::POMDP{S,A}, state, target, control, rng::AbstractRNG) where {S,A<:AbstractVector} =
    convert(A, control)

"""Optimize a direct density target; returns path, controls, loss, and kernel metric."""
function plan_trajectory(solver::AbstractErgodicSolver, start, sites, density, bounds, horizon;
    history = (), initial_controls = nothing)
    return optimize_ergodic_trajectory(start, sites, density, bounds, horizon;
        solver.density_bandwidth, solver.kernel_bandwidth, solver.dt,
        solver.optimizer_iters, solver.learning_rate, solver.momentum,
        solver.control_weight, solver.boundary_weight, solver.max_speed,
        solver.line_search_steps, solver.line_search_decay, history, initial_controls)
end

# This objective preserves the existing consumer calculation during migration.
struct LegacyInformation end
expected_information_gain(::LegacyInformation, problem, model, state, order) =
    expected_information_gain(Val(:mutual_information), problem, model, state, order)
expected_information_gain(::LegacyInformation, problem, model::GPE, state, order) =
    legacy_gp_information(problem, model, state, order)

"""
    plan_trajectory(solver::AbstractErgodicSolver, problem, state, model, horizon; ...)

Optimize a reference once, then generate a predicted problem-state trajectory
through the supplied POMDPs generator. The target density stays fixed during
optimization. Hypothetical observations condition only the local branch model.
The optimizer's reference positions are returned separately as `reference_states`.
"""
function plan_trajectory(solver::AbstractErgodicSolver, problem::Union{MDP,POMDP}, state, model, horizon;
    objective=Val(:mutual_information), remaining_steps=horizon,
    history=observation_history(solver, problem, state))
    n = isterminal(problem, state) ? 0 : min(horizon, remaining_steps)
    reference = ergodic_reference_path(problem, model, n;
        dynamics_problem=generative_problem(problem, model, solver.rng),
        initial_state=solver.backend == :kernel ? extract_location(state) : state,
        objective, history=[point_tuple(record.location) for record in history],
        infer_actions=false, ergodic_options(solver)...)
    states, taken, observations = Any[state], Any[], Any[]
    for step in 1:length(reference.states)-1
        isterminal(problem, state) && break
        a = solver.backend == :kernel ?
            select_action(generative_problem(problem, model, solver.rng), state, reference.states[step+1],
                          reference.controls[step, :], solver.rng) : reference.actions[step]
        sp, observation = generated_step(solver, problem, model, state, a, solver.rng)
        model = condition_environment_model(problem, model, sp, observation)
        state = sp
        push!(states, state)
        push!(taken, a)
        push!(observations, observation)
    end
    return merge(reference, (;states, actions=taken, observations,
                              reference_states=reference.states))
end

# Only the deprecated interface retains reference-only states and observe_fn.
function plan_trajectory(::LegacyInformation, solver::AbstractErgodicSolver,
    problem::MDP, state, model, horizon; observe_fn=nothing)
    return ergodic_reference_path(problem, model, horizon;
        initial_state=state, objective=LegacyInformation(), observe_fn,
        ergodic_options(solver)...)
end

function one_shot_ergodic_planner(problem::MDP, model, n_steps::Integer;
    initial_state = nothing, observe_fn = nothing, kwargs...)
    Base.depwarn("Use plan_trajectory(::ErgodicSolver, problem, state, model, horizon).",
                 :one_shot_ergodic_planner)
    solver = ErgodicSolver(; lookahead = n_steps, kwargs...)
    state = isnothing(initial_state) ? rand(solver.rng, initialstate(problem)) : initial_state
    return plan_trajectory(LegacyInformation(), solver, problem, state, model, n_steps; observe_fn)
end

function kernel_ergodic_trajectory(start, sites, density, bounds, n_steps::Integer; kwargs...)
    Base.depwarn("Use plan_trajectory(::ErgodicSolver, start, sites, density, bounds, horizon).",
                 :kernel_ergodic_trajectory)
    # History and initial controls belong to the new overload, not the legacy call.
    solver = ErgodicSolver(; lookahead = n_steps, kwargs...)
    return plan_trajectory(solver, start, sites, density, bounds, n_steps)
end
