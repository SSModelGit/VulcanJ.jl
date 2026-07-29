using Distributions: Bernoulli
using POMDPs
using Random: AbstractRNG, MersenneTwister
using Test
using VulcanJ

struct ToyEnvironmentModel
    observations::Int
end

struct ToyInformationMDP <: MDP{Int, Symbol} end

POMDPs.statetype(::ToyInformationMDP) = Int
POMDPs.actiontype(::ToyInformationMDP) = Symbol
POMDPs.actions(::ToyInformationMDP, ::Int) = (:left, :right)
POMDPs.discount(::ToyInformationMDP) = 1.0
POMDPs.isterminal(::ToyInformationMDP, ::Int) = false

function POMDPs.gen(::ToyInformationMDP, state::Int, action::Symbol, ::AbstractRNG)
    step = action === :left ? -1 : 1
    return (sp = state + step, r = 0.0)
end

VulcanJ.initial_environment_model(::ToyInformationMDP, ::Int) =
    ToyEnvironmentModel(0)

VulcanJ.expected_information_gain(
    ::ToyInformationMDP,
    model::ToyEnvironmentModel,
    state::Int,
    ::Integer,
) = 2.0 + 0.1 * model.observations + 0.01 * abs(state)

VulcanJ.conditional_observation_distribution(
    ::ToyInformationMDP,
    ::ToyEnvironmentModel,
    ::Int,
) = Bernoulli(0.5)

VulcanJ.condition_environment_model(
    ::ToyInformationMDP,
    model::ToyEnvironmentModel,
    ::Int,
    ::Any,
) = ToyEnvironmentModel(model.observations + 1)

VulcanJ.get_failure_prob(::ToyInformationMDP, ::Int, ::Symbol) = 0.0
VulcanJ.horizon(::ToyInformationMDP) = 4

struct OrderingEnvironmentModel
    calls::Vector{Tuple{Symbol, Int}}
end

struct OrderingMDP <: MDP{Int, Symbol} end

POMDPs.statetype(::OrderingMDP) = Int
POMDPs.actiontype(::OrderingMDP) = Symbol
POMDPs.actions(::OrderingMDP, ::Int) = (:advance,)
POMDPs.discount(::OrderingMDP) = 1.0
POMDPs.isterminal(::OrderingMDP, ::Int) = false
POMDPs.gen(::OrderingMDP, state::Int, ::Symbol, ::AbstractRNG) =
    (sp = state + 1, r = 0.0)

function VulcanJ.expected_information_gain(
    ::OrderingMDP,
    model::OrderingEnvironmentModel,
    state::Int,
    ::Integer,
)
    push!(model.calls, (:reward, state))
    return Float64(state)
end

function VulcanJ.conditional_observation_distribution(
    ::OrderingMDP,
    model::OrderingEnvironmentModel,
    state::Int,
)
    push!(model.calls, (:distribution, state))
    return Bernoulli(0.5)
end

function VulcanJ.condition_environment_model(
    ::OrderingMDP,
    model::OrderingEnvironmentModel,
    state::Int,
    ::Any,
)
    push!(model.calls, (:condition, state))
    return model
end

VulcanJ.get_failure_prob(::OrderingMDP, ::Int, ::Symbol) = 0.0
VulcanJ.horizon(::OrderingMDP) = 4

function toy_solver(; time_budget = 0.001)
    return RiskBoundedInfoMCTS(
        lookahead = 1,
        time_budget = time_budget,
        quad_order = 3,
        risk_budget = 1.0,
        alpha = 0.0,
        reference_reward = 1.0,
        rng = MersenneTwister(7),
    )
end

@testset "Black-box environment-model interface" begin
    mdp = ToyInformationMDP()
    model = initial_environment_model(mdp, 0)

    @test model == ToyEnvironmentModel(0)
    @test expected_information_gain(mdp, model, 0, 3) == 2.0

    distribution = conditional_observation_distribution(mdp, model, 0)
    observation = rand(MersenneTwister(3), distribution)
    @test observation in (false, true)
    @test condition_environment_model(mdp, model, 0, observation) ==
          ToyEnvironmentModel(1)

    policy = solve(toy_solver(), mdp)
    @test action(policy, 0) in (:left, :right)
    @test !isempty(policy.node_models)

    replacement = ToyEnvironmentModel(8)
    set_environment_model!(policy, 0, replacement)
    @test policy.node_models[VulcanJ.initialize_nodekey(0)] == replacement
    @test !haskey(policy.best_action, 0)
end

@testset "Rollouts observe successor states" begin
    mdp = OrderingMDP()
    solver = RiskBoundedInfoMCTS(
        lookahead = 2,
        time_budget = 0.0,
        quad_order = 3,
        risk_budget = 1.0,
        alpha = 0.0,
        reference_reward = 1.0,
        rng = MersenneTwister(11),
    )
    policy = solve(solver, mdp)
    model = OrderingEnvironmentModel(Tuple{Symbol, Int}[])

    reward = sample_rollout(
        policy,
        VulcanJ.initialize_nodekey(0),
        model,
        0,
        solver.lookahead,
        0.0,
        0.0,
        1.0,
    )

    @test reward == 3.0
    @test model.calls == [
        (:reward, 1),
        (:distribution, 1),
        (:condition, 1),
        (:reward, 2),
        (:distribution, 2),
        (:condition, 2),
    ]
    @test all(state != 0 for (_, state) in model.calls)
end

@testset "Simulation conditions only through the supplied interface" begin
    mdp = ToyInformationMDP()
    policy = solve(toy_solver(), mdp)
    states, observations =
        simulate_info_path(mdp, policy, (_, state) -> isodd(state), 2; initial_state = 0)

    @test length(states) == 3
    @test length(observations) == 3
    @test any(model.observations == 3 for model in values(policy.node_models))
end

include("default_gp.jl")
