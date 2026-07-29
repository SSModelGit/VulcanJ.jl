using Distributions: Normal
using GaussianProcesses: GPE, MeanZero, SE, predict_f

struct DefaultGPMDP <: MDP{Int, Symbol} end

POMDPs.statetype(::DefaultGPMDP) = Int
POMDPs.actiontype(::DefaultGPMDP) = Symbol
POMDPs.actions(::DefaultGPMDP, ::Int) = (:stay,)
POMDPs.discount(::DefaultGPMDP) = 1.0
POMDPs.isterminal(::DefaultGPMDP, ::Int) = false
POMDPs.gen(::DefaultGPMDP, state::Int, ::Symbol, ::AbstractRNG) =
    (sp = state, r = 0.0)

VulcanJ.get_failure_prob(::DefaultGPMDP, ::Int, ::Symbol) = 0.0
VulcanJ.horizon(::DefaultGPMDP) = 3
VulcanJ.cellsites(::DefaultGPMDP) = (0, 1)

function VulcanJ.get_initial_gp(::DefaultGPMDP, ::Int)
    return GPE([0.0;;], [0.0], MeanZero(), SE([0.0], 0.0))
end

function VulcanJ.add_obs_to_gp(state::Int, observation::Real, gp::GPE)
    X = reshape([Float64(state)], 1, 1)
    return GPE(hcat(gp.x, X), vcat(gp.y, Float64(observation)), gp.mean, gp.kernel)
end

function VulcanJ.posterior_phenomenon_prob(::DefaultGPMDP, gp::GPE, site::Int)
    μ, σ² = predict_f(gp, reshape([Float64(site)], 1, 1))
    z = first(μ) / sqrt(max(first(σ²), eps()))
    return 1 / (1 + exp(-z))
end

@testset "Built-in Gaussian-process environment model" begin
    mdp = DefaultGPMDP()
    gp = initial_environment_model(mdp, 0)

    @test gp isa GPE
    @test initialize_gp_belief(mdp, 0) isa GPE
    @test conditional_observation_distribution(mdp, gp, 0) isa Normal
    @test condition_environment_model(mdp, gp, 0, 0.5) isa GPE
    @test expected_information_gain(mdp, gp, 0, 3) >= 0.0

    solver = RiskBoundedInfoMCTS(
        lookahead = 1,
        time_budget = 0.001,
        quad_order = 3,
        risk_budget = 1.0,
        alpha = 0.0,
        reference_reward = 1.0,
        rng = MersenneTwister(9),
    )
    @test action(solve(solver, mdp), 0) == :stay
end
