using Distributions: Normal, AbstractMvNormal, mean, std, cov
using FastGaussQuadrature: gausshermite
using LinearAlgebra: eigen, Symmetric, Diagonal

"""Paired locations and observations from the supplied state, in temporal order."""
function observation_history end

"""Timestamp of the supplied planning state, in mission-step units."""
function state_time end

performance_alpha(t, T) = t / T
fixed_alpha(t, T) = 0.0

"""Return the supplied problem bound to a branch posterior, without mutating the caller.

Its ordinary POMDPs generator produces hypothetical states and appends predicted
measurements to their observation history. `rng` also drives predictive callbacks.
"""
function generative_problem end

function generated_step(planner, problem, model, state, action, rng)
    predictive = generative_problem(problem, model, rng)
    sp = POMDPs.@gen(:sp)(predictive, state, action, rng)
    observation = last(collect(observation_history(planner, problem, sp))).observation
    return sp, observation
end

"""Information for this observation, evaluated from the prior and conditioned model."""
function information_gain end

function observation_outcomes(distribution::Normal, order)
    abscissae, weights = gausshermite(order)
    return [(observation = mean(distribution) + sqrt(2) * std(distribution) * x,
             weight = w / sqrt(π)) for (x, w) in zip(abscissae, weights)]
end

# Cartesian product of the scalar Gauss–Hermite rule in Gaussian coordinates.
function gaussian_observation_outcomes(μ, Σ, order)
    all(iszero, Σ) && return [(observation=copy(μ), weight=1.0)]
    decomposition = eigen(Symmetric(Matrix(Σ)))
    root = decomposition.vectors * Diagonal(sqrt.(max.(decomposition.values, 0.0)))
    abscissae, weights = gausshermite(order)
    outcomes = NamedTuple[]
    for indices in Iterators.product(ntuple(_ -> eachindex(abscissae), length(μ))...)
        y = μ + sqrt(2) * root * [abscissae[i] for i in indices]
        w = prod(weights[i] / sqrt(π) for i in indices)
        push!(outcomes, (observation=y, weight=w))
    end
    return outcomes
end

observation_outcomes(distribution::AbstractMvNormal, order) =
    gaussian_observation_outcomes(mean(distribution), cov(distribution), order)

function expected_information_gain(objective, problem, model, state, order)
    distribution = conditional_observation_distribution(problem, model, state)
    return sum(observation_outcomes(distribution, order)) do outcome
        posterior = condition_environment_model(problem, model, state, outcome.observation)
        outcome.weight * information_gain(objective, problem, model, posterior,
                                          state, outcome.observation)
    end
end

function conditional_observation_distribution end
function condition_environment_model end
initial_environment_model(problem, state) = get_initial_gp(problem, state)

next_state(problem, state, action, rng::AbstractRNG) = POMDPs.@gen(:sp)(problem, state, action, rng)

extract_location(state::AbstractMatrix) = Float64.(state)
extract_location(state::AbstractVector) = reshape(Float64.(state), 1, :)
extract_location(state::Real) = reshape([Float64(state)], 1, :)

function kl_divergence(p_post::Real, p_prior::Real)
    q = clamp(float(p_prior), eps(), 1 - eps())
    p = clamp(float(p_post), eps(), 1 - eps())
    return p * log(p / q) + (1 - p) * log((1 - p) / (1 - q))
end

###
# Generics
###

get_failure_prob(problem, state, action) = 0.0
function get_initial_gp end
function posterior_phenomenon_prob end
function cellsites end
function horizon end
function add_obs_to_gp end
