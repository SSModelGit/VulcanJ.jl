# ============================================================================
# Environment-model interface
# ============================================================================

using Distributions: Normal
using FastGaussQuadrature: gausshermite
using GaussianProcesses: GPE, MeanZero, SE, predict_f

"""
    initial_environment_model(mdp, state)

Return the environment model used to begin planning at `state`.

VulcanJ stores and passes this value without inspecting it. By default this
uses VulcanJ's built-in Gaussian-process model initialization hook.
"""
initial_environment_model(mdp, state) = get_initial_gp(mdp, state)

"""
    expected_information_gain(mdp, model, state, quadrature_order)

Return the expected information obtained by observing `state` under `model`.

Environment-model integrations should specialize this function for their MDP
and model types. `GPE` models use VulcanJ's built-in Gauss-Hermite information
calculation.
"""
function expected_information_gain(::Any, model, ::Any, ::Integer)
    error(
        "The integration must implement " *
        "`expected_information_gain(mdp, model, state, quadrature_order)` " *
        "for $(typeof(model)).",
    )
end

function expected_information_gain(mdp, model::GPE, state, quadrature_order::Integer)
    quadrature_order > 0 || throw(ArgumentError("quadrature_order must be positive."))
    μ, σ² = gp_predict(model, state)
    abscissae, weights = gausshermite(quadrature_order)

    reward = 0.0
    for j in eachindex(abscissae)
        observation = sqrt(2 * max(σ², eps())) * abscissae[j] + μ
        posterior = condition_environment_model(mdp, model, state, observation)
        reward += (weights[j] / sqrt(π)) * compute_kl_reward(model, posterior, mdp)
    end
    return reward
end

"""
    conditional_observation_distribution(mdp, model, state)

Return the conditional distribution of an observation at `state`.

The returned object only needs to support `rand(rng, distribution)`. VulcanJ's
built-in `GPE` model returns its scalar Gaussian predictive distribution.
"""
function conditional_observation_distribution(::Any, model, ::Any)
    error(
        "The integration must implement " *
        "`conditional_observation_distribution(mdp, model, state)` " *
        "for $(typeof(model)).",
    )
end

function conditional_observation_distribution(::Any, model::GPE, state)
    μ, σ² = gp_predict(model, state)
    return Normal(μ, sqrt(max(σ², eps())))
end

"""
    condition_environment_model(mdp, model, state, observation)

Return the environment model conditioned on `observation` at `state`.

How, or whether, a custom model changes is entirely controlled by the
dispatched integration method. The built-in `GPE` implementation uses
`add_obs_to_gp`.
"""
function condition_environment_model(::Any, model, ::Any, ::Any)
    error(
        "The integration must implement " *
        "`condition_environment_model(mdp, model, state, observation)` " *
        "for $(typeof(model)).",
    )
end

condition_environment_model(::Any, model::GPE, state, observation) =
    add_obs_to_gp(state, observation, model)

# ============================================================================
# Built-in Gaussian-process environment model
# ============================================================================

function gp_predict(gp::GPE, state)
    location = extract_location(state)
    μ, σ² = predict_f(gp, location')
    return first(vec(μ)), first(vec(σ²))
end

function compute_kl_reward(gp_prior, gp_posterior, mdp)
    kl_sum = 0.0
    for site in cellsites(mdp)
        p_prior = posterior_phenomenon_prob(mdp, gp_prior, site)
        p_post = posterior_phenomenon_prob(mdp, gp_posterior, site)
        kl_sum += kl_divergence(p_post, p_prior)
    end
    return kl_sum
end

last_observation_location(gp) = reshape(gp.x[:, end:end], 1, :)

function gp_predictive_kl(gp_prior, gp_posterior)
    μ1, σ1² = predictive_moments(gp_prior)
    μ2, σ2² = predictive_moments(gp_posterior)
    σ1² = max(σ1², eps())
    σ2² = max(σ2², eps())
    return 0.5 * (log(σ2² / σ1²) + (σ1² + (μ1 - μ2)^2) / σ2² - 1)
end

function predictive_moments(gp)
    X = last_observation_location(gp)
    μ, Σ = predict_f(gp, X')
    return first(vec(μ)), first(vec(Σ))
end

function initial_gp_from_state(mdp, state)
    dim = observation_dim(mdp, state)
    return GPE(Matrix{Float64}(undef, dim, 0), Float64[], MeanZero(), SE(zeros(dim), 0.0))
end

observation_dim(mdp, state) = size(extract_location(state), 2)

"""GP-specific convenience name for `initial_environment_model`."""
initialize_gp_belief(mdp, state) = initial_environment_model(mdp, state)
