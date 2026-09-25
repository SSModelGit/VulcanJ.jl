using GaussianProcesses: GPE, predict_f, predict_y

# Legacy ergodic calls retain latent-field quadrature; new planning predicts
# the noisy measurement used by the GP conditioning model.
function legacy_gp_information(mdp, model::GPE, state, quadrature_order::Integer)
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

function conditional_observation_distribution(problem, model::GPE, state)
    μ, σ² = predict_y(model, extract_location(state)')
    return Normal(first(vec(μ)), sqrt(max(first(vec(σ²)), eps())))
end

condition_environment_model(problem, model::GPE, state, observation) =
    add_obs_to_gp(state, observation, model)

# A one-channel simulator records measurements as a length-one vector.
condition_environment_model(problem, model::GPE, state, observation::AbstractVector) =
    add_obs_to_gp(state, only(observation), model)

function add_obs_to_gp(state, observation::Real, model::GPE)
    location = extract_location(state)
    return GPE(hcat(model.x, location'), vcat(model.y, Float64(observation)),
               model.mean, model.kernel)
end

information_gain(::Val{:mutual_information}, problem, prior::GPE, posterior::GPE,
                 state, observation) = compute_kl_reward(prior, posterior, problem)

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


initialize_gp_belief(problem, state) = initial_environment_model(problem, state)
