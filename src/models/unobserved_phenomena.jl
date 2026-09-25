using Distributions: MixtureModel, Bernoulli, components, probs, pdf, ccdf, UnivariateDistribution

"""An explicitly specified scalar event `observation > threshold`."""
struct ThresholdPresence{T<:Real}
    threshold::T
end
(p::ThresholdPresence)(value::Real, location) = value > p.threshold
(p::ThresholdPresence)(value::AbstractArray, location) = only(value) > p.threshold

presence_probability(presence, distribution, location, order) =
    sum(o.weight * presence(o.observation,location) for o in observation_outcomes(distribution,order))
presence_probability(p::ThresholdPresence, distribution::UnivariateDistribution, location, order) =
    ccdf(distribution,p.threshold)
presence_probability(p::ThresholdPresence, distribution::Normal, location, order) =
    iszero(std(distribution)) ? Float64(mean(distribution)>p.threshold) : ccdf(distribution,p.threshold)

"""
    UnobservedPhenomenaModel(problem, base_model, presence;
        sites=cellsites(problem), history=(), quadrature_order=5)

Bayesian phenomenon layer over an existing probabilistic environment model.
`presence(value, location)` gives the conditional probability of phenomenon
presence. Its expectation under the base model's predictive distribution constructs
the presence marginal for each spatial cell. `ThresholdPresence(c)` uses the
predictive tail probability when supported; other callbacks use observation
quadrature.

`sites` are representatives of nearest-site (Voronoi) cells in the environment.
Each measurement location belongs to its nearest representative in Euclidean
distance; ties use the first site. This spatial discretization is separate from
`quadrature_order`, which integrates predicted observation values. Multiple
sensor locations are rows of `extract_location(state)` and mark each sampled cell.

Under ideal local detection, a sampled cell is no longer unobserved. `history`
contains existing `(location, observation)` records; the supplied base model
already incorporates these observations and is not conditioned again.
Predictions and branch conditioning delegate to the base model at the actual
measurement coordinates, without snapping observations to cell representatives. Information is
the sum of Bernoulli KL changes in unobserved marginals, not joint spatial MI.
"""
struct UnobservedPhenomenaModel{M,F,S}
    base_model::M
    presence::F
    sites::S
    observed::BitVector
    probabilities::Vector{Float64}
    quadrature_order::Int
end

function phenomenon_probabilities(problem, model, presence, sites, observed, order)
    return [observed[i] ? 0.0 : presence_probability(presence,
        conditional_observation_distribution(problem, model, site), site, order)
        for (i, site) in enumerate(sites)]
end

# The nearest-site partition represents cells implicitly; no mesh is needed.
function observe_cells!(observed, sites, state)
    for location in eachrow(extract_location(state))
        cell = argmin([sum(abs2, vec(extract_location(site)) - location)
                       for site in sites])
        observed[cell] = true
    end
    return observed
end

function UnobservedPhenomenaModel(problem, base_model, presence;
    sites=cellsites(problem), history=(), quadrature_order=5)
    observed = falses(length(sites))
    for record in history
        observe_cells!(observed, sites, record.location)
    end
    probabilities = phenomenon_probabilities(problem, base_model, presence,
                                              sites, observed, quadrature_order)
    return UnobservedPhenomenaModel(base_model, presence, sites, observed,
                                   probabilities, quadrature_order)
end

conditional_observation_distribution(problem, model::UnobservedPhenomenaModel, state) =
    conditional_observation_distribution(problem, model.base_model, state)

function condition_environment_model(problem, model::UnobservedPhenomenaModel, state, observation)
    base_model = condition_environment_model(problem, model.base_model, state, observation)
    observed = copy(model.observed)
    observe_cells!(observed, model.sites, state)
    probabilities = phenomenon_probabilities(problem, base_model, model.presence,
        model.sites, observed, model.quadrature_order)
    return UnobservedPhenomenaModel(base_model, model.presence, model.sites,
                                   observed, probabilities, model.quadrature_order)
end

function information_gain(::Val{:mutual_information}, problem,
    prior::UnobservedPhenomenaModel, posterior::UnobservedPhenomenaModel, state, observation)
    return sum(kl_divergence(p, q)
               for (p, q) in zip(posterior.probabilities, prior.probabilities))
end

observation_outcomes(distribution::Bernoulli, order) =
    [(observation=y, weight=pdf(distribution,y)) for y in (false,true) if pdf(distribution,y) > 0]

function observation_outcomes(distribution::MixtureModel, order)
    return [(observation=o.observation, weight=w*o.weight)
            for (component,w) in zip(components(distribution), probs(distribution)) if w > 0
            for o in observation_outcomes(component,order)]
end
