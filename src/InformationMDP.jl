module InformationMDP

import POMDPs
using FastGaussQuadrature: gausshermite
using SpecialFunctions: erf
using ..EnvironmentGP

### Data Container State
struct InfoNode
    X::Matrix{Int64} # grid world coordinates - deterministic update
    info::Float64 # Observation that we've transitioned into - pseudo-nondeterministic update, use Gauss-Hermite weights for probs
    env::EnvNode # 
    # gp::Bool # Fake Gaussian Process (Boolean) - update this during transitions to incorporate the "info" of the current state as a new measurement
end

###  MDP Construct
mutable struct InfoProblem <: POMDPs.MDP{InfoNode, Symbol}
    Xlims::Vector{Int64} # width and height limits of the world
    ū::Float64 # threshold for phenomena discovery
    p̄::Vector{Float64} # probability of phenom. when less and greater than u_bar
    abcissae::Vector{Float64} # roots of Gauss-Hermite Quadrature (via FastGaussQuadrature)
    weights::Vector{Float64} # corresponding weights of VulcanWorld.abcissae
    discount_factor::Float64 # discount factor (default: 1)
end

function InfoProblem(;sxy::Vector{Int64}=[10,10],
                      ū::Float64=0.5,
                      p̄::Vector{Float64}=[0.3,0.6],
                      gh_deg::Int64=5,
                      gamma::Float64 = 1.0)
    return InfoProblem(sxy,ū,p̄,gausshermite(gh_deg)...,gamma)
end

cellsites(p::InfoProblem) = stack([[float(x),float(y)] for x in 1:p.Xlims[1] for y in 1:p.Xlims[2]], dims=2);


########
# Reward
## Reward for moving to the next state is determined via function on the GP *before* including the new measurement on current info
## D_kl(current state || prior) = f(information prior to current state)
## ==> D_kl(..||..) = log(1/(1- [P1/2 * (1+erf((u_bar - mu(gp)) / sqrt(2*cov(gp)))) + P2/2 * (1-erf((u_bar - mu(gp)) / sqrt(2*cov(gp)))))]))
### Reward(current state, action) only needs to be D_kl(current state || prior state)
### Q-value update in MCTS will automatically do: D_kl(current state || prior state) + discount * (D_kl(next state || current state) + ...)
## Horizon and Leaf-node Value Estimation
### For now, make the horizon 1-short (i.e., at horizon-depth==0, estimate_value(.) = 0)
########

"""
Phenomena Likelihood probability

Calculates the likelihood of phenomena occuring at the particular cell described by μ and Σ.
Utilizes the thresholds described in the InfoProblem instance.
Returns: Probability of the phenomena occuring should the true continuous state fall below threshold ū
"""
phprob(μ::Float64, Σ::Float64, ū::Float64) = 1/2 * (1 - erf((ū - μ)/sqrt(2*Σ)))
site_information_divergence(μ::Float64, Σ::Float64, p::InfoProblem) = 0.5.* p.p̄ .* [phprob(μ,Σ,p.ū),1-phprob(μ,Σ,p.ū)]
Δmutual_info(node::InfoNode, p::InfoProblem) = reduce(+,site_information_divergence(predict_env(node.env, cellsites(p))...,p))

# Transition
## Deterministic update of (x,y) coordinate
## Non-deterministic update of the information (Gauss-Hermite roots), probability of transition is Gauss-Hermite abcissae
### Information is not collected (added as measurement to GP) until the state is *left*
## GP is updated to include a new measurement using the info of the *current* state, NOT the new state
#### [TODO: Fix the estimate_value function approach to be better, potentially]
######## Differences from original
### Original Vulcan included the prior history for full life-time reward
### However, the only changing parts in the sum were the rewards from the current time-step onwards, so can ignore the initial parts
### This makes the V[curr] = r + V[next] update for MDPs very feasible
########

export InfoNode, InfoProblem, cellsites, Δmutual_info

end