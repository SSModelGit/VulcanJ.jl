module InformationMDP

import POMDPs
using FastGaussQuadrature: gausshermite
using SpecialFunctions: erf
using StatsBase: Weights, sample
using LinearAlgebra: ⋅
using ..EnvironmentGP

### Data Container State

abstract type VulcanNode end

struct InitialNode <: VulcanNode
    X::Matrix{Int64}
end

struct InfoNode <: VulcanNode
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

############
# Transition
############
## Deterministic update of (x,y) coordinate
## Non-deterministic update of the information (Gauss-Hermite roots), probability of transition is Gauss-Hermite weights
### Information is not collected (added as measurement to GP) until the state is *left*
## GP is updated to include a new measurement using the info of the *current* state, NOT the new state
############

## Core Information Updates
"""Return a sample from the distribution of abcissae under Gauss-Hermite quadrature.
"""
sample_y_j(p::InfoProblem) = √2 * sample(p.abcissae, Weights(p.weights./sqrt(π)))
"""Return the mean and variance at a particular location given an environment node.
"""
μΣ_point(X::Matrix{Float64}, env::EnvNode) = map(x->x[1],predict_gp(env.gp, X))
"""Return mean-variance-corrected sample of the environment at specified location using Gauss-Hermite quadrature estimation.
"""
gh_env_sample(X::Matrix{Float64}, env::EnvNode, p::InfoProblem) = let (μ,Σ)=μΣ_point(X,env); μ+√Σ*sample_y_j(p); end

## Core Dynamics Updates
""" Returns vector of valid actions (movement vectors) that do not violate boundary counditions listed in the MDP problem description.

    Arguments:
        n::VulcanNode - Node with a position description. Typically only InfoNode, but using general VulcanNode for correctness.
        p::InfoProblem - MDP problem definition container.
    Returns:
        a_list::Vector{Matrix{Int64}} - List of valid possible actions from current node `n`.
"""
valid_actions(n::VulcanNode, p::InfoProblem) = [[x;y;;] for x in -1:1 if 0<n.X[1]+x≤p.Xlims[1] for y in -1:1 if (0<n.X[2]+y≤p.Xlims[2] && x*y+x+y≠0)]


"""Naive successor constructor for the InitialNode state. Should only be ever called on the initial node.

Does not account for prior information before sampling (assumes initial process model with zero-mean, 1-cov). Reward of transition is always zero.

    Arguments:
        node::InitialNode - The root node of the search. Based at a particular location, without any additional information.
        p::InfoProblem - MDP problem definition container.
    Returns:
        ns::InfoNode - A node at the same position as `node`, but with information value taken from a simple Gauss-Hermite curve approximation.
            - Contains an empty GP as well, using a SEard kernel (matching the dimensions of the position matrix.)
"""
make_successor(node::InitialNode, p::InfoProblem) = InfoNode(node.X, sample_y_j(p), Δmutual_info_up(node))

"""Successor constructor for the InfoNode state.

The update does the following:

    - Deterministic dynamics update. The action describes a vector of movement that never fails or deviates.
    - Generates a new sample value, using a mean-variance corrected Gauss-Hermite quadrature approximation.
        - The *new* sample value is not included in the environment.
    - Generates an updated environment node. The update adds an observation corresponding to the *current* node's sample value & location.
        - The update uses `Δmutual_info_up`, which generates a new environment node holding the next node's new information reward and environment model.
        - The newly generated GP is also optimized, performed within `EnvironmentGP.update_observations`.

    Arguments:
        node::InfoNode - The current state node.
        action::Matrix{Int64} - The action direction. Assumed to be a valid movement direction.
        p::InfoProblem - MDP problem definition container.
    Returns:
        ns::InfoNode - The next state node succeeding the current state node, based on the action provided and problem definition.
"""
function make_successor(node::InfoNode, action::Matrix{Int64}, p::InfoProblem)
    let Xnew=node.X+action, env=Δmutual_info_up(node, p)
        InfoNode(Xnew, gh_env_sample(float(Xnew),env,p), env)
    end
end

########
# Reward
########
## Reward for moving to the next state is determined via function on the GP *before* including the new measurement on current info
## D_kl(current state || prior) = f(information prior to current state)
## ==> D_kl(..||..) = log(1/(1- [P1/2 * (1+erf((u_bar - mu(gp)) / sqrt(2*cov(gp)))) + P2/2 * (1-erf((u_bar - mu(gp)) / sqrt(2*cov(gp)))))]))
### Reward(current state, action) only needs to be D_kl(current state || prior state)
### Q-value update in MCTS will automatically do: D_kl(current state || prior state) + discount * (D_kl(next state || current state) + ...)
## Horizon and Leaf-node Value Estimation
### For now, make the horizon 1-short (i.e., at horizon-depth==0, estimate_value(.) = 0)
########

"""Wrapper on the observation update function defined in EnvironmentGP.

Needs to parse out the location and observation information from the InfoNode struct, and then pass those in to the underlying observation update function.

    Arguments:
        node::InfoNode - Node being sampled.
    Returns:
        gp::GP - Gaussian process, as defined in GaussianProcesses.
"""
EnvironmentGP.update_observations(node::InfoNode) = update_observations(node.env, node.X, node.info)

"""
Phenomena Likelihood probability

Calculates the likelihood of phenomena occuring at the particular cell described by μ and Σ.
Utilizes the thresholds described in the InfoProblem instance.

Returns: Probability of the phenomena occuring should the true continuous state fall below threshold ū
"""
phprob(μ::Float64, Σ::Float64, ū::Float64) = 1/2 * (1 - erf((ū - μ)/sqrt(2*Σ)))
"""
Calculates the divergence in the information distribution on a binary variable (unobserved phenomena presence).

Only handles scalar inputs. Calculates for a given location (i.e., does not calculate globally but at a specific site within the sites of `p`.)
TODO: Need to make the divergence at locations already sampled equal to zero (i.e. no change in the expectation, didn't we just sample?)

    Arguments:
        μ::Float64 - Mean calculated at site of interest. Typically provided by the model within EnvNode.
        Σ::Float64 - Variance calculated at the site of interest. Typically provided by the model within EnvNode.
        p::InfoProblem - MDP problem definition container. Holds value for phenomena likelihood thresholding ū.
    Returns:
        d_kl::Float64 - Divergence between prior and current distributions on unobserved phenomena presence, at site of interest.
"""
site_information_divergence(μ::Float64, Σ::Float64, p::InfoProblem) = log(1/(0.5*p.p̄⋅[phprob(μ,Σ,p.ū),1-phprob(μ,Σ,p.ū)]))

"""Immediate mutual information update after sampling from the given state node. Returns an updated environment node.

Also considered as the reward of leaving the `InitialNode` towards the first "sampling node", i.e. the first `InfoNode`.

Recall that "r[t=0] is always zero." The `InitialNode` has no action preceding it, so its reward is automatically zero.
However, while transitioning to the first `InfoNode`, no sample was taken (because the `InitialNode` is only a tree root, not an actual state).
Hence, there is no reward in this transition. This is equivalent to r[t=-1], which necessarily is also 0.

    Arguments:
        node::InitialNode - The initial node.
    Returns:
        env::EnvNode = (0.0, GPE(MeanZero(),SEard(0.0,0.0,0.0))). Always zero reward.
"""
Δmutual_info_up(node::InitialNode) = EnvNode(dim=size(node.X)[1])

"""Immediate mutual information update after sampling from the given state node. Returns an updated environment node.

This is also treated as the reward of 'having entered the current state', i.e. the reward we receive at state `node` regardless of the action we take.
- Note that the typical idea of reward is one that we receive upon taking an action. In our formulation, we instead delay by one step.
- Ex. Normally, r[t=0] is the reward of "heading to location x[t=1] for a sample".
    - Now, this is r[t=1]. Likewise, r[t=2] is the reward of "heading to location x[t=2] for a sample". r[t=k] is independent of a[t=k].
    - r[t=0] is always zero. This is because there is no "mutual information gain" without having taken a sample first.

Essentially, since samples are always taken *after* entering the state at x[t=k], the gain in the mutual information is only seen in r[t=k+1].

    Arguments:
        node::InfoNode - Node being sampled.
        p::InfoProblem - MDP problem definition container.
    Returns:
        env::EnvNode - Updated environment node reflecting the change in mutual information of unobserved phenomena after sampling `node`.
"""
function Δmutual_info_up(node::InfoNode, p::InfoProblem)
    let gp = update_observations(node)
        EnvNode(mapreduce((x,y)->site_information_divergence(x,y,p),+,predict_gp(gp,cellsites(p))...), gp)
    end
end

#### [TODO: Fix the estimate_value function approach to be better, potentially]
######## Differences from original
### Original Vulcan included the prior history for full life-time reward
### However, the only changing parts in the sum were the rewards from the current time-step onwards, so can ignore the initial parts
### This makes the V[curr] = r + V[next] update for MDPs very feasible
########

export InitialNode, valid_actions, make_successor, InfoNode, InfoProblem, cellsites, Δmutual_info_up

end