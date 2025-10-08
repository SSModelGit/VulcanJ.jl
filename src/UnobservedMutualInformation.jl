"""
Calculates the Mutual Information on Unobserved Phenomena for given observations.

    Based on the derivation of unobserved phenomena noted in Ben's master's thesis.
"""
module UnobservedMutualInformation

using Random: AbstractRNG
using SpecialFunctions: erf
using LinearAlgebra: ⋅
using GaussianProcesses: GPE, MeanZero, SE, predict_f

using ..VulcanJ: VulcanProblem

"""Generic method for defining the sites at which evaluate reward.

Should be specified for UnobservedMutualInformation to work.
"""
function cellsites end

######
# Gaussian Process Construction
######
# Helps with construction of Gaussian Process models
# Should technically be replaceable with equivalent GP models that return the same settings
# Goal is to obfuscate the internals of the modeling from VulcanMDP, to let that focus on MDP qualities

"""Quick constructor of an empty Gaussian Process object with a 0-mean and a RBF-kernel.
"""
empty_gp(dim::Integer=2) = GPE(Matrix{Float64}(undef,dim,0),Float64[],MeanZero(),SE(zeros(dim),0.0))

"""Point-wise estimate of mean and variance at point in Gaussian Process.
"""
μΣ_point(X::Matrix{Float64}, gp::GPE) = map(x->x[1],predict_f(gp, X))

"""Mean-variance corrected value of an environment observation.

The sample `y` is expected to be an abcissae sample from a Gauss-Hermite quadrature.
"""
env_sample_gh(X::Matrix{Float64}, y::Float64, gp::GPE) = let (μ,Σ)=μΣ_point(X,gp); μ+√(2Σ)*y; end

"""Constructs new Gaussian Process object with updated observation.
"""
add_obs_to_gp(X::Matrix{Float64}, y::Float64, gp::GPE) = GPE(hcat(gp.x,X),vcat(gp.y, env_sample_gh(X,y,gp)),gp.mean,gp.kernel)

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
site_information_divergence(μ::Float64, Σ::Float64, p::VulcanProblem) = log(1/(0.5*p.p̄⋅[phprob(μ,Σ,p.ū),1-phprob(μ,Σ,p.ū)]))

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
# function Δmutual_info_up(node::InfoNode, p::VulcanProblem)
#     let gp = update_observations(node)
#         EnvNode(mapreduce((x,y)->site_information_divergence(x,y,p),+,predict_gp(gp,cellsites(p))...), gp)
#     end
# end

# δmi(n::InfoNode) = n.env.δmi
# δmi(n::InitialNode) = 0.0

#### [TODO: Fix the estimate_value function approach to be better, potentially]
######## Differences from original
### Original Vulcan included the prior history for full life-time reward
### However, the only changing parts in the sum were the rewards from the current time-step onwards, so can ignore the initial parts
### This makes the V[curr] = r + V[next] update for MDPs very feasible
########

export cellsites, empty_gp, add_obs_to_gp, phprob, site_information_divergence

end