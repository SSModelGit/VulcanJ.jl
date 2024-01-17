using FastGaussQuadrature: gausshermite

### Test Data Container State
struct VulcanState
    x::Matrix{Int64} # grid world coordinates - deterministic update
    info::Float64 # Observation that we've transitioned into - pseudo-nondeterministic update, use Gauss-Hermite weights for probs
    
    # gp::Bool # Fake Gaussian Process (Boolean) - update this during transitions to incorporate the "info" of the current state as a new measurement
end

###  Test MDP Construct
mutable struct VulcanWorld <: POMDPs.MDP{VulcanState, Symbol}
    size_x::Int64 # width limit of the world
    size_y::Int64 # height limit of the world
    u_bar::Float64 # threshold for phenomena discovery
    p1::Float64 # probability of phenom. when less than u_bar
    p2::Float64 # probability of phenom. when greater than u_bar
    abcissae::Vector{Float64} # roots of Gauss-Hermite Quadrature (via FastGaussQuadrature)
    weights::Vector{Float64} # corresponding weights of VulcanWorld.abcissae
    discount_factor::Float64 # discount factor (default: 1)
end

function VulcanWorld(;sx::Int64=10,
                      sy::Int64=10,
                      u_bar::Float64=0.5,
                      p1::Float64=0.3,
                      p2::Float64=0.6,
                      gh_deg::Int64=5,
                      gamma::Float64 = 1.0)
    return VulcanWorld(sx,sy,u_bar,p1,p2,gausshermite(gh_deg)...,gamma)
end

# Transition
## Deterministic update of (x,y) coordinate
## Non-deterministic update of the information (Gauss-Hermite roots), probability of transition is Gauss-Hermite abcissae
### Information is not collected (added as measurement to GP) until the state is *left*
## GP is updated to include a new measurement using the info of the *current* state, NOT the new state
# Reward
## Reward is determined via function on the GP *before* including the new measurement on current info
## D_kl(current state || prior) = f(information prior to current state)
## ==> D_kl(..||..) = log(1/(1- [P1/2 * (1+erf((u_bar - mu(gp)) / sqrt(2*cov(gp)))) + P2/2 * (1-erf((u_bar - mu(gp)) / sqrt(2*cov(gp)))))]))
### Reward(current state, action) only needs to be D_kl(current state || prior state)
### Q-value update in MCTS will automatically do: D_kl(current state || prior state) + discount * (D_kl(next state || current state) + ...)
## Horizon and Leaf-node Value Estimation
### For now, make the horizon 1-short (i.e., at horizon-depth==0, estimate_value(.) = 0)
#### [TODO: Fix the estimate_value function approach to be better, potentially]
######## Differences from original
### Original Vulcan included the prior history for full life-time reward
### However, the only changing parts in the sum were the rewards from the current time-step onwards, so can ignore the initial parts
### This makes the V[curr] = r + V[next] update for MDPs very feasible
########