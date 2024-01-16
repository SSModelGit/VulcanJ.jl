
"""
Test if VulcanJ is doing its imports correctly.

    Should print a string referencing 'Vulcan-J'.
    May potentially also return other artifacts.
"""
function trial_func()
    print("trial of Vulcan-J.")
    return 2
end

### Test Data Container State
struct WorldState
    x::Int64 # grid world x-location - deterministic update
    y::Int64 # grid world y-location - deterministic update
    info::Float64 # Observation that we've transitioned into - pseudo-nondeterministic update, use Gauss-Hermite weights for probs
    gp::Bool # Fake Gaussian Process (Boolean) - update this during transitions to incorporate the "info" of the current state as a new measurement
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