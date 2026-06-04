module VulcanJ

# Public API exports: primary solver types and utility hooks
export RiskBoundedInfoMCTS,
       RiskBoundedInfoPolicy,
       TreeNode,
       build_search_tree,
       sample_rollout,
       select_action,
       initialize_gp_belief,
       get_initial_gp,
       compute_kl_reward

###############################################
## Packages used across multiple files

### Quality-of-life packages (used throughout all files)
using Reexport
using Parameters: @with_kw, @with_kw_noshow
using Match
using ProgressMeter
###

### Below are packages used exclusively in the agent definitions (`intentional_*.jl` files)
using GaussianProcesses, FastGaussQuadrature
using Random: AbstractRNG
using SpecialFunctions: erf
using StatsBase: Weights, sample
using LinearAlgebra: normalize, ⋅, norm
using Distributions: Normal, MvNormal

using POMDPs, POMDPTools
using MuKumari
###
###############################################

# ============================================================================
# Solver Definition
# ============================================================================

struct RiskBoundedInfoMCTS <: Solver
  # Planning parameters
  lookahead::Int              # planning horizon H per solve call
  time_budget::Float64        # planning time limit τ (seconds)
  quad_order::Int             # Gauss-Hermite quadrature order J (e.g., 5)
  
  # Risk and reward parameters
  risk_budget::Float64        # total risk budget Δ
  alpha::Float64              # performance-guided scaling: 0 ≤ α ≤ 1
  reference_reward::Float64   # baseline expected reward for scaling
  
  # RNG
  rng::AbstractRNG
end


# ============================================================================
# Policy Definition (stores computed tree and values)
# ============================================================================

struct RiskBoundedInfoPolicy <: Policy
  solver::RiskBoundedInfoMCTS
  mdp::MDP
  mdp_state_type::DataType     # cache state type for tree keys
  
  # Tree bookkeeping
  tree_nodes::Dict             # state → node_metadata
  node_gps::Dict               # state → gp_posterior  
  best_action::Dict            # state → action
  
  # Running statistics
  risk_used::Float64           # cumulative risk consumed so far
  info_gained::Float64         # cumulative information reward
  time_step::Int               # current mission step
end


# ============================================================================
# Main solve() function
# ============================================================================

function POMDPs.solve(solver::RiskBoundedInfoMCTS, mdp::MDP)
  # Initialize policy with empty tree
  policy = RiskBoundedInfoPolicy(
    solver = solver,
    mdp = mdp,
    mdp_state_type = statetype(mdp),
    tree_nodes = Dict(),
    node_gps = Dict(),
    best_action = Dict(),
    risk_used = 0.0,
    info_gained = 0.0,
    time_step = 0
  )
  
  return policy
end


# ============================================================================
# Policy query: action given current state
# ============================================================================

function POMDPs.action(policy::RiskBoundedInfoPolicy, s::Any)
  # On first visit to this state: build search tree
  if s ∉ policy.tree_nodes
    build_search_tree(policy, s)
  end

  return policy.best_action[s]
end


# ============================================================================
# Search tree building (called once per query state)
# ============================================================================

function build_search_tree(policy::RiskBoundedInfoPolicy, initial_state::Any)
  solver = policy.solver
  mdp = policy.mdp
  
  # Compute risk budget for this planning step
  remaining_horizon = horizon(mdp) - policy.time_step
  remaining_risk = solver.risk_budget - policy.risk_used
  risk_per_step = remaining_risk / remaining_horizon
  
  delta_allowed = compute_performance_guided_bound(
    risk_per_step,
    policy.info_gained,
    solver.reference_reward,
    solver.alpha
  )
  
  # Initialize root node
  policy.tree_nodes[initial_state] = TreeNode(
    visits = 0,
    action_values = Dict(),
    action_counts = Dict(),
    actions_tried = Set()
  )
  
  # Retrieve or initialize GP belief at initial state
  if initial_state ∉ policy.node_gps
    policy.node_gps[initial_state] = initialize_gp_belief(mdp, initial_state)
  end
  
  initial_gp = policy.node_gps[initial_state]
  
  # Run MCTS iterations for time budget
  start_time = time()
  iteration = 0
  
  while (time() - start_time) < solver.time_budget
    _ = sample_rollout(
      policy,
      initial_state,
      initial_gp,
      policy.time_step,
      policy.time_step + solver.lookahead,
      policy.risk_used,
      0.0,
      delta_allowed
    )

    iteration += 1
  end
  
  # Extract best action from root
  root_node = policy.tree_nodes[initial_state]
  best_a = argmax(root_node.action_values)
  policy.best_action[initial_state] = best_a
end


# ============================================================================
# Recursive rollout with risk pruning
# ============================================================================

function sample_rollout(
  policy::RiskBoundedInfoPolicy,
  state::Any,
  gp::Any,
  depth::Int,
  horizon_end::Int,
  risk_accum::Real,
  info_accum::Real,
  risk_allowed::Real
)
  mdp = policy.mdp
  solver = policy.solver
  
  # Terminal condition: reached planning horizon
  if depth == horizon_end
    return info_accum  # return cumulative info as leaf value
  end
  
  # Get or initialize node metadata
  if state ∉ policy.tree_nodes
    policy.tree_nodes[state] = TreeNode(
      visits = 0,
      action_values = Dict(),
      action_counts = Dict(),
      actions_tried = Set()
    )
    policy.node_gps[state] = gp
  end
  
  node = policy.tree_nodes[state]
  
  # SELECT / EXPAND ACTION
  action = select_action(policy, state, node, risk_accum, risk_allowed)
  
  if isnothing(action)  # No feasible action
    return nothing  # Branch is infeasible
  end
  
  # OBSERVATION ESTIMATION VIA GAUSS-HERMITE QUADRATURE
  # Predict next measurement under current GP
  (μ, σ²) = gp_predict(gp, state)  # mean and variance
  
  abscissae, weights = gausshermite(solver.quad_order)
  
  # Expected information gain: average over quadrature branches
  expected_info_delta = 0.0
  
  for j in eachindex(abscissae)
    # Generate synthetic observation from Gaussian
    y_j = sqrt(2 * σ²) * abscissae[j] + μ

    # Condition GP on this observation (virtual update)
    gp_tmp = condition_gp(gp, y_j)

    # Compute KL divergence (reward for this measurement)
    kl_j = compute_kl_reward(gp, gp_tmp, mdp)

    # Accumulate weighted (Fast Gauss-Hermite formula uses sqrt(pi))
    expected_info_delta += (weights[j] / sqrt(π)) * kl_j
  end

  info_next = info_accum + expected_info_delta

  # Select one quadrature branch to use for the recursive GP update
  wsum = sum(weights)
  probs = (weights ./ (wsum > 0 ? wsum : 1.0))
  idx = sample(Weights(probs))
  y_selected = sqrt(2 * σ²) * abscissae[idx] + μ
  gp_for_recursion = condition_gp(gp, y_selected)
  
  # TRANSITION & RISK CHECK
  # Sample next state (conditioned on no collision)
  s_next = next_state(mdp, state, action, policy.solver.rng)
  
  # Compute collision/failure probability for this action
  delta_k = collision_probability(mdp, state, action)
  
  risk_next = risk_accum + delta_k
  
  if risk_next > risk_allowed
    # Prune this action and fail
    delete!(node.actions_tried, action)
    return nothing
  end
  
  # RECURSE
  result = sample_rollout(
    policy,
    s_next,
    gp_for_recursion,
    depth + 1,
    horizon_end,
    risk_next,
    info_next,
    risk_allowed
  )
  
  # BACKPROPAGATION
  if result !== nothing
    node.visits += 1

    if action ∉ keys(node.action_values)
      node.action_values[action] = 0.0
    end

    # Running average of action returns
    old_val = node.action_values[action]
    count = get(node.action_counts, action, 0) + 1
    node.action_values[action] = (old_val * (count - 1) + result) / count
    node.action_counts[action] = count

    return result
  else
    # Action failed; try next action in select_action loop
    return nothing
  end
end


# ============================================================================
# Action selection with UCB and risk feasibility
# ============================================================================

function select_action(
  policy::RiskBoundedInfoPolicy,
  state::Any,
  node::TreeNode,
  risk_accum::Real,
  risk_allowed::Real
)
  mdp = policy.mdp
  solver = policy.solver
  
  # Filter to feasible actions
  feasible_actions = Set()
  for a in actions(mdp, state)
    delta_a = collision_probability(mdp, state, a)
    if risk_accum + delta_a ≤ risk_allowed
      push!(feasible_actions, a)
    end
  end
  
  if isempty(feasible_actions)
    return nothing  # No feasible action
  end
  
  # If not all actions have been tried: pick random untried one
  untried = setdiff(feasible_actions, node.actions_tried)
  if !isempty(untried)
    # pick random untried action
    untried_vec = collect(untried)
    a = untried_vec[rand(solver.rng, 1:length(untried_vec))]
    push!(node.actions_tried, a)
    return a
  end
  
  # All actions tried: use UCB selection
  sqrt_log_visits = sqrt(2.0 * log(node.visits + 1))

  best_ucb = -Inf
  best_a = nothing

  for a in feasible_actions
    exploit = get(node.action_values, a, 0.0)
    count = max(get(node.action_counts, a, 1), 1)
    explore = sqrt_log_visits / sqrt(count)

    ucb_score = exploit + explore

    if ucb_score > best_ucb
      best_ucb = ucb_score
      best_a = a
    end
  end

  return best_a
end


# ============================================================================
# Helper structures and functions
# ============================================================================

struct TreeNode
  visits::Int
  action_values::Dict  # action → estimated value
  action_counts::Dict  # action → visit count
  actions_tried::Set   # actions that have been sampled
end


function compute_performance_guided_bound(
  risk_per_step::Real,
  info_gained::Real,
  reference_reward::Real,
  alpha::Real
)
  # Δ^pg = (1-α) * Δ_uniform + α * (g / g_ref) * Δ_uniform
  
  delta_uniform = risk_per_step
  reward_scaling = info_gained / (reference_reward + 1e-8)
  
  return (1.0 - alpha) * delta_uniform + alpha * reward_scaling * delta_uniform
end


function gp_predict(gp::Any, state::Any)
  # Extract location from state; predict mean and variance
  # Returns (μ::Float, σ²::Float)
  
  # Placeholder: assumes your GP has predict_f or similar
  location = extract_location(state)
  (mu, sigma2) = predict_f(gp, location)
  return (mu, sigma2)
end


function condition_gp(gp::Any, observation::Real)
  # Return new GP posterior conditioned on observation
  # Minimal version: add observation point to GP
  
  # Prefer the project-local GP update helper when available.
  if isdefined(@__MODULE__, :add_obs_to_gp)
    x = last_observation_location(gp)
    return add_obs_to_gp(x, observation, gp)
  elseif isdefined(@__MODULE__, :update_observations)
    x = last_observation_location(gp)
    return update_observations(gp, x, observation)
  else
    x = last_observation_location(gp)
    return append_observation_to_gp(gp, x, observation)
  end
end


function compute_kl_reward(gp_prior::Any, gp_posterior::Any, mdp::Any)
  # Sum KL divergences between posteriors for each phenomenon variable
  # ∑_i D_KL( p(X_i | posterior) || p(X_i | prior) )
  
  if isdefined(@__MODULE__, :phenomenon_indices) && isdefined(@__MODULE__, :posterior_phenomenon_prob)
    kl_sum = 0.0
    for i in phenomenon_indices(mdp)
      p_prior = posterior_phenomenon_prob(gp_prior, i)
      p_post = posterior_phenomenon_prob(gp_posterior, i)
      kl_sum += kl_divergence(p_post, p_prior)
    end
    return kl_sum
  end

  # Fallback: compare predictive Gaussians at the stored support points.
  return gp_predictive_kl(gp_prior, gp_posterior)
end


function collision_probability(mdp::Any, state::Any, action::Any)
  # Return P[C_{k+1} | state, action]
  # Placeholder: extract from MDP

  if hasmethod(get_failure_prob, Tuple{typeof(mdp), Any, Any})
    return get_failure_prob(mdp, state, action)
  elseif hasmethod(get_failure_prob, Tuple{typeof(mdp), Any})
    return get_failure_prob(mdp, (state, action))
  else
    return 0.0
  end
end


function initialize_gp_belief(mdp::Any, state::Any)
  # Initialize and return a GP belief for the environment
  # Prefer an explicit model field; otherwise fall back to a domain helper.
  if hasproperty(mdp, :gp0)
    return getproperty(mdp, :gp0)
  elseif hasmethod(get_initial_gp, Tuple{typeof(mdp)})
    return get_initial_gp(mdp)
  else
    return initial_gp_from_state(mdp, state)
  end
end


function next_state(mdp::Any, state::Any, action::Any, rng::AbstractRNG)
  genout = POMDPs.gen(mdp, state, action, rng)
  if genout isa NamedTuple && haskey(genout, :sp)
    return genout.sp
  elseif genout isa NamedTuple && haskey(genout, :state)
    return genout.state
  else
    return first(genout)
  end
end


function last_observation_location(gp::Any)
  if hasproperty(gp, :x) && size(getproperty(gp, :x), 2) > 0
    return getproperty(gp, :x)[:, end:end]
  elseif hasproperty(gp, :X) && size(getproperty(gp, :X), 2) > 0
    return getproperty(gp, :X)[:, end:end]
  else
    return zeros(1, 1)
  end
end


function gp_predictive_kl(gp_prior::Any, gp_posterior::Any)
  # Gaussian fallback: KL between scalar predictive normals at the last location.
  μ1, σ1² = predictive_moments(gp_prior)
  μ2, σ2² = predictive_moments(gp_posterior)
  σ1² = max(σ1², eps())
  σ2² = max(σ2², eps())
  return 0.5 * (log(σ2² / σ1²) + (σ1² + (μ1 - μ2)^2) / σ2² - 1)
end


function predictive_moments(gp::Any)
  if hasmethod(predict_gp, Tuple{typeof(gp), AbstractMatrix})
    X = last_observation_location(gp)
    μ, Σ = predict_gp(gp, X)
    return first(vec(μ)), first(vec(Σ))
  elseif hasmethod(predict_f, Tuple{typeof(gp), AbstractMatrix})
    X = last_observation_location(gp)
    μ, Σ = predict_f(gp, X)
    return first(vec(μ)), first(vec(Σ))
  else
    return 0.0, 1.0
  end
end


function initial_gp_from_state(mdp::Any, state::Any)
  dim = observation_dim(mdp, state)
  return GPE(Matrix{Float64}(undef, dim, 0), Float64[], MeanZero(), SE(zeros(dim), 0.0))
end


observation_dim(mdp::Any, state::Any) = isdefined(@__MODULE__, :extract_location) ? size(extract_location(state), 1) : 1


extract_location(state::Any) = state isa AbstractArray ? reshape(Float64.(state), :, 1) : reshape([Float64(state)], :, 1)


function get_failure_prob(mdp::Any, state::Any, action::Any)
  if hasproperty(mdp, :failure_probability)
    fp = getproperty(mdp, :failure_probability)
    if fp isa Function
      return Float64(fp(state, action))
    end
  end
  return 0.0
end


function get_initial_gp(mdp::Any)
  if hasproperty(mdp, :gp0)
    return getproperty(mdp, :gp0)
  end
  return GPE(Matrix{Float64}(undef, 1, 0), Float64[], MeanZero(), SE(zeros(1), 0.0))
end


function append_observation_to_gp(gp::Any, x::AbstractMatrix, y::Real)
  if hasproperty(gp, :x) && hasproperty(gp, :y) && hasproperty(gp, :mean) && hasproperty(gp, :kernel)
    Xold = getproperty(gp, :x)
    yold = getproperty(gp, :y)
    Xnew = hcat(Xold, x)
    ynew = vcat(yold, Float64(y))
    return GPE(Xnew, ynew, getproperty(gp, :mean), getproperty(gp, :kernel))
  elseif hasproperty(gp, :x) && hasproperty(gp, :y)
    Xold = getproperty(gp, :x)
    yold = getproperty(gp, :y)
    Xnew = hcat(Xold, x)
    ynew = vcat(yold, Float64(y))
    return GPE(Xnew, ynew, MeanZero(), SE(zeros(size(Xnew, 1)), 0.0))
  else
    return gp
  end
end


function kl_divergence(p_post::Real, p_prior::Real)
  q = clamp(float(p_prior), eps(), 1 - eps())
  p = clamp(float(p_post), eps(), 1 - eps())
  return p * log(p / q) + (1 - p) * log((1 - p) / (1 - q))
end

end
