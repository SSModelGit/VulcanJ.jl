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
       compute_kl_reward,
       one_shot_ergodic_planner,
       simulate_info_path,
       plot_simulated_path

###############################################
## Packages used across multiple files

### Quality-of-life packages (used throughout all files)
using Reexport
using Parameters: @with_kw, @with_kw_noshow
using Match
using ProgressMeter

using Plots
###

### Below are packages used exclusively in the agent definitions (`intentional_*.jl` files)
using GaussianProcesses, FastGaussQuadrature
using Random: AbstractRNG, GLOBAL_RNG
using SpecialFunctions: erf
using StatsBase: Weights, sample
using LinearAlgebra: normalize, ⋅, norm
using Distributions: Normal, MvNormal

using POMDPs, POMDPTools

# Register required functions with POMDPTools if available (best-effort).
# This intentionally does not provide fallbacks —
# missing functions should raise clear errors so implementers know what to provide.
# POMDPTools.add_requirement(RiskBoundedInfoMCTS, :get_initial_gp)
# POMDPTools.add_requirement(RiskBoundedInfoMCTS, :add_obs_to_gp)
# POMDPTools.add_requirement(RiskBoundedInfoMCTS, :get_failure_prob)
# POMDPTools.add_requirement(RiskBoundedInfoMCTS, :posterior_phenomenon_prob)
# POMDPTools.add_requirement(RiskBoundedInfoMCTS, :horizon)

export get_initial_gp, add_obs_to_gp, get_failure_prob, posterior_phenomenon_prob, cellsites, horizon

using MuKumari
###
###############################################

# ============================================================================
# Solver Definition
# ============================================================================

@with_kw struct RiskBoundedInfoMCTS <: Solver
    # Planning parameters
    lookahead::Int              # planning horizon H per solve call
    time_budget::Float64        # planning time limit τ (seconds)
    quad_order::Int             # Gauss-Hermite quadrature order J (e.g., 5)

    # Risk and reward parameters
    risk_budget::Float64        # total risk budget Δ
    alpha::Float64              # performance-guided scaling: 0 ≤ α ≤ 1
    reference_reward::Float64   # baseline expected reward for scaling
    risk_dereward::Float64 = -1000000000.0 # penalty for exceeding risk allowance

    # RNG
    rng::AbstractRNG
end


# ============================================================================
# Policy Definition (stores computed tree and values)
# ============================================================================

@with_kw struct RiskBoundedInfoPolicy <: Policy
    solver::RiskBoundedInfoMCTS
    mdp::MDP
    mdp_state_type::Any     # cache state type for tree keys

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
# Helper struct for tree nodes
# ============================================================================

@with_kw mutable struct TreeNode
    visits::Int
    action_values::Dict  # action → estimated value
    action_counts::Dict  # action → visit count
    actions_tried::Set   # actions that have been sampled
end

Base.copy(node::TreeNode) =
    TreeNode(node.visits, copy(node.action_values), copy(node.action_counts), copy(node.actions_tried))

@with_kw struct InfoNodeKey
    state::Any
    new_obs_hist::Vector
end

initialize_nodekey(state::Any) = InfoNodeKey(state, Vector{Any}())
successor_nodekey(nodekey::InfoNodeKey, new_state, new_obs) =
    InfoNodeKey(new_state, [nodekey.new_obs_hist; new_obs])

Base.:(==)(a::InfoNodeKey, b::InfoNodeKey) = a.state == b.state && a.new_obs_hist == b.new_obs_hist
Base.hash(k::InfoNodeKey, h::UInt) = hash(k.new_obs_hist, hash(k.state, hash(InfoNodeKey, h)))

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
        time_step = 0,
    )

    return policy
end


# ============================================================================
# Policy query: action given current state
# ============================================================================

function POMDPs.action(policy::RiskBoundedInfoPolicy, s::Any)
    # On first visit to this state: build search tree
    # we will only evaluate a pure state from an initial position (ergo, with an initial node key)
    nodekey = initialize_nodekey(s)
    if nodekey ∉ keys(policy.tree_nodes)
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

    # Compute risk budget
    # TODO: I think this is bugged
    remaining_horizon = horizon(mdp) - policy.time_step
    remaining_risk = solver.risk_budget - policy.risk_used
    risk_per_step = remaining_risk / remaining_horizon

    delta_allowed = compute_performance_guided_bound(
        risk_per_step,
        policy.info_gained,
        solver.reference_reward,
        solver.alpha,
    )
    println("this is our initial delta allowed for this step: $delta_allowed")

    # Initialize root node
    initial_nodekey = initialize_nodekey(initial_state)
    policy.tree_nodes[initial_nodekey] =
        TreeNode(visits = 0, action_values = Dict(), action_counts = Dict(), actions_tried = Set())

    # Retrieve or initialize GP belief at initial state
    if initial_nodekey ∉ keys(policy.node_gps)
        policy.node_gps[initial_nodekey] = initialize_gp_belief(mdp, initial_state)
    end

    initial_gp = policy.node_gps[initial_nodekey]

    # Run MCTS iterations for time budget
    start_time = time()
    iteration = 0

    while (time() - start_time) < solver.time_budget
        _ = sample_rollout(
            policy,
            initial_nodekey,
            initial_gp,
            policy.time_step,
            solver.lookahead,
            policy.risk_used,
            0.0,
            delta_allowed,
        )

        iteration += 1
    end

    # Extract best action from root
    root_node = policy.tree_nodes[initial_nodekey]
    if isempty(root_node.action_values)
        feasible = actions(mdp, initial_state)
        best_a = isempty(feasible) ? nothing : first(feasible)
    else
        best_a = argmax(root_node.action_values)
    end
    policy.best_action[initial_state] = best_a
end


# ============================================================================
# Recursive rollout with risk pruning
# ============================================================================

function sample_rollout(
    policy::RiskBoundedInfoPolicy,
    nodekey::Any,
    gp::Any,
    depth::Int,
    horizon_end::Int,
    risk_accum::Real,
    info_accum::Real,
    risk_allowed::Real,
)
    mdp = policy.mdp
    solver = policy.solver
    state = nodekey.state

    ### OBSERVATION ESTIMATION VIA GAUSS-HERMITE QUADRATURE
    (μ, σ²) = gp_predict(gp, state)  # GP mean and variance at current state

    abscissae, weights = gausshermite(solver.quad_order)

    expected_info_delta = 0.0
    for j in eachindex(abscissae)
        # synthetic observation
        y_j = sqrt(2 * σ²) * abscissae[j] + μ

        # Condition on this observation
        gp_tmp = add_obs_to_gp(state, y_j, gp)

        # Compute KL divergence (reward for this measurement)
        kl_j = compute_kl_reward(gp, gp_tmp, mdp)

        # Accumulate weighted (Fast Gauss-Hermite formula uses sqrt(pi))
        expected_info_delta += (weights[j] / sqrt(π)) * kl_j
    end

    info_here = info_accum + expected_info_delta

    # Terminal condition: reached planning horizon
    if depth == horizon_end
        return info_here  # return cumulative info as leaf value
    end

    # Get or initialize node metadata
    if nodekey ∉ keys(policy.tree_nodes)
        policy.tree_nodes[nodekey] =
            TreeNode(visits = 0, action_values = Dict(), action_counts = Dict(), actions_tried = Set())
        policy.node_gps[nodekey] = gp
    end

    node = policy.tree_nodes[nodekey]

    # SELECT / EXPAND ACTION
    action = select_action(policy, state, node, risk_accum, risk_allowed)

    if isnothing(action)  # No feasible action
        println(
            "infeasible at depth $depth with accumulated risk $risk_accum; returning extreme negative reward.",
        )
        return solver.risk_dereward  # Branch is infeasible
    end

    if !haskey(node.action_values, action)
        node.action_values[action] = 0.0
    end

    # Select one quadrature branch to use for the recursive GP update
    wsum = sum(weights)
    probs = (weights ./ (wsum > 0 ? wsum : 1.0))
    idx = sample(Weights(probs))
    y_selected = sqrt(2 * σ²) * abscissae[idx] + μ
    gp_for_recursion = add_obs_to_gp(state, y_selected, gp)

    # TRANSITION & RISK CHECK
    # Sample next state (conditioned on no collision)
    s_next = next_state(mdp, state, action, policy.solver.rng)

    # Compute collision/failure probability for this action
    delta_k = collision_probability(mdp, state, action)

    risk_next = risk_accum + delta_k

    # RECURSE
    result = sample_rollout(
        policy,
        successor_nodekey(nodekey, s_next, y_selected),
        gp_for_recursion,
        depth + 1,
        horizon_end,
        risk_next,
        info_here,
        risk_allowed,
    )

    # BACKPROP

    node.visits += 1

    # Running average of action returns
    old_val = node.action_values[action]
    count = get(node.action_counts, action, 0) + 1
    node.action_values[action] = (old_val * (count - 1) + result) / count
    node.action_counts[action] = count

    return result
end


# ============================================================================
# Action selection with UCB and risk feasibility
# ============================================================================

function select_action(
    policy::RiskBoundedInfoPolicy,
    state::Any,
    node::TreeNode,
    risk_accum::Real,
    risk_allowed::Real,
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
# Helper functions
# ============================================================================

function compute_performance_guided_bound(
    risk_per_step::Real,
    info_gained::Real,
    reference_reward::Real,
    alpha::Real,
)
    # Δ^pg = (1-α) * Δ_uniform + α * (g / g_ref) * Δ_uniform

    delta_uniform = risk_per_step
    reward_scaling = info_gained / (reference_reward + 1e-8)

    return (1.0 - alpha) * delta_uniform + alpha * reward_scaling * delta_uniform
end


function gp_predict(gp::Any, state::Any)
    # Extract location from state; predict mean and variance
    # Returns (μ::Float, σ²::Float)
    location = extract_location(state)
    (mu, sigma2) = predict_f(gp, location')
    # since predicting over only one state we know dimensionality of vec to be 1x1 per each
    return (mu[1], sigma2[1])
end


function compute_kl_reward(gp_prior::Any, gp_posterior::Any, mdp::Any)
    # Sum KL divergences between posteriors for each phenomenon variable
    # ∑_i D_KL( p(X_i | posterior) || p(X_i | prior) )
    kl_sum = 0.0
    for s in cellsites(mdp)
        p_prior = posterior_phenomenon_prob(mdp, gp_prior, s)
        p_post = posterior_phenomenon_prob(mdp, gp_posterior, s)
        kl_sum += kl_divergence(p_post, p_prior)
    end
    return kl_sum
end


function collision_probability(mdp::Any, state::Any, action::Any)
    # Return P[C_{k+1} | state, action]
    # Expect the MDP to implement `get_failure_prob(mdp, state, action)`.
    return get_failure_prob(mdp, state, action)
end


function initialize_gp_belief(mdp::Any, state::Any)
    # Initialize and return a GP belief for the environment
    # Expect the MDP to implement `get_initial_gp(mdp, state)`.
    return get_initial_gp(mdp, state)
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


last_observation_location(gp::Any) = reshape(gp.x[:, end:end], 1, :)


function gp_predictive_kl(gp_prior::Any, gp_posterior::Any)
    # Gaussian fallback: KL between scalar predictive normals at the last location.
    μ1, σ1² = predictive_moments(gp_prior)
    μ2, σ2² = predictive_moments(gp_posterior)
    σ1² = max(σ1², eps())
    σ2² = max(σ2², eps())
    return 0.5 * (log(σ2² / σ1²) + (σ1² + (μ1 - μ2)^2) / σ2² - 1)
end


function predictive_moments(gp::Any)
    X = last_observation_location(gp)
    μ, Σ = predict_f(gp, X')
    return first(vec(μ)), first(vec(Σ))
end


function initial_gp_from_state(mdp::Any, state::Any)
    dim = observation_dim(mdp, state)
    return GPE(Matrix{Float64}(undef, dim, 0), Float64[], MeanZero(), SE(zeros(dim), 0.0))
end


observation_dim(mdp::Any, state::Any) = size(extract_location(state), 2)

extract_location(state::Any) =
    state isa AbstractArray ? reshape(Float64.(state), 1, :) : reshape([Float64(state)], 1, :)

function kl_divergence(p_post::Real, p_prior::Real)
    q = clamp(float(p_prior), eps(), 1 - eps())
    p = clamp(float(p_post), eps(), 1 - eps())
    return p * log(p / q) + (1 - p) * log((1 - p) / (1 - q))
end

###
# Generics
###

function get_failure_prob(mdp::Any, s::Any, a::Any)
    error("MDP must implement `get_failure_prob(mdp, s, a)` to return risk for state-action pair.")
end

function get_initial_gp(mdp::Any, s::Any)
    error("MDP must implement `get_initial_gp(mdp, s)` to return initial GP belief.")
end

function posterior_phenomenon_prob(mdp, gp, s)
    error(
        "MDP must implement `posterior_phenomenon_prob(gp, idx)` to return probability of phenomenon at index.",
    )
end

function cellsites(mdp::Any)
    error("MDP must implement `cellsites(mdp)` to return discretized cells covering the searchable space.")
end

function horizon(mdp::Any)
    error("MDP must implement `horizon(mdp)` to return episode length.")
end

function add_obs_to_gp(X::Any, y::Any, gp::Any)
    error("MDP must implement `add_obs_to_gp(X, y, gp)` to return updated GP with new observation.")
end

include("ergodic_path_planner.jl")

include("viz_info_path.jl")

end
