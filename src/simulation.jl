"""Condition a policy on one executed successor and advance its mission bookkeeping.

The observation defaults to the newest record in the successor's history. The
starting policy model must already include observations at its current root.
"""
function update_environment_model!(policy::Union{RiskBoundedInfoPolicy,ErgodicPolicy}, state, a;
    observation=last(collect(observation_history(policy.solver, policy.problem, state))).observation,
    update_model=true)
    problem = policy.problem
    risk = get_failure_prob(problem, policy.root_state, a)
    posterior = update_model ? condition_environment_model(problem, policy.model, state, observation) : policy.model
    gain = update_model ? information_gain(policy.objective, problem, policy.model, posterior, state, observation) : 0.0
    if policy isa RiskBoundedInfoPolicy
        t = state_time(problem, policy.root_state) - policy.mission_start_time
        reference = update_model ? (t * policy.reference_reward + gain) / (t + 1) : policy.reference_reward
        set_environment_model!(policy, state, posterior;
            remaining_steps=policy.remaining_steps-1, risk_used=policy.risk_used+risk,
            reference_reward=reference, mission_start_time=policy.mission_start_time)
    else
        set_environment_model!(policy, state, posterior; remaining_steps=policy.remaining_steps-1)
    end
    return (;observation, gain, risk)
end

"""
    simulate_info_path(problem, policy, n_steps; initial_state, model, kwargs...)

Execute a conditional policy using the problem's real generator. `model` is the
posterior at `initial_state`; its existing observations are not conditioned again.
Returns states, actions, new observations, information rewards, risks, final model,
and policy. Planning predictions remain separate from executed observations.
"""
function simulate_info_path(problem::Union{MDP,POMDP}, policy::Union{RiskBoundedInfoPolicy,ErgodicPolicy},
    n_steps::Integer; initial_state, model, rng=policy.solver.rng,
    observe_fn=(p,s)->last(collect(observation_history(policy.solver,p,s))).observation,
    update_model=true, mission_start_time=0)
    state = initial_state
    initialize_simulation!(policy, state, model, n_steps, mission_start_time)
    states, taken, observations = Any[state], Any[], Any[]
    rewards, risks = Float64[], Float64[]
    for _ in 1:n_steps
        isterminal(problem, state) && break
        a = action(policy, state)
        isnothing(a) && break
        state = next_state(problem, state, a, rng)
        step = update_environment_model!(policy, state, a;
            observation=observe_fn(problem,state), update_model)
        push!(states,state); push!(taken,a); push!(observations,step.observation)
        push!(rewards,step.gain); push!(risks,step.risk)
    end
    return (;states, actions=taken, observations, information_rewards=rewards, risks,
             model=policy.model, policy)
end

initialize_simulation!(policy::ErgodicPolicy, state, model, n_steps, mission_start_time) =
    set_environment_model!(policy,state,model;remaining_steps=n_steps)
initialize_simulation!(policy::RiskBoundedInfoPolicy, state, model, n_steps, mission_start_time) =
    set_environment_model!(policy,state,model;remaining_steps=n_steps,mission_start_time)

# Existing callback interface: initialize a prior and include the initial reading.
function simulate_info_path(problem::Union{MDP,POMDP}, policy::Union{RiskBoundedInfoPolicy,ErgodicPolicy},
    observe_fn::Function, n_steps::Integer; initial_state=nothing, update_model=true, mission_start_time=0)
    state = isnothing(initial_state) ? rand(policy.solver.rng, initialstate(problem)) : initial_state
    model = initial_environment_model(problem,state)
    observation = observe_fn(problem,state)
    update_model && (model = condition_environment_model(problem,model,state,observation))
    result = simulate_info_path(problem,policy,n_steps;initial_state=state,model,
        observe_fn,update_model,mission_start_time)
    return result.states, [Any[observation]; result.observations]
end
