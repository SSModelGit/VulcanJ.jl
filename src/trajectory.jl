"""Generate one predicted MCTS realization by replanning after each outcome.

State timestamps are measured relative to `mission_start_time`. The supplied
reference is mean information per completed mission step. All subsequent
updates are hypothetical; the caller's state and model are not mutated.
"""
function plan_trajectory(solver::AbstractInfoMCTS, problem::Union{MDP,POMDP}, state, model, horizon;
    objective=Val(:mutual_information), remaining_steps=horizon, risk_used=0.0,
    reference_reward=solver.reference_reward, mission_start_time=0)
    policy = solve(solver, problem; objective)
    states, taken, observations = Any[state], Any[], Any[]
    rewards, risks, alphas, references = Float64[], Float64[], Float64[], Float64[]
    status = :horizon
    for _ in 1:min(horizon, remaining_steps)
        if isterminal(problem, state)
            status = :terminal
            break
        end
        set_environment_model!(policy, state, model;
            remaining_steps, risk_used, reference_reward, mission_start_time)
        a = action(policy, state)
        if isnothing(a)
            status = :infeasible
            break
        end
        branch = policy.root.branches[policy.root.best]
        sp, observation = generated_step(solver, problem, model, state, a, solver.rng)
        posterior = condition_environment_model(problem, model, sp, observation)
        gain = information_gain(objective, problem, model, posterior, sp, observation)
        push!(taken, a)
        push!(observations, observation)
        push!(states, sp)
        push!(rewards, gain)
        push!(risks, branch.risk)
        push!(alphas, policy.alpha)
        push!(references, reference_reward)
        t = state_time(problem, state) - mission_start_time
        reference_reward = (t * reference_reward + gain) / (t + 1)
        risk_used += branch.risk
        remaining_steps -= 1
        state, model = sp, posterior
    end
    isterminal(problem, state) && (status = :terminal)
    return (;states, actions=taken, observations, information_rewards=rewards,
             risks, alphas, reference_rewards=references, status)
end
