function initialize_node(policy, state, model, depth, risk, information; terminal = false)
    terminal |= depth >= min(policy.solver.lookahead, policy.remaining_steps) ||
                isterminal(policy.problem, state)
    branches = ActionNode[]
    if !terminal
        for a in actions(policy.problem, state)
            delta = get_failure_prob(policy.problem, state, a)
            if policy.risk_used + risk + delta <= policy.solver.risk_budget
                push!(branches, ActionNode(a, delta, 0, 0, 0.0, Any[], true))
            end
        end
    end
    return TreeNode(state, model, depth, risk, information, 0, branches,
                    nothing, terminal ? information : 0.0, terminal)
end

# Widen the empirical successor set as visits grow. Every new successor comes
# from the supplied generator; existing successors retain their continuation tree.
function child_node!(policy, node, branch)
    # Resume unfinished continuations before widening again.
    pending = findfirst(c -> c.visits == 0 || (!c.terminal && isnothing(c.best)), branch.children)
    !isnothing(pending) && return branch.children[pending]
    if length(branch.children) < sqrt(branch.visits + 1)
        sp, observation = generated_step(policy.solver, policy.problem, node.model,
                                         node.state, branch.action, policy.solver.rng)
        posterior = condition_environment_model(policy.problem, node.model, sp, observation)
        gain = information_gain(policy.objective, policy.problem, node.model,
                                posterior, sp, observation)
        child = initialize_node(policy, sp, posterior, node.depth + 1,
            node.sequence_risk + branch.risk, node.sequence_information + gain)
        push!(branch.children, child)
        return child
    end
    return rand(policy.solver.rng, branch.children)
end

function backup!(node)
    node.best = nothing
    for (i, branch) in pairs(node.branches)
        branch.admissible || continue
        branch.visits == 0 && continue
        any(c -> c.visits > 0 && !c.terminal && isnothing(c.best), branch.children) && continue
        branch.value = sum(child.visits * child.value for child in branch.children
            if child.visits > 0; init = 0.0) / branch.visits
        if isnothing(node.best) || branch.value > node.branches[node.best].value
            node.best = i
        end
    end
    !isnothing(node.best) && (node.value = node.branches[node.best].value)
    return node.best
end

function select_action(policy::RiskBoundedInfoPolicy, node::TreeNode)
    available = findall(branch -> branch.admissible, node.branches)
    isempty(available) && return nothing
    unvisited = filter(i -> node.branches[i].attempts == 0, available)
    !isempty(unvisited) && return rand(policy.solver.rng, unvisited)
    attempts = sum(b.attempts for b in node.branches)
    return available[argmax([node.branches[i].value +
        sqrt(2 * log(attempts) / node.branches[i].attempts) for i in available])]
end

# One forward path per iteration: backtracking returns control to root selection.
function sample_rollout(policy::RiskBoundedInfoPolicy, node::TreeNode)
    if node.terminal
        terminal_feasible(policy, node) || return false
        node.visits += 1
        return true
    end
    index = select_action(policy, node)
    isnothing(index) && return false
    branch = node.branches[index]
    branch.attempts += 1
    child = child_node!(policy, node, branch)
    outcome = sample_rollout(policy, child)
    if outcome === true
        node.visits += 1
        branch.visits += 1
    elseif outcome === false
        branch.admissible = false
    end
    backup!(node)
    outcome === true && return true
    # Failed samples do not prove that the remaining actions are infeasible.
    return any(b -> b.admissible, node.branches) ? nothing : false
end

function cleanup!(policy, node)
    node.terminal && return terminal_feasible(policy, node)
    for branch in node.branches, child in branch.children
        child.visits > 0 && cleanup!(policy, child)
    end
    # Only rollout rejects branches; unfinished work cannot invalidate a
    # completed plan, and remains available for subsequent exploration.
    return !isnothing(backup!(node))
end

function build_search_tree(policy::RiskBoundedInfoPolicy, state)
    policy.root = initialize_node(policy, state, policy.model, 0, 0.0, 0.0)
    deadline = time() + policy.solver.time_budget
    # Always attempt one rollout, including when the requested budget is zero.
    while true
        sample_rollout(policy, policy.root) === false && break
        (policy.root.terminal || time() >= deadline) && break
    end
    cleanup!(policy, policy.root)
    return policy.root
end
