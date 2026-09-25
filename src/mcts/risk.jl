function performance_bound(policy::RiskBoundedInfoPolicy, depth, information)
    remaining_risk = policy.solver.risk_budget - policy.risk_used
    isinf(remaining_risk) && return remaining_risk
    uniform = depth / policy.remaining_steps * remaining_risk
    policy.alpha == 0 && return uniform
    return min(remaining_risk,
        ((1 - policy.alpha) + policy.alpha * information /
         (depth * policy.reference_reward)) * uniform)
end

terminal_feasible(policy, node) = node.depth == 0 ||
    node.sequence_risk <= performance_bound(policy, node.depth, node.sequence_information)
