import POMDPs, VulcanJ
using POMDPs: MDP, solve, action
using VulcanJ: RiskBoundedInfoMCTS, set_environment_model!
using Random: MersenneTwister

struct SmallSearch <: MDP{NamedTuple,Symbol} end
POMDPs.actions(::SmallSearch,s) = (:safe,:risky)
POMDPs.isterminal(::SmallSearch,s) = s.time>=3
VulcanJ.state_time(::SmallSearch,s) = s.time
VulcanJ.observation_history(::RiskBoundedInfoMCTS,::SmallSearch,s) = s.history
VulcanJ.generative_problem(p::SmallSearch,m,rng) = p
VulcanJ.get_failure_prob(::SmallSearch,s,a) = a==:safe ? 0.01 : 0.12
VulcanJ.condition_environment_model(::SmallSearch,m::Float64,s,y) = m+y
VulcanJ.information_gain(::Val{:mutual_information},::SmallSearch,prior::Float64,post::Float64,s,y) = post-prior
function POMDPs.gen(::SmallSearch,s,a,rng)
    y = a==:safe ? 1.0 : 3rand(rng)
    return (sp=(time=s.time+1,history=[s.history;(location=s.time+1,observation=y)]),r=0.)
end
problem = SmallSearch()
state = (time=1,history=[(location=1,observation=1.0)])
solver = RiskBoundedInfoMCTS(;lookahead=2,time_budget=0.05,risk_budget=0.1,rng=MersenneTwister(3))
policy = solve(solver,problem)
set_environment_model!(policy,state,0.0;remaining_steps=2)
println("Performance-guided action / alpha: ",(action(policy,state),policy.alpha))
println("Caller history remains unchanged: ",length(state.history)==1)

# A rejected terminal sample must yield to the root, leaving other continuations
# available. The next iteration should try the other, unattempted root action.
struct BranchingSearch <: MDP{NamedTuple,Symbol} end
POMDPs.actions(::BranchingSearch,s) = s.time==0 ? (:bad,:safe) : (:a,:b,:c)
POMDPs.isterminal(::BranchingSearch,s) = false
VulcanJ.state_time(::BranchingSearch,s) = s.time
VulcanJ.observation_history(::RiskBoundedInfoMCTS,::BranchingSearch,s) = s.history
VulcanJ.generative_problem(p::BranchingSearch,m,rng) = p
VulcanJ.get_failure_prob(::BranchingSearch,s,a) = s.time==0 ? 0.0 : (s.mode==:bad ? 0.03 : 0.001)
VulcanJ.condition_environment_model(::BranchingSearch,m,s,y) = m
VulcanJ.information_gain(::Val{:mutual_information},::BranchingSearch,prior,post,s,y) = 0.0
function POMDPs.gen(::BranchingSearch,s,a,rng)
    return (sp=(time=s.time+1,mode=s.time==0 ? a : s.mode,
        history=[s.history;(location=s.time+1,observation=0.)]),r=0.)
end
p=BranchingSearch()
s=(time=0,mode=:start,history=[(location=0,observation=0.)])
policy=solve(RiskBoundedInfoMCTS(;lookahead=2,time_budget=0.05,risk_budget=0.6,rng=MersenneTwister(3)),p)
set_environment_model!(policy,s,0.;remaining_steps=60)
root=VulcanJ.initialize_node(policy,s,0.,0,0.,0.)
safe=pop!(root.branches) # Force the first iteration down the unsuccessful action.
VulcanJ.sample_rollout(policy,root)===nothing || error("Unfinished search classified as infeasible")
root.branches[1].admissible || error("Rejected an action before exploring its alternatives")
push!(root.branches,safe)
VulcanJ.sample_rollout(policy,root)===true || error("Unattempted safe root action was not explored")
VulcanJ.cleanup!(policy,root) || error("Unfinished branch discarded the safe plan")
root.branches[root.best].action==:safe || error("Did not select the completed safe plan")
println("Failed rollout yields to root exploration; safe continuation retained: true")

# A new, unfinished outcome of that same action must not erase its completed plan.
branch=root.branches[root.best]
pending=VulcanJ.initialize_node(policy,branch.children[1].state,0.,1,0.,0.)
push!(branch.children,pending)
VulcanJ.cleanup!(policy,root) || error("Interrupted widening discarded a completed plan")
VulcanJ.child_node!(policy,root,branch)===pending || error("Unfinished outcome was not resumed")
println("Interrupted widening preserves the completed plan and resumes later: true")

rejected=VulcanJ.initialize_node(policy,s,0.,0,0.,0.)
pop!(rejected.branches)
for _ in 1:3
    VulcanJ.sample_rollout(policy,rejected)
end
rejected.branches[1].admissible && error("Proven infeasible branch was retained")
VulcanJ.cleanup!(policy,rejected) && error("Selected a risk-violating continuation")
println("Established risk violations still reject the branch: true")
