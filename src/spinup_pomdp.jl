using ..InformationMDP
using POMDPs, POMDPTools
using MCTS

#################################
# POMDPs Dispatch Specializations
#################################

# Defining actions as valid actions that don't result in positions already sampled
useful_actions(s,m) = filter(x->vec(s.X+x) ∉ eachcol(s.env.gp.x), valid_actions(s,m))
POMDPs.actions(m::InfoProblem, s::VulcanNode) = useful_actions(s,m)

# Generative transition and reward functions
POMDPs.gen(m::InfoProblem, s::VulcanNode, a, rng) = (sp=make_successor(s,a,m,rng), r=δmi(s))

# Initial state distribution
POMDPs.initialstate(m::InfoProblem) = SparseCat(map(a->InfoNode(m.Xinit.X,a,Δmutual_info_up(m.Xinit)), m.abcissae), m.weights/(√π))

# Discount factor
POMDPs.discount(m::InfoProblem) = m.discount_factor