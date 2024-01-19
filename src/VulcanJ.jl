module VulcanJ

using Reexport
#import MCTS

# Write your package code here.
include("ascent_example.jl")
include("EnvironmentGP.jl")
include("InformationMDP.jl")

@reexport using .EnvironmentGP
@reexport using .InformationMDP

#################################
# POMDPs Dispatch Specializations
#################################

using POMDPs, POMDPTools

#= POMDPs.gen(::DDNOut{:sp}, m::InfoProblem, s::VulcanNode, a, rng) = make_successor(s,a,m,rng)
POMDPs.gen(::DDNOut{:r}, m::InfoProblem, s::VulcanNode, a, rng) = δmi(s) =#
POMDPs.actions(m::InfoProblem, s::VulcanNode) = valid_actions(s,m)
POMDPs.gen(m::InfoProblem, s::VulcanNode, a, rng) = (sp=make_successor(s,a,m,rng), r=δmi(s))
POMDPs.initialstate(m::InfoProblem) = SparseCat(map(a->InfoNode(m.Xinit.X,a,Δmutual_info_up(m.Xinit)), m.abcissae), m.weights/(√π))
POMDPs.discount(m::InfoProblem) = m.discount_factor

export trial_func

end
