module VulcanJ

using Reexport
#import MCTS

# Write your package code here.
include("ascent_example.jl")
# include("EnvironmentGP.jl")

"""
State of a Vulcan MDP (modified for code implementation):

- Split into two portions:
    - Easy-access MDP for MCTS
    - Tree representing explored searchs-space corresponding to states in above MDP
        - This is where the core Vulcan implementation lies
            - Specialize based on what aspects of Vulcan are implemented
    - Externally-stored mapping between MDP and Tree

Assumed characteristics of VulcanNode (abstract supertype):
    - `state` (representation of agent state)
    - `parent` (points to node representing previous state)
    - `action` (action agent took to enter current state from parent)
    - `obsl` (list of observations from start up until, but not including, current state)
    - `gp` (Gaussian Process based on observations thus far)
    - `children` (list of pointers to unique children nodes)
"""
abstract type VulcanNode end

struct VulcanProblem
    Xlims::Vector{Union{Integer, Float64}} # width and height limits of the world
    ū::Float64 # threshold for phenomena discovery
    p̄::Vector{Float64} # probability of phenom. when less and greater than u_bar
    vroot::Union{Nothing,VulcanNode}
    treemap::Dict # maps MDP states to VulcanMDP tree states

    function VulcanProblem(Xlims=[10.,10.], ū=0.5, p̄=[0.3,0.6], vroot=nothing,
                           treemap=Dict())
        new(Xlims, ū, p̄, vroot, treemap)
    end
end

export VulcanNode, VulcanProblem

include("UnobservedMutualInformation.jl")
include("VulcanMDP.jl")
# include("spinup_infomdp.jl")

# @reexport using .EnvironmentGP
@reexport using .UnobservedMutualInformation
@reexport using.VulcanMDP

export trial_func

end
