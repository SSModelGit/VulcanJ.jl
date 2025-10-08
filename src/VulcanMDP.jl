module VulcanMDP

using POMDPs
using Random: AbstractRNG, GLOBAL_RNG
using FastGaussQuadrature: gausshermite
using SpecialFunctions: erf
using StatsBase: Weights, sample
using LinearAlgebra: ⋅

using AbstractTrees
using GaussianProcesses: GPE, MeanZero, SE
# using ..EnvironmentGP
using ..VulcanJ: VulcanNode
using ..UnobservedMutualInformation

"""Specify the dimensions associated with the state (for GP modeling purposes).

Should only accept one argument: the state (not the Node but the state directly)
"""
function dim end
dim(state::Matrix) = size(state)[2]
dim(state::Vector) = size(state)[1]

Base.show(io::IO, z::VulcanNode) =
    print(io, "state: ", z.state,
              " | parent: ", (isnothing(z.parent) ? "none" : z.parent.state),
              " | action: ", z.action,
              " | observation: ", (isempty(z.obsl) ? "none" : z.obsl[end]))

Base.show(io::IO, ::MIME"text/plain", z::VulcanNode) =
    print(io, "state: ", z.state,
              "\nparent: ", z.parent,
              "\naction: ", z.action,
              "\nobservations: ", z.obsl,
              "\nchildren: ", mapreduce(x->"\n\t- "*repr(x),*,z.children,init=string(size(z.children)[1])))

"""Return a function that represents a change in state corresponding to the stated action.

Should take two inputs:
    state node: Mainly for specialization, but can also be used for more complex action spaces.
    action: ID of the corresponding action (either Symbol or Integer)

Action IDs can be either symbols (for simpler systems) or integers (for complex systems).

get_action can also rely on helper functions (ex. `action_space`).
"""
function get_action end

"""Convenience function for mapping observation values

This allows a mapping between integers and Gauss-Hermite approximations of observation distributions.
It is helpful for doing information state comparisons without floating point issues.
"""
function obs_map end

"""In practice, Gauss-Hermite degrees of 5 are enough. Without specialization, Vulcan uses this.

Note that internally Gauss-Hermite defaults to the physicist's Hermite polynomial.

Returns:
    - A dict mapping integers 1-5 to a vector containing [abcissae, weight]
"""
obs_map(::VulcanNode) = let (ab,wt)=gausshermite(5); Dict(1:5 .=> ab), Dict(1:5 .=> wt); end

"""Ensures that the VulcanNode state matches needs of GPE
"""
function gpe_state end

"""Returns an integer representing a Gauss-Hermite sample estimate.

Relies on the keys of the obs_map.

Returns: y_j (sample estimate, as per Vulcan's mathematical formulation - Ben's master's thesis chapter 4.5.3)
"""
function sample_y_j end

function sample_y_j(n::VulcanNode)
    let (_,wmap)=obs_map(n)
        sample(GLOBAL_RNG, collect(keys(wmap)), Weights(values(wmap)./√π))
    end
end

"""Add child node (if it does not already exist) to parent node

Returns: newly created child node OR the existing child node
"""
function get_child end

function get_child(parent::VulcanNode, action::Union{Integer, Symbol})
    child_state = get_action(parent, action)(parent.state)
    child_obs = sample_y_j(parent)
    for c in parent.children
        if (c.state==child_state) && (c.obsl[end] == child_obs)
            return c
        end
    end
    child_gp = add_obs_to_gp(gpe_state(parent), obs_map(parent)[1][child_obs], parent.gp)
    # Adds observation to observation list, creates a child of same type as parent, and then adds to parent's child roster
    push!(parent.children, typeof(parent)(child_state, child_gp, parent, action, push!(copy(parent.obsl), child_obs)))
    parent.children[end]
end

"""Deprecated function for testing only.
"""
function addchild_deterministic!(parent::VulcanNode, action::Union{Integer, Symbol})
    child_state = get_action(parent, action)(parent.state)
    child = typeof(parent)(child_state, parent, action)
    push!(parent.children, child)
    child
end

mutable struct SimpleRisklessNode{T} <: VulcanNode
    state::T
    parent::Union{Nothing,SimpleRisklessNode{T}}
    children::Vector{SimpleRisklessNode{T}}
    action::Union{Nothing,Integer, Symbol}
    obsl::Vector{Any}
    gp::GPE

    function SimpleRisklessNode{T}(state,
                                   gp=nothing,
                                   parent=nothing,action=nothing,obsl=[],
                                   children=Vector{SimpleRisklessNode{T}}()) where T <: Vector
        new{T}(state,parent,children,action,obsl,(isnothing(gp) ? empty_gp(dim(state)) : gp))
    end
end
SimpleRisklessNode(state) = SimpleRisklessNode{typeof(state)}(state)

action_space(::SimpleRisklessNode, ::Symbol) = Dict(:up => s->s+[0.0,1.0],
                                                    :down => s->s+[0.0,-1.0],
                                                    :left => s->s+[-1.0,0.0],
                                                    :right => s->s+[1.0,0.0])

action_space(::SimpleRisklessNode, ::Integer) = Dict(1 => s->s+[0.0,1.0],
                                                     5 => s->s+[0.0,-1.0],
                                                     7 => s->s+[-1.0,0.0],
                                                     3 => s->s+[1.0,0.0],
                                                     2 => s->s+[1.0,1.0],
                                                     4 => s->s+[1.0,-1.0],
                                                     6 => s->s+[-1.0,-1.0],
                                                     8 => s->s+[-1.0,1.0])

get_action(node::SimpleRisklessNode, action::Union{Integer, Symbol}) = action_space(node,action)[action]

gpe_state(node::SimpleRisklessNode) = reshape(node.state, length(node.state), 1)

export VulcanNode, SimpleRisklessNode, dim, get_action, obs_map, sample_y_j, get_child, gpe_state

end