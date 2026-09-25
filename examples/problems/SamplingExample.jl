module SamplingExample

import POMDPs, VulcanJ
using POMDPs: MDP
using VulcanJ: AbstractInfoMCTS, AbstractErgodicSolver, conditional_observation_distribution
using GaussianProcesses: GPE, MeanZero, SEIso
using Distributions: cdf
using Random: MersenneTwister

export SamplingState, SamplingProblem, setup, terrain, phenomenon_sites

struct SamplingState
    location::Matrix{Float64}
    history::Vector
end
struct SamplingProblem <: MDP{SamplingState,Symbol}
    sensor::Function
    steps::Int
    risk_scale::Float64
end
const directions = Dict(:n=>(0.,1.), :ne=>(1.,1.), :e=>(1.,0.), :se=>(1.,-1.),
                        :s=>(0.,-1.), :sw=>(-1.,-1.), :w=>(-1.,0.), :nw=>(-1.,1.))
terrain(X) = 1.4exp(-sum(abs2, X .- [0.75 0.75])/0.09) -
             0.7exp(-sum(abs2, X .- [0.25 0.8])/0.05)
location_after(s, a) = clamp.(s.location + 0.12reshape(collect(directions[a]),1,2), 0., 1.)
VulcanJ.extract_location(s::SamplingState) = s.location
VulcanJ.state_time(::SamplingProblem,s) = length(s.history)-1
VulcanJ.observation_history(::Union{AbstractInfoMCTS,AbstractErgodicSolver},::SamplingProblem,s) = s.history
# Numerical objective integration is independent of the phenomenon-cell partition.
VulcanJ.cellsites(::SamplingProblem) = [[x y] for x in range(0.,1.;length=21) for y in range(0.,1.;length=21)]
phenomenon_sites(::SamplingProblem) = [[x y] for x in 0.:0.2:1. for y in 0.:0.2:1.]
POMDPs.actions(::SamplingProblem,s) = keys(directions)
POMDPs.isterminal(p::SamplingProblem,s) = length(s.history)>p.steps
function POMDPs.gen(p::SamplingProblem,s,a,rng)
    X = location_after(s,a)
    y = p.sensor(X,rng)
    return (sp=SamplingState(X,[s.history;(location=X,observation=y)]),r=0.)
end
VulcanJ.generative_problem(p::SamplingProblem,m,rng) =
    SamplingProblem((X,r)->rand(r,conditional_observation_distribution(p,m,X)),p.steps,p.risk_scale)
VulcanJ.get_failure_prob(p::SamplingProblem,s,a) =
    p.risk_scale*(0.002+0.015exp(-sum(abs2,location_after(s,a)-[0.5 0.45])/0.04))
VulcanJ.posterior_phenomenon_prob(p::SamplingProblem,m::GPE,X) =
    1-cdf(conditional_observation_distribution(p,m,X),0.5)

function setup(;steps=60,risk_scale=0.,rng=MersenneTwister(7))
    problem = SamplingProblem((X,r)->terrain(X)+0.05randn(r),steps,risk_scale)
    X = [0.15 0.2]
    state = SamplingState(X,[(location=X,observation=problem.sensor(X,rng))])
    prior = GPE(zeros(2,0),Float64[],MeanZero(),SEIso(log(0.3),0.0))
    return problem,state,prior
end

end
