module VulcanJ

using Reexport
import MCTS

# Write your package code here.
include("ascent_example.jl")
include("measurement.jl")
include("InformationMDP.jl")

@reexport using .EnvironmentGP
@reexport using .InformationMDP

export trial_func

end
