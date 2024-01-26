module VulcanJ

using Reexport
#import MCTS

# Write your package code here.
include("ascent_example.jl")
include("EnvironmentGP.jl")
include("InformationMDP.jl")
include("spinup_infomdp.jl")

@reexport using .EnvironmentGP
@reexport using .InformationMDP

export trial_func

end
