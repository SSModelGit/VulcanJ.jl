using VulcanJ: ErgodicSolver, plan_trajectory, coordinate_bounds
using Random: MersenneTwister
sites = [(x,y) for x in 0.:0.25:1. for y in 0.:0.25:1.]
density = [exp(-((x-0.8)^2+(y-0.7)^2)/0.08) for (x,y) in sites]
density ./= sum(density)
solver = ErgodicSolver(;optimizer_iters=15,max_speed=0.15,rng=MersenneTwister(4))
path,controls,loss,metric = plan_trajectory(solver,(0.1,0.1),sites,density,coordinate_bounds(sites),8)
println("Ergodic states / controls: ",(length(path),size(controls)))
println("Initial / final optimization loss: ",(first(loss),last(loss)))
