include("problems/SamplingExample.jl")
using .SamplingExample: setup, terrain
using VulcanJ: ErgodicSolver, condition_environment_model, plot_simulated_path, simulate_info_path
using POMDPs: solve
using Random: MersenneTwister

rng = MersenneTwister(17)
problem,state,prior = setup(;steps=60,risk_scale=0.0)
solver = ErgodicSolver(;lookahead=30,optimizer_iters=30,max_speed=0.16,quad_order=3,rng)
model = condition_environment_model(problem,prior,state,last(state.history).observation)
policy = solve(solver,problem)
result = simulate_info_path(problem,policy,60;initial_state=state,model)
plot_simulated_path(problem,result.states,result.observations;
    save_path=joinpath(@__DIR__,"res","ergodic_conditional_gp","path.png"),
    observation_fn=terrain,bounds=(xmin=0.,xmax=1.,ymin=0.,ymax=1.),
    heatmap_resolution=51,marker_size=2,colorbar_title="Ground truth",
    title="ergodic conditional gp (executed)")
