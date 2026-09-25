include("problems/SamplingExample.jl")
using .SamplingExample: setup, terrain
using VulcanJ: RiskBoundedInfoMCTS, condition_environment_model, plot_simulated_path, simulate_info_path
using POMDPs: solve
using Random: MersenneTwister

rng = MersenneTwister(17)
problem,state,prior = setup(;steps=60,risk_scale=0.0)
solver = RiskBoundedInfoMCTS(;lookahead=12,time_budget=0.3,risk_budget=Inf,reference_reward=0.05,rng)
model = condition_environment_model(problem,prior,state,last(state.history).observation)
policy = solve(solver,problem)
result = simulate_info_path(problem,policy,60;initial_state=state,model)
plot_simulated_path(problem,result.states,result.observations;
    save_path=joinpath(@__DIR__,"res","riskless_conditional_gp","path.png"),
    observation_fn=terrain,bounds=(xmin=0.,xmax=1.,ymin=0.,ymax=1.),
    heatmap_resolution=51,marker_size=2,colorbar_title="Ground truth",
    title="riskless conditional gp (executed)")
