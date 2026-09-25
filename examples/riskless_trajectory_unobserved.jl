include("problems/SamplingExample.jl")
using .SamplingExample: setup, phenomenon_sites, terrain
using VulcanJ: RiskBoundedInfoMCTS, UnobservedPhenomenaModel, ThresholdPresence, condition_environment_model, observation_history, plan_trajectory, plot_simulated_path
using Random: MersenneTwister

rng = MersenneTwister(17)
problem,state,prior = setup(;steps=60,risk_scale=0.0)
solver = RiskBoundedInfoMCTS(;lookahead=12,time_budget=0.3,risk_budget=Inf,reference_reward=0.05,rng)
model = condition_environment_model(problem,prior,state,last(state.history).observation)
# Presence is a threshold event; observations resolve their containing cells.
model = UnobservedPhenomenaModel(problem,model,ThresholdPresence(0.5);
    sites=phenomenon_sites(problem),history=observation_history(solver,problem,state),quadrature_order=5)
# This is a hypothetical path; the supplied posterior/history remain unchanged.
result = plan_trajectory(solver,problem,state,model,60)
plot_simulated_path(problem,result.states,result.observations;
    save_path=joinpath(@__DIR__,"res","riskless_trajectory_unobserved","path.png"),
    observation_fn=terrain,bounds=(xmin=0.,xmax=1.,ymin=0.,ymax=1.),
    heatmap_resolution=51,marker_size=2,colorbar_title="Ground truth",
    title="riskless trajectory unobserved (predicted)")
