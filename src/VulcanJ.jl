module VulcanJ

# Public API exports: primary solver types and utility hooks
export RiskBoundedInfoMCTS,
    AbstractInfoMCTS,
    AbstractErgodicSolver,
    RiskBoundedInfoPolicy,
    ErgodicSolver,
    ErgodicPolicy,
    UnobservedPhenomenaModel,
    ThresholdPresence,
    presence_probability,
    set_environment_model!,
    update_environment_model!,
    initial_environment_model,
    conditional_observation_distribution,
    condition_environment_model,
    expected_information_gain,
    information_gain,
    generative_problem,
    observation_history,
    state_time,
    performance_alpha,
    fixed_alpha,
    plan_trajectory,
    initialize_gp_belief,
    get_initial_gp,
    compute_kl_reward,
    one_shot_ergodic_planner,
    kernel_ergodic_trajectory,
    simulate_info_path,
    plot_simulated_path,
    plot_information_reward_path

using Parameters: @with_kw
using Random: AbstractRNG, GLOBAL_RNG
import POMDPs
import POMDPs: solve, action
using POMDPs: Solver, Policy, MDP, POMDP, actions, isterminal, initialstate
import Plots

export get_initial_gp, add_obs_to_gp, get_failure_prob, posterior_phenomenon_prob, cellsites, horizon

include("interface.jl")
include("models/gaussian_process.jl")
include("models/unobserved_phenomena.jl")
include("mcts/types.jl")
include("mcts/risk.jl")
include("mcts/search.jl")
include("trajectory.jl")

include("ergodic.jl")
include("ergodic_path_planner.jl")

include("simulation.jl")
include("visualization.jl")

end
