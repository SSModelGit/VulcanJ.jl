# Core examples

Run from the VulcanJ repository root with Julia 1.11 or later:

```sh
julia --project=examples -e 'using Pkg: Pkg; Pkg.instantiate()'
julia --project=examples examples/ergodic_conditional_gp.jl
```

Run all seven examples from the repository root:

```sh
julia --project=examples/ examples/run_all_examples.jl
```

From any working directory, specify the examples environment once:

```sh
julia --project=/path/to/VulcanJ/examples/ \
  -e 'include(joinpath(dirname(Base.active_project()), "run_all_examples.jl"))'
```

The runner executes examples alphabetically in separate processes, one at a
time, with one Julia thread and one BLAS thread. A failed example stops the run.
Results save to `VulcanJ/examples/res/<example>/path.png`, independently of the
working directory. The runner excludes itself and the `problems/` helpers.

This environment contains no MuKumari, SCRIBE, or integration-package dependency.
Each script runs a 60-step demonstration and saves its figure as `res/<example>/path.png`.
The shared `problems/SamplingExample.jl` defines only the demonstration problem
and model configuration. Each script calls VulcanJ simulation and visualization
functions directly. Solver and model choices are explicit in each script.

| File | Planning mode | Model |
| --- | --- | --- |
| [ergodic_conditional_gp.jl](ergodic_conditional_gp.jl) | Ergodic conditional | Gaussian process |
| [ergodic_trajectory_unobserved.jl](ergodic_trajectory_unobserved.jl) | Ergodic full trajectory | Unobserved GP; threshold 0.5 |
| [riskless_conditional_gp.jl](riskless_conditional_gp.jl) | Riskless conditional | Gaussian process |
| [riskless_trajectory_unobserved.jl](riskless_trajectory_unobserved.jl) | Riskless full trajectory | Unobserved GP |
| [performance_conditional_gp.jl](performance_conditional_gp.jl) | Performance-guided conditional | Gaussian process |
| [performance_conditional_unobserved.jl](performance_conditional_unobserved.jl) | Performance-guided conditional | Unobserved GP |
| [fixed_risk_conditional_unobserved.jl](fixed_risk_conditional_unobserved.jl) | Fixed-risk conditional | Unobserved GP |

Conditional examples construct a policy with `solve` and pass it to
`simulate_info_path(problem, policy, 60; initial_state=state, model)`. The supplied
model already incorporates the initial reading. The simulator executes through
`POMDPs.@gen`, conditions each new observation once, and updates risk and the
solver-configured performance reference before replanning.
Full trajectories call `plan_trajectory` and represent hypothetical observations;
they do not execute the path or alter the supplied posterior/history.

The GP starts without data and is conditioned on the acquired initial reading.
Unobserved models use nearest-site spatial cells and `ThresholdPresence(0.5)`,
which evaluates the Gaussian predictive tail directly. Arbitrary presence
functions remain supported through quadrature.
Riskless examples set the problem's failure probabilities to zero. Performance
examples use a specified spatial exposure-risk field and the default
`performance_alpha` schedule. The fixed-risk example explicitly supplies
`fixed_alpha`. Risk probabilities describe failure exposure; the displayed
conditional runs are nonfailure realizations. Risk and realized information
are printed alongside the path; planning may return a shorter feasible prefix.

These are executable research examples, not a `Test` suite. The two smaller
pipelines are run with the same environment:

```sh
julia --project=examples test/performance_guided_search.jl
julia --project=examples test/ergodic_trajectory.jl
```

MCTS examples use a 12-step planning lookahead and a 0.3-second search budget
per decision. Conditional ergodic examples use a 30-step lookahead; both planner
families request 60 actions so their coverage is visually comparable. The small
core pipelines remain short. Finite-risk examples use a mission budget of 0.6 and an exposure field ranging
from approximately 0.002 to 0.017 failure probability per action.

Spatial resolutions serve separate purposes: objective evaluation and ergodic
target construction use 21×21 sites (normalized spacing 0.05); figures evaluate
the background independently on a 51×51 grid. The default ergodic density
bandwidth is 0.075 in normalized coordinates. Unobserved-phenomena examples
explicitly retain their 6×6 phenomenon-cell partition through `sites`, independent
of both grids. These resolutions do not quantize model predictions or motion.

Figures use `plot_simulated_path` with explicit world bounds, a background
callback and label, and the existing per-example output path.
