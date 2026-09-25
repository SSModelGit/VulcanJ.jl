# VulcanJ

Information-based planning for Julia 1.11 or later, using the `POMDPs.jl`
solver interface. Environment models and information objectives are supplied
through multiple dispatch. Actual sensing, estimation, and execution belong
to the caller; `simulate_info_path` is an optional example simulator.

## Planning

```julia
policy = solve(solver, problem; objective=Val(:mutual_information))
set_environment_model!(policy, state, model; remaining_steps=30)
a = action(policy, state)
```

`RiskBoundedInfoMCTS` uses sampled POMDPs successors, realized information,
performance-guided risk bounds, and partial-policy cleanup. Its root context
accepts `risk_used`, `reference_reward`, and `alpha`. Riskless models have zero
failure probability and nonbinding bounds (the default `risk_budget=Inf`).
The solver defaults to `alpha_schedule=performance_alpha`, where `performance_alpha(t,T)=t/T`.
Finite `risk_budget` with `alpha_schedule=fixed_alpha` gives fixed-risk allocation. Positive alpha
scales the allowance using sequence information and the supplied reference.
There is one search and reward evaluator for these settings.
The deprecated scalar `alpha` constructor keyword is retained for unchanged
consumers and maps to a constant schedule; it does not select a planning mode.

`ErgodicSolver` supports conditional queries and full predicted trajectories.
Vector-action problems use the optimized controls directly; discrete-action
problems use their action selector. A problem-based trajectory returns predicted
problem states with paired actions and observations, and keeps optimized
reference positions in `reference_states`. Its target density stays fixed during
optimization; hypothetical observations condition the local model.

Custom solvers can subtype `AbstractInfoMCTS` or `AbstractErgodicSolver`, expose
the corresponding solver configuration properties, and specialize
`observation_history` to reuse the existing policies and algorithms.

MCTS `plan_trajectory(solver, problem, state, model, horizon)` replans after each
predicted outcome, updating its local model, running information mean, and risk.
Define `state_time(problem,state)` in mission-step units; MuKumari delegates to
`MuKumari.t(state)`. Root and trajectory calls accept `mission_start_time=0`.
The alpha schedule is evaluated at each root and remains fixed within its search.

Direct density planning remains available:

```julia
path, controls, losses, metrics =
    plan_trajectory(ergodic_solver, start, sites, density, bounds, horizon)
```

`one_shot_ergodic_planner` and `kernel_ergodic_trajectory` are deprecated aliases.
They preserve the previous return formats and numerical conventions used by
Arrodes, including the optional observation callback and Fourier backend.

## Model and problem methods

Implement the problem's POMDPs actions/termination methods and these hooks:

```julia
generative_problem(problem, model, rng)
conditional_observation_distribution(problem, model, state)
condition_environment_model(problem, model, state, observation)
information_gain(objective, problem, prior, posterior, state, observation)
observation_history(planner, problem, state)
```

`generative_problem` binds the supplied problem to the branch posterior without
mutating the caller. VulcanJ calls `POMDPs.@gen(:sp)` on that problem and reads the
new environmental measurement from `observation_history`. The problem's generator
owns successor construction and appends the predicted measurement to its history.
Neither search nor predicted trajectories require ground truth. Conditioning
returns a branch model without mutating its parent. Actual execution, measurement
acquisition, and posterior updates remain with the caller before replanning.
Define `get_failure_prob(problem,state,action)` for nonzero risk; it defaults to zero.

`RiskBoundedInfoMCTS.quad_order` remains accepted for existing callers but no
longer controls search; numerical quadrature belongs to information/model methods.

The empirical successor set widens with the square root of action visits. Cleanup
checks retained sampled continuations; it cannot certify unseen outcomes. Even a
zero time budget attempts one rollout before cleanup.

`expected_information_gain(objective,problem,model,state,order)` integrates the
same objective, or a model adapter supplies an analytic method. Ergodic density
construction also uses `cellsites(problem)` and `extract_location(state)`.
Scalar Gaussian outcomes use Gauss–Hermite quadrature; vector observations use
the Cartesian product of that rule after a covariance transformation. Order `q`
in dimension `d` has `q^d` outcomes. This quadrature is used for expected information and phenomenon probabilities,
not for search transitions or MuKumari motion.

Default representations include Gaussian processes and a model-backed
`UnobservedPhenomenaModel`. Construct the latter with
`UnobservedPhenomenaModel(problem, base_model, presence; history, quadrature_order=5)`,
where `presence(value, location)` is a conditional presence probability and
`history` contains existing location/observation records. Evaluation sites default
to `cellsites(problem)` and define nearest-site spatial cells (Euclidean distance,
first site on ties). Each observation marks its containing cell, including when
loading history; multiple sensor positions map separately. The base model keeps
the actual measurement coordinates. Spatial cells are distinct from the
observation-value quadrature controlled by `quadrature_order`.
VulcanJ integrates the presence relationship over the
base model's observation quadrature and delegates predictions and conditioning
to that model. The default assumes ideal local detection: sampled cells have
zero unobserved probability. Its objective sums Bernoulli KL changes across
sites; it is not joint spatial mutual information. History is not replayed into
the supplied posterior. Neither representation changes the search.

## Integrations and examples

The standalone package in `integrations/VulcanJIntegrations` loads all its
MuKumari and SCRIBE adapters when imported. Its main module directly includes
the `MuKumariIntegration` and `SCRIBEIntegration` submodules, organized under
`src/MuKumari/` and `src/SCRIBE/`. Each submodule explicitly exports its adapted
functions; the parent exposes them through `Reexport`, preserving the original
VulcanJ/POMDPs function bindings. Callers can import these functions explicitly
from `VulcanJIntegrations`. Loading the package activates all adapters:

```julia
using VulcanJ: VulcanJ
using SCRIBE: SCRIBE
using MuKumari: MuKumari
using VulcanJIntegrations: VulcanJIntegrations
```

Both frameworks are dependencies of the integration package; the core VulcanJ
package does not depend on either framework or the integration package.
Existing environments should run `Pkg.resolve()` after this dependency change. SCRIBE adapters use its
posterior measurement, conditioning, and information operations; the current
adapter represents a supplied posterior snapshot. MuKumari adapters use its
state accessor, observation histories, and shared physical propagation.
SCRIBE provides analytic expected field `:variance_reduction`; realized
reduction uses its uncertainty evaluations on the supplied prior and posterior. Besides `:mutual_information`, its
adapter provides scalar objectives `:differential_entropy`, `:total_variance`, `:mean_variance`,
`:logdet_information`, and `:minimum_information_eigenvalue`. The adapter rewards
decreases in the first three and increases in the last two, using SCRIBE's own
metric evaluation on the prior and conditioned information states.

Core demonstrations are indexed in [examples/README.md](examples/README.md):
seven examples and two small pipelines under `test/`, with no integration
dependencies. The thirteen integration examples have their own environment and
[index](integrations/VulcanJIntegrations/examples/README.md), grouped into
MuKumari, SCRIBE, and combined examples. They import the integration package's
adapters rather than defining local replacements.

```sh
julia --project=examples -e 'using Pkg: Pkg; Pkg.instantiate()'
julia --project=examples examples/ergodic_conditional_gp.jl
```

See `literature/VulcanJ_implementation_status.md` for verification evidence and
consumer-pipeline limits. Actual execution bookkeeping remains caller-owned.
