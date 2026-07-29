# VulcanJ

[![Build Status](https://github.com/SSModelGit/VulcanJ.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/SSModelGit/VulcanJ.jl/actions/workflows/CI.yml?query=branch%3Amain)

Re-implements the adaptive search algorithm in Ben's work. We're using Julia as the basis for this implementation, to leverage mature third-party libraries in machine learning and numerical computation; hopefully, with its compilability and type-checking, the code will be more robust to flaws, have a smaller executable file, and run faster, approaching an actual online capacity.

## Current Architecture

The active implementation is a risk-bounded information MCTS solver on top of
`POMDPs.jl`. Spatial dynamics and risk belong to the MDP. The learned environment
is an independent model that VulcanJ treats as opaque and accesses through four
multiple-dispatch hooks.

### External Dependencies

The active solver implements its search directly against the `POMDPs.jl`
interface. `Random` supplies reproducible sampling, while
`FastGaussQuadrature` and `StatsBase` implement the optional scalar-Gaussian
outcome approximation. Plotting helpers use `Plots`.

VulcanJ includes a Gaussian-process environment model as its default toolkit
for users who do not supply another modeling system. The planner itself only
asks a supplied model for expected information, a conditional observation
distribution, and a conditioned model. The project also carries dependencies
used by historical source files that are not loaded by the active module.

### Defining VulcanJ capabilities through sub-problem classes

Vulcan is a sequence of planning operations rather than a single fixed
environment model. A problem therefore defines normal `POMDPs.jl` dynamics plus
small environment-model and risk interfaces. The environment model can be any
user-defined value with the required dispatched methods.

## General Usage

See the [environment-model interface](docs/environment_model_interface.md) for
the black-box model hooks, and the
[volcano search example](examples/volcano_search_risk_bounded.jl) for the
built-in Gaussian-process model.
