# VulcanJ

[![Build Status](https://github.com/SSModelGit/VulcanJ.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/SSModelGit/VulcanJ.jl/actions/workflows/CI.yml?query=branch%3Amain)

Re-implements the adaptive search algorithm in Ben's work. We're using Julia as the basis for this implementation, to leverage mature third-party libraries in machine learning and numerical computation; hopefully, with its compilability and type-checking, the code will be more robust to flaws, have a smaller executable file, and run faster, approaching an actual online capacity.

## Implemented Capabilities

- [X] Mutual Information on Unobserved Phenomena
- [X] Finite state space representations with information & observation transitions
- [ ] Forward Search
- [ ] Risk-aware Incremental Forward Search on Information
- [X] **Risk-unaware** MCTS on Information
- [ ] Risk-aware MCTS on Information

### External Dependencies

We extensively use the `POMDPs` library, as well as libraries that further extend its capabilities:
- `POMDPs`
- `POMDPTools`
- `MCTS`

For model-learning as we survey the environment, we use a recently-developed Gaussian Process library that aims for simple modeling over high expressivity:
- `GaussianProcesses`

We finally use a few additional libraries for some numerical operations:
- `LinearAlgebra`: dot product
- `Random`: (potentially seeded) random number generators
- `SpecialFunctions`: standard ERF calculation
- `FastGaussQuadrature`: efficient and fast calculation of Gauss-Hermite quadratures
- `StatsBase`: efficient weighted sampling of finite sets

### Defining VulcanJ capabilities through sub-problem classes

Vulcan, as a conceptual approach, is more like MCTS than a proper algorithm - it is a series of steps that can be combined in an abstract fashion to suit the specific problem you have at hand. Because of this, the actual library structure might fluctuate - where do we place the code that is readymade, versus the code that is specialized for particular purposes? Our approach here is to rely on multiple dispatch. We define the essence of a particular `MDP` instance to represent a sub-class of problems, and then further define a `spinup_<>` file to help "construct" the actual POMDP/MCTS requirements.

Right now, there is only one sub-class of problems implemented, represented by the `InformationMDP` module. This module is located in the file [InformationMDP.jl](src/InformationMDP.jl). It can represent problems with the following characteristics:

- States in space can be represented as a rectangular grid world of discrete cells.
  - Transitions are one-cell movements between states, and cannot exceed the confines of the grid space.
- The information metric is mutual information on unobserved phenomena.
- There is no risk in actions taken.
- There is a finite horizon.
- Observations are continuous.

Corresponding with the sub-class implementation, there is a [`spinup_infomdp.jl`](src/spinup_infomdp.jl) file that corresponds the definitions in the `InformationMDP` module to a `POMDPs` class instance. The instance is specialized on the special classes defined in `InformationMDP`, and is automatically exported with `VulcanJ`. To see an example of using this sub-class for an actual problem, look at the [volcano search example](test/volcano_search.jl) in the `test` folder.

### Additional support tools for VulcanJ (for developers)

To help ease the process of doing model learning, a helper module named `EnvironmentGP` is defined, in the appropriately named [`EnvironmentGP.jl`](src/EnvironmentGP.jl) file. It acts as a wrapper for doing Gaussian Process learning, taking observations and returning updated Gaussian Processes. It also provides a structure known as `EnvNode`, which is to be used by `VulcanJ` sub-problem classes for calculating rewards. (This is according to the specific way in which sequential rewards are calculated as defined for the Vulcan formulation in Ben's work.) An end-user should not have to interact with this module.

## General Usage

For now, proper documentation on usage is lacking. Look to the [volcano search example](test/volcano_search.jl) for an example of how to use the currently-implemented work.