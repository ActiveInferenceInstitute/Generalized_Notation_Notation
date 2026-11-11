# Multi-agent Trajectory Planning with RxInfer.jl

This project demonstrates multi-agent trajectory planning using probabilistic inference with the RxInfer.jl framework. The approach allows multiple agents to navigate through environments with obstacles while avoiding collisions between agents.

> **⚠️ Retired TOML validation workflow.** This example was previously used
> as the baseline for the GNN-to-RxInfer **TOML** validation workflow, where a
> GNN-generated `config.toml` was substituted for a hand-written one as a
> drop-in replacement. That TOML-based approach is no longer supported. The canonical
> pipeline now renders genuine `@model` + `infer()` Julia scripts via
> `src/gnn/render/rxinfer/rxinfer_renderer.py` and executes them under the committed
> `Project.toml` + `Manifest.toml` environment in `src/gnn/execute/rxinfer/`
>(`julia --startup-file=no --project=src/gnn/execute/rxinfer <script>`,
> no runtime `Pkg.add`). This directory is retained as a standalone example.

## RxInfer 6 compatibility boundary

The local project now resolves RxInfer **6.1.0**. Both Halfspace definitions use
`@define_factor_node` / `@define_message_update_rule`, and nonlinear
approximations use `@algorithm`. The production environment's
[native checks](../../../src/gnn/execute/rxinfer/test_models.jl) verify the custom
rules directly. `test_model.jl` also checks finite native posteriors for four
agents and three timesteps without plotting. Run it from this directory with
`julia --startup-file=no --project=. test_model.jl`. This bounded check does not
qualify every experiment, visualization or modular entry point.
This standalone research example and its modular scripts
are separate from production GNN model execution; full trajectory/visualization
acceptance must be established independently. Run production GNN via the
[maintained integration guide](../README.md).

## Project Structure

### Core Files

- **[TrajectoryPlanning.jl](TrajectoryPlanning.jl)**: Main module that combines all components, providing a unified interface for the entire system.
- **[Environment.jl](Environment.jl)**: Defines environment structures (obstacles, boundaries) and agent properties.
- **[Models.jl](Models.jl)**: Contains the probabilistic models, including the state-space formulation and Halfspace nodes for constraints.
- **[Visualizations.jl](Visualizations.jl)**: Visualization functions for rendering environments, agents, and trajectories.
- **[Experiments.jl](Experiments.jl)**: Functions for running experiments with different configurations.

### Execution Scripts

- **[run_experiment.jl](run_experiment.jl)**: Command-line script for running individual experiments with configurable parameters.
- **[MultiAgentTrajectoryPlanning.jl](MultiAgentTrajectoryPlanning.jl)**: Main control script for running the refactored version, generated from the original notebook.

### Documentation

- **[DOCUMENTATION.md](DOCUMENTATION.md)**: Comprehensive technical documentation explaining the model and implementation.
- **[meta.jl](meta.jl)**: Project metadata and description.

### Project Configuration

- **[Project.toml](Project.toml)**: Project dependencies and configuration.
- **[Manifest.toml](Manifest.toml)**: Detailed dependency information for reproducibility.

### Superseded files

- **Original_Multi-agent Trajectory Planning.jl**: Earlier script revision (not shipped in this tree).

## Running the Project

### Prerequisites

- Julia 1.10 or newer (the committed manifest is resolved with Julia 1.13.2)
- RxInfer 6.1.0 / ReactiveMP 7, Plots, LinearAlgebra, LogExpFunctions, StableRNGs; dependencies are in the local Project/Manifest.

### Quick Start

1. Navigate to the project directory, and run the main script from there:

   ```bash
   cd docs/rxinfer/multiagent_trajectory_planning
   julia --startup-file=no --project=. -e 'using Pkg; Pkg.instantiate()'
   ```

2. Run a single experiment with default parameters:

   ```bash
   julia --startup-file=no --project=. run_experiment.jl
   ```

### Command-line Parameters for run_experiment.jl

The `run_experiment.jl` script accepts several command-line arguments:

```bash
julia --startup-file=no --project=. run_experiment.jl [--env=TYPE] [--seed=VALUE] [--output=PATH] [--no-intermediates]
```

Parameters:

- `--env=TYPE`: Environment type (door, wall, combined). Default: combined
- `--seed=VALUE`: Random seed for reproducibility. Default: 42
- `--output=PATH`: Custom output directory. Default: timestamped directory in results/
- `--no-intermediates`: Disable saving intermediate steps. Default: save intermediates

Examples:

```bash
# Run with door environment and seed 123
julia --startup-file=no --project=. run_experiment.jl --env=door --seed=123

# Run with wall environment and custom output directory
julia --startup-file=no --project=. run_experiment.jl --env=wall --output=my_results

# Run with combined environment without saving intermediates
julia --startup-file=no --project=. run_experiment.jl --env=combined --no-intermediates
```

## Output Files

The experiments generate several output files in the results directory:

1. **Animations**: GIF files showing agent trajectories
   - `trajectories.gif`: Main animation of agents navigating through the environment

2. **Visualizations**:
   - `control_signals.png`: Visualization of control inputs
   - `obstacle_distances.png`: Heatmap of distances to obstacles
   - `path_uncertainty.png`: Visualization of path uncertainty

3. **Data Files** (under `data/`):
   - `paths.csv`: Raw path data for each agent
   - `controls.csv`: Control signal data
   - `uncertainties.csv`: Path uncertainty data

4. **Documentation**:
   - `experiment_summary.txt`: Summary of experiment parameters and results
   - `experiment.log`: Detailed execution log

## Customization

### Environment Customization

To create custom environments, modify the environment definitions in `Environment.jl`. Environments are defined as collections of rectangular obstacles:

```julia
custom_environment = Environment(obstacles = [
    Rectangle(center = (x1, y1), size = (width1, height1)),
    Rectangle(center = (x2, y2), size = (width2, height2)),
    # Add more obstacles as needed
])
```

### Agent Customization

To customize agents, modify the agent definitions:

```julia
custom_agents = [
    Agent(radius = r1, initial_position = (x1, y1), target_position = (tx1, ty1)),
    Agent(radius = r2, initial_position = (x2, y2), target_position = (tx2, ty2)),
    # Currently the model supports exactly 4 agents
]
```

## Model Parameters

The core planning model has several important parameters:

- **State space matrices**:
  - `A`: State transition matrix `[1 dt 0 0; 0 1 0 0; 0 0 1 dt; 0 0 0 1]`
  - `B`: Control input matrix `[0 0; dt 0; 0 0; 0 dt]`
  - `C`: Observation matrix `[1 0 0 0; 0 0 1 0]`

- **Inference parameters**:
  - `nr_steps`: Number of time steps (default: 40)
  - `nr_iterations`: Number of inference iterations (default: 350)
  - `γ`: Constraint parameter (default: 1)

These parameters can be adjusted in the `path_planning` function in `Models.jl`.

## Performance Notes

- The inference process can be computationally intensive, especially with complex environments.
- Increasing `nr_iterations` improves solution quality but increases computation time.
- The solution quality depends on the random initialization controlled by the seed parameter.


## Documents in this folder

- [Multi-agent Trajectory Planning Visualizations](VISUALIZATION.md)
