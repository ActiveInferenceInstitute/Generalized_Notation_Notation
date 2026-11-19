# Multi-agent trajectory planning model

The [standalone RxInfer example](README.md) plans paths in a shared environment.
The [production GNN renderer](../../../src/gnn/render/rxinfer/rxinfer_renderer.py)
generates Julia scripts for a separate [execution environment](../../../src/gnn/execute/rxinfer/).

## Model

[InferenceModel.jl](InferenceModel.jl) represents each agent with the state
`[x, vx, y, vy]` and the control `[ax, ay]`. A linear transition advances
the state, a projection extracts the position, and Gaussian factors tie the
initial and final states to the supplied goals. Distance functions in
[DistanceFunctions.jl](DistanceFunctions.jl) express obstacle clearance and
pairwise separation. [HalfspaceNode.jl](HalfspaceNode.jl) supplies the soft
inequality factor applied to those distances.

The inference driver uses variational message passing and linearizes the
nonlinear distance functions. Path and control posteriors are available as
`result.posteriors[:path]` and `result.posteriors[:control]`. The
`compute_diagnostics` function summarizes path means and variances, mean
controls, and maximum path uncertainty.

## Configuration and verification

[config.toml](config.toml) supplies model, environment, and experiment
settings for [config_loader.jl](config_loader.jl). The [README](README.md)
describes the entry points and the resolved environment.
[test_model.jl](test_model.jl) checks the native model on a bounded four-agent
case with three time steps and no plotting.
