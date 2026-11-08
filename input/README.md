# Input

## Overview

GNN model specifications and configuration for the 25-step pipeline.
`config.yaml` supplies pipeline defaults. `model_family_manifest.json`
declares the acceptance profile per model family.

## Exemplar GNN set

The [exemplar index](gnn_files/INDEX.md) lists models by task and calls out
special contracts. Use [model-kind detection](../docs/gnn/gnn_syntax.md#model-kind-detection)
to classify a spec and [backend admission](../docs/gnn/implementations/README.md#backend-admission)
to choose a renderer. The corpus includes categorical, linear-Gaussian,
structural, and composed models.

The registry declares these default render targets by state space. Model-specific
admission also checks composition, inference task, and backend options.
Native execution depends on its separate runtime and artifact checks.

| Model kind | Default renderer declarations | Render status `unsupported` on |
| --- | --- | --- |
| Discrete categorical | PyMDP, RxInfer.jl, ActiveInference.jl, JAX, DisCoPy, PyTorch, NumPyro, Stan, bnlearn | ngc-learn |
| Continuous linear-Gaussian | JAX, NumPyro, PyTorch, Stan, RxInfer.jl, ngc-learn | PyMDP, ActiveInference.jl, DisCoPy, bnlearn |

The [example commands](gnn_files/INDEX.md#running-an-example) show how to
select a task folder and backend. Inspect
`output/11_render_output/render_processing_summary.json` and
`output/12_execute_output/summaries/execution_summary.json` for the run's
outcomes when using the default output directory.

## Other directories

`input/` now holds only the corpus tree and its configuration: every model
fixture lives under `gnn_files/`, so the model counts the manuscript publishes
cover the whole `input/` tree.

- `gnn_files/multiagent/multi_agent_coordination_acceptance.md` — one compact
  3-agent clustered mean-field fixture in the `gnn_files/multiagent/` target
  directory for the RxInfer and DisCoPy roadmap acceptance checks.
- `gnn_files/recursive/` — reserved target for bounded `--autonomous`
  proposal-loop runs; holds no committed models, so it adds nothing to the
  model counts.
