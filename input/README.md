# Input

## Overview

Gold-standard GNN model specifications consumed by the 25-step pipeline.
`config.yaml` routes every folder through the full pipeline; `model_family_manifest.json`
declares the acceptance profile per model family.

## Exemplar GNN set

The exemplar specs live under [`gnn_files/`](gnn_files/) (see its
[`INDEX.md`](gnn_files/INDEX.md) for the cold-start index). They fall into two
model kinds with different framework coverage:

| Model kind | Folders | Renders + executes on | Reported `unsupported` on |
|---|---|---|---|
| Discrete-state POMDP / HMM (categorical `A/B/C/D[/E]`, including factored, hierarchical, multi-agent and learning variants) | `basics/`, `discrete/`, `hierarchical/`, `learning/`, `multiagent/`, `pomdp_gridworld/`, `precision/`, `pymdp_scaling_study/`, `structured/` | PyMDP, RxInfer.jl, ActiveInference.jl, JAX, DisCoPy, PyTorch, NumPyro, Stan (bnlearn renders; execution needs the intentionally unlocked `bnlearn` package) | — |
| Continuous-state linear-Gaussian (`F/H/Q/R`, `prior_mean/prior_cov`, optional closed-loop `goal_mean/control_gain`) | `continuous/` | JAX, NumPyro, PyTorch, Stan, RxInfer.jl, cpomdp | PyMDP, ActiveInference.jl, DisCoPy, bnlearn — categorical backends (DisCoPy draws categorical string diagrams); the pipeline flags these as `unsupported`, never as failures |

`unsupported` is a distinct render status: it is excluded from success rates
and Step 12 never executes those frameworks for that model. Skips at Step 12
(`skipped`) mean a *toolchain* is missing on the machine (Julia, `torch`,
`cmdstanpy`/CmdStan), not that the model is unrepresentable.

Run everything with `uv run python src/gnn/main.py --target-dir input/gnn_files`
and read `output/11_render_output/render_processing_summary.json` and
`output/12_execute_output/summaries/execution_summary.json` for the live
counts; do not trust hard-coded numbers in prose.

## Other directories

`input/` now holds only the corpus tree and its configuration: every model
fixture lives under `gnn_files/`, so the model counts the manuscript publishes
cover the whole `input/` tree.

- `gnn_files/multiagent/multi_agent_coordination_acceptance.md` — one compact
  3-agent clustered mean-field fixture (relocated from the former
  `multi_agent_models/` directory), still a hand-runnable `--target-dir` target
  for the RxInfer and DisCoPy roadmap acceptance checks.
- `gnn_files/recursive/` — reserved target for bounded `--autonomous`
  proposal-loop runs; holds no committed models, so it adds nothing to the
  model counts.

No verification command in [`../TO-DO.md`](../TO-DO.md) targets either — they
all point at `gnn_files/` broadly. Each directory's own `README.md` records what
it is for.
