# cpomdp Framework Implementation

> **GNN integration layer**: Python / JAX continuous active inference
> **Framework**: [cpomdp](https://github.com/inferogenesis/cpomdp), the continuous-state counterpart of pymdp
> **Install**: optional extra, `uv sync --extra cpomdp`

## Overview

cpomdp runs continuous (linear-Gaussian) GNN models as active inference agents. It filters with an exact Kalman filter and picks each action by scoring every policy's expected free energy (G) over a declared, finite action set. The JAX, NumPyro and PyTorch continuous scripts use a fixed proportional controller instead. cpomdp is the backend that reports G per policy and splits it into its pragmatic (goal-seeking) and epistemic (information-seeking) parts.

## What it renders

Continuous models only: `F`, `H`, `Q`, `R`, `prior_mean` and `prior_cov`, with optional `goal_mean` and `control_gain`. The examples live in `input/gnn_files/continuous/`.

Discrete models (categorical `A`/`B`/`C`/`D`) get the render status `unsupported`. That is the same status PyMDP reports for a continuous model. It counts separately from failures, and nothing is executed.

A model with `goal_mean` and `control_gain` becomes an agent that acts. A model without them becomes a passive tracker that filters and never acts.

## Installation

cpomdp is an optional extra, not a core dependency:

```bash
uv sync --extra cpomdp
```

The pipeline's setup step runs `uv sync`, which removes extras it was not asked for. Keep cpomdp installed across a full pipeline run by passing the group:

```bash
python src/gnn/main.py --target-dir input/gnn_files/continuous --optional-groups cpomdp
```

Without the extra, rendering still works, because the renderer never imports cpomdp. The execute step reports cpomdp scripts as skipped, with the install command.

## Running

```bash
python src/gnn/main.py --only-steps "3,11,12,16" --frameworks cpomdp \
  --target-dir input/gnn_files/continuous --output-dir output
```

## Outputs

| Step | Path | Contents |
| --- | --- | --- |
| Render | `output/11_render_output/<model>/cpomdp/<Model>_cpomdp.py` | Standalone script |
| Execute | `output/12_execute_output/<model>/cpomdp/simulation_data/simulation_results.json` | Results |
| Analysis | `output/16_analysis_output/cpomdp/` | Posterior-mean trajectory, G per policy over time, free energy, `cpomdp_analysis.json` per model |

The results file holds GNN's shared continuous fields (`beliefs`, `posterior_cov`, `true_states_continuous`, `controls`, `rmse_vs_true` and others). It adds `efe_history`, one row of G per step with one value per policy, plus `epistemic_term`, `pragmatic_term`, `selected_policy_index`, `n_policies`, `cost_per_cycle` and `search_warrant`. The full list is in [`src/gnn/render/cpomdp/SPEC.md`](../../../src/gnn/render/cpomdp/SPEC.md).

## Options

The pipeline renders with the defaults below. Other values need the Python API, `render_gnn_to_cpomdp(spec, path, options)`.

| Option | Default | Meaning |
| --- | --- | --- |
| `control_mode` | `efe` | `efe` applies the first action of the lowest-G policy. `parity` applies GNN's `u_t = control_gain * (goal_mean - mu_t)` on cpomdp's filter and reproduces the JAX backend on the same seed. |
| `action_scale` | `0.5` | Size of each action in the compass-rose action set. |
| `horizon` | `1` | Policy length. Every sequence of that many actions is scored each step. |
| `goal_precision` | `1.0` | How sharply the goal observation is preferred. |

## Limitations

- **The epistemic term barely matters yet.** The sensor noise `R` is fixed, so every action is equally informative. The choice of action is driven by the pragmatic term alone.
- **The agent zig-zags near the goal.** With the default `action_scale` of 0.5, the fixed step overshoots.
- **The analysis metrics are categorical.** The belief-entropy and confidence metrics are written for categorical beliefs. On Gaussian means they carry no meaning. NumPyro's continuous models have the same limitation.
- **cpomdp is pinned to a git commit.** `pyproject.toml` pins a cpomdp `main` commit through `[tool.uv.sources]` until the API it uses ships in a release.

## Source code

| Stage | Module |
| --- | --- |
| Render | [`src/gnn/render/cpomdp/`](../../../src/gnn/render/cpomdp/README.md) |
| Analysis | [`src/gnn/analysis/cpomdp/`](../../../src/gnn/analysis/cpomdp/README.md) |

Execution needs no cpomdp-specific module. The execute step runs the rendered script like the other Python backends.

## See also

- [NumPyro](numpyro.md), which renders the same continuous models with a Kalman filter and NUTS
- [GNN implementations index](README.md)
