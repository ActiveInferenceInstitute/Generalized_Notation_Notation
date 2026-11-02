# cpomdp Framework Implementation

> **GNN integration layer**: Python / JAX continuous active inference
> **Framework**: [cpomdp](https://github.com/inferogenesis/cpomdp), the continuous-state counterpart of pymdp
> **Status**: experimental; explicitly selected only
> **Install**: optional extra, `uv sync --extra cpomdp`

## Overview

cpomdp runs continuous (linear-Gaussian) GNN models as active inference agents. It filters with an exact Kalman filter and picks each action by scoring every policy's expected free energy (G) over a declared, finite action set. The JAX, NumPyro and PyTorch continuous scripts use a fixed proportional controller instead. cpomdp is the backend that reports G per policy and splits it into its pragmatic (goal-seeking) and epistemic (information-seeking) parts.

## What it renders

Continuous models only: `F`, `H`, `Q`, `R`, `prior_mean` and `prior_cov`, with optional `goal_mean` and `control_gain`. The examples live in `input/gnn_files/continuous/`.

Discrete models (categorical `A`/`B`/`C`/`D`) get the render status `unsupported`. That is the same status PyMDP reports for a continuous model. It counts separately from failures, and nothing is executed.

A model with `goal_mean` and `control_gain` becomes an agent that acts. A model without them becomes a passive tracker that filters and never acts.

## Installation

The optional extra pins the ordinary released wheel `cpomdp==0.5.0`:

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
| Analysis | `output/16_analysis_output/cpomdp/` | Posterior means with covariance uncertainty, G per policy over time, `cpomdp_analysis.json` per model |

The results file holds GNN's shared continuous fields (`beliefs`, `posterior_cov`, `true_states_continuous`, `controls`, `rmse_vs_true` and others). It adds `efe_history`, one row of G per step with one value per policy, plus `epistemic_term`, `pragmatic_term`, `selected_policy_index`, `n_policies`, `cost_per_cycle` and `search_certificate`, which carries the library's `expected`, `visited`, `warrant`, `action_set_size`, `horizon` and `action_set_version`. The full list is in [`src/gnn/render/cpomdp/SPEC.md`](../../../src/gnn/render/cpomdp/SPEC.md).

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
- Admission is capped at 4,096 policies, 32,768 step evaluations and 128 MiB estimated allocation per cycle. The estimate excludes interpreter/JIT overhead; a process deadline remains authoritative. Over-budget requests fail before allocation.
- Continuous uncertainty comes from posterior covariance. Categorical entropy and confidence remain absent with explicit reasons.
- Two-dimensional control uses nine actions; other dimensions use `2p+1`. Passive models perform no enumeration. Boolean/fractional horizons, nonfinite options and all-invalid scores fail explicitly.
- The released constructor adapter uses `dynamics_matrix`, `observation_matrix`, `dynamics_noise`, `observation_noise`, `prior`, and `control_matrix`. Source, script, dependency, policy and measured-resource receipts accompany results.

## Source code

| Stage | Module |
| --- | --- |
| Render | [`src/gnn/render/cpomdp/`](../../../src/gnn/render/cpomdp/README.md) |
| Analysis | [`src/gnn/analysis/cpomdp/`](../../../src/gnn/analysis/cpomdp/README.md) |

Execution needs no cpomdp-specific module. The execute step runs the rendered script like the other Python backends.

## See also

- [NumPyro](numpyro.md), which renders the same continuous models with a Kalman filter and NUTS
- [GNN implementations index](README.md)

## Source-declared finite controls (E3-FA1)

A flat continuous source can add this single-line Python-literal mapping to
`InitialParameterization` (or the same mapping to the parsed Python spec).
`goal_mean` and `control_gain` remain required; the gain is not used by EFE mode.

```text
cpomdp_control = {"contract": "gnn-cpomdp-finite-actions-v1", "version": "thruster-v1", "action_ids": ["forward", "reverse", "coast"], "control_units": ["newton"], "state_units": ["m/s", "m/s", "m/s"], "actions": [[0.3], [-0.15], [0.0]], "control_matrix": [[0.2], [-0.4], [0.7]]}
```

For `n` states and `p` controls, `control_matrix` is exactly `n × p` and every
action has `p` components. The transition is
`x_t = F x_(t-1) + control_matrix @ a_(t-1) + noise`. Units label each authored
control and state axis; the matrix maps control units into state increments.
These labels do not certify dimensional analysis. The prior is still over the
first observed state, with no initial predict step.

All seven keys are mandatory. The contract version is fixed above; `version`
identifies the author's action set. IDs must be unique nonempty strings and
retain source order. Equal vectors with different IDs remain distinct choices;
ties select the first authored policy in Cartesian-product order. At least two
actions are required by released cpomdp 0.5.0. Values must be finite nonboolean
numbers, with no string conversion, normalization or deduplication. Invalid
declarations refuse before replacing an existing artifact. Unknown keys and
versions refuse. These controls require `efe`; `parity` and explicit
`action_scale` overrides refuse. Without this declaration, the existing compass
rose and passive behavior apply. Other backends refuse the declaration.

Results retain the complete `finite_control_contract`, `action_set_version`,
`control_matrix`, `control_inputs` (in control coordinates), and
`policy_action_ids` for each selected policy. Existing `controls` remain state
increments for downstream continuous comparisons. `policy_score_components`
contains pragmatic and epistemic values for **every** enumerated policy and
timestep; existing selected-component fields remain available. The validated
model digest includes the complete control declaration, including labels.

Search admission allows at most 4,096 policies, 32,768 policy-step evaluations
per cycle, 262,144 across all timesteps, and 128 MiB estimated allocation. Limits
are checked before enumeration, without computing an unbounded power. The
estimate includes declared arrays and retained score histories; interpreter/JIT
overhead is separate, and the process watchdog remains authoritative. Passive
models allocate no planner.

Independent NumPy witnesses use source-authored asymmetric state, observation
and control dimensions, nonidentity sensors, nonzero means and correlated
covariances. They compare every posterior and every policy's EFE components
against the released native backend, and check duplicate-vector tie identity.
Changed control maps and an extra initial prediction must disagree. This is a
bounded numerical witness, not a universal extraction or equivalence proof.
