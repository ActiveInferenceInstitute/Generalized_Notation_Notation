# PyTorch Renderer

`src/gnn/render/pytorch/` renders a parsed GNN/POMDP model into a **standalone PyTorch** simulation script.

## Usage

```python
from pathlib import Path
from gnn.render.pytorch import render_gnn_to_pytorch

success, msg, artifacts = render_gnn_to_pytorch(
    gnn_spec=parsed_spec,
    output_path=Path("output/11_render_output/model/pytorch/model_pytorch.py"),
    options={"num_timesteps": 10, "seed": 42},
)
```

## Output

The rendered artifact is a single `.py` file. When executed, it writes `simulation_results.json` under `PYTORCH_OUTPUT_DIR` (defaults to `.`).

## Dependencies

- render-time: `numpy`
- generated script run-time: `torch`, `numpy`
- optional capture at run time: `torchlens` (`uv sync --extra torchlens`)

## Module structure

Discrete models are emitted as `torch.nn.Module` components whose buffers and
submodules carry the GNN meaning they compute
([`discrete_script.py`](discrete_script.py); addresses in
[`sites.py`](sites.py) `SITE_DESCRIPTIONS`):

| Site | GNN meaning |
| --- | --- |
| `agent.A` … `agent.E` | likelihood, transitions, preferences, prior, habit |
| `agent.state_inference` | posterior over hidden states (`s`) |
| `agent.transition` | predicted next-state prior (`s_prime`) |
| `agent.efe`, `agent.efe.ambiguity`, `agent.efe.risk` | expected free energy (`G`) and its terms |
| `agent.policy`, `agent.action` | `softmax(log E - G)` (`π`) and the sampled action (`u`) |
| `process.A`, `process.B`, `process.D`, `process.observe`, `process.transition` | the generative process (environment) |

The agent and the process hold separate buffers with the same authored values,
so a trace distinguishes inference from environment flow. One
`torch.Generator` seeded with `RANDOM_SEED` drives every sample; the rollout is
reproducible from the seed alone.

`sites.gnn_variable_sites` maps each variable named in `## Connections` onto
these sites: matrices by canonical letter (`A`–`E`), `G` and its named terms,
and state, observation and control variables through the extractor's factor
roles when exactly one variable has the role. Every other variable is listed
in `GNN_UNMAPPED_VARIABLES` with a reason instead of being guessed.

## Habit and seed

- A declared `E` must be a strict probability vector with one entry per
  action; otherwise rendering fails. Without `E` the habit is uniform, which
  leaves `softmax(log E - G)` equal to `softmax(-G)`. Results record
  `habit.source` (`declared` or `uniform_default`) and `policy_history`.
- The seed is the `seed` render option, else `model_parameters.random_seed`
  (or `seed`), else 42; it must be a nonnegative 64-bit integer.

## TorchLens capture (opt-in)

Setting `GNN_TORCHLENS_CAPTURE=1` when the script runs (Step 12 children
inherit the environment) traces the whole rollout with
[TorchLens](https://github.com/johnmarktaylor91/torchlens). The script imports
`torchlens` only inside `_torchlens_capture`; nothing changes when the variable
is unset. The capture summary is stored under `torchlens` in
`simulation_results.json` and in `torchlens/torchlens_capture.json`:

- `status`: `disabled`, `unavailable` (with the install hint), `failed` (with
  the reason) or `captured`. Capture never alters the simulation results.
- `trace_reproduces_simulation`: the traced outputs equal the plain run
  exactly.
- `agent_step_validation` / `validated`: `tl.validate` perturbation replay of
  the deterministic agent step at the first and last timestep. The full
  rollout is not perturbation-validated because categorical sampling rejects
  perturbed inputs; exact reproduction covers it instead.
- `site_edges`: counted dataflow edges between traced sites.
- `intervention_witnesses`: the last-step policy with `agent.efe.ambiguity` or
  `agent.efe.risk` zero-ablated (`fork().do(...)`), with the KL from the
  unlesioned policy.
- Artifacts: `report.html` and `rollout_rolled.svg`; with
  `GNN_TORCHLENS_ARTIFACTS=full` also the portable `rollout.tlspec` bundle and
  `ops.csv`. Each artifact failure is recorded in `artifact_errors`.

Step 16 turns the summary into the declared-connections fidelity report
([analysis/pytorch](../../analysis/pytorch/README.md)).


## Discrete filtering contract

The first observation conditions the authored initial prior. Each later prior
is the previous posterior propagated by the previous selected action's transition
matrix, before conditioning on the new observation. The recorded beliefs are
posteriors. Zero or nonfinite predictive evidence raises an error; invalid result
validation is persisted and then raises an error, so an invalid trace cannot
complete successfully. This is categorical filtering and action scoring, not a
claim of equivalence to other backends' policy objectives.
