# cpomdp Renderer

Renders a **continuous (linear-Gaussian)** GNN model into a standalone [cpomdp](https://github.com/inferogenesis/cpomdp) script. The script filters with cpomdp's exact Kalman backend and scores every policy over a declared finite action set. Discrete POMDP specs get the render status `unsupported`.

## Usage

```python
from pathlib import Path
from gnn.render.cpomdp import render_gnn_to_cpomdp

ok, msg, artifacts = render_gnn_to_cpomdp(
    parsed_spec,
    Path("output/11_render_output/model/cpomdp/model_cpomdp.py"),
    options={"control_mode": "efe"},
)
```

## Options

| Option | Default | Meaning |
| --- | --- | --- |
| `control_mode` | `efe` | `efe` applies the first action of the lowest-G policy. `parity` applies GNN's `u_t = control_gain * (goal_mean - mu_t)` on cpomdp's filter and reproduces the `jax` backend on the same seed. |
| `action_scale` | `0.5` | Magnitude of the compass-rose actions. |
| `horizon` | `1` | Policy length H. Every sequence of H actions is scored each step. |
| `goal_precision` | `1.0` | Scale of the identity precision on the preferred observation. |

Specs without `goal_mean`/`control_gain` render as passive trackers with no control matrix and no policy search. The output schema is in [SPEC.md](SPEC.md).

## Dependencies

The renderer imports nothing from cpomdp. The generated script needs `cpomdp` and `jax` (`uv sync --extra cpomdp`).

See [AGENTS.md](AGENTS.md) for the design notes.
