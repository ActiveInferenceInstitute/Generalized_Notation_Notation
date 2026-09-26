# PyMDP Render Submodule

This module renders parsed GNN model specs into executable PyMDP-oriented
Python scripts for Step 11 outputs.

## Public API

Exported from `src/gnn/render/pymdp/__init__.py`:

- `render_gnn_to_pymdp(gnn_spec, output_path, options=None) -> (success, message, warnings)`

No additional public function is exported by this package.

## Key Implementation

- `pymdp_renderer.py`
  - `PyMDPRenderer.render_spec(...)`
  - `PyMDPRenderer.render_file(...)`
  - `PyMDPRenderer.render_directory(...)`
  - wrapper `render_gnn_to_pymdp(...)`

## Contract

- Input is a parsed GNN dictionary expected by the render pipeline.
- Required execution matrices are explicit: `A`, `B`, `C`, and `D`.
  Factored models are composed into a joint PyMDP contract with matrix
  provenance. A nonstationary transition declaration (`B_t` time-indexed
  or `B_regime` + `b_regime_schedule`) is exempt from the static-`B`
  requirement: the raw tensor passes through verbatim into the embedded
  spec and `run_pymdp_simulation` rebuilds the Agent per scheduled step;
  `mode: "standalone"` refuses such specs with an
  `unsupported-nonstationary:` receipt.
- Output is a Python script file at the target `output_path`.
- Return tuple structure is:
  - `success: bool`
  - `message: str`
  - `warnings: list[str]`

## Integration

- Called by the Step 11 render pipeline (`src/gnn/render/processor/`).
- Generated scripts are consumed by Step 12 execution (`src/gnn/execute/processor.py`).
- Visualization is not handled in render.

### Step 11 → Step 12 → Step 16 layout (PyMDP)

| Step | Typical paths |
|------|----------------|
| 11 Render | `output/11_render_output/<gnn_stem>/pymdp/<ModelName>_pymdp.py` |
| 12 Execute | `output/12_execute_output/<gnn_stem>/pymdp/simulation_data/simulation_results.json` (after collection from `.../pymdp/output/pymdp_simulations/...`) |
| 16 Analysis | `output/16_analysis_output/pymdp/<model_slug>/` (plots from `generate_analysis_from_logs`) |

Generated scripts resolve the repo root via `GNN_PROJECT_ROOT` (set by Step 12 when running subprocesses) or by walking upward to a directory containing `pyproject.toml` and `src/`, then add `src/` to `sys.path` before importing the canonical executor.

## Dependency Notes

- Runtime scripts target **pymdp 1.0.0 (JAX-first)**.
- Installation guidance:
  ```bash
  uv pip install 'inferactively-pymdp>=1.0.0'
  ```
- Generated scripts import:
  ```python
  import jax.numpy as jnp
  import jax.random as jr
  from pymdp.agent import Agent
  from pymdp import utils as pymdp_utils
  ```
- Each generated script sanity-checks for ``Agent.update_empirical_prior`` at
  runtime and exits fast with a clear message if an unsupported wheel is present.

## Render Modes

`PyMDPRenderer` accepts `options={"mode": ...}`:

- `"pipeline"` (default) — thin runner that delegates to
  `src.execute.pymdp.run_pymdp_simulation` (the canonical JAX rollout).
- `"standalone"` — fully self-contained pymdp 1.0.0 script with an inline
  rollout loop. Useful for sharing a runnable example that does not require
  this repository on `PYTHONPATH`.

## Verification

Relevant tests live in `tests/`, including render+execute integration and
PyMDP simulation pathway tests. Upstream ``Agent`` / ``utils`` methods used by
generated runners are also asserted in
`tests/execute/test_pymdp_1_0_0_upstream_api.py`.
