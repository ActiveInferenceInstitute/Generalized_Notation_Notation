# PyMDP Render Submodule Agent Guide

## Purpose

Translate GNN model specs into executable Python scripts for the PyMDP
execution path used by the pipeline.

## Canonical Public Surface

From `src/gnn/render/pymdp/__init__.py`:

- `render_gnn_to_pymdp(gnn_spec: Dict[str, Any], output_path: Path, options: Optional[Dict[str, Any]] = None) -> Tuple[bool, str, List[str]]`

If a function is not exported there, it should not be documented as public API.

## Internal Components

- `pymdp_renderer.py`
  - `parse_gnn_markdown(...)`
  - `PyMDPRenderer` class:
    - `render_file(...)`
    - `render_spec(...)`
    - `render_directory(...)`
    - `_generate_code(...)` — dispatches to pipeline or standalone template
  - Constructor takes `options={"mode": "pipeline" | "standalone"}`:
    * **pipeline** (default) emits a thin runner that delegates to
      `gnn.execute.pymdp.execute_pymdp_simulation`.
    * **standalone** emits a fully self-contained pymdp 1.0.0 script.
- `pymdp_templates.py`
  - `generate_pipeline_runner_script(ctx)` — pipeline runner
  - `generate_standalone_runner_script(ctx)` — standalone runner
  - Helper functions (`generate_file_header`, `generate_example_usage_template`)
    emit pymdp 1.0.0 JAX-first code.

## Data Flow

1. Receive parsed or file-backed GNN spec.
2. Build script text with embedded state-space parameters.
3. Write script into Step 11 output structure.
4. Return `(success, message, warnings)` for pipeline orchestration.

## Integration Boundaries

- Upstream: called by render processor (`src/gnn/render/processor/`).
- Downstream: scripts are executed by `src/gnn/execute/pymdp/`.
- This module does not run simulations and does not generate analysis plots.

## Documentation Rules For This Folder

- Keep signatures synchronized with real exports and real function parameters.
- Avoid documenting speculative helper APIs.
- Keep package naming consistent with repository setup:
  - install: `uv pip install 'inferactively-pymdp>=1.0.0'`
  - imports (JAX-first):
    ```python
    import jax.numpy as jnp
    import jax.random as jr
    from pymdp.agent import Agent
    ```
- All generated scripts embed a runtime check for
  `Agent.update_empirical_prior` and exit fast on unsupported (<1.0.0) wheels.

## Tests

Primary validation is in `tests/` integration coverage for render and
render→execute handoff. The generated script calls `execute_pymdp_simulation` /
`run_pymdp_simulation`, whose upstream `Agent` API is locked by
`tests/execute/test_pymdp_1_0_0_upstream_api.py`.
