# PyMDP Framework Implementation

> **GNN Integration Layer**: Python  
> **Framework Base**: `inferactively-pymdp` 1.0.4 (1.0.3 on Python 3.12+, where flax and ngclearn need `jax>=0.11.1`)  
> **Simulation Architecture**: Structured POMDP execution  
> **Documentation Version**: 2.0.0

## Overview

The GNN pipeline renders discrete POMDP-style model specifications into
PyMDP 1.0.0 runner scripts, executes those scripts through Step 12, and
analyzes the resulting `pymdp_simulation_v1` JSON in Step 16.

What this backend cannot express: continuous linear-Gaussian models
(`input/gnn_files/continuous/`) are reported with render status `unsupported`
for PyMDP (`supports_continuous: False` in `src/gnn/render/framework_registry.py`).
Such renders are counted separately from failures under
`unsupported_framework_renderings` and are not executed by Step 12.

PyMDP integration is strict about matrix provenance and schema shape. Single
factor models expose canonical `A`, `B`, `C`, `D`, and optional `E`; factored
models preserve named components such as `A_loc`, `A_rew`, `B_loc`, `B_ctx`,
and `D_ctx` before composing a PyMDP joint contract for execution.

## Architecture

The implementation is split across three current surfaces:

1. **POMDP extraction**: `src/gnn/extract/pomdp_extractor.py` builds a structured model
   spec with modality maps, state-factor maps, control-factor maps, matrix
   shapes, and `matrix_provenance`.
2. **Rendering**: `src/gnn/render/pymdp/pymdp_renderer.py` exposes
   `render_gnn_to_pymdp(...)` and writes PyMDP 1.0 runner scripts. The default
   script delegates to `src.execute.pymdp.run_pymdp_simulation`.
3. **Execution and analysis**: `src/gnn/execute/pymdp/simulation.py` writes
   `pymdp_simulation_v1`; `src/gnn/analysis/pymdp/analyzer.py` consumes only that
   schema for PyMDP-specific plots and summaries.

## Matrix Contract

PyMDP 1.0 expects a list of JAX arrays with a leading batch dimension:

| Symbol | Shape |
|---|---|
| `A[m]` | `(batch, num_obs[m], num_states[0], num_states[1], ...)` |
| `B[f]` | `(batch, num_states[f], num_states[f], num_controls[f])` |
| `C[m]` | `(batch, num_obs[m])` |
| `D[f]` | `(batch, num_states[f])` |
| `E` | `(batch, num_policies)` when present |

The GNN extractor records whether matrices came from direct single-factor
declarations, passive-model adapters, time-indexed transition projection, or
factored joint composition. A 2-D `B` matrix is passive/single-action dynamics,
not a list of actions. Passive HMM-style models omit `control_fac_idx`, because
PyMDP 1.0 requires every listed control factor to have more than one control.

## Execution Schema

Step 12 writes `simulation_results.json` with
`"schema_version": "pymdp_simulation_v1"`. Required content includes:

- observations by modality
- hidden states by state factor
- actions by control factor
- posterior beliefs by state factor
- expected and variational free energy traces
- policy posterior / action probabilities
- validation fields
- matrix provenance
- runtime metadata

Step 16 rejects older PyMDP result shapes for PyMDP analysis and reports clear
diagnostics instead of trying to recover flat traces.

## Source Code Connections

| Pipeline Stage | Module | Public Surface |
|---|---|---|
| Extraction | [extract/pomdp_extractor.py](../../../src/gnn/extract/pomdp_extractor.py) | `extract_pomdp_from_file(...)` |
| Rendering | [pymdp_renderer.py](../../../src/gnn/render/pymdp/pymdp_renderer.py) | `render_gnn_to_pymdp(...)` |
| Execution | [simulation.py](../../../src/gnn/execute/pymdp/simulation.py) | `run_pymdp_simulation(...)` |
| Analysis | [analyzer.py](../../../src/gnn/analysis/pymdp/analyzer.py) | `generate_analysis_from_logs(...)` |
| Visualization | [visualizer.py](../../../src/gnn/analysis/pymdp/visualizer.py) | `PyMDPVisualizer` |

## Verification

```bash
uv run --extra dev python -m pytest \
    tests/execute/test_pymdp_contracts.py \
    tests/execute/test_discrete_models_pymdp.py \
    tests/analysis/test_analysis_post_simulation.py \
    tests/visualization/test_visualization_matrices.py \
    -q --tb=short
```

## See Also

- **[GNN → pymdp Integration Guide](../../pymdp/gnn_pymdp.md)**: Local PyMDP
  1.0 contract and rollout details.
- **[Cross-Framework Methodology](../integration/framework_integration_guide.md)**:
  Framework comparison context.
- **[GNN Implementations Index](README.md)**: Return to the framework index.
