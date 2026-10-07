# RxInfer Analysis - Agent Scaffolding

## Overview

Framework-specific analyzer for RxInfer.jl simulation results. Part of the Analysis module (Step 16). Consumes the genuine `@model` + `infer()` pipeline outputs: `rxinfer_simulation_v1` with real smoothed posteriors and a `variational_free_energy` trace populated with per-iteration VFE values (length = INFERENCE_ITERATIONS), enabling real convergence and free-energy analysis. Validation includes `inference_converged`, `vfe_present`, and `belief_entropy_ok`.

## Module Structure

```
analysis/rxinfer/
├── __init__.py          # Public API
├── analyzer.py          # Analysis orchestration and categorical plot dispatcher
├── metrics.py           # Result normalization, convergence and per-factor beliefs
├── result_ingestion.py  # Execution-summary discovery and JSON ingestion
├── animator.py          # Animated HTML visualizations
├── gif_animator.py      # Publication-style GIF animations + reproducibility manifest sidecar
├── dashboard.py         # Interactive HTML dashboard over the GIF batch (roadmap A5)
├── cross_framework.py   # Cross-framework comparison (roadmap A6)
├── README.md            # Human documentation
├── SPEC.md              # Module specification
└── AGENTS.md            # This file
```

### gif_animator.py

`generate_gif_animation(data, output_path, ...)` renders the 2×3
publication-style (white) GIF: beliefs, states, Bayesian graph model,
VFE, EFE-per-action heatmap (D6), and policy-posterior stackplot (D8).
For multi-factor results (`model_parameters.state_factors` with >1
size>1 factor) the top-left joint-belief panel becomes per-factor
marginal small-multiples (D4). Every GIF writes a `.manifest.json`
sidecar (A7: spec hash, Julia/RxInfer versions, seed, timesteps,
iterations, belief accuracy).

### dashboard.py

`generate_dashboard(gif_dir, output_path)` builds a single
self-contained HTML page over all GIFs + manifests with category
grouping and filtering.

### analyzer.py additions

Numerical helpers live in `metrics.py`, and filesystem/JSON discovery in
`result_ingestion.py`. Existing imports from `analyzer.py` remain available with
the same signatures. The categorical plot dispatcher remains cohesive rather
than being split to meet a line-count target; it consumes the same numerical
functions. The result reader imports no numerical or plotting dependencies.

- `compute_per_factor_beliefs(data)` — un-flattens joint posteriors into
  per-factor marginals using the `state_factors` echo in results JSON
  (C-order reshape; returns `{}` for flat models and for artifacts written
  before the echo existed, raises `ValueError` on inconsistent factor sizes).
- `_compute_convergence_diagnostics(...)` — VFE slope, convergence rate,
  iterations-to-convergence (D5), plotted alongside free energy.

### cross_framework.py

Implements roadmap **A6**: renders one GNN file to six backends — RxInfer.jl,
PyMDP, ActiveInference.jl, JAX, PyTorch, and NumPyro — from a single parsed
spec, executes each, and emits a self-contained HTML comparison.

- `compare_with_status(gnn_file, output_dir) -> tuple[str, list[FrameworkRun]]`
  — primary entry point; returns the HTML path plus one `FrameworkRun` per
  registered backend in display order. Raises `FileNotFoundError` for a
  missing GNN file.
- `run_cross_framework_comparison(gnn_file, output_dir) -> str` —
  convenience wrapper that discards the per-framework records.
- `render_comparison_html(model_name, runs, output_path) -> str` — pure
  renderer over `FrameworkRun` records, unit-testable without Julia.
- `FrameworkRun` — dataclass carrying `framework`, `status`
  (`success` / `validation_failed` / `render_failed` / `execution_failed` /
  `unavailable` / `invalid_results`), `detail`, and optional `results`.

Skip-not-failed receipts: the three Python lanes (jax, pytorch, numpyro)
probe dependency importability in-process before rendering, and the Julia
lanes probe the `julia` binary on PATH — a missing dependency yields an
`unavailable` record, never an execution failure. Two consumers build on the
same outcome records: the execute-module MCP tool
`run_cross_framework_comparison` (wraps `compare_with_status` and surfaces
per-backend statuses) and the Step-13 LLM processor, which reads the
comparison HTML plus the sibling per-framework `simulation_results.json`
files to inject a compact cross-framework meta section into its prompt.

Exit-code contract: only exit 0 with `simulation_results.json` is a clean
success; exit 1 with results is kept and flagged as `validation_failed`;
anything else is `execution_failed` with the stderr tail logged at error
level. The Python backends redirect results into the per-framework directory
via their output environment variables (`PYMDP_OUTPUT_DIR`,
`GNN_OUTPUT_DIR`, `PYTORCH_OUTPUT_DIR`, `NUMPYRO_OUTPUT_DIR`); both Julia
backends run under their committed `--project` environments resolved
relative to this file, not the CWD.
Both Julia subprocesses run with the shared Julia environment (`GKSwstype=100` headless-GR default; a `GKSwstype` set in the caller's environment wins).

## Key Functions

### analyzer.py

- `generate_analysis_from_logs(execution_dir, output_dir, verbose)` - Main entry point
- `_parse_rxinfer_outputs(filepath)` - Parse RxInfer outputs
- `_analyze_messages(data)` - Message flow analysis
- `_analyze_convergence(data)` - Convergence tracking
- `_generate_report(metrics)` - Report generation

## Integration Points

**Upstream:** Execute module (Step 12) produces RxInfer simulation results
**Downstream:** Report module (Step 23) consumes analysis outputs

## Dependencies

- pathlib, json, logging: Core Python
- numpy (optional): Numerical operations
- matplotlib (optional): Visualization

---

**Version:** 3.5.0
**Last Updated:** 2026-09-22
