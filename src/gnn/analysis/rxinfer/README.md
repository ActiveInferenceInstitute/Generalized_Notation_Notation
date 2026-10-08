# RxInfer.jl Analysis Module

Framework-specific analysis for RxInfer.jl execution results.

Filesystem/JSON discovery lives in [`result_ingestion.py`](result_ingestion.py),
normalization and numerical diagnostics in [`metrics.py`](metrics.py), and the
analysis orchestration and categorical plot dispatcher in
[`analyzer.py`](analyzer.py). Existing analyzer imports remain available; the
split preserves normalization and inference semantics.

## Public Surface

- `generate_analysis_from_logs(execution_results_dir, output_dir, verbose=False)`
- Step 16 calls this analyzer for current `rxinfer` Step 12 outputs.

## Input Contract

The primary input is:

```text
output/12_execute_output/<model>/rxinfer/simulation_data/simulation_results.json
```

The JSON schema is `rxinfer_simulation_v1` and includes observations by modality, hidden states by factor, actions by control factor, beliefs by factor, expected free energy, policy posterior, `variational_free_energy`, `vfe_per_iteration`, validation, matrix provenance, and runtime metadata. The `variational_free_energy` and `vfe_per_iteration` fields are **per-iteration VFE traces** (length = INFERENCE_ITERATIONS) — the real convergence diagnostic from RxInfer's variational message passing, NOT per-step constants. Beliefs are **smoothed posteriors** from batch inference, not filtered (online) beliefs.

## Outputs

The analyzer writes plots under:

```text
output/16_analysis_output/rxinfer/
```

Generated plots include belief evolution, belief heatmaps, observation/state traces,
entropy, inference accuracy, action frequencies, belief convergence, belief trace,
free energy, observations, EFE per action heatmap, convergence diagnostics
(VFE slope / rate / iterations-to-convergence), and — for multi-factor models —
per-factor belief-trajectory small-multiples.

### GIF Animations (roadmap D7 + D4 + A7)

`generate_analysis_from_logs` also produces a publication-style animated GIF per
model (`<model>_rxinfer_animation.gif`) via `generate_gif_animation`: 2×3 panels
covering beliefs (per-factor marginals when `state_factors` declares >1 factor),
true/inferred states, the Bayesian graph model, VFE trace, EFE-per-action
heatmap, and the policy posterior. Each GIF carries a `.manifest.json`
reproducibility sidecar (GNN spec hash, Julia/RxInfer versions, seed, timesteps,
inference iterations, belief accuracy). `generate_dashboard` builds a single
self-contained HTML dashboard over a directory of GIFs + manifests.

### Cross-Framework Comparison (roadmap A6)

`compare_with_status(gnn_file, output_dir)` renders the same GNN model to six
backends — RxInfer.jl, PyMDP, ActiveInference.jl, JAX, PyTorch, and NumPyro —
from one parsed spec, runs all six, and writes `<model>_comparison.html` into
`output_dir` alongside a per-framework subdirectory of rendered scripts and
raw results. It returns the HTML path plus one `FrameworkRun` record per
backend, in display order;
`run_cross_framework_comparison(gnn_file, output_dir)` is the convenience
wrapper that returns just the path. The execute-module MCP tool
`run_cross_framework_comparison` wraps this entry point, and the Step-13 LLM
processor injects a compact per-backend status summary derived from the same
artifacts.

Backends whose dependencies are missing are skipped with an `unavailable`
receipt — the Python lanes probe dependency importability before rendering,
the Julia lanes probe for a `julia` binary — never reported as execution
failures.

## Per-Exemplar Visualization Set

The RxInfer analyzer produces the full per-exemplar visualization suite from
`rxinfer_simulation_v1` results. The set is rendered as matplotlib PNGs (dpi 300)
under `output/16_analysis_output/rxinfer/` and is complementary to the optional
Julia-native `Plots.jl` PNGs emitted at render/execute time.

| Visualization | File suffix | Description |
|---|---|---|
| `belief_evolution` | `_rxinfer_belief_evolution.png` | Belief probability trajectories per hidden state over time |
| `obs_vs_true` | `_rxinfer_obs_vs_true.png` | Observations vs true hidden states per time step |
| `belief_heatmap` | `_rxinfer_belief_heatmap.png` | 2D heatmap of belief mass over time (state × time) |
| `belief_entropy` | `_rxinfer_belief_entropy.png` | Belief uncertainty (bits) over time with max-entropy reference |
| `accuracy` | `_rxinfer_accuracy.png` | Cumulative inference accuracy from argmax beliefs vs true states |
| `action_frequencies` | `_rxinfer_action_frequencies.png` | Distribution of selected actions across the run |
| `belief_convergence` | `_rxinfer_belief_convergence.png` | Convergence of beliefs toward a stable posterior |
| `belief_trace` | `_rxinfer_belief_trace.png` | Full belief trace / trajectory per time step |
| `free_energy` | `_rxinfer_free_energy.png` | Expected (and variational, when present) free energy dynamics |
| `observations` | `_rxinfer_observations.png` | Observation trace / modality-level observation analysis |

Each plot is generated from the current Step-12 `rxinfer_simulation_v1` payload and
is written per-model. Like the render/execute-time artifacts, these figures are
best-effort: any plot that cannot be produced (e.g. missing matplotlib or absent
input arrays) is logged as a warning and skipped, never treated as a failure. The
set preserves the fields defined by the `rxinfer_simulation_v1` schema.

## Verification

```bash
uv run --extra dev python -m pytest tests/pipeline/test_pomdp_gridworld_cross_framework.py tests/pipeline/test_cross_framework_python_lanes.py tests/analysis/test_rxinfer_cross_framework.py -q --tb=short
```
