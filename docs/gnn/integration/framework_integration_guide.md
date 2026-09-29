# GNN Framework Integration Guide

**Version**: 3.2.0 (matches repo root `pyproject.toml`)
**Status**: Maintained

## Pipeline Integration

GNN framework integration is handled by **Steps 11 and 12** of the processing pipeline:

- **`src/gnn/11_render.py`** → Code generation for PyMDP, RxInfer.jl, ActiveInference.jl, DisCoPy, JAX, PyTorch, NumPyro, Stan, bnlearn, ngclearn, cpomdp
  - See: **[src/gnn/render/AGENTS.md](../../../src/gnn/render/AGENTS.md)** for rendering module details
- **`src/gnn/12_execute.py`** → Execution of rendered simulation scripts
  - See: **[src/gnn/execute/AGENTS.md](../../../src/gnn/execute/AGENTS.md)** for execution module details

**Quick Start:**

```bash
# Generate and execute the default framework set
uv run python src/gnn/main.py --only-steps "11,12" --target-dir input/gnn_files --output-dir output --verbose

# Execute specific frameworks only
uv run python src/gnn/12_execute.py --frameworks "pymdp,jax" --verbose
```

For complete pipeline documentation, see **[src/gnn/AGENTS.md](../../../src/gnn/AGENTS.md)**.

## Overview

This guide describes how GNN models actually flow through the framework integration
surface. There is no separate framework-agnostic intermediate-representation class in
the codebase — each renderer under `src/gnn/render/<framework>/` consumes the parsed GNN
model dictionary directly (as produced by `src/gnn/parsers/`) and emits framework-native
code. Per-framework rendering, execution, and troubleshooting detail lives in the
per-framework guides linked below; this document covers the parts of the pattern that
are shared across frameworks — the render/execute/analyze pipeline shape and the
cross-framework comparison step.

## Supported Frameworks

Step 11 renders to 10 registered targets (see `src/gnn/render/AGENTS.md`). Step 12 executes all 10 — bnlearn via `src/gnn/execute/bnlearn/` (its programs skip with an install hint until the `bnlearn` extra, or Rscript plus the R `bnlearn` package for `.R` scripts, is present; Stan via the cmdstanpy driver `<stem>_stan.py`, executor `src/gnn/execute/stan/`, skipped when cmdstanpy/CmdStan is absent). PyTorch and bnlearn are registry-gated unless installed manually. Continuous (linear-Gaussian) models execute on jax, numpyro, pytorch, stan and rxinfer only; the four categorical backends (pymdp, activeinference_jl, discopy, bnlearn) return render status `unsupported` for them, recorded in `render_processing_summary.json` under `unsupported_framework_renderings` and never handed to Step 12:

| Framework | Per-framework guide |
|-----------|----------------------|
| PyMDP | [implementations/pymdp.md](../implementations/pymdp.md) |
| RxInfer.jl | [implementations/rxinfer.md](../implementations/rxinfer.md) |
| ActiveInference.jl | [implementations/activeinference_jl.md](../implementations/activeinference_jl.md) |
| JAX | [implementations/jax.md](../implementations/jax.md) |
| DisCoPy | [implementations/discopy.md](../implementations/discopy.md) |
| PyTorch | [implementations/pytorch.md](../implementations/pytorch.md) |
| NumPyro | [implementations/numpyro.md](../implementations/numpyro.md) |
| Stan | [implementations/stan.md](../implementations/stan.md) |
| bnlearn | `src/gnn/render/AGENTS.md` (no standalone guide yet) |

Use `--frameworks` (plural) on `src/gnn/11_render.py` / `src/gnn/12_execute.py` to restrict to a
subset, e.g. `--frameworks pymdp,jax`, or `--frameworks lite` for the executor's
Python-oriented quick subset. The unified `gnn render` command uses singular
`--framework`; do not confuse the two CLI surfaces.

## Render → Execute → Analyze Pipeline per Framework

The GNN pipeline processes each framework through three stages. This section documents
framework-specific behavior at each stage.

### Pipeline Overview per Framework

| Framework | Render Target | Script Type | Executor | Data Extractor | Analysis Metrics |
|-----------|---------------|-------------|----------|----------------|------------------|
| **PyMDP** | `render/pymdp/` | `.py` | Python `subprocess` | `extract_pymdp_data()` | Beliefs, actions, free energy, observations |
| **RxInfer.jl** | `render/rxinfer/` | `.jl` | Julia `subprocess` | `extract_rxinfer_data()` | Posterior distributions + genuine VFE |
| **ActiveInference.jl** | `render/activeinference_jl/` | `.jl` | Julia `subprocess` | `extract_activeinference_jl_data()` | Full Active Inference fields from CSV |
| **JAX** | `render/jax/` | `.py` | Python `subprocess` | — | GPU-accelerated simulation output |
| **DisCoPy** | `render/discopy/` | `.py` | Python `subprocess` | `extract_discopy_data()` | Diagram executions, categorical outputs |

`extract_pymdp_data`, `extract_rxinfer_data`, `extract_activeinference_jl_data`, and
`extract_discopy_data` live in `src/gnn/execute/data_extractors.py` (raw stdout/stderr
parsing) and `src/gnn/analysis/framework_extractors.py` (post-simulation JSON-payload
normalization) — see those modules for the current field-level schema.

### PyMDP Pipeline Details

**Render:** `src/gnn/render/pymdp/` generates complete PyMDP Python scripts with A, B, C, D
matrices, `Agent` instantiation, a simulation loop, and result serialization.

**Execute:** Python subprocess with `PYTHONPATH` extended for PyMDP imports. Dependency
check: `import pymdp`.

**Analyze:** `extract_pymdp_data()` reads beliefs, actions, free energy, and observations
from JSON output. Supports reading from collected files in `output/pymdp_simulations/`.

### ActiveInference.jl Pipeline Details

**Render:** `src/gnn/render/activeinference_jl/activeinference_renderer.py` generates Julia
scripts using `ActiveInference.jl` with POMDP agent setup, environment initialization,
and a simulation loop.

**Execute:** Julia subprocess. The package-availability preflight runs
`using JSON, Distributions, StatsBase, ActiveInference` (see
`src/gnn/execute/processor/single.py`). As with RxInfer, Step 12 defaults `JULIA_PROJECT` to the
committed environment at `src/gnn/execute/activeinference_jl/`, whose `Project.toml` is
deliberately minimal: `ActiveInference` (0.1), `Distributions`, `JSON`, and
`StatsBase`. Reads output from `simulation_results.csv`.

**Analyze:** `extract_activeinference_jl_data()` performs CSV parsing to extract beliefs,
actions, free energy, observations, states, and policies, with numerical parsing and
error handling.

### RxInfer.jl Pipeline Details

**Render:** `src/gnn/render/rxinfer/rxinfer_renderer.py` is the canonical renderer. It does
not emit one flat model shape for every spec: it calls `detect_model_kind()`
(`src/gnn/render/pomdp_contract.py`) and dispatches by the detected `ModelKind` to a
per-kind strategy in `src/gnn/render/rxinfer/model_strategies.py`. Detection is
*structural* — it reads the `GNNSection` value, per-level/per-agent matrix key
patterns, explicit `nr_agents`/`num_factors`, `F`/`H`/`Q`/`R` keys, and
`dirichlet_[A-E]` keys — never free text. Every strategy emits a genuine Julia
script that runs `infer()` with `free_energy = true`.

The `@model` definitions themselves are *not* inlined into each generated script. They
live in the committed Julia package `src/gnn/execute/rxinfer/src/GnnRxInferModels.jl`, and a
rendered script pulls in the one it needs — for example
`using GnnRxInferModels: pomdp_model`, then
`infer(model = pomdp_model(A=A, B=B, D=D, u=model_actions, T=TIME_STEPS), …)`. Keeping
the models in a package is what lets the environment precompile them ahead of execution;
the five model functions it defines are `pomdp_model`, `continuous_pomdp_model`,
`hierarchical_pomdp_model`, `factored_pomdp_model`, and `learning_pomdp_model`. The
generated script contributes the matrices, the simulation loop, action selection, and
result serialization.

| `ModelKind` | Strategy | Generated model |
|---|---|---|
| `FLAT` | `FlatStrategy` | `pomdp_model` — batch smoothing by default; per-timestep filtering when online mode is selected (below) |
| `HIERARCHICAL` | `HierarchicalStrategy` | `hierarchical_pomdp_model` — native two-level: the context latent is coupled into the fast-state prior through the column-normalized `A_level2` from the GNN spec, which arrives as the model's `A_ctx` argument. Mean-field constraints *and* marginal initialization are both required on RxInfer 5.5. Three or more declared levels render as the documented joint composition |
| `FACTORED` | `FactoredStrategy` | `factored_pomdp_model` — native mean-field two-factor model with a multi-parent likelihood, `DiscreteTransition(s1, A_m0, s2)`, yielding per-factor posteriors |
| `CONTINUOUS` | `ContinuousStrategy` | `continuous_pomdp_model` — native linear-Gaussian state space built from `F`/`H`/`Q`/`R` plus `prior_mean`/`prior_cov` in `InitialParameterization`. Beliefs are posterior *means* alongside `posterior_cov`, and VFE validation is sign-agnostic because a Gaussian Bethe free energy is routinely negative |
| `LEARNING` | `LearningStrategy` | `learning_pomdp_model` — the likelihood `A` is learned jointly with the states as `DirichletCollection(dirichlet_A)`; `a_learning_improved` is a hard validation gate |
| `MULTI_AGENT` | `MultiAgentStrategy` | joint composition stamping the true kind. There is no native multi-agent `@model`; per-agent marginals are recovered downstream by `compute_per_factor_beliefs()` (`src/gnn/analysis/rxinfer/analyzer.py`) from the `state_factors` echo |

The strategy table is maintained alongside the code in
[`src/gnn/render/rxinfer/README.md`](../../../src/gnn/render/rxinfer/README.md). The
TOML-emitting `toml_generator.py` is retired and retained only as a warning surface —
`render_gnn_to_rxinfer_toml` raises a `DeprecationWarning`; do not build on it.

**Online mode:** `FLAT` models may run genuine online active inference instead of batch
smoothing. Select it with `inference_mode: online` in the GNN file's `ModelParameters`
section, or pass it as a render option; `batch` is the default and any other value is
rejected at render time. In online mode the generated script calls `infer()` per
timestep on the observation prefix, and the resulting *filtered* posterior — not a
smoothed joint posterior — drives action selection via `softmax(log E − γ·EFE)`.

**Execute:** Julia subprocess invoking
`julia --startup-file=no --project=src/gnn/execute/rxinfer <script>`. Step 12 defaults
`JULIA_PROJECT` to the committed environment for the framework being run
(`_build_execution_environment()` in `src/gnn/execute/processor/single.py`), so scripts resolve
their packages without an ambient environment; an explicitly set `JULIA_PROJECT` still
wins. The committed `Project.toml` + `Manifest.toml` under `src/gnn/execute/rxinfer/` pin
RxInfer 5.5.0 and define the `GnnRxInferModels` package, which precompiles the pomdp,
continuous, hierarchical, factored, and learning models loudly — a precompilation
failure surfaces rather than being swallowed. `setup_environment.jl` activates and
instantiates the environment (`Pkg.activate()` + `Pkg.instantiate()`, no runtime
`Pkg.add`). Each run records `Random.seed!(seed)` and the script SHA256 in
`runtime_metadata`, and generated scripts embed
`const B_TENSOR_ORDER = "next_state_previous_state_action"` so the transition-tensor
axis order is self-describing.

**Analyze:** `extract_rxinfer_data()` reads posterior distributions plus the genuine
`variational_free_energy` trace from `rxinfer_simulation_v1`
(`simulation_data/simulation_results.json`). Note that `true_states[t]` records the
state that *emitted* observation `t`, so it is timing-aligned with `beliefs[t]`.
Continuous models echo `state_factors` and `observation_modalities` as empty (a
linear-Gaussian model has no categorical factors) and add `controls`,
`kalman_filter_means` and `control_mode` (`closed_loop_proportional` when the GNN
declares `goal_mean`/`control_gain`, else `passive`).

### JAX Pipeline Details

**Render:** Generates JAX-based Python scripts with JIT compilation, GPU acceleration,
and automatic differentiation. Supports POMDP simulation with vectorized operations.

**Execute:** Python subprocess, same as PyMDP but without the PyMDP-specific dependency
check.

**Analyze:** Reads simulation output from `jax_outputs_*/` directories.

### DisCoPy Pipeline Details

**Render:** Generates categorical string diagrams using DisCoPy's rigid category
framework. Creates morphisms for state-observation connections and composes them via
sequential (`>>`) or tensor (`@`) products.

**Execute:** Python subprocess. Diagram evaluation produces categorical composition
results.

**Analyze:** `extract_discopy_data()` counts diagram executions and extracts categorical
outputs from `discopy_diagrams/`. Treats individual diagram evaluations as simulation
"steps".

### Cross-Framework Analysis

After individual framework execution, Step 16 (`src/gnn/analysis/analyzer.py`) and Step 23
(`src/gnn/report/`) perform comparative analysis and dashboard generation:

1. **`analyze_framework_outputs()`** — loads and normalizes results from all frameworks
   into standard JSON targets.
2. **`generate_framework_comparison_report()`** — generates comparison metrics (execution
   time, convergence, accuracy).
3. **`visualize_cross_framework_metrics()`** — native side-by-side metric visualizations.
4. **`generate_unified_framework_dashboard()`** (`src/gnn/analysis/visualizations.py`) —
   writes multi-panel PNG comparison plots across frameworks
   (`unified_belief_comparison.png`, `unified_action_efe_comparison.png`,
   `unified_entropy_comparison.png`). The filterable HTML compare dashboard
   (category and state-size filters plus a side-by-side compare mode) is
   `analysis.rxinfer.dashboard.generate_dashboard` (see
   [16_analysis.md](../modules/16_analysis.md)).
5. **`run_cross_framework_comparison()`**
   (`src/gnn/analysis/rxinfer/cross_framework.py`) — the RxInfer-anchored cross-framework
   comparison entry point, reachable as
   `analysis.rxinfer.cross_framework.run_cross_framework_comparison`.

Alongside the comparison surface, Step 16 emits the full visualization suite,
convergence diagnostics, per-factor belief recovery via `compute_per_factor_beliefs()`
(this is how multi-agent and factored runs get per-agent and per-factor marginals), and
one animated GIF per model accompanied by a `.manifest.json` reproducibility sidecar.

## Related Documentation

- **[PyMDP Integration Guide](../implementations/pymdp.md)**: Detailed PyMDP-specific documentation
- **[RxInfer Integration Guide](../implementations/rxinfer.md)**: RxInfer.jl integration
- **[DisCoPy Integration Guide](../implementations/discopy.md)**: Categorical diagram processing
- **[Troubleshooting Guide](../operations/gnn_troubleshooting.md)**: Error and troubleshooting reference

---

**Integration Guide Version**: 3.2.0
**Render targets**: PyMDP, RxInfer.jl, ActiveInference.jl, JAX, DisCoPy, PyTorch, NumPyro, Stan, bnlearn, ngclearn, cpomdp
**Step 12 executors**: PyMDP, JAX, DisCoPy, RxInfer.jl, ActiveInference.jl, PyTorch, NumPyro, Stan, bnlearn
**Status**: Maintained
