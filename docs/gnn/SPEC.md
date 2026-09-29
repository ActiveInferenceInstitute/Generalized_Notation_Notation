# Specification: Generalized Notation Notation (GNN) Reference

## Versioning

- **GNN language / syntax** is versioned in [gnn_syntax.md](gnn_syntax.md) (v1.1).
- **This documentation tree** uses bundle version **v2.0.0** on index pages (human-oriented hub; not the same field as the language version).
- **Installable package** version is **3.2.0** from the repo root [pyproject.toml](../../pyproject.toml).

## Design Requirements
The `docs/gnn` module does not contain execution logic mapping directly to the runtime graph. Instead, it serves as the **Static Source of Truth** for format specifications.

## Components
It governs the architectural parsing constraints across three main modalities:
1. **The Syntax Standard** ([`gnn_syntax.md`](gnn_syntax.md)): Defines variable encodings, connections mapping generative pipelines, and markup validation. This is the normative source; `reference/` and `language/` expand it without overriding it.
2. **The Output Target Frameworks**: How GNN compiles down to the eleven render backends (PyMDP, RxInfer.jl, ActiveInference.jl, JAX, DisCoPy, PyTorch, NumPyro, Stan, bnlearn, ngclearn, cpomdp) registered in [`src/gnn/render/framework_registry.py`](../../src/gnn/render/framework_registry.py). Which code a backend emits depends on the **model kind** detected structurally from the declared parameterization keys — see [`gnn_syntax.md` § Parameterization families](gnn_syntax.md#parameterization-families).
3. **Agent Integration** (`mcp` / `integration`): How external services can programmatically consume, augment, and output standard GNN files.

## Interfaces
GNN `.md` files are universally consumed starting from Step 3 (`3_gnn.py`) of the main pipeline. This entire documentation tree provides the exact syntax rules that Step 3 asserts against.

## Execution & Metric Logging Capabilities
The pipeline natively instruments execution layers (e.g., PyMDP) to extract rigorous behavioral metrics from the generated simulations. Key features include:
- **Trace Extraction**: Full simulation runs capture Variational Free Energy (VFE) and Expected Free Energy (EFE) profiles alongside belief distributions and state trajectories.
- **Parametric Sweep Stability**: Scaling tests across variable dimensions (N) and horizons (T) demonstrate robust behavioral metric generation. This is automated via the **PyMDP Scaling Orchestrator** (`scripts/experiments/run_pymdp_gnn_scaling_analysis.py`), which captures analytical truths such as the baseline O(ln(N)) uncertainty scaling characteristic of flat priors.
- **Analysis Visualization**: End-to-end processing (Step 16) automatically consumes execution outputs from Step 12 to generate performance charts, agent trajectory visualizations, and energy dynamics mapping.
