# Specification: Generalized Notation Notation (GNN) Ecosystem

## Design Requirements
This repository comprises the unified `Generalized Notation Notation` (GNN) ecosystem. At its foundational architectural root, it is built to orchestrate the translation of conceptual Active Inference generative models (written in highly structured Markdown) into dynamically actionable and executable scientific artifacts over a 25-step execution pipeline.

The repository root governs real-implementation testing across all integrations, strict testing schemas, continuous documentation integration, and CI/CD security definitions. The environment handles multi-parameter rendering targets across the eleven computational engines declared in `src/gnn/render/framework_registry.py`: PyMDP, RxInfer.jl, ActiveInference.jl, JAX, DisCoPy, PyTorch, NumPyro, Stan, bnlearn, ngc-learn, cpomdp.

## Components
There are no python source packages instantiated directly within this root location. Instead, the root contains the infrastructural map that triggers the pipeline execution and validation layer:

1. **`src/`**: Master source tree for the 25-step `gnn` execution orchestrators.
2. **`docs/`**: Deep-linked, versioned, extensive framework mapping and cognitive systems documentation.
3. **`tests/`** (via `src/tests`): Real-implementation execution boundaries.
4. **`.github/`**: Declarative workflow integration interfaces for continuous validation.
5. **`scripts/`**: Specialized tooling.
6. **`input/` & `output/`**: The decoupled transactional zones mapping the inputs of configurations and `.md` models toward serialized, visualized, ML-computed deliverables.
7. Infrastructural Configs (`pyproject.toml`, `pytest.ini`, `uv.lock`): Defines environments and strict operational capacities using `uv`.

## Technical Rules
- The root tier must maintain 100% explicit module compliance manifesting exactly three primary documentation files per node (`AGENTS.md`, `README.md`, `SPEC.md`).
- Avoid exposing any `.py` script behaviors at the root; invoke logic strictly through explicit interfaces inside `src/`.
