# Input - Agent Scaffolding

## Overview

**Purpose**: The exemplar corpus and pipeline configuration consumed by the 25-step pipeline. This is a data directory, not a code module: it ships no Python, registers no MCP tools, and has no numbered pipeline script.
**Category**: Data / Configuration
**Status**: Maintained

---

## Contents

- `gnn_files/` — the maintained GNN exemplar specs, organised into task folders (`basics/`, `discrete/`, `hierarchical/`, `learning/`, `multiagent/`, `pomdp_gridworld/`, `precision/`, `pymdp_scaling_study/`, `structured/`, `continuous/`). [`gnn_files/INDEX.md`](gnn_files/INDEX.md) is the cold-start index and selection table.
- `config.yaml` — pipeline routing: default steps, skip lists, Step 2 test mode, timeouts, and per-step options.
- `model_family_manifest.json` — acceptance profile per model family (`schema`, `acceptance_profile_defaults`, `families`), consumed by `scripts/run_session_acceptance.py --manifest`.
- `gnn_files/multiagent/multi_agent_coordination_acceptance.md` — compact 3-agent clustered mean-field fixture in the `gnn_files/multiagent/` target directory.
- `gnn_files/recursive/` — reserved target for bounded `--autonomous` proposal-loop runs; holds no committed models, and no `TO-DO.md` verification command targets it (`--autonomous` runs against `gnn_files/`).

## Model kinds and framework support

Use [model-kind detection](../docs/gnn/gnn_syntax.md#model-kind-detection) and
the [backend admission guide](../docs/gnn/implementations/README.md#backend-admission)
when adding or updating an exemplar. Admission depends on the full kind set
and declared contract. Record any restricted or intentionally refused
backend routes in the exemplar index.

The registry declares these default render targets by state space. Model-specific
admission also checks composition, inference task, and backend options.
Native execution depends on its separate runtime and artifact checks.

| Model kind | Default renderer declarations | Render status `unsupported` on |
| --- | --- | --- |
| Discrete categorical | PyMDP, RxInfer.jl, ActiveInference.jl, JAX, DisCoPy, PyTorch, NumPyro, Stan, bnlearn | ngc-learn |
| Continuous linear-Gaussian | JAX, NumPyro, PyTorch, Stan, RxInfer.jl, ngc-learn | PyMDP, ActiveInference.jl, DisCoPy, bnlearn |

## Rules for agents

- Only spec files are discovered: `gnn.processing.discovery.is_model_source_path` excludes the `INDEX.md`, `AGENTS.md` and `README.md` scaffolds (plus `*.example.md` / `*.template.md`) from model discovery. Adding a new exemplar means adding a `.md` spec under an existing or new task folder and a row in `gnn_files/INDEX.md`.
- `--target-dir` must be a directory (a folder under `gnn_files/` or `gnn_files/` itself); single-file paths are not discovered.
- Do not write counts of exemplars, renders or executions into prose here. Live counts come from `output/11_render_output/render_processing_summary.json` and `output/12_execute_output/summaries/execution_summary.json`.
- Keep a single-block linear-Gaussian exemplar's parameters in that family. Adding categorical contract keys can make the model a hybrid composition.

## Related documentation

- [`README.md`](README.md) — corpus overview and run evidence
- [`SPEC.md`](SPEC.md) — data contract for this directory
- [`../src/gnn/AGENTS.md`](../src/gnn/AGENTS.md) — parser and discovery module that consumes these files
- [`../src/gnn/render/AGENTS.md`](../src/gnn/render/AGENTS.md) — renderer inventory and framework registry
