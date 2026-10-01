# Input - Agent Scaffolding

## Overview

**Purpose**: The exemplar corpus and pipeline configuration consumed by the 25-step pipeline. This is a data directory, not a code module: it ships no Python, registers no MCP tools, and has no numbered pipeline script.
**Category**: Data / Configuration
**Status**: Maintained

---

## Contents

- `gnn_files/` — the maintained GNN exemplar specs, organised into task folders (`basics/`, `discrete/`, `hierarchical/`, `learning/`, `multiagent/`, `pomdp_gridworld/`, `precision/`, `pymdp_scaling_study/`, `structured/`, `continuous/`). [`gnn_files/INDEX.md`](gnn_files/INDEX.md) is the cold-start index and selection table.
- `config.yaml` — pipeline routing: which steps run, skip lists, Step 2 test mode, timeouts, and per-step options. Every folder under `gnn_files/` goes through the full pipeline.
- `model_family_manifest.json` — acceptance profile per model family (`schema`, `acceptance_profile_defaults`, `families`), consumed by `scripts/run_session_acceptance.py --manifest`.
- `gnn_files/multiagent/multi_agent_coordination_acceptance.md` — one compact 3-agent clustered mean-field fixture (relocated from the former `multi_agent_models/` top-level directory), still hand-runnable with `--target-dir`. No `TO-DO.md` verification command targets it.
- `gnn_files/recursive/` — reserved target for bounded `--autonomous` proposal-loop runs; holds no committed models, and no `TO-DO.md` verification command targets it (`--autonomous` runs against `gnn_files/`).

## Model kinds and framework support

`render.pomdp_contract.detect_model_kind` classifies each spec; `src/gnn/render/framework_registry.py` declares the ten frameworks and their `supports_continuous` flag.

| Model kind | Folders | Renders + executes on | Render status `unsupported` on |
|---|---|---|---|
| Discrete-state POMDP / HMM (categorical `A/B/C/D[/E]`; flat, factored, hierarchical, multi-agent, learning) | every folder except `continuous/` | PyMDP, RxInfer.jl, ActiveInference.jl, JAX, DisCoPy, PyTorch, NumPyro, Stan, bnlearn | — |
| Continuous-state linear-Gaussian (`F/H/Q/R`, `prior_mean/prior_cov`, optional closed-loop `goal_mean/control_gain`) | `continuous/` | JAX, NumPyro, PyTorch, Stan, RxInfer.jl, cpomdp | PyMDP, ActiveInference.jl, DisCoPy, bnlearn (categorical backends) |

`unsupported` is a first-class render status: it is excluded from success rates, listed under `unsupported_framework_renderings` in `output/11_render_output/render_processing_summary.json`, and Step 12 never executes those frameworks for that model. A Step 12 `skipped` means a toolchain is missing on the machine (Julia, `torch`, `cmdstanpy`/CmdStan), not that the model is unrepresentable.

## Rules for agents

- Only spec files are discovered: `gnn.discovery.is_model_source_path` excludes the `INDEX.md`, `AGENTS.md` and `README.md` scaffolds (plus `*.example.md` / `*.template.md`) from model discovery. Adding a new exemplar means adding a `.md` spec under an existing or new task folder and a row in `gnn_files/INDEX.md`.
- `--target-dir` must be a directory (a folder under `gnn_files/` or `gnn_files/` itself); single-file paths are not discovered.
- Do not write counts of exemplars, renders or executions into prose here. Live counts come from `output/11_render_output/render_processing_summary.json` and `output/12_execute_output/summaries/execution_summary.json`.
- A continuous exemplar declares only the linear-Gaussian block; do not add categorical `A/B/C/D` matrices to it, or the extractor will stop classifying it as `continuous`.

## Related documentation

- [`README.md`](README.md) — purpose and the same model-kind table
- [`SPEC.md`](SPEC.md) — data contract for this directory
- [`../src/gnn/AGENTS.md`](../src/gnn/AGENTS.md) — parser and discovery module that consumes these files
- [`../src/gnn/render/AGENTS.md`](../src/gnn/render/AGENTS.md) — renderer inventory and framework registry
