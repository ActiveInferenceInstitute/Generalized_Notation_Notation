# Specification: Input

## Design Requirements

`input/` is the data tree consumed by the pipeline; it contains no Python source. Its contract is defined by the files it holds and the modules that read them:

- **GNN spec files** under `gnn_files/<task folder>/*.md` follow the [normative syntax](../docs/gnn/gnn_syntax.md) and are discovered by `gnn.processing.discovery.is_model_source_path`, which excludes the `INDEX.md`, `AGENTS.md` and `README.md` scaffolds (plus `*.example.md` / `*.template.md`).
- **`config.yaml`** routes the run: pipeline steps and skip lists, Step 2 test mode (`fast_only`), timeouts, and per-step options. It is merged with CLI arguments by `src/gnn/utils/arguments/pipeline_config_merge.py`.
- **`model_family_manifest.json`** declares acceptance profiles per model family (`schema`, `acceptance_profile_defaults`, `families`) for `scripts/run_session_acceptance.py`.
- **`gnn_files/multiagent/multi_agent_coordination_acceptance.md`** is a fixture in the `gnn_files/multiagent/` target directory. **`gnn_files/recursive/`** is reserved for bounded `--autonomous` proposal-loop runs and holds no committed models. The corpus coverage tokens in `src/gnn/manuscript/variables.py` record their status.

## Model-kind contract

`render.pomdp_contract.detect_model_kinds` classifies the complete set of
model families. For the rules, see
[model-kind detection](../docs/gnn/gnn_syntax.md#model-kind-detection).
Rendering also checks the backend's composition and execution-contract
capabilities. See [backend admission](../docs/gnn/implementations/README.md#backend-admission)
for representative combinations and how refused pairs are recorded.

## Components

No classes or functions are exported from this directory. Consumers: `src/gnn/` (parsing and discovery), `src/gnn/render/` (code generation), `src/gnn/execute/` (execution), `src/gnn/utils/arguments/pipeline_config_merge.py` (configuration).
