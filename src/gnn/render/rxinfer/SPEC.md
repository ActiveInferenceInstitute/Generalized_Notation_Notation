# RxInfer Renderer — Technical Specification

**Version**: [pyproject.toml](../../../../pyproject.toml) (canonical)

## Purpose

Generates Julia code using the RxInfer.jl genuine `@model` + `infer()` variational message-passing framework.

## Code Generation

- Maps GNN `canonical_pomdp_v1` model structure to an RxInfer factor graph specification
- Generates a genuine generative model (`@model function pomdp_model(y, A, B, D, u, T)`) with `Categorical` / `DiscreteTransition` nodes and an `infer()` call with `free_energy = true`
- `variational_free_energy` is populated with genuine values (previously `Float64[]`); EFE and policy selection remain custom Active Inference logic

## Output

- Julia script files (`.jl`) using the RxInfer API
- The former TOML parameter path (the retired `toml_generator.py` emitter, now deleted) is no longer supported; the genuine `@model` + `infer()` renderer is the only supported path

## Generated-Script Output Contract

Each rendered `.jl` script emits one required artifact plus optional guarded outputs.

### Required
- `simulation_results.json` — always written, schema `rxinfer_simulation_v1`
  (observations, hidden states, actions, beliefs by factor, expected free energy,
  policy posterior, matrix provenance, runtime metadata, validation, metrics).

### Optional (best-effort, never cause execution failure)
- `simulation.log` — human-readable chronological trace.
- `simulation_log.json` — machine-readable structured runtime events.
- `belief_evolution.png`, `efe_over_time.png`, `policy_posterior.png` — Julia-native
  `Plots.jl` figures emitted only when Plots rendering is available.

All optional artifacts are guarded: a missing dependency, failed write, or plotting
error is caught and logged, and the script continues to write `simulation_results.json`
normally. Step 12 `main()` returns non-zero only if `validation.all_valid` is false.

## Architecture

```
rxinfer/
├── rxinfer_renderer.py     # Core renderer — ModelKind dispatch, genuine @model + infer()
├── model_strategies.py     # Per-kind generators: flat (batch/online), hierarchical, factored, continuous LGSSM, Dirichlet learning
├── _strategies_multiagent.py  # Native stigmergic multi-agent generator (per-agent pomdp_model + shared env_signal trace)
├── model_contracts.py      # Matrix parsers, topology config, fail-closed validation
└── ...
```

## Dependencies

Target: `julia >= 1.8`, RxInfer 6.1.0 (pinned by the committed `Project.toml` +
`Manifest.toml` under `src/gnn/execute/rxinfer/`; execution via
`julia --startup-file=no --project=src/gnn/execute/rxinfer <script>`).
