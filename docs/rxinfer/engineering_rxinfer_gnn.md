# Engineering RxInfer.jl with Generalized Notation Notation (GNN)

## Overview

This document details how GNN specifies, renders, and executes genuine
`RxInfer.jl` models. The canonical renderer (`src/gnn/render/rxinfer/rxinfer_renderer.py`)
turns a GNN POMDP specification into an executable Julia script that defines a
generative model with `@model` and solves it with `infer()`. Execution is fully
reproducible via a committed Julia environment.

## The Genuine Pipeline

The pipeline has three main components:

1.  **The GNN Specification**: A `canonical_pomdp_v1` model description
    (`input/gnn_files/**`) with the canonical matrices `A, B, C, D` over
    states, observations, and actions.
2.  **The Rendered Julia Script**: `rxinfer_renderer.py` emits
    `output/11_render_output/<model>/rxinfer/<model>_rxinfer.jl`, defining a
    genuine generative model:

    ```julia
    @model function pomdp_model(y, A, B, D, u, T)
        s[1] ~ Categorical(D)
        y[1] ~ DiscreteTransition(s[1], A)
        for t in 2:T
            s[t] ~ DiscreteTransition(s[t-1], B[:, :, u[t-1]])
            y[t] ~ DiscreteTransition(s[t], A)
        end
    end
    ```

    Hidden states evolve via `DiscreteTransition` (the `B` matrices) and are
    emitted through the likelihood matrix `A`. This is genuine RxInfer.jl
    variational message-passing — not a hand-rolled step simulator.
3.  **The Runner / Environment**: `src/gnn/execute/rxinfer/` executes the script
    under a committed `Project.toml` + `Manifest.toml` pinning RxInfer 6.1.0.

### Inference

The rendered script calls `infer()` with `free_energy = true`:

```julia
result = infer(
    model = pomdp_model(A=A, B=B, D=D, u=action_seq, T=T_infer),
    data  = (y = observation_seq,),
    free_energy = true
)
```

`result.posteriors[:s]` yields real posterior beliefs and `result.free_energy`
feeds the `variational_free_energy` trace (genuine values, previously
`Float64[]`). EFE and policy selection remain custom logic outside RxInfer's
domain.

### Environment & Reproducibility

- `Project.toml` + `Manifest.toml` under `src/gnn/execute/rxinfer/` pin RxInfer 6.1.0
  and all dependencies.
- The runner invokes `julia --startup-file=no --project=src/gnn/execute/rxinfer <script>`.
- `setup_environment.jl` uses `Pkg.activate()` + `Pkg.instantiate()` — there is
  **no runtime `Pkg.add`**.
- Each script calls `Random.seed!(seed)` before inference and records the seed and
  script SHA256 in `runtime_metadata` (`uses_real_rxinfer: true`), giving
  byte-identical results across runs with the same seed.

### Validation

All 30 exemplar GNN files under `input/gnn_files/**` (27 discrete + 3
continuous, census: `input/gnn_files/INDEX.md`) render to and execute under
RxInfer.jl (30/30). Step 6 validation now includes `inference_converged` and
`vfe_present`, confirming genuine convergence of the variational free energy.

## Retired TOML approach

Earlier versions of the pipeline rendered a `config.toml` and used the
`multiagent_trajectory_planning/` example with a GNN-generated `config.toml` as a
drop-in replacement for a hand-written one. That path is retired:

- `toml_generator.py` and `render_gnn_to_rxinfer_toml` are gone — deleted, not deprecated.
  Nothing emits a `DeprecationWarning` because there is no callable left to emit one; the
  `rxinfer_toml` target is refused inside `render_gnn_spec`. What survives is the helper
  layer, migrated to `src/gnn/render/rxinfer/model_contracts.py` because topology contract
  tests pin it.
- The `Multiagent_GNN_RxInfer.jl` validation script and the TOML-based workflow it
  described are no longer the supported path.

The genuine `@model` + `infer()` pipeline above is the only supported render and
execute path.

## How to Run the Pipeline

```bash
# Render step
uv run --extra dev python src/gnn/main.py --only-steps "3,5,8,11,12,16" \
  --target-dir input/gnn_files --frameworks rxinfer --verbose
```

Execution happens automatically for `rxinfer` frameworks; results land in
`<model>/rxinfer/simulation_data/simulation_results.json`
(`rxinfer_simulation_v1`).

## Conclusion

GNN provides a single canonical source of truth that reaches genuine RxInfer.jl
inference through `@model` + `infer()`. The committed Julia environment and
seeded, digest-tracked execution make the result reproducible, modular, and
robust across models and runs.

There are connections to CEREBRUM and other topics, via GNN as well as
directly/separately.
