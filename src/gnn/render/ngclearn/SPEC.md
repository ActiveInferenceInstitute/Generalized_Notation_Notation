# ngc-learn Renderer — Technical Specification

**Version**: [pyproject.toml](../../../../pyproject.toml) (canonical)

## Purpose

Step 11 codegen-only render backend for the ngc-learn framework: continuous
linear-Gaussian (LGSSM) GNN models are emitted as standalone ngc-learn
simulation scripts through the shared `render/continuous_script.py` generator
(Kalman numerics byte-identical to the `jax` backend); discrete POMDPs are
first-class unsupported.

## Architecture

```
ngclearn/
├── __init__.py            # Package export (render_gnn_to_ngclearn)
├── ngclearn_renderer.py   # Entry point: continuous routing + discrete refusal
├── AGENTS.md              # Backend scaffolding (contract, integration, testing)
├── README.md              # Usage, output, dependency notes
└── SPEC.md                # This specification
```

## Contract

`render_gnn_to_ngclearn(gnn_spec, output_path, options=None) -> (success, message, generated_files)`

- Continuous specs route through `processor._render_continuous_target` and are
  written via `render.continuous_script.generate_continuous_script(backend="ngclearn")`
- Discrete refusal message: `unsupported-discrete-state: discrete POMDP: ngclearn supports continuous linear-Gaussian models only`
- Generated scripts write `simulation_results.json` under `NGCLEARN_OUTPUT_DIR`
  (default: current directory)
- Render-time dependencies: none beyond the core pipeline (codegen-only —
  the renderer never imports `ngclearn`)
- Run-time dependencies of generated scripts: `ngclearn` >=3.2.2,
  `ngcsimlib` >=3.1.1, `jax` >=0.11.1 — `uv sync --extra ngclearn`
  (py3.12 marker-gated extra)
