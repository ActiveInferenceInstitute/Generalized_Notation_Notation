# Render Module Specification

## Overview
Multi-framework rendering and code generation from GNN models.

## Components

### Framework Renderers
- `jax/jax_renderer.py` - JAX code generation
- `pymdp/pymdp_renderer.py` - PyMDP runner generation
- `discopy/translator.py` - DisCoPy translation
- `rxinfer/rxinfer_renderer.py` - Canonical RxInfer.jl renderer (genuine `@model` + `infer()`); the retired `toml_generator.py` emitter was deleted and its live helpers moved to `rxinfer/model_contracts.py` (the `rxinfer_toml` target is no longer supported)
- `stan/stan_renderer.py` - Stan program + cmdstanpy driver generation
- `pytorch/pytorch_renderer.py`, `numpyro/numpyro_renderer.py` - generator-backed continuous backends

### Core
- `generators.py` - Code generator utilities
- `processor.py` - Step 11 entry point
- `framework_registry.py` - Canonical framework inventory (`supports_continuous`, availability)

## Supported Frameworks
- `pymdp`, `rxinfer`, `activeinference_jl`, `jax`, `discopy`, `pytorch`, `numpyro`, `stan`, and `bnlearn`.

## Key Exports
```python
from gnn.render import process_render, JAXRenderer
```

The authoritative export surface is `src/gnn/render/__init__.py`.

The separately exported legacy `generate_pymdp_code` generator retains its
existing algorithms and sanitized artifact prefixes. Its model/source display
text is serialized as Python literals in documentation, metadata and report
slots. This is source-generation compatibility, not execution readiness or
scientific equivalence with the canonical PyMDP 1.0 runner.


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
