# JAX Templates Specification

## Overview

This folder defines **template strings** for JAX code generation:

- `GENERAL_TEMPLATE`
- `POMDP_TEMPLATE`
- `COMBINED_TEMPLATE`

These templates are **not** a pipeline step and do not implement an orchestrator. They are imported data used (optionally) by renderer code.

## API Surface

`src/gnn/render/jax/templates/__init__.py` must export exactly:

- `GENERAL_TEMPLATE`
- `POMDP_TEMPLATE`
- `COMBINED_TEMPLATE`

No runtime behavior beyond providing strings is expected.

## Formatting contract

Templates must remain valid Python **after** `str.format(...)` substitution.

Required keys across templates:

- `model_name`
- `timestamp`
- `source_file`
- `model_parameters`
- `n_states`
- `n_observations`
- `n_actions`

Additional keys:

- `discount` (POMDP/combined)
- `use_pomdp`, `use_neural` (combined)

## Dependencies

This module itself has no required runtime dependencies beyond Python.

Generated scripts (depending on template) import:

- `jax`, `jax.numpy`
- `optax`
- `flax.linen`

## Repo policy

- Python version: **3.12+** (repo policy)
- Any future wiring of these templates into `jax_renderer.py` must update the parent docs to reflect the dependency and behavior changes.
