# gnn.schema — Lightweight GNN Document Schema and Validation

## Overview

**Purpose:** A second, deliberately lightweight parsing/validation surface
for GNN documents: section checks, connection-edge extraction, state-space
parsing, and parameterization-vs-declaration shape cross-checks, plus a JSON
Schema for parsed model objects. Independent of the formal parser in
`gnn.parsers` by design.

**Pipeline step:** Consumed by CLI, LSP, watcher, analysis, multimodel, and
the semantic-fidelity pipeline.

**Category:** Parsing and validation.

**Status:** Maintained.

**Version:** 1.0.0

## Core Functionality

1. `validate_required_sections(content, *, file_path)` — required-section
   check (`GNN-E001`).
2. `parse_connections(content, *, known_variables, file_path)` — section-scoped
   edge extraction with cross-validation warnings (`GNN-W002`).
3. `parse_state_space(content, *, file_path)` — variable-block parsing with
   duplicate detection (`GNN-E004`).
4. `validate_matrix_dimensions(content, variables, *, file_path)` —
   InitialParameterization vs StateSpaceBlock shape cross-check
   (`GNN-E002`/`W003`).
5. `GNN_MODEL_SCHEMA` + `validate_gnn_object(obj)` — JSON Schema (draft
   2020-12) validation with a stdlib fallback when `jsonschema` is absent.
6. `REQUIRED_SECTIONS` — the required section-name set.

## Module Structure

- `__init__.py` — thin re-export surface (`__all__`).
- `parser.py` — section/connection/state parsers and the public compatibility surface.
- `types.py` — declaration and diagnostic dataclasses, re-exported by parser and package.
- `parameter_dimensions.py` — parameter assignment parsing and declared-shape checks.

## Dependencies

- Stdlib only at module scope (`ast`, `logging`, `re`, `collections.abc`,
  `dataclasses`, `typing`) — headless-importable by design.
- Lazy inside function bodies: `gnn.utils.runtime_safety.safe_eval.safe_literal_eval`,
  `jsonschema` (optional).

## Testing

```bash
uv run --extra dev python -m pytest tests/gnn/test_gnn_schema.py -q
```
