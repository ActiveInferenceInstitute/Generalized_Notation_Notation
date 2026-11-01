# Extract Module Specification

## Overview
Headless POMDP extraction from GNN specification files behind a versioned, never-raising JSON envelope (schema 1.0.0). Stdlib-only at module scope, so `python -m gnn.extract` runs without the full pipeline stack.

## Components
- `__init__.py` - The entire public surface: `extract_to_json(path, *, strict_validation=True, on_error="lenient", compact=False) -> str` (success payload is `POMDPStateSpace.to_dict()` with no `status` key; failure is the `{"status": "error", "error": {...}}` envelope; never raises) and `main(argv=None) -> int` (CLI wrapper, exit 0 on success / 1 on failure)
- `pomdp_extractor.py` - Stdlib-only extractor: `POMDPExtractor`, `POMDPStateSpace` (payload versioned via `extraction_schema_version`), `extract_pomdp_from_file` / `extract_pomdp_from_content` (overloaded for optional structured-error collection), `canonicalize_pomdp` (canonical B order `next_state, previous_state, action`), `GNNExtractionError` (codes such as `GNN-E002`, `GNN-E006`, bare `GNN-E000`), `ON_ERROR_MODES` (`lenient` / `raise` / `collect`); heavy pipeline imports stay lazy inside call paths
- `__main__.py` - Preserves the documented headless entry `python -m gnn.extract FILE [--strict|--no-strict] [--compact]`
- `mcp.py` - Single `extract_pomdp` MCP tool (`extract_pomdp_mcp`, `register_tools`): thin wrapper over `extract_to_json` returning the versioned payload or the error envelope with `success: false`

## Cross-Repo Pin
The FEP_Formal bridge `verify-document` operation imports
`gnn.extract.pomdp_extractor.extract_pomdp_from_file` as its render route
(bridge contract `docs/other/fep_lean/bridge-contract.md`). The pin record
(`specs/gnn-bridge-w2-source-custody/source-pin.json`) lives in the FEP_Formal
checkout, not here: any move or rename of that import path must be coordinated
across both checkouts. The CLI `gnn extract FILE` subcommand also calls
`extract_to_json` (wired in `src/gnn/cli/__init__.py`).

## Invariants
- Module scope of `__init__.py` and `pomdp_extractor.py` imports only the standard library; the extractor module is imported lazily inside the call path.
- The public surface is exactly `__all__ = ["extract_to_json", "main"]`; extractor functions are importable by path only.
- `extract_to_json` always returns a JSON string; every failure path serializes the error envelope, and parameter-parse failures are recorded in `matrix_provenance` in every `on_error` mode.
- Envelope schema version 1.0.0: additive keys only, no changes to the `status`/`error` shapes.

## Key Exports
```python
from gnn.extract import extract_to_json, main
```

## Receipts
```bash
uv run --extra dev python -m pytest tests/cli/test_cli_extract.py \
  tests/gnn/test_pomdp_extractor_counts.py tests/gnn/test_pomdp_extractor_errors.py \
  tests/gnn/test_pomdp_extractor_isolation.py tests/gnn/test_pomdp_extractor_orientation.py \
  tests/gnn/test_pomdp_extractor_continuous.py tests/extract/test_extract_mcp_tools.py -q
```

---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
