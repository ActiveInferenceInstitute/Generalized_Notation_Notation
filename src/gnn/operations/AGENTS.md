# Operations Module - Agent Scaffolding

## Module Overview

**Purpose**: One implementation of each shared command, called by the CLI,
HTTP API and MCP adapters

**Category**: Infrastructure / transport-neutral operations

**Version**: [pyproject.toml](../../../pyproject.toml) (canonical)

---

## Rules

- Never import `gnn.cli` or `gnn.api` here, including inside functions or via
  `importlib`; `tests/architecture/test_dependency_direction.py` scans for it.
- Put new shared behaviour here and call it from each adapter; do not add a
  second implementation in an adapter.
- Keep exit-code and HTTP status mapping in the adapters
  (`gnn.cli.handlers_*`, `gnn.api.parity`).
- `gnn.cli.commands` and `gnn.cli.templates` must keep re-exporting the same
  objects for existing importers.
- Template data stays in `src/gnn/cli/template_assets/`; read it through
  `_package_data_root()`, never by importing `gnn.cli`.

## Tests

- `tests/architecture/` - direction and fresh-interpreter import surfaces
- `tests/api/test_api_parity.py` - CLI/API outcome and exit-code agreement
- `tests/cli/` - CLI behaviour through the facades
