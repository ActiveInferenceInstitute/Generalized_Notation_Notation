# Operations Module Specification

## Overview
Transport-neutral command semantics shared by every adapter: the `gnn` CLI
(`gnn.cli`), the HTTP API (`gnn.api.parity`) and the MCP template tools
(`gnn.template.mcp`). Each transport maps the same outcome onto its own exit
code or response envelope; none re-implements the operation.

## Components
- `__init__.py` - Lazy PEP 562 facade over the names below; importing
  `gnn.operations` loads no submodule.
- `commands.py` - `run_validation_checks` / `ValidationOutcome`,
  `build_parse_payload`, `preflight_severities`, `extract_payload_defect`
  (with `EXTRACT_DEFECT_*` codes), `find_render_artifact`,
  `render_processing_succeeded`, `publish_pipeline_report`.
- `templates.py` - Maintained template index (`TemplateRecord`,
  `list_templates`, `show_template`, `pull_template`). Template data stays in
  `src/gnn/cli/template_assets/` and is read through
  `importlib.resources.files("gnn").joinpath("cli")`, so `gnn.cli` is never
  imported and source URIs remain `package://gnn.cli/template_assets/...`.

## Invariants
- Never import `gnn.cli` or `gnn.api` (static AST test in
  `tests/architecture/test_dependency_direction.py` and the import-linter
  contract "shared operations sit below every transport adapter").
- `gnn.cli.commands` and `gnn.cli.templates` are compatibility facades that
  re-export these exact objects; identity is pinned by tests.
- Exit-code and envelope mapping is owned by the adapters: CLI `0/2/1`;
  HTTP `200` with `data.exit_code` `0/2`, `400` strict/path validation,
  `404` missing path, `409` pull collision or symlink destination.

## Key Exports
```python
from gnn.operations import run_validation_checks, list_templates, pull_template
```
