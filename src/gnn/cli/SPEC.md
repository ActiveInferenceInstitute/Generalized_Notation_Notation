# CLI Module — Specification

## Purpose

Provide a unified `gnn` CLI entry point that dispatches to pipeline module APIs.

## Requirements

1. **Subcommand routing**: 20 subcommands (`run`, `validate`, `parse`, `extract`, `render`, `report`, `reproduce`, `preflight`, `health`, `serve`, `templates`, `models`, `pull`, `watch`, `graph`, `gui`, `mcp`, `lsp`, `complexity`, `benchmark`)
2. **Lazy imports**: Each handler imports its target module only when invoked
3. **Standard exit codes**: 0=success, 1=error, 2=success with warnings/skipped
4. **Verbose mode**: `--verbose` / `-v` flag enables DEBUG logging globally
5. **Path management**: All handlers ensure `src/` is on `sys.path`

## Interface

```python
def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint. Returns exit code."""
```

## Constraints

- No state between invocations (pure CLI tool)
- All domain logic lives in target modules, not in CLI handlers
- Entry point registered in `pyproject.toml` as `gnn = "gnn.cli:main"`

## Output encoding

Each `main()` invocation uses UTF-8 for native `TextIOWrapper` stdout/stderr
streams so successful diagnostics, authored Unicode file paths and error text
survive locale code pages. Original encoding and error settings are restored
when the call exits, including argument-parser `SystemExit`. Imports do not
change stream settings; nonnative captures such as `StringIO` and closed streams
retain their existing behavior. The CLI does not replace or escape Unicode to
work around output encoding failures. Existing JSON serialization, public exit
codes, dispatch semantics and scientific processing remain unchanged.

Consumers capturing CLI text should decode it as UTF-8. The installed platform
acceptance driver does this explicitly and saves its CLI logs in UTF-8; it does
not depend on a global Python UTF-8 environment override.
