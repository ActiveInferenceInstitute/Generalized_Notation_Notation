# Test Helpers — Technical Specification

**Version**: 3.2.0

## Module API

- `get_test_data_dir()` → `Path` — Path to `tests/test_data/`
- `get_sample_gnn_model()` → `Path` — Path to the on-disk `sample_gnn_model.md` fixture
- `load_sample_gnn_spec()` → `dict` — Parses the sample model; falls back to a minimal spec dict when the file is missing
- `render_gnn_files(target_dir, output_dir)` → `dict` — Recovery-friendly bulk render used by `tests/pipeline/test_pipeline_recovery.py`
- `FakeMCPTime()` → `FakeMCPTime` — Injectable clock standing in for `gnn.mcp.mcp.time`; `advance(seconds)` moves the observable clock forward instantly (registry cache expiry and sliding-window rate limiter read `time.time()`), no production seam required
- `EXPECTED_MCP_TOOLS` → `int` — Exact pin of `tools_total` in the committed `src/gnn/mcp/audit_report.json` (165 at work time, 2026-11-08)
- `EXPECTED_MCP_MODULES` → `int` — Exact pin of `modules_total` in the committed `src/gnn/mcp/audit_report.json` (36 at work time)
- `CENSUS_SOURCE` → `str` — Path of the census file the pins derive from (`src/gnn/mcp/audit_report.json`); regenerate the audit and update both constants in the same PR that adds or removes tools/modules

## Recovery Pattern

`render_gnn_files()` tolerates a patched `numpy.typing` raising `RecursionError` (bumps the recursion limit and records `recursion_limit_adjusted`), globs with string paths to dodge pathlib recursion edge cases, and writes scaffold artifacts plus a summary dict with recovery actions.
