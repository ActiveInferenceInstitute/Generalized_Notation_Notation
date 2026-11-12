---
name: API
description: Callable discovery for the two GNN HTTP applications and job MCP wrappers
---

# API capabilities

Both applications require the optional `api` extra. Start the run surface with
`uv run --frozen --no-sync gnn serve --surface runs --host 127.0.0.1 --port 8000`
and the job surface with `uv run --frozen --no-sync python -m gnn.api.server
--host 127.0.0.1 --port 8000`. [README.md](README.md) records the implemented
routes; the [typed consumer contract](../../../docs/development/interop_api.md)
records canonical models, input/output shapes, errors, and version limits.

## MCP Tools

`gnn.api.mcp.register_tools(mcp)` registers the six entries in
`_MCP_TOOL_DEFINITIONS`:

- `gnn_submit_job` → `gnn_submit_job_mcp`
- `gnn_get_job_status` → `gnn_get_job_status_mcp`
- `gnn_cancel_job` → `gnn_cancel_job_mcp`
- `gnn_delete_run` → `gnn_delete_run_mcp`
- `gnn_list_jobs` → `gnn_list_jobs_mcp`
- `gnn_get_pipeline_tools` → `gnn_get_pipeline_tools_mcp`

`register_mcp_tools()` serializes those same definitions and adds its own
`version` field, currently the literal `1.0.0` rather than the package version.
The submit wrapper creates a pending record without scheduling execution;
callers start `gnn.api.processor.execute_job_async` explicitly. Inspect the live
surface with `gnn mcp list` and `gnn mcp info <name>`.

---
## Documentation

- **[README](README.md)**: HTTP interface.
- **[AGENTS](AGENTS.md)**: Maintainer guidance.
- **[SPEC](SPEC.md)**: Contract scope.
- **[SKILL](SKILL.md)**: This guide.
