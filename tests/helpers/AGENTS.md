# Test Helpers Sub-module

## Overview

Shared, typed test utilities used across the GNN test suite: script loading,
canonical sample content, an MCP registry test double, sample-model path helpers,
and a render-recovery helper for test isolation.

## Architecture

```
helpers/
├── __init__.py            # Re-exports every public symbol below
├── script_loader.py       # load_module_from_path(): importlib loader for standalone scripts
├── gnn_samples.py         # SAMPLE_GNN_CONTENT + write_sample_gnn_markdown()
├── mcp_stubs.py           # MCPTools: in-memory MCP registry test double; FakeMCPTime: injectable clock stub
├── mcp_census.py          # exact MCP census pin (EXPECTED_MCP_TOOLS / EXPECTED_MCP_MODULES / CENSUS_SOURCE)
├── render_recovery.py     # render_gnn_files(): recovery-friendly bulk render
├── bar_labels.py          # bar value-label pixel-gap checks (submodule import only)
└── (path helpers in __init__.py for test_data/)
```

## Key Exports

- `load_module_from_path(name, path, sys_path=None)` — load a standalone script as a module; optional sibling-directory `sys.path` injection with automatic cleanup
- `SAMPLE_GNN_CONTENT` — canonical minimal POMDP GNN markdown (single source for the conftest `sample_gnn_*` fixtures)
- `write_sample_gnn_markdown(target)` — write the ontology-annotated sample markdown (creates parents)
- `MCPTools` — in-memory MCP registry test double (`register_tool` / `register_resource` / `execute_tool`); the conftest `test_mcp_tools` fixture returns an instance
- `FakeMCPTime` — injectable clock standing in for `gnn.mcp.mcp.time`; `advance(seconds)` moves the observable clock forward instantly (cache expiry, sliding-window rate limiter) instead of sleeping
- `EXPECTED_MCP_TOOLS` / `EXPECTED_MCP_MODULES` / `CENSUS_SOURCE` — exact MCP census pin from `mcp_census.py`, read from the committed `src/gnn/mcp/audit_report.json`; regenerate the audit and update the constants in the same PR that adds or removes tools/modules
- `get_test_data_dir()` / `get_sample_gnn_model()` / `load_sample_gnn_spec()` — path helpers for `tests/test_data/`
- `render_gnn_files()` — render every GNN file in a directory, capturing per-file results for recovery tests
- `tests.helpers.bar_labels` — `figures_held_open`, `open_bar_figures`, `assert_bar_labels_offset_in_points`, `assert_png_bounded`: assert bar value labels sit 3 pt above their bars at every data scale. Import the submodule directly; it is not re-exported because it imports `matplotlib.pyplot`

## Usage

```python
from tests.helpers import (
    EXPECTED_MCP_MODULES,
    EXPECTED_MCP_TOOLS,
    FakeMCPTime,
    MCPTools,
    SAMPLE_GNN_CONTENT,
    get_sample_gnn_model,
    load_module_from_path,
    load_sample_gnn_spec,
    write_sample_gnn_markdown,
)
from tests.helpers.render_recovery import render_gnn_files
```

## Parent Module

See [tests/AGENTS.md](../AGENTS.md) for the overall test architecture.

**Version**: 3.2.0
