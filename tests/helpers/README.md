# Test Helpers

Shared test utilities and fixtures used across the GNN test suite.

## Contents

- `__init__.py` — `get_test_data_dir()`, `get_sample_gnn_model()`, `load_sample_gnn_spec()`
- `script_loader.py` — `load_module_from_path()` importlib loader for standalone scripts
- `gnn_samples.py` — `SAMPLE_GNN_CONTENT` + `write_sample_gnn_markdown()`
- `mcp_stubs.py` — `MCPTools` in-memory MCP registry test double + `FakeMCPTime` injectable clock stub
- `mcp_census.py` — `EXPECTED_MCP_TOOLS` / `EXPECTED_MCP_MODULES` / `CENSUS_SOURCE` exact MCP census pin (from the committed `src/gnn/mcp/audit_report.json`; regenerate the audit and update the constants in the same PR that adds or removes tools/modules)
- `render_recovery.py` — `render_gnn_files()` for render-resilience tests
- `bar_labels.py` — checks that bar value labels sit a fixed 3 pt above their bars at every data scale (`figures_held_open`, `open_bar_figures`, `assert_bar_labels_offset_in_points`, `assert_png_bounded`); import the submodule directly, it is not re-exported because it imports `matplotlib.pyplot`

## Usage

```python
from tests.helpers import get_sample_gnn_model, load_sample_gnn_spec
from tests.helpers.render_recovery import render_gnn_files
```

## See Also

- [Parent: tests/README.md](../README.md)
