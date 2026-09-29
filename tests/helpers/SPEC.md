# Test Helpers — Technical Specification

**Version**: 3.2.0

## Module API

- `get_test_data_dir()` → `Path` — Path to `tests/test_data/`
- `get_sample_gnn_model()` → `Path` — Path to the on-disk `sample_gnn_model.md` fixture
- `load_sample_gnn_spec()` → `dict` — Parses the sample model; falls back to a minimal spec dict when the file is missing
- `render_gnn_files(target_dir, output_dir)` → `dict` — Recovery-friendly bulk render used by `tests/pipeline/test_pipeline_recovery.py`
- `FakeMCPTime()` → `FakeMCPTime` — Injectable clock standing in for `gnn.mcp.mcp.time`; `advance(seconds)` moves the observable clock forward instantly (registry cache expiry and sliding-window rate limiter read `time.time()`), no production seam required
- `EXPECTED_MCP_TOOLS` → `int` — Exact pin of `tools_total` in the committed `src/gnn/mcp/audit_report.json` (162 at work time, 2026-09-23)
- `EXPECTED_MCP_MODULES` → `int` — Exact pin of `modules_total` in the committed `src/gnn/mcp/audit_report.json` (36 at work time)
- `CENSUS_SOURCE` → `str` — Path of the census file the pins derive from (`src/gnn/mcp/audit_report.json`); regenerate the audit and update both constants in the same PR that adds or removes tools/modules

## Bar-label offsets (`bar_labels.py`)

Imported as a submodule (`from tests.helpers.bar_labels import ...`), not
re-exported from `__init__.py`, because it imports `matplotlib.pyplot`.

- `figures_held_open(monkeypatch)` — context manager that no-ops `plt.close` so figures survive the code under test, then closes them all
- `open_bar_figures()` → `list[Figure]` — every open figure with a bar container (asserts at least one)
- `bar_label_gaps_px(fig)` → `list[float]` — pixel gap between each labelled bar's top and its label
- `assert_bar_labels_offset_in_points(fig)` — every gap equals `OFFSET_POINTS` (3 pt) at the figure DPI, within 0.75 px; a data-unit offset fails at degenerate or large data scales
- `assert_png_bounded(path)` — the PNG exists and neither side exceeds `MAX_PNG_SIDE` (20,000 px)

## Recovery Pattern

`render_gnn_files()` tolerates a patched `numpy.typing` raising `RecursionError` (bumps the recursion limit and records `recursion_limit_adjusted`), globs with string paths to dodge pathlib recursion edge cases, and writes scaffold artifacts plus a summary dict with recovery actions.
