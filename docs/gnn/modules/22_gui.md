# Step 22: Gui

## Architectural Mapping

**Orchestrator**: `src/gnn/22_gui.py` (103 lines)
**Implementation Layer**: `src/gnn/gui/`

## Module Description

Step 22 is a thin orchestrator: `src/gnn/22_gui.py` (98 lines) builds a standardized
pipeline script with `create_standardized_pipeline_script` and delegates all work to
`process_gui()` in `src/gnn/gui/`. The implementation layer provides four GUI
implementations that each generate artifacts headlessly (the pipeline default) or
launch an interactive server when explicitly requested:

- **gui_1 — Form-based Interactive GNN Constructor** (port 7860): two-pane editor with
  component management and state-space editing on the left and a synchronized live GNN
  markdown editor on the right.
- **gui_2 — Visual Matrix Editor** (port 7861): real-time matrix/vector heatmap plots,
  editable value tables, and +/- dimension controls with live GNN regeneration.
- **gui_3 — State Space Design Studio** (port 7862): low-dependency visual state-space
  designer with ontology term editing, connection graphs, and parameter tuning.
- **oxdraw — Visual diagram-as-code interface** (port 5151): bidirectional
  GNN ↔ Mermaid conversion with an optional external `oxdraw` editor launch.

The same stack is reachable standalone via the `gnn gui` CLI subcommand and via the
MCP tools listed under MCP Integration below.

## Core Functionality

### `process_gui()` behavior

Defined in `src/gnn/gui/processor.py` (`process_gui`, lines 173-322):

- **Headless is derived, not defaulted**: `headless = not interactive`
  (processor.py:210-215). Every non-interactive run is headless (artifact generation
  only); a literal `headless=False` default would silently force interactive runs.
- **Caller-provided logger honored**: a `logging.Logger` passed via `logger` kwarg is
  used directly; otherwise the `gui.processor` module logger is used
  (processor.py:202-208).
- **Selected GUIs run in order**: `gui_types` is normalized by `normalize_gui_types()`
  (default `gui_1,gui_2`; accepts `gui_1`, `gui_2`, `gui_3`, `oxdraw`), each selected
  GUI runs, and per-GUI result dicts are aggregated; any failure marks the overall run
  unsuccessful.
- **Artifacts**: `_save_processing_summary()` writes `gui_processing_summary.json`;
  `_generate_navigation_page()` writes `navigation.html`.
- **Interactive keep-alive**: in interactive mode the function blocks in a loop while
  `interactive_servers_running()` (from `gui/runner.py`) reports live Gradio server
  threads, until they stop or Ctrl+C (processor.py:303-316). Server threads are daemon
  threads launched by `launch_gradio_in_thread()`.

### GUI implementations and ports

| Type | Port | Interface | Output root |
|------|------|-----------|-------------|
| `gui_1` | 7860 | Form-based constructor (Components + State Space tabs, live markdown editor) | resolved step output root |
| `gui_2` | 7861 | Visual matrix editor (heatmap plots, editable value tables, +/- dimension controls) | resolved step output root |
| `gui_3` | 7862 | State space design studio (SVG diagrams, ontology terms, connection graphs) | resolved step output root |
| `oxdraw` | 5151 | GNN ↔ Mermaid conversion with optional external editor | `oxdraw_output/` subdirectory |

Each GUI's port is defined in its own `processor.py` (`gui_1/processor.py`,
`gui_2/processor.py`, `gui_3/processor.py`, `oxdraw/processor.py`) and advertised
through the `get_gui_1_info()`/`get_gui_2_info()`/`get_gui_3_info()`/`get_oxdraw_info()`
metadata re-exported from `gnn.gui`. `gui_1`/`gui_2`/`gui_3`
normalize their output directory through `resolve_output_root()`
(`gui/runner.py`), which delegates to the shared
`gnn.pipeline.config.resolve_step_output_dir` helper (pipeline/config.py:205-237).

### Backend detection and graceful degradation

`detect_gradio_backend()` (`gui/backend.py:31-40`) probes Gradio availability once at
import. Without Gradio (no `gui` extra installed) every GUI produces static headless
artifacts and returns success instead of failing the pipeline; with Gradio installed
and `--interactive` set, servers launch on daemon threads and
`wait_for_server_launch()` (backend.py:62-98) verifies each launch with a bounded
HTTP probe.

### Timeout interaction

Step 22 runs under a **600-second declared budget** in
`gnn.pipeline.step_timeouts.STEP_TIMEOUTS` (`"22_gui.py": 600`; measured 2.9 seconds
for the headless default in the 2026-11-02 full-corpus run). Every step registered
in `step_registry.STEPS` must declare a budget in that table, and an undeclared
step or dictionary variant fails closed with `StepTimeoutNotDeclared` instead of
inheriting a silent default — a registered step with no entry fails at import of
`step_timeouts`, and `get_step_timeout()` fails at the call for an unknown script
name or a missing `default`/`comprehensive` variant. That contract is carried by
the step-timeout hardening PR #307; until it merges, an entry-less step still
receives the old implicit 180-second default.

The budget is enforced in both execution tiers — the subprocess step runner
(`gnn.pipeline.summary_wiring.execute_pipeline_step`, re-exported by `gnn.main`)
and consolidated in-process execution (`gnn.pipeline.step_executor`) — through
`get_step_timeout()`. `GNN_STEP_TIMEOUT_22` (absolute) and
`GNN_STEP_TIMEOUT_SCALE` (multiplier) are the documented overrides; see
[pipeline/README.md](../../../src/gnn/pipeline/README.md) for the knob
documentation.

Step 13 (LLM) resolves its budget from the run when one is present:
`llm.timeout_seconds` when configured, else 600 seconds per selected model
(22,800 seconds for a 38-model corpus), intersected with the invocation deadline
(`RunContext.bounded_timeout()`). The declared `13_llm.py` value (900 seconds)
applies only to a run with no run context. See [13_llm.md](13_llm.md).

Interactive GUI servers never exit on their own, and the
keep-alive loop (processor.py:303-316) holds the process while any server is live, so
an `--interactive` run inside a pipeline consumes the whole step budget and is killed
at the wall. The headless default (PR #142: `headless = not interactive`,
processor.py:210-215; regression test
`tests/gui/test_step22_headless_default.py`; CHANGELOG entry "Step-22 headless
default + GUI wrapper batch (#142)") keeps pipeline runs artifact-only;
`--interactive` is a standalone opt-in that blocks until servers stop by design.

### Performance

Headless artifact generation completes in seconds; interactive GUI startup and memory
footprint depend on the GUI backend and host. Measure on your own hardware; this
document does not track timings.

## API Reference

All public surface is importable from the package root (`gnn.*` is the only canonical
import surface):

```python
from gnn.gui import process_gui, get_available_guis, normalize_gui_types
```

### `process_gui(target_dir: Path, output_dir: Path, verbose: bool = False, **kwargs) -> bool`

Main processing function called by the orchestrator (22_gui.py), the `gnn gui` CLI
subcommand, and MCP.

**kwargs**:

- `logger` (logging.Logger, optional): caller-provided logger, honored when passed
- `gui_types` (str, optional): comma-separated list — `gui_1`, `gui_2`, `gui_3`,
  `oxdraw` (default `"gui_1,gui_2"`)
- `interactive` (bool, optional): launch interactive GUI servers (default False)
- `headless` (bool, optional): derived as `not interactive` — explicit
  `headless=True` stays headless; do not pass `headless=False` expecting servers
- `open_browser` (bool, optional): open the browser for interactive GUIs
  (default False)
- `launch_editor` (bool, optional): launch the external oxdraw editor when the
  `oxdraw` GUI type runs interactively (default False)

**Returns**: `bool` — True when every selected GUI succeeded.

```python
from pathlib import Path
import logging
from gnn.gui import process_gui

logger = logging.getLogger(__name__)

# Headless (pipeline default): artifacts only
process_gui(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/22_gui_output"),
    logger=logger,
    verbose=True,
)

# Interactive opt-in: launches servers and blocks until they stop
process_gui(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/22_gui_output"),
    gui_types="gui_1,oxdraw",
    interactive=True,
    open_browser=True,
)
```

### Module-level API (`src/gnn/gui/__init__.py`)

`__all__` exports: `process_gui`, `gui_1`, `gui_2`, `gui_3`, `oxdraw_gui`,
`get_available_guis`, `get_gui_1_info`, `get_gui_2_info`, `get_gui_3_info`,
`get_oxdraw_info`, `generate_html_navigation`, `normalize_gui_types`,
`summarize_gui_results`, `collect_pipeline_outputs`, `DEFAULT_GUI_TYPES`,
`MAX_FILES_PER_SECTION`, `PIPELINE_OUTPUT_SECTIONS`, plus the GUI 1 markdown
utilities (`add_component_to_markdown`, `update_component_states`,
`remove_component_from_markdown`, `parse_components_from_markdown`,
`parse_state_space_from_markdown`, `add_state_space_entry`,
`update_state_space_entry`, `remove_state_space_entry`).
`get_module_info()` is also defined at package level and returns module metadata
(name, version, description, feature flags) for MCP discovery.

- `get_available_guis() -> dict[str, dict]`: name → info mapping (description, port,
  category, features) for all four GUI types.
- `normalize_gui_types(value: str | Sequence[str] | None) -> list[str]`: parses a
  comma-separated string, a sequence, or `None` (the `gui_1,gui_2` default); blank
  entries are dropped.
- `summarize_gui_results(results) -> GUISummary`: aggregates per-GUI result dicts into
  `{total, succeeded, failed, failed_guis, overall_success}`.
- `generate_html_navigation(pipeline_output_dir, output_dir, logger) -> bool`: renders
  `navigation.html` from discovered pipeline outputs (called automatically by
  `process_gui()`).
- `collect_pipeline_outputs(...)`: discovery helper backing the navigation page.

### Per-GUI entry points

Each wrapper takes `(target_dir: Path, output_dir: Path, logger: logging.Logger,
**kwargs)` and returns a result dict with `success`, per-GUI metadata, and — only on
a verified interactive launch — `port` and `url`.

| Function | Extra kwargs | Defaults |
|----------|--------------|----------|
| `gui_1` | `headless`, `export_filename="constructed_model_gui1.md"`, `open_browser=True`, `verbose` | standalone default interactive (`headless=False`); pipeline runs are headless via the `process_gui` derivation |
| `gui_2` | `headless`, `export_filename="visual_model_gui2.md"`, `open_browser=True`, `verbose` | prefers `*pomdp*.md` inputs, falls back to a POMDP starter template |
| `gui_3` | `headless`, `export_filename="designed_model_gui_3.md"`, `open_browser=False` | tolerates and ignores extra pipeline kwargs |
| `oxdraw_gui` | `headless` (default True), `mode` ("interactive"/"headless"), `launch_editor`, `port=5151`, `host="127.0.0.1"`, `auto_convert`, `validate_on_save`, `verbose` | writes into `oxdraw_output/` subdirectory |

## Output Specification

Artifacts are written under `output/22_gui_output/` (per-GUI runners resolve their
output root through `resolve_output_root()`; oxdraw nests one level deeper):

```
output/22_gui_output/
├── constructed_model_gui1.md     # gui_1 export (starter template with headless marker)
├── gui_1_status.json             # gui_1 status (namespaced)
├── visual_model_gui2.md          # gui_2 export (POMDP starter or loaded model)
├── visual_matrices.json          # gui_2 matrix data snapshot
├── gui_2_status.json             # gui_2 status (namespaced)
├── designed_model_gui_3.md       # gui_3 export (design-studio starter model)
├── design_analysis.json          # gui_3 design analysis
├── design_studio_status.json     # gui_3 status
├── gui_processing_summary.json   # aggregate: kwargs, gui_types, per-GUI results, overall_success
├── navigation.html               # HTML navigation over all pipeline outputs
└── oxdraw_output/
    ├── <model>.mmd               # GNN → Mermaid conversion (one per input file)
    ├── <model>_from_mermaid.md   # Mermaid → GNN round-trip (validate_on_save)
    └── oxdraw_processing_results.json
```

Every artifact name above is constructed in code: `constructed_model_gui1.md`
(gui_1/processor.py:47), `gui_1_status.json` (gui_1/processor.py:116),
`visual_model_gui2.md` (gui_2/processor.py:44), `visual_matrices.json`
(gui_2/processor.py:99), `gui_2_status.json` (gui_2/processor.py:104),
`designed_model_gui_3.md` (gui_3/__init__.py:40), `design_analysis.json`
(gui_3/processor.py:81), `design_studio_status.json` (gui_3/processor.py:106),
`gui_processing_summary.json` (processor.py:335), `navigation.html`
(processor.py:392), `oxdraw_output/` (oxdraw/__init__.py:111), `<stem>.mmd`
(oxdraw/processor.py:107), `<stem>_from_mermaid.md` (oxdraw/processor.py:193),
`oxdraw_processing_results.json` (oxdraw/processor.py:265). Status JSONs share one
common-key schema (`backend`, `launched`, `export_file`, `gui_type`, `status`,
`reason`, `backend_reason`); `port`/`url` appear only on verified interactive
launches.

### navigation.html

Generated by `_generate_navigation_page()` from `collect_pipeline_outputs()`
discovery over `PIPELINE_OUTPUT_SECTIONS` (registry-derived from
`gnn.pipeline.step_registry.STEPS`, processor.py:47-135).
Each section reports the full discovered `file_count` while listing at most
`MAX_FILES_PER_SECTION = 20` files (processor.py:43-45); file names and paths are
HTML-escaped at render time.

## Dependencies

- `gradio>=6.7.0` — the only package in the `gui` extra (pyproject.toml:124-127).
  Without it, all GUIs degrade to static headless artifacts.
- `numpy` and `plotly` are imported by `gui_2/ui.py` for interactive heatmap/vector
  plots; both ship in the core pipeline dependencies (pyproject.toml:35, 85).
- `pandas`, `matplotlib`, and `networkx` are core pipeline dependencies but are not
  imported anywhere in `src/gnn/gui`.
- External `oxdraw` CLI (`cargo install oxdraw`) is optional and only needed for
  interactive editor launches.

## Configuration

No environment variables or configuration files are dedicated to this module. GUI
behavior is configured through `process_gui()` kwargs (`gui_types`, `interactive`,
`headless`, `open_browser`, `launch_editor`) and `input/config.yaml` pipeline
settings. Per-GUI ports are assigned in code (GUI 1: 7860, GUI 2: 7861, GUI 3: 7862,
oxdraw: 5151); processing defaults (`gui_types`, derived headless mode) live in
`process_gui()`.

## Integration Points

### Orchestrated By

- **Script**: `src/gnn/22_gui.py` (Step 22, 98 lines) via
  `create_standardized_pipeline_script`
- **CLI**: `gnn gui` subcommand (src/gnn/cli/__init__.py:467-487, handler
  `_cmd_gui` at :1317-1334) and the `just gui` recipe (justfile:139-141)
- **MCP**: `process_gui` tool registered by `src/gnn/gui/mcp.py`

### Imported By

- `src/gnn/22_gui.py:53` — pipeline step orchestrator (`from gnn.gui import process_gui`)
- `src/gnn/cli/__init__.py:1317` — `gnn gui` CLI handler (lazy import)
- `tests/gui/*` — module test suite

`main.py` does not import `gui`; its only step-22 reference is the prose step listing
at src/gnn/main.py:32. The step-20 website cross-links the GUI navigation hub.

### Imports From

- `gnn.utils.pipeline_orchestration.pipeline_template.create_standardized_pipeline_script`
  — standardized pipeline script construction (22_gui.py:54-56)
- `gnn.pipeline.config.resolve_step_output_dir` — shared output-root resolution
  (gui/runner.py:89-91)

### Data Flow

```
input/gnn_files → per-GUI headless artifacts + status JSONs → gui_processing_summary.json + navigation.html → step-20 website (cross-links the GUI navigation hub)
```

Step 22 depends on Step 3 (parsed GNN files) per the pipeline dependency graph.

## Testing

### Test Files

`tests/gui/` contains 14 test modules:

- `test_gui3_loader_and_wrapper.py`
- `test_gui3_pipeline_kwargs.py`
- `test_gui_callback_bindings.py`
- `test_gui_composability.py`
- `test_gui_functionality.py`
- `test_gui_info_ports_and_defaults.py`
- `test_gui_launch_verification.py`
- `test_gui_mcp_interactive.py`
- `test_gui_model_logic.py`
- `test_gui_overall.py`
- `test_gui_status_namespacing.py`
- `test_oxdraw_integration.py`
- `test_step22_headless_default.py`
- `test_websocket_bridge.py`

### Test Coverage

Run:

```bash
uv run --extra dev python -m pytest tests/gui/ -q
```

Measure coverage (do not treat any percentage in this document as canonical):

```bash
uv run --extra dev python -m pytest tests/gui/ --cov=gnn.gui --cov-report=term-missing
```

### Key Test Scenarios

1. Headless-default regression: `process_gui` with neither flag stays artifact-only
   (`test_step22_headless_default.py`)
2. Namespaced status artifacts and common-key schema
   (`test_gui_status_namespacing.py`)
3. Launch verification and keep-alive gating (`test_gui_launch_verification.py`)
4. oxdraw round-trip conversion and integration (`test_oxdraw_integration.py`)
5. WebSocket bridge message contracts (`test_websocket_bridge.py`)
6. MCP interactive derivation (`test_gui_mcp_interactive.py`)

## MCP Integration

### Tools Registered

`register_tools()` in `src/gnn/gui/mcp.py` registers **8 domain tools**:

- `process_gui` — run Step 22 (schema default headless: True; the handler passes
  `interactive=not headless`, so an MCP caller passing `headless=False` gets
  interactive servers)
- `list_available_guis` — list GUI implementations and ports
- `get_gui_module_info` — module metadata
- `oxdraw.convert_to_mermaid`, `oxdraw.convert_from_mermaid`,
  `oxdraw.launch_editor`, `oxdraw.check_installation`, `oxdraw.get_info` — five
  oxdraw tools registered via `gnn/gui/oxdraw/mcp.py` `register_mcp_tools()`

## References

### Related Documentation

- [Pipeline Overview](../../../README.md)
- [Architecture Guide](../../../ARCHITECTURE.md)
- [Pipeline step timeouts](../../../src/gnn/pipeline/README.md) — the
  `GNN_STEP_TIMEOUT_{N}` / `GNN_STEP_TIMEOUT_SCALE` knobs referenced above
- [oxdraw verification receipt](../../../docs/gui_oxdraw/VERIFICATION.md) — the
  maintained verification report for the oxdraw integration

### External Resources

- [Gradio Documentation](https://gradio.app/)

---

**Last Updated**: 2026-09-22
**Status**: Production Ready
**Package version**: [pyproject.toml](../../../pyproject.toml) (canonical)

---
## Documentation
- **[README](../../../src/gnn/gui/README.md)**: Module Overview
- **[AGENTS](../../../src/gnn/gui/AGENTS.md)**: Agentic Workflows
- **[SPEC](../../../src/gnn/gui/SPEC.md)**: Architectural Specification
- **[SKILL](../../../src/gnn/gui/SKILL.md)**: Capability API


---

**Source Reference**: [src/gnn/gui](../../../src/gnn/gui)