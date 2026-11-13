# GUI Module - Agent Scaffolding

## Module Overview

**Purpose**: Interactive graphical user interfaces for visual GNN model construction and editing with multiple specialized implementations

**Pipeline Step**: Step 22: GUI (Interactive GNN Constructor) (22_gui.py)

**Category**: Interactive Visualization / Model Construction

**Status**: Production Ready

**Version**: [pyproject.toml](../../../pyproject.toml) (canonical)

**Last Updated**: 2026-09-24

---

## Core Functionality

### Primary Responsibilities
1. Provide multiple GUI implementations for GNN model construction
2. Enable visual editing of model components and state spaces
3. Generate interactive visualizations for model exploration
4. Support real-time model validation and feedback
5. Export constructed models to GNN format

### Key Capabilities
- Form-based interactive GNN constructor (GUI 1)
- Visual matrix editor with drag-and-drop (GUI 2)
- State space design studio (GUI 3)
- Real-time model validation and error checking
- Interactive matrix heatmaps and network visualizations
- Component management and state space editing

---

## API Reference

### Public Functions

#### `process_gui(target_dir: Path, output_dir: Path, verbose: bool = False, logger: Optional[logging.Logger] = None, **kwargs) -> bool`
**Description**: Main GUI processing function called by orchestrator (22_gui.py). Runs all available GUI implementations.

**Parameters**:
- `target_dir` (Path): Directory containing GNN files
- `output_dir` (Path): Output directory for GUI results
- `verbose` (bool): Enable verbose logging (default: False)
- `logger` (Optional[logging.Logger]): Logger instance for progress reporting (default: None)
- `gui_types` (str, optional): Comma-separated list of GUI types ("gui_1", "gui_2", "gui_3", "oxdraw") (default: "gui_1,gui_2")
- `interactive` (bool, optional): Run GUIs in interactive mode (default: False)
- `headless` (bool, optional): Run in headless mode - artifact generation only (default: True if not interactive)
- `open_browser` (bool, optional): Automatically open browser for interactive GUIs (default: False)
- `**kwargs`: Additional GUI-specific options

**Returns**: `bool` - True if GUI processing succeeded, False otherwise

**Example**:
```python
from gnn.gui import process_gui
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
# Headless mode (pipeline default)
success = process_gui(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/22_gui_output"),
    logger=logger,
    verbose=True,
    headless=True,
)

# Interactive mode
success = process_gui(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/22_gui_output"),
    logger=logger,
    interactive=True,
    gui_types="gui_1,oxdraw",
    open_browser=True,
)
```

#### `gui_1(target_dir: Path, output_dir: Path, logger: logging.Logger, **kwargs) -> Dict[str, Any]`
**Description**: Form-based Interactive GNN Constructor (GUI 1). Two-pane editor with component management.

**Parameters**:
- `target_dir` (Path): Input directory
- `output_dir` (Path): Output directory for GUI 1
- `logger` (logging.Logger): Logger instance
- `verbose` (bool, optional): Enable verbose logging (default: False)
- `headless` (bool, optional): Run in headless mode (wrapper default: False — a standalone `gui_1()` call launches interactive servers; the pipeline always passes `headless=True` because `process_gui` derives `headless = not interactive`)
- `export_filename` (str, optional): Output filename for constructed model (default: "constructed_model_gui1.md")
- `open_browser` (bool, optional): Open browser for interactive mode (wrapper default: True; the `22_gui.py` CLI `--open-browser` flag defaults to False)

**Returns**: `Dict[str, Any]` - GUI 1 execution results with:
- `gui_type` (str): `"gui_1"`
- `description` (str): Human-readable implementation summary
- `success` (bool): Whether GUI execution succeeded
- `output_file` (Optional[str]): Path to the generated model on success, else `None`
- `backend` (str): Detected Gradio backend, `"headless"` in headless runs, `"none"`/`"error"` when unavailable
- `backend_reason` (str): Backend detection explanation
- `features` (list[str]): Capability summary

#### `gui_2(target_dir: Path, output_dir: Path, logger: logging.Logger, **kwargs) -> Dict[str, Any]`
**Description**: Visual Matrix Editor with drag-and-drop interface (GUI 2). Matrix heatmaps and POMDP template support.

**Parameters**:
- `target_dir` (Path): Input directory
- `output_dir` (Path): Output directory for GUI 2
- `logger` (logging.Logger): Logger instance
- `verbose` (bool): Enable verbose logging
- `headless` (bool): Run in headless mode
- `export_filename` (str): Output filename for visual model
- `open_browser` (bool): Open browser for interactive mode

**Returns**: Dictionary with GUI 2 execution results

#### `get_available_guis() -> Dict[str, Dict]`
**Description**: Get information about all available GUI implementations

**Returns**: Dictionary with GUI information including names, descriptions, and ports

#### `generate_html_navigation(pipeline_output_dir, output_dir, logger) -> bool`
**Description**: Generate HTML navigation page that links to all pipeline output types

**Parameters**:
- `pipeline_output_dir` (Path): Directory containing all pipeline outputs (typically output/)
- `output_dir` (Path): GUI output directory where navigation.html will be created
- `logger` (Logger): Logger instance for progress reporting

**Returns**: `True` if navigation page generated successfully, `False` otherwise

**Features**:
- Scans all 25 pipeline output directories
- Discovers files by type (JSON, MD, PNG, SVG, HTML, etc.)
- Generates organized HTML navigation with file metadata
- Provides links to all pipeline artifacts
- Includes summary statistics and integration with comprehensive reports

### Composability Helpers (module-level, pure)

#### `normalize_gui_types(value: str | Sequence[str] | None) -> list[str]`
Parse/validate the `gui_types` option: comma-separated strings are split and
stripped, blank entries dropped, `None` yields the pipeline default
`["gui_1", "gui_2"]`. Exported from `gui`.

#### `summarize_gui_results(results: Mapping[str, Mapping[str, Any]]) -> GUISummary`
Aggregate per-GUI result mappings into a typed `GUISummary` TypedDict
(`total`, `succeeded`, `failed`, `failed_guis`, `overall_success`). A GUI
counts as failed when its `success` key is missing or falsy. Exported from `gui`.

#### `collect_pipeline_outputs(pipeline_output_dir: Path, max_files_per_section: int = MAX_FILES_PER_SECTION) -> tuple[list[dict], int]`
Discover pipeline artifacts grouped by step output directory (the step
table is the registry-derived `PIPELINE_OUTPUT_SECTIONS` constant, built
from `gnn.pipeline.step_registry.STEPS`). Returns `(sections,
total_files)`; `file_count` counts every discovered file while `files` is
capped at `MAX_FILES_PER_SECTION` (20). Pass `max_files_per_section=None` to
return every matching file; navigation uses this complete listing with native
**More files** disclosure and offline filename/path/step and type filters.
An active pipeline run admits only indexed artifacts for the Step 22 model
selection. Standalone discovery remains a filesystem census, labelled separately
from current-run evidence. Missing step directories are skipped.

#### Shared internals
- `gui/runner.py` — `resolve_output_root()` (delegates to the shared
  `pipeline.config.resolve_step_output_dir` helper; caller-supplied directory on standalone use), `load_first_markdown()` (prefer-pattern
  markdown discovery), `launch_gradio_in_thread()` (background Gradio launch);
  used by the gui_1/gui_2/gui_3 processors instead of duplicated logic.
- `gui/backend.py` — `write_text_atomically()` joins `write_json_atomically()`
  as the atomic artifact writers (temp file + `os.replace`).
- `gui/websocket_bridge.py` — local-only WebSocket message contracts for reactive
  GUI synchronization (`GUIWebSocketMessage`, `GUI_WEBSOCKET_MESSAGE_TYPES`);
  covered by `tests/gui/test_websocket_bridge.py`.
- `process_gui(**kwargs)` now honors a caller-provided `logger=` kwarg
  (documented in the API table above); without it the module logger is used.
- `navigation.html` escapes file names/paths with `html.escape`, URL-encodes
  links relative to its actual destination, and renders dashboard/report
  crosslinks only for available artifacts. Artifact names containing `#`, `?`
  or `%` remain navigable. All files remain accessible without JavaScript;
  filters provide a live match count, empty state and keyboard focus styles.
- `gui_3/processor.py` uses the same `detect_gradio_backend()` recovery
  semantics as gui_1/gui_2 (a broken gradio install degrades to headless
  artifacts instead of raising).

---

## GUI Implementations

### GUI 1: Form-based Constructor
**Port**: `:7860`
**Focus**: Step-by-step model building with component management
**Features**:
- Interactive two-pane editor (components + markdown)
- Component management (observation/hidden/action/policy variables)
- State space entry management with live validation
- Synchronized plaintext GNN markdown editor
- Real-time model validation and error feedback

### GUI 2: Visual Matrix Editor
**Port**: `:7861`
**Focus**: Real-time matrix heatmaps and interactive editing
**Features**:
- Interactive DataFrame editing with +/- dimension controls
- Vector bar chart displays for C & D vectors
- Live matrix statistics (min, max, mean, sum)
- Auto-update functionality and matrix validation
- Drag-and-drop matrix manipulation

### GUI 3: State Space Design Studio
**Port**: `:7862`
**Focus**: Visual state space architecture design
**Features**:
- Visual state space architecture designer with SVG diagrams
- Ontology term editor for Active Inference concept mapping
- Interactive connection graph interface (D>s, s-A, A-o format)
- Parameter tuning controls (states, observations, actions, horizons)
- Real-time GNN export and preview with low-dependency approach

---

### Required Dependencies
Installed by the `gui` extra in `pyproject.toml` (`uv sync --extra gui`):
- `gradio` - Web-based GUI framework

Also used when present (verified imports: `gui_2/ui.py` imports `numpy` and `plotly`;
nothing in `src/gnn/gui` imports pandas, matplotlib, or networkx):
- `numpy` - Numerical operations for GUI 2 visualizations

### Optional Dependencies
- `plotly` - Interactive visualizations in GUI 2 (graceful degradation when absent)

---

## Configuration

### Environment Variables

None dedicated to this module. GUI behavior is configured through
`process_gui()` kwargs (e.g. `gui_types`, `interactive`, `headless`,
`open_browser`) and `input/config.yaml` pipeline settings. Per-GUI ports are
assigned in `gui/__init__.py` (GUI 1: 7860, GUI 2: 7861, GUI 3: 7862).

### Default Settings

Processing defaults (GUI types, headless mode) are set in `process_gui()` in
`gui/processor.py`; per-GUI options (`export_filename`, ports) are defined in
each GUI's `__init__.py`.

---

## Usage Examples

### Basic Usage (All GUIs)
```python
from gnn.gui import process_gui

success = process_gui(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/22_gui_output"),
    logger=logger,
    gui_types="gui_1,gui_2,gui_3,oxdraw",
    interactive=True,
)
```

### Specific GUI (GUI 1)
```python
from gnn.gui.gui_1 import gui_1

result = gui_1(
    target_dir=Path("input/gnn_files"),
    output_dir=Path("output/gui_1_output"),
    logger=logger,
    headless=False,
    export_filename="constructed_model_gui1.md",
    open_browser=True,
)
```

### GUI Information Query
```python
from gnn.gui import get_available_guis

guis = get_available_guis()
for gui_name, info in guis.items():
    print(f"{gui_name}: {info['description']} (Port: {info['port']})")
```

---

## Output Specification

### Output Products

#### GUI-Specific Outputs
- `constructed_model_gui1.md` - Generated GNN model from GUI 1 (Form-based Constructor)
- `visual_model_gui2.md` - Generated GNN model from GUI 2 (Visual Matrix Editor)
- `visual_matrices.json` - Matrix data and visualizations from GUI 2
- `designed_model_gui_3.md` / `design_analysis.json` / `design_studio_status.json` - GUI 3 starter model, design analysis, and status
- `<stem>.mmd`, `<stem>_from_mermaid.md`, `oxdraw_processing_results.json` - oxdraw Mermaid exports, round-trip models, and run results (written under `oxdraw_output/`)
- `gui_1_status.json` / `gui_2_status.json` - GUI execution status and backend information
- `gui_processing_summary.json` - Overall GUI processing summary with results from all GUIs

The oxdraw integration verification receipt lives at `docs/gui_oxdraw/VERIFICATION.md`.

#### Navigation and Discovery
- `navigation.html` - **Comprehensive HTML navigation page** that provides:
  - Links to all pipeline output files across all 25 steps
  - Organized by pipeline step with file counts and metadata
  - File type filtering and search capabilities
  - Direct links to visualizations, reports, execution results, and all artifacts
  - Summary statistics of pipeline outputs
  - Links to comprehensive reports

### Output Directory Structure
```
output/22_gui_output/
├── constructed_model_gui1.md          # GUI 1 output: Form-based constructed model
├── visual_model_gui2.md               # GUI 2 output: Visual matrix editor model
├── visual_matrices.json                # GUI 2 output: Matrix data and visualizations
├── gui_1_status.json                   # GUI 1 status and backend info
├── gui_2_status.json                   # GUI 2 status and backend info
├── designed_model_gui_3.md            # GUI 3 output: starter/designed model
├── design_analysis.json                # GUI 3 design analysis
├── design_studio_status.json           # GUI 3 status and backend info
├── oxdraw_output/                      # oxdraw artifacts (.mmd exports, *_from_mermaid.md round-trips, oxdraw_processing_results.json)
├── gui_processing_summary.json         # Overall processing summary
└── navigation.html                     # HTML navigation to all pipeline outputs
```

### Navigation.html Features

The `navigation.html` file provides comprehensive navigation to all pipeline outputs:

1. **Pipeline Overview**: Summary statistics showing total pipeline steps and files
2. **Output Sections**: Organized by all 25 pipeline steps:
   - Template (0_template_output)
   - Setup (1_setup_output)
   - GNN Processing (3_gnn_output)
   - Model Registry (4_model_registry_output)
   - Type Checker (5_type_checker_output)
   - Validation (6_validation_output)
   - Export (7_export_output)
   - Visualization (8_visualization_output)
   - Advanced Visualization (9_advanced_viz_output)
   - Ontology (10_ontology_output)
   - Render (11_render_output)
   - Execute (12_execute_output)
   - LLM (13_llm_output)
   - ML Integration (14_ml_integration_output)
   - Audio (15_audio_output)
   - Analysis (16_analysis_output)
   - Integration (17_integration_output)
   - Security (18_security_output)
   - Research (19_research_output)
   - Website (20_website_output)
   - MCP (21_mcp_output)
   - GUI (22_gui_output)
   - Report (23_report_output)
   - Intelligent Analysis (24_intelligent_analysis_output)

3. **File Information**: For each file, displays:
   - File name with clickable link
   - File type/extension
   - File size in MB
   - Relative path for navigation

4. **Integration**: Links to comprehensive report at `23_report_output/comprehensive_analysis_report.html`

---

## Performance Characteristics

### Latest Execution
Headless artifact generation completes in seconds; interactive GUI startup and memory
footprint depend on the GUI backend and host. Measure on your own hardware; this
document does not track timings.

---

## Error Handling

### Graceful Degradation
- **No gradio**: Recovery to HTML-based interfaces
- **No plotly**: Use matplotlib for visualizations
- **Browser unavailable**: Generate static artifacts only

### Error Categories
1. **Dependency Errors**: Missing GUI framework dependencies
2. **Port Conflicts**: GUI server port already in use
3. **Browser Errors**: Unable to launch interactive interface
4. **File System Errors**: Unable to write output files

---

## Integration Points

### Orchestrated By
- **Script**: `22_gui.py` (Step 22)
- **Function**: `process_gui()`

### Imports From
- `gnn.utils.pipeline_orchestration.pipeline_template` - Standardized processing patterns
- `pipeline.config` - Configuration management

### Imported By
- `src/gnn/22_gui.py` - Step 22 orchestrator (`from gnn.gui import process_gui`)
- `src/gnn/cli/__init__.py` - `gnn gui` CLI command (`_cmd_gui`)
- `tests/gui/*` - GUI test suite

### Data Flow
```
GNN Files → GUI Construction → Visual Editing → Model Validation → GNN Export → Pipeline Integration
```

---

## Testing

### Test Files
The `tests/gui/` suite currently holds 14 test modules:

- `test_gui3_loader_and_wrapper.py`, `test_gui3_pipeline_kwargs.py` - GUI 3 loader/wrapper and pipeline kwargs
- `test_gui_callback_bindings.py` - interactive callback binding regressions
- `test_gui_composability.py` - composability helpers
- `test_gui_functionality.py` - GUI functionality tests
- `test_gui_info_ports_and_defaults.py` - info dicts, ports, and defaults
- `test_gui_launch_verification.py` - interactive launch verification
- `test_gui_mcp_interactive.py` - MCP surface in interactive mode
- `test_gui_model_logic.py` - model logic
- `test_gui_overall.py` - overall module behavior
- `test_gui_status_namespacing.py` - per-GUI status file namespacing
- `test_oxdraw_integration.py` - oxdraw integration tests
- `test_step22_headless_default.py` - step-22 headless default
- `test_websocket_bridge.py` - websocket bridge contracts

### Test Coverage
Measure on demand:

```bash
uv run --extra dev python -m pytest tests/gui/ --cov=gnn.gui --cov-report=term-missing
```
### Key Test Scenarios
1. GUI startup and shutdown in headless mode
2. Component creation and editing workflows
3. Matrix manipulation and validation
4. Model export and format validation
5. Error handling with missing dependencies

---

## MCP Integration

### Tools Registered
- `process_gui` - Generate or launch the selected GUI surfaces
- `list_available_guis` - List GUI implementations and capabilities
- `get_gui_module_info` - Return GUI module metadata and inventory
- `oxdraw.convert_to_mermaid` - Convert a GNN model to Mermaid
- `oxdraw.convert_from_mermaid` - Convert Mermaid to GNN
- `oxdraw.launch_editor` - Launch the oxdraw editor
- `oxdraw.check_installation` - Check the oxdraw runtime
- `oxdraw.get_info` - Return oxdraw metadata

Parent GUI registration wraps the five oxdraw handlers so both the parent discovery
path and a direct `gui.oxdraw.mcp.register_tools()` call accept normal MCP keyword
arguments. Registrations live in `src/gnn/gui/mcp.py`.

---

## Recent Improvements

### Step-22 headless default and the 600s step-timeout wall
- Pipeline step 22 runs under the per-step timeout registry `pipeline/step_timeouts.py`
  (`"22_gui.py": 600`, `step_timeouts.py`; overridable via `GNN_STEP_TIMEOUT_22` and
  `GNN_STEP_TIMEOUT_SCALE`, documented in `src/gnn/pipeline/README.md`).
- Interactive GUI servers never exit on their own (`process_gui` keeps the process alive
  while servers run), so an in-pipeline `--interactive` run burns the full 600s budget
  before the step is killed.
- The headless default (PR #142) derives `headless = not interactive`, so pipeline runs
  produce artifacts and return immediately; regression: `tests/gui/test_step22_headless_default.py`.
- Use `--interactive` only for standalone sessions outside a pipeline run.
---

## Recent Enhancements (January 5, 2026)

### HTML Navigation Generation
- **Added**: `generate_html_navigation()` function to create comprehensive navigation page
- **Features**:
  - Scans all 25 pipeline output directories automatically
  - Discovers and catalogs all output files by type
  - Generates organized HTML with file metadata (type, size, path)
  - Provides direct links to all pipeline artifacts
  - Includes summary statistics and integration with comprehensive reports
- **Output**: `navigation.html` file in GUI output directory
- **Integration**: Automatically generated during `process_gui()` execution
- **Access**: Open `output/22_gui_output/navigation.html` in web browser for full pipeline navigation

### Output File Discovery
- **Enhanced**: All output files are now properly discovered and documented
- **Files Generated**:
  - `constructed_model_gui1.md` - GUI 1 form-based model construction
  - `visual_model_gui2.md` - GUI 2 visual matrix editor model
  - `visual_matrices.json` - GUI 2 matrix data and visualizations
  - `gui_1_status.json` / `gui_2_status.json` - GUI execution status and backend information
  - `gui_processing_summary.json` - Overall processing summary
  - `navigation.html` - Comprehensive navigation to all pipeline outputs

---

## Troubleshooting

### Common Issues

#### Issue 1: GUI fails to launch
**Symptom**: GUI server doesn't start or browser doesn't open  
**Cause**: Port already in use or dependencies missing  
**Solution**: 
- Check if port is already in use: `lsof -i :7860`
- Use different port: `--port 7861`
- Verify Gradio dependencies installed
- Use `--headless` mode if GUI not needed

#### Issue 2: Model export fails
**Symptom**: GUI completes but model file not generated  
**Cause**: Export function errors or file permissions  
**Solution**:
- Check output directory permissions
- Verify export filename is valid
- Review export function logs
- Ensure GNN format validation passes

---

## Version History

### Current Version: [pyproject.toml](../../../pyproject.toml) (canonical)

**Features**:
- Multiple GUI implementations (GUI 1, GUI 2, GUI 3, oxdraw)
- Interactive model construction
- Real-time validation
- Model export to GNN format

**Known Issues**:
- None currently

### Roadmap
- **Next Version**: Enhanced visual editing
- **Future**: Collaborative editing

---

## References

### Related Documentation
- [Pipeline Overview](../../../README.md)
- [Architecture Guide](../../../ARCHITECTURE.md)
- [GUI Guide](../../../docs/gui_oxdraw/)

### External Resources
- [Gradio Documentation](https://gradio.app/)

---

**Last Updated**: 2026-09-22
**Status**: Production Ready
**Package version**: [pyproject.toml](../../../pyproject.toml) (canonical)
**Architecture Compliance**: Thin Orchestrator Pattern


---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
- **[SKILL](SKILL.md)**: Capability API
