# GUI 1: Interactive GNN Constructor - Agent Scaffolding

## Module Overview

**Purpose**: Provides the primary form-based visual interface for constructing and editing Active Inference generative models. It features a Gradio-based application allowing real-time editing of GNN components, state spaces, and immediate markdown synchronization.

**Pipeline Step**: Step 22: GUI Processing (22_gui.py)

**Category**: Interactive Visualization / Model Construction

**Status**: Production Ready

**Version**: [pyproject.toml](../../../../pyproject.toml) (canonical)

---

## Core Functionality

### Primary Responsibilities

1. **Interactive Model Editing**: Renders generalized model components (matrices, states, observations, connections) into editable form fields.
2. **State Space Validation**: Validates dimension entries as comma-separated integers (`parse_dims_csv` in `ui.py`); invalid entries surface an inline error message in the validation output pane.
3. **Live Markdown Synchronization**: Two-pane interface synchronizing web changes directly to the target `.md` file representation.

### Key Capabilities

- Two-pane Gradio web interface (Controls + Markdown).
- Form-based component management.
- Live validation mapping to Active Inference terminology.
- Integration as a headless backend or interactive server.

### Agent Capabilities

This module provides specialized agent capabilities for visual construction:

#### Form Interface Agent
- **Core Function**: Maps structured dict strings into visual schemas.
- **Input Processing**: Processes `target_dir` GNN files.
- **Output Generation**: Saves supplied Markdown as a headless artifact, or provides
  interactive editing and validation when Gradio is available. Headless export does
  not execute interactive callbacks or certify scientific validity.

---

## API Reference

### Public Functions

#### `gui_1(target_dir: Path, output_dir: Path, logger: logging.Logger, **kwargs) -> Dict[str, Any]`

**Description**: Main orchestrator to run GUI 1. Handles parameter extraction for headless vs. browser mode.

**Parameters**:
- `target_dir` (Path): Directory containing GNN files to load.
- `output_dir` (Path): Output directory for the saved models.
- `logger` (logging.Logger): Standard pipeline logger.
- `**kwargs`: Supported kwargs include `headless` (bool), `export_filename` (str), and `open_browser` (bool).

**Returns**: A dictionary containing `gui_type`, `success`, `output_file`, and execution details.

#### `get_gui_1_info() -> Dict[str, Any]`

**Description**: Exposes metadata capabilities (e.g. features, requirements, layout type) to the aggregator.

---

## Dependencies

### Required Dependencies
- `gradio` - Core web framework for the interactive application.
- `pathlib` - Filesystem resolution.

### Internal Dependencies
- `gnn.gui.gui_1.markdown` - Text synchronization helpers.
- `gnn.gui.gui_1.processor` - Controller logic coordinating Gradio and standard logic.

---

## Configuration

### Default Settings

```
# Headless defaults
headless = False
export_filename = "constructed_model_gui1.md"
open_browser = True
```

---

## Error Handling

### Recovery Strategies
- **Parse Failures**: If a parsing or validation operation fails mid-edit, the editor content is kept as-is and the error is reported in the `validation_output` pane beneath the editor; no rollback or buffered state mechanism exists.
- **Dependency Missing**: A missing `gradio` is detected at import time (`detect_gradio_backend` in `../backend.py`); GUI 1 then degrades to headless artifact generation instead of raising, and logs `uv sync --extra gui` as the remediation.

---

## Development Guidelines

### Adding New Component Forms
1. Update `markdown.py` to support inserting the component text safely.
2. Register the Gradio inputs in `ui.py`.
3. Add to the `__all__` exported registry in `__init__.py`.

State insertion belongs inside an existing canonical `StateSpaceBlock`, before
the next section. Keep manual text and line endings outside the insertion intact;
retain physical duplicates, existing editor heading aliases, the linear delimiter
parser and the 8,388,608-character input limit. Native readers still apply their
own section grammar.

**Last Updated**: 2026-10-08
**Architecture Compliance**: 100% Thin Orchestrator Pattern
