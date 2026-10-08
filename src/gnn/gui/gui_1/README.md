# GUI 1: Form-based Interactive GNN Constructor

## Overview
This module provides the primary form-based GUI functionality for the Generalized Notation Notation (GNN) pipeline. Operating as a component of **Step 22**, it provides a real-time, interactive environment to scaffold, edit, and orchestrate generative model parameters directly from the browser.

## Key Features
- **Form-Based Component Management**: Interactive Gradio forms to dynamically add, update, and remove model components such as matrices (A, B, C, D) and vectors.
- **State Space Editing**: Live validation and manipulation of hierarchical state spaces and modalities.
- **Live Markdown Synchronization**: A split-pane layout giving real-time visualization of changes mapping directly into the standard GNN `.md` underlying storage format.
- **Headless Mode Capable**: Saves supplied Markdown as a headless pipeline artifact; interactive editing requires Gradio.

## Quick Start
You can launch this GUI directly via the pipeline's GUI orchestrator:
```bash
# General invocation
uv run python src/gnn/22_gui.py --gui-types gui_1 --target-dir input/gnn_files/ --output-dir output/22_gui_output/ --interactive

# Headless setup
uv run python src/gnn/22_gui.py --gui-types gui_1 --headless
```

## Internal Architecture
- **`__init__.py`**: Public module registry pointing to execution entry points.
- **`ui.py`**: Gradio interface layouts and event endpoint bindings.
- **`processor.py`**: Intermediary logic connecting interface behaviors to core GNN parsing structures.
- **`markdown.py`**: Text synchronization helpers that add, update, remove, and parse component blocks and state-space entries without corrupting manual edits.
- **`mcp.py`**: Optional Model Context Protocol registration — `register_gui_tools()` delegates tool registration to the parent `gui` module (`register_module_tools("gui")`); the submodule defines no tools of its own.

## Saved state declarations

`add_state_space_entry` places a new declaration inside an existing canonical
`## StateSpaceBlock`, before the following section, so native GNN file readers
can discover it after export. Existing text and line endings outside the added
declaration remain intact. The editor also accepts legacy spaced state headings;
these editor aliases do not change the native readers' supported section names.
Physical duplicates are retained rather than silently combined.

The public `gui_1(..., headless=True)` entry point saves supplied edited Markdown
without starting a browser or requiring Gradio. It adds a headless artifact
marker; it does not run interactive callbacks or certify the model's scientific
validity. Text helpers retain the 8,388,608-character input limit.
