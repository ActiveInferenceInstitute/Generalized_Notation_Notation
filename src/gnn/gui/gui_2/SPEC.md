# GUI 2 — Technical Specification

**Version**: [pyproject.toml](../../../../pyproject.toml) (canonical)

## Purpose

Second-generation GNN constructor GUI: a Gradio web app for visual matrix editing.

## Features

- Interactive DataFrame matrix editing with +/- dimension controls
- Matrix heatmaps and vector bar charts (Plotly when available)
- Tabbed matrix/vector editing (A, B, C, D) with live regeneration of the GNN markdown
- Real-time validation feedback
- POMDP template-based initialization
- Saved transition tensors retain the declared GNN order
  `B[next_state, previous_state, action]`; the editor's internal action-first
  planes are converted only at the named B parsing/serialization boundaries.
- Exported `type=float` values use shortest round-trip representations of
  accepted finite Python floats, preserving their precision, signed values
  and structural zeros. Original decimal formatting is not retained.

See [saved-model migration](README.md#saved-models-and-migration) for earlier
six-digit exports and transition/vector artifacts that require regeneration
from authoritative source.

## Technology

- `gradio` (from the `gui` extra in `pyproject.toml`)
- `plotly` (interactive plots; recovery: basic displays)
- `numpy` (matrix handling)

## Architecture

- `ui.py` — Main UI layout and event handling
- `matrix_editor.py` — Matrix parsing, serialization, and validation helpers
- `processor.py` — Pipeline entry point, headless artifacts, and server launch
