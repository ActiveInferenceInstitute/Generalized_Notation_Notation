# GUI 2: Visual Matrix Editor

## Overview
This module represents the second major GUI implementation for the repository: an advanced Visual Matrix Editor dedicated to matrix-level parameter adjustments using intuitive, visual dashboards rather than text forms.

## Key Features
- **Visual Matrix Representation**: High-performance rendering of Active Inference generative matrices via Plotly.
- **Interactive Dimension Controls**: Resize matrices live with +/- controls while preserving existing values.
- **POMDP Bootstrapping**: Features logic (via `get_pomdp_template`) to launch instantly with default Dirichlet matrices populated correctly.

## Architecture Structure
- **`ui.py`**: Gradio layout — tabs for A/B/C/D matrices with heatmaps and dimension controls.
- **`matrix_editor.py`**: Matrix parsing/serialization and validation helpers (`create_matrix_from_gnn`, `update_gnn_from_matrix`, `validate_visual_matrix_dimensions`).
- **`processor.py`**: Execution orchestrator — headless artifact generation and interactive server launch.
- **`__init__.py`**: Facade wrapping and integration parameters for the `gui` super-module.

## Usage
GUI 2 is executed via the standard orchestrator flow:
```bash
uv run python src/gnn/22_gui.py --gui-types gui_2 --interactive
```
When launched interactively, the server listens on port 7861 (http://localhost:7861); without the `gui` extra installed (or with `--headless`), GUI 2 writes static artifacts (`visual_model_gui2.md`, `visual_matrices.json`, `gui_2_status.json`) instead.

## Saved models and migration

GNN declares the transition tensor as `B[next_state, previous_state, action]`.
The editor displays one action plane at a time and restores that declared
order on export, including when the number of actions differs from the number
of states. Other model parameters remain in their authored order.

From 4.1.0, saved A/B/C/D values retain the precision of the finite Python
floats accepted by the editor, using the shortest decimal representation that
reads back to the same float. Declared `type=float` values, signed preferences
and structural zeros retain their numerical meaning. This does not preserve
arbitrary-precision decimal text or original numeric formatting.

Earlier exports rounded numbers to six significant digits and could write
malformed vector braces or action-first transition bytes. Reload the
authoritative canonical source and regenerate those exports before using them
for inference. Values already rounded cannot be recovered from the rounded
file; previously saved axis order is not automatically inferred or repaired.
