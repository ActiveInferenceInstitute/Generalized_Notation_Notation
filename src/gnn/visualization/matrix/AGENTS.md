# Visualization Matrix Sub-module

## Overview

Matrix visualization module for GNN model parameters. Generates heatmaps,
statistics, correlation plots, and POMDP tensor analysis from the
`MatrixVisualizer` class.

## Architecture

```
matrix/
├── __init__.py       # Package exports
├── visualizer.py     # MatrixVisualizer class
├── transition_analysis.py # Canonical POMDP tensor diagnostics and figure assembly
├── extract.py        # Matrix extraction from parsed models
└── compat.py         # Shared helper exports
```

## Key Class: `MatrixVisualizer`

### Core Methods

- **`generate_matrix_heatmap(matrix_name, matrix, output_path, **kwargs) -> bool`** — Annotated heatmap with auto-suppressed annotations beyond `_ANNOTATION_CELL_LIMIT` (25 cells).
- **`generate_3d_tensor_visualization(tensor_name, tensor, output_path, **kwargs) -> bool`** — Per-action-slice POMDP transition tensor visualization. For `B`, shape is `(next_state, previous_state, action)`.
- **`generate_pomdp_transition_analysis(tensor, output_path) -> bool`** — Comprehensive POMDP B-matrix analysis with entropy, stochasticity checks over next-state columns, and state dominance.
- **`generate_matrix_analysis(parameters, output_path) -> bool`** — Multi-panel matrix analysis from raw parameter lists.
- **`generate_matrix_statistics(parameters, output_path) -> bool`** — Statistical summary panels (distributions, ranges, sparsity).
- **`generate_combined_matrix_overview(matrices, output_path) -> bool`** — Side-by-side comparison of all matrices in a model.

### Extraction Methods

- **`extract_matrix_data_from_parameters(parameters) -> Dict[str, ndarray]`** — Extracts named matrices (A, B, C, D, E) from parameter lists.
- **`extract_from_parsed_gnn(parsed_data) -> Dict[str, ndarray]`** — Extracts matrices from parsed GNN model dict.

### Module-level functions (extract.py)

- **`convert_to_matrix(value, name="") -> Optional[ndarray]`** — Coerce nested lists/tuples to a numpy array (None when non-numeric or empty).
- **`extract_matrix_data_from_parameters(parameters) -> Dict[str, ndarray]`** — Pure parameter → matrix extraction; the `MatrixVisualizer` method delegates here.
- **`collect_visualization_matrices(parsed_data) -> Dict[str, ndarray]`** — Step-8 resolution order in one pure call: parameters → variables → raw `matrices` entries (with `data` keys); first non-empty stage wins. Re-exported at the package root as `collect_visualization_matrices`.

### Safety Features

- `_ANNOTATION_CELL_LIMIT = 25` — Skip per-cell text annotation on large matrices.
- `_MAX_FIGURE_DIMENSION = 200` — Prevent matplotlib RendererAgg pixel overflow.
- `_safe_figsize(width, height) -> Tuple[float, float]` — Clamp figure dimensions to prevent rendering crashes.
- CSV export alongside every heatmap and every 3-D tensor action slice for downstream accessibility.

## Parent Module

See [visualization/AGENTS.md](../AGENTS.md) for the overall visualization architecture.

**Version**: [pyproject.toml](../../../../pyproject.toml) (canonical)
**Last Updated**: 2026-09-24
