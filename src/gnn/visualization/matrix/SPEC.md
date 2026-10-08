# Matrix Visualization — Technical Specification

**Version**: [pyproject.toml](../../../../pyproject.toml) (canonical)

## Supported Matrix Types

- 2-D likelihood matrices (`A`)
- 2-D passive/single-action transition matrices (`B`)
- 3D POMDP tensors (per-action slices)
- Prior distributions (D vectors)

## Heatmap Configuration

- Colormap: `viridis` (default), configurable
- Annotations: cell values shown when matrix ≤ 10×10
- Statistical sidebar: mean, std, min, max per row/column

## 3D Tensor Handling

Per-action slice visualization with shared colorbar and cross-slice statistics.
For POMDP transition tensors, `B` shape is `(next_state, previous_state,
action)`. Stochastic validation sums over `next_state` for each
`previous_state`/`action` column. Every action slice is exported to CSV.

## Output Formats

- PNG (native matrix artifacts, 300 DPI)
- Matplotlib formats selected by the Python caller's output filename; the
  pipeline does not expose a separate SVG command-line option
- CSV (always, alongside visual output)
