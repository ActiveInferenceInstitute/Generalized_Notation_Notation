# DisCoPy Renderer — Technical Specification

**Version**: [pyproject.toml](../../../../pyproject.toml) (canonical)

## Purpose

Generates Python code using DisCoPy for categorical diagram construction.

## Code Generation

- Maps GNN model structure to categorical diagrams
- Generates morphism compositions and functorial mappings
- Produces executable DisCoPy Python scripts

## Output

- Python script files using `discopy` API
- Diagram serialization (JSON)
- Model display names are serialized as Python literals in generated module
  documentation and executable slots. Quotes, newlines, backslashes, Unicode and
  braces remain data and retain their values in the exported JSON.

## Architecture

- `discopy_renderer.py` owns the supported `render_gnn_to_discopy` generator.
- `translator.py` preserves compatibility imports for separate parsing,
  abstract diagram, file translation, bootstrap and template owners.
- `matrix_builders.py` and the file-level JAX-backed MatrixDiagram utility
  remain experimental and unsupported for numerical evaluation. Real-table
  pair decoding and tensor/matrix constructor semantics require an explicit
  compatibility decision; see [the support boundary](README.md#experimental-matrix-route).
- Permutation options validate metadata without reordering diagram structure.

## Dependencies

Target: `discopy >= 1.1.0`
