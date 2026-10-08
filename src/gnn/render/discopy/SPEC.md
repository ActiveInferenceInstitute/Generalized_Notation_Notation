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

- `translator.py` (1648 lines) — Core GNN-to-DisCoPy translation
- Template-based code generation with parametric diagram construction

## Dependencies

Target: `discopy >= 1.1.0`
