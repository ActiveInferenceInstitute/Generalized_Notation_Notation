# PyMDP Renderer — Technical Specification

**Version**: [pyproject.toml](../../../../pyproject.toml) (canonical)

## Purpose

Generates Python code for PyMDP POMDP simulation.

## Code Generation

- Maps GNN model structure to PyMDP agent constructor
- Generates transition matrices (A, B), preference vectors (C), prior (D)
- Produces standalone simulation scripts

## Output

- Python scripts using `pymdp` API
- JSON parameter files

## Architecture

```
pymdp/
├── __init__.py             # Package exports
├── pymdp_renderer.py       # Canonical renderer
├── pymdp_templates.py      # Pipeline and standalone runner templates
└── ...
```

## Runtime Contract

- Generated scripts target `inferactively-pymdp>=1.0.0`.
- Required matrices are `A`, `B`, `C`, and `D`; factored POMDP specs are
  composed by `render/pomdp_processor/_spec_generation.py` before this renderer is called.
- Pipeline source embeds the JSON-clean specification as a Python literal.
  Authored string contents, including `true`, `false`, `null`, escapes and Unicode,
  retain their values; JSON booleans and null become Python booleans and `None`.
- Both runner modes serialize authored model names and annotations in the module
  docstring and executable text slots. Quotes, backslashes, newlines, Unicode and
  braces remain data. Default `output/pymdp_simulations/<model_name>` path values
  and the pipeline `PYMDP_OUTPUT_DIR` override retain their existing behavior;
  this serialization does not change filesystem name selection or model execution.
