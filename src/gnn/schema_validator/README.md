# Schema Validator Module

Syntax-level validation and parsing for GNN model files: a regex-based parser, a
multi-level validator with round-trip and schema checks, and cross-format
consistency verification. The package was split in 3.2.0 from a former
single-file module (parser half → `syntax.py`, validation half →
`validator.py`) plus a standalone cross-format module (→ `cross_format.py`).

## Module Structure

```
src/gnn/schema_validator/
├── __init__.py             # Curated public surface (__all__)
├── syntax.py               # GNNParser facade: parse_file/parse_content + ROUND_TRIP_AVAILABLE
├── format_detection.py     # FormatDetectionMixin: extension/content/binary format detection
├── section_parsers.py      # SectionParsersMixin: per-section regex parsers
├── validator.py            # GNNValidator facade: validate_file + validate_gnn_file_comprehensive
├── validation_levels.py    # LevelResolverMixin: level resolution and rank comparison
├── structural_checks.py    # StructuralChecksMixin: format/structure/strict/research gates
├── round_trip_checks.py    # RoundTripChecksMixin: round-trip + cross-format validation
├── semantic_checks.py      # SemanticChecksMixin: semantic/mathematical checks
├── cross_format.py         # CrossFormatValidator: cross-format consistency validation
├── README.md               # This documentation
└── AGENTS.md               # Agent scaffolding documentation
```

## Usage

```python
from gnn.schema_validator import (
    GNNParser,
    GNNValidator,
    validate_gnn_file_comprehensive,
    validate_cross_format_consistency,
)

# Parse GNN source text into a structured model
parser = GNNParser(enhanced_validation=True)
parsed = parser.parse_file("my_model.gnn")

# Full validation (schema + semantic + optional round-trip)
result = GNNValidator().validate_file("my_model.gnn")
assert result.is_valid

# One-shot convenience
result = validate_gnn_file_comprehensive("my_model.gnn")
```

Notes:

- `FORMAL_PARSER_AVAILABLE` is `False` (the Lark-based formal parser was
  removed); `ROUND_TRIP_AVAILABLE` reflects whether `gnn.parsers` imports
  cleanly. `GNNValidator` degrades accordingly.
- `ValidationLevel`, `ValidationResult`, and `ParsedGNN` live in
  `src/gnn/types/` and are re-exported here for convenience.
- Enhanced saved-format parsing exposes a registry variable's canonical datatype
  value (for example, `integer` or `binary`) in `GNNVariable.data_type`. A plain
  string remains a string; enum display names such as `DataType.INTEGER` do not
  replace the saved datatype. Markdown continues to retain its authored spelling.

## See Also

- [Package root](../../../README.md)
- [Validation module](../validation/README.md) — semantic/quality validation layered on top
- [Parsers module](../parsers/README.md) — formal parsing system used for round-trip testing
