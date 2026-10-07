# Schema Validator Module Specification

## Overview
Syntax-level GNN validation: the regex-based parser that turns GNN source text into `ParsedGNN`, the multi-level validator (schema, round-trip, semantic checks), and cross-format consistency validation.

## Components
- `syntax.py` - `GNNParser` (regex-based; `parse_file` / `parse_content` with extension- and content-based format detection including binary pickle; parses StateSpaceBlock, Connections, parameters, Equations, Time, ActInfOntologyAnnotation, and Signature sections; computes semantic checksums), `ROUND_TRIP_AVAILABLE` flag (true when `gnn.parsers` imports cleanly)
- `validator.py` - `GNNValidator.validate_file(file_path, validation_level)` (level resolution and rank comparison, basic structure, strict requirements, research standards, round-trip validation, cross-format consistency, semantic consistency, mathematical checks including stochasticity), `validate_gnn_file_comprehensive(file_path) -> ValidationResult`
- `format_detection.py` / `section_parsers.py` - `FormatDetectionMixin` / `SectionParsersMixin`: the parsing halves of `GNNParser` (format and binary-pickle detection; the per-section regex parsers), mixed into the facade class
- `validation_levels.py` / `structural_checks.py` / `round_trip_checks.py` / `semantic_checks.py` - `LevelResolverMixin` / `StructuralChecksMixin` / `RoundTripChecksMixin` / `SemanticChecksMixin`: the validation halves of `GNNValidator` (level resolution and rank comparison; format/structure/strict/research gates; round-trip and cross-format checks; semantic and mathematical checks), mixed into the facade class
- `cross_format.py` - `CrossFormatValidator` (`validate(files)`, `validate_cross_format_consistency(gnn_content)`, `validate_schema_definitions_consistency()`) and `CrossFormatValidationResult` (consistency rate, per-format issues, semantic checksums), convenience `validate_cross_format_consistency` / `validate_schema_consistency`, and a small `__main__` CLI
- `__init__.py` - Curated public surface; re-exports `ParsedGNN`, `ValidationLevel`, `ValidationResult` from `gnn.types`

## Invariants
- Parser and validator are separated by design: `validator.py` imports `GNNParser` from `syntax.py`, never the reverse.
- `ROUND_TRIP_AVAILABLE` degrades gracefully when the enhanced
  `gnn.parsers.GNNParsingSystem` import fails; basic parsing and validation
  remain available when required core types are importable.
- `syntax.py` owns that capability flag; validator and public package exports
  reuse it. Optional import failure logs its exception type and cause. Known
  parser failures retain the chained source exception in `parse_degraded`;
  the basic markdown fallback and public result shape remain available.
- Validation levels compare through rank mapping; accepted string forms resolve to `ValidationLevel` members (`basic`/`standard`/`strict`/`research`/`round_trip`).

## Key Exports
```python
from gnn.schema_validator import (
    ROUND_TRIP_AVAILABLE,
    CrossFormatValidationResult,
    CrossFormatValidator,
    GNNParser,
    GNNValidator,
    ParsedGNN,
    ValidationLevel,
    ValidationResult,
    validate_cross_format_consistency,
    validate_gnn_file_comprehensive,
    validate_schema_consistency,
)
```

## Receipts
```bash
uv run --extra dev python -m pytest tests/schema_validator/test_cross_format.py \
  tests/schema_validator/test_validation_level_resolution.py \
  tests/schema_validator/test_time_varying_dynamics_parsing.py \
  tests/gnn/test_gnn_validator.py tests/gnn/test_gnn_cross_format_validator.py -q
```

---
## Documentation
- **[README](README.md)**: Module Overview
- **[AGENTS](AGENTS.md)**: Agentic Workflows
- **[SPEC](SPEC.md)**: Architectural Specification
