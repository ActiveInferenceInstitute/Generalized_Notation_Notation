"""Syntax-level GNN validation: regex parser, multi-level validator, cross-format consistency.

This package owns the syntax side of GNN validation:

- ``gnn.schema_validator.syntax`` — ``GNNParser``, the regex-based parser that
  turns GNN source text into a ``ParsedGNN`` structure.
- ``gnn.schema_validator.validator`` — ``GNNValidator`` and the
  ``validate_gnn_file_comprehensive`` entry point: schema, round-trip, and
  semantic checks.
- ``gnn.schema_validator.cross_format`` — ``CrossFormatValidator`` and the
  ``validate_cross_format_consistency`` / ``validate_schema_consistency``
  entry points for consistency across output formats.
"""

from gnn.schema_validator.cross_format import (
    CrossFormatValidationResult,
    CrossFormatValidator,
    validate_cross_format_consistency,
    validate_schema_consistency,
)
from gnn.schema_validator.syntax import ROUND_TRIP_AVAILABLE, GNNParser
from gnn.schema_validator.validator import (
    GNNValidator,
    validate_gnn_file_comprehensive,
)
from gnn.types import ParsedGNN, ValidationLevel, ValidationResult

__all__ = [
    "ROUND_TRIP_AVAILABLE",
    "CrossFormatValidationResult",
    "CrossFormatValidator",
    "GNNParser",
    "GNNValidator",
    "ParsedGNN",
    "ValidationLevel",
    "ValidationResult",
    "validate_cross_format_consistency",
    "validate_gnn_file_comprehensive",
    "validate_schema_consistency",
]
