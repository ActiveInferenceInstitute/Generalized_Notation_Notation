"""
GNN schema validation module.

This module provides comprehensive validation, parsing, and analysis capabilities
for GNN (Generalized Notation Notation) model files with enhanced round-trip
testing support and cross-format validation.

The regex-based GNNParser lives in gnn/schema_validator/syntax.py; this module
owns GNNValidator and the validate_gnn_file_comprehensive entry point.

Enhanced Features:
- Complete format ecosystem support (23 formats)
- Binary file validation for pickle/binary formats
- Cross-format consistency validation
- Round-trip semantic preservation testing
- Enhanced error reporting and suggestions
"""

import logging
from pathlib import Path
from typing import Any, Optional, Union, cast

from gnn.schema_validator.round_trip_checks import RoundTripChecksMixin
from gnn.schema_validator.semantic_checks import SemanticChecksMixin
from gnn.schema_validator.structural_checks import StructuralChecksMixin
from gnn.schema_validator.syntax import ROUND_TRIP_AVAILABLE, GNNParser
from gnn.schema_validator.validation_levels import LevelResolverMixin
from gnn.schemas.section_contract import REQUIRED_SECTIONS
from gnn.types import (
    ValidationLevel,
    ValidationResult,
)

# RoundTripResult is already imported from gnn.types above
# No need to import it again from testing module to avoid circular deps

logger = logging.getLogger(__name__)

# syntax.py owns optional parser availability. Its required module import above
# preserves original failure instead of substituting an unusable GNNParser.


class GNNValidator(
    LevelResolverMixin,
    StructuralChecksMixin,
    RoundTripChecksMixin,
    SemanticChecksMixin,
):
    def __init__(
        self,
        schema_path: Optional[Path] = None,
        enable_round_trip_testing: bool = False,
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        enable_cross_validation: bool = True,
    ) -> None:
        """Initialize the instance.

        ``schema_path`` is retained for cross-format callers that bind a
        validator to a specific schema file; it is not parsed at init.
        """
        if schema_path is None:
            schema_path = Path(__file__).parent.parent / "schemas/json.json"

        self.schema_path = schema_path
        self.enable_round_trip_testing = (
            enable_round_trip_testing and ROUND_TRIP_AVAILABLE
        )
        self.validation_level = validation_level

        # Initialize enhanced parser
        self.parser = GNNParser(enhanced_validation=True)

        # Initialize cross-format validator (optionally, to avoid recursion)
        self.cross_validator = None
        if enable_cross_validation:
            try:
                from gnn.schema_validator.cross_format import CrossFormatValidator

                self.cross_validator = CrossFormatValidator()
            except ImportError:
                self.cross_validator = None

        # Initialize round-trip tester if enabled
        if self.enable_round_trip_testing:
            try:
                from gnn.testing.round_trip_tester import GNNRoundTripTester

                self.round_trip_tester = GNNRoundTripTester()
                logger.info("Round-trip testing enabled for comprehensive validation")
            except Exception as e:
                logger.warning(f"Could not initialize round-trip tester: {e}")
                self.enable_round_trip_testing = False

    def validate_file(
        self,
        file_path: Union[str, Path],
        validation_level: Union[ValidationLevel, str, None] = None,
    ) -> ValidationResult:
        """Enhanced validation with comprehensive testing capabilities."""
        import time

        start_time = time.time()

        validation_level = validation_level or self.validation_level
        # Resolve string levels before the broad exception handler below:
        # an invalid level string must raise at the public API, not be
        # downgraded to a generic validation error.
        if isinstance(validation_level, str):
            validation_level = self._resolve_level(validation_level)
        file_path = Path(file_path)

        result = ValidationResult(
            is_valid=True,
            validation_level=validation_level,
            format_tested=self._detect_file_format(file_path),
        )

        try:
            # Step 1: Basic file access and format detection
            file_format = self._detect_file_format(file_path)
            result.format_tested = file_format

            # Handle binary formats
            if file_format in ["binary", "pickle"]:
                return self._validate_binary_file(file_path, result)

            # Read text content
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError as e:
                result.errors.append(f"File encoding error: {e}")
                result.is_valid = False
                return result

            # Step 2: Parse the file using enhanced parser
            try:
                parsed_gnn = self.parser.parse_file(file_path)
                result.semantic_checksum = parsed_gnn.semantic_checksum
                result.metadata["parsed_successfully"] = True
                result.metadata["source_format"] = parsed_gnn.source_format
            except Exception as e:
                result.errors.append(f"Parsing failed: {e}")
                result.is_valid = False
                if validation_level == ValidationLevel.BASIC:
                    return result
                # Continue with content-based validation for higher levels
                parsed_gnn = cast(Any, None)

            # Step 3: Validation based on level
            # Use safe rank comparisons to avoid Enum comparison issues
            if self._level_rank(validation_level) >= self._level_rank(
                ValidationLevel.BASIC
            ):
                self._validate_basic_structure(content, result, file_format)

            if (
                self._level_rank(validation_level)
                >= self._level_rank(ValidationLevel.STANDARD)
                and parsed_gnn
            ):
                self._validate_semantics(parsed_gnn, result)

            if self._level_rank(validation_level) >= self._level_rank(
                ValidationLevel.STRICT
            ):
                self._validate_strict_requirements(parsed_gnn, content, result)

            if self._level_rank(validation_level) >= self._level_rank(
                ValidationLevel.RESEARCH
            ):
                self._validate_research_standards(parsed_gnn, content, result)

            # Step 4: Round-trip testing if enabled and requested
            if (
                (
                    isinstance(validation_level, ValidationLevel)
                    and validation_level == ValidationLevel.ROUND_TRIP
                )
                and self.enable_round_trip_testing
                and parsed_gnn
            ):
                self._perform_round_trip_validation(parsed_gnn, result)

            # Step 5: Cross-format consistency if available
            # Use safe comparison via rank to avoid Enum ordering errors
            if (
                self._level_rank(validation_level)
                >= self._level_rank(ValidationLevel.STRICT)
                and self.cross_validator
                and parsed_gnn
            ):
                self._validate_cross_format_consistency(content, result)

            # Final result determination
            result.is_valid = len(result.errors) == 0

            # Performance metrics
            end_time = time.time()
            result.performance_metrics = {
                "validation_time": end_time - start_time,
                "content_length": float(len(content)),
            }
            result.validation_level = validation_level

            return result

        except Exception as e:
            result.errors.append(f"Validation failed with exception: {e}")
            result.is_valid = False
            return result

    def _validate_structure(self, content: str, result: ValidationResult) -> Any:
        """Previous method - now delegates to markdown validation."""
        self._validate_markdown_structure(content, result)


def validate_gnn_file_comprehensive(file_path: Union[str, Path]) -> ValidationResult:
    """Validate a GNN file with the full GNNValidator pipeline."""
    validator = GNNValidator()
    return validator.validate_file(file_path)
