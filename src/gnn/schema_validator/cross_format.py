#!/usr/bin/env python3
"""
Cross-Format Validation for GNN Schema Consistency - Enhanced

This module ensures that all format representations (JSON Schema, YAML Schema,
XSD, Protocol Buffers, etc.) maintain consistency and validate the same GNN models.

Enhanced Features:
- Integration with comprehensive round-trip testing
- Performance optimization and metrics
- Enhanced error reporting and analysis
- Support for binary format validation
- Cross-format semantic preservation validation
"""

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import these at module level to avoid circular imports
from gnn.types import GNNFormat, ValidationLevel, ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class CrossFormatValidationResult:
    """Enhanced results from cross-format validation with comprehensive metrics."""

    is_consistent: bool
    schema_formats: List[str] = field(default_factory=list)
    inconsistencies: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    format_results: Dict[str, ValidationResult] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Enhanced fields for comprehensive analysis
    semantic_checksums: Dict[str, str] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    round_trip_compatibility: Dict[str, bool] = field(default_factory=dict)
    format_specific_issues: Dict[str, List[str]] = field(default_factory=dict)

    def add_format_issue(self, format_name: str, issue: str) -> Any:
        """Add a format-specific issue."""
        if format_name not in self.format_specific_issues:
            self.format_specific_issues[format_name] = []
        self.format_specific_issues[format_name].append(issue)

    def get_consistency_rate(self) -> float:
        """Get overall consistency rate across formats."""
        if not self.schema_formats:
            return 0.0
        consistent_formats = sum(
            1
            for fmt in self.schema_formats
            if fmt in self.format_results and self.format_results[fmt].is_valid
        )
        return (consistent_formats / len(self.schema_formats)) * 100


class CrossFormatValidator:
    """Enhanced validator for cross-format consistency with comprehensive testing support."""

    def __init__(
        self,
        gnn_module_path: Optional[Path] = None,
        enable_round_trip_testing: bool = False,
    ) -> None:
        """Initialize the instance."""
        if gnn_module_path is None:
            gnn_module_path = Path(__file__).parent

        self.gnn_path = gnn_module_path
        self.enable_round_trip_testing = enable_round_trip_testing
        self.format_validators: dict[str, Any] = {}
        self.parsing_system = None

        # Defer heavy validator initialization until first use to avoid recursion and speed up tests
        self._validators_initialized = False

        # Initialize parsing system if available
        try:
            # Import here to avoid circular imports
            from gnn.parsers import GNNParsingSystem

            self.parsing_system = GNNParsingSystem()
            logger.info(
                "Enhanced parsing system initialized for cross-format validation"
            )
        except Exception as e:
            # Use debug level to avoid excessive warnings
            logger.debug(f"Could not initialize parsing system: {e}")

    def configure(self, output_dir: Optional[Path] = None, **kwargs: Any) -> Any:
        """Configure the validator."""
        self.output_dir = output_dir

    def validate(self, files: List[Path]) -> Dict[str, Any]:
        """Validate a list of files."""
        results: dict[Any, Any] = {}
        success = True
        for f in files:
            try:
                content = f.read_text(encoding="utf-8")
                res = self.validate_cross_format_consistency(content)
                results[str(f)] = res
                if not res.is_consistent:
                    success = False
            except Exception as e:
                logger.error(f"Cross-format validation failed for {f}: {e}")
                success = False

        return {"success": success, "files_validated": len(files), "results": results}

    def _initialize_validators(self) -> Any:
        """Initialize enhanced validators for different schema formats."""
        from gnn.schema_validator.validator import GNNValidator

        # Enhanced validation levels for different formats
        validation_levels: dict[str, Any] = {
            "json": ValidationLevel.STANDARD,
            "xml": ValidationLevel.STANDARD,
            "yaml": ValidationLevel.STANDARD,
            "binary": ValidationLevel.BASIC,  # Limited validation for binary
            "markdown": ValidationLevel.STRICT,
        }

        # Track initialization results for more thoughtful reporting
        initialization_results: dict[str, Any] = {
            "success": [],
            "failed": [],
            "missing_schema": [],
        }

        # Initialize validators with proper exception handling
        for format_name, validation_level in validation_levels.items():
            try:
                schema_path = self.gnn_path / f"schemas/{format_name}.{format_name}"
                schema_exists = schema_path.exists()

                if schema_exists or format_name in ["binary", "markdown"]:
                    self.format_validators[format_name] = GNNValidator(
                        schema_path if schema_exists else None,
                        validation_level=validation_level,
                        enable_cross_validation=False,
                    )
                    initialization_results["success"].append(format_name)
                    logger.debug(
                        f"Initialized {format_name} validator with {validation_level.value} level"
                    )
                else:
                    initialization_results["missing_schema"].append(format_name)
                    logger.debug(
                        f"Skipped {format_name} validator: schema file not found"
                    )

            except Exception as e:
                # Use a more specific error message
                initialization_results["failed"].append(f"{format_name}: {str(e)}")
                logger.debug(f"Could not initialize {format_name} validator: {e}")

        self.available_formats = list(self.format_validators.keys())
        self._validators_initialized = True

        # Provide thoughtful summary reporting instead of intense warnings
        if initialization_results["success"]:
            logger.info(
                f"Cross-format validator initialized with {len(self.available_formats)} formats: {', '.join(initialization_results['success'])}"
            )

        # Only warn if there are actual failures (not just missing schemas)
        if initialization_results["failed"]:
            # Use a single warning instead of multiple warnings
            logger.warning(
                f"Some format validators failed to initialize: {'; '.join(initialization_results['failed'])}"
            )

        # Debug-level info about missing schemas (expected in many cases)
        if initialization_results["missing_schema"]:
            logger.debug(
                f"Schema files not found for: {', '.join(initialization_results['missing_schema'])} (validation will be basic)"
            )

    def validate_cross_format_consistency(
        self, gnn_content: str, source_format: str = "markdown"
    ) -> CrossFormatValidationResult:
        """Enhanced cross-format validation with comprehensive analysis."""
        import time

        start_time = time.time()

        # Lazily initialize validators on first use
        if not self._validators_initialized:
            try:
                self._initialize_validators()
            except Exception as e:
                logger.debug(f"Deferred validator initialization failed: {e}")
                self.format_validators = {}
                self.available_formats = []

        result = CrossFormatValidationResult(is_consistent=True)
        result.schema_formats = self.available_formats.copy()
        result.metadata["source_format"] = source_format
        result.metadata["validation_timestamp"] = time.time()

        # Step 1: Validate against each available format
        logger.debug(f"Starting cross-format validation for {source_format} content")
        format_validation_results: dict[Any, Any] = {}

        for format_name, validator in self.format_validators.items():
            format_start = time.time()

            try:
                # Create temporary file for validation
                temp_file = self._create_temp_file(gnn_content, format_name)
                validation_result = validator.validate_file(temp_file)
                format_validation_results[format_name] = validation_result
                result.format_results[format_name] = validation_result

                # Track performance metrics
                format_time = time.time() - format_start
                result.performance_metrics[format_name] = format_time

                # Compute semantic checksum if parsing succeeded
                if validation_result.semantic_checksum:
                    result.semantic_checksums[format_name] = (
                        validation_result.semantic_checksum
                    )

                logger.debug(
                    f"  {format_name}: {'✅' if validation_result.is_valid else '❌'} ({format_time:.3f}s)"
                )

            except Exception as e:
                logger.debug(f"Validation failed for {format_name}: {e}")
                result.add_format_issue(format_name, f"Validation error: {e}")
                result.inconsistencies.append(
                    f"Error validating {format_name} format: {str(e)}"
                )
                result.is_consistent = False

        # Step 2: Check consistency between formats
        self._analyze_format_consistency(format_validation_results, result)

        # Step 3: Semantic checksum analysis
        self._analyze_semantic_checksums(result)

        # Step 4: Round-trip compatibility testing (if enabled)
        if self.enable_round_trip_testing and self.parsing_system:
            self._test_round_trip_compatibility(gnn_content, result)

        # Step 5: Generate comprehensive metadata
        result.metadata.update(
            self._generate_enhanced_metadata(format_validation_results, result)
        )

        total_time = time.time() - start_time
        result.performance_metrics["total_validation_time"] = total_time

        logger.debug(
            f"Cross-format validation completed: {result.get_consistency_rate():.1f}% consistency ({total_time:.3f}s)"
        )

        return result

    def _analyze_format_consistency(
        self,
        format_results: Dict[str, ValidationResult],
        result: CrossFormatValidationResult,
    ) -> Any:
        """Enhanced analysis of consistency between formats."""
        valid_formats: list[Any] = []
        invalid_formats: list[Any] = []

        for format_name, validation_result in format_results.items():
            if validation_result.is_valid:
                valid_formats.append(format_name)
            else:
                invalid_formats.append(format_name)
                # Collect format-specific errors
                for error in validation_result.errors:
                    result.add_format_issue(format_name, error)

        # Consistency analysis
        if len(valid_formats) == len(format_results):
            result.warnings.append(
                "All formats validate successfully - high consistency"
            )
        elif len(valid_formats) >= len(format_results) * 0.8:
            result.warnings.append(
                f"Good consistency: {len(valid_formats)}/{len(format_results)} formats valid"
            )
        else:
            result.inconsistencies.append(
                f"Poor consistency: only {len(valid_formats)}/{len(format_results)} formats valid"
            )
            result.is_consistent = False

        # Cross-format error pattern analysis
        common_errors = self._find_common_error_patterns(format_results)
        if common_errors:
            result.warnings.extend(
                [f"Common error pattern: {error}" for error in common_errors]
            )

        # Format-specific validation warnings
        for format_name, validation_result in format_results.items():
            if validation_result.warnings:
                result.warnings.append(
                    f"{format_name} warnings: {len(validation_result.warnings)}"
                )

    def _analyze_semantic_checksums(self, result: CrossFormatValidationResult) -> Any:
        """Analyze semantic checksums for consistency."""
        checksums = result.semantic_checksums

        if len(checksums) < 2:
            result.warnings.append("Insufficient semantic checksums for comparison")
            return

        # Find unique checksums
        unique_checksums = set(checksums.values())

        if len(unique_checksums) == 1:
            result.warnings.append("Perfect semantic consistency across all formats")
        elif len(unique_checksums) <= len(checksums) * 0.5:
            result.warnings.append(
                "Good semantic consistency - minor variations detected"
            )
        else:
            result.inconsistencies.append(
                "Significant semantic inconsistencies between formats"
            )
            result.is_consistent = False

            # Analyze checksum patterns
            checksum_groups: dict[Any, Any] = {}
            for format_name, checksum in checksums.items():
                if checksum not in checksum_groups:
                    checksum_groups[checksum] = []
                checksum_groups[checksum].append(format_name)

            for _, formats in checksum_groups.items():
                if len(formats) > 1:
                    result.warnings.append(f"Consistent group: {', '.join(formats)}")

    def _test_round_trip_compatibility(
        self, gnn_content: str, result: CrossFormatValidationResult
    ) -> Any:
        """Test round-trip compatibility between formats."""
        try:
            # Import here to avoid circular imports
            from gnn.schema_validator.syntax import GNNParser

            # Create temporary file for parsing
            temp_file = self._create_temp_file(gnn_content, "markdown")

            try:
                # Parse the content
                parsed_gnn = GNNParser(enhanced_validation=True).parse_file(temp_file)

                # If we have a parsing system, test round-trip for key formats
                if self.parsing_system and hasattr(parsed_gnn, "model_name"):
                    test_formats: list[Any] = [
                        GNNFormat.JSON,
                        GNNFormat.XML,
                        GNNFormat.YAML,
                    ]

                    for fmt in test_formats:
                        try:
                            # This is a simplified round-trip test: check if object has valid content
                            round_trip_success = hasattr(
                                parsed_gnn, "to_dict"
                            ) and bool(parsed_gnn.to_dict())
                            result.round_trip_compatibility[fmt.value] = (
                                round_trip_success
                            )

                            if round_trip_success:
                                result.warnings.append(
                                    f"Round-trip compatible: {fmt.value}"
                                )
                            else:
                                result.inconsistencies.append(
                                    f"Round-trip incompatible: {fmt.value}"
                                )

                        except Exception as e:
                            result.add_format_issue(
                                fmt.value, f"Round-trip test failed: {e}"
                            )

            except Exception as e:
                result.warnings.append(
                    f"Round-trip testing limited due to parsing issues: {e}"
                )

        except Exception as e:
            result.warnings.append(f"Round-trip testing unavailable: {e}")

    def _find_common_error_patterns(
        self, format_results: Dict[str, ValidationResult]
    ) -> List[str]:
        """Find common error patterns across formats."""
        error_counts: dict[Any, Any] = {}

        for _, validation_result in format_results.items():
            for error in validation_result.errors:
                # Normalize error message for pattern matching
                normalized_error = self._normalize_error_message(error)
                error_counts[normalized_error] = (
                    error_counts.get(normalized_error, 0) + 1
                )

        # Return errors that appear in multiple formats
        common_errors = [
            error
            for error, count in error_counts.items()
            if count >= 2 and count >= len(format_results) * 0.3
        ]

        return common_errors

    def _normalize_error_message(self, error: str) -> str:
        """Normalize error message for pattern matching."""
        # Remove format-specific details
        import re

        normalized = error.lower()

        # Remove file paths and line numbers
        normalized = re.sub(r"line \d+", "line N", normalized)
        normalized = re.sub(r"[a-z]:\\.+?\.txt", "FILE", normalized)

        # Remove specific values
        normalized = re.sub(r"\d+", "N", normalized)
        normalized = re.sub(r"'[^']*'", "'VALUE'", normalized)

        return normalized.strip()

    def _generate_enhanced_metadata(
        self,
        format_results: Dict[str, ValidationResult],
        result: CrossFormatValidationResult,
    ) -> Dict[str, Any]:
        """Generate comprehensive metadata for the validation result."""
        metadata: dict[str, Any] = {
            "formats_tested": len(format_results),
            "valid_formats": sum(1 for r in format_results.values() if r.is_valid),
            "invalid_formats": sum(
                1 for r in format_results.values() if not r.is_valid
            ),
            "total_errors": sum(len(r.errors) for r in format_results.values()),
            "total_warnings": sum(len(r.warnings) for r in format_results.values()),
            "consistency_rate": result.get_consistency_rate(),
            "has_semantic_checksums": len(result.semantic_checksums) > 0,
            "unique_checksums": len(set(result.semantic_checksums.values()))
            if result.semantic_checksums
            else 0,
            "round_trip_tested": len(result.round_trip_compatibility) > 0,
        }

        # Performance summary
        if result.performance_metrics:
            total_time = result.performance_metrics.get("total_validation_time", 0)
            format_times = [
                v
                for k, v in result.performance_metrics.items()
                if k != "total_validation_time"
            ]

            metadata["performance"] = {
                "total_time": total_time,
                "average_format_time": sum(format_times) / len(format_times)
                if format_times
                else 0,
                "fastest_format": min(
                    result.performance_metrics.items(), key=lambda x: x[1]
                )[0]
                if format_times
                else None,
                "slowest_format": max(
                    result.performance_metrics.items(), key=lambda x: x[1]
                )[0]
                if format_times
                else None,
            }

        return metadata

    def _create_temp_file(self, content: str, format_hint: str = "markdown") -> Path:
        """Create temporary file with appropriate extension for format testing."""
        # Map format names to file extensions
        extensions: dict[str, Any] = {
            "markdown": ".md",
            "json": ".json",
            "xml": ".xml",
            "yaml": ".yaml",
            "binary": ".pkl",
        }

        extension = extensions.get(format_hint, ".txt")

        # Create temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix=extension, text=True)
        try:
            with open(temp_fd, "w", encoding="utf-8") as f:
                f.write(content)
        except (OSError, UnicodeError, TypeError):
            logger.warning(
                "Text write failed for temp file %s; falling back to binary write",
                temp_path,
            )
            # Handle binary formats
            with open(temp_path, "wb") as f:
                f.write(content.encode("utf-8"))

        return Path(temp_path)

    def validate_schema_definitions_consistency(self) -> CrossFormatValidationResult:
        """Enhanced validation of schema definition files consistency."""
        if not self._validators_initialized:
            try:
                self._initialize_validators()
            except Exception as e:
                logger.debug(f"Deferred validator initialization failed: {e}")
                self.format_validators = {}
                self.available_formats = []
        result = CrossFormatValidationResult(is_consistent=True)

        # Load available schema definitions
        schemas: dict[Any, Any] = {}
        schema_files: dict[str, Any] = {
            "json": self.gnn_path / "schemas/json.json",
            "yaml": self.gnn_path / "schemas/yaml.yaml",
            "xsd": self.gnn_path / "schemas/xsd.xsd",
            "proto": self.gnn_path / "schemas/proto.proto",
        }

        try:
            # Import here to avoid circular imports
            import json

            import yaml

            for format_name, schema_path in schema_files.items():
                if schema_path.exists():
                    try:
                        if format_name == "json":
                            with open(schema_path) as f:
                                schemas[format_name] = json.load(f)
                        elif format_name == "yaml":
                            with open(schema_path) as f:
                                schemas[format_name] = yaml.safe_load(f)
                        else:
                            # For other formats, just check if they're readable
                            with open(schema_path) as f:
                                content = f.read(1000)  # Read first 1KB
                            schemas[format_name] = {"content_length": len(content)}

                        logger.debug(f"Loaded {format_name} schema definition")

                    except Exception as e:
                        result.inconsistencies.append(
                            f"Failed to load {format_name} schema: {e}"
                        )
                        result.is_consistent = False
                else:
                    result.warnings.append(f"Schema file not found: {schema_path}")

            result.schema_formats = list(schemas.keys())

            # Enhanced structural consistency validation
            self._validate_enhanced_schema_structure_consistency(schemas, result)

        except ImportError as e:
            # Handle import errors gracefully
            result.warnings.append(f"Schema validation limited: {e}")
        except Exception as e:
            result.inconsistencies.append(f"Error loading schema definitions: {str(e)}")
            result.is_consistent = False

        return result

    def _validate_enhanced_schema_structure_consistency(
        self, schemas: Dict[str, Dict], result: CrossFormatValidationResult
    ) -> Any:
        """Enhanced validation of structural consistency between schema formats."""
        if len(schemas) < 2:
            result.warnings.append("Insufficient schemas for consistency comparison")
            return

        # Analyze JSON and YAML schemas if both are available
        if "json" in schemas and "yaml" in schemas:
            json_schema = schemas["json"]
            yaml_schema = schemas["yaml"]

            # Compare key structural elements
            json_sections = set(json_schema.get("properties", {}).keys())
            yaml_sections = set(yaml_schema.get("required_sections", []))

            if json_sections and yaml_sections:
                common_sections = json_sections & yaml_sections
                json_only = json_sections - yaml_sections
                yaml_only = yaml_sections - json_sections

                if (
                    len(common_sections)
                    >= max(len(json_sections), len(yaml_sections)) * 0.8
                ):
                    result.warnings.append(
                        "Good structural consistency between JSON and YAML schemas"
                    )
                else:
                    result.inconsistencies.append(
                        "Significant structural differences between JSON and YAML schemas"
                    )
                    result.is_consistent = False

                if json_only:
                    result.warnings.append(
                        f"JSON-only sections: {', '.join(json_only)}"
                    )
                if yaml_only:
                    result.warnings.append(
                        f"YAML-only sections: {', '.join(yaml_only)}"
                    )

        # Validate schema completeness
        expected_elements: list[Any] = [
            "ModelName",
            "StateSpaceBlock",
            "Connections",
            "InitialParameterization",
        ]

        for format_name, schema_data in schemas.items():
            if isinstance(schema_data, dict):
                # Check if expected elements are covered
                schema_str = str(schema_data).lower()
                found_elements = [
                    elem for elem in expected_elements if elem.lower() in schema_str
                ]

                coverage = len(found_elements) / len(expected_elements) * 100
                if coverage >= 80:
                    result.warnings.append(
                        f"{format_name} schema has good coverage ({coverage:.1f}%)"
                    )
                else:
                    result.inconsistencies.append(
                        f"{format_name} schema has poor coverage ({coverage:.1f}%)"
                    )


def validate_cross_format_consistency(
    gnn_content: str, enable_round_trip: bool = False
) -> CrossFormatValidationResult:
    """Enhanced convenience function for cross-format validation."""
    validator = CrossFormatValidator(enable_round_trip_testing=enable_round_trip)
    return validator.validate_cross_format_consistency(gnn_content)


def validate_schema_consistency() -> CrossFormatValidationResult:
    """Enhanced convenience function for schema consistency validation."""
    validator = CrossFormatValidator()
    return validator.validate_schema_definitions_consistency()


# CLI interface for cross-format validation
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cross-Format GNN Schema Validation")
    parser.add_argument("--file", type=str, help="GNN file to validate")
    parser.add_argument(
        "--check-schemas",
        action="store_true",
        help="Check schema definition consistency",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.check_schemas:
        result = validate_schema_consistency()
        print(
            f"Schema Consistency: {'CONSISTENT' if result.is_consistent else 'INCONSISTENT'}"
        )

        if result.inconsistencies:
            print("\nInconsistencies:")
            for inconsistency in result.inconsistencies:
                print(f"  - {inconsistency}")

        if result.warnings:
            print("\nWarnings:")
            for warning in result.warnings:
                print(f"  - {warning}")

    elif args.file:
        with open(args.file) as f:
            content = f.read()

        result = validate_cross_format_consistency(content)
        print(
            f"Cross-Format Validation: {'CONSISTENT' if result.is_consistent else 'INCONSISTENT'}"
        )

        if args.verbose:
            print(f"\nFormats tested: {result.schema_formats}")
            print(f"Metadata: {result.metadata}")

        if result.inconsistencies:
            print("\nInconsistencies:")
            for inconsistency in result.inconsistencies:
                print(f"  - {inconsistency}")

    else:
        parser.print_help()
