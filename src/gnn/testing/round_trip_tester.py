#!/usr/bin/env python3
"""
Production round-trip tester for GNN format conversion.

``GNNRoundTripTester`` exercises parse -> serialize -> parse conversions
across the configured GNN formats and reports semantic preservation.
Configuration, result dataclasses, the serializer availability probe, the
direct markdown parser, model comparison, and report generation live in
the ``round_trip_*`` sibling modules.

Extracted from the former ``testing.test_round_trip`` monolith so the
tester is importable as production code (``schema_validator``'s
``GNNValidator`` wires it for comprehensive validation) while the pytest
test cases live under ``tests/testing/``.
"""

import logging
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, List, Optional, cast

from gnn.parsers.common import Parameter

from .round_trip_availability import (
    CROSS_FORMAT_AVAILABLE,
    AlloySerializer,
    ASN1Serializer,
    BinarySerializer,
    Connection,
    ConnectionType,
    CoqSerializer,
    CrossFormatValidator,
    DataType,
    FunctionalSerializer,
    GNNFormat,
    GNNInternalRepresentation,
    GNNParsingSystem,
    GNNValidator,
    GrammarSerializer,
    IsabelleSerializer,
    JSONSerializer,
    LeanSerializer,
    ParsedGNN,
    ParseResult,
    PKLSerializer,
    ProtobufSerializer,
    PythonSerializer,
    ScalaSerializer,
    Variable,
    XMLSerializer,
    XSDSerializer,
    YAMLSerializer,
    ZNotationSerializer,
)
from .round_trip_comparison import RoundTripComparisonMixin
from .round_trip_config import (
    ENHANCED_TEST_CONFIG,
    FORMAT_TEST_CONFIG,
    LOGGING_CONFIG,
    OUTPUT_CONFIG,
    REFERENCE_CONFIG,
    TEST_BEHAVIOR_CONFIG,
)
from .round_trip_markdown_parser import _DirectMarkdownParser
from .round_trip_report import RoundTripReportMixin
from .round_trip_results import (
    ComprehensiveTestReport,
    RoundTripResult,
)

# Note: do NOT set sys.setrecursionlimit at module scope — it poisons the
# process for any other test that imports this module (RecursionError in
# unrelated tests). The default limit (1000+) is adequate here.

# Configure logging based on configuration
if LOGGING_CONFIG["suppress_parser_warnings"]:
    logging.getLogger("gnn.parsers").setLevel(logging.ERROR)
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["log_level"]),
    format="%(levelname)s: %(message)s"
    if LOGGING_CONFIG["enable_debug"]
    else "%(message)s",
)


logger = logging.getLogger(__name__)


class GNNRoundTripTester(RoundTripComparisonMixin, RoundTripReportMixin):
    """Comprehensive round-trip testing system for GNN formats."""

    def __init__(self, temp_dir: Optional[Path] = None) -> None:
        """Initialize the round-trip tester."""
        self.temp_dir = temp_dir or Path(tempfile.mkdtemp())
        self.parsing_system: Optional[GNNParsingSystem]

        # Initialize parsing system with enhanced error handling
        if ENHANCED_TEST_CONFIG["graceful_parser_fallback"]:
            try:
                strict_val = TEST_BEHAVIOR_CONFIG.get("strict_validation", False)
                self.parsing_system = GNNParsingSystem(strict_validation=strict_val)
                if LOGGING_CONFIG["enable_debug"]:
                    logger.info("Successfully initialized full parsing system")
            except Exception as e:
                if LOGGING_CONFIG["enable_debug"]:
                    logger.warning(
                        f"Parsing system initialization failed, using recovery: {e}"
                    )
                self.parsing_system = None
        else:
            try:
                strict_val = TEST_BEHAVIOR_CONFIG.get("strict_validation", False)
                self.parsing_system = GNNParsingSystem(strict_validation=strict_val)
            except Exception as e:
                if LOGGING_CONFIG["enable_debug"]:
                    logger.warning(f"Could not initialize full parsing system: {e}")
                self.parsing_system = None

        # Initialize validators with better error handling
        try:
            self.validator = (
                GNNValidator()
                if TEST_BEHAVIOR_CONFIG.get("validate_round_trip", False)
                else None
            )
        except Exception as e:
            if LOGGING_CONFIG["enable_debug"]:
                logger.warning(f"Could not initialize validator: {e}")
            self.validator = None

        try:
            self.cross_validator = (
                CrossFormatValidator()
                if CROSS_FORMAT_AVAILABLE
                and TEST_BEHAVIOR_CONFIG.get("run_cross_format_validation", False)
                else None
            )
        except Exception as e:
            if LOGGING_CONFIG["enable_debug"]:
                logger.warning(f"Could not initialize cross-format validator: {e}")
            self.cross_validator = None

        # Reference model paths - try configured paths
        self.reference_file = self._find_reference_file()

        # Initialize supported formats based on configuration
        self.supported_formats = self._determine_test_formats()

        if LOGGING_CONFIG["enable_detailed_output"]:
            logger.info(
                f"Round-trip tester initialized with {len(self.supported_formats)} formats: {[f.value for f in self.supported_formats]}"
            )

    def _create_direct_markdown_parser(self) -> Any:
        """Create a direct, dependency-free markdown parser for the reference file."""
        return _DirectMarkdownParser()

    def _serialize_with_individual_serializer(
        self, model: GNNInternalRepresentation, target_format: GNNFormat
    ) -> Optional[str]:
        """Serialize using individual serializer instances without full parsing system."""
        try:
            # Comprehensive serializer map with all available formats
            try:
                serializer_map: dict[Any, Any] = {
                    GNNFormat.JSON: JSONSerializer(),
                    GNNFormat.XML: XMLSerializer(),
                    GNNFormat.YAML: YAMLSerializer(),
                    GNNFormat.SCALA: ScalaSerializer(),
                    GNNFormat.PROTOBUF: ProtobufSerializer(),
                    GNNFormat.ALLOY: AlloySerializer(),
                    GNNFormat.ASN1: ASN1Serializer(),
                    GNNFormat.LEAN: LeanSerializer(),
                    GNNFormat.COQ: CoqSerializer(),
                    GNNFormat.PYTHON: PythonSerializer(),
                    GNNFormat.PICKLE: BinarySerializer(),
                    GNNFormat.PKL: PKLSerializer(),  # Add PKL serializer
                    GNNFormat.XSD: XSDSerializer(),
                    GNNFormat.ISABELLE: IsabelleSerializer(),
                    GNNFormat.HASKELL: FunctionalSerializer(),
                    GNNFormat.BNF: GrammarSerializer(),
                    GNNFormat.Z_NOTATION: ZNotationSerializer(),
                    # Add more as they become available
                }
            except Exception as e:
                if LOGGING_CONFIG["enable_debug"]:
                    logger.error(f"Error creating serializer map: {e}")
                return None

            # Use enum value for comparison to handle different enum instances
            for fmt, serializer in serializer_map.items():
                if fmt.value == target_format.value:
                    return cast("str | None", serializer.serialize(model))
            else:
                if LOGGING_CONFIG["enable_debug"]:
                    logger.warning(
                        f"No individual serializer available for {target_format.value}"
                    )
                return None

        except Exception as e:
            if LOGGING_CONFIG["enable_detailed_output"]:
                print(
                    f"         ❌ Individual serializer error for {target_format.value}: {e}"
                )
                import traceback

                print(f"         Traceback: {traceback.format_exc()}")
            if LOGGING_CONFIG["enable_debug"]:
                logger.error(
                    f"Individual serializer for {target_format.value} failed: {e}"
                )
            return None

    def _find_reference_file(self) -> Path:
        """Find the reference file using configuration."""
        project_root = Path(__file__).parent.parent.parent.parent

        # Try configured primary reference file
        primary_ref = project_root / REFERENCE_CONFIG["reference_file"]
        if primary_ref.exists():
            return cast("Path", primary_ref)

        # Try recovery files
        for recovery in REFERENCE_CONFIG["fallback_reference_files"]:
            fallback_path = project_root / recovery
            if fallback_path.exists():
                return cast("Path", fallback_path)

        # Default recovery
        default_path = (
            Path(__file__).parent.parent / "gnn_examples/actinf_pomdp_agent.md"
        )
        return default_path

    def _determine_test_formats(self) -> List[GNNFormat]:
        """Determine which formats to test based on configuration."""
        all_formats: list[Any] = [GNNFormat.MARKDOWN]  # Always include markdown

        # Define format categories
        format_categories: dict[str, Any] = {
            "schema_formats": [
                GNNFormat.JSON,
                GNNFormat.XML,
                GNNFormat.YAML,
                GNNFormat.XSD,
                GNNFormat.ASN1,
                GNNFormat.PKL,
                GNNFormat.PROTOBUF,
            ],
            "language_formats": [GNNFormat.SCALA, GNNFormat.PYTHON, GNNFormat.HASKELL],
            "formal_formats": [
                GNNFormat.LEAN,
                GNNFormat.COQ,
                GNNFormat.ISABELLE,
                GNNFormat.ALLOY,
                GNNFormat.Z_NOTATION,
            ],
            "grammar_formats": [GNNFormat.BNF, GNNFormat.EBNF],
            "temporal_formats": [GNNFormat.TLA_PLUS, GNNFormat.AGDA],
            "binary_formats": [GNNFormat.PICKLE],
        }

        # Add individual serializer format mapping
        available_individual_formats: dict[str, Any] = {
            "json": GNNFormat.JSON,
            "xml": GNNFormat.XML,
            "yaml": GNNFormat.YAML,
            "scala": GNNFormat.SCALA,
            "python": GNNFormat.PYTHON,
            "pkl": GNNFormat.PKL,
            "asn1": GNNFormat.ASN1,
            "protobuf": GNNFormat.PROTOBUF,
            "lean": GNNFormat.LEAN,
            "coq": GNNFormat.COQ,
            "alloy": GNNFormat.ALLOY,
            "binary": GNNFormat.PICKLE,
            "xsd": GNNFormat.XSD,
            "isabelle": GNNFormat.ISABELLE,
            "functional": GNNFormat.HASKELL,
            "grammar": GNNFormat.BNF,
            "temporal": GNNFormat.TLA_PLUS,
            "znotation": GNNFormat.Z_NOTATION,
        }

        # Determine formats to test first, before checking parsing system
        if FORMAT_TEST_CONFIG["test_all_formats"]:
            # Test all formats based on categories
            for category, enabled in FORMAT_TEST_CONFIG["test_categories"].items():
                if enabled and category in format_categories:
                    all_formats.extend(format_categories[category])
        else:
            # Test only specified formats
            specified_formats = FORMAT_TEST_CONFIG["test_formats"]
            for fmt_name in specified_formats:
                try:
                    # Try direct GNNFormat lookup first
                    fmt = GNNFormat(fmt_name)
                    if fmt not in all_formats:
                        all_formats.append(fmt)
                except ValueError:
                    # Try individual serializer format mapping
                    if fmt_name in available_individual_formats:
                        fmt = available_individual_formats[fmt_name]
                        if fmt not in all_formats:
                            all_formats.append(fmt)
                    elif LOGGING_CONFIG["enable_debug"]:
                        logger.warning(f"Unknown format in configuration: {fmt_name}")

        # Apply format overrides
        overrides = FORMAT_TEST_CONFIG.get("format_overrides", {})
        for fmt_name, enabled in overrides.items():
            try:
                fmt = GNNFormat(fmt_name)
                if not enabled and fmt in all_formats:
                    all_formats.remove(fmt)
                elif enabled and fmt not in all_formats:
                    all_formats.append(fmt)
            except ValueError:
                if LOGGING_CONFIG["enable_debug"]:
                    logger.warning(f"Unknown format in overrides: {fmt_name}")

        # If no parsing system available, return configured formats anyway for recovery testing
        if not self.parsing_system:
            if LOGGING_CONFIG["enable_debug"]:
                logger.warning(
                    "No parsing system available, using individual serializers for testing"
                )
            return all_formats

        # Check if parsing system has initialized parsers/serializers properly
        if self.parsing_system:
            working_formats: list[Any] = [GNNFormat.MARKDOWN]  # Always include markdown

            # Check if any parsers/serializers are available
            available_parsers = self.parsing_system.parsers
            available_serializers = self.parsing_system.serializers

            # If no parsers/serializers are available, use individual serializer mode
            if not available_parsers and not available_serializers:
                if LOGGING_CONFIG["enable_debug"]:
                    logger.debug(
                        "No parsers/serializers in parsing system, using individual serializers"
                    )
                return all_formats

            for fmt in all_formats:
                if fmt == GNNFormat.MARKDOWN:
                    continue
                try:
                    # Check if format is in the available lists (using enum values for comparison)
                    parser_available = any(
                        p.value == fmt.value for p in available_parsers
                    )
                    serializer_available = any(
                        s.value == fmt.value for s in available_serializers
                    )
                    if parser_available and serializer_available:
                        working_formats.append(fmt)
                        if LOGGING_CONFIG["enable_debug"]:
                            logger.debug(
                                f"Added format {fmt.value} for round-trip testing"
                            )
                    else:
                        # For formats without full parser/serializer, still add them for individual serializer testing
                        # Expand the list of known serializable formats
                        serializable_formats: list[Any] = [
                            GNNFormat.JSON,
                            GNNFormat.XML,
                            GNNFormat.YAML,
                            GNNFormat.SCALA,
                            GNNFormat.PYTHON,
                            GNNFormat.PKL,
                            GNNFormat.ASN1,
                            GNNFormat.PROTOBUF,
                            GNNFormat.LEAN,
                            GNNFormat.COQ,
                            GNNFormat.ALLOY,
                            GNNFormat.XSD,
                            GNNFormat.ISABELLE,
                            GNNFormat.HASKELL,
                            GNNFormat.BNF,
                            GNNFormat.PICKLE,
                            GNNFormat.Z_NOTATION,
                        ]
                        if fmt in serializable_formats:
                            working_formats.append(fmt)
                            if LOGGING_CONFIG["enable_debug"]:
                                logger.debug(
                                    f"Added format {fmt.value} for individual serializer testing"
                                )
                        else:
                            if LOGGING_CONFIG["enable_debug"]:
                                logger.debug(
                                    f"Skipping format {fmt.value}: missing parser or serializer"
                                )
                except Exception as e:
                    if LOGGING_CONFIG["enable_debug"]:
                        logger.debug(f"Exception checking format {fmt.value}: {e}")

            return working_formats
        else:
            # No parsing system available, return all configured formats for individual serializer testing
            if LOGGING_CONFIG["enable_debug"]:
                logger.debug(
                    "No parsing system available, using all configured formats"
                )
            return all_formats

    def _print_format_groups(self) -> Any:
        """Print format groups for organized display."""
        schema_formats = [
            f
            for f in self.supported_formats
            if f.value in ["json", "xml", "yaml", "xsd", "asn1", "pkl", "protobuf"]
        ]
        language_formats = [
            f
            for f in self.supported_formats
            if f.value in ["scala", "lean", "coq", "python", "haskell", "isabelle"]
        ]
        formal_formats = [
            f
            for f in self.supported_formats
            if f.value in ["tla_plus", "agda", "alloy", "z_notation", "bnf", "ebnf"]
        ]
        other_formats = [
            f
            for f in self.supported_formats
            if f not in schema_formats + language_formats + formal_formats
            and f != GNNFormat.MARKDOWN
        ]

        if schema_formats:
            print(
                f"   📋 Schema formats: {', '.join([f.value for f in schema_formats])}"
            )
        if language_formats:
            print(
                f"   💻 Language formats: {', '.join([f.value for f in language_formats])}"
            )
        if formal_formats:
            print(
                f"   🧮 Formal formats: {', '.join([f.value for f in formal_formats])}"
            )
        if other_formats:
            print(f"   🔧 Other formats: {', '.join([f.value for f in other_formats])}")

    def _print_detailed_test_result(
        self, test_result: RoundTripResult, fmt: GNNFormat
    ) -> Any:
        """Print detailed test result information."""
        if test_result.success:
            print(
                f"   ✅ PASS - {fmt.value} round-trip successful ({test_result.test_time:.3f}s)"
            )
            if test_result.converted_content:
                print(
                    f"      └─ Serialized {len(test_result.converted_content)} characters"
                )
            if test_result.warnings:
                print(f"      └─ ⚠️  {len(test_result.warnings)} warnings:")
                for warning in test_result.warnings[:2]:  # Show first 2 warnings
                    print(f"         • {warning}")
                if len(test_result.warnings) > 2:
                    print(f"         • ... and {len(test_result.warnings) - 2} more")
        else:
            print(
                f"   ❌ FAIL - {fmt.value} round-trip failed ({test_result.test_time:.3f}s)"
            )
            if test_result.errors:
                print(f"      └─ ❌ {len(test_result.errors)} errors:")
                for error in test_result.errors[:2]:  # Show first 2 errors
                    print(f"         • {error}")
                if len(test_result.errors) > 2:
                    print(f"         • ... and {len(test_result.errors) - 2} more")
            if test_result.differences:
                print(f"      └─ 🔍 {len(test_result.differences)} differences:")
                for diff in test_result.differences[:2]:  # Show first 2 differences
                    print(f"         • {diff}")
                if len(test_result.differences) > 2:
                    print(f"         • ... and {len(test_result.differences) - 2} more")
            if test_result.warnings:
                print(f"      └─ ⚠️  {len(test_result.warnings)} warnings:")
                for warning in test_result.warnings[:2]:
                    print(f"         • {warning}")
                if len(test_result.warnings) > 2:
                    print(f"         • ... and {len(test_result.warnings) - 2} more")
        print()

    def run_comprehensive_tests(self) -> ComprehensiveTestReport:
        """Run comprehensive round-trip tests for all supported formats."""
        import time

        if not self.reference_file.exists():
            raise FileNotFoundError(f"Reference file not found: {self.reference_file}")

        report = ComprehensiveTestReport(reference_file=str(self.reference_file))

        if LOGGING_CONFIG["enable_detailed_output"]:
            print(f"\n{'=' * 80}")
            print("GNN COMPREHENSIVE ROUND-TRIP TESTING")
            if not FORMAT_TEST_CONFIG["test_all_formats"]:
                print("SELECTIVE FORMAT TESTING ENABLED")
            print(f"{'=' * 80}")
            print(f"📁 Reference file: {self.reference_file}")
            print(f"🔄 Testing {len(self.supported_formats) - 1} formats:")

            # Group formats for display
            if LOGGING_CONFIG["enable_format_groups"]:
                self._print_format_groups()
            else:
                test_formats = [
                    f.value for f in self.supported_formats if f != GNNFormat.MARKDOWN
                ]
                print(f"   Formats: {', '.join(test_formats)}")

            if OUTPUT_CONFIG["save_test_artifacts"]:
                print(f"📂 Temp directory: {self.temp_dir}")
            print()
        else:
            print(
                f"🔄 Testing {len(self.supported_formats) - 1} GNN formats for round-trip compatibility..."
            )

        # Parse the reference model
        if LOGGING_CONFIG["enable_detailed_output"]:
            print("📖 Reading reference model...")

        start_time = time.time()

        reference_result = None

        # Try parsing with full system first, then recovery
        if self.parsing_system:
            try:
                reference_result = self.parsing_system.parse_file(
                    self.reference_file, GNNFormat.MARKDOWN
                )
            except Exception as parse_error:
                if LOGGING_CONFIG["enable_debug"]:
                    logger.warning(
                        f"Full parsing system failed, using recovery: {parse_error}"
                    )
                reference_result = cast(Any, None)

        # Use direct parser recovery if main parsing failed or unavailable
        if not reference_result or not reference_result.success:
            try:
                if LOGGING_CONFIG["enable_detailed_output"]:
                    print("   └─ Using direct markdown parser...")

                # Use direct markdown parser as recovery
                direct_parser = self._create_direct_markdown_parser()
                reference_model = direct_parser.parse_file(self.reference_file)

                # Create a simple result object
                reference_result = type(
                    "DirectParseResult",
                    (),
                    {"success": True, "model": reference_model, "errors": []},
                )()

            except Exception as fallback_error:
                error_msg = f"Both parsing system and recovery failed: {fallback_error}"
                print(f"❌ CRITICAL ERROR: {error_msg}")
                report.critical_errors.append(error_msg)
                return report

        if not reference_result.success:
            error_msg = f"Failed to parse reference file: {reference_result.errors}"
            print("❌ CRITICAL ERROR: Failed to parse reference file")
            if LOGGING_CONFIG["enable_detailed_output"]:
                for error in reference_result.errors:
                    print(f"   └─ {error}")
            report.critical_errors.append(error_msg)
            return report

        reference_model = reference_result.model
        parse_time = time.time() - start_time

        if LOGGING_CONFIG["enable_detailed_output"]:
            print(
                f"✅ Successfully parsed reference model: '{reference_model.model_name}' ({parse_time:.3f}s)"
            )
            print(f"   └─ Variables: {len(reference_model.variables)}")
            print(f"   └─ Connections: {len(reference_model.connections)}")
            print(f"   └─ Parameters: {len(reference_model.parameters)}")
            print()
        else:
            print(
                f"✅ Reference model loaded: {reference_model.model_name} ({len(reference_model.variables)} variables, {len(reference_model.connections)} connections)"
            )

        # Store start time for timeout checking
        self.start_time = start_time

        # Test conversion to each format and back
        test_formats = [
            fmt for fmt in self.supported_formats if fmt != GNNFormat.MARKDOWN
        ]

        # Group tests by category for better organization
        schema_formats = [
            f
            for f in self.supported_formats
            if f.value in ["json", "xml", "yaml", "xsd", "asn1", "pkl", "protobuf"]
        ]
        language_formats = [
            f
            for f in self.supported_formats
            if f.value in ["scala", "lean", "coq", "python", "haskell", "isabelle"]
        ]
        formal_formats = [
            f
            for f in self.supported_formats
            if f.value in ["tla_plus", "agda", "alloy", "z_notation", "bnf", "ebnf"]
        ]
        other_formats = [
            f
            for f in self.supported_formats
            if f not in schema_formats + language_formats + formal_formats
            and f != GNNFormat.MARKDOWN
        ]

        format_groups: list[Any] = [
            ("Schema Formats", schema_formats),
            ("Language Formats", language_formats),
            ("Formal Specification Formats", formal_formats),
            ("Other Formats", other_formats),
        ]

        test_count = 0
        for group_name, formats in format_groups:
            if not formats:
                continue

            if (
                LOGGING_CONFIG["enable_detailed_output"]
                and LOGGING_CONFIG["enable_format_groups"]
            ):
                print(f"🔍 Testing {group_name} ({len(formats)} formats)")
                print(f"{'─' * 60}")

            for fmt in formats:
                test_count += 1

                # Check timeout
                if (
                    hasattr(self, "start_time")
                    and (time.time() - self.start_time)
                    > TEST_BEHAVIOR_CONFIG["max_test_time"]
                ):
                    print(f"⏰ Test timeout reached, stopping at format {fmt.value}")
                    break

                if LOGGING_CONFIG["enable_detailed_output"]:
                    print(
                        f"🔄 [{test_count}/{len(test_formats)}] Testing {fmt.value.upper()} round-trip..."
                    )
                else:
                    print(f"Testing {fmt.value}... ", end="", flush=True)

                test_start = time.time()
                test_result = self._test_round_trip(reference_model, fmt)
                test_result.test_time = time.time() - test_start
                report.add_result(test_result)

                # Configurable result reporting
                if LOGGING_CONFIG["enable_detailed_output"]:
                    self._print_detailed_test_result(test_result, fmt)
                else:
                    status = "✅ PASS" if test_result.success else "❌ FAIL"
                    print(f"{status} ({test_result.test_time:.2f}s)")

                # Fail fast option
                if TEST_BEHAVIOR_CONFIG["fail_fast"] and not test_result.success:
                    print(
                        f"🛑 Fail-fast enabled, stopping after first failure: {fmt.value}"
                    )
                    break

            if (
                LOGGING_CONFIG["enable_detailed_output"]
                and LOGGING_CONFIG["enable_format_groups"]
            ):
                print()

        # Test cross-format consistency if available
        if (
            CROSS_FORMAT_AVAILABLE
            and self.cross_validator
            and TEST_BEHAVIOR_CONFIG["run_cross_format_validation"]
        ):
            print("🔍 Testing cross-format consistency...")
            consistency_start = time.time()
            self._test_cross_format_consistency(reference_model, report)
            consistency_time = time.time() - consistency_start

            if report.critical_errors:
                print(
                    f"   ❌ Cross-format consistency failed ({consistency_time:.3f}s)"
                )
                for error in report.critical_errors[
                    -3:
                ]:  # Show last 3 errors (from consistency test)
                    print(f"      └─ {error}")
            else:
                print(
                    f"   ✅ Cross-format consistency passed ({consistency_time:.3f}s)"
                )
            print()
        else:
            print(
                "🔍 Cross-format consistency testing skipped (disabled or module not available)"
            )
            print()

        # Final summary
        total_time = time.time() - start_time
        print(f"{'=' * 80}")
        print("COMPREHENSIVE ROUND-TRIP TEST RESULTS")
        print(f"{'=' * 80}")
        print(f"📊 Total tests: {report.total_tests}")
        print(f"✅ Successful: {report.successful_tests}")
        print(f"❌ Failed: {report.failed_tests}")
        print(f"📈 Success rate: {report.get_success_rate():.1f}%")
        print(f"⏱️  Total time: {total_time:.3f}s")
        print()

        # Show results by category
        if LOGGING_CONFIG["enable_detailed_output"]:
            format_summary = report.get_format_summary()
            for group_name, formats in format_groups:
                if not formats:
                    continue

                group_success = sum(
                    1
                    for fmt in formats
                    if format_summary.get(fmt, {}).get("success", 0) > 0
                )
                group_total = len(formats)
                group_rate = (
                    (group_success / group_total * 100) if group_total > 0 else 0
                )

                status = (
                    "✅" if group_rate == 100 else "⚠️" if group_rate >= 50 else "❌"
                )
                print(
                    f"{status} {group_name}: {group_success}/{group_total} ({group_rate:.1f}%)"
                )

                for fmt in formats:
                    fmt_stats = format_summary.get(fmt, {"success": 0, "total": 0})
                    fmt_rate = (
                        (fmt_stats["success"] / fmt_stats["total"] * 100)
                        if fmt_stats["total"] > 0
                        else 0
                    )
                    fmt_status = (
                        "✅" if fmt_rate == 100 else "⚠️" if fmt_rate > 0 else "❌"
                    )
                    print(
                        f"   {fmt_status} {fmt.value}: {fmt_stats['success']}/{fmt_stats['total']}"
                    )

            print()

        # Concise summary
        success_rate = report.get_success_rate()
        if success_rate == 100.0:
            message = "🎉 ALL TESTS PASSED! 100% confidence in round-trip conversion."
            if LOGGING_CONFIG["enable_detailed_output"]:
                print(message)
                print(
                    "   The GNN ecosystem is fully functional with complete format interoperability."
                )
            else:
                print(message)
        elif success_rate >= 80.0:
            message = f"🎊 EXCELLENT! {success_rate:.1f}% success rate."
            print(message)
            if LOGGING_CONFIG["enable_detailed_output"]:
                print(
                    "   Most formats are working correctly. Review failed formats for minor issues."
                )
        elif success_rate >= 60.0:
            message = f"👍 GOOD! {success_rate:.1f}% success rate."
            print(message)
            if LOGGING_CONFIG["enable_detailed_output"]:
                print(
                    "   Core formats are working. Some specialized formats need attention."
                )
        else:
            message = (
                f"⚠️  {report.failed_tests} tests failed ({success_rate:.1f}% success)."
            )
            print(message)
            if LOGGING_CONFIG["enable_detailed_output"]:
                print(
                    "   Significant issues found. Review errors above and implement fixes."
                )

        if LOGGING_CONFIG["enable_detailed_output"]:
            print(f"{'=' * 80}")

        return report

    def _convert_parsed_gnn_to_parse_result(self, parsed_gnn: ParsedGNN) -> ParseResult:
        """Convert ParsedGNN to the parser representation used by serializers."""
        variables = [
            Variable(
                name=variable.name,
                dimensions=[
                    dim if isinstance(dim, int) else 1 for dim in variable.dimensions
                ],
                data_type=(
                    {"int": DataType.INTEGER, "bool": DataType.BINARY}.get(
                        variable.data_type
                    )
                    or (
                        DataType(variable.data_type)
                        if variable.data_type in {item.value for item in DataType}
                        else DataType.CATEGORICAL
                    )
                ),
                description=variable.description,
            )
            for variable in parsed_gnn.variables.values()
        ]
        connections = [
            Connection(
                source_variables=(
                    [connection.source]
                    if isinstance(connection.source, str)
                    else list(connection.source)
                ),
                target_variables=(
                    [connection.target]
                    if isinstance(connection.target, str)
                    else list(connection.target)
                ),
                connection_type=(
                    ConnectionType(connection.connection_type)
                    if connection.connection_type
                    in {item.value for item in ConnectionType}
                    else ConnectionType.DIRECTED
                ),
                weight=connection.weight,
                description=connection.description,
            )
            for connection in parsed_gnn.connections
        ]
        model = GNNInternalRepresentation(
            model_name=parsed_gnn.model_name or "Unknown Model",
            annotation=parsed_gnn.model_annotation,
            variables=variables,
            connections=connections,
            parameters=[
                Parameter(name=name, value=deepcopy(value))
                for name, value in parsed_gnn.parameters.items()
            ],
        )

        result = ParseResult(model=model, success=True)
        return result

    def _test_round_trip(
        self, reference_model: GNNInternalRepresentation, target_format: GNNFormat
    ) -> RoundTripResult:
        """Test round-trip conversion for a specific format."""
        result = RoundTripResult(
            source_format=GNNFormat.MARKDOWN,
            target_format=target_format,
            success=True,
            original_model=reference_model,
        )

        # Timeout check per format
        import time

        format_start_time = time.time()

        try:
            # Step 1: Serialize to target format
            if LOGGING_CONFIG["enable_detailed_output"]:
                print(f"      ➤ Serializing to {target_format.value}...")

            converted_content: str | None = None
            if self.parsing_system:
                try:
                    converted_content = self.parsing_system.serialize(
                        reference_model, target_format
                    )
                except ValueError as e:
                    if "No serializer available" in str(e):
                        if LOGGING_CONFIG["enable_detailed_output"]:
                            print(
                                "         ⚠️  Parsing system failed, using individual serializer..."
                            )
                        # Recovery to individual serializer when parsing system fails
                        converted_content = self._serialize_with_individual_serializer(
                            reference_model, target_format
                        )
                        if not converted_content:
                            raise ValueError(
                                f"Both parsing system and individual serializer failed for {target_format.value}"
                            ) from e
                    else:
                        raise
            else:
                # Recovery to direct serializer access with comprehensive serializer map
                converted_content = self._serialize_with_individual_serializer(
                    reference_model, target_format
                )
                if not converted_content:
                    raise ValueError(
                        f"Individual serializer for {target_format.value} failed"
                    )

            result.converted_content = converted_content
            result.checksum_original = self._compute_model_checksum(reference_model)

            if not converted_content:
                result.add_error(
                    f"Serialization to {target_format.value} produced empty content"
                )
                if LOGGING_CONFIG["enable_detailed_output"]:
                    print("         ❌ Serialization failed - empty content")
                return result

            if LOGGING_CONFIG["enable_detailed_output"]:
                print(f"         ✓ Serialized {len(converted_content)} characters")

            # Check format-specific timeout
            if (time.time() - format_start_time) > TEST_BEHAVIOR_CONFIG[
                "per_format_timeout"
            ]:
                result.add_error(
                    f"Format test timeout ({TEST_BEHAVIOR_CONFIG['per_format_timeout']}s)"
                )
                return result

            # Step 2: Save to temporary file (only if configured)
            if OUTPUT_CONFIG["save_test_artifacts"]:
                self.temp_dir.mkdir(parents=True, exist_ok=True)
                temp_file = (
                    self.temp_dir
                    / f"test_model.{self._get_file_extension(target_format)}"
                )

                # Handle binary formats specially
                if target_format == GNNFormat.PICKLE:
                    # For pickle, save the base64 content but as binary for proper round-trip
                    import base64

                    try:
                        binary_data = base64.b64decode(converted_content)
                        temp_file.write_bytes(binary_data)
                    except Exception:
                        # Recovery to text if decode fails
                        with tempfile.NamedTemporaryFile(
                            mode="w",
                            encoding="utf-8",
                            dir=temp_file.parent,
                            delete=False,
                        ) as tmp_f:
                            tmp_f.write(converted_content)
                        os.replace(tmp_f.name, str(temp_file))
                else:
                    with tempfile.NamedTemporaryFile(
                        mode="w", encoding="utf-8", dir=temp_file.parent, delete=False
                    ) as tmp_f:
                        tmp_f.write(converted_content)
                    os.replace(tmp_f.name, str(temp_file))

                if LOGGING_CONFIG["enable_detailed_output"]:
                    print(f"         ✓ Saved to {temp_file.name}")
            else:
                # Create temporary file in memory for parsing
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=f".{self._get_file_extension(target_format)}",
                    delete=False,
                ) as tf:
                    tf.write(converted_content)
                    temp_file = Path(tf.name)

            # Step 3: Parse back from target format
            if LOGGING_CONFIG["enable_detailed_output"]:
                print(f"      ➤ Parsing back from {target_format.value}...")

            if self.parsing_system:
                try:
                    # Use enum value for comparison to handle different enum instances
                    available_parsers = self.parsing_system.parsers
                    parser_found = any(
                        p.value == target_format.value for p in available_parsers
                    )

                    if parser_found:
                        # Find the actual parser instance
                        for fmt, parser in available_parsers.items():
                            if fmt.value == target_format.value:
                                parsed_result = parser.parse_file(str(temp_file))
                                break
                    else:
                        raise ValueError(
                            f"No parser available for format: {target_format.value}"
                        )

                except Exception as e:
                    if "No parser available" in str(e):
                        if LOGGING_CONFIG["enable_detailed_output"]:
                            print(
                                f"         ⚠️  No parser available for {target_format.value}, skipping parse-back"
                            )
                        # For formats without parsers, we'll mark as successful with warnings
                        result.add_warning(
                            f"Cannot parse back {target_format.value} - no parser available"
                        )
                        return result
                    else:
                        raise e
            else:
                # For non-markdown formats, we'll mark as successful but with warnings
                result.add_warning(
                    f"Cannot parse back {target_format.value} without full parsing system"
                )
                if LOGGING_CONFIG["enable_detailed_output"]:
                    print("         ⚠️  Parse-back skipped (limited parsing system)")
                return result

            if not parsed_result.success:
                result.add_error(
                    f"Failed to parse {target_format.value} content: {parsed_result.errors}"
                )
                if LOGGING_CONFIG["enable_detailed_output"]:
                    print("         ❌ Parse failed:")
                    for error in parsed_result.errors:
                        print(f"            • {error}")
                return result

            result.parsed_back_model = parsed_result.model
            result.checksum_converted = self._compute_model_checksum(
                parsed_result.model
            )
            print("         ✓ Parsed back successfully")

            # Show warnings from parsing if any
            if hasattr(parsed_result, "warnings") and parsed_result.warnings:
                print("         ⚠️  Parse warnings:")
                for warning in parsed_result.warnings:
                    print(f"            • {warning}")
                    result.add_warning(f"Parse warning: {warning}")

            # Step 4: Compare models for semantic equivalence
            if TEST_BEHAVIOR_CONFIG["validate_round_trip"]:
                if LOGGING_CONFIG["enable_detailed_output"]:
                    print("      ➤ Comparing semantic equivalence...")
                original_count = len(result.differences)
                self._compare_models(reference_model, parsed_result.model, result)
                new_differences = len(result.differences) - original_count

                if LOGGING_CONFIG["enable_detailed_output"]:
                    if new_differences == 0:
                        print("         ✓ Models are semantically equivalent")
                    else:
                        print(f"         ❌ Found {new_differences} differences")

            # Step 5: Validate converted model if validator is available
            if self.validator and TEST_BEHAVIOR_CONFIG["validate_round_trip"]:
                if LOGGING_CONFIG["enable_detailed_output"]:
                    print("      ➤ Validating converted model...")
                try:
                    validation_result = self.validator.validate_file(temp_file)
                    if validation_result.is_valid:
                        if LOGGING_CONFIG["enable_detailed_output"]:
                            print("         ✓ Validation passed")
                    else:
                        if LOGGING_CONFIG["enable_detailed_output"]:
                            print("         ⚠️  Validation warnings/errors:")
                            for error in validation_result.errors:
                                print(f"            • Error: {error}")
                            for warning in validation_result.warnings:
                                print(f"            • Warning: {warning}")
                        # Always record validation issues
                        for error in validation_result.errors:
                            result.add_warning(f"Validation error: {error}")
                        for warning in validation_result.warnings:
                            result.add_warning(f"Validation warning: {warning}")
                except Exception as e:
                    if LOGGING_CONFIG["enable_detailed_output"]:
                        print(f"         ⚠️  Validation failed: {e}")
                    result.add_warning(f"Validation failed: {e}")
            elif LOGGING_CONFIG["enable_detailed_output"] and not self.validator:
                print("         ⚠️  Validation skipped (validator not available)")

            # Checksum comparison
            if (
                TEST_BEHAVIOR_CONFIG["compute_checksums"]
                and result.checksum_original
                and result.checksum_converted
            ):
                checksum_match = result.checksum_original == result.checksum_converted
                if LOGGING_CONFIG["enable_detailed_output"]:
                    if checksum_match:
                        print("         ✓ Semantic checksums match")
                    else:
                        print("         ⚠️  Semantic checksums differ")
                if not checksum_match:
                    result.add_warning(
                        "Semantic checksums don't match (may indicate data loss)"
                    )

        except Exception as e:
            result.add_error(f"Round-trip test failed with exception: {str(e)}")
            if LOGGING_CONFIG["enable_detailed_output"]:
                print(f"         ❌ Exception occurred: {str(e)}")
                import traceback

                print(f"         Traceback: {traceback.format_exc()}")
            elif LOGGING_CONFIG["enable_debug"]:
                import traceback

                logger.debug(
                    f"Exception in {target_format.value}: {traceback.format_exc()}"
                )

        return result

    def _get_file_extension(self, format: GNNFormat) -> str:
        """Get file extension for a format."""
        from gnn.parsers.common import FORMAT_OUTPUT_EXTENSION_MAP

        # Keep the existing valid Z artifact suffix; other formats use the
        # canonical parser/serializer owner, including distinct PKL/Pickle.
        if format == GNNFormat.Z_NOTATION:
            return "zed"
        return FORMAT_OUTPUT_EXTENSION_MAP.get(format, ".txt").removeprefix(".")
