#!/usr/bin/env python3
"""
Unit tests for gnn.types (MED-T4).

Covers the observable public behavior of the shared dataclass types in
gnn/types/definitions.py — construction, default handling, mutation
helpers (add_error/add_warning/add_round_trip_result), accounting
(success rates, report tallies, format summaries), error formatting,
and the package facade re-export of GNNFormat.
"""

from typing import Any

import pytest

from gnn.types import (
    ComprehensiveTestReport,
    GNNConnection,
    GNNFormat,
    GNNInternalRepresentation,
    GNNSyntaxError,
    GNNVariable,
    ParsedGNN,
    ParseResult,
    RoundTripResult,
    ValidationLevel,
    ValidationResult,
)


def _round_trip(
    source: GNNFormat = GNNFormat.MARKDOWN,
    target: GNNFormat = GNNFormat.JSON,
    success: bool = True,
) -> RoundTripResult:
    return RoundTripResult(source_format=source, target_format=target, success=success)


class TestValidationLevel:
    """ValidationLevel is the canonical level vocabulary."""

    def test_member_values(self) -> Any:
        assert ValidationLevel.BASIC.value == "basic"
        assert ValidationLevel.STANDARD.value == "standard"
        assert ValidationLevel.STRICT.value == "strict"
        assert ValidationLevel.RESEARCH.value == "research"
        assert ValidationLevel.ROUND_TRIP.value == "round_trip"

    def test_lookup_by_value_round_trip(self) -> Any:
        assert ValidationLevel("strict") is ValidationLevel.STRICT


class TestGNNSyntaxError:
    """Error type carries position/format context and formats messages."""

    def test_stores_context(self) -> Any:
        err = GNNSyntaxError("bad token", line=3, column=7, format_context="json")
        assert str(err) == "bad token"
        assert err.line == 3
        assert err.column == 7
        assert err.format_context == "json"

    def test_format_message_full(self) -> Any:
        err = GNNSyntaxError("bad token", line=3, column=7, format_context="json")
        assert err.format_message() == "bad token (line 3, column 7) in json format"

    def test_format_message_line_only(self) -> Any:
        err = GNNSyntaxError("bad token", line=3)
        assert err.format_message() == "bad token (line 3)"

    def test_format_message_bare(self) -> Any:
        assert GNNSyntaxError("bad token").format_message() == "bad token"


class TestValidationResult:
    """ValidationResult aggregates round-trip outcomes and rates."""

    def test_defaults(self) -> Any:
        result = ValidationResult(is_valid=True)
        assert result.errors == []
        assert result.warnings == []
        assert result.suggestions == []
        assert result.metadata == {}
        assert result.validation_level is ValidationLevel.STANDARD
        assert result.format_tested is None
        assert result.round_trip_results == []
        assert result.cross_format_consistent is None
        assert result.semantic_checksum is None

    def test_add_round_trip_result_success_is_silent(self) -> Any:
        result = ValidationResult(is_valid=True)
        result.add_round_trip_result(_round_trip(success=True))
        assert result.errors == []
        assert result.warnings == []

    def test_add_round_trip_result_failure_propagates(self) -> Any:
        failed = _round_trip(success=False)
        failed.errors.append("conversion failed")
        failed.warnings.append("lossy")
        result = ValidationResult(is_valid=True)
        result.add_round_trip_result(failed)
        assert result.errors == ["conversion failed"]
        assert result.warnings == ["lossy"]

    def test_round_trip_success_rate(self) -> Any:
        result = ValidationResult(is_valid=True)
        assert result.get_round_trip_success_rate() == 0.0
        result.add_round_trip_result(_round_trip(success=True))
        result.add_round_trip_result(_round_trip(success=False))
        result.add_round_trip_result(_round_trip(success=True))
        assert result.get_round_trip_success_rate() == pytest.approx(2 / 3 * 100)


class TestDataclassDefaults:
    """Default factories must not share mutable state across instances."""

    def test_gnn_variable_defaults(self) -> Any:
        var = GNNVariable(name="s_t", dimensions=[3, 1], data_type="float")
        assert var.description is None
        assert var.constraints is None
        assert var.line_number is None
        assert var.ontology_mapping is None
        assert var.format_specific_metadata == {}
        var.format_specific_metadata["fmt"] = "md"
        assert (
            GNNVariable(
                name="o_t", dimensions=[3], data_type="int"
            ).format_specific_metadata
            == {}
        )

    def test_gnn_connection_defaults(self) -> Any:
        conn = GNNConnection(
            source="s_t", target="o_t", connection_type="directed", symbol=">"
        )
        assert conn.description is None
        assert conn.line_number is None
        assert conn.weight is None
        assert conn.metadata == {}

    def test_parsed_gnn_requires_model_fields(self) -> Any:
        parsed = ParsedGNN(
            gnn_section="ActInfPOMDP",
            version="GNN v1",
            model_name="Demo",
            model_annotation="annotation",
            variables={},
            connections=[],
            parameters={},
            equations=[],
            time_config={},
            ontology_mappings={},
            model_parameters={},
            footer="footer",
        )
        assert parsed.signature is None
        assert parsed.source_format is None
        assert parsed.round_trip_verified is False


class TestParseResult:
    """ParseResult has one declaration behind every public import path.

    Two constructor patterns are in use: ``ParseResult()`` without a model
    (typed consumers) and ``ParseResult(model=...)`` (parser consumers). Each
    must keep the default documented by the surviving single shape, and no
    behavior may depend on which public surface supplied the class.
    """

    def test_single_declaration_across_public_surfaces(self) -> Any:
        import gnn.parsers.common as parsers_common
        import gnn.types.definitions as types_definitions

        assert ParseResult is parsers_common.ParseResult
        assert ParseResult is types_definitions.ParseResult

    def test_bare_construction_keeps_typed_defaults(self) -> Any:
        result = ParseResult()
        assert result.model is None
        assert result.success is False
        assert result.errors == []
        assert result.warnings == []
        assert result.parse_time is None
        assert result.source_file is None
        assert result.validation_result is None
        assert result.metadata == {}

    def test_model_construction_reports_success(self) -> Any:
        model = GNNInternalRepresentation(model_name="Demo")
        result = ParseResult(model=model)
        assert result.model is model
        assert result.success is True
        assert result.errors == []
        assert result.warnings == []

    def test_explicit_success_wins_over_model_presence(self) -> Any:
        model = GNNInternalRepresentation(model_name="Demo")
        assert ParseResult(model=model, success=False).success is False
        assert ParseResult(success=True).success is True

    def test_add_error_flips_success(self) -> Any:
        result = ParseResult(success=True)
        result.add_error("oops")
        assert result.errors == ["oops"]
        assert result.success is False

    def test_add_error_on_model_construction(self) -> Any:
        result = ParseResult(model=GNNInternalRepresentation(model_name="Demo"))
        result.add_error("oops")
        assert result.has_errors() is True
        assert result.success is False

    def test_add_warning_keeps_success(self) -> Any:
        result = ParseResult(success=True)
        result.add_warning("lossy detail")
        assert result.warnings == ["lossy detail"]
        assert result.has_warnings() is True
        assert result.success is True


class TestRoundTripResult:
    """RoundTripResult tracks conversion fidelity per direction."""

    def test_defaults(self) -> Any:
        result = _round_trip()
        assert result.converted_content is None
        assert result.parsed_back_model is None
        assert result.differences == []
        assert result.warnings == []
        assert result.errors == []
        assert result.test_time == 0.0
        assert result.success is True

    def test_add_difference_flips_success(self) -> Any:
        result = _round_trip()
        result.add_difference("missing equation")
        assert result.differences == ["missing equation"]
        assert result.success is False

    def test_add_error_flips_success(self) -> Any:
        result = _round_trip()
        result.add_error("decode failure")
        assert result.errors == ["decode failure"]
        assert result.success is False

    def test_add_warning_keeps_success(self) -> Any:
        result = _round_trip()
        result.add_warning("rounding")
        assert result.warnings == ["rounding"]
        assert result.success is True


class TestComprehensiveTestReport:
    """ComprehensiveTestReport tallies results and summarizes per format."""

    def test_empty_report(self) -> Any:
        report = ComprehensiveTestReport(reference_file="model.md")
        assert report.total_tests == 0
        assert report.get_success_rate() == 0.0
        assert report.get_format_summary() == {}

    def test_add_result_tallies_and_matrix(self) -> Any:
        report = ComprehensiveTestReport(reference_file="model.md")
        report.add_result(_round_trip(source=GNNFormat.MARKDOWN, target=GNNFormat.JSON))
        failed = _round_trip(
            source=GNNFormat.MARKDOWN, target=GNNFormat.XML, success=False
        )
        failed.differences.append("drift")
        failed.errors.append("boom")
        report.add_result(failed)
        assert report.total_tests == 2
        assert report.successful_tests == 1
        assert report.failed_tests == 1
        assert report.get_success_rate() == pytest.approx(50.0)
        assert report.format_matrix == {
            (GNNFormat.MARKDOWN, GNNFormat.JSON): True,
            (GNNFormat.MARKDOWN, GNNFormat.XML): False,
        }
        assert report.semantic_differences == ["drift"]
        assert report.critical_errors == ["boom"]

    def test_format_summary_groups_by_target(self) -> Any:
        report = ComprehensiveTestReport(reference_file="model.md")
        report.add_result(_round_trip(source=GNNFormat.MARKDOWN, target=GNNFormat.JSON))
        report.add_result(_round_trip(source=GNNFormat.XML, target=GNNFormat.JSON))
        report.add_result(
            _round_trip(source=GNNFormat.MARKDOWN, target=GNNFormat.YAML, success=False)
        )
        assert report.get_format_summary() == {
            GNNFormat.JSON: {"success": 2, "failure": 0, "total": 2},
            GNNFormat.YAML: {"success": 0, "failure": 1, "total": 1},
        }


class TestTypesFacade:
    """gnn.types re-exports the authoritative GNNFormat from parsers."""

    def test_gnn_format_reexport_identity(self) -> Any:
        from gnn.parsers.common import GNNFormat as AuthoritativeGNNFormat

        assert GNNFormat is AuthoritativeGNNFormat

    def test_gnn_format_members(self) -> Any:
        assert GNNFormat.MARKDOWN.value == "markdown"
        assert GNNFormat.JSON.value == "json"
        assert GNNFormat.XML.value == "xml"
        assert GNNFormat.YAML.value == "yaml"
