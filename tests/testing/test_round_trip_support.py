"""Unit tests for the ``gnn.testing`` round-trip support modules.

The support modules (direct markdown parser, result dataclasses, report
mixin, comparison mixin, model checksum) ship inside ``src/gnn/testing``
where pytest's ``testpaths`` never collects them, so they measured 0% in
the verification harness despite their bundled ``test_round_trip.py``
files. They are exercised here directly against synthetic GNN documents.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from gnn.parsers.common import GNNFormat, GNNInternalRepresentation
from gnn.testing.round_trip_comparison import RoundTripComparisonMixin
from gnn.testing.round_trip_markdown_parser import _DirectMarkdownParser
from gnn.testing.round_trip_report import RoundTripReportMixin
from gnn.testing.round_trip_results import (
    ComprehensiveTestReport,
    RoundTripResult,
)

SAMPLE_GNN = """## ModelName
TestModel

## ModelAnnotation
A test active inference model.

## StateSpaceBlock
s_f[2,2] # hidden state factor
A[2,2,2,type=float] # likelihood matrix
o[2] # observations

## Connections
s_f > o

## InitialParameterization
A = uniform

## Time
Dynamic
Horizon=5

## ActInfOntologyAnnotation
s_f = hidden_state_factor
"""


def _parse(content: str) -> GNNInternalRepresentation:
    return _DirectMarkdownParser().parse_content(content)


def _vars_by_name(
    model: GNNInternalRepresentation,
) -> dict[str, object]:
    return {var.name: var for var in model.variables}


class TestDirectMarkdownParser:
    def test_empty_content_yields_unknown_model(self) -> None:
        model = _parse("")

        assert model.model_name == "Unknown Model"
        assert model.variables == []
        assert model.connections == []
        assert model.parameters == []
        assert model.ontology_mappings == []
        assert model.time_specification is None

    def test_full_document_sections_and_metadata(self) -> None:
        model = _parse(SAMPLE_GNN)

        assert model.model_name == "TestModel"
        assert model.annotation == "A test active inference model."
        assert set(model.raw_sections) == {
            "ModelName",
            "ModelAnnotation",
            "StateSpaceBlock",
            "Connections",
            "InitialParameterization",
            "Time",
            "ActInfOntologyAnnotation",
        }

    def test_variable_roles_and_type_mapping(self) -> None:
        model = _parse(SAMPLE_GNN)
        variables = _vars_by_name(model)

        assert set(variables) == {"s_f", "A", "o"}
        assert variables["s_f"].dimensions == [2, 2]
        assert variables["s_f"].var_type.value == "hidden_state"
        assert variables["A"].var_type.value == "likelihood_matrix"
        assert variables["A"].data_type.value == "float"
        assert variables["o"].var_type.value == "observation"

    def test_connection_parameter_time_and_ontology(self) -> None:
        model = _parse(SAMPLE_GNN)

        assert len(model.connections) == 1
        conn = model.connections[0]
        assert conn.source_variables == ["s_f"]
        assert conn.target_variables == ["o"]
        assert conn.connection_type.value == "directed"

        assert [(p.name, p.value) for p in model.parameters] == [("A", "uniform")]

        assert model.time_specification.time_type == "Dynamic"
        assert model.time_specification.horizon == "5"

        ontology = [(m.variable_name, m.ontology_term) for m in model.ontology_mappings]
        assert ontology == [("s_f", "hidden_state_factor")]


class TestRoundTripResults:
    def test_add_error_marks_result_unsuccessful(self) -> None:
        result = RoundTripResult(source_format="markdown", target_format="json")

        assert result.success is True
        result.add_error("checksum drifted")
        result.add_warning("cosmetic")
        result.add_difference("name casing")

        assert result.success is False
        assert result.errors == ["checksum drifted"]
        assert result.warnings == ["cosmetic"]
        assert result.differences == ["name casing"]

    def test_report_aggregates_and_format_summary(self) -> None:
        report = ComprehensiveTestReport(reference_file="ref.md")
        ok = RoundTripResult(
            source_format=GNNFormat.MARKDOWN, target_format=GNNFormat.JSON
        )
        bad = RoundTripResult(source_format=GNNFormat.JSON, target_format=GNNFormat.PKL)
        bad.add_error("round trip failed")
        report.add_result(ok)
        report.add_result(bad)

        assert report.total_tests == 2
        assert report.successful_tests == 1
        assert report.failed_tests == 1
        assert report.get_success_rate() == 50.0
        assert report.get_format_summary() == {
            GNNFormat.JSON: {"success": 1, "total": 1},
            GNNFormat.PKL: {"success": 0, "total": 1},
        }

    def test_empty_report_success_rate_is_zero(self) -> None:
        report = ComprehensiveTestReport()

        assert report.total_tests == 0
        assert report.get_success_rate() == 0.0


class TestRoundTripReportMixin:
    def _report(self, *, with_critical: bool = False) -> ComprehensiveTestReport:
        report = ComprehensiveTestReport(reference_file="ref.md")
        ok = RoundTripResult(
            source_format=GNNFormat.MARKDOWN, target_format=GNNFormat.JSON
        )
        bad = RoundTripResult(source_format=GNNFormat.JSON, target_format=GNNFormat.PKL)
        bad.add_error("round trip failed")
        report.add_result(ok)
        report.add_result(bad)
        if with_critical:
            report.critical_errors.append("parser exploded")
        return report

    def test_report_text_contains_summary_and_sections(self) -> None:
        text = RoundTripReportMixin().generate_report(self._report())

        assert "# GNN Round-Trip Testing Report" in text
        assert "**Reference File:** `ref.md`" in text
        assert "**Total Tests:** 2" in text
        assert "**Failed:** 1" in text
        assert "## Format Summary" in text
        assert "## Detailed Results" in text
        assert "✅ PASS" in text
        assert "❌ FAIL" in text
        assert "## Recommendations" in text

    def test_report_includes_critical_issues_when_present(self) -> None:
        text = RoundTripReportMixin().generate_report(self._report(with_critical=True))

        assert "## Critical Issues" in text
        assert "parser exploded" in text

    def test_report_writes_output_file(self, tmp_path: Path) -> None:
        output_file = tmp_path / "report.md"

        text = RoundTripReportMixin().generate_report(
            self._report(), output_file=output_file
        )

        assert output_file.read_text(encoding="utf-8") == text


class _ComparisonHost(RoundTripComparisonMixin):
    def __init__(self) -> None:
        self.logger = logging.getLogger("test_round_trip_support")


class TestRoundTripComparisonMixin:
    def test_identical_models_have_no_differences(self) -> None:
        host = _ComparisonHost()
        result = RoundTripResult()

        host._compare_models(_parse(SAMPLE_GNN), _parse(SAMPLE_GNN), result)

        assert result.success is True
        assert result.differences == []

    def test_name_mismatch_is_recorded(self) -> None:
        host = _ComparisonHost()
        result = RoundTripResult()
        renamed = _parse(SAMPLE_GNN.replace("TestModel", "OtherModel"))

        host._compare_models(_parse(SAMPLE_GNN), renamed, result)

        assert any("Model name mismatch" in d for d in result.differences)

    def test_missing_variable_is_recorded(self) -> None:
        host = _ComparisonHost()
        result = RoundTripResult()
        dropped = _parse(SAMPLE_GNN.replace("o[2] # observations\n", ""))

        host._compare_models(_parse(SAMPLE_GNN), dropped, result)

        assert any("Variable missing in converted: o" in d for d in result.differences)

    def test_data_type_mismatch_is_recorded(self) -> None:
        host = _ComparisonHost()
        result = RoundTripResult()
        retyped = _parse(SAMPLE_GNN.replace("type=float", "type=int"))

        host._compare_models(_parse(SAMPLE_GNN), retyped, result)

        assert any("data type mismatch" in d for d in result.differences)
        assert any("float" in d and "integer" in d for d in result.differences)

    def test_checksum_is_stable_and_sensitive(self) -> None:
        host = _ComparisonHost()
        # The checksum covers name/variables/connections/parameters only, so
        # the changed model must alter a checksummed field (dimensions).
        changed = _parse(SAMPLE_GNN.replace("o[2]", "o[3]"))

        checksum_a = host._compute_model_checksum(_parse(SAMPLE_GNN))
        checksum_b = host._compute_model_checksum(_parse(SAMPLE_GNN))
        checksum_c = host._compute_model_checksum(changed)

        assert checksum_a == checksum_b
        assert checksum_a != checksum_c


@pytest.mark.parametrize(
    "mixin",
    [RoundTripComparisonMixin, RoundTripReportMixin],
    ids=["comparison", "report"],
)
def test_mixins_do_not_require_constructor_state(mixin: type) -> None:
    """The mixins are mixed into ``GNNRoundTripTester``; instantiating them
    bare must stay possible (no ``__init__`` state required)."""

    instance = mixin()
    assert isinstance(instance, mixin)
