"""Unit tests for ``gnn.testing.round_trip_comparison``.

Covers the ``RoundTripComparisonMixin`` paths not exercised by
``test_round_trip_support.py``: every per-section difference recorder,
cross-format consistency aggregation, and the plain-string attribute
fallbacks. Comparison inputs come from the real direct markdown parser on
tiny inline GNN documents, plus minimal synthetic section objects for
states the parser cannot produce.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gnn.parsers.common import GNNFormat, GNNInternalRepresentation
from gnn.testing.round_trip_comparison import RoundTripComparisonMixin
from gnn.testing.round_trip_markdown_parser import _DirectMarkdownParser
from gnn.testing.round_trip_results import ComprehensiveTestReport, RoundTripResult

COMPARE_GNN = """## ModelName
CompareModel

## ModelAnnotation
Original annotation.

## StateSpaceBlock
s[2,2] # state
o[2] # observation

## Connections
s > o

## InitialParameterization
s = 0.1

## Time
Dynamic

## ActInfOntologyAnnotation
s = hidden_state
"""


def _parse(content: str) -> GNNInternalRepresentation:
    return _DirectMarkdownParser().parse_content(content)


class _Host(RoundTripComparisonMixin):
    """Bare mixin host; the comparison methods touch no instance state."""


def _result() -> RoundTripResult:
    return RoundTripResult()


def _var(
    name: str,
    var_type: object = "hidden_state",
    data_type: object = "float",
    dimensions: tuple[int, ...] = (2,),
) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        var_type=var_type,
        data_type=data_type,
        dimensions=list(dimensions),
    )


def _param(name: str, value: object) -> SimpleNamespace:
    return SimpleNamespace(name=name, value=value)


class TestCompareModels:
    def test_annotation_mismatch_is_recorded(self) -> None:
        converted = _parse(COMPARE_GNN.replace("Original annotation.", "Changed."))
        result = _result()

        _Host()._compare_models(_parse(COMPARE_GNN), converted, result)

        assert result.differences == ["Annotation mismatch"]

    def test_extra_variable_in_converted_is_recorded(self) -> None:
        without_o = _parse(COMPARE_GNN.replace("o[2] # observation\n", ""))
        result = _result()

        _Host()._compare_models(without_o, _parse(COMPARE_GNN), result)

        assert "Extra variable in converted: o" in result.differences


class TestCompareVariables:
    def test_var_type_mismatch_via_enum_value(self) -> None:
        result = _result()
        conv_var = _var("s", var_type=SimpleNamespace(value="observation"))

        _Host()._compare_variables([_var("s")], [conv_var], result)

        assert result.differences == [
            "Variable s type mismatch: hidden_state vs observation"
        ]

    def test_plain_string_var_type_uses_str_fallback(self) -> None:
        result = _result()

        _Host()._compare_variables(
            [_var("s", var_type="hidden_state")], [_var("s", var_type="action")], result
        )

        assert result.differences == [
            "Variable s type mismatch: hidden_state vs action"
        ]

    def test_dimensions_mismatch_is_recorded(self) -> None:
        result = _result()

        _Host()._compare_variables([_var("s")], [_var("s", dimensions=(2, 3))], result)

        assert result.differences == ["Variable s dimensions mismatch: [2] vs [2, 3]"]


class TestCompareConnections:
    def test_count_mismatch_is_recorded(self) -> None:
        model = _parse(COMPARE_GNN)
        doubled = _parse(COMPARE_GNN)
        doubled.connections.append(model.connections[0])
        result = _result()

        _Host()._compare_connections(model.connections, doubled.connections, result)

        assert result.differences == ["Connection count mismatch: 1 vs 2"]

    def test_missing_connection_content_is_recorded(self) -> None:
        result = _result()
        orig_conn = _parse(COMPARE_GNN).connections[0]

        _Host()._compare_connections([orig_conn], [], result)

        # The recorder reports both the arity drift and the missing content.
        assert result.differences == [
            "Connection count mismatch: 1 vs 0",
            "Missing connection: s--directed-->o",
        ]

    def test_extra_connection_content_is_recorded(self) -> None:
        result = _result()
        conv_conn = _parse(COMPARE_GNN).connections[0]

        _Host()._compare_connections([], [conv_conn], result)

        assert result.differences == [
            "Connection count mismatch: 0 vs 1",
            "Extra connection: s--directed-->o",
        ]


class TestCompareParameters:
    def test_missing_and_extra_parameters_are_recorded(self) -> None:
        result = _result()

        _Host()._compare_parameters(
            [_param("s", 0.1), _param("dropped", 1)],
            [_param("s", 0.1), _param("invented", 2)],
            result,
        )

        assert result.differences == [
            "Missing parameter: dropped",
            "Extra parameter: invented",
        ]

    def test_value_mismatch_is_recorded(self) -> None:
        result = _result()

        _Host()._compare_parameters([_param("s", 0.1)], [_param("s", 0.9)], result)

        assert result.differences == ["Parameter s value mismatch: 0.1 vs 0.9"]


class TestCompareEquationsAndTime:
    def test_equation_count_mismatch_is_recorded(self) -> None:
        result = _result()

        _Host()._compare_equations(["x = y"], [], result)

        assert result.differences == ["Equation count mismatch: 1 vs 0"]

    def test_time_presence_mismatch_is_recorded(self) -> None:
        result = _result()

        _Host()._compare_time_specification(
            _parse(COMPARE_GNN).time_specification,
            _parse(COMPARE_GNN.replace("## Time\nDynamic\n", "")).time_specification,
            result,
        )

        assert result.differences == ["Time specification presence mismatch"]

    def test_time_type_mismatch_is_recorded(self) -> None:
        result = _result()
        orig = _parse(COMPARE_GNN)
        conv = _parse(COMPARE_GNN.replace("## Time\nDynamic\n", "## Time\nDiscrete\n"))

        _Host()._compare_time_specification(
            orig.time_specification, conv.time_specification, result
        )

        assert result.differences == ["Time type mismatch: Dynamic vs Discrete"]


class TestCompareOntologyMappings:
    def test_mapping_term_mismatch_is_recorded(self) -> None:
        result = _result()
        conv = _parse(COMPARE_GNN.replace("s = hidden_state", "s = state_factor"))

        _Host()._compare_ontology_mappings(
            _parse(COMPARE_GNN).ontology_mappings, conv.ontology_mappings, result
        )

        assert result.differences == ["Ontology mappings mismatch"]


class _CrossHost(RoundTripComparisonMixin):
    """Host stubbing the parsing/validator surfaces used by cross-format checks."""

    def __init__(
        self,
        supported_formats: list[GNNFormat],
        reference_file: Path,
        parsing_system: Any | None = None,
        cross_validator: Any | None = None,
    ) -> None:
        self.supported_formats = supported_formats
        self.reference_file = reference_file
        self.parsing_system = parsing_system
        self.cross_validator = cross_validator


class _Validator:
    """Stub cross-format validator returning a fixed consistency verdict."""

    def __init__(self, is_consistent: bool, inconsistencies: list[str]) -> None:
        self._verdict = SimpleNamespace(
            is_consistent=is_consistent, inconsistencies=inconsistencies
        )
        self.seen: list[str] = []

    def validate_cross_format_consistency(self, content: str) -> SimpleNamespace:
        self.seen.append(content)
        return self._verdict


@pytest.fixture
def reference_file(tmp_path: Path) -> Path:
    path = tmp_path / "reference.md"
    path.write_text(COMPARE_GNN, encoding="utf-8")
    return path


class TestCrossFormatConsistency:
    def test_consistent_content_records_no_critical_errors(
        self, reference_file: Path
    ) -> None:
        host = _CrossHost(
            [GNNFormat.MARKDOWN, GNNFormat.JSON],
            reference_file,
            parsing_system=SimpleNamespace(
                serialize=lambda model, fmt: f"{fmt.value}:serialized"
            ),
            cross_validator=_Validator(is_consistent=True, inconsistencies=[]),
        )
        report = ComprehensiveTestReport(reference_file=str(reference_file))

        host._test_cross_format_consistency(_parse(COMPARE_GNN), report)

        assert report.critical_errors == []

    def test_inconsistent_content_appends_inconsistencies(
        self, reference_file: Path
    ) -> None:
        host = _CrossHost(
            [GNNFormat.MARKDOWN, GNNFormat.JSON],
            reference_file,
            parsing_system=SimpleNamespace(
                serialize=lambda model, fmt: f"{fmt.value}:serialized"
            ),
            cross_validator=_Validator(
                is_consistent=False, inconsistencies=["json drift"]
            ),
        )
        report = ComprehensiveTestReport(reference_file=str(reference_file))

        host._test_cross_format_consistency(_parse(COMPARE_GNN), report)

        # Both the markdown read and the json serialization are validated,
        # so the stub verdict appends its inconsistency once per format.
        assert report.critical_errors == ["json drift", "json drift"]

    def test_validation_error_is_recorded_per_format(
        self, reference_file: Path
    ) -> None:
        def explode(content: str) -> SimpleNamespace:
            raise RuntimeError("validator boom")

        host = _CrossHost(
            [GNNFormat.MARKDOWN, GNNFormat.JSON],
            reference_file,
            parsing_system=SimpleNamespace(
                serialize=lambda model, fmt: f"{fmt.value}:serialized"
            ),
            cross_validator=SimpleNamespace(validate_cross_format_consistency=explode),
        )
        report = ComprehensiveTestReport(reference_file=str(reference_file))

        host._test_cross_format_consistency(_parse(COMPARE_GNN), report)

        # Validation runs per non-empty format content: markdown, then json.
        assert report.critical_errors == [
            "Cross-format validation failed for markdown: validator boom",
            "Cross-format validation failed for json: validator boom",
        ]

    def test_serialization_failure_appends_critical_error(
        self, reference_file: Path
    ) -> None:
        def fail_serialize(model: object, fmt: GNNFormat) -> str:
            raise RuntimeError("serializer boom")

        host = _CrossHost(
            [GNNFormat.MARKDOWN, GNNFormat.JSON],
            reference_file,
            parsing_system=SimpleNamespace(serialize=fail_serialize),
            cross_validator=_Validator(is_consistent=True, inconsistencies=[]),
        )
        report = ComprehensiveTestReport(reference_file=str(reference_file))

        host._test_cross_format_consistency(_parse(COMPARE_GNN), report)

        assert report.critical_errors == [
            "Failed to serialize to json: serializer boom"
        ]

    def test_unavailable_cross_format_module_is_skipped(
        self,
        reference_file: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import gnn.testing.round_trip_comparison as comparison_module

        monkeypatch.setattr(comparison_module, "CROSS_FORMAT_AVAILABLE", False)
        host = _CrossHost([GNNFormat.MARKDOWN, GNNFormat.JSON], reference_file)
        report = ComprehensiveTestReport(reference_file=str(reference_file))

        host._test_cross_format_consistency(_parse(COMPARE_GNN), report)

        assert report.critical_errors == []

    def test_reference_read_failure_bubbles_to_critical_errors(
        self, tmp_path: Path
    ) -> None:
        host = _CrossHost([GNNFormat.MARKDOWN], tmp_path / "missing.md")
        report = ComprehensiveTestReport(reference_file="missing.md")

        host._test_cross_format_consistency(_parse(COMPARE_GNN), report)

        assert len(report.critical_errors) == 1
        assert report.critical_errors[0].startswith(
            "Cross-format consistency test failed:"
        )
