"""Public saved-file round trips retain supplied model values and failures."""

import hashlib
import json
import logging
from copy import deepcopy
from pathlib import Path

import pytest

from gnn.parsers import GNNParsingSystem
from gnn.schema_validator import GNNValidator, ValidationLevel
from gnn.testing.round_trip_tester import GNNRoundTripTester


@pytest.fixture(autouse=True)
def restore_parser_logging():
    logger = logging.getLogger("gnn.parsers")
    before = logger.level
    try:
        yield
    finally:
        logger.setLevel(before)


def supplied_parameters() -> dict:
    return {
        "A": [[float(row == column) for column in range(4)] for row in range(4)],
        "B": [[[float(row == column)] * 4 for column in range(4)] for row in range(4)],
        "C": [-2.0, 0.0, 0.0, 3.0],
        "D": [0.0, 0.2, 0.3, 0.5],
        "offset": 0,
        "enabled": False,
        "calibration": [[-0.125, 0.0], [0.0, 1.5]],
        "settings": {"offset": 0, "enabled": False, "labels": ["a", "b"]},
    }


def authored_source(tmp_path: Path) -> Path:
    original = (
        Path(__file__).resolve().parents[2] / "input/gnn_files/discrete/simple_mdp.md"
    ).read_text(encoding="utf-8")
    source = tmp_path / "zero_prior.md"
    before, remainder = original.split("## InitialParameterization", 1)
    _, after = remainder.split("## Equations", 1)
    parameters = "\n".join(
        f"{name}={json.dumps(value)}" for name, value in supplied_parameters().items()
    )
    source.write_text(
        before.replace("Simple MDP Agent", "Authored zero prior agent").replace(
            "G[π,type=float]", "G[4,type=float]"
        )
        + "## InitialParameterization\n\n"
        + parameters
        + "\n\n## Equations"
        + after,
        encoding="utf-8",
    )
    return source


def test_validator_saved_roundtrips_preserve_supplied_parameter_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gnn.testing import round_trip_config

    source = authored_source(tmp_path)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    expected = supplied_parameters()
    saved = tmp_path / "converted"
    monkeypatch.setitem(round_trip_config.OUTPUT_CONFIG, "save_test_artifacts", True)
    validator = GNNValidator(
        enable_round_trip_testing=True,
        enable_cross_validation=False,
        validation_level=ValidationLevel.ROUND_TRIP,
    )
    validator.round_trip_tester.temp_dir = saved
    parsed = validator.parser.parse_file(source)
    assert parsed.parameters == expected
    dimensions = {
        name: variable.dimensions for name, variable in parsed.variables.items()
    }
    descriptions = {
        name: variable.description for name, variable in parsed.variables.items()
    }
    data_types = {
        "A": "float",
        "B": "float",
        "C": "float",
        "D": "float",
        "s": "float",
        "s_prime": "float",
        "o": "integer",
        "π": "float",
        "u": "integer",
        "G": "float",
        "t": "integer",
    }
    assert {
        name: variable.data_type for name, variable in parsed.variables.items()
    } == {name: "int" if name in {"o", "u", "t"} else "float" for name in data_types}

    result = validator.validate_file(source)

    assert len(result.round_trip_results) == 3, result.warnings
    assert {item.target_format.value for item in result.round_trip_results} == {
        "json",
        "xml",
        "yaml",
    }
    for item in result.round_trip_results:
        assert item.success, (item.errors, item.differences)
        for model in (item.original_model, item.parsed_back_model):
            values = {parameter.name: parameter.value for parameter in model.parameters}
            assert values == expected
            assert values["enabled"] is False
            assert type(values["offset"]) is int
            assert type(values["calibration"][0][1]) is float
            assert {
                variable.name: variable.dimensions for variable in model.variables
            } == dimensions
            assert {
                variable.name: variable.description for variable in model.variables
            } == descriptions
            assert {
                variable.name: variable.data_type.value for variable in model.variables
            } == data_types
            assert model.model_name == "Authored zero prior agent"
            assert model.annotation == parsed.model_annotation
    assert len(list(saved.glob("test_model.*"))) == 3
    for artifact in saved.glob("test_model.*"):
        reopened = GNNParsingSystem().parse_file(artifact)
        assert reopened.success, reopened.errors
        values = {
            parameter.name: parameter.value for parameter in reopened.model.parameters
        }
        assert values == expected
        assert values["enabled"] is False
        assert type(values["offset"]) is int
        assert {
            variable.name: variable.dimensions for variable in reopened.model.variables
        } == dimensions
        assert {
            variable.name: variable.description for variable in reopened.model.variables
        } == descriptions
        assert {
            variable.name: variable.data_type.value
            for variable in reopened.model.variables
        } == data_types
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    assert parsed.parameters == expected

    # Each public result is independently editable, including nested values.
    first = result.round_trip_results[0].original_model
    first_parameters = {parameter.name: parameter for parameter in first.parameters}
    first_parameters["settings"].value["labels"].append("edited")
    first_parameters["calibration"].value[0][0] = 99
    for item in result.round_trip_results[1:]:
        assert {
            parameter.name: parameter.value
            for parameter in item.original_model.parameters
        } == expected
    assert parsed.parameters == expected


def configure_saved_run(monkeypatch: pytest.MonkeyPatch, formats: list[str]) -> None:
    from gnn.testing import round_trip_config

    monkeypatch.setitem(round_trip_config.OUTPUT_CONFIG, "save_test_artifacts", True)
    monkeypatch.setitem(round_trip_config.FORMAT_TEST_CONFIG, "test_all_formats", False)
    monkeypatch.setitem(round_trip_config.FORMAT_TEST_CONFIG, "test_formats", formats)
    monkeypatch.setitem(round_trip_config.FORMAT_TEST_CONFIG, "format_overrides", {})


def test_category_selected_saved_artifacts_are_reopenable_and_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gnn.testing import round_trip_config

    source = authored_source(tmp_path)
    # The registry Markdown reader supports these numeric/boolean containers;
    # mapping fidelity belongs to the schema-validator test above.
    source.write_text(
        "\n".join(
            line
            for line in source.read_text(encoding="utf-8").splitlines()
            if not line.startswith("settings=")
        ),
        encoding="utf-8",
    )
    source_bytes = source.read_bytes()
    supplied = supplied_parameters()
    del supplied["settings"]
    admitted = GNNParsingSystem().parse_file(source)
    assert admitted.success, admitted.errors
    expected_parameters = {
        parameter.name: deepcopy(parameter.value)
        for parameter in admitted.model.parameters
    }
    assert {name: expected_parameters[name] for name in supplied} == supplied
    configure_saved_run(monkeypatch, [])
    monkeypatch.setitem(round_trip_config.FORMAT_TEST_CONFIG, "test_all_formats", True)
    monkeypatch.setitem(
        round_trip_config.FORMAT_TEST_CONFIG,
        "test_categories",
        {
            "schema_formats": True,
            "language_formats": False,
            "formal_formats": False,
            "grammar_formats": True,
            "temporal_formats": False,
            "binary_formats": True,
        },
    )
    saved = tmp_path / "selected"
    tester = GNNRoundTripTester(temp_dir=saved)
    tester.reference_file = source

    report = tester.run_comprehensive_tests()

    expected_formats = {
        "json",
        "xml",
        "yaml",
        "xsd",
        "asn1",
        "pkl",
        "protobuf",
        "bnf",
        "ebnf",
        "pickle",
    }
    assert {
        item.target_format.value for item in report.round_trip_results
    } == expected_formats
    assert report.successful_tests == 10, [
        (item.target_format, item.errors, item.differences)
        for item in report.round_trip_results
    ]
    assert report.failed_tests == 0
    artifacts = list(saved.glob("test_model.*"))
    assert len(artifacts) == 10
    assert (saved / "test_model.pkl").is_file()
    assert (saved / "test_model.pickle").is_file()
    assert (saved / "test_model.ebnf").is_file()
    for artifact in artifacts:
        reopened = GNNParsingSystem().parse_file(artifact)
        assert reopened.success, (artifact, reopened.errors)
        assert {
            parameter.name: parameter.value for parameter in reopened.model.parameters
        } == expected_parameters
        assert {
            variable.name: variable.dimensions for variable in reopened.model.variables
        }["B"] == [4, 4, 4]
    report_path = tmp_path / "report.md"
    rendered = tester.generate_report(report, report_path)
    assert report_path.read_text(encoding="utf-8") == rendered
    assert f"**Reference File:** `{source}`" in rendered
    assert "**Total Tests:** 10" in rendered
    for target in expected_formats:
        assert f"### {target} ✅ PASS" in rendered
    assert source.read_bytes() == source_bytes


def test_unwritable_artifact_directory_is_a_failed_conversion_not_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = authored_source(tmp_path)
    blocked = tmp_path / "occupied"
    blocked.write_bytes(b"previous caller artifact")
    configure_saved_run(monkeypatch, ["json", "pickle"])
    tester = GNNRoundTripTester(temp_dir=blocked)
    tester.reference_file = source

    report = tester.run_comprehensive_tests()

    assert report.total_tests == 2
    assert report.successful_tests == 0
    assert report.failed_tests == 2
    assert report.get_success_rate() == 0
    for item in report.round_trip_results:
        assert not item.success
        assert any(
            "File exists" in error and str(blocked) in error for error in item.errors
        ), item.errors
    assert blocked.read_bytes() == b"previous caller artifact"
    text = tester.generate_report(report)
    assert "**Failed:** 2" in text
    assert "### json ❌ FAIL" in text
    assert "### pickle ❌ FAIL" in text


def test_missing_roundtrip_source_retains_actual_path(tmp_path: Path) -> None:
    missing = tmp_path / "not-supplied.md"
    tester = GNNRoundTripTester(temp_dir=tmp_path / "saved")
    tester.reference_file = missing

    with pytest.raises(FileNotFoundError, match="not-supplied") as error:
        tester.run_comprehensive_tests()

    assert str(missing) in str(error.value)
    assert not tester.temp_dir.exists()


def test_source_directory_refuses_conversion_with_causal_report(tmp_path: Path) -> None:
    source = tmp_path / "actually-a-directory.md"
    source.mkdir()
    tester = GNNRoundTripTester(temp_dir=tmp_path / "saved")
    tester.reference_file = source

    report = tester.run_comprehensive_tests()

    assert report.reference_file == str(source)
    assert report.total_tests == 0
    assert report.get_success_rate() == 0
    assert any(
        "Is a directory" in error and str(source) in error
        for error in report.critical_errors
    ), report.critical_errors
    text = tester.generate_report(report)
    assert "## Critical Issues" in text
    assert str(source) in text
    assert not tester.temp_dir.exists()
