"""Saved round-trip artifacts preserve a real supplied model across formats."""

import json
import logging
from pathlib import Path

import pytest

from gnn.parsers import GNNFormat, GNNParsingSystem


@pytest.fixture(autouse=True)
def preserve_parser_logging():
    """Restore the import-time parser setting even for an isolated strategy run."""
    logger = logging.getLogger("gnn.parsers")
    prior_level = logger.level
    try:
        yield
    finally:
        logger.setLevel(prior_level)


def test_saved_round_trip_artifacts_reparse_to_the_source_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gnn.testing import round_trip_config
    from gnn.testing.round_trip_tester import GNNRoundTripTester

    source = (
        Path(__file__).resolve().parents[2] / "input/gnn_files/discrete/simple_mdp.md"
    )
    original = GNNParsingSystem().parse_file(source, GNNFormat.MARKDOWN)
    assert original.success, original.errors
    monkeypatch.setitem(
        round_trip_config.REFERENCE_CONFIG, "reference_file", str(source)
    )
    monkeypatch.setitem(round_trip_config.OUTPUT_CONFIG, "save_test_artifacts", True)
    tester = GNNRoundTripTester(temp_dir=tmp_path / "converted")
    report = tester.run_comprehensive_tests()
    published = tmp_path / "round_trip_report.md"
    text = tester.generate_report(report, published)

    assert not report.critical_errors, report.critical_errors
    assert report.round_trip_results, "A report must contain actual source conversions"
    assert published.read_text(encoding="utf-8") == text
    assert str(source) in text
    source_variables = {
        variable.name: variable for variable in original.model.variables
    }
    source_parameters = {
        parameter.name: parameter.value for parameter in original.model.parameters
    }
    for result in report.round_trip_results:
        assert result.success, (result.target_format, result.errors, result.differences)
        assert result.parsed_back_model is not None, result.target_format
        assert result.parsed_back_model.model_name == "Simple MDP Agent"
        converted_variables = {
            variable.name: variable for variable in result.parsed_back_model.variables
        }
        assert set(converted_variables) == set(source_variables)
        for name, variable in source_variables.items():
            assert converted_variables[name].dimensions == variable.dimensions
            assert converted_variables[name].data_type == variable.data_type
            assert converted_variables[name].var_type == variable.var_type
        assert {
            parameter.name: parameter.value
            for parameter in result.parsed_back_model.parameters
        } == source_parameters
        assert result.checksum_original == result.checksum_converted
        assert f"### {result.target_format.value} ✅ PASS" in text

    # Re-open the actual saved files, independently of the report's in-memory models.
    saved = list((tmp_path / "converted").glob("test_model.*"))
    assert len(saved) == len(report.round_trip_results)
    for artifact in saved:
        parsed = GNNParsingSystem().parse_file(artifact)
        assert parsed.success, (artifact.name, parsed.errors)
        assert parsed.model.model_name == "Simple MDP Agent"
        variables = {variable.name: variable for variable in parsed.model.variables}
        assert variables["A"].dimensions == [4, 4]
        assert variables["B"].dimensions == [4, 4, 4]
        assert {
            parameter.name: parameter.value for parameter in parsed.model.parameters
        }["D"] == source_parameters["D"]


def test_native_round_trip_strategy_distinguishes_supplied_source_from_missing_file(
    tmp_path: Path,
) -> None:
    from gnn.testing import RoundTripTestStrategy

    source = tmp_path / "supplied_sensor.md"
    source.write_bytes(
        (
            Path(__file__).resolve().parents[2]
            / "input/gnn_files/discrete/simple_mdp.md"
        ).read_bytes()
    )
    missing = tmp_path / "missing_sensor.md"
    output = tmp_path / "round_trip_results"
    strategy = RoundTripTestStrategy()
    strategy.configure(output_dir=output)

    result = strategy.test([source, missing])

    valid = result["file_results"][str(source)]
    assert valid["success"], valid
    assert valid["format_results"], (
        "The supplied model must have real per-format evidence"
    )
    assert all(format_result["success"] for format_result in valid["format_results"])
    assert {
        format_result["target_format"] for format_result in valid["format_results"]
    } >= {"json", "xml", "yaml", "pickle"}
    failed = result["file_results"][str(missing)]
    assert not failed["success"]
    assert str(missing) in failed["error"]
    assert not failed["format_results"]
    assert not result["success"]
    assert result["summary"]["successful_files"] == 1
    assert result["summary"]["success_rate"] == 50
    records = list(output.glob("round_trip_test_results_*.json"))
    assert len(records) == 1
    saved = json.loads(records[0].read_text(encoding="utf-8"))
    assert saved == result
    assert set(saved["file_results"]) == {str(source), str(missing)}
