"""Configured public saved conversions, real partial reports, and source custody.

These are authored Markdown models, not inference output or compiler acceptance.
The independently reopened files are the value oracle; report flags alone are not.
"""

import hashlib
import logging

import pytest

from gnn.parsers import GNNParsingSystem, UnifiedGNNParser
from gnn.testing import round_trip_config
from gnn.testing.round_trip_tester import GNNRoundTripTester

SOURCE = """# Saved consumer model
## GNNSection
ActInfPOMDP
## GNNVersionAndFlags
GNN v1
## ModelName
Independent saved report model
## ModelAnnotation
Authored signed calibration with zero and false; no inference is performed.
## StateSpaceBlock
state[2, 1, type=float] # two state slots
observation[2, type=int] # integer observations
action[1, type=bool] # binary control
## Connections
state>observation
action>state
## InitialParameterization
calibration=[[-0.125, 0.0], [0.0, 1.5]]
offset=0
enabled=False
mask=[[true, false], [false, true]]
## Time
Dynamic
DiscreteTime=t
ModelTimeHorizon=7
## Footer
The source footer is retained.
"""
VALUES = {
    "calibration": [[-0.125, 0.0], [0.0, 1.5]],
    "offset": 0,
    "enabled": False,
    "mask": [[True, False], [False, True]],
}


@pytest.fixture(autouse=True)
def restore_parser_logging():
    logger = logging.getLogger("gnn.parsers")
    prior = logger.level
    yield
    logger.setLevel(prior)


def configure_saved_request(monkeypatch, source, formats, overrides=None):
    monkeypatch.setitem(
        round_trip_config.REFERENCE_CONFIG, "reference_file", str(source)
    )
    monkeypatch.setitem(round_trip_config.FORMAT_TEST_CONFIG, "test_all_formats", False)
    monkeypatch.setitem(round_trip_config.FORMAT_TEST_CONFIG, "test_formats", formats)
    monkeypatch.setitem(
        round_trip_config.FORMAT_TEST_CONFIG, "format_overrides", overrides or {}
    )
    monkeypatch.setitem(round_trip_config.OUTPUT_CONFIG, "save_test_artifacts", True)
    monkeypatch.setitem(
        round_trip_config.LOGGING_CONFIG, "enable_detailed_output", True
    )
    monkeypatch.setitem(round_trip_config.LOGGING_CONFIG, "enable_debug", True)


def assert_authored_model(model):
    assert model.model_name == "Independent saved report model"
    assert model.annotation == (
        "Authored signed calibration with zero and false; no inference is performed."
    )
    assert {p.name: p.value for p in model.parameters} == VALUES
    assert type(next(p.value for p in model.parameters if p.name == "offset")) is int
    assert next(p.value for p in model.parameters if p.name == "enabled") is False
    assert {v.name: (v.dimensions, v.data_type.value) for v in model.variables} == {
        "state": ([2, 1], "float"),
        "observation": ([2], "integer"),
        "action": ([1], "binary"),
    }
    assert sorted(
        (tuple(c.source_variables), tuple(c.target_variables), c.connection_type.value)
        for c in model.connections
    ) == [
        (("action",), ("state",), "directed"),
        (("state",), ("observation",), "directed"),
    ]
    assert model.time_specification.time_type == "Dynamic"
    assert model.time_specification.discretization == "DiscreteTime"
    assert model.time_specification.horizon == 7


@pytest.mark.parametrize("grouped", [True, False])
def test_public_selective_aliases_emit_independent_saved_values_and_report(
    tmp_path, monkeypatch, capsys, caplog, grouped
):
    source = tmp_path / "authored.md"
    source.write_text(SOURCE, encoding="utf-8")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    # These aliases are caller configuration, not replacement parsers. The
    # override removes XML and adds YAML; duplicate aliases cannot duplicate files.
    configure_saved_request(
        monkeypatch,
        source,
        [
            "json",
            "xml",
            "binary",
            "functional",
            "grammar",
            "temporal",
            "znotation",
            "binary",
            "unrecognized_request",
        ],
        {"xml": False, "yaml": True, "unrecognized_override": True},
    )
    monkeypatch.setitem(
        round_trip_config.LOGGING_CONFIG, "enable_format_groups", grouped
    )
    artifacts = tmp_path / "saved"
    with caplog.at_level(logging.WARNING):
        tester = GNNRoundTripTester(temp_dir=artifacts)
        report = tester.run_comprehensive_tests()
    targets = {"json", "yaml", "pickle", "haskell", "bnf", "tla_plus", "z_notation"}
    assert {r.target_format.value for r in report.round_trip_results} == targets
    assert report.total_tests == report.successful_tests == 7
    assert report.failed_tests == 0, [r.errors for r in report.round_trip_results]
    assert report.critical_errors == []
    assert report.reference_file == str(source)
    assert "Unknown format in configuration: unrecognized_request" in caplog.text
    assert "Unknown format in overrides: unrecognized_override" in caplog.text
    assert not (artifacts / "test_model.xml").exists()
    files = list(artifacts.iterdir())
    assert len(files) == 7
    assert (artifacts / "test_model.pickle").read_bytes().startswith(b"\x80")
    for saved in files:
        reopened = GNNParsingSystem().parse_file(saved)
        assert reopened.success, (saved, reopened.errors)
        assert_authored_model(reopened.model)
        alternate = UnifiedGNNParser().parse_file(saved)
        assert alternate.success, (saved, alternate.errors)
        assert alternate.source_file == str(saved)
        assert alternate.model.source_format is reopened.model.source_format
        assert_authored_model(alternate.model)
    for result in report.round_trip_results:
        assert result.success, (result.errors, result.differences)
        assert result.errors == []
        assert result.differences == []
        assert_authored_model(result.original_model)
        assert_authored_model(result.parsed_back_model)
    destination = tmp_path / "round-trip-report.md"
    destination.write_bytes(b"old report remains until generated")
    rendered = tester.generate_report(report, destination)
    assert destination.read_text(encoding="utf-8") == rendered
    assert f"**Reference File:** `{source}`" in rendered
    assert "**Total Tests:** 7" in rendered
    assert "**Failed:** 0" in rendered
    for target in targets:
        assert f"### {target} ✅ PASS" in rendered
    output = capsys.readouterr().out
    assert "Independent saved report model" in output
    assert "Total tests: 7" in output
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash


@pytest.mark.parametrize("fail_fast", [False, True])
def test_real_target_path_conflict_retains_partial_failure_report_and_prior_bytes(
    tmp_path, monkeypatch, capsys, fail_fast
):
    source = tmp_path / "authored.md"
    source.write_text(SOURCE, encoding="utf-8")
    original = source.read_bytes()
    configure_saved_request(monkeypatch, source, ["json", "xml", "yaml"])
    monkeypatch.setitem(round_trip_config.TEST_BEHAVIOR_CONFIG, "fail_fast", fail_fast)
    artifacts = tmp_path / "saved"
    obstruction = artifacts / "test_model.json"
    obstruction.mkdir(parents=True)
    prior = obstruction / "previous.json"
    prior.write_bytes(b'{"previous":false,"offset":0}')
    prior_bytes = prior.read_bytes()
    tester = GNNRoundTripTester(temp_dir=artifacts)

    report = tester.run_comprehensive_tests()

    assert report.total_tests == (1 if fail_fast else 3)
    assert report.failed_tests == 1
    assert report.successful_tests == (0 if fail_fast else 2)
    failure = report.round_trip_results[0]
    assert failure.target_format.value == "json"
    assert failure.success is False
    assert failure.parsed_back_model is None
    assert any(str(obstruction) in e for e in failure.errors), failure.errors
    assert failure.converted_content
    for suffix in ("xml", "yaml"):
        destination = artifacts / f"test_model.{suffix}"
        if fail_fast:
            assert not destination.exists()
        else:
            reopened = GNNParsingSystem().parse_file(destination)
            assert reopened.success, reopened.errors
            assert_authored_model(reopened.model)
    path = tmp_path / "partial-report.md"
    rendered = tester.generate_report(report, path)
    assert path.read_text(encoding="utf-8") == rendered
    assert "### json ❌ FAIL" in rendered
    assert str(obstruction) in rendered
    assert "Fix serialization/parsing for json" in rendered
    assert "**Failed:** 1" in rendered
    assert "json" in capsys.readouterr().out
    assert obstruction.is_dir()
    assert prior.read_bytes() == prior_bytes
    assert source.read_bytes() == original
