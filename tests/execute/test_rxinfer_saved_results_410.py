"""Authored saved legacy result consumers; no Julia inference is executed.

Canonical refusal fixtures are manufactured from the documented producer schema,
not labelled as captured native backend output.
"""

import json
import logging
from pathlib import Path

import pytest

from gnn.analysis.rxinfer.result_ingestion import read_result_object
from gnn.execute.rxinfer.rxinfer_results import (
    collect_rxinfer_results,
    format_rxinfer_report,
    parse_rxinfer_output,
    summarize_posteriors,
)


def save_result(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_authored_legacy_saved_result_retains_values_through_report(
    tmp_path: Path,
) -> None:
    artifact = save_result(
        tmp_path / "sensor" / "simulation_results.json",
        {
            "model_name": "sensor",
            "iterations": 3,
            "converged": True,
            "free_energy": [4.0, 3.0, 0.0],
            "elapsed_seconds": 0.0,
            "posteriors": {
                "zero_gaussian": {
                    "type": "NormalMeanVariance",
                    "mean": 0.0,
                    "variance": 0.0,
                },
                "zero_scalar": 0.0,
                "scalar": 2.5,
                "prior": {"type": "dirichlet", "alpha": [1.0, 3.0]},
                "state": {"type": "categorical", "probabilities": [0.0, 1.0]},
            },
        },
    )

    parsed = parse_rxinfer_output(artifact)
    assert parsed is not None
    assert parsed["free_energy"] == [4.0, 3.0, 0.0]
    assert parsed["elapsed_seconds"] == 0.0
    summary = summarize_posteriors(parsed)
    assert summary["zero_gaussian"] == {
        "type": "NormalMeanVariance",
        "mean": 0.0,
        "std": 0.0,
    }
    assert summary["zero_scalar"] == {"type": "scalar", "value": 0.0}
    assert summary["scalar"] == {"type": "scalar", "value": 2.5}
    assert summary["prior"]["mean"] == [0.25, 0.75]
    assert summary["prior"]["argmax"] == 1
    assert summary["prior"]["entropy"] == pytest.approx(0.5623)
    assert summary["state"]["mean"] == [0.0, 1.0]

    results = collect_rxinfer_results(tmp_path)
    assert len(results) == 1
    assert results[0]["source_file"] == str(artifact)
    assert results[0]["posterior_summary"] == summary
    metrics = results[0]["convergence_metrics"]
    assert metrics["total_iterations"] == 3
    assert metrics["final_free_energy"] == 0.0
    assert metrics["free_energy_change"] == -4.0
    assert metrics["relative_change"] == 1.0
    report = format_rxinfer_report(results)
    assert "**Models analyzed**: 1" in report
    assert "**Final Free Energy**: 0.0000" in report
    assert "| zero_gaussian | NormalMeanVariance | 0.0000 |" in report
    assert "| zero_scalar | scalar | 0.0000 |" in report
    assert "| scalar | scalar | 2.5000 |" in report


def test_overlapping_discovery_collects_distinct_saved_files_once_and_filters(
    tmp_path: Path,
) -> None:
    alpha = save_result(
        tmp_path / "alpha_rxinfer_results.json",
        {"model_name": "alpha", "free_energy": [2.0, 1.0]},
    )
    beta = save_result(
        tmp_path / "nested" / "rxinfer_output.json",
        {"model_name": "beta", "posteriors": {"x": 4.0}},
    )
    results = collect_rxinfer_results(tmp_path)
    assert len(results) == 2
    assert {result["source_file"] for result in results} == {str(alpha), str(beta)}
    assert {result["model_name"] for result in results} == {"alpha", "beta"}
    report = format_rxinfer_report(results)
    assert report.count("## alpha\n") == report.count("## beta\n") == 1
    filtered = collect_rxinfer_results(tmp_path, model_name="beta")
    assert len(filtered) == 1
    assert filtered[0]["source_file"] == str(beta)
    assert filtered[0]["posterior_summary"]["x"]["value"] == 4.0


def test_saved_zero_primary_fields_take_precedence_over_legacy_aliases(
    tmp_path: Path,
) -> None:
    artifact = save_result(
        tmp_path / "zero_rxinfer.json",
        {
            "modelName": "zero",
            "iterations": 0,
            "n_iterations": 9,
            "elapsed_seconds": 0.0,
            "elapsed": 99.0,
            "free_energy": 0.0,
            "F": 8.0,
            "posteriors": {
                "x": {"mean": 0.0, "mu": 7.0, "variance": 0.0, "sigma2": 4.0}
            },
        },
    )
    actual = parse_rxinfer_output(artifact)
    assert actual is not None
    assert actual["model_name"] == "zero"
    assert actual["iterations"] == 0
    assert actual["elapsed_seconds"] == 0.0
    assert actual["free_energy"] == [0.0]
    assert actual["posteriors"]["x"]["mean"] == [0.0]
    assert actual["posteriors"]["x"]["variance"] == [0.0]


def test_empty_primary_fields_keep_supported_legacy_aliases(tmp_path: Path) -> None:
    artifact = save_result(
        tmp_path / "aliases_rxinfer.json",
        {
            "model_name": "",
            "modelName": "aliases",
            "n_iterations": 2,
            "elapsed": 1.25,
            "free_energy": [],
            "freeEnergy": {"values": [], "data": [3, 1]},
            "posteriors": {},
            "q": {
                "x": {
                    "distribution": "NormalMeanVariance",
                    "mean": [],
                    "μ": [2.0],
                    "σ²": [4.0],
                }
            },
        },
    )
    actual = parse_rxinfer_output(artifact)
    assert actual is not None
    assert actual["model_name"] == "aliases"
    assert actual["iterations"] == 2
    assert actual["elapsed_seconds"] == 1.25
    assert actual["free_energy"] == [3.0, 1.0]
    assert summarize_posteriors(actual)["x"] == {
        "type": "NormalMeanVariance",
        "mean": 2.0,
        "std": 2.0,
    }


@pytest.mark.parametrize(
    ("contents", "cause"),
    [
        (b'{"posteriors":', "JSONDecodeError"),
        (b"[]", "ValueError"),
        (b"\xff", "UnicodeDecodeError"),
        (b'{"free_energy":["bad-number"]}', "ValueError"),
        (b'{"posteriors":[1]}', "ValueError"),
        (json.dumps({"free_energy": [10**400]}).encode("utf-8"), "OverflowError"),
    ],
)
def test_unreadable_saved_results_return_none_with_path_and_cause(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, contents: bytes, cause: str
) -> None:
    artifact = tmp_path / "bad_rxinfer.json"
    artifact.write_bytes(contents)
    valid = save_result(
        tmp_path / "good_rxinfer.json", {"model_name": "good", "posteriors": {"x": 1.5}}
    )
    with caplog.at_level(logging.WARNING):
        assert parse_rxinfer_output(artifact) is None
        results = collect_rxinfer_results(tmp_path)
    assert str(artifact) in caplog.text
    assert cause in caplog.text
    assert len(results) == 1 and results[0]["source_file"] == str(valid)


def test_missing_saved_result_keeps_public_none_and_empty_collection(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    missing = tmp_path / "missing" / "simulation_results.json"
    with caplog.at_level(logging.WARNING):
        assert parse_rxinfer_output(missing) is None
        assert collect_rxinfer_results(missing.parent) == []
    assert str(missing) in caplog.text
    assert "No results found" in format_rxinfer_report([])


@pytest.mark.parametrize("schema_marker", [True, False])
@pytest.mark.parametrize("family", ["categorical", "continuous"])
def test_manufactured_canonical_saved_schema_is_refused_without_losing_payload(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, family: str, schema_marker: bool
) -> None:
    # These source-schema fixtures do not claim execution by a native backend.
    payload = {
        "framework": "RxInfer.jl",
        "model_name": "canonical",
        "num_timesteps": 2,
        "model_parameters": {"inference_iterations": 3},
        "runtime_metadata": {"model_kind": family},
        "variational_free_energy": [6.0, 4.0, 3.0],
        "vfe_per_iteration": [6.0, 4.0, 3.0],
    }
    if schema_marker:
        payload["schema_version"] = "rxinfer_simulation_v1"
    if family == "continuous":
        payload.update(
            beliefs=[[0.0, -0.5], [1.0, 2.0]],
            posterior_cov=[[[1.0, 0.2], [0.2, 2.0]], [[0.5, 0.1], [0.1, 1.0]]],
        )
    else:
        payload.update(
            beliefs=[[0.25, 0.75], [0.5, 0.5]], expected_free_energy=[-1.0, -2.0]
        )
    artifact = save_result(tmp_path / "simulation_results.json", payload)
    with caplog.at_level(logging.WARNING):
        assert parse_rxinfer_output(artifact) is None
        assert collect_rxinfer_results(tmp_path) == []
    assert str(artifact) in caplog.text
    assert "Unsupported canonical" in caplog.text
    assert "read_result_object" in caplog.text
    # The actual current reader preserves the full saved object, including the
    # distinct timestep/iteration counts, energy signs and covariance layout.
    assert read_result_object(artifact) == payload


def test_saved_gaussian_covariance_without_vfe_keeps_canonical_reader_identity(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Missing optional VFE does not make a Gaussian trajectory a legacy named
    # posterior. This manufactured fixture exercises only saved-file reading.
    payload = {
        "model_name": "gaussian-without-vfe",
        "model_kind": "continuous",
        "num_timesteps": 1,
        "beliefs": [[0.0, -0.5]],
        "posterior_cov": [[[1.0, 0.2], [0.2, 2.0]]],
    }
    artifact = save_result(tmp_path / "simulation_results.json", payload)
    with caplog.at_level(logging.WARNING):
        assert parse_rxinfer_output(artifact) is None
        assert collect_rxinfer_results(tmp_path) == []
    assert str(artifact) in caplog.text and "read_result_object" in caplog.text
    assert read_result_object(artifact) == payload


@pytest.mark.parametrize(
    "name", ["sensor_rxinfer_execution_log.json", "rxinfer_summary.json"]
)
def test_discovered_execution_metadata_is_not_reported_as_legacy_inference(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, name: str
) -> None:
    artifact = save_result(
        tmp_path / name,
        {
            "script_path": "sensor_rxinfer.jl",
            "success": True,
            "duration_seconds": 0.2,
            "return_code": 0,
        },
    )
    with caplog.at_level(logging.WARNING):
        assert parse_rxinfer_output(artifact) is None
        assert collect_rxinfer_results(tmp_path) == []
    assert str(artifact) in caplog.text
    assert "Unsupported" in caplog.text
