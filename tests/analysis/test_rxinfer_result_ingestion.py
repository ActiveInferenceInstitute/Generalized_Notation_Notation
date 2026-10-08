"""Real result-file consumers refuse malformed and explicitly empty evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gnn.analysis.rxinfer import extract_simulation_data, generate_analysis_from_logs
from gnn.analysis.rxinfer.result_ingestion import (
    RxInferResultReadError,
    read_result_object,
)


@pytest.mark.parametrize("summary", ["{", "[]", '{"execution_details": []}'])
def test_current_summary_cannot_admit_orphan_results(
    tmp_path: Path, summary: str, caplog: pytest.LogCaptureFixture
) -> None:
    execution = tmp_path / "execution"
    results = execution / "inherited-model/rxinfer/simulation_data"
    results.mkdir(parents=True)
    (results / "simulation_results.json").write_text(
        json.dumps(
            {
                "schema_version": "rxinfer_simulation_v1",
                "beliefs": [[0.2, 0.8], [0.3, 0.7]],
                "observations": [0, 1],
                "true_states": [1, 1],
            }
        ),
        encoding="utf-8",
    )
    summaries = execution / "summaries"
    summaries.mkdir()
    summary_file = summaries / "execution_summary.json"
    summary_file.write_text(summary, encoding="utf-8")
    output = tmp_path / "analysis"

    generated = generate_analysis_from_logs(execution, output)

    assert generated == []
    assert not list(output.glob("*"))
    if summary != '{"execution_details": []}':
        assert str(summary_file) in caplog.text
        assert "RxInferResultReadError" in caplog.text


def test_invalid_encoding_keeps_failure_context_and_legacy_empty_result(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    results = tmp_path / "simulation_data"
    results.mkdir()
    malformed = results / "simulation_results.json"
    malformed.write_bytes(b"\xff\xfe")

    data = extract_simulation_data(tmp_path)

    assert data["beliefs"] == []
    assert data["framework"] == "rxinfer"
    assert str(malformed) in caplog.text
    assert "UnicodeDecodeError" in caplog.text


def test_standalone_result_extraction_preserves_native_values(tmp_path: Path) -> None:
    results = tmp_path / "simulation_data"
    results.mkdir()
    source = {
        "schema_version": "rxinfer_simulation_v1",
        "beliefs": [[0.2, 0.8], [0.3, 0.7]],
        "model_name": "asymmetric",
        "variational_free_energy": [3.25, 2.5],
    }
    (results / "simulation_results.json").write_text(
        json.dumps(source), encoding="utf-8"
    )

    extracted = extract_simulation_data(tmp_path)

    for key, value in source.items():
        assert extracted[key] == value


@pytest.mark.parametrize("contents", ["{", "[]"])
def test_read_failure_keeps_original_cause(tmp_path: Path, contents: str) -> None:
    source = tmp_path / "result.json"
    source.write_text(contents, encoding="utf-8")

    with pytest.raises(RxInferResultReadError) as raised:
        read_result_object(source)

    assert raised.value.path == source
    assert isinstance(raised.value.__cause__, ValueError)
    assert type(raised.value.__cause__).__name__ in raised.value.reason
