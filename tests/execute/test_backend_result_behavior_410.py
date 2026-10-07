"""Step 12 result consumers preserve source values and diagnose bad artifacts."""

import json
import logging
from pathlib import Path

import pytest

from gnn.execute.data_extractors import extract_simulation_data_from_files


@pytest.mark.parametrize(
    "framework", ["pymdp", "jax", "numpyro", "pytorch", "ngclearn"]
)
def test_backend_artifact_consumer_retains_categorical_trace_values(
    tmp_path: Path, framework: str
) -> None:
    directory = tmp_path / "simulation_data"
    directory.mkdir()
    payload = {
        "beliefs": [[0.3, 0.7], [0.1, 0.9]],
        "actions": [1, 0],
        "observations": [2, 1],
        "expected_free_energy": [[0.5, -0.4], [-0.7, 0.25]],
        "expected_free_energy_convention": "lower_is_better",
        "num_timesteps": 2,
    }
    (directory / "sensor_simulation_results.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    actual = extract_simulation_data_from_files(
        tmp_path, framework, logging.getLogger("test.backend.consumer")
    )

    for key, value in payload.items():
        assert actual[key] == value


@pytest.mark.parametrize(
    "framework", ["pymdp", "jax", "numpyro", "pytorch", "ngclearn"]
)
def test_backend_artifact_consumer_diagnoses_corrupt_result_without_success_data(
    tmp_path: Path, framework: str, caplog: pytest.LogCaptureFixture
) -> None:
    directory = tmp_path / "simulation_data"
    directory.mkdir()
    artifact = directory / "sensor_simulation_results.json"
    artifact.write_text('{"beliefs":', encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        actual = extract_simulation_data_from_files(
            tmp_path, framework, logging.getLogger("test.backend.consumer")
        )

    assert actual == {}
    assert str(artifact) in caplog.text
    assert "Failed to parse" in caplog.text
