"""Live JAX analysis consumes authored saved logs without executing a model."""

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from gnn.analysis.jax import extract_simulation_data, generate_analysis_from_logs


@pytest.fixture(autouse=True)
def close_consumer_figures():
    from matplotlib import pyplot

    yield
    pyplot.close("all")


def saved_log(tmp_path, *, raw_output=None):
    """Existing saved execution envelope; its values are authored test input."""
    data = {
        "model_name": "Authored saved trajectory Ω",
        "actions": [1, 0, 1],
        "beliefs": [[0.5, 0.5], [0.25, 0.75], [0.125, 0.875]],
        "free_energy": [1.75, -0.25, -0.75],
        "raw_output": (
            "Actions taken: [1 0 1]\n"
            "Final belief: [0.125 0.875]\n"
            "Average EFE: 0.25\n"
            "EFE for all actions: [1.0 -0.5]\n"
            "A matrix shape: (2, 2)\n"
            "B matrix shape: (2, 2, 2)\n"
            "C vector shape: (2,)\n"
            "D vector shape: (2,)\n"
            "Number of states: 2\n"
            "Number of observations: 2\n"
            "Number of actions: 2\n"
            if raw_output is None
            else raw_output
        ),
    }
    jax_dir = tmp_path / "execution" / "saved_model" / "jax"
    log = jax_dir / "execution_logs" / "saved_model_results.json"
    log.parent.mkdir(parents=True)
    log.write_text(
        json.dumps({"simulation_data": data}, ensure_ascii=False), encoding="utf-8"
    )
    return jax_dir, log, data


def verify_pngs(files, expected, output):
    assert len(files) == len(set(files)) == len(expected)
    assert {Path(file).name for file in files} == expected
    for file in files:
        path = Path(file)
        assert path.parent == output
        before = path.read_bytes()
        with Image.open(path) as image:
            assert image.format == "PNG"
            assert image.width >= 500 and image.height >= 300
            image.load()
            assert any(low < high for low, high in image.convert("RGB").getextrema())
        assert path.read_bytes() == before


ALL_PLOTS = {
    "saved_model_jax_free_energy.png",
    "saved_model_jax_action_dist.png",
    "saved_model_jax_action_timeline.png",
    "saved_model_jax_efe_comparison.png",
    "saved_model_jax_model_summary.png",
    "saved_model_jax_beliefs.png",
}


def test_public_saved_jax_log_extraction_preserves_exact_signed_trajectory(tmp_path):
    jax_dir, log, authored = saved_log(tmp_path)
    before = log.read_bytes()
    source_hash = hashlib.sha256(before).hexdigest()
    extracted = extract_simulation_data(jax_dir)
    assert extracted == {"framework": "jax", **authored}
    assert extracted["actions"].count(1) == 2
    assert sum(extracted["free_energy"]) == 0.75
    assert extracted["beliefs"][-1] == [0.125, 0.875]
    assert hashlib.sha256(log.read_bytes()).hexdigest() == source_hash
    assert log.read_bytes() == before


@pytest.mark.parametrize("failure", ["json", "nonobject", "directory"])
def test_public_saved_jax_log_read_failures_return_existing_empty_envelope(
    tmp_path, caplog, failure
):
    jax_dir = tmp_path / "jax"
    log = jax_dir / "execution_logs" / "unreadable_results.json"
    log.parent.mkdir(parents=True)
    if failure == "directory":
        log.mkdir()
        marker = log / "caller_note.txt"
        marker.write_bytes(b"retained unreadable directory\n")
        before = marker.read_bytes()
    else:
        log.write_text(
            "{invalid saved json" if failure == "json" else "[1, 2]", encoding="utf-8"
        )
        before = log.read_bytes()
    assert extract_simulation_data(jax_dir) == {
        "actions": [],
        "beliefs": [],
        "free_energy": [],
        "model_name": "",
        "framework": "jax",
    }
    assert "Failed to extract JAX data" in caplog.text
    assert (marker if failure == "directory" else log).read_bytes() == before


def test_public_saved_jax_analysis_reopens_all_native_pngs_and_leaves_log_unchanged(
    tmp_path,
):
    jax_dir, log, _ = saved_log(tmp_path)
    before = log.read_bytes()
    output = tmp_path / "analysis"
    files = generate_analysis_from_logs(jax_dir.parents[1], output, verbose=True)
    verify_pngs(files, ALL_PLOTS, output)
    assert log.read_bytes() == before


@pytest.mark.parametrize(
    "filename", ["saved_model_jax_free_energy.png", "saved_model_jax_beliefs.png"]
)
def test_public_saved_jax_analysis_returns_only_written_artifacts_after_collision(
    tmp_path, caplog, filename
):
    jax_dir, log, _ = saved_log(tmp_path)
    before = log.read_bytes()
    output = tmp_path / "analysis"
    collision = output / filename
    collision.mkdir(parents=True)
    marker = collision / "caller_note.txt"
    marker.write_bytes(b"retained caller plot collision\n")
    files = generate_analysis_from_logs(jax_dir.parents[1], output)
    verify_pngs(files, ALL_PLOTS - {filename}, output)
    assert filename not in {Path(file).name for file in files}
    assert "Failed to create" in caplog.text
    assert marker.read_bytes() == b"retained caller plot collision\n"
    assert log.read_bytes() == before


def test_public_saved_jax_analysis_malformed_log_keeps_healthy_sibling_outputs(
    tmp_path, caplog
):
    jax_dir, log, _ = saved_log(tmp_path)
    malformed = log.parent / "malformed_results.json"
    malformed.write_bytes(b"{invalid saved execution result\n")
    before = {path: path.read_bytes() for path in (log, malformed)}
    output = tmp_path / "analysis"
    files = generate_analysis_from_logs(jax_dir.parents[1], output)
    verify_pngs(files, ALL_PLOTS, output)
    assert "Failed to process" in caplog.text
    assert "malformed_results.json" in caplog.text
    assert all(path.read_bytes() == data for path, data in before.items())


def test_public_saved_jax_analysis_invalid_stdout_preserves_saved_trajectory_fallback(
    tmp_path, caplog
):
    jax_dir, log, data = saved_log(tmp_path, raw_output="Actions taken: [bad-token]\n")
    before = log.read_bytes()
    output = tmp_path / "analysis"
    files = generate_analysis_from_logs(jax_dir.parents[1], output)
    verify_pngs(
        files,
        ALL_PLOTS
        - {"saved_model_jax_efe_comparison.png", "saved_model_jax_model_summary.png"},
        output,
    )
    assert "Failed to parse JAX raw_output" in caplog.text
    assert extract_simulation_data(jax_dir)["actions"] == data["actions"]
    assert extract_simulation_data(jax_dir)["free_energy"] == data["free_energy"]
    assert log.read_bytes() == before


def test_public_saved_jax_analysis_blocked_output_refuses_without_replacing_it(
    tmp_path, caplog
):
    jax_dir, log, _ = saved_log(tmp_path)
    before = log.read_bytes()
    output = tmp_path / "blocked_analysis"
    output.write_bytes(b"retained caller output path\n")
    assert generate_analysis_from_logs(jax_dir.parents[1], output) == []
    assert "Failed to process" in caplog.text
    assert log.name in caplog.text
    assert str(output) in caplog.text
    assert output.read_bytes() == b"retained caller output path\n"
    assert log.read_bytes() == before


def test_public_saved_jax_missing_inputs_produce_empty_results_without_output_writes(
    tmp_path,
):
    absent = tmp_path / "absent_execution"
    output = tmp_path / "absent_analysis"
    assert generate_analysis_from_logs(absent, output) == []
    assert extract_simulation_data(absent) == {
        "actions": [],
        "beliefs": [],
        "free_energy": [],
        "model_name": "",
        "framework": "jax",
    }
    assert not absent.exists()
    assert not output.exists()
