"""Public saved-episode utilities preserve authored data without model execution."""

import hashlib
import json
import math
import pickle
import re
from pathlib import Path

import numpy as np
import pytest

from gnn.execute.pymdp import (
    create_output_directory_with_timestamp,
    format_duration,
    generate_simulation_summary,
    safe_json_dump,
    save_simulation_results,
)


def saved_episodes(tmp_path):
    """Authored saved-input fixture, not evidence of a backend run."""
    payload = {
        "traces": [
            {
                "episode": 0,
                "true_states": [0, 1],
                "observations": [0, 1],
                "actions": [1, 0],
                "rewards": [-0.25, 1.25],
                "beliefs": [[0.5, 0.5], [0.75, 0.25]],
            },
            {
                "episode": 1,
                "true_states": [1, 1, 0],
                "observations": [1, 1, 0],
                "actions": [0, 1, 0],
                "rewards": [-1.5, 0.5, -0.5],
                "beliefs": [[1.0, 0.0], [0.25, 0.75], None],
            },
            {
                "episode": 2,
                "true_states": [],
                "observations": [],
                "actions": [],
                "rewards": [],
                "beliefs": [],
            },
        ],
        "metrics": {"episode_rewards": [1.0, -1.5, 0.0]},
        "config": {"model_name": "Authored saved episodes Ω", "num_states": 2},
    }
    source = tmp_path / "authored_episodes.json"
    source.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return source, json.loads(source.read_text(encoding="utf-8"))


def test_public_saved_episode_summary_has_independent_reward_and_entropy_oracles(
    tmp_path,
):
    source, saved = saved_episodes(tmp_path)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    caller_bytes = pickle.dumps(saved, protocol=5)
    summary = generate_simulation_summary(saved["traces"], saved["metrics"])
    assert summary["total_episodes"] == 3
    assert summary["successful_episodes"] == 1
    assert summary["total_steps"] == 5
    assert summary["success_rate"] == pytest.approx(1 / 3)
    assert summary["average_episode_length"] == pytest.approx(5 / 3)
    assert summary["average_reward"] == pytest.approx(-1 / 6)
    episodes = summary["episode_statistics"]
    assert [item["episode"] for item in episodes] == [0, 1, 2]
    assert [item["total_reward"] for item in episodes] == [1.0, -1.5, 0.0]
    assert [item["episode_length"] for item in episodes] == [2, 3, 0]
    assert [item["final_state"] for item in episodes] == [1, 0, None]
    assert [item["success"] for item in episodes] == [True, False, False]
    # H(1/2,1/2)=ln2; H(3/4,1/4)=2ln2-(3/4)ln3, independently.
    quarter_entropy = 2 * math.log(2) - 0.75 * math.log(3)
    assert episodes[0]["mean_belief_entropy"] == pytest.approx(
        (math.log(2) + quarter_entropy) / 2, abs=1e-12
    )
    assert episodes[1]["mean_belief_entropy"] == pytest.approx(
        quarter_entropy / 2, abs=1e-12
    )
    assert episodes[2]["mean_belief_entropy"] == 0.0
    destination = tmp_path / "summary" / "simulation_summary.json"
    assert safe_json_dump(summary, destination)
    assert json.loads(destination.read_text()) == summary
    assert pickle.dumps(saved, protocol=5) == caller_bytes
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash


def test_public_saved_summary_empty_episodes_and_missing_metrics_are_truthful():
    assert generate_simulation_summary([], {}) == {
        "total_episodes": 0,
        "successful_episodes": 0,
        "total_steps": 0,
        "average_reward": 0.0,
        "average_episode_length": 0.0,
        "success_rate": 0.0,
        "episode_statistics": [],
    }
    summary = generate_simulation_summary([{"beliefs": [None, []]}], {})
    assert summary["total_episodes"] == 1
    assert summary["successful_episodes"] == summary["total_steps"] == 0
    assert summary["average_reward"] == summary["average_episode_length"] == 0.0
    assert summary["episode_statistics"][0] == {
        "episode": 0,
        "total_reward": 0.0,
        "episode_length": 0,
        "mean_belief_entropy": 0.0,
        "final_state": None,
        "success": False,
    }


@pytest.mark.parametrize("with_matrices", [False, True])
def test_public_saved_results_reopen_json_pickle_and_exact_model_arrays(
    tmp_path, with_matrices
):
    source, saved = saved_episodes(tmp_path)
    source_bytes = source.read_bytes()
    matrices = {"A": np.array([[0.75, 0.25], [0.25, 0.75]])} if with_matrices else None
    before = pickle.dumps((saved, matrices), protocol=5)
    destination = tmp_path / "saved"
    flags = save_simulation_results(
        saved["traces"], saved["metrics"], saved["config"], matrices, destination
    )
    assert flags == {
        "config": True,
        "metrics": True,
        "traces_pickle": True,
        "traces_json": True,
        "matrices": True if with_matrices else None,
    }
    for filename, key in (
        ("simulation_config.json", "config"),
        ("performance_metrics.json", "metrics"),
        ("simulation_traces.json", "traces"),
    ):
        artifact = destination / filename
        artifact_bytes = artifact.read_bytes()
        assert json.loads(artifact_bytes) == saved[key]
        assert artifact.read_bytes() == artifact_bytes
    with (destination / "simulation_traces.pkl").open("rb") as stream:
        assert pickle.load(stream) == saved["traces"]
    if with_matrices:
        with (destination / "model_matrices.pkl").open("rb") as stream:
            reopened = pickle.load(stream)
        assert set(reopened) == {"A"}
        np.testing.assert_array_equal(reopened["A"], [[0.75, 0.25], [0.25, 0.75]])
    else:
        assert not (destination / "model_matrices.pkl").exists()
    assert source.read_bytes() == source_bytes
    assert pickle.dumps((saved, matrices), protocol=5) == before


@pytest.mark.parametrize(
    "filename,role",
    [
        ("simulation_config.json", "config"),
        ("performance_metrics.json", "metrics"),
        ("simulation_traces.pkl", "traces_pickle"),
        ("simulation_traces.json", "traces_json"),
        ("model_matrices.pkl", "matrices"),
    ],
)
def test_public_saved_result_flags_report_actual_blocked_artifact_and_keep_siblings(
    tmp_path, caplog, filename, role
):
    source, saved = saved_episodes(tmp_path)
    source_bytes = source.read_bytes()
    destination = tmp_path / "saved"
    collision = destination / filename
    collision.mkdir(parents=True)
    marker = collision / "caller_note.txt"
    marker.write_bytes(b"caller-owned artifact collision\n")
    flags = save_simulation_results(
        saved["traces"], saved["metrics"], saved["config"], {"A": [[1.0]]}, destination
    )
    assert flags[role] is False
    assert all(status is True for key, status in flags.items() if key != role)
    assert marker.read_bytes() == b"caller-owned artifact collision\n"
    assert filename in caplog.text
    assert "Failed to save" in caplog.text
    assert source.read_bytes() == source_bytes
    if role != "traces_json":
        assert (
            json.loads((destination / "simulation_traces.json").read_text())
            == saved["traces"]
        )
    if role != "traces_pickle":
        with (destination / "simulation_traces.pkl").open("rb") as stream:
            assert pickle.load(stream) == saved["traces"]


def test_public_saved_results_blocked_parent_refuses_all_saves_without_replacing_it(
    tmp_path, caplog
):
    source, saved = saved_episodes(tmp_path)
    destination = tmp_path / "blocked"
    destination.write_bytes(b"caller-owned output path\n")
    source_bytes = source.read_bytes()
    flags = save_simulation_results(
        saved["traces"], saved["metrics"], saved["config"], {"A": [[1.0]]}, destination
    )
    assert flags == dict.fromkeys(
        ("config", "metrics", "traces_pickle", "traces_json", "matrices"), False
    )
    assert "Failed to save JSON" in caplog.text
    assert "Failed to save pickle" in caplog.text
    assert destination.read_bytes() == b"caller-owned output path\n"
    assert source.read_bytes() == source_bytes


def test_public_json_saved_numpy_types_are_reopened_as_declared_native_values(tmp_path):
    payload = {
        "samples": [np.int64(3), np.float32(0.25), np.bool_(False), np.str_("Ω")],
        "diagnostic_scalar": np.complex128(1.25 - 0.5j),
        "beliefs": np.array([[0.25, 0.75]]),
    }
    before = pickle.dumps(payload, protocol=5)
    destination = tmp_path / "native_types.json"
    assert safe_json_dump(payload, destination)
    reopened = json.loads(destination.read_text())
    assert reopened == {
        "samples": [3, 0.25, False, "Ω"],
        "diagnostic_scalar": {"real": 1.25, "imag": -0.5},
        "beliefs": [[0.25, 0.75]],
    }
    assert [type(value) for value in reopened["samples"]] == [int, float, bool, str]
    assert reopened["samples"][2] is False
    assert pickle.dumps(payload, protocol=5) == before


@pytest.mark.parametrize(
    "seconds,label",
    [(15.25, "15.25 seconds"), (120, "2.00 minutes"), (7200, "2.00 hours")],
)
def test_public_saved_duration_labels_retain_seconds_minutes_and_hours(seconds, label):
    assert format_duration(seconds) == label


@pytest.mark.parametrize("as_string", [False, True])
def test_public_timestamp_directory_keeps_caller_root_and_prefix(tmp_path, as_string):
    root = str(tmp_path) if as_string else tmp_path
    directory = create_output_directory_with_timestamp(root, prefix="saved_episode")
    assert directory.parent == tmp_path
    assert re.fullmatch(r"saved_episode_\d{8}_\d{6}", directory.name)
    assert directory.is_dir()
    assert list(directory.iterdir()) == []


def test_public_timestamp_directory_refuses_a_regular_file_root_without_replacing_it(
    tmp_path,
):
    root = tmp_path / "caller_file"
    root.write_bytes(b"retained caller bytes\n")
    with pytest.raises(OSError):
        create_output_directory_with_timestamp(root)
    assert root.read_bytes() == b"retained caller bytes\n"
