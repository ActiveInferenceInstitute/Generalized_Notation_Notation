"""cpomdp results reach the analysis step: path inference and the analyzer."""

from __future__ import annotations

import json
from pathlib import Path

from gnn.analysis.cpomdp import generate_analysis_from_logs
from gnn.analysis.framework_common import framework_from_path, model_name_from_path

STEPS = 5
N_POLICIES = 9


def _write_result(root: Path, model: str) -> Path:
    path = root / model / "cpomdp" / "simulation_data" / "simulation_results.json"
    path.parent.mkdir(parents=True)
    payload = {
        "model_name": model,
        "framework": "cpomdp",
        "model_kind": "continuous",
        "beliefs": [[0.1 * t, 0.2 * t] for t in range(STEPS)],
        "efe_history": [
            [float(t + k) for k in range(N_POLICIES)] for t in range(STEPS)
        ],
        "actions": [],
        "observations": [],
        "validation": {"all_valid": True},
    }
    path.write_text(json.dumps(payload))
    return path


def test_cpomdp_result_paths_name_their_model_and_framework(tmp_path: Path) -> None:
    path = _write_result(tmp_path, "continuous_navigation")
    assert framework_from_path(path) == "cpomdp"
    assert model_name_from_path(path) == "continuous_navigation"


def test_analyzer_writes_trajectory_efe_plot_and_summary(tmp_path: Path) -> None:
    _write_result(tmp_path / "results", "navigator")
    out = tmp_path / "analysis"

    generated = generate_analysis_from_logs(tmp_path / "results", out)

    summary_path = out / "navigator" / "cpomdp_analysis.json"
    assert generated == [str(summary_path)]
    summary = json.loads(summary_path.read_text())
    assert summary["framework"] == "cpomdp"
    assert summary["num_states"] == 2
    assert summary["plots_generated"] is True
    assert (out / "navigator" / "belief_trajectory.png").is_file()
    assert (out / "navigator" / "efe_history.png").is_file()
