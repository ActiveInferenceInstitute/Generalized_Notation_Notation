"""Backend analysis consumers retain source data in real scientific artifacts."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure
from PIL import Image

from gnn.analysis.activeinference_jl.analyzer import (
    generate_analysis_from_logs as analyze_activeinference,
)
from gnn.analysis.jax.analyzer import generate_analysis_from_logs as analyze_jax


def record_plot_data(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Observe the external rendering seam while still writing native figures."""
    recorded = {}
    save = Figure.savefig

    def record(figure, filename, *args, **kwargs):
        recorded[Path(filename).name] = [
            {
                "title": axis.get_title(),
                "xlabel": axis.get_xlabel(),
                "ylabel": axis.get_ylabel(),
                "images": [
                    np.asarray(image.get_array()).copy() for image in axis.images
                ],
                "lines": [
                    (
                        np.asarray(line.get_xdata()).copy(),
                        np.asarray(line.get_ydata()).copy(),
                    )
                    for line in axis.lines
                ],
                "bars": [
                    patch.get_height()
                    for patch in axis.patches
                    if hasattr(patch, "get_height")
                ],
                "texts": [text.get_text() for text in axis.texts],
            }
            for axis in figure.axes
        ]
        return save(figure, filename, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", record)
    return recorded


@pytest.fixture
def saved_plot_data(monkeypatch: pytest.MonkeyPatch) -> dict:
    return record_plot_data(monkeypatch)


def assert_native_pngs(paths: list[str]) -> None:
    for path in paths:
        with Image.open(path) as image:
            assert image.format == "PNG"
            assert image.width > 100 and image.height > 100
            image.verify()


def test_jax_structured_analysis_preserves_beliefs_actions_and_efe(
    tmp_path: Path, saved_plot_data: dict
) -> None:
    execution = tmp_path / "execution"
    native_dir = execution / "sensor" / "jax" / "simulation_data"
    native_dir.mkdir(parents=True)
    beliefs = [[0.5, 0.5], [0.25, 0.75], [0.1, 0.9]]
    actions = [0, 1, 1]
    efe = [1.5, 0.75, -0.25]
    (native_dir / "simulation_results.json").write_text(
        json.dumps(
            {
                "simulation_trace": {
                    "beliefs": beliefs,
                    "actions": actions,
                    "efe_history": efe,
                    "belief_confidence": [0.5, 0.75, 0.9],
                },
                "model_parameters": {
                    "num_states": 2,
                    "num_observations": 3,
                    "num_actions": 2,
                },
                "validation": {"all_beliefs_valid": True, "actions_in_range": True},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "analysis"

    paths = analyze_jax(execution, output, verbose=True)

    assert {Path(path).name for path in paths} == {
        "sensor_jax_belief_evolution.png",
        "sensor_jax_efe.png",
        "sensor_jax_actions.png",
        "sensor_jax_confidence.png",
        "sensor_jax_dashboard.png",
    }
    assert_native_pngs(paths)
    belief_plot = saved_plot_data["sensor_jax_belief_evolution.png"]
    np.testing.assert_array_equal(belief_plot[0]["images"][0], np.asarray(beliefs).T)
    entropy_plot = next(axis for axis in belief_plot if "Entropy" in axis["ylabel"])
    entropy = [-sum(p * math.log(p) for p in row) for row in beliefs]
    np.testing.assert_allclose(entropy_plot["lines"][0][1], entropy, atol=3e-10)
    efe_plot = saved_plot_data["sensor_jax_efe.png"][0]
    np.testing.assert_array_equal(efe_plot["lines"][0][1], efe)
    np.testing.assert_array_equal(efe_plot["lines"][0][0], [0, 1, 2])
    action_plot = saved_plot_data["sensor_jax_actions.png"]
    assert action_plot[0]["bars"] == [1, 2]
    np.testing.assert_array_equal(action_plot[1]["lines"][0][1], actions)
    summary_text = "\n".join(
        text
        for axis in saved_plot_data["sensor_jax_dashboard.png"]
        for text in axis["texts"]
    )
    assert "States: 2" in summary_text
    assert "Observations: 3" in summary_text
    assert "Actions: 2" in summary_text
    assert "Timesteps: 3" in summary_text


def test_activeinference_current_schema_uses_joint_axis_values(
    tmp_path: Path, saved_plot_data: dict
) -> None:
    execution = tmp_path / "execution"
    native_dir = execution / "sensor" / "activeinference_jl" / "simulation_data"
    native_dir.mkdir(parents=True)
    beliefs = [[0.4, 0.6], [0.2, 0.8], [0.1, 0.9]]
    (native_dir / "sensor_simulation_results.json").write_text(
        json.dumps(
            {
                "schema_version": "activeinference_jl_simulation_v1",
                "beliefs_by_factor": {"joint_state": beliefs},
                "observations_by_modality": {"joint_observation": [2, 0, 1]},
                "actions_by_control_factor": {"joint_action": [1, 0, 1]},
                "expected_free_energy": [0.3, -0.5, -0.9],
            }
        ),
        encoding="utf-8",
    )

    paths = analyze_activeinference(execution, tmp_path / "analysis")

    assert_native_pngs(paths)
    np.testing.assert_array_equal(
        saved_plot_data["activeinference_jl_belief_heatmap.png"][0]["images"][0],
        np.asarray(beliefs).T,
    )
    for name, expected in (
        ("observations", [2, 0, 1]),
        ("actions", [1, 0, 1]),
        ("expected_free_energy", [0.3, -0.5, -0.9]),
    ):
        plot = saved_plot_data[f"activeinference_jl_{name}.png"][0]
        np.testing.assert_array_equal(plot["lines"][0][1], expected)
        np.testing.assert_array_equal(plot["lines"][0][0], [0, 1, 2])


def test_jax_execution_log_analysis_retains_full_logged_action_sequence(
    tmp_path: Path, saved_plot_data: dict
) -> None:
    execution = tmp_path / "execution"
    native_dir = execution / "sensor" / "jax" / "execution_logs"
    native_dir.mkdir(parents=True)
    (native_dir / "sensor_results.json").write_text(
        json.dumps(
            {
                "simulation_data": {
                    "actions": [1],
                    "free_energy": [0.75, -0.5],
                    "beliefs": [[0.5, 0.5], [0.1, 0.9]],
                    "raw_output": "\n".join(
                        [
                            "Actions taken: [1 0 1]",
                            "Final belief: [0.1 0.9]",
                            "Average EFE: -0.25",
                            "EFE for all actions: [0.75 -0.5]",
                            "A matrix shape: (3, 2)",
                            "B matrix shape: (2, 2, 2)",
                            "C vector shape: (3,)",
                            "D vector shape: (2,)",
                            "Number of states: 2",
                            "Number of observations: 3",
                            "Number of actions: 2",
                        ]
                    ),
                }
            }
        ),
        encoding="utf-8",
    )

    paths = analyze_jax(execution, tmp_path / "analysis")

    assert_native_pngs(paths)
    timeline = saved_plot_data["sensor_jax_action_timeline.png"][0]
    np.testing.assert_array_equal(timeline["lines"][0][1], [1, 0, 1])
    np.testing.assert_array_equal(timeline["lines"][0][0], [0, 1, 2])
    assert saved_plot_data["sensor_jax_action_dist.png"][0]["bars"] == [2, 1]
    np.testing.assert_array_equal(
        saved_plot_data["sensor_jax_efe_comparison.png"][0]["bars"], [0.75, -0.5]
    )
    model_text = "\n".join(saved_plot_data["sensor_jax_model_summary.png"][0]["texts"])
    assert "A: (3, 2)" in model_text
    assert "B: (2, 2, 2)" in model_text
    assert "Simulation: 3 steps" in model_text
    belief_plot = saved_plot_data["sensor_jax_beliefs.png"][0]
    np.testing.assert_array_equal(belief_plot["lines"][0][1], [0.5, 0.1])
    np.testing.assert_array_equal(belief_plot["lines"][1][1], [0.5, 0.9])


def test_activeinference_csv_and_julia_matrices_preserve_axes_and_calibration(
    tmp_path: Path, saved_plot_data: dict
) -> None:
    execution = tmp_path / "execution"
    native_dir = execution / "activeinference_outputs_sensor"
    native_dir.mkdir(parents=True)
    (native_dir / "simulation_results.csv").write_text(
        "# generated trace, zero-based timestep\n"
        "step,observation,action,belief_state_1\n"
        "0,2,1,0.5\n1,0,0,0.2\n2,1,1,0.1\n",
        encoding="utf-8",
    )
    (native_dir / "model_parameters.json").write_text(
        json.dumps(
            {
                "A_matrix": "[0.8 0.1; 0.1 0.6; 0.1 0.3]",
                "B_matrix": "[0.9 0.2; 0.1 0.8;;; 0.3 0.7; 0.7 0.3]",
                "C_vector": "[-0.5, 0.25, 1.0]",
                "D_vector": "[0.3, 0.7]",
                "E_vector": "[0.4, 0.6]",
            }
        ),
        encoding="utf-8",
    )

    paths = analyze_activeinference(execution, tmp_path / "analysis")

    assert_native_pngs(paths)
    trace = saved_plot_data["activeinference_jl_trace_reconstruction.png"]
    for axis, expected in zip(trace, ([2, 0, 1], [1, 0, 1], [0.5, 0.2, 0.1])):
        np.testing.assert_array_equal(axis["lines"][0][0], [0, 1, 2])
        np.testing.assert_array_equal(axis["lines"][0][1], expected)
    matrices = saved_plot_data["activeinference_jl_model_matrices.png"]
    panels = [axis for axis in matrices if axis["images"]]
    np.testing.assert_array_equal(
        panels[0]["images"][0], [[0.8, 0.1], [0.1, 0.6], [0.1, 0.3]]
    )
    np.testing.assert_array_equal(panels[1]["images"][0], [[0.9, 0.2], [0.1, 0.8]])
    np.testing.assert_array_equal(panels[2]["images"][0], [[0.3, 0.7], [0.7, 0.3]])
    np.testing.assert_array_equal(panels[3]["images"][0], [[-0.5, 0.25, 1.0]])
    np.testing.assert_array_equal(panels[4]["images"][0], [[0.3, 0.7]])
    np.testing.assert_array_equal(panels[5]["images"][0], [[0.4, 0.6]])


def test_activeinference_malformed_csv_row_cannot_shift_valid_trace_alignment(
    tmp_path: Path, saved_plot_data: dict, caplog: pytest.LogCaptureFixture
) -> None:
    execution = tmp_path / "execution"
    native_dir = execution / "activeinference_outputs_sensor"
    native_dir.mkdir(parents=True)
    (native_dir / "simulation_results.csv").write_text(
        "# source trace\n0,2,1,0.5\n1,not-a-symbol,0,0.2\n2,1,1,0.1\n",
        encoding="utf-8",
    )

    paths = analyze_activeinference(execution, tmp_path / "analysis")

    assert [Path(path).name for path in paths] == [
        "activeinference_jl_trace_reconstruction.png"
    ]
    assert_native_pngs(paths)
    trace = saved_plot_data["activeinference_jl_trace_reconstruction.png"]
    for axis, expected in zip(trace, ([2, 1], [1, 1], [0.5, 0.1])):
        np.testing.assert_array_equal(axis["lines"][0][0], [0, 2])
        np.testing.assert_array_equal(axis["lines"][0][1], expected)
    assert "malformed CSV row" in caplog.text
