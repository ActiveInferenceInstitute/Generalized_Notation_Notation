"""Native visualization consumers expose verifiable data and export artifacts."""

import csv
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pytest
from matplotlib import pyplot as plt

from tests.analysis.test_analysis_consumer_behavior_410 import (
    assert_native_pngs,
    record_plot_data,
)


@pytest.fixture
def saved_plot_data(monkeypatch: pytest.MonkeyPatch) -> dict:
    return record_plot_data(monkeypatch)


@pytest.fixture(autouse=True)
def isolated_plot_style():
    """Keep the consumer's rendering style local to each acceptance check."""
    with plt.rc_context():
        yield


def test_simulation_visualization_suite_retains_numeric_trace_evidence(
    tmp_path: Path, saved_plot_data: dict
) -> None:
    from gnn.render.visualization_suite import VisualizationSuite

    # Two nonconstant traces are enough to establish slopes, correlations,
    # empirical distributions and source-relative temporal identity.
    signal = [1.0, 2.0, 3.0, 4.0]
    error = [4.0, 3.0, 2.0, 1.0]
    traces = {"signal": signal, "error": error}
    suite = VisualizationSuite(tmp_path, "measured_sensor")

    paths = suite.create_comprehensive_suite(
        {"traces": traces, "summary": {"source": "sensor-17", "timesteps": 4}}
    )

    assert_native_pngs([str(path) for path in paths])
    axes = [axis for plots in saved_plot_data.values() for axis in plots]
    by_title = {axis["title"]: axis for axis in axes}
    primary = by_title["Primary Time Series"]
    np.testing.assert_array_equal(primary["lines"][0][1], signal)
    np.testing.assert_array_equal(primary["lines"][1][1], error)
    cumulative = by_title["Cumulative Evolution"]
    np.testing.assert_array_equal(cumulative["lines"][0][1], [1, 3, 6, 10])
    np.testing.assert_array_equal(cumulative["lines"][1][1], [4, 7, 9, 10])
    changes = by_title["Rate of Change"]
    np.testing.assert_array_equal(changes["lines"][0][1], [1, 1, 1])
    np.testing.assert_array_equal(changes["lines"][1][1], [-1, -1, -1])
    empirical = by_title["Empirical Cumulative Distribution"]
    np.testing.assert_array_equal(empirical["lines"][1][0], [1, 2, 3, 4])
    np.testing.assert_array_equal(empirical["lines"][1][1], [0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(
        by_title["Correlation Matrix"]["images"][0], [[1, -1], [-1, 1]]
    )
    np.testing.assert_allclose(
        by_title["Performance Trends (Slope Analysis)"]["bars"], [1, -1]
    )
    assert by_title["Value Ranges"]["bars"] == [3, 3]
    np.testing.assert_allclose(by_title["Variance Comparison"]["bars"], [1.25, 1.25])


def test_simulation_exports_preserve_arrays_and_csv_axis_values(tmp_path: Path) -> None:
    from gnn.render.visualization_suite import ComprehensiveDataExporter

    exporter = ComprehensiveDataExporter(tmp_path, "measured_sensor")
    source = {
        "traces": {"action": [1, 0], "belief": [[0.25, 0.75], [0.1, 0.9]]},
        "parameters": {"A": np.array([[0.8, 0.2], [0.2, 0.8]])},
        "seed": np.int64(7),
        "calibration": complex(2, -3),
    }

    paths = exporter.export_all_formats(source)

    data = json.loads(
        next(
            path for path in paths if "_data_" in path.name and path.suffix == ".json"
        ).read_text()
    )
    assert data["parameters"]["A"] == [[0.8, 0.2], [0.2, 0.8]]
    assert data["seed"] == 7
    assert data["calibration"] == {"real": 2.0, "imag": -3.0}
    csv_path = next(path for path in paths if path.suffix == ".csv")
    with csv_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows == [
        {"step": "1", "action": "1", "belief_0": "0.25", "belief_1": "0.75"},
        {"step": "2", "action": "0", "belief_0": "0.1", "belief_1": "0.9"},
    ]
    metadata = json.loads(
        next(path for path in paths if "_metadata_" in path.name).read_text()
    )
    assert metadata["simulation_name"] == "measured_sensor"
    assert set(metadata["exported_files"]) == {
        path.name for path in paths if "_metadata_" not in path.name
    }
    hdf5_path = next(path for path in paths if path.suffix == ".h5")
    with h5py.File(hdf5_path, "r") as artifact:
        np.testing.assert_array_equal(
            artifact["parameters/A"][()], source["parameters"]["A"]
        )
        np.testing.assert_array_equal(
            artifact["traces/belief"][()], source["traces"]["belief"]
        )
        np.testing.assert_array_equal(artifact["traces/action"][()], [1, 0])
        assert artifact["calibration"][()] == complex(2, -3)


def test_parsed_model_matrix_artifacts_keep_action_axes_and_every_csv(
    tmp_path: Path, saved_plot_data: dict
) -> None:
    from gnn.visualization import generate_matrix_visualizations

    likelihood = [[0.8, 0.2, 0.5], [0.2, 0.8, 0.5]]
    action_zero = [[0.7, 0.1, 0.2], [0.2, 0.6, 0.3], [0.1, 0.3, 0.5]]
    action_one = [[0.2, 0.4, 0.6], [0.3, 0.4, 0.1], [0.5, 0.2, 0.3]]
    transition = np.stack([action_zero, action_one], axis=2)
    matrices = {
        "A": likelihood,
        "B": transition.tolist(),
        "C": [1, -2],
        "D": [0.1, 0.3, 0.6],
    }

    paths = generate_matrix_visualizations(
        {
            "parameters": [
                {"name": name, "value": value} for name, value in matrices.items()
            ]
        },
        tmp_path,
        "asymmetric_sensor",
    )

    assert {Path(path).name for path in paths} == {
        "matrix_analysis.png",
        "matrix_statistics.png",
        "pomdp_transition_analysis.png",
    }
    assert_native_pngs(paths)
    overview = saved_plot_data["matrix_analysis.png"]
    a_panel = next(axis for axis in overview if axis["title"].startswith("A ("))
    b_panel = next(axis for axis in overview if axis["title"].startswith("B ("))
    np.testing.assert_array_equal(a_panel["meshes"][0].reshape(2, 3), likelihood)
    np.testing.assert_array_equal(b_panel["meshes"][0].reshape(3, 3), action_zero)
    assert next(axis for axis in overview if axis["title"] == "C (Vector)")["bars"] == [
        1,
        -2,
    ]
    np.testing.assert_array_equal(
        next(axis for axis in overview if axis["title"] == "D (Vector)")["bars"],
        [0.1, 0.3, 0.6],
    )
    statistics = {
        axis["title"]: axis for axis in saved_plot_data["matrix_statistics.png"]
    }
    assert statistics["Matrix Sizes"]["bars"] == [6, 18, 2, 3]
    np.testing.assert_allclose(
        statistics["Matrix Means"]["bars"], [0.5, 1 / 3, -0.5, 1 / 3]
    )
    transitions = {
        axis["title"]: axis for axis in saved_plot_data["pomdp_transition_analysis.png"]
    }
    np.testing.assert_array_equal(
        transitions["Action 0 Transition Matrix"]["images"][0], action_zero
    )
    np.testing.assert_array_equal(
        transitions["Action 1 Transition Matrix"]["images"][0], action_one
    )
    entropy = []
    for plane in (action_zero, action_one):
        columns = zip(*plane)
        entropy.append(
            sum(-sum(p * math.log(p) for p in column) for column in columns) / 3
        )
    entropy_plot = transitions["Transition Entropy by Action"]
    np.testing.assert_allclose(entropy_plot["bars"], entropy)
    assert entropy_plot["ylabel"] == "Mean Entropy (nats)"

    exported = {}
    for path in (tmp_path / "asymmetric_sensor").glob("*.csv"):
        with path.open(newline="") as stream:
            rows = list(csv.reader(stream))
        name = rows[0][0].removeprefix("Matrix: ")
        assert name not in exported, (
            "Every matrix needs an independent source-identified export"
        )
        exported[name] = rows
    assert set(exported) == set(matrices)
    a_rows = [row for row in exported["A"] if row and row[0].startswith("Row ")]
    np.testing.assert_array_equal(
        [[float(value) for value in row[1:]] for row in a_rows], likelihood
    )
    for name in ("C", "D"):
        np.testing.assert_array_equal(
            [float(value) for value in exported[name][-1]], matrices[name]
        )
    b_rows = exported["B"]
    for action, plane in enumerate((action_zero, action_one)):
        start = b_rows.index([f"Action slice {action}"]) + 2
        np.testing.assert_array_equal(
            [[float(value) for value in row[1:]] for row in b_rows[start : start + 3]],
            plane,
        )
