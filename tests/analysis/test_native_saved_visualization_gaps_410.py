"""Real file consumers preserve authored diagnostics and reject unavailable data.

The scaling fixture contains explicitly authored runtime numbers for testing
the report algebra. It is not a backend run or performance measurement.
"""

import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from gnn.analysis import (
    extract_activeinference_jl_data,
    extract_rxinfer_data,
    generate_action_analysis,
    generate_belief_heatmaps,
    generate_free_energy_plots,
    generate_observation_analysis,
    visualize_all_framework_outputs,
)
from gnn.analysis.framework_comparison import (
    analyze_framework_outputs,
    generate_framework_comparison_report,
)
from gnn.analysis.viz_plots import generate_vfe_vs_efe_plot
from gnn.integration import run_meta_analysis
from tests.advanced_visualization.test_native_network_consumers_410 import (
    figures as figures,
)
from tests.visualization.test_native_matrix_views_410 import native_png


@pytest.fixture(autouse=True)
def close_native_figures():
    yield
    plt.close("all")


def saved_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def tree_bytes(root):
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("subdirectory", ["", "simulation_data"])
def test_public_saved_legacy_rxinfer_reader_preserves_signed_and_zero_values(
    tmp_path, subdirectory
):
    implementation = tmp_path / "rxinfer"
    payload = {
        "beliefs": [[1, 0], [0.25, 0.75]],
        "true_states": [0, 1],
        "observations": [0, 4],
        "actions": [0, 1],
        "efe_history": [[-4, 0], [-2, 1]],
        "action_probabilities": [[1, 0], [0.2, 0.8]],
    }
    saved_json(implementation / subdirectory / "simulation_results.json", payload)
    before = tree_bytes(implementation)
    result = extract_rxinfer_data({"implementation_directory": str(implementation)})
    assert result == {
        "beliefs": payload["beliefs"],
        "true_states": [0, 1],
        "observations": [0, 4],
        "actions": [0, 1],
        "free_energy": [[-4, 0], [-2, 1]],
        "action_probabilities": payload["action_probabilities"],
        "posterior": [],
        "inference_data": [],
    }
    assert tree_bytes(implementation) == before


def test_public_saved_rxinfer_malformed_json_retains_explicit_read_error(tmp_path):
    implementation = tmp_path / "rxinfer"
    implementation.mkdir()
    path = implementation / "simulation_results.json"
    path.write_text('{"beliefs":', encoding="utf-8")
    before = tree_bytes(implementation)
    result = extract_rxinfer_data({"implementation_directory": str(implementation)})
    assert result["beliefs"] == [] and result["free_energy"] == []
    assert "Expecting value" in result["extraction_error"]
    assert tree_bytes(implementation) == before


@pytest.mark.parametrize("timestamped", [False, True])
def test_public_saved_activeinference_csv_reader_preserves_native_columns(
    tmp_path, timestamped
):
    implementation = tmp_path / "activeinference_jl"
    directory = (
        implementation / "activeinference_outputs_fixture"
        if timestamped
        else implementation
    )
    directory.mkdir(parents=True)
    csv = directory / "simulation_results.csv"
    csv.write_text(
        "# Authored saved CSV control, no Julia execution\nstep,observation,action,q0,q1\n"
        "0,0,1,1,0\n1,4,0,0.25,0.75\n2,0,1,0.5,0.5\n",
        encoding="utf-8",
    )
    before = tree_bytes(implementation)
    result = extract_activeinference_jl_data(
        {"implementation_directory": str(implementation)}
    )
    assert result["beliefs"] == [[1, 0], [0.25, 0.75], [0.5, 0.5]]
    assert result["observations"] == [0, 4, 0] and result["actions"] == [1, 0, 1]
    assert result["traces"] == [
        {"step": 0, "observation": 0, "action": 1},
        {"step": 1, "observation": 4, "action": 0},
        {"step": 2, "observation": 0, "action": 1},
    ]
    assert result["num_timesteps"] == 3
    assert tree_bytes(implementation) == before


def test_saved_activeinference_csv_comparison_writes_exact_data_and_exclusion(tmp_path):
    execution = tmp_path / "12_execute_output"
    implementation = execution / "authored_csv" / "activeinference_jl"
    implementation.mkdir(parents=True)
    csv = implementation / "simulation_data" / "simulation_results.csv"
    csv.parent.mkdir()
    csv.write_text(
        "# saved categorical fixture\nstep,o,u,q0,q1\n0,0,1,1,0\n1,4,0,0.25,0.75\n",
        encoding="utf-8",
    )
    saved_json(
        execution / "summaries" / "execution_summary.json",
        {
            "execution_details": [
                {
                    "framework": "activeinference_jl",
                    "model_name": "authored_csv",
                    "success": True,
                    "implementation_directory": str(implementation),
                }
            ],
        },
    )
    before = tree_bytes(execution)
    data = analyze_framework_outputs(execution)
    framework = data["frameworks"]["activeinference_jl"]
    assert framework["beliefs"] == [[1, 0], [0.25, 0.75]]
    assert framework["actions"] == [1, 0] and framework["observations"] == [0, 4]
    assert framework["data_source"] == str(csv)
    assert framework["num_timesteps"] == 2
    assert framework["validation"] == {
        "all_beliefs_valid": True,
        "beliefs_sum_to_one": True,
        "actions_in_range": True,
    }
    report = Path(generate_framework_comparison_report(data, tmp_path / "reports"))
    reopened = json.loads(
        report.with_name("framework_comparison_data.json").read_text()
    )
    assert reopened == data
    content = report.read_text()
    assert "beliefs=2, actions=2, observations=2, free_energy=0" in content
    assert str(csv) in content
    assert "Unavailable Scientific Comparisons" in content
    assert "missing declared model_id" in content
    assert tree_bytes(execution) == before


def test_saved_gaussian_comparison_preserves_mean_covariance_and_source(tmp_path):
    execution = tmp_path / "12_execute_output"
    implementation = execution / "authored_gaussian" / "rxinfer"
    payload = {
        "model_name": "authored_gaussian",
        "model_id": "fixtures/gaussian",
        "model_kind": "continuous",
        "source_sha256": hashlib.sha256(
            b"authored Gaussian saved-data control"
        ).hexdigest(),
        "source_relative_path": "fixtures/gaussian.md",
        "inference_mode": "kalman_filter",
        "dtype": "float64",
        "belief_axes": ["timestep", "state"],
        "beliefs": [[-2, 0], [-1, 3]],
        "posterior_cov": [[[4, 0], [0, 9]], [[1, 0], [0, 4]]],
        "observations_continuous": [[0], [-1]],
        "num_timesteps": 2,
    }
    result_path = saved_json(implementation / "simulation_results.json", payload)
    saved_json(
        execution / "summaries" / "execution_summary.json",
        {
            "execution_details": [
                {
                    "framework": "rxinfer",
                    "model_name": "authored_gaussian",
                    "success": True,
                    "output_file": str(result_path),
                }
            ]
        },
    )
    before = tree_bytes(execution)
    data = analyze_framework_outputs(execution)
    framework = data["frameworks"]["rxinfer"]
    assert framework["data_source"] == str(result_path)
    assert framework["beliefs"] == [[-2, 0], [-1, 3]]
    stats = data["comparisons"]["simulation_statistics"]["rxinfer"]
    report = Path(generate_framework_comparison_report(data, tmp_path / "reports"))
    content = report.read_text()
    assert "fixtures/gaussian | joint | continuous | 2 | N/A | 2.0000" in content
    assert "mean_confidence" not in stats
    assert (
        stats["unavailable_metrics"]["mean_confidence"]
        == "Gaussian means are not categorical probabilities"
    )
    assert (
        json.loads(report.with_name("framework_comparison_data.json").read_text())
        == data
    )
    assert tree_bytes(execution) == before


def test_saved_visualization_filter_ignores_malformed_and_unselected_siblings(
    tmp_path, figures
):
    execution = tmp_path / "12_execute_output"
    for name, framework in [("chosen", "pytorch"), ("other", "numpyro")]:
        saved_json(
            execution / name / framework / "execution_logs" / "fixture_results.json",
            {
                "framework": framework,
                "model_name": f"Display {name}",
                "simulation_data": {"observations": [0, 0, 4]},
            },
        )
    broken = execution / "broken_results.json"
    broken.write_text('{"framework":', encoding="utf-8")
    before = tree_bytes(execution)
    output = tmp_path / "plots"
    files = visualize_all_framework_outputs(
        execution,
        output,
        allowed_frameworks={"pytorch"},
        allowed_model_names={"chosen"},
    )
    assert len(files) == 1
    path = Path(files[0])
    assert path.name == "Display chosen_pytorch_observations.png"
    native_png(path)
    assert [bar.get_height() for bar in figures[path.name].axes[0].patches] == [2, 1]
    assert tree_bytes(execution) == before


def test_native_many_policy_energy_heatmap_preserves_signed_zero_scores(
    tmp_path, figures
):
    scores = [[-12 + i for i in range(12)], [0] * 12, [12 - i for i in range(12)]]
    before = deepcopy(scores)
    path = tmp_path / "authored_many_policy.png"
    assert generate_free_energy_plots(
        scores, path, units="declared diagnostic units"
    ) == str(path)
    native_png(path)
    axis = figures[path.name].axes[0]
    np.testing.assert_array_equal(axis.images[0].get_array(), np.array(scores).T)
    np.testing.assert_array_equal(axis.lines[0].get_ydata(), [-12, 0, 1])
    assert axis.lines[0].get_label() == "Minimum reported policy score"
    assert scores == before


def test_native_single_action_has_no_invented_transition(tmp_path, figures):
    path = tmp_path / "one_action.png"
    assert generate_action_analysis([0], path) == str(path)
    native_png(path)
    axes = figures[path.name].axes
    assert [bar.get_height() for bar in axes[0].patches] == [1]
    np.testing.assert_array_equal(axes[1].lines[0].get_ydata(), [0])
    assert len(axes[2].images) == 0
    assert [text.get_text() for text in axes[2].texts] == [
        "Need > 1 action\nfor transitions"
    ]


@pytest.mark.parametrize(
    "builder,values,error",
    [
        (generate_action_analysis, [], "No actions provided"),
        (generate_observation_analysis, [], "No observations provided"),
        (generate_observation_analysis, list(range(500)), "limit is 64,000,000"),
        (generate_belief_heatmaps, [[1, 0]], "Need at least 2 timesteps"),
        (generate_free_energy_plots, [], "No free energy values provided"),
        (generate_free_energy_plots, [float("nan")], "finite, nonempty"),
    ],
)
def test_public_native_plot_refusal_preserves_callers_existing_file(
    tmp_path, builder, values, error
):
    path = tmp_path / "caller_owned.png"
    path.write_bytes(b"caller-owned existing bytes")
    before = tree_bytes(tmp_path)
    with pytest.raises(ValueError, match=error):
        builder(values, path)
    assert tree_bytes(tmp_path) == before


@pytest.mark.parametrize(
    "vfe,efe,error",
    [
        ([], [1], "Need both VFE and EFE"),
        ([0], [float("inf")], "EFE must contain finite"),
    ],
)
def test_public_native_paired_energy_refuses_unavailable_samples(
    tmp_path, vfe, efe, error
):
    path = tmp_path / "pair.png"
    with pytest.raises(ValueError, match=error):
        generate_vfe_vs_efe_plot(vfe, efe, path)
    assert not path.exists() and not path.with_suffix(".conventions.json").exists()


def test_public_saved_scaling_report_uses_authored_fit_without_invented_cause(tmp_path):
    execution = tmp_path / "12_execute_output"
    # Explicitly authored algebra controls, not measured execution times. All
    # observations lie on runtime=(N/2)^2, giving slope 2 and R² 1 independently.
    details = [
        {
            "model_name": f"authored_diagnostic_N{n}_T2",
            "framework": "pymdp",
            "success": True,
            "execution_time": seconds,
            "fixture_provenance": "authored report-algebra control; not an execution measurement",
        }
        for n, seconds in [(2, 1), (4, 4), (8, 16)]
    ]
    saved_json(
        execution / "summaries" / "execution_summary.json",
        {"execution_details": details},
    )
    before = tree_bytes(execution)
    output = tmp_path / "meta_report"
    result = run_meta_analysis(execution, output)
    assert result is not None and result["records"] == 3
    assert result["plots"]
    plots = [Path(path) for path in result["plots"] if Path(path).suffix == ".png"]
    assert {
        "runtime_scaling_curves.png",
        "framework_runtime_comparison.png",
        "time_per_step.png",
        "throughput_vs_n.png",
    } <= {plot.name for plot in plots}
    for plot in plots:
        native_png(plot)
    with (output / "visualizations" / "data" / "sweep_data.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [
        (int(row["num_states"]), float(row["execution_time_s"])) for row in rows
    ] == [(2, 1), (4, 4), (8, 16)]
    stats = json.loads(Path(result["statistics_json"]).read_text())
    fit = stats["loglog_runtime_vs_n_by_T"]["2"]
    assert fit["slope"] == pytest.approx(2.0)
    assert fit["r_squared"] == pytest.approx(1.0)
    content = Path(result["report"]).read_text()
    assert "O(N^2.00)" in content and "$R^2$=1.000" in content
    assert "N=2.0→8.0: 1.0s → 16.0s" in content
    assert "α ≈ 0.1–0.3" not in content
    assert "β ≈ 0.5–0.7" not in content
    assert "N ≤ 128" not in content
    assert "dominated by constant **JIT compilation overhead**" not in content
    assert "descriptive" in content.lower() and "causal" in content.lower()
    assert tree_bytes(execution) == before
