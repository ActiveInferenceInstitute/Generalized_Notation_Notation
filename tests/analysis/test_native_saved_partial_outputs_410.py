"""Public consumers of authored saved partial results; no backend execution."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure
from PIL import Image

from gnn.analysis import (
    analyze_execution_results,
    visualize_all_framework_outputs,
    viz_plots,
)


@pytest.fixture
def native_figures(monkeypatch):
    observed = {}
    original = Figure.savefig

    def forward(figure, filename, *args, **kwargs):
        result = original(figure, filename, *args, **kwargs)
        observed[Path(filename).name] = figure
        return result

    monkeypatch.setattr(Figure, "savefig", forward)
    return observed


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def _saved_partial(tmp_path, *, energy_source="metrics", energy=None):
    """Authored saved fixture: field labels are declarations, never measurements."""
    execution = tmp_path / "12_execute_output"
    implementation = execution / "finite" / "numpyro"
    energy = [-6.0, 0.0, 6.0] if energy is None else energy
    log = {
        "framework": "NumPyro",
        "model_name": "finite",
        "implementation_directory": str(implementation),
        "policy_posterior": [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
    }
    if energy_source == "metrics":
        log["metrics"] = {"expected_free_energy": energy}
    else:
        log["simulation_trace"] = {"efe_history": energy}
    structured = _write(implementation / "execution_logs" / "finite_results.json", log)
    simulation = _write(
        implementation / "simulation_data" / "saved_trace.json",
        {
            "beliefs": [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
            "actions": [1, 0, 1],
            "observations": [0, 1, 0],
            "expected_free_energy": energy,
            "expected_free_energy_convention": "Authored signed saved scalar EFE diagnostic",
            "units": {"expected_free_energy": "authored units"},
        },
    )
    malformed = implementation / "simulation_data" / "unreadable.json"
    malformed.write_text("{not-json")
    return execution, structured, simulation, malformed


def _custody(paths):
    return {path: path.read_bytes() for path in paths}


@pytest.mark.parametrize("energy_source", ["metrics", "simulation_trace"])
def test_saved_partial_public_analysis_retains_declared_signed_zero_efe_and_policy(
    tmp_path, energy_source
):
    execution, *sources = _saved_partial(tmp_path, energy_source=energy_source)
    original = _custody(sources)
    data = analyze_execution_results(execution, allowed_frameworks={"numpyro"})
    assert set(data["framework_results"]) == {"numpyro"}
    assert data["cross_framework_comparison"] == {}
    framework = data["framework_results"]["numpyro"]
    assert framework["result_count"] == 1
    analyses = framework["analyses"]
    energy = next(item for item in analyses if "free_energy_values" in item)
    assert energy["free_energy_values"] == [-6.0, 0.0, 6.0]
    assert energy["model_name"] == "finite"
    assert energy["framework"] == "numpyro"
    assert energy["free_energy_count"] == 3
    assert energy["mean_free_energy"] == 0.0
    assert energy["std_free_energy"] == pytest.approx(math.sqrt(24.0))
    assert energy["min_free_energy"] == -6.0
    assert energy["max_free_energy"] == 6.0
    assert energy["free_energy_trend"] == pytest.approx(6.0)
    assert not energy["free_energy_decreasing"]
    policy = next(item for item in analyses if "policy_entropy" in item)
    assert policy["policy_count"] == 3
    assert policy["policy_entropy"] == pytest.approx([0.0, math.log(2), 0.0])
    assert policy["policy_stability"]["entropy_mean"] == pytest.approx(math.log(2) / 3)
    assert policy["policy_stability"]["entropy_std"] == pytest.approx(
        math.sqrt(2) * math.log(2) / 3
    )
    assert _custody(sources) == original


def test_saved_partial_public_native_plots_recover_healthy_source_and_filter_sibling(
    tmp_path, native_figures
):
    execution, *sources = _saved_partial(tmp_path)
    sibling = _write(
        execution / "other" / "numpyro" / "other_results.json",
        {
            "framework": "NumPyro",
            "model_name": "other",
            "simulation_data": {"observations": [99]},
        },
    )
    original = _custody([*sources, sibling])
    output = tmp_path / "16_analysis_output" / "cross_framework"
    files = visualize_all_framework_outputs(
        execution,
        output,
        allowed_frameworks={"numpyro"},
        allowed_model_names={"finite"},
        generate_animations=False,
    )
    assert {Path(item).name for item in files} == {
        "finite_numpyro_free_energy.png",
        "finite_numpyro_observations.png",
    }
    for item in files:
        with Image.open(item) as image:
            image.load()
            assert image.format == "PNG"
            assert image.width > 100 and image.height > 100
    energy = native_figures["finite_numpyro_free_energy.png"].axes[0]
    np.testing.assert_array_equal(energy.lines[0].get_ydata(), [-6.0, 0.0, 6.0])
    assert any(
        "Authored signed saved scalar EFE diagnostic" in text.get_text()
        for text in native_figures["finite_numpyro_free_energy.png"].texts
    )
    observations = native_figures["finite_numpyro_observations.png"].axes[0]
    assert [bar.get_height() for bar in observations.patches] == [2, 1]
    assert _custody([*sources, sibling]) == original
    assert not list(output.parent.rglob("other*.png"))


@pytest.mark.parametrize(
    "energy", [[0.0, float("inf"), -1.0], [[1.0, 0.0], [0.0, 1.0]]]
)
def test_public_saved_analysis_refuses_nonfinite_or_unreduced_energy(tmp_path, energy):
    execution, *sources = _saved_partial(tmp_path, energy=energy)
    original = _custody(sources)
    data = analyze_execution_results(execution, allowed_frameworks={"numpyro"})
    analyses = data["framework_results"]["numpyro"]["analyses"]
    rejected = next(item for item in analyses if item.get("status") == "failed")
    assert "finite scalar series" in rejected["error"]
    assert "mean_free_energy" not in rejected
    assert any("policy_entropy" in item for item in analyses)
    assert _custody(sources) == original


def test_public_saved_analysis_refuses_invalid_declared_policy_without_normalizing(
    tmp_path,
):
    execution, structured, simulation, malformed = _saved_partial(tmp_path)
    payload = json.loads(structured.read_text())
    payload["policy_posterior"] = [[0.4, 0.4]]
    structured.write_text(json.dumps(payload))
    sources = [structured, simulation, malformed]
    original = _custody(sources)
    data = analyze_execution_results(execution, allowed_frameworks={"numpyro"})
    analyses = data["framework_results"]["numpyro"]["analyses"]
    rejected = next(item for item in analyses if item.get("status") == "failed")
    assert "probabilit" in rejected["error"].lower()
    assert "policy_entropy" not in rejected
    assert any(item.get("free_energy_values") == [-6.0, 0.0, 6.0] for item in analyses)
    assert _custody(sources) == original


def test_public_saved_visualization_does_not_claim_refused_energy_artifact(tmp_path):
    execution, *sources = _saved_partial(tmp_path, energy=[0.0, float("inf"), -1.0])
    original = _custody(sources)
    output = tmp_path / "16_analysis_output" / "cross_framework"
    files = visualize_all_framework_outputs(
        execution,
        output,
        allowed_frameworks={"numpyro"},
        allowed_model_names={"finite"},
        generate_animations=False,
    )
    assert {Path(item).name for item in files} == {"finite_numpyro_observations.png"}
    assert all(Path(item).is_file() for item in files)
    assert not (output.parent / "numpyro" / "finite_numpyro_free_energy.png").exists()
    assert _custody(sources) == original


@pytest.fixture
def native_operational_inputs(monkeypatch):
    """Observe input delivery while retaining the real native consumer/output."""
    observed = {}
    original = viz_plots.generate_cross_framework_comparison

    def forward(data, path):
        observed["data"] = deepcopy(data)
        result = original(data, path)
        observed["result"] = result
        return result

    monkeypatch.setattr(viz_plots, "generate_cross_framework_comparison", forward)
    return observed


def _saved_active_partial(tmp_path, csv_rows):
    execution = tmp_path / "12_execute_output"
    implementation = execution / "finite" / "activeinference_jl"
    structured = _write(
        implementation / "execution_logs" / "finite_results.json",
        {
            "framework": "activeinference_jl",
            "model_name": "finite",
            "implementation_directory": str(implementation),
        },
    )
    simulation = _write(
        implementation / "simulation_data" / "simulation_results.json",
        {
            "schema_version": "activeinference_jl_simulation_v1",
            "framework": "activeinference_jl",
            "model_name": "finite",
            "beliefs": [],
            "actions": [],
            "observations": [],
            "expected_free_energy": [-3.0, 0.0, 3.0],
            "expected_free_energy_convention": "Authored partial saved EFE diagnostic",
        },
    )
    sources = [structured, simulation]
    if csv_rows is not None:
        csv = implementation / "simulation_data" / "simulation_results.csv"
        csv.write_text(
            "# Authored saved CSV; no Julia/model execution\n"
            "step,observation,action,belief0,belief1\n" + csv_rows
        )
        sources.append(csv)
    # A second saved record invokes the actual operational consumer. No model
    # binding or scientific cross-framework agreement is asserted.
    sources.append(
        _write(
            execution / "finite" / "numpyro" / "finite_results.json",
            {
                "framework": "numpyro",
                "model_name": "finite",
                "simulation_data": {"observations": [1, 0], "actions": [0, 1]},
            },
        )
    )
    sources.append(
        _write(
            execution / "other" / "activeinference_jl" / "other_results.json",
            {
                "framework": "activeinference_jl",
                "model_name": "other",
                "schema_version": "activeinference_jl_simulation_v1",
                "observations": [99],
            },
        )
    )
    return execution, sources


def _run_active_partial(execution, tmp_path):
    output = tmp_path / "16_analysis_output" / "cross_framework"
    files = visualize_all_framework_outputs(
        execution,
        output,
        allowed_frameworks={"activeinference_jl", "numpyro"},
        allowed_model_names={"finite"},
        generate_animations=False,
    )
    assert all(Path(item).is_file() for item in files)
    for item in files:
        with Image.open(item) as image:
            image.load()
            assert image.format == "PNG"
            assert image.width > 100 and image.height > 100
    assert not list(output.parent.rglob("other*.png"))
    return output, {Path(item).name for item in files}


def test_public_active_partial_schema_recovers_exact_saved_csv_rows_and_native_artifacts(
    tmp_path, native_figures, native_operational_inputs
):
    execution, sources = _saved_active_partial(
        tmp_path, "0,0,1,1.0,0.0\n1,1,0,0.5,0.5\n2,0,1,0.0,1.0\n"
    )
    original = _custody(sources)
    output, names = _run_active_partial(execution, tmp_path)
    assert "finite_activeinference_jl_observations.png" in names
    assert "finite_activeinference_jl_free_energy.png" in names
    delivered = native_operational_inputs["data"]["activeinference_jl_finite"][
        "simulation_data"
    ]
    assert delivered["observations"] == [0, 1, 0]
    assert delivered["actions"] == [1, 0, 1]
    assert delivered["beliefs"] == [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]
    observations = native_figures["finite_activeinference_jl_observations.png"].axes[0]
    assert [bar.get_height() for bar in observations.patches] == [2, 1]
    energy = native_figures["finite_activeinference_jl_free_energy.png"].axes[0]
    np.testing.assert_array_equal(energy.lines[0].get_ydata(), [-3.0, 0.0, 3.0])
    operational = json.loads(
        (output / "cross_framework_comparison.operational.json").read_text()
    )["frameworks"]["activeinference_jl"]
    assert operational["steps_completed"] == [3]
    assert operational["missing_timing_count"] == 1
    assert operational["execution_times"] == []
    assert _custody(sources) == original


def test_public_active_partial_csv_rejects_entire_malformed_numeric_row(
    tmp_path, native_figures, native_operational_inputs
):
    execution, sources = _saved_active_partial(
        tmp_path, "0,0,1,1.0,0.0\n1,1,0,not-numeric,0.5\n2,1,0,0.0,1.0\n"
    )
    original = _custody(sources)
    output, names = _run_active_partial(execution, tmp_path)
    assert "finite_activeinference_jl_observations.png" in names
    delivered = native_operational_inputs["data"]["activeinference_jl_finite"][
        "simulation_data"
    ]
    assert delivered["observations"] == [0, 1]
    assert delivered["actions"] == [1, 0]
    assert delivered["beliefs"] == [[1.0, 0.0], [0.0, 1.0]]
    observations = native_figures["finite_activeinference_jl_observations.png"].axes[0]
    assert [bar.get_height() for bar in observations.patches] == [1, 1]
    operational = json.loads(
        (output / "cross_framework_comparison.operational.json").read_text()
    )["frameworks"]["activeinference_jl"]
    assert operational["steps_completed"] == [2]
    assert _custody(sources) == original


def test_public_active_partial_missing_csv_keeps_available_efe_without_invented_rows(
    tmp_path, native_figures, native_operational_inputs
):
    execution, sources = _saved_active_partial(tmp_path, None)
    original = _custody(sources)
    _, names = _run_active_partial(execution, tmp_path)
    assert "finite_activeinference_jl_free_energy.png" in names
    assert "finite_activeinference_jl_observations.png" not in names
    delivered = native_operational_inputs["data"]["activeinference_jl_finite"][
        "simulation_data"
    ]
    assert delivered["observations"] == []
    assert delivered["actions"] == []
    assert delivered["beliefs"] == []
    energy = native_figures["finite_activeinference_jl_free_energy.png"].axes[0]
    np.testing.assert_array_equal(energy.lines[0].get_ydata(), [-3.0, 0.0, 3.0])
    assert _custody(sources) == original
