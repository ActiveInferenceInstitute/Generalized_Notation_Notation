#!/usr/bin/env python3
"""End-to-end smoke tests for the analysis module split (MAJ-04 2/6).

Covers the moved bodies that the targeted suites under-exercise:
``visualize_all_framework_outputs`` (418-line orchestrator),
``generate_unified_framework_dashboard`` (3% coverage pre-smoke),
``simulation_visualizations`` (10%), and the cross-framework
comparison/report chain — fed by shapes produced by construction
(``analyze_framework_outputs`` output feeds the consumers directly).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gnn.analysis.analyzer import analyze_framework_outputs
from gnn.analysis.simulation_visualizations import (
    generate_matrix_visualizations,
    visualize_simulation_results,
)
from gnn.analysis.viz_dashboard import (
    generate_confidence_comparison,
    generate_cross_framework_comparison,
    generate_efe_convergence_comparison,
    generate_unified_framework_dashboard,
)
from gnn.analysis.viz_plots import visualize_all_framework_outputs

PYMDP_PAYLOAD: dict[str, Any] = {
    "schema_version": "pymdp_simulation_v1",
    "beliefs_by_factor": {"joint_state": [[0.8, 0.2], [0.7, 0.3]]},
    "actions_by_control_factor": {"joint_action": [0, 1]},
    "observations_by_modality": {"joint_observation": [0, 1]},
    "expected_free_energy": [-0.1, -0.2],
    "variational_free_energy": [-0.3, -0.4],
    "metrics": {},
    "model_parameters": {"num_states": 3, "num_observations": 3},
}


def _execution_tree(tmp_path: Path) -> Path:
    """Minimal Step 12 output tree: summary + one pymdp implementation dir."""
    execution_dir = tmp_path / "12_execute_output"
    summaries = execution_dir / "summaries"
    summaries.mkdir(parents=True)
    impl_dir = execution_dir / "current_model" / "pymdp"
    impl_dir.mkdir(parents=True)
    summary: dict[str, Any] = {
        "requested_frameworks": ["pymdp"],
        "execution_details": [
            {
                "framework": "pymdp",
                "model_name": "current_model",
                "success": True,
                "execution_time": 0.1,
                "implementation_directory": str(impl_dir),
            }
        ],
        "framework_status": {"pymdp": {"status": "success"}},
    }
    (summaries / "execution_summary.json").write_text(json.dumps(summary))
    logs = impl_dir / "execution_logs" / "run1"
    logs.mkdir(parents=True)
    (logs / "simulation_results.json").write_text(json.dumps(PYMDP_PAYLOAD))
    (impl_dir / "structured_result.json").write_text(json.dumps(PYMDP_PAYLOAD))
    return execution_dir


def _framework_results(tmp_path: Path) -> dict[str, Any]:
    return analyze_framework_outputs(_execution_tree(tmp_path))


def test_generate_unified_framework_dashboard_smoke(tmp_path: Path) -> None:
    framework_data: dict[str, Any] = {
        "pymdp": {
            "framework": "pymdp",
            "simulation_data": {
                "beliefs": [[0.8, 0.2], [0.6, 0.4]],
                "actions": [0, 1],
                "expected_free_energy": [-0.1, -0.2],
                "variational_free_energy": [-0.3, -0.4],
            },
        }
    }
    generated = generate_unified_framework_dashboard(
        framework_data, tmp_path / "dash", model_name="current_model"
    )
    assert isinstance(generated, list)
    assert generated == [], "unbound populated data must not establish a comparison"
    admission = json.loads(
        (tmp_path / "dash" / "unified_dashboard_admission.json").read_text()
    )
    assert "missing declared model_id" in admission["excluded"]["pymdp[0]"]


def test_analyze_then_dashboard_chain(tmp_path: Path) -> None:
    results = _framework_results(tmp_path)
    framework_data = results["frameworks"]
    generated = generate_unified_framework_dashboard(
        framework_data, tmp_path / "dash", model_name="current_model"
    )
    assert isinstance(generated, list)  # graceful when data is thin


def test_analyze_then_comparison_family(tmp_path: Path) -> None:
    results = _framework_results(tmp_path)
    out = tmp_path / "cmp"
    comparison = generate_cross_framework_comparison(
        results["frameworks"], out / "comparison.png"
    )
    assert Path(comparison).exists()
    # EFE convergence degrades gracefully when traces are absent.
    generate_efe_convergence_comparison(results["frameworks"], out)
    generate_confidence_comparison(results["frameworks"], out)


def test_analyze_then_framework_report(tmp_path: Path) -> None:
    results = _framework_results(tmp_path)
    from gnn.analysis.framework_comparison import generate_framework_comparison_report

    report = generate_framework_comparison_report(results, tmp_path / "reports")
    assert Path(report).exists()


def test_visualize_all_framework_outputs_smoke(tmp_path: Path) -> None:
    generated = visualize_all_framework_outputs(
        _execution_tree(tmp_path), tmp_path / "viz"
    )
    assert isinstance(generated, list)
    assert generated, "no visualizations generated from a populated execution tree"


def test_visualize_simulation_results_smoke(tmp_path: Path) -> None:
    results: dict[str, Any] = {
        "execution_details": [
            {
                "model_name": "current_model",
                "framework": "pymdp",
                "success": True,
                "beliefs": [[0.8, 0.2], [0.6, 0.4]],
                "actions": [0, 1],
                "observations": [0, 1],
                "expected_free_energies": [-0.1, -0.2],
            }
        ]
    }
    generated = visualize_simulation_results(results, tmp_path / "simviz")
    assert isinstance(generated, list)


def test_generate_matrix_visualizations_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "matrices"
    out_dir.mkdir(parents=True, exist_ok=True)
    parsed_data: dict[str, Any] = {
        "matrices": [
            {"name": "A", "data": np.array([[0.9, 0.1], [0.1, 0.9]])},
            {"name": "B", "data": np.array([[0.8, 0.2], [0.2, 0.8]])},
        ]
    }
    generated = generate_matrix_visualizations(parsed_data, out_dir, "current_model")
    assert isinstance(generated, list)
    assert generated, "matrix visualizations generated nothing from valid matrices"


def test_module_smoke_no_unexpected_errors(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.ERROR)
    visualize_all_framework_outputs(_execution_tree(tmp_path), tmp_path / "viz2")
    errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors == [], f"unexpected ERROR-level log records: {errors}"


@pytest.mark.parametrize(
    ("payload", "expected_label", "expected_updates"),
    [
        (
            {"model_kind": "continuous", "beliefs": [[-2, 1], [1, -3], [1, -3]]},
            "Posterior Mean Update (Euclidean Norm)",
            [5.0, 0.0],
        ),
        (
            {
                "runtime_metadata": {"model_kind": "continuous"},
                "beliefs": [[2, 1], [4, 3]],
            },
            "Posterior Mean Update (Euclidean Norm)",
            [np.sqrt(8)],
        ),
        (
            {"beliefs": [[1, 0], [0, 1]]},
            "Jensen-Shannon Distance",
            [np.sqrt(np.log(2))],
        ),
        ({"states": [[-2, 1], [1, -3]]}, "State Update (Euclidean Norm)", [5.0]),
        (
            {"model_kind": "continuous", "states": [[-2, 1], [1, -3]]},
            "State Update (Euclidean Norm)",
            [5.0],
        ),
    ],
)
def test_simulation_update_metric_preserves_family(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, Any],
    expected_label: str,
    expected_updates: list[float],
) -> None:
    """Gaussian means retain their values; categorical distance admits exact zeros."""
    from gnn.analysis import simulation_visualizations as module

    source = tmp_path / "implementation" / "simulation_data"
    source.mkdir(parents=True)
    (source / "simulation_results.json").write_text(json.dumps(payload))
    captured: list[tuple[str, list[float], str]] = []
    savefig = module.plt.savefig

    def inspect_figure(filename: Any, **kwargs: Any) -> None:
        if str(filename).endswith("_belief_convergence.png"):
            ax = module.plt.gcf().axes[0]
            captured.append(
                (
                    ax.get_ylabel(),
                    list(ax.lines[0].get_ydata()),
                    ax.lines[0].get_label(),
                )
            )
        savefig(filename, **kwargs)

    monkeypatch.setattr(module.plt, "savefig", inspect_figure)
    files = module.visualize_simulation_results(
        {
            "execution_details": [
                {
                    "model_name": "metric",
                    "framework": "jax",
                    "implementation_directory": str(source.parent),
                }
            ]
        },
        tmp_path / "plots",
    )
    assert len(captured) == 1
    assert captured[0][0] == expected_label
    np.testing.assert_allclose(captured[0][1], expected_updates)
    assert captured[0][2] == (
        "State Update"
        if "beliefs" not in payload
        else "Posterior Mean Update"
        if "Mean" in expected_label
        else "Belief Update Magnitude"
    )
    assert all(Path(name).read_bytes().startswith(b"\x89PNG") for name in files)


@pytest.mark.parametrize(
    "beliefs",
    [
        [[-0.2, 1.2], [0.5, 0.5]],
        [[0.2, 0.2], [0.5, 0.5]],
        [[float("nan"), 1], [0.5, 0.5]],
    ],
)
def test_simulation_update_refuses_malformed_probabilities(
    tmp_path: Path,
    beliefs: list[list[float]],
) -> None:
    """Malformed rows cannot become evidence through clipping or normalization."""
    source = tmp_path / "implementation" / "simulation_data"
    source.mkdir(parents=True)
    (source / "simulation_results.json").write_text(json.dumps({"beliefs": beliefs}))
    assert (
        visualize_simulation_results(
            {
                "execution_details": [
                    {
                        "model_name": "invalid",
                        "framework": "jax",
                        "implementation_directory": str(source.parent),
                    }
                ]
            },
            tmp_path / "plots",
        )
        == []
    )
    assert not list((tmp_path / "plots").rglob("*.png"))
