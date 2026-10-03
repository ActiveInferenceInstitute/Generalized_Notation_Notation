"""Tests for the RxInfer animated HTML visualization generator.

Validates that generate_animated_html produces self-contained HTML files
with all expected animation functions, embedded data, and correct structure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gnn.analysis.rxinfer.animator import generate_animated_html


def _build_synthetic_results(
    n_steps: int = 10,
    n_states: int = 3,
    n_actions: int = 2,
) -> dict[str, Any]:
    """Build a synthetic rxinfer_simulation_v1 result for testing."""
    beliefs = []
    for t in range(n_steps):
        row = [0.2 / (n_states - 1)] * n_states
        row[t % n_states] = 0.8
        beliefs.append(row)

    return {
        "schema_version": "rxinfer_simulation_v1",
        "success": True,
        "framework": "RxInfer.jl",
        "model_name": "TestModel",
        "num_timesteps": n_steps,
        "beliefs": beliefs,
        "observations": [t % n_states for t in range(n_steps)],
        "true_states": [(t + 1) % n_states for t in range(n_steps)],
        "actions": [t % n_actions for t in range(n_steps)],
        "variational_free_energy": [6.11 - 0.1 * i for i in range(20)],
        "vfe_per_iteration": [6.11 - 0.1 * i for i in range(20)],
        "expected_free_energy": [3.0 - 0.1 * t for t in range(n_steps)],
        "validation": {
            "all_valid": True,
            "inference_converged": True,
            "vfe_present": True,
            "belief_entropy_ok": True,
            "belief_accuracy": 0.8,
            "belief_accuracy_ok": True,
        },
        "runtime_metadata": {
            "uses_real_rxinfer": True,
            "model_kind": "flat",
        },
    }


def test_generates_html_file(tmp_path: Path) -> None:
    """generate_animated_html produces an HTML file on disk."""
    results = _build_synthetic_results()
    output = tmp_path / "test_animation.html"
    path = generate_animated_html(results, output, "TestModel")
    assert Path(path).exists()
    assert Path(path).stat().st_size > 0


def test_html_contains_embedded_data(tmp_path: Path) -> None:
    """The HTML file contains embedded JSON data."""
    results = _build_synthetic_results()
    output = tmp_path / "test_animation.html"
    generate_animated_html(results, output, "TestModel")
    content = output.read_text()
    assert "DATA = {" in content
    assert "TestModel" in content


def test_html_contains_animation_functions(tmp_path: Path) -> None:
    """The HTML contains all expected animation JS functions."""
    results = _build_synthetic_results()
    output = tmp_path / "test_animation.html"
    generate_animated_html(results, output, "TestModel")
    content = output.read_text()
    assert "drawBeliefs" in content
    assert "drawStates" in content
    assert "drawActions" in content
    assert "drawVFE" in content
    assert "togglePlay" in content


def test_html_contains_metadata(tmp_path: Path) -> None:
    """The HTML displays model metadata (uses_real_rxinfer, accuracy, etc.)."""
    results = _build_synthetic_results()
    output = tmp_path / "test_animation.html"
    generate_animated_html(results, output, "TestModel")
    content = output.read_text()
    assert "uses_real_rxinfer" in content
    assert "belief_accuracy" in content
    assert "inference_converged" in content
    assert "model_kind" in content


def test_html_is_self_contained(tmp_path: Path) -> None:
    """The HTML has no external script/style dependencies."""
    results = _build_synthetic_results()
    output = tmp_path / "test_animation.html"
    generate_animated_html(results, output, "TestModel")
    content = output.read_text()
    assert "<script>" in content
    assert "<style>" in content
    assert "src=" not in content or "src=" not in content.split("<canvas>")[0]
    assert "<link" not in content


def test_dict_shaped_beliefs_handled(tmp_path: Path) -> None:
    """Dict-shaped beliefs_by_factor are handled correctly."""
    results = _build_synthetic_results()
    beliefs = results.pop("beliefs")
    results["beliefs_by_factor"] = {"joint_state": beliefs}
    output = tmp_path / "test_dict_animation.html"
    generate_animated_html(results, output, "DictModel")
    content = output.read_text()
    assert "DATA = {" in content
    assert "DictModel" in content


def test_empty_beliefs_returns_empty(tmp_path: Path) -> None:
    """Empty beliefs produces empty string (no animation)."""
    results = {"schema_version": "rxinfer_simulation_v1", "success": True}
    output = tmp_path / "empty_animation.html"
    path = generate_animated_html(results, output, "EmptyModel")
    assert path == ""


def test_different_state_counts(tmp_path: Path) -> None:
    """Models with different state counts produce valid animations."""
    for n_states in [2, 4, 8, 16]:
        results = _build_synthetic_results(n_states=n_states)
        output = tmp_path / f"test_{n_states}_states.html"
        generate_animated_html(results, output, f"Model_{n_states}")
        assert output.exists()
        content = output.read_text()
        # Verify the embedded data has the right number of states
        data_start = content.index("DATA = ") + len("DATA = ")
        # Find the matching closing brace by counting braces
        depth = 0
        for i, ch in enumerate(content[data_start:], data_start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    data_json = content[data_start : i + 1]
                    break
        data = json.loads(data_json)
        assert data["n_states"] == n_states
