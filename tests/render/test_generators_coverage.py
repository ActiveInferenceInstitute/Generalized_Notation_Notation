"""Provides helper functions: _explicit_pomdp_spec, test_bnlearn_generator, test_pymdp_generator, test_activeinference_jl_generator, and 4 more.

Public functions: _explicit_pomdp_spec, test_bnlearn_generator, test_pymdp_generator, test_activeinference_jl_generator, test_rxinfer_generator, test_discopy_generator, test_matrix_to_julia, test_sanitizers
"""

from pathlib import Path
from typing import Any

import pytest

from gnn.render.bnlearn import generate_bnlearn_code
from gnn.render.generators import (
    _matrix_to_julia,
    _sanitize_identifier,
    _to_pascal_case,
    generate_activeinference_jl_code,
    generate_discopy_code,
    generate_pymdp_code,
    generate_rxinfer_code,
)
from tests.helpers.bar_labels import (
    assert_bar_labels_offset_in_points,
    assert_png_bounded,
    figures_held_open,
    open_bar_figures,
)


def _explicit_pomdp_spec() -> dict[str, Any]:
    return {
        "model_name": "TestModel",
        "description": 'A model with quoted labels like "left" and "reward context".',
        "initialparameterization": {
            "A": [[0.9, 0.1], [0.1, 0.9]],
            "B": [[[0.8, 0.2], [0.2, 0.8]]],
            "C": [0.0, 1.0],
            "D": [0.5, 0.5],
            "E": [1.0],
        },
        "model_parameters": {"num_hidden_states": 2, "num_obs": 2, "num_actions": 1},
    }


def test_bnlearn_generator() -> Any:
    res = generate_bnlearn_code({"model_name": "TestModel"})
    assert isinstance(res, str)
    assert "TestModel" in res


def test_pymdp_generator() -> Any:
    res = generate_pymdp_code({"model_name": "TestModel"})
    assert isinstance(res, str)
    # The template might enforce certain things


def test_activeinference_jl_generator() -> Any:
    res = generate_activeinference_jl_code(_explicit_pomdp_spec())
    assert isinstance(res, str)
    assert "TestModel" in res
    assert "using Base64" in res
    assert "base64decode" in res
    assert "JSON.parse(raw" not in res


def test_rxinfer_generator() -> Any:
    res = generate_rxinfer_code(_explicit_pomdp_spec())
    assert isinstance(res, str)
    assert "TestModel" in res
    assert "using Base64" in res
    assert "base64decode" in res
    assert "JSON.parse(raw" not in res


def test_discopy_generator() -> Any:
    res = generate_discopy_code({"model_name": "TestModel"})
    assert isinstance(res, str)
    assert "TestModel" in res


def test_matrix_to_julia() -> Any:
    assert _matrix_to_julia([1, 2, 3]) == "[1, 2, 3]"
    assert _matrix_to_julia([[1, 2], [3, 4]]) == "[1 2; 3 4]"
    assert "cat(" in _matrix_to_julia([[[1, 2]], [[3, 4]]])


def test_sanitizers() -> Any:
    assert _sanitize_identifier("Test Model 1") == "test_model_1"
    assert _to_pascal_case("test model") == "TestModel"


def _discopy_results(score: float) -> dict[str, Any]:
    steps = [
        {
            "step": i,
            "semantic_score": score,
            "duration_ms": 1.0,
            "complexity_measure": 1,
            "analysis_type": "composition",
        }
        for i in range(4)
    ]
    return {
        "analysis_steps": steps,
        "semantic_scores": [score] * len(steps),
        "performance_metrics": {
            "average_semantic_score": score,
            "analysis_efficiency": 100 * score,
            "semantic_score_stability": 10 * score,
        },
        "diagrams_info": {
            "main_domain": "State",
            "main_codomain": "Observation",
            "total_morphisms": 2,
        },
    }


@pytest.mark.parametrize("score", [0.0, 0.9])
def test_discopy_dashboard_bar_labels_are_offset_in_points(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, score: float
) -> None:
    code = generate_discopy_code({"model_name": "TestModel"})
    namespace: dict[str, Any] = {"__name__": "generated_discopy"}
    exec(compile(code, "generated_discopy.py", "exec"), namespace)
    analyzer = namespace[f"Enhanced{_to_pascal_case('TestModel')}CategoricalAnalyzer"]()

    with figures_held_open(monkeypatch):
        viz_files = analyzer.create_enhanced_visualizations(
            _discopy_results(score), tmp_path
        )
        for png in viz_files:
            assert_png_bounded(Path(png))
        (dashboard,) = open_bar_figures()
        assert_bar_labels_offset_in_points(dashboard)
