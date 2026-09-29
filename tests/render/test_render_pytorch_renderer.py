"""Focused tests for the PyTorch render backend."""

from __future__ import annotations

import py_compile
from typing import Any


def _small_gnn_spec() -> dict:
    return {
        "modelName": "pytorch_smoke",
        "model_parameters": {
            "num_hidden_states": 2,
            "num_obs": 2,
            "num_timesteps": 7,
        },
        "initialparameterization": {
            "A": [[0.9, 0.1], [0.1, 0.9]],
            "B": [
                [[0.9, 0.2], [0.1, 0.8]],
                [[0.8, 0.1], [0.2, 0.9]],
            ],
            "C": [0.0, 1.0],
            "D": [0.5, 0.5],
        },
    }


def test_pytorch_renderer_generates_compilable_script_with_timestep(
    tmp_path: Any,
) -> None:
    from gnn.render.pytorch.pytorch_renderer import render_gnn_to_pytorch

    output_path = tmp_path / "pytorch_smoke.py"

    success, message, artifacts = render_gnn_to_pytorch(_small_gnn_spec(), output_path)

    assert success, message
    assert artifacts == [str(output_path)]
    generated = output_path.read_text(encoding="utf-8")
    assert "T = 7" in generated
    py_compile.compile(str(output_path), doraise=True)


def test_render_success_policy_can_be_strict_about_framework_failures() -> None:
    from gnn.render.processor import _render_succeeded

    assert (
        _render_succeeded(
            success_count=1,
            total_files=1,
            total_framework_successes=1,
            total_framework_attempts=2,
            strict_framework_success=False,
        )
        is True
    )
    assert (
        _render_succeeded(
            success_count=1,
            total_files=1,
            total_framework_successes=1,
            total_framework_attempts=2,
            strict_framework_success=True,
        )
        is False
    )


def test_pytorch_discrete_code_declares_simulation_schema(tmp_path: Any) -> None:
    """The generated discrete PyTorch script stamps pytorch_simulation_v1."""
    from gnn.render.pytorch.pytorch_renderer import render_gnn_to_pytorch

    output_path = tmp_path / "pytorch_schema.py"

    success, message, _ = render_gnn_to_pytorch(_small_gnn_spec(), output_path)

    assert success, message
    generated = output_path.read_text(encoding="utf-8")
    assert "pytorch_simulation_v1" in generated


def _emitted_matrix_provenance(code: str) -> Any:
    """Return the literal ``matrix_provenance`` value in the generated results dict."""
    import ast

    tree = ast.parse(code)  # raises SyntaxError on malformed emission
    results_dict = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(getattr(target, "id", None) == "results" for target in node.targets)
    ).value
    assert isinstance(results_dict, ast.Dict)
    provenance_node = next(
        value
        for key, value in zip(results_dict.keys, results_dict.values)
        if isinstance(key, ast.Constant) and key.value == "matrix_provenance"
    )
    return ast.literal_eval(provenance_node)


def test_pytorch_emits_spec_matrix_provenance(tmp_path: Any) -> None:
    """Generated code carries the spec's matrix_provenance verbatim.

    Without it, a ``--frameworks all`` GridWorld run with the torch extra
    breaks ``matrix_provenance_equal`` in the Step 16 manifest.
    """
    from gnn.render.pytorch.pytorch_renderer import render_gnn_to_pytorch

    output_path = tmp_path / "pytorch_provenance.py"
    spec = _small_gnn_spec()
    spec["matrix_provenance"] = {
        "B": {"canonical_order": "next_state_previous_state_action"}
    }

    success, message, _ = render_gnn_to_pytorch(spec, output_path)

    assert success, message
    assert _emitted_matrix_provenance(output_path.read_text(encoding="utf-8")) == {
        "B": {"canonical_order": "next_state_previous_state_action"}
    }


def test_pytorch_defaults_matrix_provenance_to_empty_dict(tmp_path: Any) -> None:
    """Specs without matrix_provenance still emit the key as {}."""
    from gnn.render.pytorch.pytorch_renderer import render_gnn_to_pytorch

    output_path = tmp_path / "pytorch_no_provenance.py"

    success, message, _ = render_gnn_to_pytorch(_small_gnn_spec(), output_path)

    assert success, message
    assert _emitted_matrix_provenance(output_path.read_text(encoding="utf-8")) == {}
