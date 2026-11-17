#!/usr/bin/env python3
"""
Tests for render/jax/jax_renderer.py. JAX and NumPy are hard project
dependencies (see pyproject.toml), so live generated-script checks import
them explicitly and fail if absent (repo zero-skip contract).
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np


def _jax_spec(model_name: str = "TestModel") -> dict[str, Any]:
    """Return a canonical non-square-action POMDP for generator tests."""
    return {
        "model_name": model_name,
        "model_parameters": {
            "num_hidden_states": 3,
            "num_obs": 2,
            "num_actions": 2,
            "num_timesteps": 3,
            "b_tensor_order": "next_state_previous_state_action",
        },
        "initialparameterization": {
            "A": [[0.8, 0.3, 0.1], [0.2, 0.7, 0.9]],
            "B": [
                [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]],
                [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]],
                [[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]],
            ],
            "C": [0.0, 1.0],
            "D": [1.0, 0.0, 0.0],
        },
    }


@pytest.fixture(scope="module")
def mod() -> Any:
    try:
        import gnn.render.jax.jax_renderer as m

        return m
    except ImportError:
        raise AssertionError("gnn.render.jax.jax_renderer not importable")


@pytest.fixture(scope="module")
def spec_mod() -> Any:
    """Canonical home of the matrix-string parsers and spec extraction."""
    import gnn.render.jax.jax_spec_extract as m

    return m


class TestParseGnnMatrixString:
    def test_3x3_matrix_with_braces(self, spec_mod: Any) -> Any:
        matrix = spec_mod._parse_gnn_matrix_string(
            "{(0.9,0.05,0.05),(0.05,0.9,0.05),(0.05,0.05,0.9)}"
        )
        assert matrix.shape == (3, 3)
        assert abs(matrix[0, 0] - 0.9) < 1e-6

    def test_returns_numpy_array(self, spec_mod: Any) -> Any:
        result = spec_mod._parse_gnn_matrix_string("{(0.5,0.5)}")
        assert isinstance(result, np.ndarray)

    def test_invalid_returns_fallback_array(self, spec_mod: Any) -> Any:
        result = spec_mod._parse_gnn_matrix_string("NOT_A_MATRIX")
        assert isinstance(result, np.ndarray)


class TestParseMatrixString:
    def test_semicolon_separated_rows(self, spec_mod: Any) -> Any:
        matrix = spec_mod._parse_matrix_string("0.9,0.1;0.1,0.9")
        assert matrix.shape == (2, 2)
        assert abs(matrix[0, 0] - 0.9) < 1e-6

    def test_single_row(self, spec_mod: Any) -> Any:
        matrix = spec_mod._parse_matrix_string("0.5,0.3,0.2")
        assert matrix.ndim >= 1
        assert matrix.shape[-1] == 3

    def test_empty_string_returns_array(self, spec_mod: Any) -> Any:
        result = spec_mod._parse_matrix_string("")
        assert isinstance(result, np.ndarray)


class TestParseVectorString:
    def test_basic_vector(self, spec_mod: Any) -> Any:
        vec = spec_mod._parse_vector_string("0.5,0.3,0.2")
        assert vec.shape == (3,)
        assert abs(vec.sum() - 1.0) < 1e-6

    def test_returns_numpy_array(self, spec_mod: Any) -> Any:
        result = spec_mod._parse_vector_string("0.5,0.5")
        assert isinstance(result, np.ndarray)

    def test_invalid_falls_back(self, spec_mod: Any) -> Any:
        result = spec_mod._parse_vector_string("INVALID")
        assert isinstance(result, np.ndarray)


class TestExtractGnnMatrices:
    def test_empty_spec_returns_dict(self, spec_mod: Any) -> Any:
        result = spec_mod._extract_gnn_matrices({})
        assert isinstance(result, dict)

    def test_spec_with_parameters_returns_dict(self, spec_mod: Any) -> Any:
        spec: dict[str, Any] = {
            "parameters": {"A": "{(0.9,0.1),(0.1,0.9)}"},
            "state_space": {"A": {"dimensions": [2, 2], "type": "float"}},
        }
        result = spec_mod._extract_gnn_matrices(spec)
        assert isinstance(result, dict)

    def test_passive_pomdp_b_stays_canonical_single_action(self, spec_mod: Any) -> Any:
        spec: dict[str, Any] = {
            "model_name": "passive_chain",
            "model_parameters": {
                "num_hidden_states": 3,
                "num_obs": 3,
                "num_actions": 1,
            },
            "initialparameterization": {
                "A": np.eye(3).tolist(),
                "B": np.eye(3).tolist(),
                "C": [0.0, 0.0, 0.0],
                "D": [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            },
        }

        result = spec_mod._extract_gnn_matrices(spec)

        assert result["B"].shape == (3, 3, 1)
        np.testing.assert_allclose(result["B"][:, :, 0], np.eye(3))

    def test_controlled_canonical_b_is_not_transposed(self, spec_mod: Any) -> Any:
        b_matrix = np.zeros((3, 3, 3))
        for action in range(3):
            b_matrix[:, :, action] = np.roll(np.eye(3), action, axis=0)
        spec: dict[str, Any] = {
            "model_name": "controlled_chain",
            "model_parameters": {
                "num_hidden_states": 3,
                "num_obs": 3,
                "num_actions": 3,
                "b_tensor_order": "next_state_previous_state_action",
            },
            "initialparameterization": {
                "A": np.eye(3).tolist(),
                "B": b_matrix.tolist(),
                "C": [0.0, 0.0, 0.0],
                "D": [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            },
        }

        result = spec_mod._extract_gnn_matrices(spec)

        assert result["B"].shape == (3, 3, 3)
        np.testing.assert_allclose(result["B"], b_matrix)


class TestGenerateJaxModelCode:
    """Tests for _generate_jax_model_code code generation."""

    VALID_SPEC = _jax_spec()

    def test_returns_string(self, mod: Any) -> Any:
        """_generate_jax_model_code returns a non-empty string."""
        result = mod._generate_jax_model_code(self.VALID_SPEC, None)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_import_jax(self, mod: Any) -> Any:
        """Generated code imports jax."""
        result = mod._generate_jax_model_code(self.VALID_SPEC, None)
        assert "import jax" in result or "jax" in result

    def test_model_name_in_output(self, mod: Any) -> Any:
        """Generated code uses the ModelName from spec."""
        spec = _jax_spec("MyUniqueModel")
        result = mod._generate_jax_model_code(spec, None)
        assert "MyUniqueModel" in result

    def test_empty_spec_fails_instead_of_emitting_recovery_stub(self, mod: Any) -> Any:
        with pytest.raises(ValueError, match="requires canonical A/B/C/D"):
            mod._generate_jax_model_code({}, None)

    def test_passive_pomdp_generates_one_action(self, mod: Any) -> Any:
        spec: dict[str, Any] = {
            "model_name": "passive_chain",
            "model_parameters": {
                "num_hidden_states": 3,
                "num_obs": 3,
                "num_actions": 1,
            },
            "initialparameterization": {
                "A": np.eye(3).tolist(),
                "B": np.eye(3).tolist(),
                "C": [0.0, 0.0, 0.0],
                "D": [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            },
        }

        result = mod._generate_jax_model_code(spec, None)

        assert "NUM_ACTIONS = 1" in result


class TestGenerateJaxPomdpCode:
    """Tests for _generate_jax_pomdp_code code generation."""

    def test_returns_string(self, mod: Any) -> Any:
        """_generate_jax_pomdp_code returns a non-empty string."""
        result = mod._generate_jax_pomdp_code(_jax_spec("POMDPModel"), None)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_jax_import(self, mod: Any) -> Any:
        """Generated POMDP code always includes jax import (even in recovery path)."""
        result = mod._generate_jax_pomdp_code(_jax_spec("POMDPTest"), None)
        assert "jax" in result

    def test_empty_spec_fails_instead_of_emitting_recovery_stub(self, mod: Any) -> Any:
        with pytest.raises(ValueError, match="requires canonical A/B/C/D"):
            mod._generate_jax_pomdp_code({}, None)

    def test_transition_uses_canonical_b_axis_order(self, mod: Any) -> None:
        code = mod._generate_jax_pomdp_code(_jax_spec(), None)
        assert "self.models.B[:, :, action]" in code
        assert "self.models.B[:, action, :]" not in code
        assert "jnp.dot(self.models.A.T, self.models.C)" in code

    def test_generated_solver_never_installs_dependencies(self, mod: Any) -> None:
        code = mod._generate_jax_pomdp_code(_jax_spec(), None)
        assert "pip install" not in code
        assert "subprocess.run" not in code


def test_render_failure_is_explicit_and_writes_no_stub(
    mod: Any, tmp_path: Path
) -> None:
    output_path = tmp_path / "invalid_jax.py"
    success, message, artifacts = mod.render_gnn_to_jax({}, output_path)

    assert success is False
    assert "requires canonical A/B/C/D" in message
    assert artifacts == []
    assert not output_path.exists()


@pytest.mark.integration
@pytest.mark.parametrize(
    ("renderer_name", "filename", "success_marker"),
    [
        ("render_gnn_to_jax", "model_general.py", "model test successful"),
        ("render_gnn_to_jax_pomdp", "model_solver.py", "solver test successful"),
    ],
)
def test_generated_jax_code_executes_with_unequal_dimensions(
    mod: Any,
    renderer_name: str,
    filename: str,
    success_marker: str,
    tmp_path: Path,
) -> None:
    import jax  # noqa: F401  # hard project dep; explicit import per zero-skip contract

    script = tmp_path / filename
    renderer = getattr(mod, renderer_name)
    success, message, artifacts = renderer(_jax_spec(), script)
    assert success, message
    assert artifacts == [str(script)]

    env = os.environ.copy()
    env["GNN_OUTPUT_DIR"] = str(tmp_path / "jax_outputs")
    result = subprocess.run(  # nosec B603
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert success_marker in result.stdout


def test_render_to_path_failure_message_names_label(tmp_path: Path) -> None:
    """The returned failure message must name the target and the cause,
    not degrade to a bare ``str(e)``."""
    from gnn.render.jax.jax_renderer import _render_to_path

    def _boom(gnn_spec: dict, options: object) -> str:
        raise RuntimeError("boom")

    success, message, artifacts = _render_to_path(
        _boom, "JAX model", {}, tmp_path / "out.py", None
    )
    assert success is False
    assert "JAX model" in message
    assert "boom" in message
    assert artifacts == []
