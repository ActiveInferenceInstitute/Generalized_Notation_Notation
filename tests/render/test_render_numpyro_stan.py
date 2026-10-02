"""End-to-end render tests for NumPyro and Stan backends.

Validates that the GNN → Render → Validate pipeline works for both
probabilistic programming frameworks.  Follows the same pattern as
test_render_pytorch_renderer.py (inline spec dict, compile check).
"""

from __future__ import annotations

import ast
import py_compile
from pathlib import Path
from typing import Any, Dict

# ──────────────────────────────────────────────
# Shared fixtures
# ──────────────────────────────────────────────


def _small_gnn_spec() -> Dict[str, Any]:
    """Minimal 2-state POMDP spec for render smoke tests."""
    return {
        "modelName": "render_e2e_test",
        "model_parameters": {
            "num_hidden_states": 2,
            "num_obs": 2,
            "num_actions": 2,
            "num_timesteps": 5,
            "b_tensor_order": "next_state_previous_state_action",
        },
        "initialparameterization": {
            "A": [[0.9, 0.1], [0.1, 0.9]],
            "B": [
                [[0.9, 0.8], [0.2, 0.1]],
                [[0.1, 0.2], [0.8, 0.9]],
            ],
            "C": [0.0, 1.0],
            "D": [0.5, 0.5],
        },
    }


def _gnn_variables() -> list:
    """Minimal variable list for the Stan renderer interface."""
    return [
        {"name": "A", "dimensions": [2, 2], "dtype": "float"},
        {"name": "B", "dimensions": [2, 2, 2], "dtype": "float"},
        {"name": "s", "dimensions": [2], "dtype": "float"},
        {"name": "o", "dimensions": [2], "dtype": "float"},
    ]


def _gnn_connections() -> list:
    """Minimal connection list for the Stan renderer interface."""
    return [
        {"source": "s", "target": "o", "directed": True},
        {"source": "s", "target": "s", "directed": True},
    ]


# ──────────────────────────────────────────────
# NumPyro Renderer Tests
# ──────────────────────────────────────────────


class TestNumPyroRenderer:
    """E2E tests for the NumPyro render backend."""

    def test_numpyro_renders_successfully(self, tmp_path: Path) -> None:
        """NumPyro renderer produces a success result."""
        from gnn.render.numpyro.numpyro_renderer import render_gnn_to_numpyro

        output = tmp_path / "numpyro_model.py"
        success, message, artifacts = render_gnn_to_numpyro(_small_gnn_spec(), output)

        assert success, f"NumPyro render failed: {message}"
        assert output.exists(), "Output file was not created"
        assert len(artifacts) > 0, "No artifact paths returned"

    def test_numpyro_generates_valid_python(self, tmp_path: Path) -> None:
        """Generated NumPyro code is syntactically valid Python."""
        from gnn.render.numpyro.numpyro_renderer import render_gnn_to_numpyro

        output = tmp_path / "numpyro_syntax.py"
        success, _, _ = render_gnn_to_numpyro(_small_gnn_spec(), output)
        assert success, "NumPyro render should succeed"

        # Compile check — raises SyntaxError on malformed code
        py_compile.compile(str(output), doraise=True)

    def test_numpyro_code_parses_to_ast(self, tmp_path: Path) -> None:
        """Generated code is parseable as a Python AST (stronger than compile)."""
        from gnn.render.numpyro.numpyro_renderer import render_gnn_to_numpyro

        output = tmp_path / "numpyro_ast.py"
        success, _, _ = render_gnn_to_numpyro(_small_gnn_spec(), output)
        assert success, "NumPyro render should succeed"

        code = output.read_text(encoding="utf-8")
        tree = ast.parse(code)
        # Should contain function definitions (run_simulation at minimum)
        func_names = [
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        ]
        assert len(func_names) > 0, "No functions found in generated code"
        assert "run_simulation" in func_names, (
            f"Expected 'run_simulation' function, found: {func_names}"
        )

    def test_numpyro_contains_expected_imports(self, tmp_path: Path) -> None:
        """Generated code imports numpyro and jax."""
        from gnn.render.numpyro.numpyro_renderer import render_gnn_to_numpyro

        output = tmp_path / "numpyro_imports.py"
        success, _, _ = render_gnn_to_numpyro(_small_gnn_spec(), output)
        assert success, "NumPyro render should succeed"

        code = output.read_text(encoding="utf-8")
        assert "import numpyro" in code or "from numpyro" in code, (
            "NumPyro import not found in generated code"
        )
        assert "import jax" in code or "from jax" in code, (
            "JAX import not found in generated code"
        )

    def test_numpyro_contains_model_name(self, tmp_path: Path) -> None:
        """Generated code references the original model name."""
        from gnn.render.numpyro.numpyro_renderer import render_gnn_to_numpyro

        output = tmp_path / "numpyro_name.py"
        success, _, _ = render_gnn_to_numpyro(_small_gnn_spec(), output)
        assert success, "NumPyro render should succeed"

        code = output.read_text(encoding="utf-8")
        assert "render_e2e_test" in code, "Model name not found in output"

    def test_numpyro_respects_timesteps(self, tmp_path: Path) -> None:
        """Generated code uses the num_timesteps from the spec."""
        from gnn.render.numpyro.numpyro_renderer import render_gnn_to_numpyro

        output = tmp_path / "numpyro_timesteps.py"
        spec = _small_gnn_spec()
        spec["model_parameters"]["num_timesteps"] = 42

        success, _, _ = render_gnn_to_numpyro(spec, output)
        assert success, "NumPyro render should succeed"

        code = output.read_text(encoding="utf-8")
        assert "42" in code, "Custom num_timesteps not reflected in output"

    def test_numpyro_discrete_code_declares_simulation_schema(
        self, tmp_path: Path
    ) -> None:
        """The generated discrete NumPyro script stamps numpyro_simulation_v1."""
        from gnn.render.numpyro.numpyro_renderer import render_gnn_to_numpyro

        output = tmp_path / "numpyro_schema.py"
        success, message, _ = render_gnn_to_numpyro(_small_gnn_spec(), output)

        assert success, message
        code = output.read_text(encoding="utf-8")
        assert "numpyro_simulation_v1" in code

    def test_numpyro_emits_spec_matrix_provenance(self, tmp_path: Path) -> None:
        """Generated code carries the spec's matrix_provenance verbatim."""
        import ast

        from gnn.render.numpyro.numpyro_renderer import render_gnn_to_numpyro

        output = tmp_path / "numpyro_provenance.py"
        spec = _small_gnn_spec()
        spec["matrix_provenance"] = {
            "B": {"canonical_order": "next_state_previous_state_action"}
        }

        success, message, _ = render_gnn_to_numpyro(spec, output)

        assert success, message
        code = output.read_text(encoding="utf-8")
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
        assert ast.literal_eval(provenance_node) == {
            "B": {"canonical_order": "next_state_previous_state_action"}
        }

    def test_numpyro_defaults_matrix_provenance_to_empty_dict(
        self, tmp_path: Path
    ) -> None:
        """Specs without matrix_provenance still emit the key as {}."""
        import ast

        from gnn.render.numpyro.numpyro_renderer import render_gnn_to_numpyro

        output = tmp_path / "numpyro_no_provenance.py"
        success, message, _ = render_gnn_to_numpyro(_small_gnn_spec(), output)

        assert success, message
        code = output.read_text(encoding="utf-8")
        tree = ast.parse(code)
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
        assert ast.literal_eval(provenance_node) == {}


# ──────────────────────────────────────────────
# Stan Renderer Tests
# ──────────────────────────────────────────────


class TestStanRenderer:
    """E2E tests for the Stan render backend."""

    def test_stan_renders_model_code(self) -> None:
        """Stan renderer produces non-empty model code."""
        from gnn.render.stan.stan_renderer import render_stan

        code = render_stan(_gnn_variables(), _gnn_connections(), model_name="e2e_test")

        assert isinstance(code, str)
        assert len(code) > 0, "Stan code is empty"

    def test_stan_has_required_blocks(self) -> None:
        """Generated Stan code contains data{}, parameters{}, model{} blocks."""
        from gnn.render.stan.stan_renderer import render_stan

        code = render_stan(
            _gnn_variables(), _gnn_connections(), model_name="block_test"
        )

        assert "data {" in code, "Missing data{} block"
        assert "parameters {" in code, "Missing parameters{} block"
        assert "model {" in code, "Missing model{} block"

    def test_stan_classifies_observed_variables(self) -> None:
        """Observed variables (o) go to data{}, latent (s) to parameters{}."""
        from gnn.render.stan.stan_renderer import render_stan

        code = render_stan(
            _gnn_variables(), _gnn_connections(), model_name="classify_test"
        )

        # 'o' should be in data block
        data_section = code.split("data {")[1].split("}")[0]
        assert "o" in data_section, "Observed variable 'o' not in data{}"

        # 's' should be in parameters block
        params_section = code.split("parameters {")[1].split("}")[0]
        assert "s" in params_section, "Latent variable 's' not in parameters{}"

    def test_stan_generates_connection_statements(self) -> None:
        """Directed connections are documented as edges in the sketch's model{}.

        The structural sketch deliberately emits no ``~`` likelihoods — the
        runnable inference program comes from ``render_gnn_to_stan``.
        """
        from gnn.render.stan.stan_renderer import render_stan

        code = render_stan(_gnn_variables(), _gnn_connections(), model_name="conn_test")

        model_section = code.split("model {")[1].split("}")[0]
        assert "~ normal(" not in model_section, "sketch must not fabricate likelihoods"
        assert "s → o" in model_section or "s →" in model_section, (
            "Connection s→o not documented in model{}"
        )

    def test_stan_type_mapping_vector(self) -> None:
        """1D float variables map to Stan vector[] type."""
        from gnn.render.stan.stan_renderer import render_stan

        vars_1d: list[Any] = [{"name": "x", "dimensions": [3], "dtype": "float"}]
        code = render_stan(vars_1d, [], model_name="type_test")

        assert "vector[3]" in code, "1D float not mapped to vector[3]"

    def test_stan_type_mapping_matrix(self) -> None:
        """2D float variables map to Stan matrix[] type."""
        from gnn.render.stan.stan_renderer import render_stan

        vars_2d: list[Any] = [{"name": "M", "dimensions": [3, 4], "dtype": "float"}]
        code = render_stan(vars_2d, [], model_name="matrix_test")

        assert "matrix[3, 4]" in code, "2D float not mapped to matrix[3, 4]"

    def test_stan_type_mapping_3d_array(self) -> None:
        """3D variables use Stan array[] matrix[] syntax."""
        from gnn.render.stan.stan_renderer import render_stan

        vars_3d: list[Any] = [{"name": "T", "dimensions": [2, 3, 4], "dtype": "float"}]
        code = render_stan(vars_3d, [], model_name="array_test")

        assert "array[2]" in code, "3D not mapped to array syntax"
        assert "matrix[3, 4]" in code, "Inner dimensions not mapped to matrix"

    def test_stan_model_name_in_header(self) -> None:
        """Generated code contains the model name in a comment header."""
        from gnn.render.stan.stan_renderer import render_stan

        code = render_stan(
            _gnn_variables(), _gnn_connections(), model_name="header_check"
        )

        assert "header_check" in code, "Model name not in generated header"

    def test_stan_empty_inputs(self) -> None:
        """Stan renderer handles empty variable/connection lists gracefully."""
        from gnn.render.stan.stan_renderer import render_stan

        code = render_stan([], [], model_name="empty_test")

        assert "data {" in code
        assert "parameters {" in code
        assert "model {" in code
        assert isinstance(code, str) and len(code) > 0
