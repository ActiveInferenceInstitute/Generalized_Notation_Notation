"""Tests for the Kronecker-factorized pipeline integration (MAJ-02 residual).

Pins the numbered-pipeline route for factor-separable GNN specs:

- **Step 11 (render)** — ``render.pomdp_processor`` composes factorised specs
  into a correct joint Kronecker model (per-factor action spaces, product
  action space) without crashing, and ``render.jax.render_gnn_to_jax`` routes
  them to the native factorized generator (sparse mean-field script).
- **Step 12 (execute)** — the rendered script runs standalone and writes
  ``jax_kronecker_factorized_v1`` ``simulation_results.json`` under
  ``GNN_OUTPUT_DIR`` with ``joint_materialized: False``.
- **Step 16 (analysis)** — ``analysis.framework_extractors.extract_jax_data``
  recognises the factorized schema (top-level, nested ``simulation_data``, or
  implementation-directory payloads) and maps per-factor traces into the
  standard analysis fields.

Pure Python — no Julia, no GPU, zero skips.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gnn.extract.pomdp_extractor import (
    extract_pomdp_from_content,
    extract_pomdp_from_file,
)
from gnn.render.jax import render_gnn_to_jax
from gnn.render.pomdp_processor import (
    _factor_action_counts,
    _is_kronecker_factorized_spec,
    pomdp_to_gnn_spec,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GNN_FILES = PROJECT_ROOT / "input" / "gnn_files"
SWARM_FILE = GNN_FILES / "multiagent" / "stigmergic_swarm.md"
FLAT_FILE = GNN_FILES / "discrete" / "simple_mdp.md"


def _require_pomdp(state: Any) -> Any:
    """Narrow the Optional parse result to a state space for type-checking."""
    assert state is not None, "extraction must yield a POMDP state space"
    return state


def _load_module(name: str, relative_path: Path) -> Any:
    """Load a scripts/ module (the tests/execute suite's established pattern)."""
    scripts_dir = str(relative_path.parent)
    need_cleanup = scripts_dir not in sys.path
    if need_cleanup:
        sys.path.insert(0, scripts_dir)
    try:
        spec = importlib.util.spec_from_file_location(name, relative_path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if need_cleanup and scripts_dir in sys.path:
            sys.path.remove(scripts_dir)


def _factorized_spec(factor_sizes: list[int], t: int = 10) -> dict[str, Any]:
    """Parse a generated factorized GNN file into a canonical renderer spec."""
    generator = _load_module(
        "pymdp_spec_generator",
        PROJECT_ROOT / "scripts" / "pymdp_spec_generator.py",
    )
    content = generator.generate_factorized_gnn_file(factor_sizes, t)
    pomdp = _require_pomdp(extract_pomdp_from_content(content, strict_validation=True))
    return pomdp_to_gnn_spec(pomdp)


def _col_norm(matrix: np.ndarray) -> np.ndarray:
    out = np.asarray(matrix, dtype=np.float64).copy()
    out /= out.sum(axis=0, keepdims=True)
    return out


def _canonical_b(raw: Any, num_actions: int) -> np.ndarray:
    """Mirror the processor's per-factor B canonicalisation."""
    tensor = np.asarray(raw, dtype=np.float64)
    if tensor.ndim == 2:
        tensor = tensor[:, :, np.newaxis]
    elif (
        tensor.ndim == 3
        and tensor.shape[0] == num_actions
        and tensor.shape[1] == tensor.shape[2]
    ):
        tensor = tensor.transpose(2, 1, 0)
    elif (
        tensor.ndim == 3 and tensor.shape[0] == 1 and tensor.shape[1] == tensor.shape[2]
    ):
        tensor = tensor.transpose(2, 1, 0)
    for action in range(tensor.shape[2]):
        tensor[:, :, action] = _col_norm(tensor[:, :, action])
    return tensor


class TestKroneckerDetection:
    """Spec-level detection of Kronecker-factorized (independent-action) models."""

    def test_factorized_spec_detected(self) -> None:
        pomdp = _require_pomdp(
            extract_pomdp_from_content(
                _load_module(
                    "pymdp_spec_generator",
                    PROJECT_ROOT / "scripts" / "pymdp_spec_generator.py",
                ).generate_factorized_gnn_file([3, 4], 10),
                strict_validation=True,
            )
        )
        assert _is_kronecker_factorized_spec(pomdp) is True

    def test_multi_agent_spec_not_detected(self) -> None:
        pomdp = _require_pomdp(
            extract_pomdp_from_file(SWARM_FILE, strict_validation=True)
        )
        assert _is_kronecker_factorized_spec(pomdp) is False

    def test_flat_spec_not_detected(self) -> None:
        pomdp = _require_pomdp(
            extract_pomdp_from_file(FLAT_FILE, strict_validation=True)
        )
        assert _is_kronecker_factorized_spec(pomdp) is False


class TestProcessorComposition:
    """Step 11 canonical joint composition for factorized specs."""

    def test_joint_shapes_and_action_space(self) -> None:
        spec = _factorized_spec([3, 4], 10)
        initial = spec["initialparameterization"]
        assert np.asarray(initial["A"]).shape == (12, 12)
        assert np.asarray(initial["B"]).shape == (12, 12, 12)
        assert len(initial["C"]) == 12
        assert len(initial["D"]) == 12
        assert spec["model_parameters"]["num_actions"] == 12
        assert spec["model_parameters"]["num_hidden_states"] == 12
        assert spec["matrix_provenance"]["B"]["kronecker_factorized"] is True
        assert spec["matrix_provenance"]["B"]["factor_action_counts"] == [3, 4]

    def test_joint_A_is_kronecker_product(self) -> None:
        spec = _factorized_spec([3, 4], 10)
        pomdp = _require_pomdp(
            extract_pomdp_from_content(
                _load_module(
                    "pymdp_spec_generator",
                    PROJECT_ROOT / "scripts" / "pymdp_spec_generator.py",
                ).generate_factorized_gnn_file([3, 4], 10),
                strict_validation=True,
            )
        )
        a0 = np.asarray(pomdp.matrices["A_f0"])
        a1 = np.asarray(pomdp.matrices["A_f1"])
        got = np.asarray(spec["initialparameterization"]["A"])
        assert np.allclose(_col_norm(got), _col_norm(np.kron(a0, a1)), atol=1e-10)

    def test_joint_B_is_exact_composition(self) -> None:
        spec = _factorized_spec([3, 4], 10)
        pomdp = _require_pomdp(
            extract_pomdp_from_content(
                _load_module(
                    "pymdp_spec_generator",
                    PROJECT_ROOT / "scripts" / "pymdp_spec_generator.py",
                ).generate_factorized_gnn_file([3, 4], 10),
                strict_validation=True,
            )
        )
        matrices = pomdp.matrices
        b0 = _canonical_b(matrices["B_f0"], 3)
        b1 = _canonical_b(matrices["B_f1"], 4)
        got = np.asarray(spec["initialparameterization"]["B"])
        for action in range(12):
            a0i = action % 3
            a1i = (action // 3) % 4
            for state in range(12):
                s0, s1 = state // 4, state % 4
                for previous in range(12):
                    p0, p1 = previous // 4, previous % 4
                    expected = b0[s0, p0, a0i] * b1[s1, p1, a1i]
                    assert np.isclose(got[state, previous, action], expected), (
                        f"B[{state},{previous},{action}] = {got[state, previous, action]} "
                        f"!= {expected}"
                    )

    def test_joint_B_columns_normalised(self) -> None:
        spec = _factorized_spec([3, 4], 10)
        b = np.asarray(spec["initialparameterization"]["B"])
        for action in range(b.shape[2]):
            assert np.allclose(b[:, :, action].sum(axis=0), 1.0)

    def test_swarm_composition_unchanged(self) -> None:
        pomdp = _require_pomdp(
            extract_pomdp_from_file(SWARM_FILE, strict_validation=True)
        )
        spec = pomdp_to_gnn_spec(pomdp)
        assert spec["model_parameters"]["num_hidden_states"] == 729
        assert spec["model_parameters"]["num_actions"] == 4
        assert np.asarray(spec["initialparameterization"]["B"]).shape == (729, 729, 4)


class TestRenderRouting:
    """Step 11 JAX renderer routing for factorized specs."""

    def test_factorized_spec_routes_to_factorized_generator(
        self, tmp_path: Path
    ) -> None:
        spec = _factorized_spec([3, 4], 10)
        output = tmp_path / "kronecker_model_jax.py"
        success, message, artifacts = render_gnn_to_jax(spec, output)
        assert success, message
        assert artifacts == [str(output)]
        script = output.read_text()
        assert "run_factorized_active_inference" in script
        assert "jax_kronecker_factorized_v1" in script
        assert "GNN_OUTPUT_DIR" in script
        assert "FactorizedPOMDP" in script

    def test_flat_spec_keeps_general_generator(self, tmp_path: Path) -> None:
        pomdp = _require_pomdp(
            extract_pomdp_from_file(FLAT_FILE, strict_validation=True)
        )
        spec = pomdp_to_gnn_spec(pomdp)
        output = tmp_path / "flat_model_jax.py"
        success, message, _ = render_gnn_to_jax(spec, output)
        assert success, message
        script = output.read_text()
        assert "run_factorized_active_inference" not in script


class TestScriptExecution:
    """Step 12: the rendered factorized script writes the v1 schema."""

    def test_rendered_script_executes_and_writes_schema(self, tmp_path: Path) -> None:
        spec = _factorized_spec([2, 2, 2], 5)
        script_path = tmp_path / "binary_factor_model_jax.py"
        success, message, _ = render_gnn_to_jax(spec, script_path)
        assert success, message

        out_dir = tmp_path / "execute_out"
        env = dict(os.environ)
        env["GNN_PROJECT_ROOT"] = str(PROJECT_ROOT)
        env["GNN_OUTPUT_DIR"] = str(out_dir)
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
            cwd=tmp_path,
        )
        assert result.returncode == 0, result.stderr

        results_file = out_dir / "simulation_results.json"
        assert results_file.exists()
        payload = json.loads(results_file.read_text(encoding="utf-8"))
        assert payload["schema_version"] == "jax_kronecker_factorized_v1"
        assert payload["model_kind"] == "factorized_kronecker"
        assert payload["num_factors"] == 3
        assert payload["num_timesteps"] == 5
        assert payload["model_parameters"]["joint_state_space_size"] == 8
        assert payload["model_parameters"]["joint_materialized"] is False
        assert payload["validation"]["all_valid"] is True
        assert len(payload["beliefs_by_factor"]) == 3
        assert all(len(trace) == 5 for trace in payload["beliefs_by_factor"].values())


class TestAnalysisExtraction:
    """Step 16: analysis consumes the factorized schema."""

    @pytest.fixture()
    def payload(self, tmp_path: Path) -> dict[str, Any]:
        spec = _factorized_spec([3, 4], 10)
        script_path = tmp_path / "gnn_test_analysis_model_jax.py"
        success, _, _ = render_gnn_to_jax(spec, script_path)
        assert success
        out_dir = tmp_path / "gnn_test_analysis_out"
        env = dict(os.environ)
        env["GNN_PROJECT_ROOT"] = str(PROJECT_ROOT)
        env["GNN_OUTPUT_DIR"] = str(out_dir)
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
            cwd=tmp_path,
        )
        assert result.returncode == 0, result.stderr
        return dict(
            json.loads(
                (out_dir / "simulation_results.json").read_text(encoding="utf-8")
            )
        )

    def test_top_level_payload(self, payload: dict[str, Any]) -> None:
        from gnn.analysis.framework_extractors import extract_jax_data

        extracted = extract_jax_data(payload)
        assert extracted["schema_version"] == "jax_kronecker_factorized_v1"
        assert extracted["model_kind"] == "factorized_kronecker"
        assert extracted["factors"] == ["factor0", "factor1"]
        assert len(extracted["beliefs"]) == 2
        assert all(len(trace) == 10 for trace in extracted["beliefs"])
        assert extracted["model_parameters"]["joint_state_space_size"] == 12
        assert extracted["model_parameters"]["joint_materialized"] is False
        assert extracted["validation"]["all_valid"] is True

    def test_free_energy_is_per_step_total(self, payload: dict[str, Any]) -> None:
        from gnn.analysis.framework_extractors import extract_jax_data

        extracted = extract_jax_data(payload)
        efe_by_factor = payload["efe_per_factor"]
        expected = [sum(step) for step in zip(*efe_by_factor.values())]
        assert extracted["free_energy"] == expected
        assert len(extracted["free_energy"]) == 10

    def test_nested_simulation_data_dispatch(self, payload: dict[str, Any]) -> None:
        from gnn.analysis.framework_extractors import extract_jax_data

        extracted = extract_jax_data({"simulation_data": payload})
        assert extracted["schema_version"] == "jax_kronecker_factorized_v1"

    def test_implementation_directory_dispatch(
        self, payload: dict[str, Any], tmp_path: Path
    ) -> None:
        from gnn.analysis.framework_extractors import extract_jax_data

        sim_dir = tmp_path / "simulation_data"
        sim_dir.mkdir()
        (sim_dir / "simulation_results.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        extracted = extract_jax_data({"implementation_directory": str(tmp_path)})
        assert extracted["schema_version"] == "jax_kronecker_factorized_v1"
        assert extracted["factors"] == ["factor0", "factor1"]

    def test_pymdp_payload_falls_back(self) -> None:
        from gnn.analysis.framework_extractors import extract_jax_data

        extracted = extract_jax_data(
            {"schema_version": "pymdp_simulation_v1", "beliefs": [0.5], "actions": [0]}
        )
        # The pymdp path returns the flat schema (no schema_version key) and
        # reads beliefs from ``beliefs_by_factor.joint_state``.
        assert "schema_version" not in extracted
        assert extracted["beliefs"] == []
