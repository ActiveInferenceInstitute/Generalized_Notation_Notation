"""
Integration tests for the POMDP pipeline: render → execute → analysis.

Validates that a GNN specification with explicit POMDP matrices can be:
1. Rendered to PyMDP code
2. Executed as a PyMDPSimulation
3. Analyzed via Active Inference metrics
"""

import json
from typing import Any

import numpy as np
import pytest

from gnn.analysis.post_simulation import (
    analyze_active_inference_metrics,
    compute_shannon_entropy,
    compute_variational_free_energy,
    extract_pymdp_data,
)
from gnn.execute.pymdp.pymdp_simulation import PyMDPSimulation
from gnn.execute.pymdp.pymdp_utils import safe_json_dump
from gnn.render.processor import render_gnn_spec

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def pomdp_gnn_spec() -> Any:
    """A GNN spec representing a simple 3-state, 3-obs, 2-action POMDP."""
    return {
        "name": "test_pomdp_model",
        "model_name": "test_pomdp_model",
        "states": ["s0", "s1", "s2"],
        "observations": ["o0", "o1", "o2"],
        "actions": ["left", "right"],
        "parameters": {
            "A": [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]],
            "B_left": [[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.0, 0.1, 0.9]],
            "B_right": [[0.1, 0.0, 0.9], [0.1, 0.8, 0.1], [0.8, 0.2, 0.0]],
            "C": [1.0, 0.0, -1.0],
            "D": [1.0, 0.0, 0.0],
            "E": [0.5, 0.5],
        },
        "initialparameterization": {
            "A": [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]],
            "B": [
                [[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.0, 0.1, 0.9]],
                [[0.1, 0.0, 0.9], [0.1, 0.8, 0.1], [0.8, 0.2, 0.0]],
            ],
            "C": [1.0, 0.0, -1.0],
            "D": [1.0, 0.0, 0.0],
            "E": [0.5, 0.5],
        },
        "model_parameters": {"num_hidden_states": 3, "num_obs": 3, "num_actions": 2},
        "num_states": 3,
        "num_observations": 3,
        "num_actions": 2,
    }


@pytest.fixture
def pymdp_gnn_config() -> Any:
    """A GNN config dict suitable for PyMDPSimulation initialization.
    Uses the same format as the existing passing tests in test_execute_pymdp_simulation.py.
    """
    return {
        "states": ["s0", "s1", "s2", "s3"],
        "observations": ["o0", "o1"],
        "actions": ["a0", "a1", "a2", "a3"],
        "model_type": "discrete_pomdp",
        "initialparameterization": {
            "A": [
                [0.85, 0.15, 0.85, 0.15],
                [0.15, 0.85, 0.15, 0.85],
            ],
            "B": [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0],
                ],
                [
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                ],
            ],
            "C": [0.0, 1.0],
            "D": [0.25, 0.25, 0.25, 0.25],
        },
    }


# =============================================================================
# Render Tests
# =============================================================================


class TestRenderPOMDP:
    """Test that GNN POMDP specs can be rendered to framework code."""

    def test_render_to_pymdp(self, pomdp_gnn_spec: Any, tmp_path: Any) -> Any:
        """Rendering a POMDP spec to PyMDP should produce a Python file."""
        success, msg, artifacts = render_gnn_spec(pomdp_gnn_spec, "pymdp", tmp_path)
        assert success is True, f"PyMDP rendering failed: {msg}"

        # Verify at least one .py file was created
        py_files = list(tmp_path.rglob("*.py"))
        assert len(py_files) > 0, "No .py files generated"

    def test_render_to_discopy(self, pomdp_gnn_spec: Any, tmp_path: Any) -> Any:
        """Rendering a POMDP spec with declared wiring to DisCoPy should succeed."""
        spec = dict(pomdp_gnn_spec)
        spec["variables"] = [
            {"name": "s0", "dimensions": [3]},
            {"name": "o0", "dimensions": [3]},
        ]
        spec["connections"] = [
            {"source": "s0", "relation": ">", "target": "o0"},
        ]
        success, msg, artifacts = render_gnn_spec(spec, "discopy", tmp_path)
        assert success is True, f"DisCoPy rendering failed: {msg}"

    def test_render_to_rxinfer(self, pomdp_gnn_spec: Any, tmp_path: Any) -> Any:
        """Rendering a POMDP spec to RxInfer should succeed."""
        success, msg, artifacts = render_gnn_spec(pomdp_gnn_spec, "rxinfer", tmp_path)
        assert success is True, f"RxInfer rendering failed: {msg}"

    def test_render_to_activeinference_jl(
        self, pomdp_gnn_spec: Any, tmp_path: Any
    ) -> Any:
        """Rendering a POMDP spec to ActiveInference.jl should succeed."""
        success, msg, artifacts = render_gnn_spec(
            pomdp_gnn_spec, "activeinference_jl", tmp_path
        )
        assert success is True, f"ActiveInference.jl rendering failed: {msg}"


# =============================================================================
# Execute Tests
# =============================================================================


@pytest.mark.needs_pymdp
class TestExecutePOMDP:
    """Test that POMDP simulations run and produce valid results."""

    def test_simulation_creation(self, pymdp_gnn_config: Any, tmp_path: Any) -> Any:
        """PyMDPSimulation should initialize from GNN config."""
        sim = PyMDPSimulation(gnn_config=pymdp_gnn_config, output_dir=tmp_path)
        assert sim is not None
        assert sim.num_states == 4
        assert sim.num_observations == 2
        assert sim.num_actions == 4
        assert sim.agent is not None

    def test_simulation_run_produces_results(
        self, pymdp_gnn_config: Any, tmp_path: Any
    ) -> Any:
        """Running a simulation should produce observations, actions, and beliefs."""
        sim = PyMDPSimulation(gnn_config=pymdp_gnn_config, output_dir=tmp_path)
        results = sim.run_simulation(num_timesteps=10)

        assert results is not None
        assert "observations" in results
        assert "actions" in results
        assert "beliefs" in results

        assert len(results["observations"]) == 10
        assert len(results["actions"]) == 10

    def test_simulation_beliefs_valid(
        self, pymdp_gnn_config: Any, tmp_path: Any
    ) -> Any:
        """Beliefs at each timestep should be valid probability distributions."""
        sim = PyMDPSimulation(gnn_config=pymdp_gnn_config, output_dir=tmp_path)
        results = sim.run_simulation(num_timesteps=10)

        beliefs = results.get("beliefs", [])
        assert len(beliefs) > 0, "No beliefs recorded"
        for t, b in enumerate(beliefs):
            b_arr = np.asarray(b).flatten()
            assert np.all(b_arr >= 0), f"Negative belief at t={t}"
            total = np.sum(b_arr)
            assert abs(total - 1.0) < 0.01, (
                f"Beliefs at t={t} sum to {total}, expected ~1.0"
            )

    def test_serialization_roundtrip(self, pymdp_gnn_config: Any, tmp_path: Any) -> Any:
        """Simulation results should be JSON-serializable via safe_json_dump."""
        sim = PyMDPSimulation(gnn_config=pymdp_gnn_config, output_dir=tmp_path)
        results = sim.run_simulation(num_timesteps=5)

        output_file = tmp_path / "sim_results.json"
        safe_json_dump(results, output_file)
        assert output_file.exists()

        with open(output_file, "r") as f:
            loaded = json.load(f)
        assert "observations" in loaded
        assert "actions" in loaded


# =============================================================================
# Analysis Tests
# =============================================================================


@pytest.mark.needs_pymdp
class TestAnalyzePOMDP:
    """Test the analysis pipeline on simulation results."""

    def test_analyze_simulation_output(
        self, pymdp_gnn_config: Any, tmp_path: Any
    ) -> Any:
        """Running analyze_active_inference_metrics on sim results should produce
        meaningful statistics."""
        sim = PyMDPSimulation(gnn_config=pymdp_gnn_config, output_dir=tmp_path)
        results = sim.run_simulation(num_timesteps=15)

        beliefs_list = [np.asarray(b).flatten().tolist() for b in results["beliefs"]]
        actions_int = [int(a) for a in results["actions"]]
        fe = [float(5.0 - 0.1 * t) for t in range(len(beliefs_list))]

        analysis = analyze_active_inference_metrics(
            beliefs_list, fe, actions_int, "pipeline_test"
        )

        assert analysis["model_name"] == "pipeline_test"
        assert analysis["num_timesteps"] == len(beliefs_list)
        assert "metrics" in analysis
        assert "belief_entropy" in analysis["metrics"]

    def test_entropy_computation_on_sim_beliefs(
        self, pymdp_gnn_config: Any, tmp_path: Any
    ) -> Any:
        """Shannon entropy should be computable on simulation beliefs."""
        sim = PyMDPSimulation(gnn_config=pymdp_gnn_config, output_dir=tmp_path)
        results = sim.run_simulation(num_timesteps=10)

        for b in results["beliefs"]:
            b_arr = np.asarray(b).flatten()
            entropy = compute_shannon_entropy(b_arr)
            assert entropy >= 0, "Entropy should be non-negative"
            assert np.isfinite(entropy), "Entropy should be finite"

    def test_extract_pymdp_data_structure(
        self, pymdp_gnn_config: Any, tmp_path: Any
    ) -> Any:
        """extract_pymdp_data should return a well-structured dict."""
        sim = PyMDPSimulation(gnn_config=pymdp_gnn_config, output_dir=tmp_path)
        results = sim.run_simulation(num_timesteps=10)

        execution_result: dict[str, Any] = {
            "framework": "pymdp",
            "simulation_data": results,
            "implementation_directory": str(tmp_path),
        }

        extracted = extract_pymdp_data(execution_result)
        assert isinstance(extracted, dict)
        # Should have standard Active Inference fields
        for key in [
            "traces",
            "free_energy",
            "beliefs",
            "states",
            "observations",
            "actions",
        ]:
            assert key in extracted, f"Missing key: {key}"


# =============================================================================
# End-to-End Pipeline Test
# =============================================================================


@pytest.mark.needs_pymdp
class TestEndToEndPOMDPPipeline:
    """Full pipeline: render → execute → analyze."""

    def test_full_pipeline_flow(
        self, pomdp_gnn_spec: Any, pymdp_gnn_config: Any, tmp_path: Any
    ) -> Any:
        """Exercise the complete render → execute → analyze path."""
        # 1. Render
        render_dir = tmp_path / "rendered"
        success, msg, _ = render_gnn_spec(pomdp_gnn_spec, "pymdp", render_dir)
        assert success, f"Render failed: {msg}"

        # 2. Execute
        sim = PyMDPSimulation(gnn_config=pymdp_gnn_config, output_dir=tmp_path)
        results = sim.run_simulation(num_timesteps=15)
        assert len(results.get("observations", [])) > 0, (
            "Simulation produced no observations"
        )

        # 3. Analyze
        beliefs = [np.asarray(b).flatten().tolist() for b in results.get("beliefs", [])]
        actions = [int(a) for a in results.get("actions", [])]
        fe = [5.0 - 0.1 * t for t in range(len(beliefs))]

        assert len(beliefs) > 0, "No beliefs to analyze"

        analysis = analyze_active_inference_metrics(
            beliefs, fe, actions[: len(beliefs)], "e2e_test"
        )
        assert analysis["num_timesteps"] > 0
        assert "belief_entropy" in analysis["metrics"]
        assert analysis["metrics"]["belief_entropy"]["mean"] > 0

        # 4. Verify VFE computation with simulation data
        b_arr = np.array(beliefs[0])
        A = np.eye(len(b_arr))
        obs = np.zeros(len(b_arr))
        obs[0] = 1.0
        vfe = compute_variational_free_energy(obs, b_arr, A)
        assert np.isfinite(vfe), "VFE should be finite"
