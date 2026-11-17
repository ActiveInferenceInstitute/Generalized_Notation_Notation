#!/usr/bin/env python3
"""
PyMDP Simulation Tests

Comprehensive test suite for PyMDP simulations configured from GNN specifications.
Tests both the simulation functionality and GNN integration.

Features:
- GNN configuration testing
- PyMDP simulation validation
- Visualization testing
- Pipeline integration validation

Author: GNN PyMDP Integration
Date: 2024
"""

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gnn.analysis.pymdp.visualizer import PyMDPVisualizer
from gnn.execute.pymdp.pymdp_simulation import PyMDPSimulation
from gnn.execute.pymdp.pymdp_utils import convert_numpy_for_json, safe_json_dump


# Pipeline imports
# Local simple temp helpers to avoid cross-package test import issues
def create_test_temp_dir(prefix: str) -> Path:
    d = Path(tempfile.mkdtemp(prefix=f"{prefix}_"))
    return d


def cleanup_test_temp_dir(path: Path) -> None:
    try:
        import shutil

        shutil.rmtree(path)
    except Exception:
        pass


@pytest.mark.needs_pymdp
class TestPyMDPSimulation(unittest.TestCase):
    """Test PyMDP simulation functionality with GNN integration."""

    def setUp(self) -> None:
        """Set up test environment."""
        self.test_dir = create_test_temp_dir("pymdp_simulation_test")

        # Default GNN configuration for testing
        self.gnn_config = {
            "states": ["s1", "s2", "s3", "s4"],
            "observations": ["o1", "o2"],
            "actions": ["up", "down", "left", "right"],
            "model_type": "discrete_pomdp",
            "environment_type": "gridworld",
            "grid_size": [2, 2],
            "agent_position": [0, 0],
            "goal_position": [1, 1],
            "wall_positions": [],
            "preferences": {
                "goal_reward": 10.0,
                "step_cost": -0.1,
                "wall_penalty": -1.0,
            },
            "learning_rate": 0.5,
            "exploration_factor": 0.1,
        }

        # Alternative minimal configuration
        self.minimal_config = {
            "states": 4,
            "observations": 2,
            "actions": 4,
            "model_type": "discrete_pomdp",
        }
        self.initialparameterization = {
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
        }
        self.gnn_config["initialparameterization"] = self.initialparameterization
        self.minimal_config["initialparameterization"] = self.initialparameterization

    def test_simulation_creation_with_gnn_config(self) -> None:
        """Test creating PyMDP simulation from GNN configuration."""
        simulation = PyMDPSimulation(
            gnn_config=self.gnn_config, output_dir=self.test_dir
        )

        self.assertEqual(simulation.num_states, 4)
        self.assertEqual(simulation.num_observations, 2)
        self.assertEqual(simulation.num_actions, 4)
        self.assertIsNotNone(simulation.agent)

    def test_simulation_creation_with_minimal_config(self) -> None:
        """Test creating PyMDP simulation with minimal configuration."""
        simulation = PyMDPSimulation(
            gnn_config=self.minimal_config, output_dir=self.test_dir
        )

        self.assertEqual(simulation.num_states, 4)
        self.assertEqual(simulation.num_observations, 2)
        self.assertEqual(simulation.num_actions, 4)
        self.assertIsNotNone(simulation.agent)

    def test_simulation_run(self) -> None:
        """Test running a complete simulation."""
        simulation = PyMDPSimulation(
            gnn_config=self.gnn_config, output_dir=self.test_dir
        )

        results = simulation.run_simulation(num_timesteps=10)

        self.assertIn("observations", results)
        self.assertIn("actions", results)
        self.assertIn("beliefs", results)
        self.assertIn("performance", results)
        self.assertEqual(len(results["observations"]), 10)
        self.assertEqual(len(results["actions"]), 10)

    def test_matrix_construction(self) -> None:
        """Test that PyMDP matrices are properly constructed."""
        simulation = PyMDPSimulation(
            gnn_config=self.gnn_config, output_dir=self.test_dir
        )

        # Test A matrix properties
        A = simulation.A[0]
        self.assertEqual(A.shape, (2, 4))  # observations x states
        self.assertTrue(np.allclose(A.sum(axis=0), 1.0))  # columns sum to 1

        # Test B matrix properties
        B = simulation.B[0]
        self.assertEqual(B.shape, (4, 4, 4))  # states x states x actions
        for action in range(4):
            self.assertTrue(np.allclose(B[:, :, action].sum(axis=0), 1.0))

        # Test C vector properties
        C = simulation.C[0]
        self.assertEqual(len(C), 2)  # number of observations

        # Test D vector properties
        D = simulation.D[0]
        self.assertEqual(len(D), 4)  # number of states
        self.assertTrue(np.allclose(D.sum(), 1.0))  # sums to 1

    def test_serialization(self) -> None:
        """Test that simulation results can be serialized."""
        simulation = PyMDPSimulation(
            gnn_config=self.gnn_config, output_dir=self.test_dir
        )

        results = simulation.run_simulation(num_timesteps=5)

        # Test JSON serialization
        json_file = self.test_dir / "test_results.json"
        safe_json_dump(results, json_file)
        self.assertTrue(json_file.exists())

        # Load and verify
        with open(json_file) as f:
            loaded_results = json.load(f)

        self.assertIn("observations", loaded_results)
        self.assertIn("actions", loaded_results)

    def test_visualization_creation(self) -> None:
        """Test that visualizations can be created."""
        simulation = PyMDPSimulation(
            gnn_config=self.gnn_config, output_dir=self.test_dir
        )

        results = simulation.run_simulation(num_timesteps=10)

        visualizer = PyMDPVisualizer(config=self.gnn_config, output_dir=self.test_dir)

        # Test visualization creation (should not raise errors)
        try:
            visualizer.plot_belief_evolution(results["beliefs"])
            visualizer.plot_action_sequence(results["actions"])
            visualizer.plot_performance_metrics(results["performance"])
            viz_success = True
        except Exception as e:
            viz_success = False
            print(f"Visualization error: {e}")

        self.assertTrue(viz_success)

    def test_gnn_parameter_integration(self) -> None:
        """Test that GNN parameters are properly integrated."""
        # Test with explicit state names
        config_with_names: dict[str, Any] = {
            "states": ["location_00", "location_01", "location_10", "location_11"],
            "observations": ["visible", "occluded"],
            "actions": ["move_north", "move_south", "move_east", "move_west"],
            "model_type": "discrete_pomdp",
            "initialparameterization": self.initialparameterization,
            "preferences": {"goal_reward": 5.0},
        }

        simulation = PyMDPSimulation(
            gnn_config=config_with_names, output_dir=self.test_dir
        )

        self.assertEqual(simulation.num_states, 4)
        self.assertEqual(simulation.num_observations, 2)
        self.assertEqual(simulation.num_actions, 4)

        # Verify state names are stored
        self.assertEqual(simulation.state_names, config_with_names["states"])
        self.assertEqual(
            simulation.observation_names, config_with_names["observations"]
        )
        self.assertEqual(simulation.action_names, config_with_names["actions"])

    def test_simulation_execution_metadata_accelerator_record(self) -> None:
        """Test that execution_metadata records accelerator_type and vectorization."""
        from gnn.execute.pymdp.pymdp_simulation import create_demo_pymdp_simulation

        simulation = create_demo_pymdp_simulation()
        results = simulation.run_simulation(num_timesteps=1)
        self.assertIn("execution_metadata", results)
        meta = results["execution_metadata"]
        self.assertIn("accelerator_type", meta)
        self.assertIn("duration_seconds", meta)
        self.assertIn("vectorization", meta)

    def tearDown(self) -> None:
        """Clean up test environment."""
        cleanup_test_temp_dir(self.test_dir)


class TestPyMDPUtils(unittest.TestCase):
    """Test PyMDP utility functions."""

    def test_numpy_conversion(self) -> None:
        """Test numpy to JSON conversion utilities."""
        # Test various numpy types
        test_data: dict[str, Any] = {
            "int_array": np.array([1, 2, 3]),
            "float_array": np.array([1.5, 2.5, 3.5]),
            "nested": {
                "matrix": np.array([[1, 2], [3, 4]]),
                "scalar": np.float64(3.14),
            },
        }

        converted = convert_numpy_for_json(test_data)

        # Should be JSON serializable
        json_str = json.dumps(converted)
        self.assertIsInstance(json_str, str)

        # Load back and verify
        loaded = json.loads(json_str)
        self.assertIn("int_array", loaded)
        self.assertIn("float_array", loaded)
        self.assertIn("nested", loaded)


if __name__ == "__main__":
    # Run specific test methods or all tests
    unittest.main(argv=[""], exit=False, verbosity=2)
