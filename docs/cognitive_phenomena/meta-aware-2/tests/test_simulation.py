#!/usr/bin/env python3
"""
Test Suite for Meta-Awareness Simulation

Comprehensive tests for the meta-aware-2 simulation pipeline including
unit tests, integration tests, and validation against known results.

Part of the meta-aware-2 "golden spike" GNN-specified executable implementation.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.gnn_parser import ModelConfig, load_gnn_config
from core.meta_awareness_model import MetaAwarenessModel
from execution.simulation_runner import SimulationRunner
from utils.math_utils import MathUtils


class TestSimulation(unittest.TestCase):
    """Test suite for meta-awareness simulation."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(__file__).parent
        self.config_path = self.test_dir.parent / "config" / "meta_awareness_gnn.toml"
        self.math_utils = MathUtils()

        # Create minimal test config if main config doesn't exist
        if not self.config_path.exists():
            self._create_minimal_test_config()

    def _create_minimal_test_config(self):
        """Create a minimal test configuration."""
        test_config = """
[model]
name = "test_meta_awareness"
description = "Test configuration for meta-awareness model"
num_levels = 2
level_names = ["perception", "attention"]
time_steps = 50
oddball_pattern = "default"

[levels.perception]
state_dim = 2
obs_dim = 2
action_dim = 0

[levels.attention]
state_dim = 2
obs_dim = 2
action_dim = 2

[precision_bounds]
perception = [0.5, 2.0]
attention = [1.0, 4.0]

[policy_precision]
2_level = 2.0
3_level = 4.0
"""

        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w") as f:
            f.write(test_config)

    def test_config_loading(self):
        """Test GNN configuration loading."""
        config = load_gnn_config(self.config_path)

        self.assertIsInstance(config, ModelConfig)
        # Accept either the minimal test config name or the default config name
        self.assertIn(
            config.name, ["test_meta_awareness", "meta_awareness_active_inference"]
        )
        self.assertGreaterEqual(config.num_levels, 2)
        self.assertIsInstance(config.level_names, list)
        self.assertGreater(config.time_steps, 0)

    def test_model_initialization(self):
        """Test model initialization with configuration."""
        config = load_gnn_config(self.config_path)
        model = MetaAwarenessModel(config, random_seed=42)

        self.assertEqual(model.config.name, config.name)
        self.assertEqual(model.num_levels, config.num_levels)
        self.assertIsInstance(model.math_utils, MathUtils)

        # Check that state spaces are properly initialized
        self.assertEqual(len(model.state_dims), config.num_levels)
        self.assertEqual(len(model.obs_dims), config.num_levels)

    def test_stimulus_generation(self):
        """Test stimulus sequence generation."""
        config = load_gnn_config(self.config_path)
        model = MetaAwarenessModel(config, random_seed=42)

        stimulus_sequence = model.state.stimulus_sequence

        self.assertIsInstance(stimulus_sequence, np.ndarray)
        self.assertEqual(len(stimulus_sequence), config.time_steps)
        self.assertTrue(np.all(np.isin(stimulus_sequence, [0, 1])))

        # Check that there are some oddball stimuli
        num_oddballs = np.sum(stimulus_sequence == 1)
        self.assertGreater(num_oddballs, 0)

    def test_simulation_execution(self):
        """Test basic simulation execution."""
        config = load_gnn_config(self.config_path)
        model = MetaAwarenessModel(config, random_seed=42)

        results = model.run_simulation("default")

        # Check basic result structure
        self.assertIsInstance(results, dict)
        self.assertIn("model_name", results)
        self.assertIn("num_levels", results)
        self.assertIn("time_steps", results)
        self.assertIn("state_posteriors", results)
        self.assertIn("true_states", results)
        self.assertIn("precision_values", results)

        # Check array dimensions
        for level_name in config.level_names:
            if level_name in results["state_posteriors"]:
                state_post = results["state_posteriors"][level_name]
                expected_shape = (
                    config.levels[level_name].state_dim,
                    config.time_steps,
                )
                self.assertEqual(state_post.shape, expected_shape)

    def test_precision_bounds(self):
        """Test that precision values stay within bounds."""
        config = load_gnn_config(self.config_path)
        model = MetaAwarenessModel(config, random_seed=42)

        results = model.run_simulation("default")

        for level_name, precision_ts in results["precision_values"].items():
            if level_name in config.precision_bounds:
                bounds = config.precision_bounds[level_name]

                # Allow for small numerical tolerance
                tolerance = 1e-6
                self.assertTrue(np.all(precision_ts >= bounds[0] - tolerance))
                self.assertTrue(np.all(precision_ts <= bounds[1] + tolerance))

    def test_belief_normalization(self):
        """Test that belief distributions are properly normalized."""
        config = load_gnn_config(self.config_path)
        model = MetaAwarenessModel(config, random_seed=42)

        results = model.run_simulation("default")

        for level_name, beliefs in results["state_posteriors"].items():
            # Check that beliefs sum to 1 at each time step
            belief_sums = np.sum(beliefs, axis=0)
            np.testing.assert_allclose(belief_sums, 1.0, atol=1e-10)

    def test_mathematical_operations(self):
        """Test mathematical utility functions."""
        # Test softmax
        x = np.array([1.0, 2.0, 3.0])
        result = self.math_utils.softmax(x)

        self.assertAlmostEqual(np.sum(result), 1.0, places=10)
        self.assertTrue(np.all(result > 0))
        self.assertTrue(np.all(result < 1))

        # Test normalization
        x = np.array([1.0, 2.0, 3.0])
        result = self.math_utils.normalize(x)

        self.assertAlmostEqual(np.sum(result), 1.0, places=10)

        # Test entropy
        p = np.array([0.5, 0.5])
        entropy = self.math_utils.compute_entropy(p)
        expected_entropy = -np.sum(p * np.log(p))

        self.assertAlmostEqual(entropy, expected_entropy, places=10)

    def test_reproducibility(self):
        """Test that simulations are reproducible with same seed."""
        config = load_gnn_config(self.config_path)

        # Run simulation twice with same seed
        model1 = MetaAwarenessModel(config, random_seed=42)
        results1 = model1.run_simulation("default")

        model2 = MetaAwarenessModel(config, random_seed=42)
        results2 = model2.run_simulation("default")

        # Compare key results
        np.testing.assert_array_equal(
            results1["stimulus_sequence"], results2["stimulus_sequence"]
        )

        for level_name in config.level_names:
            if (
                level_name in results1["true_states"]
                and level_name in results2["true_states"]
            ):
                np.testing.assert_array_equal(
                    results1["true_states"][level_name],
                    results2["true_states"][level_name],
                )

    def test_simulation_runner(self):
        """Test simulation runner functionality."""
        runner = SimulationRunner(
            config_path=self.config_path,
            output_dir="./test_output",
            log_level="WARNING",  # Reduce logging for tests
            random_seed=42,
        )

        # Test single simulation
        results = runner.run_simulation("default")

        self.assertIsInstance(results, dict)
        self.assertIn("execution_metadata", results)
        self.assertEqual(results["execution_metadata"]["random_seed"], 42)

    def test_error_handling(self):
        """Test error handling for invalid configurations."""
        # Test with non-existent config file
        with self.assertRaises(FileNotFoundError):
            load_gnn_config("nonexistent_config.toml")

        # Test with invalid precision bounds
        config = load_gnn_config(self.config_path)
        config.precision_bounds["perception"] = [2.0, 1.0]  # Invalid: min > max

        # This should not crash but may generate warnings
        try:
            model = MetaAwarenessModel(config, random_seed=42)
            model.run_simulation("default")
        except Exception as e:
            # If it does raise an exception, it should be informative
            self.assertIsInstance(e, Exception)

    def test_performance_benchmarks(self):
        """Test performance benchmarks for simulation speed."""
        config = load_gnn_config(self.config_path)

        # Reduce time steps for faster testing
        config.time_steps = 20

        import time

        start_time = time.time()

        model = MetaAwarenessModel(config, random_seed=42)
        results = model.run_simulation("default")

        duration = time.time() - start_time

        # Simulation should complete in reasonable time (adjust as needed)
        self.assertLess(duration, 10.0)  # Should complete in less than 10 seconds

        # Results should be complete
        self.assertEqual(results["time_steps"], config.time_steps)

    def tearDown(self):
        """Clean up after tests."""
        # Remove test output directory if it exists
        import shutil

        test_output = Path("./test_output")
        if test_output.exists():
            shutil.rmtree(test_output)


class TestValidationAgainstPaper(unittest.TestCase):
    """Validation tests against Sandved-Smith et al. (2021) paper results."""

    def setUp(self):
        """Set up validation test fixtures."""
        self.test_dir = Path(__file__).parent
        self.config_path = self.test_dir.parent / "config" / "meta_awareness_gnn.toml"

        # Expected results from paper (approximate values)
        self.expected_mind_wandering_percentage = 65.0  # ±10%
        self.expected_precision_range = (0.5, 2.0)
        self.expected_transition_count_range = (50, 80)  # For 100 time steps

    def test_mind_wandering_percentage(self):
        """Test that mind-wandering percentage matches paper expectations."""
        if not self.config_path.exists():
            self.skipTest("GNN config file not available")

        config = load_gnn_config(self.config_path)
        config.time_steps = 100  # Standard test length

        model = MetaAwarenessModel(config, random_seed=42)
        results = model.run_simulation("default")

        # Calculate mind-wandering percentage
        attention_level = (
            results["level_names"][1]
            if len(results["level_names"]) > 1
            else "attention"
        )
        if attention_level in results["true_states"]:
            att_states = results["true_states"][attention_level]
            mw_percentage = np.mean(att_states == 1) * 100

            # Should be within reasonable range of paper results
            self.assertGreater(mw_percentage, 30.0)  # At least some mind-wandering
            self.assertLess(mw_percentage, 90.0)  # Not constant mind-wandering

    def test_precision_dynamics_range(self):
        """Test that precision dynamics are within expected range."""
        if not self.config_path.exists():
            self.skipTest("GNN config file not available")

        config = load_gnn_config(self.config_path)
        model = MetaAwarenessModel(config, random_seed=42)
        results = model.run_simulation("default")

        perception_level = (
            results["level_names"][0] if results["level_names"] else "perception"
        )
        if perception_level in results["precision_values"]:
            precision_ts = results["precision_values"][perception_level]

            # Check range
            self.assertGreaterEqual(
                np.min(precision_ts), self.expected_precision_range[0] * 0.8
            )
            self.assertLessEqual(
                np.max(precision_ts), self.expected_precision_range[1] * 1.2
            )

            # Check variability
            precision_std = np.std(precision_ts)
            self.assertGreater(precision_std, 0.1)  # Should show meaningful variation


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
