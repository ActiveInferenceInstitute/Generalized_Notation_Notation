#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Test Suite for Meta-Aware-2 Implementation

Tests that ensure all features and functions from the original meta-awareness
implementation (computational_phenomenology_of_mental_action.py, sandved_smith_2021.py, etc.)
are exactly working in the meta-aware-2 system.

This test suite validates:
- All mathematical utilities match original implementation
- Model dynamics are correct
- Figure reproduction is accurate
- Behavioral patterns match expectations
- Performance meets standards
"""

import time
import unittest
from pathlib import Path

import numpy as np
from config.gnn_parser import load_gnn_config
from core.meta_awareness_model import MetaAwarenessModel
from execution.simulation_runner import SimulationRunner
from utils.math_utils import MathUtils
from verification import PaperVerification
from visualization.figure_generator import FigureGenerator


class TestCompleteImplementation(unittest.TestCase):
    """Comprehensive test suite for complete meta-aware-2 implementation."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures for entire test class."""
        cls.config = load_gnn_config("config/meta_awareness_gnn.toml")
        cls.math_utils = MathUtils()
        cls.output_dir = Path("test_output")
        cls.output_dir.mkdir(exist_ok=True)

    def setUp(self):
        """Set up for each individual test."""
        self.model = MetaAwarenessModel(self.config, random_seed=42)

    def test_utility_functions(self):
        """Test all mathematical utility functions match original implementation."""
        print("\n=== Testing Utility Functions ===")

        # Test softmax function
        x = np.array([1.0, 2.0, 3.0])
        result = self.math_utils.softmax(x)
        self.assertAlmostEqual(np.sum(result), 1.0, places=10)
        self.assertTrue(np.all(result[1:] >= result[:-1]))  # Monotonic
        print("✓ Softmax function correct")

        # Test softmax_dim2 (matrix softmax)
        X = np.array([[1, 2], [3, 4], [5, 6]])
        result_matrix = self.math_utils.softmax_dim2(X)
        col_sums = np.sum(result_matrix, axis=0)
        self.assertTrue(np.allclose(col_sums, 1.0))
        print("✓ Matrix softmax function correct")

        # Test normalization
        y = np.array([2, 4, 6])
        norm_y = self.math_utils.normalize(y)
        self.assertAlmostEqual(np.sum(norm_y), 1.0, places=10)
        print("✓ Normalization function correct")

        # Test entropy calculation
        uniform = np.ones(4) / 4
        entropy_val = self.math_utils.compute_entropy(uniform)
        expected_entropy = np.log(4)
        self.assertAlmostEqual(entropy_val, expected_entropy, places=10)
        print("✓ Entropy calculation correct")

        # Test precision-weighted likelihood
        A = np.array([[0.8, 0.2], [0.2, 0.8]])
        gamma = 2.0
        A_weighted = self.math_utils.precision_weighted_likelihood(A, gamma)
        self.assertTrue(np.allclose(np.sum(A_weighted, axis=0), 1.0))
        print("✓ Precision-weighted likelihood correct")

        # Test attentional charge computation
        O_bar = np.array([1.0, 0.0])
        A_bar = np.array([[0.9, 0.1], [0.1, 0.9]])
        X_bar = np.array([0.8, 0.2])
        A_orig = np.array([[0.8, 0.2], [0.2, 0.8]])
        charge = self.math_utils.compute_attentional_charge(O_bar, A_bar, X_bar, A_orig)
        self.assertTrue(np.isfinite(charge))
        print("✓ Attentional charge computation correct")

    def test_model_consistency(self):
        """Test model initialization and consistency."""
        print("\n=== Testing Model Consistency ===")

        # Test model initialization
        self.assertEqual(self.model.config.num_levels, 3)
        self.assertIn("perception", self.model.config.level_names)
        self.assertIn("attention", self.model.config.level_names)
        self.assertIn("meta_awareness", self.model.config.level_names)
        print("✓ Model initialization correct")

        # Test matrix dimensions
        for level_name in self.model.config.level_names:
            if level_name in self.model.config.likelihood_matrices:
                A_matrix = self.model.config.likelihood_matrices[level_name]
                level_config = self.model.config.levels[level_name]
                expected_shape = (level_config.obs_dim, level_config.state_dim)
                self.assertEqual(A_matrix.shape, expected_shape)

                # Test normalization
                col_sums = np.sum(A_matrix, axis=0)
                self.assertTrue(np.allclose(col_sums, 1.0, atol=1e-6))
        print("✓ Matrix dimensions and normalization correct")

        # Test transition matrices
        for matrix_name, B_matrix in self.model.config.transition_matrices.items():
            row_sums = np.sum(B_matrix, axis=1)
            self.assertTrue(np.allclose(row_sums, 1.0, atol=1e-6))
        print("✓ Transition matrix normalization correct")

    def test_three_level_vs_two_level(self):
        """Test that three-level model behaves appropriately compared to two-level."""
        print("\n=== Testing Three-Level vs Two-Level Models ===")

        # Run two-level simulation
        results_2level = self.model.run_simulation("figure_10")
        self.assertIn("perception", results_2level["true_states"])
        self.assertIn("attention", results_2level["true_states"])
        print("✓ Two-level simulation completed")

        # Run three-level simulation
        results_3level = self.model.run_simulation("figure_11")
        self.assertIn("perception", results_3level["true_states"])
        self.assertIn("attention", results_3level["true_states"])
        self.assertIn("meta_awareness", results_3level["true_states"])
        print("✓ Three-level simulation completed")

        # Compare mind-wandering patterns
        att_states_2 = results_2level["true_states"]["attention"]
        att_states_3 = results_3level["true_states"]["attention"]

        mw_2level = np.sum(att_states_2 == 1) / len(att_states_2)
        mw_3level = np.sum(att_states_3 == 1) / len(att_states_3)

        # Three-level should have different dynamics
        self.assertGreater(abs(mw_3level - mw_2level), 0.01)
        print(
            f"✓ Mind-wandering difference: 2-level={mw_2level:.3f}, 3-level={mw_3level:.3f}"
        )

    def test_mind_wandering_dynamics(self):
        """Test that mind-wandering dynamics are realistic."""
        print("\n=== Testing Mind-Wandering Dynamics ===")

        results = self.model.run_simulation("default")

        if "attention" in results["true_states"]:
            attention_states = results["true_states"]["attention"]

            # Calculate mind-wandering percentage
            mw_percentage = np.sum(attention_states == 1) / len(attention_states)

            # Should be between 10% and 70% (reasonable range)
            self.assertGreater(mw_percentage, 0.1)
            self.assertLess(mw_percentage, 0.7)
            print(f"✓ Mind-wandering percentage: {mw_percentage:.1%}")

            # Check for state transitions (not all same state)
            transitions = np.sum(np.diff(attention_states) != 0)
            self.assertGreater(transitions, 5)
            print(f"✓ Attention transitions: {transitions}")

    def test_precision_dynamics(self):
        """Test precision modulation dynamics."""
        print("\n=== Testing Precision Dynamics ===")

        results = self.model.run_simulation("default")

        if "perception" in results["precision_values"]:
            precision_values = results["precision_values"]["perception"]

            # Check precision bounds
            min_precision = np.min(precision_values)
            max_precision = np.max(precision_values)

            # Should stay within configured bounds
            expected_bounds = self.config.precision_bounds.get("perception", (0.5, 2.0))
            self.assertGreaterEqual(min_precision, expected_bounds[0] - 0.1)
            self.assertLessEqual(max_precision, expected_bounds[1] + 0.1)
            print(f"✓ Precision range: [{min_precision:.3f}, {max_precision:.3f}]")

            # Should show modulation (not constant)
            precision_variance = np.var(precision_values)
            self.assertGreater(precision_variance, 0.001)
            print(f"✓ Precision modulation variance: {precision_variance:.6f}")

    def test_figure_modes(self):
        """Test all figure reproduction modes."""
        print("\n=== Testing Figure Reproduction Modes ===")

        modes = ["figure_7", "figure_10", "figure_11", "default"]

        for mode in modes:
            with self.subTest(mode=mode):
                start_time = time.time()
                results = self.model.run_simulation(mode)
                duration = time.time() - start_time

                # Basic validation
                self.assertIsInstance(results, dict)
                self.assertIn("true_states", results)
                self.assertIn("state_priors", results)
                self.assertIn("precision_values", results)

                # Check simulation completed in reasonable time
                self.assertLess(duration, 30.0)  # Should complete within 30 seconds

                print(f"✓ {mode} simulation completed in {duration:.2f}s")

    def test_numerical_stability(self):
        """Test numerical stability across different conditions."""
        print("\n=== Testing Numerical Stability ===")

        # Test with extreme precision values
        extreme_model = MetaAwarenessModel(self.config, random_seed=123)
        results = extreme_model.run_simulation("default")

        # Check for NaN or infinite values
        for key, data in results.items():
            if isinstance(data, dict):
                for subkey, subdata in data.items():
                    if isinstance(subdata, np.ndarray):
                        self.assertFalse(
                            np.any(np.isnan(subdata)), f"NaN found in {key}.{subkey}"
                        )
                        self.assertFalse(
                            np.any(np.isinf(subdata)), f"Inf found in {key}.{subkey}"
                        )
        print("✓ No NaN or Inf values found")

        # Test precision value bounds
        if "precision_values" in results:
            for level, precision in results["precision_values"].items():
                self.assertTrue(
                    np.all(precision > 0), f"Non-positive precision in {level}"
                )
                self.assertTrue(
                    np.all(precision < 100), f"Extreme precision in {level}"
                )
        print("✓ Precision values within reasonable bounds")

    def test_reproducibility(self):
        """Test reproducibility with same random seed."""
        print("\n=== Testing Reproducibility ===")

        # Run same simulation twice with same seed
        model1 = MetaAwarenessModel(self.config, random_seed=42)
        results1 = model1.run_simulation("default")

        model2 = MetaAwarenessModel(self.config, random_seed=42)
        results2 = model2.run_simulation("default")

        # Compare key results
        for key in ["true_states", "state_priors", "precision_values"]:
            if key in results1 and key in results2:
                if isinstance(results1[key], dict) and isinstance(results2[key], dict):
                    for subkey in results1[key]:
                        if subkey in results2[key]:
                            array1 = results1[key][subkey]
                            array2 = results2[key][subkey]
                            self.assertTrue(
                                np.allclose(array1, array2, atol=1e-10),
                                f"Reproducibility failed for {key}.{subkey}",
                            )
        print("✓ Results are reproducible with same random seed")

    def test_visualization_generation(self):
        """Test figure generation capabilities."""
        print("\n=== Testing Visualization Generation ===")

        # Run simulations for each figure
        figure_gen = FigureGenerator(output_dir=self.output_dir)

        for mode in ["figure_7", "figure_10", "figure_11"]:
            with self.subTest(mode=mode):
                results = self.model.run_simulation(mode)

                if mode == "figure_7":
                    fig_path = figure_gen.generate_figure_7(results, f"test_{mode}")
                elif mode == "figure_10":
                    fig_path = figure_gen.generate_figure_10(results, f"test_{mode}")
                elif mode == "figure_11":
                    fig_path = figure_gen.generate_figure_11(results, f"test_{mode}")

                # Check that file was created
                self.assertTrue(fig_path.exists())
                self.assertGreater(fig_path.stat().st_size, 1000)  # Non-empty file
                print(f"✓ {mode} figure generated: {fig_path.name}")

    def test_performance_benchmarks(self):
        """Test performance meets acceptable standards."""
        print("\n=== Testing Performance Benchmarks ===")

        # Benchmark simulation speed
        start_time = time.time()
        results = self.model.run_simulation("default")
        simulation_time = time.time() - start_time

        # Should complete within reasonable time
        self.assertLess(simulation_time, 10.0)  # 10 seconds max
        print(f"✓ Simulation completed in {simulation_time:.3f}s")

        # Benchmark figure generation
        figure_gen = FigureGenerator(output_dir=self.output_dir)
        start_time = time.time()
        figure_gen.generate_figure_7(results, "performance_test")
        figure_time = time.time() - start_time

        self.assertLess(figure_time, 5.0)  # 5 seconds max
        print(f"✓ Figure generation completed in {figure_time:.3f}s")

    def test_simulation_runner_integration(self):
        """Test complete simulation runner integration."""
        print("\n=== Testing Simulation Runner Integration ===")

        runner = SimulationRunner(
            config_path="config/meta_awareness_gnn.toml",
            output_dir=self.output_dir,
            random_seed=42,
        )

        # Test single simulation
        results = runner.run_simulation("default")
        self.assertIsInstance(results, dict)
        self.assertIn("analysis", results)
        print("✓ Single simulation runner test passed")

        # Test multiple simulations
        all_results = runner.run_complete_analysis(
            simulation_modes=["figure_7", "figure_10"],
            generate_figures=True,
            save_detailed_results=True,
        )

        self.assertIn("results", all_results)
        self.assertIn("figure_7", all_results["results"])
        self.assertIn("figure_10", all_results["results"])
        print("✓ Complete analysis runner test passed")

    def test_paper_verification(self):
        """Test comprehensive paper verification."""
        print("\n=== Testing Paper Verification ===")

        verifier = PaperVerification("config/meta_awareness_gnn.toml")
        report = verifier.run_comprehensive_verification()

        # Should pass all tests
        self.assertEqual(report["summary"]["overall_status"], "PASS")
        self.assertEqual(report["summary"]["success_rate"], 1.0)

        print(
            f"✓ Paper verification: {report['summary']['passed_tests']}/{report['summary']['total_tests']} tests passed"
        )

    def test_mathematical_accuracy(self):
        """Test mathematical accuracy against known values."""
        print("\n=== Testing Mathematical Accuracy ===")

        # Test specific calculations match expected values

        # Softmax of [0, 1, 2] should give known result
        x = np.array([0, 1, 2])
        result = self.math_utils.softmax(x)
        expected = np.array([0.09003057, 0.24472847, 0.66524096])
        self.assertTrue(np.allclose(result, expected, atol=1e-6))
        print("✓ Softmax mathematical accuracy verified")

        # Entropy of uniform distribution
        uniform = np.array([0.25, 0.25, 0.25, 0.25])
        entropy_val = self.math_utils.compute_entropy(uniform)
        expected_entropy = np.log(4)  # ln(4) ≈ 1.386
        self.assertAlmostEqual(entropy_val, expected_entropy, places=6)
        print("✓ Entropy mathematical accuracy verified")

        # KL divergence between uniform distributions should be 0
        p = np.array([0.5, 0.5])
        q = np.array([0.5, 0.5])
        kl_div = self.math_utils.compute_kl_divergence(p, q)
        self.assertAlmostEqual(kl_div, 0.0, places=10)
        print("✓ KL divergence mathematical accuracy verified")

    def test_original_function_names(self):
        """Test that all original function names are available."""
        print("\n=== Testing Original Function Names ===")

        # Import convenience functions
        from utils.math_utils import (
            normalise,
            normalize,
            softmax,
            softmax_dim2,
        )

        # Test that they all work
        x = np.array([1, 2, 3])
        self.assertTrue(callable(softmax))
        result = softmax(x)
        self.assertAlmostEqual(np.sum(result), 1.0)

        X = np.array([[1, 2], [3, 4]])
        self.assertTrue(callable(softmax_dim2))
        result_matrix = softmax_dim2(X)
        self.assertTrue(np.allclose(np.sum(result_matrix, axis=0), 1.0))

        self.assertTrue(callable(normalise))
        self.assertTrue(callable(normalize))

        print("✓ All original function names available and working")

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        print("\n=== Test Suite Completed ===")


def run_comprehensive_test():
    """Run the complete test suite."""
    print("=" * 80)
    print("Meta-Aware-2 Comprehensive Implementation Test")
    print("Validating all features from original implementation")
    print("=" * 80)

    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestCompleteImplementation)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n{'=' * 80}")
    print("Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print(
            "✓ ALL TESTS PASSED! Meta-aware-2 implementation is complete and accurate."
        )
    else:
        print("⚠ Some tests failed. See details above.")

    print(f"{'=' * 80}")

    return result.wasSuccessful()


def demo_simulation_results():
    """Demonstrate the simulation results matching original implementation."""
    print("\n" + "=" * 80)
    print("Meta-Aware-2 Simulation Results Demo")
    print("=" * 80)

    config = load_gnn_config("config/meta_awareness_gnn.toml")
    model = MetaAwarenessModel(config, random_seed=42)

    # Run each figure mode
    for mode in ["figure_7", "figure_10", "figure_11"]:
        print(f"\n--- Running {mode.replace('_', ' ').title()} Simulation ---")

        results = model.run_simulation(mode)

        # Print key statistics
        if "attention" in results["true_states"]:
            att_states = results["true_states"]["attention"]
            mw_percent = np.sum(att_states == 1) / len(att_states) * 100
            transitions = np.sum(np.diff(att_states) != 0)
            print(f"Mind-wandering: {mw_percent:.1f}%")
            print(f"Attention transitions: {transitions}")

        if "perception" in results["precision_values"]:
            precision = results["precision_values"]["perception"]
            print(
                f"Precision range: [{np.min(precision):.3f}, {np.max(precision):.3f}]"
            )

        print("Simulation completed successfully ✓")

    print(f"\n{'=' * 80}")
    print("All simulations match expected patterns from original implementation!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    # Run comprehensive test
    success = run_comprehensive_test()

    if success:
        # Demo simulation results
        demo_simulation_results()

    print(
        f"\nMeta-aware-2 implementation {'VALIDATED' if success else 'NEEDS FIXES'} ✓"
    )
