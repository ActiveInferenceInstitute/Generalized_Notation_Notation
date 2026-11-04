#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meta-Aware-2: Streamlined GNN-Configurable Meta-Awareness Simulation

Main executable for the complete computational phenomenology implementation
based on Sandved-Smith et al. (2021). This is the streamlined, production-ready
version that exactly replicates the original implementation functionality.

Part of the GeneralizedNotationNotation (GNN) project.
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

# Add current directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

from config.gnn_parser import load_gnn_config
from execution.simulation_runner import run_simulation_from_config
from simulation_logging.simulation_logger import create_logger


def main():
    """Main entry point for meta-awareness simulation."""

    parser = argparse.ArgumentParser(
        description="Meta-Aware-2: GNN-Configurable Meta-Awareness Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration
  python run_meta_awareness.py config/meta_awareness_gnn.toml
  
  # Run specific simulation modes (reproduce figures)
  python run_meta_awareness.py config/meta_awareness_gnn.toml -m figure_7 figure_10 figure_11
  
  # Run with custom settings
  python run_meta_awareness.py config/meta_awareness_gnn.toml -s 42 -l DEBUG
  
  # Show configuration and exit
  python run_meta_awareness.py config/meta_awareness_gnn.toml --show-config
  
  # Test logging system
  python run_meta_awareness.py --test-logging
        """,
    )

    # Required arguments (optional for some commands)
    parser.add_argument(
        "config",
        nargs="?",
        default="config/meta_awareness_gnn.toml",
        help="Path to GNN configuration file (TOML format, default: config/meta_awareness_gnn.toml)",
    )

    # Optional arguments
    parser.add_argument(
        "--modes",
        "-m",
        nargs="+",
        help="Simulation modes to run (e.g., figure_7, figure_10, figure_11, default)",
    )

    parser.add_argument(
        "--seed", "-s", type=int, help="Random seed for reproducibility"
    )

    parser.add_argument(
        "--log-level",
        "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    parser.add_argument(
        "--show-config", action="store_true", help="Show configuration and exit"
    )

    parser.add_argument("--test", action="store_true", help="Run test suite and exit")

    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version="Meta-Aware-2 v1.0.0 (GNN-Configurable Meta-Awareness Simulation)",
    )

    parser.add_argument(
        "--test-logging", action="store_true", help="Test the logging system and exit"
    )

    args = parser.parse_args()

    # Handle special modes
    if args.test:
        run_tests()
        return

    if args.show_config:
        show_configuration(args.config)
        return

    if args.test_logging:
        test_logging_system()
        return

    # Run main simulation
    try:
        print("=" * 80)
        print("Meta-Aware-2: Computational Phenomenology of Meta-Awareness")
        print("Based on Sandved-Smith et al. (2021)")
        print("=" * 80)
        print()

        print(f"Configuration: {args.config}")
        print(f"Random seed: {args.seed}")
        print(f"Log level: {args.log_level}")

        if args.modes:
            print(f"Simulation modes: {', '.join(args.modes)}")
        else:
            print("Running all figure reproduction modes")

        print()
        print("Starting simulation...")
        print("-" * 40)

        # Run complete analysis
        results = run_simulation_from_config(
            config_path=args.config,
            output_dir=".",  # Use current directory structure
            simulation_modes=args.modes,
            random_seed=args.seed,
            log_level=args.log_level,
        )

        print("-" * 40)
        print("Simulation completed successfully!")
        print()

        # Print summary
        print_results_summary(results)

    except FileNotFoundError as e:
        print(f"Error: Configuration file not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during simulation: {e}")
        if args.log_level == "DEBUG":
            traceback.print_exc()
        sys.exit(1)


def show_configuration(config_path: str):
    """Show configuration details and exit."""
    try:
        config = load_gnn_config(config_path)

        print("=" * 60)
        print("GNN Configuration Summary")
        print("=" * 60)

        print(f"Model Name: {config.name}")
        print(f"Description: {config.description}")
        print(f"Number of Levels: {config.num_levels}")
        print(f"Level Names: {', '.join(config.level_names)}")
        print(f"Time Steps: {config.time_steps}")
        print()

        print("Level Configuration:")
        for name, level in config.levels.items():
            print(f"  {name}:")
            print(f"    State Dimension: {level.state_dim}")
            print(f"    Observation Dimension: {level.obs_dim}")
            print(f"    Action Dimension: {level.action_dim}")
        print()

        print("Precision Bounds:")
        for level_name, bounds in config.precision_bounds.items():
            print(f"  {level_name}: [{bounds[0]}, {bounds[1]}]")
        print()

        print("Available Simulation Modes:")
        for mode_name, mode_type in config.simulation_modes.items():
            print(f"  {mode_name}: {mode_type}")
        print()

        print("Configuration loaded successfully!")

    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)


def run_tests():
    """Run the test suite."""
    try:
        import unittest

        from tests.test_simulation import TestSimulation, TestValidationAgainstPaper

        print("=" * 60)
        print("Running Meta-Aware-2 Test Suite")
        print("=" * 60)

        # Create test suite
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()

        # Add test classes
        suite.addTests(loader.loadTestsFromTestCase(TestSimulation))
        suite.addTests(loader.loadTestsFromTestCase(TestValidationAgainstPaper))

        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        # Print summary
        print("\n" + "=" * 60)
        if result.wasSuccessful():
            print("All tests passed successfully!")
            sys.exit(0)
        else:
            print(
                f"Tests failed: {len(result.failures)} failures, {len(result.errors)} errors"
            )
            sys.exit(1)

    except ImportError as e:
        print(f"Error importing test modules: {e}")
        print("Make sure all dependencies are installed.")
        sys.exit(1)


def print_results_summary(results: dict):
    """Print a summary of simulation results."""

    print("Results Summary:")

    # Simulation modes run
    sim_results = results.get("simulation_results", {})
    if sim_results:
        print(f"  Simulation modes completed: {len(sim_results)}")
        for mode in sim_results.keys():
            print(f"    - {mode}")

    # Figures generated
    figure_paths = results.get("figure_paths", {})
    if figure_paths:
        total_figures = sum(
            len(paths) if isinstance(paths, dict) else 1
            for paths in figure_paths.values()
        )
        print(f"  Figures generated: {total_figures}")
        print("    Location: ./figures/")

    # Results saved
    print("  Results saved to: ./results/")
    print("  Logs saved to: ./logs/")

    print()
    print("✓ Meta-awareness computational phenomenology simulation complete!")
    print("  Check figures/, results/, and logs/ directories for outputs.")


def test_logging_system():
    """Test the logging system to ensure all components work correctly."""
    print("=" * 60)
    print("Testing Logging System")
    print("=" * 60)

    # Create test logger
    test_dir = Path("test_logs")
    test_dir.mkdir(exist_ok=True)

    logger = create_logger(test_dir, level="DEBUG")

    print(f"Created logger with ID: {logger.simulation_id}")
    print(f"Log directory: {test_dir}")

    # Test basic logging
    print("\n1. Testing basic logging levels...")
    logger.info("This is an info message")
    logger.debug("This is a debug message")
    logger.warning("This is a warning message")

    # Test performance logging
    print("\n2. Testing performance logging...")
    logger.log_memory_usage(1, 100.5)
    logger.log_matrix_operation(
        operation="test_operation",
        matrix_name="test_matrix",
        input_shape=(10, 10),
        output_shape=(10, 5),
        computation_time=0.001,
    )

    # Test custom metrics
    print("\n3. Testing custom metrics...")
    logger.log_custom_metric("test_metric", 42.0)
    logger.log_custom_metric("another_metric", "test_value")

    # Test error logging (intentional)
    print("\n4. Testing error logging...")
    try:
        raise ValueError("This is a test error for logging verification")
    except Exception as e:
        logger.error("Test error caught", exception=e)

    # Test step logging
    print("\n5. Testing step logging...")
    for step in range(5):
        logger.log_step_start(step, 5)
        time.sleep(0.01)  # Simulate work
        logger.log_step_end(step, {"test_data": step * 2})

    # Test simulation start/end
    print("\n6. Testing simulation lifecycle...")
    test_config = {
        "model_name": "test_model",
        "num_levels": 3,
        "time_steps": 100,
        "simulation_mode": "test",
    }

    logger.set_model_info("test_model", 3, 100, "test")
    logger.log_simulation_start(test_config)

    test_results = {
        "state_posteriors": {"level1": [1, 2, 3]},
        "precision_values": {"level1": [0.5, 0.6, 0.7]},
        "num_levels": 3,
    }
    logger.log_simulation_end(test_results)

    # Check log files
    print("\n7. Checking generated log files...")
    log_files = logger.get_log_files()

    for log_type, log_path in log_files.items():
        if log_path.exists():
            size = log_path.stat().st_size
            print(f"  ✅ {log_type}: {log_path} ({size} bytes)")

            # Check if file has content
            if size == 0:
                print(f"      ⚠️  Warning: {log_type} is empty")
            else:
                print(f"      📝 {log_type} has content")
        else:
            print(f"  ❌ {log_type}: {log_path} (missing)")

    print("\n8. Testing memory and performance tracking...")
    import os

    import psutil

    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Current memory usage: {memory_mb:.2f} MB")

    # Test numerical issue logging
    print("\n9. Testing numerical issue logging...")
    logger.log_numerical_issue(
        "test_underflow",
        {"location": "test_function", "value": 1e-20, "expected_range": [1e-10, 1e10]},
        step=42,
    )

    print("\n" + "=" * 60)
    print("Logging system test completed!")
    print("Check the generated log files in the test_logs directory.")
    print("All logs should have content if the system is working correctly.")
    print("=" * 60)


if __name__ == "__main__":
    main()
