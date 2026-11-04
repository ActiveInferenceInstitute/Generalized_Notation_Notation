#!/usr/bin/env python3
"""
Test script for numpy serialization utilities

This script tests the JSON serialization fixes for numpy types
that were causing the original PyMDP simulation to fail.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
from pymdp_utils import (
    clean_trace_for_serialization,
    convert_numpy_for_json,
    safe_json_dump,
    save_simulation_results,
)


def test_numpy_serialization():
    """Test numpy type serialization that was causing the original error"""

    print("Testing numpy serialization utilities...")

    # Create test data that mimics the problematic traces
    test_trace = {
        "episode": 1,
        "true_states": [np.int64(8), np.int64(3), np.int64(4)],
        "observations": [0, 2, 1],
        "actions": [np.float64(0.0), np.float64(2.0)],
        "rewards": [np.float64(-0.10), np.float64(10.00)],
        "beliefs": [np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.05, 0.15, 0.25, 0.55])],
        "positions": [(np.int64(1), np.int64(3)), (0, 4)],
        "policies": [None, None],
        "expected_free_energies": [None, None],
        "variational_free_energies": [np.float64(2.5), np.float64(1.8)],
    }

    print(f"Original trace sample: {test_trace}")

    # Test convert_numpy_for_json function
    print("\n1. Testing convert_numpy_for_json...")
    converted = convert_numpy_for_json(test_trace)
    print(f"Converted trace: {converted}")

    # Test that it's now JSON serializable
    try:
        json.dumps(converted)
        print("✓ Successfully converted to JSON string")
    except Exception as e:
        print(f"✗ JSON conversion failed: {e}")
        return False

    # Test clean_trace_for_serialization
    print("\n2. Testing clean_trace_for_serialization...")
    cleaned = clean_trace_for_serialization(test_trace)
    print(f"Cleaned trace keys: {list(cleaned.keys())}")

    try:
        json.dumps(cleaned)
        print("✓ Successfully cleaned and converted to JSON")
    except Exception as e:
        print(f"✗ Cleaned trace JSON conversion failed: {e}")
        return False

    # Test safe_json_dump
    print("\n3. Testing safe_json_dump...")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "test_trace.json"

        success = safe_json_dump(test_trace, test_file)
        if success and test_file.exists():
            print(f"✓ Successfully saved to {test_file}")
            print(f"File size: {test_file.stat().st_size} bytes")

            # Try to load it back
            with open(test_file, "r") as f:
                json.load(f)
            print("✓ Successfully loaded back from JSON")
        else:
            print("✗ Failed to save JSON file")
            return False

    # Test performance metrics
    print("\n4. Testing performance metrics serialization...")
    test_metrics = {
        "episode_rewards": [np.float64(9.9), np.float64(8.7), np.float64(10.0)],
        "episode_lengths": [np.int64(2), np.int64(5), np.int64(1)],
        "belief_entropies": [np.float64(2.893), np.float64(1.446), np.float64(0.734)],
        "success_rates": [np.float64(1.0), np.float64(1.0), np.float64(1.0)],
    }

    converted_metrics = convert_numpy_for_json(test_metrics)
    try:
        json.dumps(converted_metrics)
        print("✓ Performance metrics successfully serialized")
    except Exception as e:
        print(f"✗ Performance metrics serialization failed: {e}")
        return False

    # Test complete simulation results saving
    print("\n5. Testing complete simulation results saving...")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        test_traces = [test_trace, test_trace.copy()]
        test_config = {
            "grid_size": 5,
            "num_episodes": 2,
            "grid_layout": [[0, 1, 2], [1, 0, 1]],  # Simple test layout
            "random_seed": 42,
        }

        save_results = save_simulation_results(
            traces=test_traces,
            metrics=test_metrics,
            config=test_config,
            model_matrices=None,
            output_dir=temp_path,
        )

        print(f"Save results: {save_results}")

        # Check all files were created
        expected_files = [
            "simulation_config.json",
            "performance_metrics.json",
            "simulation_traces.pkl",
            "simulation_traces.json",
        ]

        all_success = True
        for filename in expected_files:
            filepath = temp_path / filename
            if filepath.exists():
                print(f"✓ {filename} created ({filepath.stat().st_size} bytes)")
            else:
                print(f"✗ {filename} missing")
                all_success = False

        if not all_success:
            return False

    print("\n🎉 All numpy serialization tests passed!")
    return True


if __name__ == "__main__":
    success = test_numpy_serialization()
    exit(0 if success else 1)
