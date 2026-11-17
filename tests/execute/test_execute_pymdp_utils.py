#!/usr/bin/env python3
"""
Test script for PyMDP utilities

This script tests the PyMDP utilities for proper numpy serialization and
GNN integration functionality within the pipeline.
"""

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

try:
    from gnn.execute.pymdp.pymdp_utils import (
        check_gnn_pomdp_spec,
        clean_trace_for_serialization,
        convert_numpy_for_json,
        extract_gnn_dimensions,
        parse_gnn_matrix_string,
        parse_gnn_vector_string,
        safe_json_dump,
        save_simulation_results,
    )
except ImportError:
    from gnn.execute.pymdp.pymdp_utils import (
        check_gnn_pomdp_spec,
        clean_trace_for_serialization,
        convert_numpy_for_json,
        extract_gnn_dimensions,
        parse_gnn_matrix_string,
        parse_gnn_vector_string,
        safe_json_dump,
        save_simulation_results,
    )


@pytest.fixture
def test_trace() -> Any:
    """Create test data that mimics PyMDP simulation traces."""
    return {
        "episode": 1,
        "true_states": [np.int64(2), np.int64(1), np.int64(0)],
        "observations": [0, 1, 2],
        "actions": [np.float64(0.0), np.float64(1.0)],
        "rewards": [np.float64(-0.10), np.float64(1.00)],
        "beliefs": [np.array([0.1, 0.6, 0.3]), np.array([0.05, 0.25, 0.7])],
    }


@pytest.fixture
def test_gnn_spec() -> Any:
    """Create test GNN specification."""
    return {
        "model_parameters": {"num_hidden_states": 3, "num_obs": 3, "num_actions": 3},
        "variables": [
            {"name": "A", "dimensions": [3, 3]},
            {"name": "B", "dimensions": [3, 3, 3]},
            {"name": "C", "dimensions": [3]},
        ],
    }


class TestNumpySerialization:
    """Test numpy type serialization functionality."""

    def test_convert_numpy_for_json(self, test_trace: Any) -> Any:
        """Test convert_numpy_for_json function."""
        converted = convert_numpy_for_json(test_trace)

        # Should be JSON serializable
        json_str = json.dumps(converted)
        assert json_str is not None
        assert len(json_str) > 0

    def test_clean_trace_for_serialization(self, test_trace: Any) -> Any:
        """Test clean_trace_for_serialization function."""
        cleaned = clean_trace_for_serialization(test_trace)

        # Should be JSON serializable
        json_str = json.dumps(cleaned)
        assert json_str is not None
        assert "episode" in cleaned

    def test_safe_json_dump(self, test_trace: Any) -> Any:
        """Test safe_json_dump function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_file = temp_path / "test_trace.json"

            success = safe_json_dump(test_trace, test_file)
            assert success, "safe_json_dump should return True on success"
            assert test_file.exists(), "JSON file should be created"

            # Should be loadable
            with open(test_file) as f:
                loaded_data = json.load(f)
            assert loaded_data is not None


class TestGNNParsing:
    """Test GNN parsing utilities."""

    def test_parse_gnn_matrix_string(self) -> Any:
        """Test GNN matrix string parsing."""
        matrix_str = "{(0.9, 0.05, 0.05), (0.05, 0.9, 0.05), (0.05, 0.05, 0.9)}"

        matrix = parse_gnn_matrix_string(matrix_str)

        assert matrix is not None
        assert matrix.shape == (3, 3), f"Expected shape (3,3), got {matrix.shape}"

    def test_parse_gnn_vector_string(self) -> Any:
        """Test GNN vector string parsing."""
        vector_str = "{(0.33333, 0.33333, 0.33333)}"

        vector = parse_gnn_vector_string(vector_str)

        assert vector is not None
        assert len(vector) == 3, f"Expected length 3, got {len(vector)}"

    def test_extract_gnn_dimensions(self, test_gnn_spec: Any) -> Any:
        """Test GNN dimension extraction."""
        dimensions = extract_gnn_dimensions(test_gnn_spec)

        assert dimensions["num_states"] == 3, "States dimension mismatch"
        assert dimensions["num_observations"] == 3, "Observations dimension mismatch"
        assert dimensions["num_actions"] == 3, "Actions dimension mismatch"

    def test_check_gnn_pomdp_spec(self, test_gnn_spec: Any) -> Any:
        """Test POMDP structure validation."""
        validation = check_gnn_pomdp_spec(test_gnn_spec)

        assert validation["valid"], "POMDP structure should be valid"


class TestIntegration:
    """Test full integration with simulated data."""

    def test_save_simulation_results(self) -> Any:
        """Test saving simulation results."""
        test_traces: list[Any] = [
            {
                "episode": 0,
                "true_states": [0, 1, 2],
                "observations": [0, 1, 2],
                "actions": [0, 1],
                "rewards": [-0.1, 1.0],
                "beliefs": [np.array([0.8, 0.2, 0.0]), np.array([0.1, 0.1, 0.8])],
            }
        ]

        test_metrics: dict[str, Any] = {
            "episode_rewards": [0.9],
            "episode_lengths": [3],
            "belief_entropies": [1.5],
            "success_rates": [1.0],
        }

        test_config: dict[str, Any] = {
            "num_states": 3,
            "num_observations": 3,
            "num_actions": 3,
            "model_name": "TestModel",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            save_results = save_simulation_results(
                traces=test_traces,
                metrics=test_metrics,
                config=test_config,
                model_matrices=None,
                output_dir=temp_path,
            )

            assert save_results is not None

            # Check files were created
            expected_files: list[Any] = [
                "simulation_config.json",
                "performance_metrics.json",
                "simulation_traces.pkl",
                "simulation_traces.json",
            ]

            for filename in expected_files:
                temp_path / filename


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
