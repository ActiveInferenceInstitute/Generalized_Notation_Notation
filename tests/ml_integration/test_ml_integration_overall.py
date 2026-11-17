"""Test suite for ML Integration module.

Public classes: TestMLIntegrationOverall
"""

import json
from typing import Any

from gnn.ml_integration.processor import process_ml_integration


class TestMLIntegrationOverall:
    """Test suite for ML Integration module."""

    def test_process_ml_integration_flow(
        self, safe_filesystem: Any, sample_gnn_file: Any
    ) -> Any:
        """Test the ML integration workflow."""
        target_dir = sample_gnn_file.parent
        output_dir = safe_filesystem.create_dir("ml_output")

        success = process_ml_integration(target_dir, output_dir, verbose=True)

        assert success is True

        # Check output
        results_file = output_dir / "ml_integration_results.json"
        assert results_file.exists()

        with open(results_file) as f:
            data = json.load(f)

        assert data["status"] == "completed"
        # We expect at least one model processed
        assert len(data.get("models_trained", [])) == 1

        # Check framework status
        assert "framework_status" in data
        # Sklearn might be available or missing, code handles both.
        # If available, type should be 'decision_tree_classifier'
        # If missing, 'structural_analysis'

        model_info = data["models_trained"][0]
        assert model_info["source"] == sample_gnn_file.name
        assert model_info["type"] == "structural_analysis"
        assert model_info["validation_status"] == "not_applicable"
        assert "accuracy" not in model_info

    def test_process_ml_integration_no_files(self, safe_filesystem: Any) -> Any:
        """Test with empty input."""
        empty_dir = safe_filesystem.create_dir("empty_input")
        output_dir = safe_filesystem.create_dir("empty_output")

        success = process_ml_integration(empty_dir, output_dir)

        # The code iterates over empty list and returns True
        assert success is True

        results_file = output_dir / "ml_integration_results.json"
        assert results_file.exists()
        with open(results_file) as f:
            data = json.load(f)
        assert len(data["models_trained"]) == 0
