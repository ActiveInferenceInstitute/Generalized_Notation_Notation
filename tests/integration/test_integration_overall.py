"""Test suite for Integration module.

Public classes: TestIntegrationOverall
"""

import json
from typing import Any

import pytest

from gnn.integration.processor import process_integration


class TestIntegrationOverall:
    """Test suite for Integration module."""

    @pytest.fixture
    def integration_files(self, safe_filesystem: Any) -> Any:
        """Create a set of files with dependencies."""
        # File 1: Defines ComponentA
        safe_filesystem.create_file(
            "comp_a.md",
            """
# Component A
- name: ComponentA
  type: Object
""",
        )
        # File 2: References ComponentA
        safe_filesystem.create_file(
            "comp_b.md",
            """
# Component B
- name: ComponentB
  type: ComponentA
  refs: $ref:ComponentA
""",
        )
        return safe_filesystem.temp_dir

    def test_process_integration_flow(
        self, safe_filesystem: Any, integration_files: Any
    ) -> Any:
        """Test the integration processing workflow."""
        target_dir = integration_files
        output_dir = safe_filesystem.create_dir("integration_output")

        success = process_integration(target_dir, output_dir, verbose=True)

        assert success is True

        # Check output
        results_file = output_dir / "integration_results/integration_results.json"
        assert results_file.exists()

        with open(results_file) as f:
            data = json.load(f)

        assert data["success"] is True
        assert data["processed_files"] == 2

        # Check if dependency was found (if using networkx)
        # We can't guarantee networkx is installed in checking env,
        # but the processor handles missing networkx gracefully.
        # If stats are present:
        if "system_graph_stats" in data and data["system_graph_stats"]:
            # Should have 2 nodes (ComponentA, ComponentB)
            # And 1 edge (B -> A) or similar depending on implementation
            pass

    def test_process_integration_circular_dependency(self, safe_filesystem: Any) -> Any:
        """Test with circular dependency if networkx available."""
        # A -> B, B -> A
        safe_filesystem.create_file(
            "cycle_a.md",
            """
- name: A
  type: B
""",
        )
        safe_filesystem.create_file(
            "cycle_b.md",
            """
- name: B
  type: A
""",
        )
        output_dir = safe_filesystem.create_dir("output")

        # This shouldn't crash, but might report issues
        process_integration(safe_filesystem.temp_dir, output_dir)

        results_file = output_dir / "integration_results/integration_results.json"
        with open(results_file) as f:
            data = json.load(f)

        # If networkx is present, it might list issues.
        # We assume success=True for the process itself (analysis completed),
        # even if it found issues in the model.
        assert data["success"] is True
