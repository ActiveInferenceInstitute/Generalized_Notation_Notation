#!/usr/bin/env python3
"""
Test Visualization Ontology Tests

This file contains tests migrated from test_visualization.py.
"""

import sys
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gnn.visualization.ontology_visualizer import OntologyVisualizer


# Migrated from test_visualization.py
class TestOntologyVisualizer:
    """Test cases for the OntologyVisualizer class."""

    def test_extract_ontology_mappings(self, sample_gnn_file: Path) -> None:
        """Test ontology mapping extraction from GNN file."""
        # Read test file
        with open(sample_gnn_file) as f:
            content = f.read()

        # Extract ontology section
        import re

        ontology_match = re.search(
            r"## ActInfOntologyAnnotation\n(.*?)(?=\n##|\Z)", content, re.DOTALL
        )
        assert ontology_match is not None

        # Extract mappings
        visualizer = OntologyVisualizer()
        mappings = visualizer._extract_ontology_mappings(ontology_match.group(1))

        # Verify mappings
        assert len(mappings) == 13  # All variables and matrices

        # Verify matrix mappings
        assert ("A", "LikelihoodMatrix") in mappings
        assert ("B", "TransitionMatrix") in mappings
        assert ("C", "LogPreferenceVector") in mappings
        assert ("D", "PriorOverHiddenStates") in mappings
        assert ("E", "Habit") in mappings
        assert ("F", "VariationalFreeEnergy") in mappings
        assert ("G", "ExpectedFreeEnergy") in mappings

        # Verify variable mappings
        assert ("s", "HiddenState") in mappings
        assert ("s_prime", "NextHiddenState") in mappings
        assert ("o", "Observation") in mappings
        assert ("π", "PolicyVector") in mappings  # Distribution over actions
        assert ("u", "Action") in mappings  # Chosen action
        assert ("t", "Time") in mappings

    def test_create_ontology_table(self, temp_output_dir: Path) -> None:
        """Test creation of ontology visualization table."""
        visualizer = OntologyVisualizer()
        mappings: list[Any] = [
            ("s1", "state_location_1"),
            ("s2", "state_location_2"),
            ("a1", "action_move_forward"),
            ("o1", "observation_position"),
        ]

        # Create visualization
        output_path = visualizer._create_ontology_table(mappings, temp_output_dir)

        # Verify output
        assert output_path is not None
        assert Path(output_path).exists()
        assert Path(output_path).is_file()
        assert Path(output_path).suffix == ".png"

    def test_visualize_directory(
        self, test_data_dir: Path, temp_output_dir: Path
    ) -> None:
        """Test visualization of all ontology annotations in a directory."""
        visualizer = OntologyVisualizer()
        output_files = visualizer.visualize_directory(test_data_dir, temp_output_dir)

        # Verify outputs
        assert len(output_files) > 0
        for file_path in output_files:
            assert Path(file_path).exists()
            assert Path(file_path).is_file()
            assert Path(file_path).suffix == ".png"
