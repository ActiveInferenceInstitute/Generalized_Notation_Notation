#!/usr/bin/env python3
"""
Test PyMDP Visualization Module

Test script for the PyMDP visualization capabilities within the GNN pipeline.
This verifies that visualization works with pipeline-configured simulations.

Features:
- Test discrete state visualization
- Test belief distribution plots
- Test performance metrics
- Pipeline integration verification

Author: GNN PyMDP Integration
Date: 2024
"""

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gnn.analysis.pymdp.visualizer import PyMDPVisualizer, save_all_visualizations


@pytest.fixture
def test_data() -> Any:
    """Create test simulation data for visualization testing."""
    num_timesteps = 50
    num_states = 9  # 3x3 grid
    num_observations = 9  # Full observability for testing
    num_actions = 4  # Up, Down, Left, Right

    # Generate synthetic trajectory data
    rng = np.random.default_rng(42)  # Local generator: no global RNG mutation
    states = rng.integers(0, num_states, num_timesteps).tolist()
    observations = states.copy()  # Full observability for testing
    actions = rng.integers(0, num_actions, num_timesteps).tolist()
    rewards = rng.standard_normal(num_timesteps).tolist()

    # Generate synthetic belief distributions
    beliefs: list[Any] = []
    for _t in range(num_timesteps):
        belief = rng.dirichlet(np.ones(num_states))
        beliefs.append(belief)

    return {
        "states": states,
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "beliefs": beliefs,
        "num_states": num_states,
        "num_observations": num_observations,
        "num_actions": num_actions,
        "metrics": {
            "expected_free_energy": rng.standard_normal(num_timesteps).tolist(),
            "actions": actions,
            "belief_confidence": [float(max(b)) for b in beliefs],
            "cumulative_preference": rewards,
        },
    }


@pytest.fixture
def visualizer(temp_output_dir: Any) -> Any:
    """Create a PyMDPVisualizer for testing."""
    return PyMDPVisualizer(output_dir=temp_output_dir, show_plots=False)


class TestPyMDPVisualizationModule:
    """Test suite for PyMDP visualization module."""

    def test_visualizer_creation(self, temp_output_dir: Any) -> Any:
        """Test PyMDPVisualizer can be created with output directory."""
        visualizer = PyMDPVisualizer(output_dir=temp_output_dir, show_plots=False)
        assert visualizer is not None
        assert visualizer.save_dir == temp_output_dir

    def test_discrete_state_visualization(
        self, visualizer: Any, test_data: Any, temp_output_dir: Any
    ) -> Any:
        """Test discrete state visualization from test data."""
        save_path = temp_output_dir / "discrete_states.png"

        fig = visualizer.plot_discrete_states(
            state_sequence=test_data["states"],
            num_states=test_data["num_states"],
            title="Test State Sequence",
            save_path=save_path,
        )

        assert fig is not None
        assert save_path.exists(), "Discrete state plot should be created"

        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_belief_evolution_visualization(
        self, visualizer: Any, test_data: Any, temp_output_dir: Any
    ) -> Any:
        """Test belief evolution visualization."""
        save_path = temp_output_dir / "belief_evolution.png"

        fig = visualizer.plot_belief_evolution(
            belief_traces=test_data["beliefs"],
            title="Test Belief Evolution",
            save_path=save_path,
        )

        assert fig is not None
        assert save_path.exists(), "Belief evolution plot should be created"

        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_performance_metrics_visualization(
        self, visualizer: Any, test_data: Any, temp_output_dir: Any
    ) -> Any:
        """Test performance metrics visualization."""
        save_path = temp_output_dir / "performance_metrics.png"

        fig = visualizer.plot_performance_metrics(
            metrics=test_data["metrics"], save_path=save_path
        )

        assert fig is not None
        assert save_path.exists(), "Performance metrics plot should be created"

        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_action_sequence_visualization(
        self, visualizer: Any, test_data: Any, temp_output_dir: Any
    ) -> Any:
        """Test action sequence visualization."""
        save_path = temp_output_dir / "action_sequence.png"

        fig = visualizer.plot_action_sequence(
            action_sequence=test_data["actions"],
            num_actions=test_data["num_actions"],
            title="Test Actions",
            save_path=save_path,
        )

        assert fig is not None
        assert save_path.exists(), "Action sequence plot should be created"

        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_observation_sequence_visualization(
        self, visualizer: Any, test_data: Any, temp_output_dir: Any
    ) -> Any:
        """Test observation sequence visualization."""
        save_path = temp_output_dir / "observation_sequence.png"

        fig = visualizer.plot_observation_sequence(
            observation_sequence=test_data["observations"],
            num_observations=test_data["num_observations"],
            title="Test Observations",
            save_path=save_path,
        )

        assert fig is not None
        assert save_path.exists(), "Observation sequence plot should be created"

        import matplotlib.pyplot as plt

        plt.close(fig)


def test_visualizer() -> Any:
    """Test the PyMDPVisualizer with synthetic data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)

        # Create synthetic test data
        rng = np.random.default_rng(42)
        num_timesteps = 20
        num_states = 9

        test_data: dict[str, Any] = {
            "states": rng.integers(0, num_states, num_timesteps).tolist(),
            "beliefs": [
                rng.dirichlet(np.ones(num_states)) for _ in range(num_timesteps)
            ],
            "metrics": {
                "expected_free_energy": rng.standard_normal(num_timesteps).tolist(),
                "actions": rng.integers(0, 4, num_timesteps).tolist(),
                "belief_confidence": rng.random(num_timesteps).tolist(),
                "cumulative_preference": rng.standard_normal(num_timesteps).tolist(),
            },
            "num_states": num_states,
        }

        # Initialize visualizer
        visualizer = PyMDPVisualizer(output_dir=output_dir, show_plots=False)

        # Test discrete states plot
        fig = visualizer.plot_discrete_states(
            state_sequence=test_data["states"],
            num_states=test_data["num_states"],
            title="Test States",
            save_path=output_dir / "states.png",
        )
        assert fig is not None

        # Test belief evolution plot
        fig = visualizer.plot_belief_evolution(
            belief_traces=test_data["beliefs"],
            title="Test Beliefs",
            save_path=output_dir / "beliefs.png",
        )
        assert fig is not None

        # Test performance metrics plot
        fig = visualizer.plot_performance_metrics(
            metrics=test_data["metrics"], save_path=output_dir / "performance.png"
        )
        assert fig is not None

        # Check files were created
        png_files = list(output_dir.glob("*.png"))
        assert len(png_files) >= 3, (
            f"Expected at least 3 PNG files, got {len(png_files)}"
        )

        visualizer.close_all_plots()


def test_pipeline_integration() -> Any:
    """Test integration with pipeline configuration using save_all_visualizations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)

        # Create test data in the format expected by save_all_visualizations
        rng = np.random.default_rng(42)
        num_timesteps = 20
        num_states = 9

        sim_results: dict[str, Any] = {
            "states": rng.integers(0, num_states, num_timesteps).tolist(),
            "beliefs": [
                rng.dirichlet(np.ones(num_states)) for _ in range(num_timesteps)
            ],
            "metrics": {
                "expected_free_energy": rng.standard_normal(num_timesteps).tolist(),
                "actions": rng.integers(0, 4, num_timesteps).tolist(),
                "belief_confidence": rng.random(num_timesteps).tolist(),
                "cumulative_preference": rng.standard_normal(num_timesteps).tolist(),
            },
            "num_states": num_states,
        }

        # Test save_all_visualizations function
        saved_files = save_all_visualizations(
            simulation_results=sim_results,
            output_dir=output_dir,
            config={"save_dir": output_dir},
        )

        assert len(saved_files) > 0, "save_all_visualizations should create files"

        # Verify files were actually created
        for _name, path in saved_files.items():
            assert Path(path).exists(), f"File should exist: {path}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
