#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNN Configuration Parser for Meta-Awareness Active Inference Model

This module provides comprehensive parsing of GNN configuration files,
creating all model variables and parameters in a generic, dimensionally-flexible way.

Part of the meta-aware-2 "golden spike" GNN-specified executable implementation.
"""

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore
    except ImportError:
        try:
            import toml as tomllib  # type: ignore
        except ImportError:

            class _TomlFallback:
                @staticmethod
                def load(f):
                    import json

                    return json.load(f)

            tomllib = _TomlFallback()  # type: ignore
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LevelConfig:
    """Configuration for a single hierarchical level."""

    level_id: int
    state_dim: int
    obs_dim: int
    action_dim: int
    state_labels: List[str]
    obs_labels: List[str]
    action_labels: Optional[List[str]] = None


@dataclass
class ModelConfig:
    """Complete model configuration parsed from GNN file."""

    # Model metadata
    name: str
    description: str
    version: str
    paper_reference: str

    # Temporal parameters
    time_steps: int
    discrete_time: bool
    model_time_horizon: int

    # Hierarchical structure
    num_levels: int
    level_names: List[str]
    levels: Dict[str, LevelConfig] = field(default_factory=dict)

    # Matrices (generic dimensions)
    transition_matrices: Dict[str, np.ndarray] = field(default_factory=dict)
    likelihood_matrices: Dict[str, np.ndarray] = field(default_factory=dict)
    prior_beliefs: Dict[str, np.ndarray] = field(default_factory=dict)

    # Policy configuration
    policy_priors: Dict[str, np.ndarray] = field(default_factory=dict)
    policy_preferences: Dict[str, np.ndarray] = field(default_factory=dict)
    policy_precision: Dict[str, float] = field(default_factory=dict)

    # Precision parameters
    precision_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # Stimulus configuration
    oddball_pattern: str = "default"
    oddball_times: List[int] = field(default_factory=list)

    # Simulation modes
    simulation_modes: Dict[str, str] = field(default_factory=dict)
    simulation_schedules: Dict[str, List[int]] = field(default_factory=dict)

    # Configuration for outputs
    logging_config: Dict[str, Any] = field(default_factory=dict)
    visualization_config: Dict[str, Any] = field(default_factory=dict)
    output_config: Dict[str, Any] = field(default_factory=dict)

    # Numerical and validation settings
    numerical_config: Dict[str, Any] = field(default_factory=dict)
    validation_config: Dict[str, Any] = field(default_factory=dict)
    extensions_config: Dict[str, Any] = field(default_factory=dict)


class GNNConfigParser:
    """
    Parser for GNN configuration files.

    Reads TOML configuration files and creates ModelConfig objects
    with all necessary parameters for the meta-awareness model.
    """

    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize parser with configuration file path.

        Args:
            config_path: Path to GNN configuration file (.toml)
        """
        self.config_path = Path(config_path)
        self.raw_config = None
        self.model_config = None

        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

    def load_config(self) -> Dict[str, Any]:
        """Load raw configuration from TOML file."""
        try:
            with open(self.config_path, "rb") as f:
                self.raw_config = tomllib.load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return self.raw_config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    def parse_config(self) -> ModelConfig:
        """
        Parse raw configuration into structured ModelConfig object.

        Returns:
            ModelConfig object with all parsed parameters
        """
        if self.raw_config is None:
            self.load_config()

        logger.info("Parsing GNN configuration...")

        # Parse model metadata
        model_info = self.raw_config.get("model", {})
        temporal_info = self.raw_config.get("temporal", {})
        hierarchical_info = self.raw_config.get("hierarchical_structure", {})

        # Initialize ModelConfig
        self.model_config = ModelConfig(
            name=model_info.get("name", "meta_awareness_model"),
            description=model_info.get("description", ""),
            version=model_info.get("version", "1.0"),
            paper_reference=model_info.get("paper_reference", ""),
            time_steps=model_info.get(
                "time_steps", temporal_info.get("time_steps", 100)
            ),
            discrete_time=temporal_info.get("discrete_time", True),
            model_time_horizon=temporal_info.get("model_time_horizon", 100),
            num_levels=model_info.get(
                "num_levels", hierarchical_info.get("num_levels", 3)
            ),
            level_names=model_info.get(
                "level_names", hierarchical_info.get("level_names", [])
            ),
        )

        # Parse hierarchical levels
        self._parse_levels()

        # Parse matrices
        self._parse_transition_matrices()
        self._parse_likelihood_matrices()
        self._parse_prior_beliefs()

        # Parse policy configuration
        self._parse_policies()

        # Parse precision parameters
        self._parse_precision()

        # Parse stimulus configuration
        self._parse_stimuli()

        # Parse simulation modes
        self._parse_simulation_modes()

        # Parse output configurations
        self._parse_output_configs()

        # Validate configuration
        self._validate_config()

        logger.info("Configuration parsing completed successfully")
        return self.model_config

    def _parse_levels(self):
        """Parse hierarchical level configurations."""
        levels_config = self.raw_config.get("levels", {})

        for level_name, level_data in levels_config.items():
            level_config = LevelConfig(
                level_id=level_data.get("level_id", 1),
                state_dim=level_data.get("state_dim", 2),
                obs_dim=level_data.get("obs_dim", 2),
                action_dim=level_data.get("action_dim", 0),
                state_labels=level_data.get("state_labels", []),
                obs_labels=level_data.get("obs_labels", []),
                action_labels=level_data.get("action_labels", None),
            )
            self.model_config.levels[level_name] = level_config
            logger.debug(f"Parsed level '{level_name}': {level_config}")

    def _parse_transition_matrices(self):
        """Parse transition matrices with generic dimensions."""
        transition_config = self.raw_config.get("transition_matrices", {})

        for matrix_name, matrix_data in transition_config.items():
            if "matrix" in matrix_data:
                matrix = np.array(matrix_data["matrix"], dtype=float)
                self.model_config.transition_matrices[matrix_name] = matrix
                logger.debug(
                    f"Parsed transition matrix '{matrix_name}': shape {matrix.shape}"
                )

    def _parse_likelihood_matrices(self):
        """Parse likelihood matrices with generic dimensions."""
        likelihood_config = self.raw_config.get("likelihood_matrices", {})

        for matrix_name, matrix_data in likelihood_config.items():
            if "matrix" in matrix_data:
                matrix = np.array(matrix_data["matrix"], dtype=float)
                self.model_config.likelihood_matrices[matrix_name] = matrix
                logger.debug(
                    f"Parsed likelihood matrix '{matrix_name}': shape {matrix.shape}"
                )

    def _parse_prior_beliefs(self):
        """Parse prior belief vectors."""
        priors_config = self.raw_config.get("prior_beliefs", {})

        for prior_name, prior_data in priors_config.items():
            if "vector" in prior_data:
                vector = np.array(prior_data["vector"], dtype=float)
                self.model_config.prior_beliefs[prior_name] = vector
                logger.debug(f"Parsed prior '{prior_name}': shape {vector.shape}")

    def _parse_policies(self):
        """Parse policy configuration."""
        # Parse policy preferences
        preferences_config = self.raw_config.get("policy_preferences", {})
        for level_name, preferences in preferences_config.items():
            if isinstance(preferences, list):
                self.model_config.policy_preferences[level_name] = np.array(
                    preferences, dtype=float
                )

        # Parse policy priors
        priors_config = self.raw_config.get("policy_priors", {})
        for level_name, priors in priors_config.items():
            if isinstance(priors, list):
                self.model_config.policy_priors[level_name] = np.array(
                    priors, dtype=float
                )

        # Parse policy precision
        precision_config = self.raw_config.get("policy_precision", {})
        for precision_type, value in precision_config.items():
            if isinstance(value, (int, float)):
                self.model_config.policy_precision[precision_type] = float(value)

    def _parse_precision(self):
        """Parse precision parameters."""
        precision_config = self.raw_config.get("precision_bounds", {})

        for level_name, bounds_list in precision_config.items():
            if isinstance(bounds_list, list) and len(bounds_list) >= 2:
                bounds = tuple(bounds_list[:2])  # Take first two values as min, max
                self.model_config.precision_bounds[level_name] = bounds
                logger.debug(f"Parsed precision bounds for '{level_name}': {bounds}")
            else:
                logger.warning(
                    f"Invalid precision bounds format for '{level_name}': {bounds_list}"
                )

    def _parse_stimuli(self):
        """Parse stimulus configuration."""
        # Parse oddball pattern from model section
        model_info = self.raw_config.get("model", {})
        self.model_config.oddball_pattern = model_info.get("oddball_pattern", "default")
        self.model_config.oddball_times = model_info.get("oddball_times", [])

    def _parse_simulation_modes(self):
        """Parse simulation mode configurations."""
        modes_config = self.raw_config.get("simulation_modes", {})
        schedules_config = self.raw_config.get("simulation_schedules", {})

        # Extract mode mappings
        for key, value in modes_config.items():
            self.model_config.simulation_modes[key] = value

        # Extract simulation schedules
        for schedule_name, schedule_data in schedules_config.items():
            self.model_config.simulation_schedules[schedule_name] = schedule_data

    def _parse_output_configs(self):
        """Parse output, logging, and visualization configurations."""
        self.model_config.logging_config = self.raw_config.get("logging_config", {})
        self.model_config.visualization_config = self.raw_config.get(
            "visualization_config", {}
        )
        self.model_config.output_config = self.raw_config.get("output_config", {})
        self.model_config.numerical_config = self.raw_config.get("numerical_config", {})
        self.model_config.validation_config = self.raw_config.get(
            "validation_config", {}
        )
        self.model_config.extensions_config = self.raw_config.get(
            "extensions_config", {}
        )

    def _validate_config(self):
        """Validate parsed configuration for consistency."""
        logger.info("Validating configuration...")

        # Check that all required levels are defined
        expected_levels = set(self.model_config.level_names)
        actual_levels = set(self.model_config.levels.keys())

        if expected_levels != actual_levels:
            missing = expected_levels - actual_levels
            extra = actual_levels - expected_levels
            if missing:
                logger.warning(f"Missing level definitions: {missing}")
            if extra:
                logger.warning(f"Extra level definitions: {extra}")

        # Validate matrix dimensions
        if self.model_config.validation_config.get("check_matrix_dimensions", True):
            self._validate_matrix_dimensions()

        # Check normalization of probability matrices
        if self.model_config.validation_config.get("check_normalization", True):
            self._validate_normalization()

        logger.info("Configuration validation completed")

    def _validate_matrix_dimensions(self):
        """Validate that matrix dimensions are consistent with level definitions."""
        self.model_config.validation_config.get("tolerance", 1e-10)

        # Check transition matrices
        for matrix_name, matrix in self.model_config.transition_matrices.items():
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                logger.warning(
                    f"Transition matrix '{matrix_name}' is not square: {matrix.shape}"
                )

        # Check likelihood matrices
        for matrix_name, matrix in self.model_config.likelihood_matrices.items():
            if matrix.ndim != 2:
                logger.warning(
                    f"Likelihood matrix '{matrix_name}' is not 2D: {matrix.shape}"
                )

    def _validate_normalization(self):
        """Check that probability matrices are properly normalized."""
        tolerance = self.model_config.validation_config.get("tolerance", 1e-10)

        # Check transition matrices (rows should sum to 1)
        for matrix_name, matrix in self.model_config.transition_matrices.items():
            row_sums = np.sum(matrix, axis=1)
            if not np.allclose(row_sums, 1.0, atol=tolerance):
                logger.warning(
                    f"Transition matrix '{matrix_name}' rows don't sum to 1: {row_sums}"
                )

        # Check likelihood matrices (columns should sum to 1)
        for matrix_name, matrix in self.model_config.likelihood_matrices.items():
            col_sums = np.sum(matrix, axis=0)
            if not np.allclose(col_sums, 1.0, atol=tolerance):
                logger.warning(
                    f"Likelihood matrix '{matrix_name}' columns don't sum to 1: {col_sums}"
                )

        # Check prior beliefs (should sum to 1)
        for prior_name, prior in self.model_config.prior_beliefs.items():
            prior_sum = np.sum(prior)
            if not np.allclose(prior_sum, 1.0, atol=tolerance):
                logger.warning(f"Prior '{prior_name}' doesn't sum to 1: {prior_sum}")

    def get_level_config(self, level_name: str) -> Optional[LevelConfig]:
        """Get configuration for a specific level."""
        return self.model_config.levels.get(level_name)

    def get_matrix(self, matrix_type: str, matrix_name: str) -> Optional[np.ndarray]:
        """
        Get a specific matrix from the configuration.

        Args:
            matrix_type: 'transition', 'likelihood', or 'prior'
            matrix_name: Name of the specific matrix

        Returns:
            Matrix as numpy array, or None if not found
        """
        if matrix_type == "transition":
            return self.model_config.transition_matrices.get(matrix_name)
        elif matrix_type == "likelihood":
            return self.model_config.likelihood_matrices.get(matrix_name)
        elif matrix_type == "prior":
            return self.model_config.prior_beliefs.get(matrix_name)
        else:
            logger.error(f"Unknown matrix type: {matrix_type}")
            return None

    def export_config_summary(
        self, output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Export a summary of the parsed configuration."""
        summary = {
            "model_info": {
                "name": self.model_config.name,
                "description": self.model_config.description,
                "version": self.model_config.version,
                "paper_reference": self.model_config.paper_reference,
            },
            "structure": {
                "num_levels": self.model_config.num_levels,
                "level_names": self.model_config.level_names,
                "time_steps": self.model_config.time_steps,
            },
            "level_dimensions": {
                name: {
                    "state_dim": config.state_dim,
                    "obs_dim": config.obs_dim,
                    "action_dim": config.action_dim,
                }
                for name, config in self.model_config.levels.items()
            },
            "matrix_shapes": {
                "transition": {
                    name: matrix.shape
                    for name, matrix in self.model_config.transition_matrices.items()
                },
                "likelihood": {
                    name: matrix.shape
                    for name, matrix in self.model_config.likelihood_matrices.items()
                },
                "priors": {
                    name: vector.shape
                    for name, vector in self.model_config.prior_beliefs.items()
                },
            },
        }

        if output_path:
            import json

            with open(output_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info(f"Configuration summary exported to {output_path}")

        return summary


def load_gnn_config(config_path: Union[str, Path]) -> ModelConfig:
    """
    Convenience function to load and parse a GNN configuration file.

    Args:
        config_path: Path to GNN configuration file

    Returns:
        Parsed ModelConfig object
    """
    # Resolve configuration path robustly for different working directories
    cp = Path(config_path)
    if not cp.exists():
        # Try alongside this module (the config directory)
        candidates = [
            Path(__file__).parent / cp,
            Path(__file__).parent / cp.name,
        ]
        for c in candidates:
            if c.exists():
                cp = c
                break
    parser = GNNConfigParser(cp)
    return parser.parse_config()


# Example usage
if __name__ == "__main__":
    # Example of loading a configuration
    config_file = Path(__file__).parent / "meta_awareness_gnn.toml"

    try:
        parser = GNNConfigParser(config_file)
        model_config = parser.parse_config()

        print("Configuration loaded successfully!")
        print(f"Model: {model_config.name}")
        print(f"Levels: {model_config.level_names}")
        print(f"Time steps: {model_config.time_steps}")

        # Export summary
        summary = parser.export_config_summary("config_summary.json")
        print("Configuration summary exported.")

    except Exception as e:
        print(f"Error loading configuration: {e}")
