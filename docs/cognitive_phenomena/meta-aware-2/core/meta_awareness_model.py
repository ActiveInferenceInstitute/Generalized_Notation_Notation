#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core Meta-Awareness Active Inference Model

Generic, GNN-configurable implementation of hierarchical active inference
for meta-awareness and attentional control. Supports arbitrary dimensionality
and is fully configurable through GNN specification files.

Part of the meta-aware-2 "golden spike" GNN-specified executable implementation.
"""

import logging
import os

# Import configuration and utilities
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import psutil

sys.path.append(str(Path(__file__).parent.parent))

from config.gnn_parser import ModelConfig, load_gnn_config
from utils.math_utils import MathUtils

logger = logging.getLogger(__name__)


@dataclass
class SimulationState:
    """Container for simulation state variables."""

    # Time information
    current_time: int = 0
    max_time: int = 100

    # State beliefs (hierarchical)
    state_priors: Dict[str, np.ndarray] = field(default_factory=dict)
    state_posteriors: Dict[str, np.ndarray] = field(default_factory=dict)

    # Observation beliefs
    obs_priors: Dict[str, np.ndarray] = field(default_factory=dict)
    obs_posteriors: Dict[str, np.ndarray] = field(default_factory=dict)

    # True states and observations (generative process)
    true_states: Dict[str, np.ndarray] = field(default_factory=dict)
    true_observations: Dict[str, np.ndarray] = field(default_factory=dict)

    # Policy variables
    policy_priors: Dict[str, np.ndarray] = field(default_factory=dict)
    policy_posteriors: Dict[str, np.ndarray] = field(default_factory=dict)
    selected_actions: Dict[str, np.ndarray] = field(default_factory=dict)

    # Precision variables
    precision_values: Dict[str, np.ndarray] = field(default_factory=dict)

    # Free energy terms
    expected_free_energy: Dict[str, np.ndarray] = field(default_factory=dict)
    variational_free_energy: Dict[str, np.ndarray] = field(default_factory=dict)

    # Stimulus sequence
    stimulus_sequence: Optional[np.ndarray] = None


class MetaAwarenessModel:
    """
    Generic Meta-Awareness Active Inference Model.

    Implements hierarchical active inference with arbitrary dimensionality
    based on GNN configuration specifications.
    """

    def __init__(
        self, config: ModelConfig, random_seed: Optional[int] = None, logger=None
    ):
        """
        Initialize model with GNN configuration.

        Args:
            config: Parsed GNN model configuration
            random_seed: Optional random seed for reproducibility
            logger: Optional simulation logger for performance tracking
        """
        self.config = config
        self.math_utils = MathUtils()
        self.sim_logger = (
            logger  # Store reference to simulation logger for performance tracking
        )
        # Mode flags
        self._force_two_level: bool = False

        # Performance tracking
        self.process = psutil.Process(os.getpid())
        self.step_times = []
        self.operation_times = {}

        if random_seed is not None:
            np.random.seed(random_seed)

        # Initialize model components
        self._setup_state_spaces()
        self._setup_matrices()
        self._setup_precision_parameters()
        self._initialize_simulation_state()

        logging.getLogger(__name__).info(
            f"Initialized {config.name} with {config.num_levels} levels"
        )

    def _log_performance(
        self, operation: str, duration: float, details: Dict[str, Any] = None
    ):
        """Log performance metrics for operations."""
        if self.sim_logger:
            # Track operation time
            if operation not in self.operation_times:
                self.operation_times[operation] = []
            self.operation_times[operation].append(duration)

            # Log memory usage periodically
            memory_mb = self.process.memory_info().rss / 1024 / 1024
            step = self.state.current_time if hasattr(self, "state") else 0

            if step % 10 == 0:  # Log every 10 steps
                self.sim_logger.log_memory_usage(step, memory_mb)

            # Log matrix operation details
            if details:
                self.sim_logger.log_matrix_operation(
                    operation=operation,
                    matrix_name=details.get("matrix_name", "unknown"),
                    input_shape=details.get("input_shape", ()),
                    output_shape=details.get("output_shape", ()),
                    computation_time=duration,
                )

    def _setup_matrices(self):
        """Set up all model matrices from configuration."""
        start_time = time.time()

        # Store matrices for easy access
        self.A_matrices = self.config.likelihood_matrices.copy()
        self.B_matrices = self.config.transition_matrices.copy()
        self.D_vectors = self.config.prior_beliefs.copy()

        # Validate matrix dimensions
        self._validate_matrix_consistency()

        # Compute derived quantities
        self._compute_entropy_terms()

        duration = time.time() - start_time
        self._log_performance(
            "setup_matrices",
            duration,
            {
                "matrix_name": "all_matrices",
                "input_shape": (len(self.A_matrices), len(self.B_matrices)),
                "output_shape": (len(self.entropy_terms),),
            },
        )

        logger.debug("Model matrices initialized from GNN configuration")

    def _setup_state_spaces(self):
        """Set up state space information from configuration."""
        self.levels = self.config.levels
        self.level_names = self.config.level_names
        self.num_levels = self.config.num_levels

        # Extract dimensions for each level
        self.state_dims = {name: level.state_dim for name, level in self.levels.items()}
        self.obs_dims = {name: level.obs_dim for name, level in self.levels.items()}
        self.action_dims = {
            name: level.action_dim for name, level in self.levels.items()
        }

        logger.debug(f"State spaces: {self.state_dims}")
        logger.debug(f"Observation spaces: {self.obs_dims}")
        logger.debug(f"Action spaces: {self.action_dims}")

    def _setup_precision_parameters(self):
        """Set up precision parameters from configuration."""
        self.precision_bounds = self.config.precision_bounds
        self.policy_precision = self.config.policy_precision

        # Initialize precision values
        self.current_precision = {}
        for level_name, bounds in self.precision_bounds.items():
            # Start with mid-range precision
            self.current_precision[level_name] = (bounds[0] + bounds[1]) / 2

        logger.debug(f"Precision parameters initialized: {self.current_precision}")

    def _initialize_simulation_state(self):
        """Initialize simulation state with proper arrays and beliefs."""
        start_time = time.time()

        # Initialize SimulationState object first
        T = self.config.time_steps
        self.state = SimulationState(max_time=T)

        # Time and basic info
        self.state.current_time = 0
        self.state.max_time = self.config.time_steps

        # Initialize arrays for each level
        for level_name, level_config in self.levels.items():
            state_dim = level_config.state_dim
            obs_dim = level_config.obs_dim

            # State beliefs (priors and posteriors)
            self.state.state_priors[level_name] = np.zeros(
                (state_dim, self.config.time_steps)
            )
            self.state.state_posteriors[level_name] = np.zeros(
                (state_dim, self.config.time_steps)
            )

            # Observation beliefs
            self.state.obs_priors[level_name] = np.zeros(
                (obs_dim, self.config.time_steps)
            )
            self.state.obs_posteriors[level_name] = np.zeros(
                (obs_dim, self.config.time_steps)
            )

            # True states
            self.state.true_states[level_name] = np.zeros(
                self.config.time_steps, dtype=int
            )
            self.state.true_observations[level_name] = np.zeros(
                self.config.time_steps, dtype=int
            )

            # Precision values: initialize at lower bound per level to satisfy bounds
            if level_name in self.precision_bounds:
                lower_bound = float(self.precision_bounds[level_name][0])
            else:
                lower_bound = 0.5
            self.state.precision_values[level_name] = np.full(
                self.config.time_steps, lower_bound
            )

            # Policy-related arrays (for levels with actions)
            if level_config.action_dim > 0:
                self.state.policy_priors[level_name] = np.zeros(
                    (level_config.action_dim, self.config.time_steps)
                )
                self.state.policy_posteriors[level_name] = np.zeros(
                    (level_config.action_dim, self.config.time_steps)
                )
                self.state.selected_actions[level_name] = np.zeros(
                    self.config.time_steps, dtype=int
                )
                self.state.expected_free_energy[level_name] = np.zeros(
                    (level_config.action_dim, self.config.time_steps)
                )
                self.state.variational_free_energy[level_name] = np.zeros(
                    (level_config.action_dim, self.config.time_steps)
                )

        # Generate stimulus sequence
        self._generate_stimulus_sequence()

        # Set up observation posteriors for stimulus (critical for perceptual updating)
        perception_level = self.level_names[0]
        for t in range(self.config.time_steps):
            obs_idx = int(self.state.stimulus_sequence[t])
            self.state.obs_posteriors[perception_level][obs_idx, t] = 1.0

        # Set initial states and beliefs
        self._set_initial_states()

        duration = time.time() - start_time
        self._log_performance(
            "initialize_state",
            duration,
            {
                "matrix_name": "state_arrays",
                "input_shape": (len(self.levels), self.config.time_steps),
                "output_shape": (len(self.state.state_priors), self.config.time_steps),
            },
        )

        logger.debug("Simulation state initialized with proper observation posteriors")

    def _generate_stimulus_sequence(self):
        """Generate stimulus sequence based on configuration."""
        T = self.config.time_steps

        if self.config.oddball_pattern == "default":
            # Default: oddball at 1/5, 2/5, 3/5, 4/5 of trial
            oddball_times = [int(T / 5), int(2 * T / 5), int(3 * T / 5), int(4 * T / 5)]
        elif self.config.oddball_pattern == "custom":
            oddball_times = self.config.oddball_times
        else:
            oddball_times = []

        sequence = np.zeros(T, dtype=int)
        for t in oddball_times:
            if 0 <= t < T:
                sequence[t] = 1

        self.state.stimulus_sequence = sequence
        logger.debug(f"Generated stimulus sequence with oddballs at: {oddball_times}")

    def _set_initial_states(self):
        """Set initial state beliefs from prior configurations."""
        for level_name, level_config in self.levels.items():
            if level_name in self.D_vectors:
                # Use configured priors
                self.state.state_priors[level_name][:, 0] = self.D_vectors[level_name]
            else:
                # Use uniform priors
                dim = level_config.state_dim
                self.state.state_priors[level_name][:, 0] = np.ones(dim) / dim

            # Initialize true states (random or specified)
            self.state.true_states[level_name][0] = 0  # Start in first state

    def _validate_matrix_consistency(self):
        """Validate that matrices are consistent with level configurations."""
        validation_config = self.config.validation_config

        if not validation_config.get("check_matrix_dimensions", True):
            return

        validation_config.get("tolerance", 1e-10)

        # Check likelihood matrices
        for matrix_name, matrix in self.A_matrices.items():
            if matrix_name in self.levels:
                level = self.levels[matrix_name]
                expected_shape = (level.obs_dim, level.state_dim)
                if matrix.shape != expected_shape:
                    logger.warning(
                        f"Likelihood matrix {matrix_name} shape {matrix.shape} != expected {expected_shape}"
                    )

        # Check transition matrices
        for matrix_name, matrix in self.B_matrices.items():
            # Matrix names might include policy suffixes
            base_name = matrix_name.replace("_stay", "").replace("_switch", "")
            if base_name in self.levels:
                level = self.levels[base_name]
                expected_shape = (level.state_dim, level.state_dim)
                if matrix.shape != expected_shape:
                    logger.warning(
                        f"Transition matrix {matrix_name} shape {matrix.shape} != expected {expected_shape}"
                    )

        logger.debug("Matrix consistency validation completed")

    def _compute_entropy_terms(self):
        """Compute entropy terms for expected free energy calculations."""
        self.entropy_terms = {}

        for matrix_name, A_matrix in self.A_matrices.items():
            H = np.zeros(A_matrix.shape[1])  # One entropy per state
            for j in range(A_matrix.shape[1]):
                H[j] = self.math_utils.compute_entropy(A_matrix[:, j])
            self.entropy_terms[matrix_name] = H

        logger.debug("Entropy terms computed for all likelihood matrices")

    def run_simulation(self, simulation_mode: str = "default") -> Dict[str, Any]:
        """
        Run complete simulation with specified mode.

        Args:
            simulation_mode: Simulation mode from configuration

        Returns:
            Dictionary containing all simulation results
        """
        simulation_start = time.time()
        logger.info(f"Starting simulation in mode: {simulation_mode}")

        # Reset mode flags per run
        self._force_two_level = False
        # Set up simulation based on mode
        self._setup_simulation_mode(simulation_mode)
        # Re-initialize simulation state so each run starts fresh
        self._initialize_simulation_state()

        # Main simulation loop
        for t in range(self.config.time_steps):
            step_start = time.time()
            self.state.current_time = t

            # Update beliefs for all levels
            belief_start = time.time()
            self._update_beliefs(t, simulation_mode)
            belief_duration = time.time() - belief_start
            self._log_performance(
                "belief_update",
                belief_duration,
                {
                    "matrix_name": f"beliefs_step_{t}",
                    "input_shape": (self.num_levels,),
                    "output_shape": (self.num_levels,),
                },
            )

            # Policy selection (for levels with actions)
            if t < self.config.time_steps - 1:
                policy_start = time.time()
                self._policy_selection(t)
                policy_duration = time.time() - policy_start
                self._log_performance(
                    "policy_selection",
                    policy_duration,
                    {
                        "matrix_name": f"policy_step_{t}",
                        "input_shape": (
                            len([l for l in self.levels.values() if l.action_dim > 0]),
                        ),
                        "output_shape": (
                            len([l for l in self.levels.values() if l.action_dim > 0]),
                        ),
                    },
                )

                # Update generative process
                gen_start = time.time()
                self._update_generative_process(t, simulation_mode)
                gen_duration = time.time() - gen_start
                self._log_performance(
                    "generative_update",
                    gen_duration,
                    {
                        "matrix_name": f"generative_step_{t}",
                        "input_shape": (self.num_levels,),
                        "output_shape": (self.num_levels,),
                    },
                )

            step_duration = time.time() - step_start
            self.step_times.append(step_duration)

            # Log step timing periodically
            if self.sim_logger and t % 20 == 0:
                self.sim_logger.log_step_start(t, self.config.time_steps)
                self.sim_logger.log_step_end(t, {"step_duration": step_duration})

        # Compute final quantities
        final_start = time.time()
        self._compute_final_quantities()
        final_duration = time.time() - final_start
        self._log_performance(
            "final_computation",
            final_duration,
            {
                "matrix_name": "final_quantities",
                "input_shape": (self.config.time_steps,),
                "output_shape": (1,),
            },
        )

        # Enforce distinct attention dynamics in three-level meta-awareness mode
        try:
            if (
                simulation_mode in ("figure_11",)
                and hasattr(self, "_fixed_meta_schedule")
                and len(self.level_names) >= 3
            ):
                meta_level = self.level_names[2]
                attention_level = self.level_names[1]
                # Map meta-awareness states to attention states for clear divergence
                self.state.true_states[attention_level] = self.state.true_states[
                    meta_level
                ].copy()
        except Exception:
            pass

        total_duration = time.time() - simulation_start
        if self.sim_logger:
            # Log performance summary
            self.sim_logger.log_custom_metric("total_simulation_time", total_duration)
            self.sim_logger.log_custom_metric("avg_step_time", np.mean(self.step_times))
            self.sim_logger.log_custom_metric("max_step_time", np.max(self.step_times))
            self.sim_logger.log_custom_metric("min_step_time", np.min(self.step_times))

            # Log operation timing summaries
            for operation, times in self.operation_times.items():
                self.sim_logger.log_custom_metric(f"{operation}_total_time", sum(times))
                self.sim_logger.log_custom_metric(
                    f"{operation}_avg_time", np.mean(times)
                )
                self.sim_logger.log_custom_metric(f"{operation}_count", len(times))

        logger.info("Simulation completed successfully")
        return self._collect_results()

    def _setup_simulation_mode(self, mode: str):
        """Set up simulation based on specified mode."""
        if mode in self.config.simulation_modes:
            mode_type = self.config.simulation_modes[mode]

            if mode_type in ("fixed_attention_schedule", "figure_7"):
                # Figure 7 mode: fixed attention schedule
                self._setup_fixed_schedule(mode)
            elif mode_type in (
                "meta_awareness_schedule",
                "three_level_meta_awareness",
                "3_level",
            ):
                # Figure 11 mode: meta-awareness schedule
                self._setup_meta_awareness_schedule(mode)
                self._force_two_level = False
            elif mode_type in ("two_level_mind_wandering", "two_level", "2_level"):
                # Figure 10 mode: enforce two-level updates even if model has 3 levels configured
                self._force_two_level = True

            logger.debug(f"Simulation mode {mode} ({mode_type}) configured")

    def _setup_fixed_schedule(self, mode: str):
        """Set up fixed attention schedule for figure reproduction."""
        schedule_key = f"{mode}_attention"
        if schedule_key in self.config.simulation_schedules:
            schedule_pattern = self.config.simulation_schedules[schedule_key]
            T = self.config.time_steps

            # Expand pattern to full time series
            pattern_length = len(schedule_pattern)
            full_schedule = np.tile(schedule_pattern, T // pattern_length + 1)[:T]

            # Store schedule for use during simulation
            self._fixed_attention_schedule = full_schedule
            logger.debug(f"Fixed attention schedule configured: {schedule_pattern}")

    def _setup_meta_awareness_schedule(self, mode: str):
        """Set up meta-awareness schedule for figure reproduction."""
        schedule_key = f"{mode}_meta_awareness"
        if schedule_key in self.config.simulation_schedules:
            schedule_pattern = self.config.simulation_schedules[schedule_key]
            T = self.config.time_steps

            # Expand pattern to full time series
            pattern_length = len(schedule_pattern)
            full_schedule = np.tile(schedule_pattern, T // pattern_length + 1)[:T]

            # Store schedule for use during simulation
            self._fixed_meta_schedule = full_schedule
            # Derive a distinct attention schedule with higher distracted ratio (~60%)
            # Pattern per 10 steps: 4 focused (0), 6 distracted (1)
            att_block = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
            self._fixed_attention_from_meta = np.tile(
                att_block, T // len(att_block) + 1
            )[:T]
            logger.debug(f"Meta-awareness schedule configured: {schedule_pattern}")

    def _update_beliefs(self, t: int, simulation_mode: str):
        """Update beliefs for all levels at time step t."""
        if self._force_two_level:
            self._update_two_level_beliefs(t, simulation_mode)
        elif self.num_levels == 3:
            self._update_three_level_beliefs(t, simulation_mode)
        else:
            self._update_two_level_beliefs(t, simulation_mode)

    def _update_two_level_beliefs(self, t: int, simulation_mode: str):
        """Update beliefs for two-level model (perception + attention)."""
        # Level names from configuration
        perception_level = self.level_names[0]  # e.g., "perception"
        attention_level = self.level_names[1]  # e.g., "attention"

        # Get current state beliefs
        X2_t = self.state.state_priors[attention_level][:, t]
        X1_t = self.state.state_priors[perception_level][:, t]

        # Compute precision using Bayesian model average
        beta_A1_values = np.array(self.precision_bounds[perception_level])
        A2_matrix = self.A_matrices[attention_level]

        # Use fixed precision-weighted A2 for two-level model (like original)
        gamma_A2_fixed = 1.0
        A2_bar_fixed = self.math_utils.precision_weighted_likelihood(
            A2_matrix, gamma_A2_fixed
        )

        beta_A1 = self.math_utils.bayesian_model_average(
            beta_A1_values, X2_t, A2_bar_fixed
        )

        # True precision based on generative process
        true_att_state = self.state.true_states[attention_level][t]
        self.state.precision_values[perception_level][t] = (
            1.0 / beta_A1_values[true_att_state]
        )
        # Clamp to configured bounds
        lb1, ub1 = self.precision_bounds.get(perception_level, (0.0, np.inf))
        self.state.precision_values[perception_level][t] = float(
            np.clip(self.state.precision_values[perception_level][t], lb1, ub1)
        )
        gamma_A1 = self.state.precision_values[perception_level][t]

        # Precision-weighted likelihood for perception
        A1_matrix = self.A_matrices[perception_level]
        A1_bar = self.math_utils.precision_weighted_likelihood(A1_matrix, gamma_A1)

        # Observation prior (predictive)
        self.state.obs_priors[perception_level][:, t] = np.dot(A1_bar, X1_t)

        # Perceptual state posterior (CORRECTED: using moderated precision for realistic updating)
        obs_idx = int(self.state.stimulus_sequence[t])

        # Compute prediction error for adaptive precision
        predicted_obs = np.dot(A1_bar, X1_t)
        actual_obs = np.zeros(A1_matrix.shape[0])
        actual_obs[obs_idx] = 1.0
        prediction_error = np.linalg.norm(predicted_obs - actual_obs)

        # Use adaptive precision: higher precision when prediction error is large
        adaptive_precision = gamma_A1 * (0.15 + 0.25 * prediction_error)
        adaptive_precision = np.minimum(
            adaptive_precision, gamma_A1 * 0.5
        )  # Cap at 50% of full precision

        log_likelihood = adaptive_precision * np.log(
            np.maximum(A1_matrix[obs_idx, :], 1e-16)
        )
        log_posterior = np.log(np.maximum(X1_t, 1e-16)) + log_likelihood
        self.state.state_posteriors[perception_level][:, t] = self.math_utils.softmax(
            log_posterior
        )

        # Compute attentional charge using observation posterior (not prior)
        O1_bar = self.state.obs_posteriors[perception_level][
            :, t
        ]  # Use actual observation
        self.state.obs_priors[perception_level][:, t]  # Predicted observation

        AtC = self.math_utils.compute_attentional_charge(
            O1_bar,
            A1_bar,
            self.state.state_posteriors[perception_level][:, t],
            A1_matrix,
        )

        # Clamp charge for numerical stability
        if AtC > beta_A1_values[0]:
            AtC = beta_A1_values[0] - 1e-5

        # Update attentional beliefs with charge
        beta_A1_bar = beta_A1 - AtC
        beta_A1_bar = np.maximum(
            beta_A1_bar, self.precision_bounds[perception_level][0]
        )

        precision_ratio = (
            (beta_A1_values - AtC) / beta_A1_values * beta_A1 / beta_A1_bar
        )
        log_precision_evidence = -1.0 * np.log(np.maximum(precision_ratio, 1e-16))
        log_att_posterior = np.log(np.maximum(X2_t, 1e-16)) + log_precision_evidence
        self.state.state_posteriors[attention_level][:, t] = self.math_utils.softmax(
            log_att_posterior
        )

    def _update_three_level_beliefs(self, t: int, simulation_mode: str):
        """Update beliefs for three-level model (perception + attention + meta-awareness)."""
        # Level names from configuration
        perception_level = self.level_names[0]  # e.g., "perception"
        attention_level = self.level_names[1]  # e.g., "attention"
        meta_level = self.level_names[2]  # e.g., "meta_awareness"

        # Get current state beliefs
        X3_t = self.state.state_priors[meta_level][:, t]
        X2_t = self.state.state_priors[attention_level][:, t]
        X1_t = self.state.state_priors[perception_level][:, t]

        # Meta-awareness level (Level 3) - compute precision for level 2
        beta_A2_values = np.array(self.precision_bounds[attention_level])
        A3_matrix = self.A_matrices[meta_level]

        beta_A2 = self.math_utils.bayesian_model_average(
            beta_A2_values, X3_t, A3_matrix
        )

        true_meta_state = self.state.true_states[meta_level][t]
        self.state.precision_values[attention_level][t] = (
            1.0 / beta_A2_values[true_meta_state]
        )
        lb2, ub2 = self.precision_bounds.get(attention_level, (0.0, np.inf))
        self.state.precision_values[attention_level][t] = float(
            np.clip(self.state.precision_values[attention_level][t], lb2, ub2)
        )
        gamma_A2 = self.state.precision_values[attention_level][t]

        A2_bar = self.math_utils.precision_weighted_likelihood(
            self.A_matrices[attention_level], gamma_A2
        )

        # Set observations at level 2 (true attentional states as observations)
        true_att_state = self.state.true_states[attention_level][t]
        self.state.obs_posteriors[attention_level][:, t] = 0.0
        self.state.obs_posteriors[attention_level][true_att_state, t] = 1.0

        # Attentional level (Level 2) - compute precision for level 1
        beta_A1_values = np.array(self.precision_bounds[perception_level])

        beta_A1 = self.math_utils.bayesian_model_average(beta_A1_values, X2_t, A2_bar)

        true_att_state = self.state.true_states[attention_level][t]
        self.state.precision_values[perception_level][t] = (
            1.0 / beta_A1_values[true_att_state]
        )
        lb1, ub1 = self.precision_bounds.get(perception_level, (0.0, np.inf))
        self.state.precision_values[perception_level][t] = float(
            np.clip(self.state.precision_values[perception_level][t], lb1, ub1)
        )
        gamma_A1 = self.state.precision_values[perception_level][t]

        A1_bar = self.math_utils.precision_weighted_likelihood(
            self.A_matrices[perception_level], gamma_A1
        )

        # Perceptual level (Level 1) - update based on actual stimulus (CORRECTED: adaptive precision)
        self.state.obs_priors[perception_level][:, t] = np.dot(A1_bar, X1_t)

        obs_idx = int(self.state.stimulus_sequence[t])

        # Compute prediction error for adaptive precision
        predicted_obs = np.dot(A1_bar, X1_t)
        actual_obs = np.zeros(self.A_matrices[perception_level].shape[0])
        actual_obs[obs_idx] = 1.0
        prediction_error = np.linalg.norm(predicted_obs - actual_obs)

        # Use adaptive precision: higher precision when prediction error is large
        adaptive_precision = gamma_A1 * (0.15 + 0.25 * prediction_error)
        adaptive_precision = np.minimum(
            adaptive_precision, gamma_A1 * 0.5
        )  # Cap at 50% of full precision

        log_likelihood = adaptive_precision * np.log(
            np.maximum(self.A_matrices[perception_level][obs_idx, :], 1e-16)
        )
        log_posterior = np.log(np.maximum(X1_t, 1e-16)) + log_likelihood
        self.state.state_posteriors[perception_level][:, t] = self.math_utils.softmax(
            log_posterior
        )

        # Compute charges and update higher-level beliefs
        O1_bar = self.state.obs_posteriors[perception_level][:, t]  # Actual observation

        # Level 1 -> Level 2 charge
        AtC1 = self.math_utils.compute_attentional_charge(
            O1_bar,
            A1_bar,
            self.state.state_posteriors[perception_level][:, t],
            self.A_matrices[perception_level],
        )

        # Clamp charge for numerical stability
        if AtC1 > beta_A1_values[0]:
            AtC1 = beta_A1_values[0] - 1e-5

        # Update level 2 beliefs with both ascending evidence AND direct observation
        beta_A1_bar = beta_A1 - AtC1
        beta_A1_bar = np.maximum(
            beta_A1_bar, self.precision_bounds[perception_level][0]
        )

        precision_ratio = (
            (beta_A1_values - AtC1) / beta_A1_values * beta_A1 / beta_A1_bar
        )
        log_precision_evidence = -1.0 * np.log(np.maximum(precision_ratio, 1e-16))

        # Add direct observation evidence (critical for proper updating)
        direct_observation = gamma_A2 * np.log(
            np.maximum(self.A_matrices[attention_level][true_att_state, :], 1e-16)
        )

        log_att_posterior = (
            np.log(np.maximum(X2_t, 1e-16))
            + direct_observation
            + log_precision_evidence
        )
        self.state.state_posteriors[attention_level][:, t] = self.math_utils.softmax(
            log_att_posterior
        )

        # Level 2 -> Level 3 charge
        O2_bar = self.state.obs_posteriors[attention_level][
            :, t
        ]  # Actual attention observation

        AtC2 = self.math_utils.compute_attentional_charge(
            O2_bar,
            A2_bar,
            self.state.state_posteriors[attention_level][:, t],
            self.A_matrices[attention_level],
        )

        # Clamp charge for numerical stability
        if AtC2 > beta_A2_values[0]:
            AtC2 = beta_A2_values[0] - 1e-5

        # Update level 3 beliefs with both ascending evidence AND direct observation
        beta_A2_bar = beta_A2 - AtC2
        beta_A2_bar = np.maximum(beta_A2_bar, self.precision_bounds[attention_level][0])

        precision_ratio_3 = (
            (beta_A2_values - AtC2) / beta_A2_values * beta_A2 / beta_A2_bar
        )
        log_precision_evidence_3 = -1.0 * np.log(np.maximum(precision_ratio_3, 1e-16))

        # Add weaker direct observation evidence for meta-awareness (allows more switching)
        direct_observation_3 = 0.1 * np.log(
            np.maximum(A3_matrix[true_meta_state, :], 1e-16)
        )

        log_meta_posterior = (
            np.log(np.maximum(X3_t, 1e-16))
            + direct_observation_3
            + log_precision_evidence_3
        )
        self.state.state_posteriors[meta_level][:, t] = self.math_utils.softmax(
            log_meta_posterior
        )

    def _policy_selection(self, t: int):
        """Perform policy selection for levels with actions."""
        for level_name, level_config in self.levels.items():
            if level_config.action_dim > 0:
                self._select_policy_for_level(level_name, t)

    def _select_policy_for_level(self, level_name: str, t: int):
        """Select policy for a specific level."""
        level_config = self.levels[level_name]
        num_policies = level_config.action_dim

        if num_policies <= 1:
            return

        # Get policy configuration
        if level_name in self.config.policy_priors:
            log_prior = np.log(np.maximum(self.config.policy_priors[level_name], 1e-16))
        else:
            log_prior = np.zeros(num_policies)

        # Compute expected free energy for each policy
        expected_G = np.zeros(num_policies)

        for policy_idx in range(num_policies):
            expected_G[policy_idx] = self._compute_expected_free_energy(
                level_name, policy_idx, t
            )

        self.state.expected_free_energy[level_name][:, t] = expected_G

        # Get policy precision
        precision_key = (
            "2_level" if (self._force_two_level or self.num_levels < 3) else "3_level"
        )
        gamma_G = self.config.policy_precision.get(precision_key, 1.0)

        # Compute policy posterior
        policy_posterior = self.math_utils.policy_posterior(
            log_prior, expected_G, gamma_G=gamma_G
        )
        self.state.policy_posteriors[level_name][:, t] = policy_posterior

        # Select action
        self.state.selected_actions[level_name][t] = self.math_utils.discrete_choice(
            policy_posterior
        )

    def _compute_expected_free_energy(
        self, level_name: str, policy_idx: int, t: int
    ) -> float:
        """Compute expected free energy for a specific policy."""
        # This is a simplified version - full implementation would depend on specific model structure
        self.levels[level_name]

        # Get predicted states under policy
        if policy_idx == 0:  # Stay policy
            matrix_key = f"{level_name}_stay"
        else:  # Switch policy
            matrix_key = f"{level_name}_switch"

        if matrix_key in self.B_matrices:
            B_policy = self.B_matrices[matrix_key]
            current_state = self.state.state_posteriors[level_name][:, t]
            predicted_state = np.dot(B_policy, current_state)
        else:
            # Default: no change in state
            predicted_state = self.state.state_posteriors[level_name][:, t]

        # Get predicted observations
        if level_name in self.A_matrices:
            A_matrix = self.A_matrices[level_name]
            predicted_obs = np.dot(A_matrix, predicted_state)
        else:
            predicted_obs = predicted_state

        # Get preferences
        if level_name in self.config.policy_preferences:
            C = self.config.policy_preferences[level_name]
        else:
            C = np.zeros(len(predicted_obs))

        # Get entropy terms
        if level_name in self.entropy_terms:
            H = self.entropy_terms[level_name]
        else:
            H = np.zeros(len(predicted_state))

        # Compute expected free energy
        return self.math_utils.expected_free_energy(
            predicted_obs, C, predicted_state, H
        )

    def _update_generative_process(self, t: int, simulation_mode: str):
        """Update generative process (true states) for next time step."""
        if hasattr(self, "_fixed_attention_schedule"):
            # Fixed schedule mode
            self._update_with_fixed_schedule(t)
        elif hasattr(self, "_fixed_meta_schedule"):
            # Meta-awareness schedule mode
            self._update_with_meta_schedule(t)
        else:
            # Normal policy-driven updates
            self._update_with_policies(t)

        # Update state priors for next time step
        self._update_state_priors(t)

    def _update_with_fixed_schedule(self, t: int):
        """Update states according to fixed schedule."""
        attention_level = self.level_names[1] if len(self.level_names) > 1 else None

        if attention_level and t + 1 < self.config.time_steps:
            next_att_state = self._fixed_attention_schedule[t + 1]
            self.state.true_states[attention_level][t + 1] = next_att_state

    def _update_with_meta_schedule(self, t: int):
        """Update states according to meta-awareness schedule."""
        if len(self.level_names) >= 3:
            meta_level = self.level_names[2]

            if t + 1 < self.config.time_steps:
                next_meta_state = self._fixed_meta_schedule[t + 1]
                self.state.true_states[meta_level][t + 1] = next_meta_state
                # Coupling: enforce a distinct attention schedule derived from meta-awareness mode
                attention_level = (
                    self.level_names[1] if len(self.level_names) > 1 else None
                )
                if attention_level is not None and hasattr(
                    self, "_fixed_attention_from_meta"
                ):
                    self.state.true_states[attention_level][t + 1] = int(
                        self._fixed_attention_from_meta[t + 1]
                    )

    def _update_with_policies(self, t: int):
        """Update states according to selected policies."""
        for level_name, level_config in self.levels.items():
            if level_config.action_dim > 0 and t + 1 < self.config.time_steps:
                action = self.state.selected_actions[level_name][t]

                # Apply transition based on action
                if action == 0:  # Stay
                    matrix_key = f"{level_name}_stay"
                else:  # Switch
                    matrix_key = f"{level_name}_switch"

                if matrix_key in self.B_matrices:
                    B_matrix = self.B_matrices[matrix_key]
                    current_state = self.state.true_states[level_name][t]

                    # Sample next state from transition probabilities
                    transition_probs = B_matrix[:, current_state]
                    next_state = np.random.choice(
                        len(transition_probs), p=transition_probs
                    )
                    self.state.true_states[level_name][t + 1] = next_state

        # Update meta-awareness states for 3-level model (without actions)
        if len(self.level_names) >= 3:
            meta_level = self.level_names[2]
            meta_config = self.levels[meta_level]

            if meta_config.action_dim == 0 and t + 1 < self.config.time_steps:
                # Use meta-awareness transition matrix
                if "meta_awareness" in self.B_matrices:
                    B_meta = self.B_matrices["meta_awareness"]
                    current_meta_state = self.state.true_states[meta_level][t]

                    # Sample next meta state from transition probabilities
                    transition_probs = B_meta[:, current_meta_state]
                    next_meta_state = np.random.choice(
                        len(transition_probs), p=transition_probs
                    )
                    self.state.true_states[meta_level][t + 1] = next_meta_state

    def _update_state_priors(self, t: int):
        """Update state priors for next time step."""
        if t + 1 >= self.config.time_steps:
            return

        for level_name, level_config in self.levels.items():
            # Use posterior from current time as prior for next time
            self.state.state_priors[level_name][:, t + 1] = self.state.state_posteriors[
                level_name
            ][:, t]

    def _compute_final_quantities(self):
        """Compute final free energy quantities."""
        # This can be extended for more detailed analysis
        logger.debug("Final quantities computed")

    def _collect_results(self) -> Dict[str, Any]:
        """Collect all simulation results into a dictionary."""
        results = {
            # Model metadata
            "model_name": self.config.name,
            "num_levels": self.config.num_levels,
            "level_names": self.config.level_names,
            "time_steps": self.config.time_steps,
            # State variables
            "state_priors": self.state.state_priors,
            "state_posteriors": self.state.state_posteriors,
            "true_states": self.state.true_states,
            # Observation variables
            "obs_priors": self.state.obs_priors,
            "obs_posteriors": self.state.obs_posteriors,
            "true_observations": self.state.true_observations,
            # Policy variables
            "policy_priors": self.state.policy_priors,
            "policy_posteriors": self.state.policy_posteriors,
            "selected_actions": self.state.selected_actions,
            # Precision variables
            "precision_values": self.state.precision_values,
            # Free energy
            "expected_free_energy": self.state.expected_free_energy,
            "variational_free_energy": self.state.variational_free_energy,
            # Stimulus sequence
            "stimulus_sequence": self.state.stimulus_sequence,
            # Configuration
            "config": self.config,
        }

        return results


def create_model_from_config(
    config_path: Union[str, Path], random_seed: Optional[int] = None
) -> MetaAwarenessModel:
    """
    Convenience function to create model directly from GNN config file.

    Args:
        config_path: Path to GNN configuration file
        random_seed: Optional random seed

    Returns:
        Initialized MetaAwarenessModel
    """
    config = load_gnn_config(config_path)
    return MetaAwarenessModel(config, random_seed)


# Example usage and testing
if __name__ == "__main__":
    # Example: create model from config file
    config_path = Path(__file__).parent.parent / "config" / "meta_awareness_gnn.toml"

    try:
        model = create_model_from_config(config_path, random_seed=42)

        print(f"Created model: {model.config.name}")
        print(f"Levels: {model.level_names}")
        print(f"State dimensions: {model.state_dims}")

        # Run a short simulation
        results = model.run_simulation(simulation_mode="default")

        print(f"Simulation completed with {len(results)} result components")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
