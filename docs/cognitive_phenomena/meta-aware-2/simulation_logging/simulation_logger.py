#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation Logging Module for Meta-Aware-2

Comprehensive logging system for tracking simulation progress, performance,
and results. Provides structured logging with correlation contexts for
the meta-awareness computational phenomenology pipeline.

Part of the meta-aware-2 "golden spike" GNN-specified executable implementation.
"""

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np


@dataclass
class SimulationMetrics:
    """Container for simulation performance metrics."""

    simulation_id: str
    model_name: str
    start_time: datetime
    end_time: Optional[datetime] = None

    # Performance metrics
    total_duration: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_time_seconds: float = 0.0

    # Model metrics
    num_levels: int = 0
    time_steps: int = 0
    simulation_mode: str = ""

    # Quality metrics
    convergence_achieved: bool = False
    numerical_stability: bool = True
    error_count: int = 0
    warning_count: int = 0

    # Custom metrics
    custom_metrics: Dict[str, Any] = None

    def __post_init__(self):
        if self.custom_metrics is None:
            self.custom_metrics = {}

    def finalize(self, end_time: datetime):
        """Finalize metrics with end time."""
        self.end_time = end_time
        if self.start_time:
            self.total_duration = (end_time - self.start_time).total_seconds()


class SimulationLogger:
    """
    Comprehensive simulation logger with structured output and metrics tracking.
    """

    def __init__(
        self,
        log_dir: Union[str, Path] = "./logs",
        simulation_id: Optional[str] = None,
        level: int = logging.INFO,
    ):
        """
        Initialize simulation logger.

        Args:
            log_dir: Directory for log files
            simulation_id: Unique simulation identifier
            level: Logging level
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.simulation_id = simulation_id or self._generate_simulation_id()
        self.level = level

        # Initialize metrics
        self.metrics = SimulationMetrics(
            simulation_id=self.simulation_id, model_name="", start_time=datetime.now()
        )

        # Set up loggers
        self._setup_loggers()

        # Performance tracking
        self._start_time = time.time()
        self._step_times = []

        self.info(f"Simulation logger initialized: {self.simulation_id}")

    def _generate_simulation_id(self) -> str:
        """Generate unique simulation ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"meta_aware_sim_{timestamp}"

    def _setup_loggers(self):
        """Set up file and console loggers."""
        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Main simulation logger
        self.logger = logging.getLogger(f"meta_aware.{self.simulation_id}")
        self.logger.setLevel(self.level)

        # Clear existing handlers
        self.logger.handlers.clear()

        # File handler
        log_file = self.log_dir / f"{self.simulation_id}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(self.level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Performance logger (file only)
        self.perf_logger = logging.getLogger(f"meta_aware.perf.{self.simulation_id}")
        self.perf_logger.setLevel(logging.DEBUG)
        self.perf_logger.handlers.clear()

        perf_file = self.log_dir / f"{self.simulation_id}_performance.log"
        perf_handler = logging.FileHandler(perf_file)
        perf_handler.setLevel(logging.DEBUG)
        perf_handler.setFormatter(formatter)
        self.perf_logger.addHandler(perf_handler)

        # Error logger (file only)
        self.error_logger = logging.getLogger(f"meta_aware.error.{self.simulation_id}")
        self.error_logger.setLevel(logging.ERROR)
        self.error_logger.handlers.clear()

        error_file = self.log_dir / f"{self.simulation_id}_errors.log"
        error_handler = logging.FileHandler(error_file)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        self.error_logger.addHandler(error_handler)

    def set_model_info(
        self, model_name: str, num_levels: int, time_steps: int, simulation_mode: str
    ):
        """Set model information for metrics tracking."""
        self.metrics.model_name = model_name
        self.metrics.num_levels = num_levels
        self.metrics.time_steps = time_steps
        self.metrics.simulation_mode = simulation_mode

        self.info(
            f"Model configured: {model_name} ({num_levels} levels, {time_steps} steps, mode: {simulation_mode})"
        )

    def log_simulation_start(self, config: Dict[str, Any]):
        """Log simulation start with configuration."""
        self.info("=" * 60)
        self.info("SIMULATION STARTED")
        self.info("=" * 60)

        self.info(f"Simulation ID: {self.simulation_id}")
        self.info(f"Start time: {self.metrics.start_time}")

        # Log configuration summary
        self.info("Configuration summary:")
        for key, value in config.items():
            if isinstance(value, (dict, list)):
                self.info(f"  {key}: {type(value).__name__} with {len(value)} items")
            else:
                self.info(f"  {key}: {value}")

    def log_simulation_end(self, results: Dict[str, Any]):
        """Log simulation completion with results summary."""
        end_time = datetime.now()
        self.metrics.finalize(end_time)

        self.info("=" * 60)
        self.info("SIMULATION COMPLETED")
        self.info("=" * 60)

        self.info(f"End time: {end_time}")
        self.info(f"Duration: {self.metrics.total_duration:.2f} seconds")

        # Log results summary
        self.info("Results summary:")
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                self.info(f"  {key}: array {value.shape} ({value.dtype})")
            elif isinstance(value, dict):
                self.info(f"  {key}: dict with {len(value)} items")
            elif isinstance(value, list):
                self.info(f"  {key}: list with {len(value)} items")
            else:
                self.info(f"  {key}: {type(value).__name__}")

        # Save metrics
        self._save_metrics()

    def log_step_start(self, step: int, total_steps: int):
        """Log start of simulation step."""
        if step % 10 == 0 or step < 10:  # Log every 10th step or first 10 steps
            self.debug(f"Step {step}/{total_steps} starting...")

        self._step_start_time = time.time()

    def log_step_end(self, step: int, step_data: Optional[Dict[str, Any]] = None):
        """Log end of simulation step with optional data."""
        step_duration = time.time() - self._step_start_time
        self._step_times.append(step_duration)

        if step % 50 == 0:  # Log every 50th step
            avg_step_time = np.mean(self._step_times[-50:])
            self.info(f"Step {step} completed (avg step time: {avg_step_time:.4f}s)")

        # Log performance data
        if step_data:
            self.perf_logger.debug(f"Step {step}: {json.dumps(step_data, default=str)}")

    def log_level_update(self, level_name: str, step: int, update_data: Dict[str, Any]):
        """Log hierarchical level update."""
        self.debug(f"Level {level_name} update at step {step}")

        # Log key quantities
        for key, value in update_data.items():
            if isinstance(value, np.ndarray) and value.size <= 10:
                self.debug(f"  {key}: {value}")
            elif isinstance(value, (int, float)):
                self.debug(f"  {key}: {value:.6f}")

    def log_policy_selection(
        self,
        level_name: str,
        step: int,
        policy_probs: np.ndarray,
        selected_action: int,
        expected_free_energy: Optional[np.ndarray] = None,
    ):
        """Log policy selection details."""
        self.debug(f"Policy selection for {level_name} at step {step}")
        self.debug(f"  Policy probabilities: {policy_probs}")
        self.debug(f"  Selected action: {selected_action}")

        if expected_free_energy is not None:
            self.debug(f"  Expected free energy: {expected_free_energy}")

    def log_precision_update(
        self,
        level_name: str,
        step: int,
        old_precision: float,
        new_precision: float,
        precision_bounds: tuple,
    ):
        """Log precision parameter updates."""
        self.debug(f"Precision update for {level_name} at step {step}")
        self.debug(f"  Previous: {old_precision:.6f}")
        self.debug(f"  New: {new_precision:.6f}")
        self.debug(f"  Bounds: {precision_bounds}")

        # Check for bound violations
        if new_precision < precision_bounds[0] or new_precision > precision_bounds[1]:
            self.warning(
                f"Precision {new_precision:.6f} outside bounds {precision_bounds}"
            )

    def log_numerical_issue(
        self, issue_type: str, details: Dict[str, Any], step: Optional[int] = None
    ):
        """Log numerical stability issues."""
        self.metrics.numerical_stability = False

        step_info = f" at step {step}" if step is not None else ""
        self.warning(f"Numerical issue ({issue_type}){step_info}: {details}")

        # Log to error file as well
        self.error_logger.error(
            f"Numerical issue: {issue_type}, step: {step}, details: {details}"
        )

    def log_convergence_check(
        self, level_name: str, step: int, converged: bool, criterion: str, value: float
    ):
        """Log convergence checking."""
        status = "CONVERGED" if converged else "NOT CONVERGED"
        self.debug(f"Convergence check for {level_name} at step {step}: {status}")
        self.debug(f"  Criterion: {criterion}, Value: {value:.8f}")

        if converged and not self.metrics.convergence_achieved:
            self.metrics.convergence_achieved = True
            self.info(f"Convergence achieved for {level_name} at step {step}")

    def log_matrix_operation(
        self,
        operation: str,
        matrix_name: str,
        input_shape: tuple,
        output_shape: tuple,
        computation_time: float,
    ):
        """Log matrix operations for performance analysis."""
        self.perf_logger.debug(
            f"Matrix operation: {operation}, matrix: {matrix_name}, "
            f"input: {input_shape}, output: {output_shape}, "
            f"time: {computation_time:.6f}s"
        )

    def log_memory_usage(self, step: int, memory_mb: float):
        """Log memory usage."""
        self.metrics.memory_usage_mb = max(self.metrics.memory_usage_mb, memory_mb)

        if step % 100 == 0:  # Log every 100th step
            self.perf_logger.debug(f"Memory usage at step {step}: {memory_mb:.2f} MB")

    def log_custom_metric(
        self, metric_name: str, value: Any, step: Optional[int] = None
    ):
        """Log custom metric."""
        self.metrics.custom_metrics[metric_name] = value

        step_info = f" at step {step}" if step is not None else ""
        self.debug(f"Custom metric '{metric_name}'{step_info}: {value}")

    def add_tag(self, tag: str):
        """Add tag to simulation for categorization."""
        if "tags" not in self.metrics.custom_metrics:
            self.metrics.custom_metrics["tags"] = []

        if tag not in self.metrics.custom_metrics["tags"]:
            self.metrics.custom_metrics["tags"].append(tag)
            self.info(f"Added tag: {tag}")

    def _save_metrics(self):
        """Save metrics to JSON file."""
        metrics_file = self.log_dir / f"{self.simulation_id}_metrics.json"

        # Convert metrics to dict, handling numpy arrays and datetime objects
        metrics_dict = asdict(self.metrics)

        # Convert datetime objects
        for key, value in metrics_dict.items():
            if isinstance(value, datetime):
                metrics_dict[key] = value.isoformat()

        # Add step timing statistics
        if self._step_times:
            metrics_dict["step_timing"] = {
                "mean_step_time": float(np.mean(self._step_times)),
                "std_step_time": float(np.std(self._step_times)),
                "min_step_time": float(np.min(self._step_times)),
                "max_step_time": float(np.max(self._step_times)),
                "total_steps": len(self._step_times),
            }

        with open(metrics_file, "w") as f:
            json.dump(metrics_dict, f, indent=2, default=str)

        self.info(f"Metrics saved to: {metrics_file}")

    # Convenience methods for different log levels
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)

    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message."""
        self.metrics.warning_count += 1
        self.logger.warning(message)

    def error(self, message: str, exception: Optional[Exception] = None):
        """Log error message."""
        self.metrics.error_count += 1
        self.logger.error(message)

        if exception:
            self.error_logger.error(f"Exception: {message}", exc_info=exception)

    def critical(self, message: str, exception: Optional[Exception] = None):
        """Log critical message."""
        self.metrics.error_count += 1
        self.logger.critical(message)

        if exception:
            self.error_logger.critical(
                f"Critical exception: {message}", exc_info=exception
            )

    def get_metrics(self) -> SimulationMetrics:
        """Get current metrics."""
        return self.metrics

    def get_log_files(self) -> Dict[str, Path]:
        """Get paths to all log files."""
        return {
            "main_log": self.log_dir / f"{self.simulation_id}.log",
            "performance_log": self.log_dir / f"{self.simulation_id}_performance.log",
            "error_log": self.log_dir / f"{self.simulation_id}_errors.log",
            "metrics": self.log_dir / f"{self.simulation_id}_metrics.json",
        }


def create_logger(
    log_dir: Union[str, Path] = "./logs",
    simulation_id: Optional[str] = None,
    level: Union[int, str] = logging.INFO,
) -> SimulationLogger:
    """
    Convenience function to create a simulation logger.

    Args:
        log_dir: Directory for log files
        simulation_id: Unique simulation identifier
        level: Logging level (int or string)

    Returns:
        Configured SimulationLogger instance
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    return SimulationLogger(log_dir, simulation_id, level)


# Example usage and testing
if __name__ == "__main__":
    # Example: create logger and test functionality
    logger = create_logger("./test_logs", level="DEBUG")

    # Test configuration
    config = {
        "model_name": "test_meta_awareness",
        "num_levels": 3,
        "time_steps": 100,
        "simulation_mode": "figure_11",
    }

    logger.set_model_info(
        config["model_name"],
        config["num_levels"],
        config["time_steps"],
        config["simulation_mode"],
    )

    logger.log_simulation_start(config)

    # Simulate some steps
    for step in range(10):
        logger.log_step_start(step, 10)

        # Simulate step processing
        time.sleep(0.01)

        step_data = {
            "precision": np.random.random(),
            "free_energy": np.random.random() * 10,
        }

        logger.log_step_end(step, step_data)

        if step == 5:
            logger.log_convergence_check(
                "attention", step, True, "precision_change", 0.001
            )

    # Test logging various events
    logger.log_policy_selection("attention", 5, np.array([0.3, 0.7]), 1)
    logger.log_precision_update("perception", 3, 0.5, 0.8, (0.1, 2.0))
    logger.log_numerical_issue("underflow", {"location": "softmax", "value": 1e-20}, 7)
    logger.add_tag("test_simulation")
    logger.log_custom_metric("mind_wandering_percentage", 0.35)

    # Finish simulation
    results = {
        "state_posteriors": np.random.random((3, 100)),
        "precision_values": np.random.random((2, 100)),
        "final_free_energy": 45.2,
    }

    logger.log_simulation_end(results)

    print(f"Test completed. Log files: {logger.get_log_files()}")
