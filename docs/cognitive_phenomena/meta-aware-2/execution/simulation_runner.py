#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation Runner for Meta-Aware-2

Main execution module that orchestrates all components of the meta-awareness
simulation pipeline. Provides high-level interface for running simulations
with comprehensive logging, visualization, and analysis.

Part of the meta-aware-2 "golden spike" GNN-specified executable implementation.
"""

import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.gnn_parser import ModelConfig, load_gnn_config
from core.meta_awareness_model import MetaAwarenessModel
from simulation_logging.simulation_logger import create_logger
from utils.math_utils import MathUtils
from visualization.figure_generator import FigureGenerator


class SimulationRunner:
    """
    Main simulation runner that orchestrates the complete meta-awareness pipeline.
    """

    def __init__(
        self,
        config_path: Union[str, Path],
        output_dir: Union[str, Path] = ".",
        log_level: str = "INFO",
        random_seed: Optional[int] = None,
    ):
        """
        Initialize simulation runner.

        Args:
            config_path: Path to GNN configuration file
            output_dir: Base output directory for results (default: current directory)
            log_level: Logging level
            random_seed: Random seed for reproducibility
        """
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.random_seed = random_seed

        # Create output subdirectories (simple structure)
        self.results_dir = self.output_dir / "results"
        self.figures_dir = self.output_dir / "figures"
        self.logs_dir = self.output_dir / "logs"

        for directory in [self.results_dir, self.figures_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Load configuration
        self.config = self._load_and_validate_config()

        # Initialize components
        self.logger = create_logger(self.logs_dir, level=log_level)
        self.figure_generator = FigureGenerator(self.figures_dir)
        self.math_utils = MathUtils()

        # Initialize model (will be created on run)
        self.model = None

        self.logger.info(
            f"SimulationRunner initialized with config: {self.config_path}"
        )
        self.logger.info(f"Output directory: {self.output_dir}")

    def _load_and_validate_config(self) -> ModelConfig:
        """Load and validate GNN configuration."""
        try:
            config = load_gnn_config(self.config_path)
            return config
        except Exception as e:
            print(f"Error loading configuration from {self.config_path}: {e}")
            raise

    def run_complete_analysis(
        self,
        simulation_modes: Optional[List[str]] = None,
        generate_figures: bool = True,
        save_detailed_results: bool = True,
    ) -> Dict[str, Any]:
        """
        Run complete meta-awareness analysis pipeline.

        Args:
            simulation_modes: List of simulation modes to run
            generate_figures: Whether to generate publication figures
            save_detailed_results: Whether to save detailed simulation results

        Returns:
            Dictionary containing all analysis results
        """
        self.logger.info("Starting complete meta-awareness analysis pipeline")

        # Set default simulation modes if not provided
        if simulation_modes is None:
            simulation_modes = list(self.config.simulation_modes.keys())

        # Initialize results container
        complete_results = {
            "config": self.config,
            "simulation_results": {},
            # Public alias expected by tests
            "results": {},
            "analysis_summary": {},
            "figure_paths": {},
            "execution_info": {
                "start_time": time.time(),
                "simulation_modes": simulation_modes,
                "random_seed": self.random_seed,
            },
        }

        try:
            # Run simulations for each mode
            for mode in simulation_modes:
                self.logger.info(f"Running simulation mode: {mode}")

                # Run single simulation
                results = self.run_simulation(mode)
                complete_results["simulation_results"][mode] = results

                # Generate figures if requested
                if generate_figures:
                    figure_paths = self._generate_mode_figures(results, mode)
                    complete_results["figure_paths"][mode] = figure_paths

                # Save detailed results if requested
                if save_detailed_results:
                    self._save_simulation_results(results, mode)

            # Generate comparative analysis
            if len(simulation_modes) > 1:
                comparative_analysis = self._run_comparative_analysis(
                    complete_results["simulation_results"]
                )
                complete_results["analysis_summary"]["comparative"] = (
                    comparative_analysis
                )

                if generate_figures:
                    comparison_figures = self._generate_comparison_figures(
                        complete_results["simulation_results"]
                    )
                    complete_results["figure_paths"]["comparisons"] = comparison_figures

            # Finalize execution info
            complete_results["execution_info"]["end_time"] = time.time()
            complete_results["execution_info"]["total_duration"] = (
                complete_results["execution_info"]["end_time"]
                - complete_results["execution_info"]["start_time"]
            )

            self.logger.info("Complete analysis pipeline finished successfully")
            self.logger.info(
                f"Total duration: {complete_results['execution_info']['total_duration']:.2f} seconds"
            )
            # Keep alias in sync at the end
            complete_results["results"] = complete_results["simulation_results"]

            return complete_results

        except Exception as e:
            self.logger.error(f"Error in complete analysis pipeline: {e}", exception=e)
            raise

    def run_simulation(self, simulation_mode: str = "default") -> Dict[str, Any]:
        """
        Run a single meta-awareness simulation.

        Args:
            simulation_mode: Simulation mode to run

        Returns:
            Dictionary containing simulation results
        """
        self.logger.info(f"Starting simulation in mode: {simulation_mode}")

        # Set model info for logging
        self.logger.set_model_info(
            self.config.name,
            self.config.num_levels,
            self.config.time_steps,
            simulation_mode,
        )

        try:
            # Initialize model with logger for performance tracking
            self.model = MetaAwarenessModel(self.config, self.random_seed, self.logger)

            # Log simulation start
            config_summary = {
                "model_name": self.config.name,
                "num_levels": self.config.num_levels,
                "time_steps": self.config.time_steps,
                "simulation_mode": simulation_mode,
                "level_names": self.config.level_names,
                "state_dimensions": {
                    name: level.state_dim for name, level in self.config.levels.items()
                },
                "random_seed": self.random_seed,
            }
            self.logger.log_simulation_start(config_summary)

            # Run simulation
            results = self.model.run_simulation(simulation_mode)

            # Add execution metadata
            results["execution_metadata"] = {
                "simulation_mode": simulation_mode,
                "random_seed": self.random_seed,
                "config_path": str(self.config_path),
            }

            # Compute additional analysis
            analysis_results = self._compute_additional_analysis(results)
            # Expose under a top-level 'analysis' key for consumers/tests
            results["analysis"] = analysis_results
            # Also merge flat keys for direct result consumers.
            results.update(analysis_results)

            # Log simulation completion
            self.logger.log_simulation_end(results)

            return results

        except Exception as e:
            self.logger.error(f"Error during simulation: {e}", exception=e)
            raise

    def _compute_additional_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute additional analysis metrics."""
        analysis = {}

        try:
            # Mind-wandering analysis
            analysis["mind_wandering_analysis"] = self._analyze_mind_wandering(results)

            # Precision analysis
            analysis["precision_analysis"] = self._analyze_precision_dynamics(results)

            # Free energy analysis
            analysis["free_energy_analysis"] = self._analyze_free_energy(results)

            # Behavioral patterns
            analysis["behavioral_patterns"] = self._analyze_behavioral_patterns(results)

            self.logger.info("Additional analysis completed")

        except Exception as e:
            self.logger.warning(f"Error in additional analysis: {e}")
            analysis["analysis_error"] = str(e)

        return analysis

    def _analyze_mind_wandering(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze mind-wandering patterns."""
        attention_level = self._get_attention_level_name(results)
        true_states = results.get("true_states", {})

        if attention_level not in true_states:
            return {"error": "No attention level data available"}

        att_states = true_states[attention_level]

        # Calculate basic statistics
        focused_percentage = np.mean(att_states == 0) * 100
        distracted_percentage = np.mean(att_states == 1) * 100

        # Calculate transitions
        transitions = np.sum(np.diff(att_states) != 0)

        # Calculate episode lengths
        episodes = self._identify_episodes(att_states)
        focused_episodes = [ep for ep in episodes if ep["state"] == 0]
        distracted_episodes = [ep for ep in episodes if ep["state"] == 1]

        analysis = {
            "focused_percentage": focused_percentage,
            "distracted_percentage": distracted_percentage,
            "num_transitions": transitions,
            "num_focused_episodes": len(focused_episodes),
            "num_distracted_episodes": len(distracted_episodes),
            "avg_focused_episode_length": np.mean(
                [ep["length"] for ep in focused_episodes]
            )
            if focused_episodes
            else 0,
            "avg_distracted_episode_length": np.mean(
                [ep["length"] for ep in distracted_episodes]
            )
            if distracted_episodes
            else 0,
            "max_focused_episode_length": max([ep["length"] for ep in focused_episodes])
            if focused_episodes
            else 0,
            "max_distracted_episode_length": max(
                [ep["length"] for ep in distracted_episodes]
            )
            if distracted_episodes
            else 0,
        }

        self.logger.log_custom_metric(
            "mind_wandering_percentage", distracted_percentage
        )
        self.logger.log_custom_metric("attention_transitions", transitions)

        return analysis

    def _analyze_precision_dynamics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze precision parameter dynamics."""
        precision_values = results.get("precision_values", {})

        analysis = {}

        for level_name, precision_ts in precision_values.items():
            level_analysis = {
                "mean": float(np.mean(precision_ts)),
                "std": float(np.std(precision_ts)),
                "min": float(np.min(precision_ts)),
                "max": float(np.max(precision_ts)),
                "range": float(np.max(precision_ts) - np.min(precision_ts)),
                "variability_coefficient": float(
                    np.std(precision_ts) / np.mean(precision_ts)
                )
                if np.mean(precision_ts) > 0
                else 0,
            }

            # Calculate precision change rate
            precision_changes = np.abs(np.diff(precision_ts))
            level_analysis["mean_change_rate"] = float(np.mean(precision_changes))
            level_analysis["max_change_rate"] = float(np.max(precision_changes))

            analysis[level_name] = level_analysis

            self.logger.log_custom_metric(
                f"{level_name}_precision_mean", level_analysis["mean"]
            )
            self.logger.log_custom_metric(
                f"{level_name}_precision_variability",
                level_analysis["variability_coefficient"],
            )

        return analysis

    def _analyze_free_energy(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze free energy dynamics."""
        expected_G = results.get("expected_free_energy", {})
        variational_F = results.get("variational_free_energy", {})

        analysis = {"expected_free_energy": {}, "variational_free_energy": {}}

        # Analyze expected free energy
        for level_name, G_values in expected_G.items():
            if G_values.size > 0:
                G_flat = G_values.flatten()
                analysis["expected_free_energy"][level_name] = {
                    "mean": float(np.mean(G_flat)),
                    "std": float(np.std(G_flat)),
                    "min": float(np.min(G_flat)),
                    "max": float(np.max(G_flat)),
                }

        # Analyze variational free energy
        for level_name, F_values in variational_F.items():
            if F_values.size > 0:
                F_flat = F_values.flatten()
                analysis["variational_free_energy"][level_name] = {
                    "mean": float(np.mean(F_flat)),
                    "std": float(np.std(F_flat)),
                    "min": float(np.min(F_flat)),
                    "max": float(np.max(F_flat)),
                }

        return analysis

    def _analyze_behavioral_patterns(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze behavioral response patterns."""
        stimulus_sequence = results.get("stimulus_sequence", np.array([]))

        # Find oddball stimuli
        oddball_times = np.where(stimulus_sequence == 1)[0]

        analysis = {
            "stimulus_analysis": {
                "total_stimuli": len(stimulus_sequence),
                "num_oddballs": len(oddball_times),
                "oddball_percentage": len(oddball_times) / len(stimulus_sequence) * 100
                if len(stimulus_sequence) > 0
                else 0,
            }
        }

        if len(oddball_times) > 1:
            intervals = np.diff(oddball_times)
            analysis["stimulus_analysis"].update(
                {
                    "mean_oddball_interval": float(np.mean(intervals)),
                    "std_oddball_interval": float(np.std(intervals)),
                    "min_oddball_interval": float(np.min(intervals)),
                    "max_oddball_interval": float(np.max(intervals)),
                }
            )

        return analysis

    def _get_perception_level_name(self, results: Dict[str, Any]) -> str:
        """Infer the perception level name from results or configuration.
        Preference order:
        - If results contain precision_values for a config level, use that level
        - Otherwise, select the first level in config with action_dim == 0
        - Fallback to the first configured level name or 'perception'
        """
        precision_values = results.get("precision_values", {})
        if isinstance(precision_values, dict):
            for level in self.config.levels.keys():
                if level in precision_values:
                    return level
        for level_name, lvl in self.config.levels.items():
            if getattr(lvl, "action_dim", 0) == 0:
                return level_name
        names = list(self.config.levels.keys())
        return names[0] if names else "perception"

    def _generate_mode_figures(
        self, results: Dict[str, Any], mode: str
    ) -> Dict[str, Path]:
        """Generate figures for a specific simulation mode."""
        figure_paths = {}

        try:
            # Determine figure type based on mode and model structure
            if "figure_7" in mode or "fixed" in mode:
                fig_path = self.figure_generator.generate_figure_7(
                    results, f"{mode}_figure_7"
                )
                figure_paths["main_figure"] = fig_path

            elif "figure_10" in mode or ("2" in mode and "level" in mode):
                fig_path = self.figure_generator.generate_figure_10(
                    results, f"{mode}_figure_10"
                )
                figure_paths["main_figure"] = fig_path

            elif "figure_11" in mode or ("3" in mode and "level" in mode):
                fig_path = self.figure_generator.generate_figure_11(
                    results, f"{mode}_figure_11"
                )
                figure_paths["main_figure"] = fig_path

            else:
                # Generate appropriate figure based on number of levels
                if results.get("num_levels", 2) >= 3:
                    fig_path = self.figure_generator.generate_figure_11(
                        results, f"{mode}_three_level"
                    )
                    figure_paths["main_figure"] = fig_path
                else:
                    fig_path = self.figure_generator.generate_figure_10(
                        results, f"{mode}_two_level"
                    )
                    figure_paths["main_figure"] = fig_path

            self.logger.info(
                f"Generated main figure for mode {mode}: {figure_paths.get('main_figure')}"
            )

        except Exception as e:
            self.logger.warning(f"Error generating figures for mode {mode}: {e}")

        return figure_paths

    def _generate_comparison_figures(
        self, all_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Path]:
        """Generate comparison figures across simulation modes."""
        comparison_paths = {}

        try:
            # Find 2-level and 3-level results for comparison
            results_2level = None
            results_3level = None

            for mode, results in all_results.items():
                if results.get("num_levels", 2) == 2:
                    results_2level = results
                elif results.get("num_levels", 2) >= 3:
                    results_3level = results

            # Generate model comparison if both are available
            if results_2level is not None and results_3level is not None:
                comparison_path = self.figure_generator.generate_model_comparison(
                    results_2level, results_3level, "model_comparison"
                )
                comparison_paths["model_comparison"] = comparison_path
                self.logger.info(
                    f"Generated model comparison figure: {comparison_path}"
                )

        except Exception as e:
            self.logger.warning(f"Error generating comparison figures: {e}")

        return comparison_paths

    def _run_comparative_analysis(
        self, all_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run comparative analysis across simulation modes."""
        comparative_analysis = {}

        try:
            # Compare mind-wandering patterns
            mw_comparison = {}
            for mode, results in all_results.items():
                mw_analysis = results.get("mind_wandering_analysis", {})
                if mw_analysis:
                    mw_comparison[mode] = {
                        "distracted_percentage": mw_analysis.get(
                            "distracted_percentage", 0
                        ),
                        "num_transitions": mw_analysis.get("num_transitions", 0),
                        "avg_episode_length": mw_analysis.get(
                            "avg_distracted_episode_length", 0
                        ),
                    }

            comparative_analysis["mind_wandering_comparison"] = mw_comparison

            # Compare precision dynamics
            precision_comparison = {}
            for mode, results in all_results.items():
                precision_analysis = results.get("precision_analysis", {})
                if precision_analysis:
                    # Get perception level analysis
                    perception_level = self._get_perception_level_name(results)
                    if perception_level in precision_analysis:
                        precision_comparison[mode] = precision_analysis[
                            perception_level
                        ]

            comparative_analysis["precision_comparison"] = precision_comparison

            self.logger.info("Comparative analysis completed")

        except Exception as e:
            self.logger.warning(f"Error in comparative analysis: {e}")
            comparative_analysis["analysis_error"] = str(e)

        return comparative_analysis

    def _save_simulation_results(self, results: Dict[str, Any], mode: str):
        """Save detailed simulation results to files."""
        try:
            import json
            import pickle

            # Save as JSON (excluding numpy arrays)
            json_results = self._prepare_results_for_json(results)
            json_path = self.results_dir / f"{mode}_results.json"
            with open(json_path, "w") as f:
                json.dump(json_results, f, indent=2, default=str)

            # Save full results as pickle (including numpy arrays)
            pickle_path = self.results_dir / f"{mode}_results.pkl"
            with open(pickle_path, "wb") as f:
                pickle.dump(results, f)

            self.logger.info(
                f"Saved results for mode {mode}: {json_path}, {pickle_path}"
            )

        except Exception as e:
            self.logger.warning(f"Error saving results for mode {mode}: {e}")

    def _prepare_results_for_json(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare results for JSON serialization by converting numpy arrays."""
        json_results = {}

        for key, value in results.items():
            if isinstance(value, np.ndarray):
                if value.size <= 1000:  # Only include small arrays
                    json_results[key] = value.tolist()
                else:
                    json_results[key] = {
                        "type": "numpy_array",
                        "shape": value.shape,
                        "dtype": str(value.dtype),
                        "note": "Array too large for JSON, see pickle file",
                    }
            elif isinstance(value, dict):
                json_results[key] = self._prepare_results_for_json(value)
            else:
                json_results[key] = value

        return json_results

    # Helper methods
    def _get_attention_level_name(self, results: Dict[str, Any]) -> str:
        """Get the name of the attention level."""
        level_names = results.get("level_names", [])
        if len(level_names) > 1:
            return level_names[1]
        return "attention"

    def _identify_episodes(self, states: np.ndarray) -> List[Dict[str, Any]]:
        """Identify episodes of continuous states."""
        episodes = []
        current_state = states[0]
        start_time = 0

        for t, state in enumerate(states[1:], 1):
            if state != current_state:
                # End of current episode
                episodes.append(
                    {
                        "state": current_state,
                        "start": start_time,
                        "end": t - 1,
                        "length": t - start_time,
                    }
                )
                current_state = state
                start_time = t

        # Add final episode
        episodes.append(
            {
                "state": current_state,
                "start": start_time,
                "end": len(states) - 1,
                "length": len(states) - start_time,
            }
        )

        return episodes

    def get_results_summary(self) -> Dict[str, Any]:
        """Get summary of all generated results."""
        summary = {
            "output_directories": {
                "base": str(self.output_dir),
                "results": str(self.results_dir),
                "figures": str(self.figures_dir),
                "logs": str(self.logs_dir),
            },
            "config_path": str(self.config_path),
            "model_info": {
                "name": self.config.name,
                "num_levels": self.config.num_levels,
                "time_steps": self.config.time_steps,
                "level_names": self.config.level_names,
            },
        }

        if hasattr(self, "logger"):
            summary["log_files"] = {
                str(k): str(v) for k, v in self.logger.get_log_files().items()
            }

        return summary


def run_simulation_from_config(
    config_path: Union[str, Path],
    output_dir: Union[str, Path] = ".",
    simulation_modes: Optional[List[str]] = None,
    random_seed: Optional[int] = None,
    log_level: str = "INFO",
) -> Dict[str, Any]:
    """
    Convenience function to run complete simulation from config file.

    Args:
        config_path: Path to GNN configuration file
        output_dir: Output directory for results
        simulation_modes: List of simulation modes to run
        random_seed: Random seed for reproducibility
        log_level: Logging level

    Returns:
        Complete analysis results
    """
    runner = SimulationRunner(
        config_path=config_path,
        output_dir=output_dir,
        log_level=log_level,
        random_seed=random_seed,
    )

    return runner.run_complete_analysis(
        simulation_modes=simulation_modes,
        generate_figures=True,
        save_detailed_results=True,
    )


# Example usage and testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run meta-awareness simulation")
    parser.add_argument("config", help="Path to GNN configuration file")
    parser.add_argument("--output", "-o", default="./output", help="Output directory")
    parser.add_argument("--modes", "-m", nargs="+", help="Simulation modes to run")
    parser.add_argument("--seed", "-s", type=int, help="Random seed")
    parser.add_argument("--log-level", "-l", default="INFO", help="Logging level")

    args = parser.parse_args()

    try:
        results = run_simulation_from_config(
            config_path=args.config,
            output_dir=args.output,
            simulation_modes=args.modes,
            random_seed=args.seed,
            log_level=args.log_level,
        )

        print("Simulation completed successfully!")
        print(f"Results saved to: {args.output}")
        print(
            f"Total duration: {results['execution_info']['total_duration']:.2f} seconds"
        )

    except Exception as e:
        print(f"Error running simulation: {e}")
        traceback.print_exc()
        sys.exit(1)
