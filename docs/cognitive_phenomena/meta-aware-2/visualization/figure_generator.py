#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure Generation Module for Meta-Aware-2

Comprehensive visualization system for generating publication-quality figures
from meta-awareness simulation results. Supports reproduction of all figures
from Sandved-Smith et al. (2021) and custom analysis visualizations.

Part of the meta-aware-2 "golden spike" GNN-specified executable implementation.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Set style for publication-quality figures
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


class FigureGenerator:
    """
    Publication-quality figure generator for meta-awareness simulations.
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "./figures",
        dpi: int = 300,
        figure_format: str = "png",
    ):
        """
        Initialize figure generator.

        Args:
            output_dir: Directory for saving figures
            dpi: Figure resolution
            figure_format: Output format (png, pdf, svg, etc.)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.dpi = dpi
        self.figure_format = figure_format

        # Set up color schemes
        self._setup_color_schemes()

        # Figure size presets
        self.figure_sizes = {
            "single_column": (3.5, 2.5),
            "double_column": (7.0, 5.0),
            "square": (4.0, 4.0),
            "wide": (8.0, 3.0),
            "tall": (4.0, 6.0),
        }

    def _setup_color_schemes(self):
        """Set up publication-quality color schemes."""
        # Color scheme for different model levels
        self.level_colors = {
            "perception": "#1f77b4",  # Blue
            "attention": "#ff7f0e",  # Orange
            "meta_awareness": "#2ca02c",  # Green
            "level_1": "#1f77b4",
            "level_2": "#ff7f0e",
            "level_3": "#2ca02c",
        }

        # Color scheme for states
        self.state_colors = {
            "focused": "#d62728",  # Red
            "distracted": "#9467bd",  # Purple
            "meta_on": "#2ca02c",  # Green
            "meta_off": "#8c564b",  # Brown
        }

        # Precision colormap
        self.precision_cmap = LinearSegmentedColormap.from_list(
            "precision", ["#440154", "#31688e", "#35b779", "#fde725"]
        )

        # Free energy colormap
        self.free_energy_cmap = LinearSegmentedColormap.from_list(
            "free_energy",
            ["#0d0887", "#6a00a8", "#b12a90", "#e16462", "#fca636", "#f0f921"],
        )

    def generate_figure_7(
        self, results: Dict[str, Any], save_name: str = "figure_7_fixed_attention"
    ) -> Path:
        """
        Generate Figure 7: Fixed attention schedule (Sandved-Smith et al. 2021).

        Args:
            results: Simulation results dictionary
            save_name: Output filename (without extension)

        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(
            3,
            1,
            figsize=self.figure_sizes["double_column"],
            sharex=True,
            height_ratios=[1, 1, 1],
        )

        time_steps = results["time_steps"]
        time_range = np.arange(time_steps)

        # Extract data
        stimulus_sequence = results["stimulus_sequence"]
        true_states = results["true_states"]
        precision_values = results["precision_values"]

        # Panel A: Stimulus sequence
        ax = axes[0]
        ax.eventplot(
            [np.where(stimulus_sequence == 1)[0]],
            lineoffsets=0.5,
            linelengths=0.8,
            linewidths=2,
            colors=["black"],
            label="Oddball stimulus",
        )
        ax.set_ylim(0, 1)
        ax.set_ylabel("Stimulus")
        ax.set_title("A. Stimulus Sequence", fontweight="bold", loc="left")
        ax.grid(True, alpha=0.3)

        # Panel B: Attentional states
        ax = axes[1]
        attention_level = self._get_attention_level_name(results)
        if attention_level in true_states:
            att_states = true_states[attention_level]

            # Create color-coded time series
            for t in range(time_steps):
                color = (
                    self.state_colors["focused"]
                    if att_states[t] == 0
                    else self.state_colors["distracted"]
                )
                ax.bar(t, 1, width=1, color=color, alpha=0.7, edgecolor="none")

        ax.set_ylim(0, 1)
        ax.set_ylabel("Attentional State")
        ax.set_title("B. True Attentional States", fontweight="bold", loc="left")

        # Create legend
        focused_patch = patches.Patch(
            color=self.state_colors["focused"], label="Focused"
        )
        distracted_patch = patches.Patch(
            color=self.state_colors["distracted"], label="Distracted"
        )
        ax.legend(handles=[focused_patch, distracted_patch], loc="upper right")

        # Panel C: Precision dynamics
        ax = axes[2]
        perception_level = self._get_perception_level_name(results)
        if perception_level in precision_values:
            precision_ts = precision_values[perception_level]
            ax.plot(
                time_range,
                precision_ts,
                linewidth=2,
                color=self.level_colors["perception"],
                label="Precision",
            )

            # Highlight oddball times
            oddball_times = np.where(stimulus_sequence == 1)[0]
            for oddball_t in oddball_times:
                ax.axvline(oddball_t, color="red", linestyle="--", alpha=0.5)

        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Precision (γ)")
        ax.set_title("C. Precision Dynamics", fontweight="bold", loc="left")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()

        # Save figure
        output_path = self.output_dir / f"{save_name}.{self.figure_format}"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        return output_path

    def generate_figure_10(
        self, results: Dict[str, Any], save_name: str = "figure_10_two_level"
    ) -> Path:
        """
        Generate Figure 10: Two-level model with mind-wandering.

        Args:
            results: Simulation results dictionary
            save_name: Output filename (without extension)

        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(
            4,
            1,
            figsize=self.figure_sizes["double_column"],
            sharex=True,
            height_ratios=[0.5, 1, 1, 1],
        )

        time_steps = results["time_steps"]
        time_range = np.arange(time_steps)

        # Extract data
        stimulus_sequence = results["stimulus_sequence"]
        state_posteriors = results["state_posteriors"]
        true_states = results["true_states"]
        precision_values = results["precision_values"]

        # Panel A: Stimulus sequence
        ax = axes[0]
        ax.eventplot(
            [np.where(stimulus_sequence == 1)[0]],
            lineoffsets=0.5,
            linelengths=0.8,
            linewidths=2,
            colors=["black"],
        )
        ax.set_ylim(0, 1)
        ax.set_ylabel("Stimulus")
        ax.set_title("A. Stimulus Sequence", fontweight="bold", loc="left")
        ax.grid(True, alpha=0.3)

        # Panel B: Perceptual beliefs
        ax = axes[1]
        perception_level = self._get_perception_level_name(results)
        if perception_level in state_posteriors:
            beliefs = state_posteriors[perception_level]

            # Plot belief dynamics
            for state_idx in range(beliefs.shape[0]):
                ax.plot(
                    time_range,
                    beliefs[state_idx, :],
                    linewidth=1.5,
                    label=f"State {state_idx}",
                )

        ax.set_ylabel("Belief Strength")
        ax.set_title("B. Perceptual Beliefs", fontweight="bold", loc="left")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Panel C: Attentional states
        ax = axes[2]
        attention_level = self._get_attention_level_name(results)
        if attention_level in true_states:
            att_states = true_states[attention_level]

            # Create color-coded time series
            for t in range(time_steps):
                color = (
                    self.state_colors["focused"]
                    if att_states[t] == 0
                    else self.state_colors["distracted"]
                )
                ax.bar(t, 1, width=1, color=color, alpha=0.7, edgecolor="none")

        ax.set_ylim(0, 1)
        ax.set_ylabel("Attentional State")
        ax.set_title("C. Attentional States", fontweight="bold", loc="left")

        # Panel D: Precision dynamics
        ax = axes[3]
        if perception_level in precision_values:
            precision_ts = precision_values[perception_level]
            ax.plot(
                time_range,
                precision_ts,
                linewidth=2,
                color=self.level_colors["perception"],
            )

            # Highlight mind-wandering episodes
            if attention_level in true_states:
                wandering_periods = self._identify_mind_wandering_periods(
                    true_states[attention_level]
                )
                for start, end in wandering_periods:
                    ax.axvspan(
                        start, end, alpha=0.2, color=self.state_colors["distracted"]
                    )

        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Precision (γ)")
        ax.set_title("D. Precision Dynamics", fontweight="bold", loc="left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        output_path = self.output_dir / f"{save_name}.{self.figure_format}"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        return output_path

    def generate_figure_11(
        self, results: Dict[str, Any], save_name: str = "figure_11_three_level"
    ) -> Path:
        """
        Generate Figure 11: Three-level model with meta-awareness.

        Args:
            results: Simulation results dictionary
            save_name: Output filename (without extension)

        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(
            5,
            1,
            figsize=self.figure_sizes["tall"],
            sharex=True,
            height_ratios=[0.5, 1, 1, 1, 1],
        )

        time_steps = results["time_steps"]
        time_range = np.arange(time_steps)

        # Extract data
        stimulus_sequence = results["stimulus_sequence"]
        state_posteriors = results["state_posteriors"]
        true_states = results["true_states"]
        precision_values = results["precision_values"]

        # Panel A: Stimulus sequence
        ax = axes[0]
        ax.eventplot(
            [np.where(stimulus_sequence == 1)[0]],
            lineoffsets=0.5,
            linelengths=0.8,
            linewidths=2,
            colors=["black"],
        )
        ax.set_ylim(0, 1)
        ax.set_ylabel("Stimulus")
        ax.set_title("A. Stimulus Sequence", fontweight="bold", loc="left")
        ax.grid(True, alpha=0.3)

        # Panel B: Perceptual beliefs
        ax = axes[1]
        perception_level = self._get_perception_level_name(results)
        if perception_level in state_posteriors:
            beliefs = state_posteriors[perception_level]

            for state_idx in range(beliefs.shape[0]):
                ax.plot(
                    time_range,
                    beliefs[state_idx, :],
                    linewidth=1.5,
                    label=f"State {state_idx}",
                )

        ax.set_ylabel("Belief Strength")
        ax.set_title("B. Perceptual Beliefs", fontweight="bold", loc="left")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Panel C: Attentional states
        ax = axes[2]
        attention_level = self._get_attention_level_name(results)
        if attention_level in true_states:
            att_states = true_states[attention_level]

            for t in range(time_steps):
                color = (
                    self.state_colors["focused"]
                    if att_states[t] == 0
                    else self.state_colors["distracted"]
                )
                ax.bar(t, 1, width=1, color=color, alpha=0.7, edgecolor="none")

        ax.set_ylim(0, 1)
        ax.set_ylabel("Attentional State")
        ax.set_title("C. Attentional States", fontweight="bold", loc="left")

        # Panel D: Meta-awareness states
        ax = axes[3]
        if len(results["level_names"]) >= 3:
            meta_level = results["level_names"][2]
            if meta_level in true_states:
                meta_states = true_states[meta_level]

                for t in range(time_steps):
                    color = (
                        self.state_colors["meta_on"]
                        if meta_states[t] == 0
                        else self.state_colors["meta_off"]
                    )
                    ax.bar(t, 1, width=1, color=color, alpha=0.7, edgecolor="none")

        ax.set_ylim(0, 1)
        ax.set_ylabel("Meta-awareness")
        ax.set_title("D. Meta-awareness States", fontweight="bold", loc="left")

        # Create legend for meta-awareness
        meta_on_patch = patches.Patch(
            color=self.state_colors["meta_on"], label="Meta-aware"
        )
        meta_off_patch = patches.Patch(
            color=self.state_colors["meta_off"], label="Meta-unaware"
        )
        ax.legend(handles=[meta_on_patch, meta_off_patch], loc="upper right")

        # Panel E: Precision dynamics
        ax = axes[4]
        if perception_level in precision_values:
            precision_ts = precision_values[perception_level]
            ax.plot(
                time_range,
                precision_ts,
                linewidth=2,
                color=self.level_colors["perception"],
            )

            # Highlight meta-awareness control periods
            if len(results["level_names"]) >= 3:
                meta_level = results["level_names"][2]
                if meta_level in true_states:
                    meta_control_periods = self._identify_meta_control_periods(
                        true_states[meta_level]
                    )
                    for start, end in meta_control_periods:
                        ax.axvspan(
                            start, end, alpha=0.2, color=self.state_colors["meta_on"]
                        )

        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Precision (γ)")
        ax.set_title("E. Precision Dynamics", fontweight="bold", loc="left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        output_path = self.output_dir / f"{save_name}.{self.figure_format}"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        return output_path

    def generate_precision_analysis(
        self, results: Dict[str, Any], save_name: str = "precision_analysis"
    ) -> Path:
        """
        Generate comprehensive precision analysis figure.

        Args:
            results: Simulation results dictionary
            save_name: Output filename (without extension)

        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figure_sizes["double_column"])

        precision_values = results["precision_values"]
        time_steps = results["time_steps"]
        time_range = np.arange(time_steps)

        # Panel A: Precision time series
        ax = axes[0, 0]
        for level_name, precision_ts in precision_values.items():
            color = self.level_colors.get(level_name, "gray")
            ax.plot(
                time_range,
                precision_ts,
                linewidth=2,
                color=color,
                label=level_name.title(),
            )

        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Precision (γ)")
        ax.set_title("A. Precision Time Series", fontweight="bold", loc="left")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Panel B: Precision distributions
        ax = axes[0, 1]
        for level_name, precision_ts in precision_values.items():
            color = self.level_colors.get(level_name, "gray")
            ax.hist(
                precision_ts,
                bins=30,
                alpha=0.6,
                color=color,
                label=level_name.title(),
                density=True,
            )

        ax.set_xlabel("Precision (γ)")
        ax.set_ylabel("Density")
        ax.set_title("B. Precision Distributions", fontweight="bold", loc="left")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Panel C: Precision variability
        ax = axes[1, 0]
        level_names = list(precision_values.keys())
        variabilities = [np.std(precision_values[name]) for name in level_names]
        colors = [self.level_colors.get(name, "gray") for name in level_names]

        ax.bar(range(len(level_names)), variabilities, color=colors, alpha=0.7)
        ax.set_xticks(range(len(level_names)))
        ax.set_xticklabels([name.title() for name in level_names], rotation=45)
        ax.set_ylabel("Standard Deviation")
        ax.set_title("C. Precision Variability", fontweight="bold", loc="left")
        ax.grid(True, alpha=0.3)

        # Panel D: Precision correlation matrix (if multiple levels)
        ax = axes[1, 1]
        if len(precision_values) > 1:
            precision_matrix = np.array(
                [precision_values[name] for name in level_names]
            )
            correlation_matrix = np.corrcoef(precision_matrix)

            im = ax.imshow(correlation_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(len(level_names)))
            ax.set_yticks(range(len(level_names)))
            ax.set_xticklabels([name.title() for name in level_names], rotation=45)
            ax.set_yticklabels([name.title() for name in level_names])

            # Add correlation values
            for i in range(len(level_names)):
                for j in range(len(level_names)):
                    ax.text(
                        j,
                        i,
                        f"{correlation_matrix[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="black",
                    )

            plt.colorbar(im, ax=ax, shrink=0.8)
        else:
            ax.text(
                0.5,
                0.5,
                "Requires multiple\nlevels for correlation",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        ax.set_title("D. Level Correlations", fontweight="bold", loc="left")

        plt.tight_layout()

        # Save figure
        output_path = self.output_dir / f"{save_name}.{self.figure_format}"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        return output_path

    def generate_free_energy_analysis(
        self, results: Dict[str, Any], save_name: str = "free_energy_analysis"
    ) -> Path:
        """
        Generate free energy analysis figure.

        Args:
            results: Simulation results dictionary
            save_name: Output filename (without extension)

        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figure_sizes["double_column"])

        expected_free_energy = results.get("expected_free_energy", {})
        variational_free_energy = results.get("variational_free_energy", {})
        time_steps = results["time_steps"]
        time_range = np.arange(time_steps)

        # Panel A: Expected free energy over time
        ax = axes[0, 0]
        for level_name, G_values in expected_free_energy.items():
            if G_values.ndim > 1:  # Multiple policies
                for policy_idx in range(G_values.shape[0]):
                    color = self.level_colors.get(level_name, "gray")
                    alpha = 0.7 if policy_idx == 0 else 0.4
                    ax.plot(
                        time_range,
                        G_values[policy_idx, :],
                        linewidth=2,
                        color=color,
                        alpha=alpha,
                        label=f"{level_name.title()} Policy {policy_idx}",
                    )
            else:
                color = self.level_colors.get(level_name, "gray")
                ax.plot(
                    time_range,
                    G_values,
                    linewidth=2,
                    color=color,
                    label=level_name.title(),
                )

        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Expected Free Energy")
        ax.set_title("A. Expected Free Energy", fontweight="bold", loc="left")
        ax.grid(True, alpha=0.3)
        if ax.get_legend_handles_labels()[0]:  # Check if there are any legend entries
            ax.legend()

        # Panel B: Variational free energy over time
        ax = axes[0, 1]
        for level_name, F_values in variational_free_energy.items():
            color = self.level_colors.get(level_name, "gray")
            if F_values.ndim > 1:
                # Average over policies if multiple
                F_avg = np.mean(F_values, axis=0)
                ax.plot(
                    time_range,
                    F_avg,
                    linewidth=2,
                    color=color,
                    label=level_name.title(),
                )
            else:
                ax.plot(
                    time_range,
                    F_values,
                    linewidth=2,
                    color=color,
                    label=level_name.title(),
                )

        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Variational Free Energy")
        ax.set_title("B. Variational Free Energy", fontweight="bold", loc="left")
        ax.grid(True, alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()

        # Panel C: Free energy distributions
        ax = axes[1, 0]
        all_G_values = []
        all_F_values = []

        for level_name in expected_free_energy.keys():
            G_vals = expected_free_energy[level_name]
            if G_vals.ndim > 1:
                all_G_values.extend(G_vals.flatten())
            else:
                all_G_values.extend(G_vals)

        for level_name in variational_free_energy.keys():
            F_vals = variational_free_energy[level_name]
            if F_vals.ndim > 1:
                all_F_values.extend(F_vals.flatten())
            else:
                all_F_values.extend(F_vals)

        if all_G_values:
            ax.hist(
                all_G_values,
                bins=30,
                alpha=0.6,
                color="blue",
                label="Expected G",
                density=True,
            )
        if all_F_values:
            ax.hist(
                all_F_values,
                bins=30,
                alpha=0.6,
                color="red",
                label="Variational F",
                density=True,
            )

        ax.set_xlabel("Free Energy")
        ax.set_ylabel("Density")
        ax.set_title("C. Free Energy Distributions", fontweight="bold", loc="left")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Panel D: Free energy summary statistics
        ax = axes[1, 1]
        ax.axis("off")

        # Create summary text
        summary_text = "Free Energy Summary:\n\n"

        if all_G_values:
            summary_text += "Expected Free Energy (G):\n"
            summary_text += f"  Mean: {np.mean(all_G_values):.3f}\n"
            summary_text += f"  Std:  {np.std(all_G_values):.3f}\n"
            summary_text += f"  Min:  {np.min(all_G_values):.3f}\n"
            summary_text += f"  Max:  {np.max(all_G_values):.3f}\n\n"

        if all_F_values:
            summary_text += "Variational Free Energy (F):\n"
            summary_text += f"  Mean: {np.mean(all_F_values):.3f}\n"
            summary_text += f"  Std:  {np.std(all_F_values):.3f}\n"
            summary_text += f"  Min:  {np.min(all_F_values):.3f}\n"
            summary_text += f"  Max:  {np.max(all_F_values):.3f}\n"

        ax.text(
            0.1,
            0.9,
            summary_text,
            transform=ax.transAxes,
            fontfamily="monospace",
            fontsize=10,
            verticalalignment="top",
        )
        ax.set_title("D. Summary Statistics", fontweight="bold", loc="left")

        plt.tight_layout()

        # Save figure
        output_path = self.output_dir / f"{save_name}.{self.figure_format}"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        return output_path

    def generate_model_comparison(
        self,
        results_2level: Dict[str, Any],
        results_3level: Dict[str, Any],
        save_name: str = "model_comparison",
    ) -> Path:
        """
        Generate comparison figure between 2-level and 3-level models.

        Args:
            results_2level: Two-level model results
            results_3level: Three-level model results
            save_name: Output filename (without extension)

        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(3, 2, figsize=self.figure_sizes["double_column"])

        time_steps = min(results_2level["time_steps"], results_3level["time_steps"])
        time_range = np.arange(time_steps)

        # Panel A1 & A2: Precision dynamics comparison
        for idx, (results, title_suffix) in enumerate(
            [(results_2level, "2-Level"), (results_3level, "3-Level")]
        ):
            ax = axes[0, idx]
            precision_values = results["precision_values"]
            perception_level = self._get_perception_level_name(results)

            if perception_level in precision_values:
                precision_ts = precision_values[perception_level][:time_steps]
                ax.plot(
                    time_range,
                    precision_ts,
                    linewidth=2,
                    color=self.level_colors["perception"],
                )

            ax.set_ylabel("Precision (γ)")
            ax.set_title(
                f"A{idx + 1}. Precision ({title_suffix})", fontweight="bold", loc="left"
            )
            ax.grid(True, alpha=0.3)

        # Panel B1 & B2: Attentional state dynamics
        for idx, (results, title_suffix) in enumerate(
            [(results_2level, "2-Level"), (results_3level, "3-Level")]
        ):
            ax = axes[1, idx]
            true_states = results["true_states"]
            attention_level = self._get_attention_level_name(results)

            if attention_level in true_states:
                att_states = true_states[attention_level][:time_steps]

                for t in range(time_steps):
                    color = (
                        self.state_colors["focused"]
                        if att_states[t] == 0
                        else self.state_colors["distracted"]
                    )
                    ax.bar(t, 1, width=1, color=color, alpha=0.7, edgecolor="none")

            ax.set_ylim(0, 1)
            ax.set_ylabel("Attentional State")
            ax.set_title(
                f"B{idx + 1}. Attention ({title_suffix})", fontweight="bold", loc="left"
            )

        # Panel C1 & C2: Mind-wandering statistics
        ax = axes[2, 0]

        # Calculate mind-wandering statistics
        stats_2level = self._calculate_mind_wandering_stats(results_2level)
        stats_3level = self._calculate_mind_wandering_stats(results_3level)

        categories = ["Focused %", "Distracted %", "Transitions", "Avg Episode Length"]
        values_2level = [
            stats_2level["focused_percentage"],
            stats_2level["distracted_percentage"],
            stats_2level["num_transitions"],
            stats_2level["avg_episode_length"],
        ]
        values_3level = [
            stats_3level["focused_percentage"],
            stats_3level["distracted_percentage"],
            stats_3level["num_transitions"],
            stats_3level["avg_episode_length"],
        ]

        x = np.arange(len(categories))
        width = 0.35

        ax.bar(
            x - width / 2,
            values_2level,
            width,
            label="2-Level",
            color=self.level_colors["attention"],
            alpha=0.7,
        )
        ax.bar(
            x + width / 2,
            values_3level,
            width,
            label="3-Level",
            color=self.level_colors["meta_awareness"],
            alpha=0.7,
        )

        ax.set_ylabel("Value")
        ax.set_title("C1. Mind-wandering Statistics", fontweight="bold", loc="left")
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel C2: Precision comparison statistics
        ax = axes[2, 1]

        prec_stats_2level = self._calculate_precision_stats(results_2level)
        prec_stats_3level = self._calculate_precision_stats(results_3level)

        prec_categories = ["Mean", "Std", "Min", "Max"]
        prec_values_2level = [
            prec_stats_2level["mean"],
            prec_stats_2level["std"],
            prec_stats_2level["min"],
            prec_stats_2level["max"],
        ]
        prec_values_3level = [
            prec_stats_3level["mean"],
            prec_stats_3level["std"],
            prec_stats_3level["min"],
            prec_stats_3level["max"],
        ]

        x = np.arange(len(prec_categories))

        ax.bar(
            x - width / 2,
            prec_values_2level,
            width,
            label="2-Level",
            color=self.level_colors["attention"],
            alpha=0.7,
        )
        ax.bar(
            x + width / 2,
            prec_values_3level,
            width,
            label="3-Level",
            color=self.level_colors["meta_awareness"],
            alpha=0.7,
        )

        ax.set_ylabel("Precision (γ)")
        ax.set_title("C2. Precision Statistics", fontweight="bold", loc="left")
        ax.set_xticks(x)
        ax.set_xticklabels(prec_categories)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        output_path = self.output_dir / f"{save_name}.{self.figure_format}"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        return output_path

    def save_analysis_summary(
        self, results: Dict[str, Any], save_name: str = "analysis_summary"
    ) -> Path:
        """
        Save comprehensive analysis summary as JSON.

        Args:
            results: Simulation results dictionary
            save_name: Output filename (without extension)

        Returns:
            Path to saved JSON file
        """
        # Calculate comprehensive statistics
        summary = {
            "model_info": {
                "name": results.get("model_name", "Unknown"),
                "num_levels": results.get("num_levels", 0),
                "level_names": results.get("level_names", []),
                "time_steps": results.get("time_steps", 0),
            },
            "mind_wandering_stats": self._calculate_mind_wandering_stats(results),
            "precision_stats": self._calculate_precision_stats(results),
            "free_energy_stats": self._calculate_free_energy_stats(results),
            "behavioral_analysis": self._calculate_behavioral_analysis(results),
        }

        # Save to JSON
        output_path = self.output_dir / f"{save_name}.json"
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        return output_path

    # Helper methods
    def _get_perception_level_name(self, results: Dict[str, Any]) -> str:
        """Get the name of the perception level."""
        level_names = results.get("level_names", [])
        if level_names:
            return level_names[0]
        return "perception"

    def _get_attention_level_name(self, results: Dict[str, Any]) -> str:
        """Get the name of the attention level."""
        level_names = results.get("level_names", [])
        if len(level_names) > 1:
            return level_names[1]
        return "attention"

    def _identify_mind_wandering_periods(
        self, att_states: np.ndarray
    ) -> List[Tuple[int, int]]:
        """Identify continuous mind-wandering periods."""
        periods = []
        in_period = False
        start = 0

        for t, state in enumerate(att_states):
            if state == 1 and not in_period:  # Start of distracted period
                start = t
                in_period = True
            elif state == 0 and in_period:  # End of distracted period
                periods.append((start, t - 1))
                in_period = False

        # Handle case where period extends to end
        if in_period:
            periods.append((start, len(att_states) - 1))

        return periods

    def _identify_meta_control_periods(
        self, meta_states: np.ndarray
    ) -> List[Tuple[int, int]]:
        """Identify meta-awareness control periods."""
        periods = []
        in_period = False
        start = 0

        for t, state in enumerate(meta_states):
            if state == 0 and not in_period:  # Start of meta-aware period
                start = t
                in_period = True
            elif state == 1 and in_period:  # End of meta-aware period
                periods.append((start, t - 1))
                in_period = False

        if in_period:
            periods.append((start, len(meta_states) - 1))

        return periods

    def _calculate_mind_wandering_stats(
        self, results: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate mind-wandering statistics."""
        attention_level = self._get_attention_level_name(results)
        true_states = results.get("true_states", {})

        if attention_level not in true_states:
            return {
                "focused_percentage": 0,
                "distracted_percentage": 0,
                "num_transitions": 0,
                "avg_episode_length": 0,
            }

        att_states = true_states[attention_level]

        # Calculate percentages
        focused_percentage = np.mean(att_states == 0) * 100
        distracted_percentage = np.mean(att_states == 1) * 100

        # Calculate transitions
        transitions = np.sum(np.diff(att_states) != 0)

        # Calculate average episode length
        episodes = self._identify_mind_wandering_periods(att_states)
        if episodes:
            avg_episode_length = np.mean([end - start + 1 for start, end in episodes])
        else:
            avg_episode_length = 0

        return {
            "focused_percentage": focused_percentage,
            "distracted_percentage": distracted_percentage,
            "num_transitions": transitions,
            "avg_episode_length": avg_episode_length,
        }

    def _calculate_precision_stats(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate precision statistics."""
        perception_level = self._get_perception_level_name(results)
        precision_values = results.get("precision_values", {})

        if perception_level not in precision_values:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}

        precision_ts = precision_values[perception_level]

        return {
            "mean": float(np.mean(precision_ts)),
            "std": float(np.std(precision_ts)),
            "min": float(np.min(precision_ts)),
            "max": float(np.max(precision_ts)),
        }

    def _calculate_free_energy_stats(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate free energy statistics."""
        expected_G = results.get("expected_free_energy", {})
        variational_F = results.get("variational_free_energy", {})

        stats = {"expected": {}, "variational": {}}

        for level_name, G_values in expected_G.items():
            if G_values.size > 0:
                stats["expected"][level_name] = {
                    "mean": float(np.mean(G_values)),
                    "std": float(np.std(G_values)),
                    "min": float(np.min(G_values)),
                    "max": float(np.max(G_values)),
                }

        for level_name, F_values in variational_F.items():
            if F_values.size > 0:
                stats["variational"][level_name] = {
                    "mean": float(np.mean(F_values)),
                    "std": float(np.std(F_values)),
                    "min": float(np.min(F_values)),
                    "max": float(np.max(F_values)),
                }

        return stats

    def _calculate_behavioral_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate behavioral analysis metrics."""
        stimulus_sequence = results.get("stimulus_sequence", np.array([]))

        # Find oddball responses (simplified)
        oddball_times = np.where(stimulus_sequence == 1)[0]

        analysis = {
            "num_oddballs": len(oddball_times),
            "oddball_intervals": [],
            "response_patterns": {},
        }

        if len(oddball_times) > 1:
            intervals = np.diff(oddball_times)
            analysis["oddball_intervals"] = intervals.tolist()
            analysis["mean_interval"] = float(np.mean(intervals))
            analysis["std_interval"] = float(np.std(intervals))

        return analysis


# Example usage and testing
if __name__ == "__main__":
    # Create figure generator
    generator = FigureGenerator("./test_figures")

    # Create test results for validation
    time_steps = 200
    sample_results = {
        "model_name": "test_meta_awareness",
        "num_levels": 3,
        "level_names": ["perception", "attention", "meta_awareness"],
        "time_steps": time_steps,
        "stimulus_sequence": np.zeros(time_steps),
        "state_posteriors": {"perception": np.random.random((2, time_steps))},
        "true_states": {
            "attention": np.random.choice([0, 1], time_steps),
            "meta_awareness": np.random.choice([0, 1], time_steps),
        },
        "precision_values": {
            "perception": np.random.uniform(0.5, 2.0, time_steps),
            "attention": np.random.uniform(1.0, 4.0, time_steps),
        },
        "expected_free_energy": {"attention": np.random.random((2, time_steps))},
        "variational_free_energy": {"attention": np.random.random((2, time_steps))},
    }

    # Add some oddballs
    sample_results["stimulus_sequence"][[40, 80, 120, 160]] = 1

    # Generate test figures
    fig7_path = generator.generate_figure_7(sample_results)
    fig10_path = generator.generate_figure_10(sample_results)
    fig11_path = generator.generate_figure_11(sample_results)

    precision_path = generator.generate_precision_analysis(sample_results)
    free_energy_path = generator.generate_free_energy_analysis(sample_results)

    # Generate model comparison (using same results for both models for testing)
    comparison_path = generator.generate_model_comparison(
        sample_results, sample_results
    )

    # Save analysis summary
    summary_path = generator.save_analysis_summary(sample_results)

    print("Test figures generated:")
    print(f"  Figure 7: {fig7_path}")
    print(f"  Figure 10: {fig10_path}")
    print(f"  Figure 11: {fig11_path}")
    print(f"  Precision analysis: {precision_path}")
    print(f"  Free energy analysis: {free_energy_path}")
    print(f"  Model comparison: {comparison_path}")
    print(f"  Analysis summary: {summary_path}")
