#!/usr/bin/env python3
"""
PyMDP Gridworld Visualization Module

Comprehensive visualization utilities for PyMDP gridworld simulations.
This module provides plotting capabilities for gridworld environments,
agent trajectories, belief distributions, and performance metrics.

Features:
- Gridworld state visualization with agent position
- Belief distribution heatmaps
- Trajectory plotting
- Performance metrics visualization
- Configurable plot styles and output formats

Author: GNN PyMDP Integration
Date: 2024
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

warnings.filterwarnings("ignore")


class GridworldVisualizer:
    """Visualization utilities for gridworld simulation"""

    def __init__(
        self,
        grid_layout: np.ndarray,
        output_dir: Path,
        plot_style: str = "seaborn-v0_8",
        figure_size: Tuple[int, int] = (12, 8),
        show_plots: bool = False,
    ):
        """
        Initialize the GridworldVisualizer.

        Args:
            grid_layout: 2D numpy array representing the gridworld
            output_dir: Directory to save visualizations
            plot_style: Matplotlib style to use
            figure_size: Figure size for plots
            show_plots: Whether to display plots during execution
        """
        self.grid_layout = grid_layout
        self.output_dir = output_dir
        self.grid_size = grid_layout.shape[0]
        self.show_plots = show_plots
        self.figure_size = figure_size

        # Set up plotting style
        plt.style.use(plot_style)
        sns.set_palette("husl")

    def plot_gridworld(
        self,
        agent_pos: Tuple[int, int] = None,
        beliefs: np.ndarray = None,
        title: str = "Gridworld State",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Plot the gridworld with agent position and beliefs"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figure_size)

        # Plot 1: Gridworld layout with agent position
        self._plot_grid_layout(ax1, agent_pos, title)

        # Plot 2: Belief distribution
        if beliefs is not None:
            self._plot_beliefs(ax2, beliefs, "Agent Beliefs")
        else:
            ax2.text(
                0.5,
                0.5,
                "No beliefs provided",
                ha="center",
                va="center",
                transform=ax2.transAxes,
                fontsize=14,
            )
            ax2.set_title("Agent Beliefs", fontsize=14, fontweight="bold")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if self.show_plots:
            plt.show()

        return fig

    def _plot_grid_layout(self, ax: plt.Axes, agent_pos: Tuple[int, int], title: str):
        """Plot gridworld layout"""
        # Create color map for different cell types
        colors = {
            0: "white",  # Empty
            1: "gray",  # Wall
            2: "green",  # Goal
            3: "red",  # Hazard
        }

        # Plot grid cells
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell_type = self.grid_layout[i, j]
                color = colors.get(cell_type, "blue")

                rect = patches.Rectangle(
                    (j, self.grid_size - 1 - i),
                    1,
                    1,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=2,
                )
                ax.add_patch(rect)

                # Add text labels
                if cell_type == 2:
                    ax.text(
                        j + 0.5,
                        self.grid_size - 1 - i + 0.5,
                        "G",
                        ha="center",
                        va="center",
                        fontsize=14,
                        fontweight="bold",
                    )
                elif cell_type == 3:
                    ax.text(
                        j + 0.5,
                        self.grid_size - 1 - i + 0.5,
                        "H",
                        ha="center",
                        va="center",
                        fontsize=14,
                        fontweight="bold",
                    )

        # Plot agent position if provided
        if agent_pos is not None:
            ax.plot(
                agent_pos[1] + 0.5,
                self.grid_size - 1 - agent_pos[0] + 0.5,
                "ko",
                markersize=15,
                markerfacecolor="blue",
                markeredgecolor="black",
            )
            ax.text(
                agent_pos[1] + 0.5,
                self.grid_size - 1 - agent_pos[0] + 0.5,
                "A",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="white",
            )

        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.grid(True, alpha=0.3)

    def _plot_beliefs(self, ax: plt.Axes, beliefs: np.ndarray, title: str):
        """Plot belief distribution over states"""
        # Reshape beliefs to grid
        belief_grid = beliefs.reshape(self.grid_size, self.grid_size)

        # Create heatmap
        im = ax.imshow(belief_grid, cmap="viridis", aspect="equal")

        # Add text annotations
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                ax.text(
                    j,
                    i,
                    f"{belief_grid[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

        # Add colorbar
        plt.colorbar(im, ax=ax, label="Belief Probability")

    def plot_trajectory(
        self,
        trajectory: List[Tuple[int, int]],
        title: str = "Agent Trajectory",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Plot agent trajectory through the gridworld"""
        fig, ax = plt.subplots(figsize=self.figure_size)

        # Plot grid layout
        self._plot_grid_layout(ax, None, title)

        # Plot trajectory
        if len(trajectory) > 1:
            x_coords = [pos[1] + 0.5 for pos in trajectory]
            y_coords = [self.grid_size - 1 - pos[0] + 0.5 for pos in trajectory]

            ax.plot(x_coords, y_coords, "b-", linewidth=3, alpha=0.7)
            ax.plot(x_coords, y_coords, "bo", markersize=8, alpha=0.8)

            # Mark start and end
            ax.plot(x_coords[0], y_coords[0], "go", markersize=12, label="Start")
            ax.plot(x_coords[-1], y_coords[-1], "ro", markersize=12, label="End")

            ax.legend()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if self.show_plots:
            plt.show()

        return fig

    def plot_performance_metrics(
        self, metrics: Dict[str, List], save_path: Optional[Path] = None
    ) -> plt.Figure:
        """Plot performance metrics over episodes"""
        fig, axes = plt.subplots(2, 2, figsize=self.figure_size)

        # Episode rewards
        if "episode_rewards" in metrics and metrics["episode_rewards"]:
            axes[0, 0].plot(metrics["episode_rewards"])
            axes[0, 0].set_title("Episode Rewards")
            axes[0, 0].set_xlabel("Episode")
            axes[0, 0].set_ylabel("Total Reward")
            axes[0, 0].grid(True, alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, "No reward data", ha="center", va="center")
            axes[0, 0].set_title("Episode Rewards")

        # Episode lengths
        if "episode_lengths" in metrics and metrics["episode_lengths"]:
            axes[0, 1].plot(metrics["episode_lengths"])
            axes[0, 1].set_title("Episode Lengths")
            axes[0, 1].set_xlabel("Episode")
            axes[0, 1].set_ylabel("Steps")
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, "No length data", ha="center", va="center")
            axes[0, 1].set_title("Episode Lengths")

        # Average belief entropy
        if "belief_entropies" in metrics and metrics["belief_entropies"]:
            axes[1, 0].plot(metrics["belief_entropies"])
            axes[1, 0].set_title("Average Belief Entropy")
            axes[1, 0].set_xlabel("Episode")
            axes[1, 0].set_ylabel("Entropy")
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, "No entropy data", ha="center", va="center")
            axes[1, 0].set_title("Average Belief Entropy")

        # Success rate (rolling average)
        if "success_rates" in metrics and metrics["success_rates"]:
            axes[1, 1].plot(metrics["success_rates"])
            axes[1, 1].set_title("Success Rate (Rolling Average)")
            axes[1, 1].set_xlabel("Episode")
            axes[1, 1].set_ylabel("Success Rate")
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, "No success rate data", ha="center", va="center")
            axes[1, 1].set_title("Success Rate (Rolling Average)")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if self.show_plots:
            plt.show()

        return fig

    def plot_free_energy_evolution(
        self,
        variational_fe: List[float],
        expected_fe: List[np.ndarray],
        actions: List[int],
        title: str = "Free Energy Evolution",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Plot evolution of free energy values and action selection over time"""
        if not variational_fe and not expected_fe:
            return None

        fig, axes = plt.subplots(
            3, 1, figsize=(self.figure_size[0], self.figure_size[1] * 1.5)
        )

        # Plot 1: Variational Free Energy (state inference)
        if variational_fe:
            time_steps = range(len(variational_fe))
            axes[0].plot(time_steps, variational_fe, "b-o", linewidth=2, markersize=6)
            axes[0].set_title(
                "Variational Free Energy (State Inference)", fontweight="bold"
            )
            axes[0].set_xlabel("Time Step")
            axes[0].set_ylabel("Variational FE")
            axes[0].grid(True, alpha=0.3)
            axes[0].set_ylim(bottom=0)
        else:
            axes[0].text(
                0.5,
                0.5,
                "No variational FE data",
                ha="center",
                va="center",
                transform=axes[0].transAxes,
            )
            axes[0].set_title(
                "Variational Free Energy (State Inference)", fontweight="bold"
            )

        # Plot 2: Expected Free Energy of Policies (policy inference)
        if expected_fe and len(expected_fe) > 0:
            # Convert expected_fe to array for plotting
            fe_array = np.array(
                [efe if efe is not None else np.array([]) for efe in expected_fe]
            )
            if len(fe_array) > 0 and len(fe_array[0]) > 0:
                time_steps = range(len(fe_array))
                num_policies = len(fe_array[0])

                # Create heatmap of expected free energies
                fe_matrix = np.array(
                    [
                        efe
                        if len(efe) == num_policies
                        else np.full(num_policies, np.nan)
                        for efe in fe_array
                    ]
                )

                im = axes[1].imshow(
                    fe_matrix.T, aspect="auto", cmap="viridis", origin="lower"
                )
                axes[1].set_title("Expected Free Energy of Policies", fontweight="bold")
                axes[1].set_xlabel("Time Step")
                axes[1].set_ylabel("Policy Index")
                plt.colorbar(im, ax=axes[1], label="Expected FE")

                # Mark selected actions (handle length mismatch)
                if actions:
                    action_steps = min(len(actions), len(time_steps))
                    for t in range(action_steps):
                        action = actions[t]
                        if action < num_policies:
                            axes[1].plot(
                                t,
                                action,
                                "ro",
                                markersize=8,
                                markeredgecolor="white",
                                markeredgewidth=2,
                            )
            else:
                axes[1].text(
                    0.5,
                    0.5,
                    "No policy FE data",
                    ha="center",
                    va="center",
                    transform=axes[1].transAxes,
                )
                axes[1].set_title("Expected Free Energy of Policies", fontweight="bold")
        else:
            axes[1].text(
                0.5,
                0.5,
                "No policy FE data",
                ha="center",
                va="center",
                transform=axes[1].transAxes,
            )
            axes[1].set_title("Expected Free Energy of Policies", fontweight="bold")

        # Plot 3: Action Selection Timeline
        if actions:
            time_steps = range(len(actions))
            action_names = ["North", "South", "East", "West"]

            # Create action timeline
            axes[2].step(
                time_steps, actions, "g-", linewidth=3, where="post", alpha=0.7
            )
            axes[2].scatter(
                time_steps,
                actions,
                c=actions,
                cmap="tab10",
                s=100,
                edgecolors="black",
                linewidth=2,
            )

            axes[2].set_title("Selected Actions Over Time", fontweight="bold")
            axes[2].set_xlabel("Time Step")
            axes[2].set_ylabel("Action")
            axes[2].set_yticks(range(min(4, max(actions) + 1)))
            axes[2].set_yticklabels(
                [
                    action_names[i] if i < len(action_names) else f"Action {i}"
                    for i in range(min(4, max(actions) + 1))
                ]
            )
            axes[2].grid(True, alpha=0.3)

            # Add action labels on points
            for t, action in enumerate(actions):
                axes[2].annotate(
                    action_names[action]
                    if action < len(action_names)
                    else f"A{action}",
                    (t, action),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )
        else:
            axes[2].text(
                0.5,
                0.5,
                "No action data",
                ha="center",
                va="center",
                transform=axes[2].transAxes,
            )
            axes[2].set_title("Selected Actions Over Time", fontweight="bold")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if self.show_plots:
            plt.show()

        return fig

    def plot_belief_evolution(
        self,
        belief_traces: List[np.ndarray],
        title: str = "Belief Evolution",
        save_path: Optional[Path] = None,
        max_steps: int = 20,
    ) -> plt.Figure:
        """Plot evolution of beliefs over time steps"""
        if not belief_traces:
            return None

        # Limit number of steps to plot
        steps_to_plot = min(len(belief_traces), max_steps)

        fig, axes = plt.subplots(
            2, min(5, steps_to_plot), figsize=(3 * min(5, steps_to_plot), 6)
        )

        if steps_to_plot == 1:
            axes = [axes]
        elif steps_to_plot <= 5:
            axes = [axes[0], axes[1]]

        for step in range(steps_to_plot):
            row = step // 5
            col = step % 5

            if steps_to_plot <= 5:
                ax = axes[row] if isinstance(axes[row], plt.Axes) else axes[row][col]
            else:
                ax = axes[row, col]

            belief_grid = belief_traces[step].reshape(self.grid_size, self.grid_size)

            ax.imshow(belief_grid, cmap="viridis", aspect="equal", vmin=0, vmax=1)
            ax.set_title(f"Step {step}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        # Hide unused subplots
        total_subplots = 2 * 5 if steps_to_plot > 5 else 2
        for i in range(steps_to_plot, total_subplots):
            row = i // 5
            col = i % 5
            if steps_to_plot <= 5:
                ax = axes[row] if isinstance(axes[row], plt.Axes) else axes[row][col]
            else:
                ax = axes[row, col]
            ax.axis("off")

        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if self.show_plots:
            plt.show()

        return fig

    def plot_active_inference_analysis(
        self,
        belief_traces: List[np.ndarray],
        variational_fe: List[float],
        expected_fe: List[np.ndarray],
        actions: List[int],
        positions: List[Tuple[int, int]],
        episode_num: int,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Create Active Inference analysis visualization"""

        if not belief_traces and not variational_fe and not expected_fe:
            return None

        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[2, 1, 1])

        # Plot 1: Belief evolution heatmap (large plot)
        ax1 = fig.add_subplot(gs[:, 0])
        if belief_traces:
            # Create belief evolution matrix
            belief_matrix = np.array([belief.reshape(-1) for belief in belief_traces])

            im1 = ax1.imshow(
                belief_matrix.T, aspect="auto", cmap="viridis", origin="lower"
            )
            ax1.set_title(
                f"Episode {episode_num}: Belief Evolution Over Time",
                fontsize=14,
                fontweight="bold",
            )
            ax1.set_xlabel("Time Step")
            ax1.set_ylabel("State Index")

            # Add trajectory overlay
            if positions:
                trajectory_states = [
                    pos[0] * self.grid_size + pos[1] for pos in positions
                ]
                time_steps = range(len(trajectory_states))
                ax1.plot(
                    time_steps,
                    trajectory_states,
                    "r-",
                    linewidth=3,
                    alpha=0.8,
                    label="True Trajectory",
                )
                ax1.scatter(
                    time_steps,
                    trajectory_states,
                    c="red",
                    s=50,
                    edgecolors="white",
                    linewidth=2,
                )
                ax1.legend()

            plt.colorbar(im1, ax=ax1, label="Belief Probability")
        else:
            ax1.text(
                0.5,
                0.5,
                "No belief data available",
                ha="center",
                va="center",
                transform=ax1.transAxes,
            )
            ax1.set_title(
                f"Episode {episode_num}: Belief Evolution Over Time",
                fontsize=14,
                fontweight="bold",
            )

        # Plot 2: Variational Free Energy
        ax2 = fig.add_subplot(gs[0, 1])
        if variational_fe:
            time_steps = range(len(variational_fe))
            ax2.plot(time_steps, variational_fe, "b-o", linewidth=2, markersize=4)
            ax2.set_title("Variational Free Energy", fontweight="bold")
            ax2.set_xlabel("Time Step")
            ax2.set_ylabel("VFE")
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(
                0.5,
                0.5,
                "No VFE data",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )
            ax2.set_title("Variational Free Energy", fontweight="bold")

        # Plot 3: Policy Free Energy (simplified)
        ax3 = fig.add_subplot(gs[0, 2])
        if expected_fe and actions:
            # Show average expected free energy and selected actions
            avg_efe = [
                np.mean(efe) if efe is not None and len(efe) > 0 else 0
                for efe in expected_fe
            ]
            time_steps = range(len(avg_efe))

            # Plot average expected free energy
            ax3_twin = ax3.twinx()
            ax3.plot(time_steps, avg_efe, "g-", linewidth=2, alpha=0.7, label="Avg EFE")

            # Plot selected actions as bars (adjust for length mismatch)
            action_time_steps = range(min(len(time_steps), len(actions)))
            action_values = actions[: len(action_time_steps)]
            ax3_twin.bar(
                action_time_steps,
                action_values,
                alpha=0.5,
                color="orange",
                label="Actions",
            )

            ax3.set_title("Policy Analysis", fontweight="bold")
            ax3.set_xlabel("Time Step")
            ax3.set_ylabel("Avg Expected FE", color="green")
            ax3_twin.set_ylabel("Selected Action", color="orange")
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(
                0.5,
                0.5,
                "No policy data",
                ha="center",
                va="center",
                transform=ax3.transAxes,
            )
            ax3.set_title("Policy Analysis", fontweight="bold")

        # Plot 4: Belief entropy over time
        ax4 = fig.add_subplot(gs[1, 1])
        if belief_traces:
            entropies = [
                -np.sum(belief * np.log(belief + 1e-16)) for belief in belief_traces
            ]
            time_steps = range(len(entropies))
            ax4.plot(time_steps, entropies, "m-o", linewidth=2, markersize=4)
            ax4.set_title("Belief Entropy", fontweight="bold")
            ax4.set_xlabel("Time Step")
            ax4.set_ylabel("Entropy (nats)")
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(
                0.5,
                0.5,
                "No belief data",
                ha="center",
                va="center",
                transform=ax4.transAxes,
            )
            ax4.set_title("Belief Entropy", fontweight="bold")

        # Plot 5: Action confidence (if policy data available)
        ax5 = fig.add_subplot(gs[1, 2])
        if expected_fe and actions:
            # Calculate action confidence as difference between selected and other policies
            confidences = []
            for t, (efe, action) in enumerate(zip(expected_fe, actions)):
                if efe is not None and len(efe) > action:
                    # Confidence as difference between best and worst policy
                    conf = np.max(efe) - np.min(efe) if len(efe) > 1 else 0
                    confidences.append(conf)
                else:
                    confidences.append(0)

            if confidences:
                time_steps = range(len(confidences))
                ax5.plot(time_steps, confidences, "r-o", linewidth=2, markersize=4)
                ax5.set_title("Action Confidence", fontweight="bold")
                ax5.set_xlabel("Time Step")
                ax5.set_ylabel("EFE Range")
                ax5.grid(True, alpha=0.3)
            else:
                ax5.text(
                    0.5,
                    0.5,
                    "No confidence data",
                    ha="center",
                    va="center",
                    transform=ax5.transAxes,
                )
                ax5.set_title("Action Confidence", fontweight="bold")
        else:
            ax5.text(
                0.5,
                0.5,
                "No policy data",
                ha="center",
                va="center",
                transform=ax5.transAxes,
            )
            ax5.set_title("Action Confidence", fontweight="bold")

        # Plot 6: Spatial belief distribution (last time step)
        ax6 = fig.add_subplot(gs[2, 1:])
        if belief_traces:
            # Show final belief distribution as gridworld heatmap
            final_beliefs = belief_traces[-1].reshape(self.grid_size, self.grid_size)

            im6 = ax6.imshow(final_beliefs, cmap="viridis", aspect="equal")
            ax6.set_title("Final Belief Distribution", fontweight="bold")
            ax6.set_xlabel("Column")
            ax6.set_ylabel("Row")

            # Add grid and values
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    ax6.text(
                        j,
                        i,
                        f"{final_beliefs[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=8,
                    )

            # Mark final position
            if positions:
                final_pos = positions[-1]
                ax6.plot(
                    final_pos[1],
                    final_pos[0],
                    "ro",
                    markersize=12,
                    markeredgecolor="white",
                    markeredgewidth=2,
                )

            plt.colorbar(im6, ax=ax6, label="Belief Probability")
        else:
            ax6.text(
                0.5,
                0.5,
                "No belief data",
                ha="center",
                va="center",
                transform=ax6.transAxes,
            )
            ax6.set_title("Final Belief Distribution", fontweight="bold")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if self.show_plots:
            plt.show()

        return fig

    def create_summary_figure(
        self,
        final_pos: Tuple[int, int],
        final_beliefs: np.ndarray,
        trajectory: List[Tuple[int, int]],
        metrics: Dict[str, List],
        episode_num: int,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """Create a comprehensive summary figure for an episode"""
        fig = plt.figure(figsize=(16, 12))

        # Create a 3x2 grid layout
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2])

        # Final state visualization
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_grid_layout(ax1, final_pos, f"Episode {episode_num} Final State")

        # Final beliefs
        ax2 = fig.add_subplot(gs[0, 1])
        if final_beliefs is not None:
            self._plot_beliefs(
                ax2, final_beliefs, f"Episode {episode_num} Final Beliefs"
            )
        else:
            ax2.text(0.5, 0.5, "No beliefs data", ha="center", va="center")
            ax2.set_title(f"Episode {episode_num} Final Beliefs")

        # Trajectory
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_grid_layout(ax3, None, f"Episode {episode_num} Trajectory")
        if len(trajectory) > 1:
            x_coords = [pos[1] + 0.5 for pos in trajectory]
            y_coords = [self.grid_size - 1 - pos[0] + 0.5 for pos in trajectory]

            ax3.plot(x_coords, y_coords, "b-", linewidth=3, alpha=0.7)
            ax3.plot(x_coords, y_coords, "bo", markersize=8, alpha=0.8)

            # Mark start and end
            ax3.plot(x_coords[0], y_coords[0], "go", markersize=12, label="Start")
            ax3.plot(x_coords[-1], y_coords[-1], "ro", markersize=12, label="End")
            ax3.legend()

        # Performance metrics (simplified)
        ax4 = fig.add_subplot(gs[2, :])
        if "episode_rewards" in metrics and metrics["episode_rewards"]:
            episodes = range(1, len(metrics["episode_rewards"]) + 1)
            ax4.plot(
                episodes, metrics["episode_rewards"], "b-o", label="Episode Rewards"
            )
            ax4.set_xlabel("Episode")
            ax4.set_ylabel("Total Reward")
            ax4.set_title("Performance Over Episodes")
            ax4.grid(True, alpha=0.3)
            ax4.legend()

            # Highlight current episode
            if episode_num <= len(metrics["episode_rewards"]):
                current_reward = metrics["episode_rewards"][episode_num - 1]
                ax4.plot(
                    episode_num,
                    current_reward,
                    "ro",
                    markersize=10,
                    label=f"Current Episode: {current_reward:.2f}",
                )
                ax4.legend()
        else:
            ax4.text(0.5, 0.5, "No performance data", ha="center", va="center")
            ax4.set_title("Performance Over Episodes")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if self.show_plots:
            plt.show()

        return fig


def create_visualizer(
    grid_layout: np.ndarray, output_dir: Path, **kwargs
) -> GridworldVisualizer:
    """Factory function to create a GridworldVisualizer instance"""
    return GridworldVisualizer(grid_layout, output_dir, **kwargs)


def save_all_visualizations(
    visualizer: GridworldVisualizer,
    all_traces: List[Dict],
    performance_metrics: Dict[str, List],
    grid_layout: np.ndarray,
) -> None:
    """Save all visualization outputs for a simulation"""

    # Create episode-specific visualizations in subdirectories
    for i, trace in enumerate(all_traces):
        episode_num = i + 1
        episode_dir = visualizer.output_dir / f"episode_{episode_num}"
        episode_dir.mkdir(exist_ok=True)

        # Final state visualization
        if trace["positions"]:
            final_pos = trace["positions"][-1]
            final_beliefs = trace["beliefs"][-1] if trace["beliefs"] else None

            # Individual plots
            fig = visualizer.plot_gridworld(
                agent_pos=final_pos,
                beliefs=final_beliefs,
                title=f"Episode {episode_num} Final State",
                save_path=episode_dir / "final_state.png",
            )
            plt.close(fig)

            # Trajectory plot
            if len(trace["positions"]) > 1:
                fig = visualizer.plot_trajectory(
                    trace["positions"],
                    title=f"Episode {episode_num} Trajectory",
                    save_path=episode_dir / "trajectory.png",
                )
                plt.close(fig)

            # Comprehensive summary
            fig = visualizer.create_summary_figure(
                final_pos=final_pos,
                final_beliefs=final_beliefs,
                trajectory=trace["positions"],
                metrics=performance_metrics,
                episode_num=episode_num,
                save_path=episode_dir / "summary.png",
            )
            plt.close(fig)

            # Belief evolution for longer episodes
            if len(trace["beliefs"]) > 1:
                fig = visualizer.plot_belief_evolution(
                    trace["beliefs"],
                    title=f"Episode {episode_num} Belief Evolution",
                    save_path=episode_dir / "belief_evolution.png",
                )
                if fig:
                    plt.close(fig)

            # Free energy evolution (if data available)
            variational_fe = trace.get("variational_free_energies", [])
            expected_fe = trace.get("expected_free_energies", [])
            actions = trace.get("actions", [])

            if variational_fe or expected_fe:
                fig = visualizer.plot_free_energy_evolution(
                    variational_fe=variational_fe,
                    expected_fe=expected_fe,
                    actions=actions,
                    title=f"Episode {episode_num} Free Energy Evolution",
                    save_path=episode_dir / "free_energy.png",
                )
                if fig:
                    plt.close(fig)

            # Active Inference analysis
            if trace["beliefs"] and len(trace["beliefs"]) > 1:
                fig = visualizer.plot_active_inference_analysis(
                    belief_traces=trace["beliefs"],
                    variational_fe=variational_fe,
                    expected_fe=expected_fe,
                    actions=actions,
                    positions=trace["positions"],
                    episode_num=episode_num,
                    save_path=episode_dir / "active_inference_analysis.png",
                )
                if fig:
                    plt.close(fig)

            # Save episode-specific data
            episode_data = {
                "episode_number": episode_num,
                "total_reward": sum(trace.get("rewards", [])),
                "episode_length": len(trace.get("positions", [])),
                "final_position": trace["positions"][-1]
                if trace.get("positions")
                else None,
                "mean_variational_fe": float(
                    np.mean(trace.get("variational_free_energies", [0]))
                ),
                "positions": trace.get("positions", []),
                "actions": trace.get("actions", []),
                "rewards": trace.get("rewards", []),
            }

            from pymdp_utils import safe_json_dump

            safe_json_dump(episode_data, episode_dir / "episode_data.json")

    # Cross-episode analysis at top level
    fig = visualizer.plot_performance_metrics(
        performance_metrics, save_path=visualizer.output_dir / "performance_metrics.png"
    )
    plt.close(fig)


if __name__ == "__main__":
    print(
        "PyMDP Gridworld Visualizer - Visualization utilities for gridworld simulations"
    )
    print("This module should be imported and used with the main simulation script.")
