"""
Visualization Utilities for AXIOM Implementation
==============================================

Implements visualization functions for AXIOM agent analysis including
slot positions, learning curves, model complexity evolution, and
performance metrics.

Authors: AXIOM Research Team
Institution: VERSES AI / Active Inference Institute
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


def visualize_slots(
    s_slot: np.ndarray,
    z_slot_present: np.ndarray,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 8),
    title: str = "AXIOM Object Slots",
) -> plt.Figure:
    """
    Visualize object slots with position, color, and shape information.

    Args:
        s_slot: Slot features [K, 7] - position(2) + color(3) + shape(2)
        z_slot_present: Slot presence indicators [K]
        save_path: Optional path to save figure
        figsize: Figure size
        title: Plot title

    Returns:
        matplotlib Figure object
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left plot: Spatial layout
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect("equal")
    ax1.set_title("Slot Positions and Shapes")
    ax1.set_xlabel("X Position")
    ax1.set_ylabel("Y Position")

    for k in range(len(s_slot)):
        if z_slot_present[k]:
            # Extract features
            x, y = s_slot[k, 0:2]  # Position
            r, g, b = s_slot[k, 2:5]  # Color
            width, height = s_slot[k, 5:7]  # Shape

            # Create colored rectangle
            color = np.clip([r, g, b], 0, 1)
            rect = patches.Rectangle(
                (x - width / 2, y - height / 2),
                width,
                height,
                facecolor=color,
                edgecolor="black",
                alpha=0.7,
                linewidth=2,
            )
            ax1.add_patch(rect)

            # Add slot label
            ax1.text(
                x, y, f"S{k}", ha="center", va="center", fontweight="bold", fontsize=10
            )

    # Right plot: Feature distributions
    ax2.set_title("Slot Feature Distributions")

    # Create feature matrix for present slots
    present_slots = s_slot[z_slot_present]
    if len(present_slots) > 0:
        feature_names = ["X", "Y", "R", "G", "B", "W", "H"]

        # Box plot of features
        box_data = []
        box_labels = []
        for i, name in enumerate(feature_names):
            if len(present_slots[:, i]) > 0:
                box_data.append(present_slots[:, i])
                box_labels.append(name)

        if box_data:
            ax2.boxplot(box_data, labels=box_labels)
            ax2.set_ylabel("Feature Values")
            ax2.set_xlabel("Features")
    else:
        ax2.text(
            0.5,
            0.5,
            "No active slots",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return None

    return fig


def plot_reward_history(
    rewards: List[float],
    window_size: int = 100,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 6),
    title: str = "AXIOM Learning Curve",
) -> plt.Figure:
    """
    Plot reward history with moving average.

    Args:
        rewards: List of rewards over time
        window_size: Window size for moving average
        save_path: Optional path to save figure
        figsize: Figure size
        title: Plot title

    Returns:
        matplotlib Figure object
    """

    if len(rewards) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No reward data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(title)
        return fig

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    timesteps = np.arange(len(rewards))
    rewards_array = np.array(rewards)

    # Top plot: Raw rewards and moving average
    ax1.plot(timesteps, rewards_array, alpha=0.3, color="blue", label="Raw Rewards")

    if len(rewards) >= window_size:
        moving_avg = pd.Series(rewards).rolling(window=window_size).mean()
        ax1.plot(
            timesteps,
            moving_avg,
            color="red",
            linewidth=2,
            label=f"Moving Average ({window_size} steps)",
        )

    ax1.set_ylabel("Reward")
    ax1.set_title("Learning Progress")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bottom plot: Cumulative reward
    cumulative_rewards = np.cumsum(rewards_array)
    ax2.plot(timesteps, cumulative_rewards, color="green", linewidth=2)
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Cumulative Reward")
    ax2.set_title("Cumulative Performance")
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return None

    return fig


def plot_model_complexity(
    complexity_history: List[Dict[str, int]],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 8),
    title: str = "AXIOM Model Complexity Evolution",
) -> plt.Figure:
    """
    Plot evolution of model complexity over time.

    Args:
        complexity_history: List of complexity dictionaries
        save_path: Optional path to save figure
        figsize: Figure size
        title: Plot title

    Returns:
        matplotlib Figure object
    """

    if len(complexity_history) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No complexity data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(title)
        return fig

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

    timesteps = np.arange(len(complexity_history))

    # Extract time series data
    keys = ["K_slots", "V_identities", "L_dynamics", "M_contexts"]
    data = {key: [item.get(key, 0) for item in complexity_history] for key in keys}

    # Plot component counts
    ax1.plot(timesteps, data["K_slots"], "o-", label="Slots (K)", markersize=4)
    ax1.set_ylabel("Number of Slots")
    ax1.set_title("Slot Complexity")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(
        timesteps,
        data["V_identities"],
        "o-",
        label="Identities (V)",
        color="orange",
        markersize=4,
    )
    ax2.set_ylabel("Number of Identities")
    ax2.set_title("Identity Complexity")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3.plot(
        timesteps,
        data["L_dynamics"],
        "o-",
        label="Dynamics (L)",
        color="green",
        markersize=4,
    )
    ax3.set_xlabel("Measurement Points")
    ax3.set_ylabel("Number of Dynamics Modes")
    ax3.set_title("Dynamics Complexity")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    ax4.plot(
        timesteps,
        data["M_contexts"],
        "o-",
        label="Contexts (M)",
        color="red",
        markersize=4,
    )
    ax4.set_xlabel("Measurement Points")
    ax4.set_ylabel("Number of Context Modes")
    ax4.set_title("Context Complexity")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return None

    return fig


def plot_performance_metrics(
    performance_history: List[Dict[str, Any]],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (15, 10),
    title: str = "AXIOM Performance Metrics",
) -> plt.Figure:
    """
    Plot comprehensive performance metrics.

    Args:
        performance_history: List of performance metric dictionaries
        save_path: Optional path to save figure
        figsize: Figure size
        title: Plot title

    Returns:
        matplotlib Figure object
    """

    if len(performance_history) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No performance data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(title)
        return fig

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    timesteps = np.arange(len(performance_history))

    # Extract common metrics
    def safe_extract(key, default=0):
        return [item.get(key, default) for item in performance_history]

    # Total parameters over time
    ax1 = fig.add_subplot(gs[0, 0])
    total_params = safe_extract("total_parameters")
    ax1.plot(timesteps, total_params, "o-", color="purple", markersize=4)
    ax1.set_ylabel("Total Parameters")
    ax1.set_title("Model Size Evolution")
    ax1.grid(True, alpha=0.3)

    # Computation time metrics
    ax2 = fig.add_subplot(gs[0, 1])
    if "total_step_time" in performance_history[0]:
        step_times = safe_extract("total_step_time")
        ax2.plot(timesteps, step_times, "o-", color="red", markersize=4)
        ax2.set_ylabel("Step Time (s)")
        ax2.set_title("Computation Time")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(
            0.5,
            0.5,
            "No timing data",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )

    # Memory usage
    ax3 = fig.add_subplot(gs[0, 2])
    if "memory_usage" in performance_history[0]:
        memory = safe_extract("memory_usage")
        ax3.plot(timesteps, memory, "o-", color="green", markersize=4)
        ax3.set_ylabel("Memory (MB)")
        ax3.set_title("Memory Usage")
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(
            0.5,
            0.5,
            "No memory data",
            ha="center",
            va="center",
            transform=ax3.transAxes,
        )

    # Learning rates by module
    ax4 = fig.add_subplot(gs[1, :])
    module_names = ["perception", "identity", "dynamics", "interaction", "planning"]
    colors = ["blue", "orange", "green", "red", "purple"]

    for i, (module, color) in enumerate(zip(module_names, colors)):
        if f"{module}_time" in performance_history[0]:
            times = safe_extract(f"{module}_time")
            ax4.plot(
                timesteps,
                times,
                "o-",
                label=module.capitalize(),
                color=color,
                markersize=3,
            )

    ax4.set_xlabel("Measurement Points")
    ax4.set_ylabel("Computation Time (s)")
    ax4.set_title("Module-wise Performance")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Performance summary table
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis("tight")
    ax5.axis("off")

    if len(performance_history) > 0:
        latest = performance_history[-1]
        summary_data = [
            ["Metric", "Value"],
            ["Total Parameters", f"{latest.get('total_parameters', 0):,}"],
            ["Average Step Time", f"{np.mean(safe_extract('total_step_time')):.4f}s"],
            ["Peak Memory Usage", f"{max(safe_extract('memory_usage', [0])):.1f} MB"],
            [
                "Model Complexity (K,V,L,M)",
                f"({latest.get('K_slots', 0)}, {latest.get('V_identities', 0)}, {latest.get('L_dynamics', 0)}, {latest.get('M_contexts', 0)})",
            ],
        ]

        table = ax5.table(
            cellText=summary_data[1:],
            colLabels=summary_data[0],
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

    plt.suptitle(title, fontsize=16, fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return None

    return fig


def plot_mixture_components(
    component_data: Dict[str, np.ndarray],
    component_type: str,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 8),
    max_components: int = 10,
) -> plt.Figure:
    """
    Visualize mixture model components.

    Args:
        component_data: Dictionary with component parameters
        component_type: Type of component ('slots', 'identities', 'dynamics', 'contexts')
        save_path: Optional path to save figure
        figsize: Figure size
        max_components: Maximum number of components to visualize

    Returns:
        matplotlib Figure object
    """

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    # Component weights
    if "weights" in component_data:
        weights = component_data["weights"][:max_components]
        axes[0].bar(range(len(weights)), weights)
        axes[0].set_title(f"{component_type.capitalize()} Mixing Weights")
        axes[0].set_xlabel("Component Index")
        axes[0].set_ylabel("Weight")
        axes[0].grid(True, alpha=0.3)

    # Component means (if available)
    if "means" in component_data:
        means = component_data["means"][:max_components]
        im = axes[1].imshow(means.T, aspect="auto", cmap="viridis")
        axes[1].set_title(f"{component_type.capitalize()} Component Means")
        axes[1].set_xlabel("Component Index")
        axes[1].set_ylabel("Feature Dimension")
        plt.colorbar(im, ax=axes[1])

    # Usage statistics
    if "usage" in component_data:
        usage = component_data["usage"][:max_components]
        axes[2].plot(range(len(usage)), usage, "o-", markersize=6)
        axes[2].set_title(f"{component_type.capitalize()} Usage Statistics")
        axes[2].set_xlabel("Component Index")
        axes[2].set_ylabel("Usage Count")
        axes[2].grid(True, alpha=0.3)

    # Quality metrics
    if "quality" in component_data:
        quality = component_data["quality"][:max_components]
        axes[3].bar(range(len(quality)), quality)
        axes[3].set_title(f"{component_type.capitalize()} Quality Metrics")
        axes[3].set_xlabel("Component Index")
        axes[3].set_ylabel("Log Likelihood")
        axes[3].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return None

    return fig


def create_axiom_dashboard(
    agent_summary: Dict[str, Any],
    recent_history: Dict[str, List],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (20, 12),
) -> plt.Figure:
    """
    Create comprehensive AXIOM agent dashboard.

    Args:
        agent_summary: Current agent summary statistics
        recent_history: Recent history data
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

    # Title
    fig.suptitle("AXIOM Agent Dashboard", fontsize=20, fontweight="bold")

    # Summary statistics (top row)
    ax_summary = fig.add_subplot(gs[0, :])
    ax_summary.axis("off")

    summary_text = f"""
    Episode: {agent_summary.get("episode", 0)} | Timestep: {agent_summary.get("timestep", 0)} | Total Reward: {agent_summary.get("total_reward", 0):.2f}
    Parameters: {agent_summary.get("total_parameters", 0):,} | Avg Reward: {agent_summary.get("average_reward", 0):.3f} | Recent Reward: {agent_summary.get("recent_reward", 0):.3f}
    Model Complexity - Slots: {agent_summary.get("model_complexity", {}).get("K_slots", 0)} | 
    Identities: {agent_summary.get("model_complexity", {}).get("V_identities", 0)} | 
    Dynamics: {agent_summary.get("model_complexity", {}).get("L_dynamics", 0)} | 
    Contexts: {agent_summary.get("model_complexity", {}).get("M_contexts", 0)}
    """

    ax_summary.text(
        0.5,
        0.5,
        summary_text,
        ha="center",
        va="center",
        transform=ax_summary.transAxes,
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5),
    )

    # Recent rewards
    if "rewards" in recent_history and len(recent_history["rewards"]) > 0:
        ax_rewards = fig.add_subplot(gs[1, :2])
        rewards = recent_history["rewards"][-200:]  # Last 200 steps
        ax_rewards.plot(rewards, color="blue", alpha=0.7)
        ax_rewards.set_title("Recent Rewards")
        ax_rewards.set_ylabel("Reward")
        ax_rewards.grid(True, alpha=0.3)

    # Model complexity evolution
    if (
        "model_complexity" in recent_history
        and len(recent_history["model_complexity"]) > 0
    ):
        ax_complexity = fig.add_subplot(gs[1, 2:])
        complexity = recent_history["model_complexity"][-50:]  # Last 50 measurements

        if complexity:
            timesteps = range(len(complexity))
            ax_complexity.plot(
                timesteps,
                [c.get("K_slots", 0) for c in complexity],
                "o-",
                label="Slots",
                markersize=3,
            )
            ax_complexity.plot(
                timesteps,
                [c.get("V_identities", 0) for c in complexity],
                "o-",
                label="Identities",
                markersize=3,
            )
            ax_complexity.plot(
                timesteps,
                [c.get("L_dynamics", 0) for c in complexity],
                "o-",
                label="Dynamics",
                markersize=3,
            )
            ax_complexity.plot(
                timesteps,
                [c.get("M_contexts", 0) for c in complexity],
                "o-",
                label="Contexts",
                markersize=3,
            )

            ax_complexity.set_title("Model Complexity Evolution")
            ax_complexity.set_ylabel("Component Count")
            ax_complexity.legend()
            ax_complexity.grid(True, alpha=0.3)

    # Performance metrics (if available)
    if (
        "performance_metrics" in recent_history
        and len(recent_history["performance_metrics"]) > 0
    ):
        perf_data = recent_history["performance_metrics"][-20:]  # Last 20 measurements

        if perf_data:
            ax_perf = fig.add_subplot(gs[2, :])
            timesteps = range(len(perf_data))

            # Extract timing data for different modules
            modules = ["perception", "identity", "dynamics", "interaction", "planning"]
            colors = ["blue", "orange", "green", "red", "purple"]

            for module, color in zip(modules, colors):
                times = [p.get(f"{module}_time", 0) for p in perf_data]
                if any(t > 0 for t in times):
                    ax_perf.plot(
                        timesteps,
                        times,
                        "o-",
                        label=module.capitalize(),
                        color=color,
                        markersize=3,
                    )

            ax_perf.set_title("Module Performance")
            ax_perf.set_xlabel("Measurement Points")
            ax_perf.set_ylabel("Computation Time (s)")
            ax_perf.legend()
            ax_perf.grid(True, alpha=0.3)

    # Action distribution (if available)
    if "actions" in recent_history and len(recent_history["actions"]) > 0:
        ax_actions = fig.add_subplot(gs[3, :2])
        actions = recent_history["actions"][-1000:]  # Last 1000 actions

        unique_actions, counts = np.unique(actions, return_counts=True)
        ax_actions.bar(unique_actions, counts)
        ax_actions.set_title("Recent Action Distribution")
        ax_actions.set_xlabel("Action")
        ax_actions.set_ylabel("Count")
        ax_actions.grid(True, alpha=0.3)

    # Free energy evolution (if available)
    if "free_energy" in recent_history and len(recent_history["free_energy"]) > 0:
        ax_fe = fig.add_subplot(gs[3, 2:])
        free_energy = recent_history["free_energy"][-200:]  # Last 200 measurements
        ax_fe.plot(free_energy, color="red", alpha=0.7)
        ax_fe.set_title("Expected Free Energy")
        ax_fe.set_xlabel("Timestep")
        ax_fe.set_ylabel("Free Energy")
        ax_fe.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return None

    return fig


# Example usage and testing
def test_visualization_utilities():
    """Test visualization utilities with example data."""

    print("Testing AXIOM Visualization Utilities...")

    # Generate test data
    np.random.seed(42)

    # Test slot visualization
    K_slots = 6
    s_slot = np.random.rand(K_slots, 7)
    s_slot[:, 0:2] = np.random.rand(K_slots, 2)  # Positions in [0,1]
    s_slot[:, 2:5] = np.random.rand(K_slots, 3)  # Colors in [0,1]
    s_slot[:, 5:7] = 0.05 + 0.1 * np.random.rand(K_slots, 2)  # Shapes
    z_slot_present = np.random.choice([True, False], K_slots, p=[0.8, 0.2])

    visualize_slots(s_slot, z_slot_present)
    print("✓ Slot visualization created")

    # Test reward history
    rewards = np.cumsum(np.random.randn(1000) * 0.1) + np.random.randn(1000) * 0.5
    plot_reward_history(rewards)
    print("✓ Reward history plot created")

    # Test model complexity
    complexity_history = []
    for i in range(50):
        complexity_history.append(
            {
                "K_slots": 4 + i // 10,
                "V_identities": 2 + i // 15,
                "L_dynamics": 5 + i // 8,
                "M_contexts": 10 + i // 5,
                "total_parameters": 1000 + i * 20,
            }
        )

    plot_model_complexity(complexity_history)
    print("✓ Model complexity plot created")

    # Test performance metrics
    performance_history = []
    for i in range(20):
        performance_history.append(
            {
                "total_parameters": 1000 + i * 50,
                "total_step_time": 0.1 + 0.01 * np.random.randn(),
                "perception_time": 0.02 + 0.005 * np.random.randn(),
                "planning_time": 0.05 + 0.01 * np.random.randn(),
                "memory_usage": 100 + i * 5 + 10 * np.random.randn(),
            }
        )

    plot_performance_metrics(performance_history)
    print("✓ Performance metrics plot created")

    plt.show()
    print("All visualization tests completed successfully!")


if __name__ == "__main__":
    test_visualization_utilities()
