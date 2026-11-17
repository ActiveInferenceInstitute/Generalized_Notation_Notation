"""
Visualization functions for Sandved-Smith et al. (2021) computational phenomenology model.
Generates Figures 7, 10, and 11 from the paper.
"""

import os
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np


def setup_matplotlib_style():
    """Set up matplotlib style for publication-quality figures."""
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 600,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 12,
        }
    )


def plot_figure_7(
    results: Dict[str, Any], save_path: Optional[str] = None
) -> plt.Figure:
    """
    Generate Figure 7: Influence of attentional state on perception.

    Args:
        results: Dictionary containing simulation results
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    setup_matplotlib_style()

    X1_bar = results["X1_bar"]
    x2 = results["x2"]
    O = results["O"]
    T = results["T"]

    fig = plt.figure(figsize=(8, 3.5))

    # Create background shading for attentional states
    ax = plt.subplot(1, 1, 1)

    # Background coloring based on attentional state
    attention_background = 0.5 * (0.5 * x2)
    extent = [0.5, T + 0.5, -0.3, 1.3]
    ax.imshow(
        [attention_background, attention_background],
        aspect="auto",
        cmap="Oranges",
        alpha=0.5,
        extent=extent,
    )

    # Plot perceptual state beliefs
    ax.plot(
        1 + np.arange(T),
        X1_bar[0, :],
        label=r"${\bar{s}}^{(1)}$",
        color="royalblue",
        linewidth=2,
    )

    # Plot true stimuli
    ax.scatter(
        1 + np.arange(T), 1 - O, label="stimulus", color="steelblue", s=30, alpha=0.8
    )

    # Labels and formatting
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["deviant", "standard"])
    ax.set_xlabel(r"time ($\tau$)")
    ax.set_title("Influence of attentional state on perception")
    ax.legend(loc="lower right")
    ax.set_ylim([-0.3, 1.3])
    ax.set_xlim([0.5, T + 0.5])

    # Add text annotations
    ax.text(19, 1.14, r"$s^{(2)}$ = focused", fontsize=12)
    ax.text(67, 1.14, r"$s^{(2)}$ = distracted", color="white", fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")

    return fig


def plot_figure_10(
    results: Dict[str, Any], save_path: Optional[str] = None
) -> plt.Figure:
    """
    Generate Figure 10: Two-level model with attentional cycles.

    Args:
        results: Dictionary containing simulation results
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    setup_matplotlib_style()

    Pi2_bar = results["Pi2_bar"]
    u2 = results["u2"]
    X2_bar = results["X2_bar"]
    x2 = results["x2"]
    X1_bar = results["X1_bar"]
    O = results["O"]
    T = results["T"]

    fig = plt.figure(figsize=(8, 6))

    # Top subplot: Attentional Actions
    plt.subplot(3, 1, 1)
    plt.plot(
        1.5 + np.arange(0, T - 1),
        Pi2_bar[0, :-1],
        label=r"${\bar{u}}^{(2)}$",
        color="coral",
        linewidth=2,
    )
    plt.scatter(
        1 + np.arange(T),
        1 - u2,
        label="true action",
        color="orangered",
        s=20,
        alpha=0.8,
    )
    plt.legend(loc="lower right")
    plt.ylim([0, 1.0])
    plt.yticks([0, 1], ["switch", "stay"])
    plt.title("Second Level: Attentional Action")
    plt.xlim([0.5, T + 0.5])

    # Middle subplot: Attentional States
    plt.subplot(3, 1, 2)
    plt.plot(
        1 + np.arange(T),
        X2_bar[0, :],
        label=r"${\bar{s}}^{(2)}$",
        color="darkorange",
        linewidth=2,
    )
    plt.scatter(
        1 + np.arange(T),
        1 - x2,
        label="true state",
        color="sandybrown",
        s=20,
        alpha=0.8,
    )
    plt.ylim([-0.1, 1.1])
    plt.yticks([0, 1], ["distracted", "focused"])
    plt.title("Second Level: Attentional State")
    plt.legend(loc="lower right")
    plt.xlim([0.5, T + 0.5])

    # Bottom subplot: Perceptual States
    plt.subplot(3, 1, 3)
    plt.plot(
        1 + np.arange(T),
        X1_bar[0, :],
        label=r"${\bar{s}}^{(1)}$",
        color="royalblue",
        linewidth=2,
    )
    plt.scatter(
        1 + np.arange(T), 1 - O, label="true state", color="steelblue", s=20, alpha=0.8
    )
    plt.yticks([0, 1], ["deviant", "standard"])
    plt.xlabel(r"time ($\tau$)")
    plt.title("First Level: Perceptual State")
    plt.legend(loc="lower right")
    plt.ylim([-0.1, 1.1])
    plt.xlim([0.5, T + 0.5])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")

    return fig


def plot_figure_11(
    results: Dict[str, Any], save_path: Optional[str] = None
) -> plt.Figure:
    """
    Generate Figure 11: Three-level model with meta-awareness.

    Args:
        results: Dictionary containing simulation results
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    setup_matplotlib_style()

    X3_bar = results["X3_bar"]
    x3 = results["x3"]
    X2_bar = results["X2_bar"]
    x2 = results["x2"]
    X1_bar = results["X1_bar"]
    O = results["O"]
    T = results["T"]

    fig = plt.figure(figsize=(10, 6))

    # Top subplot: Meta-awareness States
    plt.subplot(3, 1, 1)
    plt.plot(
        1 + np.arange(T),
        X3_bar[0, :],
        label=r"${\bar{s}}^{(3)}$",
        color="darkgreen",
        linewidth=2,
    )
    plt.scatter(
        1 + np.arange(T), 1 - x3, label="true state", color="limegreen", s=20, alpha=0.8
    )
    plt.ylim([-0.1, 1.1])
    plt.yticks([0, 1], ["low meta-awareness", "high meta-awareness"])
    plt.title("Third Level: Meta-awareness State")
    plt.legend(loc="lower right")
    plt.xlim([0.5, T + 0.5])

    # Middle subplot: Attentional States
    plt.subplot(3, 1, 2)
    plt.plot(
        1 + np.arange(T),
        X2_bar[0, :],
        label=r"${\bar{s}}^{(2)}$",
        color="darkorange",
        linewidth=2,
    )
    plt.scatter(
        1 + np.arange(T),
        1 - x2,
        label="true state",
        color="sandybrown",
        s=20,
        alpha=0.8,
    )
    plt.ylim([-0.1, 1.1])
    plt.yticks([0, 1], ["distracted", "focused"])
    plt.title("Second Level: Attentional State")
    plt.legend(loc="lower right")
    plt.xlim([0.5, T + 0.5])

    # Bottom subplot: Perceptual States
    plt.subplot(3, 1, 3)
    plt.plot(
        1 + np.arange(T),
        X1_bar[0, :],
        label=r"${\bar{s}}^{(1)}$",
        color="royalblue",
        linewidth=2,
    )
    plt.scatter(
        1 + np.arange(T), 1 - O, label="true state", color="steelblue", s=20, alpha=0.8
    )
    plt.yticks([0, 1], ["deviant", "standard"])
    plt.xlabel(r"time ($\tau$)")
    plt.title("First Level: Perceptual State")
    plt.legend(loc="lower right")
    plt.ylim([-0.1, 1.1])
    plt.xlim([0.5, T + 0.5])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")

    return fig


def plot_precision_dynamics(
    results: Dict[str, Any], save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot precision dynamics over time (additional analysis figure).

    Args:
        results: Dictionary containing simulation results
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    setup_matplotlib_style()

    gamma_A1 = results["gamma_A1"]
    gamma_A2 = results.get("gamma_A2", None)
    T = results["T"]

    fig = plt.figure(figsize=(8, 4))

    if gamma_A2 is not None:
        # Three-level model
        plt.subplot(2, 1, 1)
        plt.plot(
            1 + np.arange(T),
            gamma_A2,
            label=r"$\gamma_{A_2}$ (attention precision)",
            color="orange",
            linewidth=2,
        )
        plt.ylabel("Precision")
        plt.title("Level 2: Attentional Precision")
        plt.legend()
        plt.xlim([0.5, T + 0.5])

        plt.subplot(2, 1, 2)
    else:
        plt.subplot(1, 1, 1)

    plt.plot(
        1 + np.arange(T),
        gamma_A1,
        label=r"$\gamma_{A_1}$ (perceptual precision)",
        color="blue",
        linewidth=2,
    )
    plt.xlabel(r"time ($\tau$)")
    plt.ylabel("Precision")
    plt.title("Level 1: Perceptual Precision")
    plt.legend()
    plt.xlim([0.5, T + 0.5])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")

    return fig


def plot_free_energy_dynamics(
    results: Dict[str, Any], save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot free energy dynamics over time (additional analysis figure).

    Args:
        results: Dictionary containing simulation results
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    setup_matplotlib_style()

    G2 = results["G2"]
    F2 = results["F2"]
    T = results["T"]

    fig = plt.figure(figsize=(8, 4))

    # Expected Free Energy
    plt.subplot(2, 1, 1)
    plt.plot(
        1 + np.arange(T - 1), G2[0, :-1], label="stay policy", color="blue", linewidth=2
    )
    plt.plot(
        1 + np.arange(T - 1),
        G2[1, :-1],
        label="switch policy",
        color="red",
        linewidth=2,
    )
    plt.ylabel("Expected Free Energy")
    plt.title("Expected Free Energy by Policy")
    plt.legend()
    plt.xlim([0.5, T - 0.5])

    # Variational Free Energy
    plt.subplot(2, 1, 2)
    plt.plot(
        1 + np.arange(T - 1), F2[0, :-1], label="stay policy", color="blue", linewidth=2
    )
    plt.plot(
        1 + np.arange(T - 1),
        F2[1, :-1],
        label="switch policy",
        color="red",
        linewidth=2,
    )
    plt.xlabel(r"time ($\tau$)")
    plt.ylabel("Variational Free Energy")
    plt.title("Variational Free Energy by Policy")
    plt.legend()
    plt.xlim([0.5, T - 0.5])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches="tight")

    return fig


def save_all_figures(results: Dict[str, Any], output_dir: str = "figures") -> None:
    """
    Save all figures to specified directory.

    Args:
        results: Dictionary containing simulation results
        output_dir: Directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving figures to {output_dir}/")

    # Main paper figures
    plot_figure_7(results, os.path.join(output_dir, "figure_7.png"))
    plot_figure_10(results, os.path.join(output_dir, "figure_10.png"))

    if "X3_bar" in results:  # Three-level model
        plot_figure_11(results, os.path.join(output_dir, "figure_11.png"))

    # Additional analysis figures
    plot_precision_dynamics(results, os.path.join(output_dir, "precision_dynamics.png"))
    plot_free_energy_dynamics(
        results, os.path.join(output_dir, "free_energy_dynamics.png")
    )

    print("All figures saved successfully!")


def display_results_summary(results: Dict[str, Any]) -> None:
    """
    Display a summary of simulation results.

    Args:
        results: Dictionary containing simulation results
    """
    T = results["T"]

    print("\n" + "=" * 60)
    print("SANDVED-SMITH ET AL. (2021) SIMULATION RESULTS")
    print("=" * 60)

    print(f"Simulation duration: {T} time steps")
    print(
        f"Model levels: {'3 (with meta-awareness)' if 'X3_bar' in results else '2 (attention only)'}"
    )

    # Attentional state statistics
    x2 = results["x2"]
    focused_time = np.sum(x2 == 0) / T * 100
    distracted_time = np.sum(x2 == 1) / T * 100

    print("\nAttentional State Distribution:")
    print(f"  - Focused: {focused_time:.1f}%")
    print(f"  - Distracted: {distracted_time:.1f}%")

    # Mind-wandering episodes
    state_changes = np.diff(x2)
    distractions = np.sum(state_changes == 1)  # Transitions to distracted
    refocuses = np.sum(state_changes == -1)  # Transitions to focused

    print("\nMind-wandering Episodes:")
    print(f"  - Number of distractions: {distractions}")
    print(f"  - Number of refocuses: {refocuses}")

    if distractions > 0:
        # Average mind-wandering duration
        distracted_periods = []
        in_distraction = False
        start_time = 0

        for t, state in enumerate(x2):
            if state == 1 and not in_distraction:  # Start of distraction
                in_distraction = True
                start_time = t
            elif state == 0 and in_distraction:  # End of distraction
                in_distraction = False
                distracted_periods.append(t - start_time)

        if distracted_periods:
            avg_distraction = np.mean(distracted_periods)
            print(
                f"  - Average mind-wandering duration: {avg_distraction:.1f} time steps"
            )

    # Precision statistics
    gamma_A1 = results["gamma_A1"]
    print("\nPerceptual Precision (γ_A1):")
    print(f"  - Mean: {np.mean(gamma_A1):.3f}")
    print(f"  - Range: [{np.min(gamma_A1):.3f}, {np.max(gamma_A1):.3f}]")

    if "gamma_A2" in results:
        gamma_A2 = results["gamma_A2"]
        print("\nAttentional Precision (γ_A2):")
        print(f"  - Mean: {np.mean(gamma_A2):.3f}")
        print(f"  - Range: [{np.min(gamma_A2):.3f}, {np.max(gamma_A2):.3f}]")

    print("\n" + "=" * 60)
