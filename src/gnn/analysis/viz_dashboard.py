#!/usr/bin/env python3
"""
Cross-framework dashboards and comparison plots for GNN Step 16 analysis visualizations.

Extracted from ``analysis.visualizations``.
"""

import json
import logging
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
)

from .viz_base import (
    MATPLOTLIB_AVAILABLE,
    np,
    plt,
    safe_savefig,
)
from .viz_schema import _normalize_framework_name

logger = logging.getLogger(__name__)


def generate_unified_framework_dashboard(
    framework_data: Dict[str, Dict[str, Any]],
    output_dir: Path,
    model_name: str = "Active Inference Model",
) -> List[str]:
    """
    Generate comprehensive unified dashboard comparing all frameworks.

    Creates a multi-panel visualization that directly compares:
    - Belief evolution trajectories across all frameworks
    - Action selection patterns
    - Expected free energy dynamics
    - Key performance metrics

    Args:
        framework_data: Dictionary mapping framework keys to their data
        output_dir: Directory to save visualizations
        model_name: Model name for titles

    Returns:
        List of generated file paths
    """
    generated_files: list[Any] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract data per framework
    framework_beliefs: dict[Any, Any] = {}
    framework_actions: dict[Any, Any] = {}
    framework_efe: dict[Any, Any] = {}
    framework_vfe: dict[Any, Any] = {}
    framework_metrics: dict[Any, Any] = {}

    for _key, data in framework_data.items():
        framework = data.get("framework", "unknown")
        sim_data = data.get("simulation_data", {})

        # Try to get from results if not in simulation_data
        if not sim_data and data.get("results"):
            result = data["results"][0]
            sim_data = result.get("simulation_data", {})

        if sim_data:
            if sim_data.get("beliefs"):
                framework_beliefs[framework] = np.array(sim_data["beliefs"])
            if sim_data.get("actions"):
                framework_actions[framework] = sim_data["actions"]
            if sim_data.get("efe_history") or sim_data.get("expected_free_energy"):
                efe_data = (
                    sim_data.get("efe_history")
                    or sim_data.get("expected_free_energy")
                    or []
                )
                if efe_data:
                    efe_arr = np.array(efe_data)
                    if efe_arr.ndim == 2:
                        efe_data = np.mean(efe_arr, axis=1).tolist()
                    framework_efe[framework] = efe_data

            if sim_data.get("vfe_history") or sim_data.get("variational_free_energy"):
                vfe_data = (
                    sim_data.get("vfe_history")
                    or sim_data.get("variational_free_energy")
                    or []
                )
                if vfe_data:
                    framework_vfe[framework] = vfe_data

            # Collect metrics
            framework_metrics[framework] = {
                "num_timesteps": len(sim_data.get("beliefs", []))
                or len(sim_data.get("actions", [])),
                "num_states": len(sim_data["beliefs"][0])
                if sim_data.get("beliefs")
                else 0,
                "unique_actions": len(set(sim_data.get("actions", []))),
            }

    # === Dashboard 1: Belief Evolution Comparison ===
    if len(framework_beliefs) >= 2:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        colors = plt.get_cmap("tab10")(np.linspace(0, 1, 10))
        _frameworks = list(framework_beliefs.keys())

        # Individual framework belief plots
        for idx, (fw, beliefs) in enumerate(framework_beliefs.items()):
            if idx >= 5:
                break
            ax = axes[idx]
            n_states = beliefs.shape[1] if beliefs.ndim > 1 else 1

            for state_idx in range(min(n_states, 5)):
                if beliefs.ndim > 1:
                    ax.plot(
                        beliefs[:, state_idx],
                        label=f"State {state_idx + 1}",
                        color=colors[state_idx],
                        linewidth=1.5,
                    )
                else:
                    ax.plot(beliefs, label="Belief", linewidth=1.5)

            ax.set_title(f"{fw.upper()}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Probability")
            ax.set_ylim(0, 1.05)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

        # Combined comparison in last panel
        ax_combined = axes[5]
        for fw_idx, (fw, beliefs) in enumerate(framework_beliefs.items()):
            if beliefs.ndim > 1:
                dominant_belief = np.max(beliefs, axis=1)
                ax_combined.plot(
                    dominant_belief,
                    label=f"{fw}",
                    linewidth=2,
                    linestyle=[
                        "solid",
                        "dashed",
                        "dotted",
                        "dashdot",
                        (0, (3, 5, 1, 5)),
                    ][fw_idx % 5],
                )

        ax_combined.set_title(
            "Dominant Belief Confidence", fontsize=12, fontweight="bold"
        )
        ax_combined.set_xlabel("Time Step")
        ax_combined.set_ylabel("Max Probability")
        ax_combined.legend(loc="best")
        ax_combined.grid(True, alpha=0.3)

        plt.suptitle(
            f"Belief Evolution Comparison - {model_name}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        belief_file = output_dir / "unified_belief_comparison.png"
        saved = safe_savefig(belief_file, log=logger)
        if saved:
            generated_files.append(saved)

    # === Dashboard 2: Action & EFE Comparison ===
    n_panels = max(len(framework_actions), len(framework_efe), 1)
    if framework_actions or framework_efe:
        fig, axes = plt.subplots(2, max(3, (n_panels + 1) // 2), figsize=(18, 10))

        # Action distribution comparison
        if framework_actions:
            all_action_set: set[Any] = set()
            for actions in framework_actions.values():
                all_action_set.update(actions)
            all_actions = sorted(all_action_set)

            ax_action = axes[0, 0] if axes.ndim > 1 else axes[0]
            bar_width = 0.8 / len(framework_actions)

            for fw_idx, (fw, actions) in enumerate(framework_actions.items()):
                action_counts = [actions.count(a) for a in all_actions]
                x_positions = np.arange(len(all_actions)) + fw_idx * bar_width
                ax_action.bar(
                    x_positions, action_counts, bar_width, label=fw, alpha=0.8
                )

            ax_action.set_xlabel("Action")
            ax_action.set_ylabel("Count")
            ax_action.set_title("Action Distribution by Framework")
            ax_action.set_xticks(
                np.arange(len(all_actions))
                + bar_width * (len(framework_actions) - 1) / 2
            )
            ax_action.set_xticklabels([f"A{a}" for a in all_actions])
            ax_action.legend()
            ax_action.grid(True, alpha=0.3, axis="y")

        # EFE evolution comparison
        if framework_efe:
            ax_efe = axes[0, 1] if axes.ndim > 1 else axes[1]

            for fw, efe_values in framework_efe.items():
                ax_efe.plot(efe_values, label=fw, linewidth=2)

            ax_efe.set_xlabel("Time Step")
            ax_efe.set_ylabel("Expected Free Energy")
            ax_efe.set_title("EFE Evolution by Framework")
            ax_efe.legend()
            ax_efe.grid(True, alpha=0.3)

        # Metrics summary table
        ax_table = axes[1, 0] if axes.ndim > 1 else axes[2]
        ax_table.axis("off")

        if framework_metrics:
            table_data: list[Any] = []
            headers: list[Any] = ["Framework", "Timesteps", "States", "Actions Used"]

            for fw, metrics in framework_metrics.items():
                table_data.append(
                    [
                        fw.upper(),
                        str(metrics.get("num_timesteps", "N/A")),
                        str(metrics.get("num_states", "N/A")),
                        str(metrics.get("unique_actions", "N/A")),
                    ]
                )

            table = ax_table.table(
                cellText=table_data, colLabels=headers, loc="center", cellLoc="center"
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            ax_table.set_title(
                "Framework Metrics Summary", fontsize=12, fontweight="bold", pad=20
            )

        # Hide unused axes
        for idx in range(2, axes.shape[1] if axes.ndim > 1 else 1):
            for row in range(2):
                if axes.ndim > 1:
                    axes[row, idx].axis("off")

        plt.suptitle(
            f"Action & EFE Comparison - {model_name}", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()

        action_efe_file = output_dir / "unified_action_efe_comparison.png"
        saved = safe_savefig(action_efe_file, log=logger)
        if saved:
            generated_files.append(saved)

    # === Dashboard 3: Belief Entropy Comparison ===
    if len(framework_beliefs) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Calculate entropy for each framework
        framework_entropy: dict[Any, Any] = {}
        for fw, beliefs in framework_beliefs.items():
            if beliefs.ndim > 1:
                entropy: list[Any] = []
                for t in range(len(beliefs)):
                    p = np.clip(beliefs[t], 1e-10, 1.0)
                    p = p / np.sum(p)
                    entropy.append(-np.sum(p * np.log(p)))
                framework_entropy[fw] = entropy

        # Plot entropy trajectories
        ax1 = axes[0]
        for fw, entropy in framework_entropy.items():
            ax1.plot(entropy, label=fw, linewidth=2)

        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Belief Entropy (nats)")
        ax1.set_title("Belief Uncertainty Over Time")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Box plot comparison
        ax2 = axes[1]
        entropy_data = list(framework_entropy.values())
        labels = list(framework_entropy.keys())

        bp = ax2.boxplot(entropy_data, tick_labels=labels, patch_artist=True)
        colors = plt.get_cmap("Set2")(np.linspace(0, 1, len(labels)))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)

        ax2.set_xlabel("Framework")
        ax2.set_ylabel("Entropy Distribution")
        ax2.set_title("Entropy Statistics by Framework")
        ax2.grid(True, alpha=0.3, axis="y")

        plt.suptitle(
            f"Belief Entropy Analysis - {model_name}", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()

        entropy_file = output_dir / "unified_entropy_comparison.png"
        saved = safe_savefig(entropy_file, log=logger)
        if saved:
            generated_files.append(saved)

    return generated_files


def generate_cross_framework_comparison(
    framework_data: Dict[str, Dict[str, Any]], output_path: Path
) -> str:
    """
    Generate cross-framework comparison visualization.

    Args:
        framework_data: Dictionary mapping framework keys to their data
        output_path: Path to save the visualization

    Returns:
        Path to the generated file
    """
    # Aggregate by UNIQUE framework name to avoid duplicates
    aggregated: Dict[str, Dict[str, Any]] = {}

    for _key, data in framework_data.items():
        framework = data.get("framework", "unknown")

        if framework not in aggregated:
            aggregated[framework] = {
                "execution_times": [],
                "steps_completed": [],
                "success_count": 0,
                "total_count": 0,
            }

        agg = aggregated[framework]
        agg["total_count"] += 1

        # Extract execution time from results
        results = data.get("results", [])
        if results:
            result = results[0]
            exec_time = result.get("execution_time", 0)
            if exec_time:
                agg["execution_times"].append(exec_time)
            if result.get("success", False):
                agg["success_count"] += 1

        # Extract steps_completed from simulation data
        sim_data = data.get("simulation_data", {})
        if not sim_data and results:
            sim_data = results[0].get("simulation_data", {})

        steps = 0
        if sim_data:
            beliefs = sim_data.get("beliefs", [])
            actions = sim_data.get("actions", [])
            observations = sim_data.get("observations", [])
            steps = max(len(beliefs), len(actions), len(observations))

        if steps > 0:
            agg["steps_completed"].append(steps)

    if not aggregated:
        raise ValueError("No framework data for comparison")

    # Build final metrics lists
    frameworks = sorted(aggregated.keys())
    metrics: dict[str, Any] = {
        "execution_time": [],
        "steps_completed": [],
        "success_rate": [],
    }

    for fw in frameworks:
        agg = aggregated[fw]
        if agg["execution_times"]:
            metrics["execution_time"].append(
                sum(agg["execution_times"]) / len(agg["execution_times"])
            )
        else:
            metrics["execution_time"].append(0)

        if agg["steps_completed"]:
            metrics["steps_completed"].append(max(agg["steps_completed"]))
        else:
            metrics["steps_completed"].append(0)

        if agg["total_count"] > 0:
            metrics["success_rate"].append(agg["success_count"] / agg["total_count"])
        else:
            metrics["success_rate"].append(0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = plt.get_cmap("Set3")(np.linspace(0, 1, len(frameworks)))

    # Execution time comparison
    ax1 = axes[0]
    _bars = ax1.bar(range(len(frameworks)), metrics["execution_time"], color=colors)
    ax1.set_xlabel("Framework")
    ax1.set_ylabel("Execution Time (s)")
    ax1.set_title("Execution Time Comparison")
    ax1.set_xticks(range(len(frameworks)))
    ax1.set_xticklabels(frameworks, rotation=45, ha="right")

    # Steps completed comparison
    ax2 = axes[1]
    ax2.bar(range(len(frameworks)), metrics["steps_completed"], color=colors)
    ax2.set_xlabel("Framework")
    ax2.set_ylabel("Steps Completed")
    ax2.set_title("Simulation Steps Comparison")
    ax2.set_xticks(range(len(frameworks)))
    ax2.set_xticklabels(frameworks, rotation=45, ha="right")

    # Success rate comparison
    ax3 = axes[2]
    success_colors = [
        "green" if s >= 1.0 else ("orange" if s > 0 else "red")
        for s in metrics["success_rate"]
    ]
    ax3.bar(
        range(len(frameworks)), metrics["success_rate"], color=success_colors, alpha=0.7
    )
    ax3.set_xlabel("Framework")
    ax3.set_ylabel("Success Rate")
    ax3.set_title("Execution Success Comparison")
    ax3.set_xticks(range(len(frameworks)))
    ax3.set_xticklabels(frameworks, rotation=45, ha="right")
    ax3.set_ylim(-0.1, 1.1)

    plt.suptitle("Cross-Framework Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    saved = safe_savefig(output_path, log=logger)
    return saved or str(output_path)


def generate_efe_convergence_comparison(
    framework_data: Dict[str, Dict[str, Any]], output_path: Path
) -> List[str]:
    """
    Generate an overlay plot comparing Expected Free Energy convergence across frameworks.

    Compares EFE trajectories from frameworks that provide efe_history (JAX, PyMDP).

    Args:
        framework_data: Dictionary of framework data keyed by framework_modelname
        output_path: Path to save the visualization

    Returns:
        List of generated file paths
    """
    if not MATPLOTLIB_AVAILABLE:
        return []

    # Collect EFE data per framework
    efe_series: dict[Any, Any] = {}
    colors: dict[str, Any] = {
        "jax": "#E74C3C",
        "pymdp": "#3498DB",
        "rxinfer": "#2ECC71",
        "activeinference_jl": "#9B59B6",
        "discopy": "#F39C12",
    }

    for _key, data in framework_data.items():
        framework = data.get("framework", "unknown")
        sim_data = data.get("simulation_data", {})

        # Check simulation_trace and metrics for EFE
        efe = (
            sim_data.get("simulation_trace", {}).get("efe_history")
            or sim_data.get("metrics", {}).get("expected_free_energy")
            or sim_data.get("efe_history", [])
        )

        if efe and len(efe) > 1:
            try:
                efe_arr = np.array(efe)
                if efe_arr.ndim == 2:
                    # Take mean across action/policy dimension to get 1D sequence
                    efe_1d = np.mean(efe_arr, axis=1)
                else:
                    efe_1d = efe_arr
                efe_series[framework] = efe_1d
            except Exception as e:
                logger.warning(f"Error flattening EFE array for {framework}: {e}")

    if len(efe_series) < 2:
        logger.debug("Not enough frameworks with EFE data for comparison")
        return []

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Raw EFE trajectories
    for fw, efe in efe_series.items():
        color = colors.get(fw, "#7F8C8D")
        ax1.plot(
            efe,
            "o-",
            label=fw.upper(),
            color=color,
            linewidth=2,
            markersize=5,
            alpha=0.8,
        )

    ax1.set_xlabel("Time Step", fontweight="bold")
    ax1.set_ylabel("Expected Free Energy", fontweight="bold")
    ax1.set_title("EFE Convergence Comparison", fontweight="bold", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right: Cumulative/running mean EFE
    for fw, efe in efe_series.items():
        color = colors.get(fw, "#7F8C8D")
        running_mean = np.cumsum(efe) / (np.arange(len(efe)) + 1)
        ax2.plot(
            running_mean, "-", label=f"{fw.upper()} (mean)", color=color, linewidth=2.5
        )
        ax2.fill_between(
            range(len(running_mean)), running_mean, alpha=0.15, color=color
        )

    ax2.set_xlabel("Time Step", fontweight="bold")
    ax2.set_ylabel("Running Mean EFE", fontweight="bold")
    ax2.set_title("EFE Running Mean", fontweight="bold", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Cross-Framework EFE Analysis", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    saved = safe_savefig(output_path, log=logger)
    if saved:
        logger.info(f"Generated EFE convergence comparison: {output_path.name}")
        return [str(output_path)]
    return []


def generate_confidence_comparison(
    framework_data: Dict[str, Dict[str, Any]], output_path: Path
) -> List[str]:
    """
    Generate a comparison of belief confidence convergence across frameworks.

    Uses belief_confidence directly when available, or derives it from beliefs
    by taking max probability per timestep.

    Args:
        framework_data: Dictionary of framework data
        output_path: Path to save the visualization

    Returns:
        List of generated file paths
    """
    if not MATPLOTLIB_AVAILABLE:
        return []

    confidence_series: dict[Any, Any] = {}
    colors: dict[str, Any] = {
        "jax": "#E74C3C",
        "pymdp": "#3498DB",
        "rxinfer": "#2ECC71",
        "activeinference_jl": "#9B59B6",
        "discopy": "#F39C12",
    }

    for _key, data in framework_data.items():
        framework = data.get("framework", "unknown")
        sim_data = data.get("simulation_data", {})

        # Direct confidence data
        confidence = (
            sim_data.get("simulation_trace", {}).get("belief_confidence")
            or sim_data.get("metrics", {}).get("belief_confidence")
            or sim_data.get("belief_confidence", [])
        )

        # Derive from beliefs if not available directly
        if not confidence:
            beliefs = sim_data.get("beliefs", [])
            if beliefs and isinstance(beliefs[0], list):
                confidence = [max(b) for b in beliefs]

        if confidence and len(confidence) > 1:
            confidence_series[framework] = confidence

    if len(confidence_series) < 2:
        logger.debug("Not enough frameworks with confidence data for comparison")
        return []

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Confidence over time
    for fw, conf in confidence_series.items():
        color = colors.get(fw, "#7F8C8D")
        ax1.plot(conf, "o-", label=fw.upper(), color=color, linewidth=2, markersize=5)

    ax1.set_xlabel("Time Step", fontweight="bold")
    ax1.set_ylabel("Max Belief Probability", fontweight="bold")
    ax1.set_title("Belief Confidence Over Time", fontweight="bold", fontsize=13)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1 / 3, color="gray", linestyle=":", alpha=0.5, label="Chance level")

    # Right: Entropy over time (inverse of confidence)
    for fw, conf in confidence_series.items():
        color = colors.get(fw, "#7F8C8D")
        # Reconstruct entropy from max confidence (approximation)
        entropy = [-np.log2(c + 1e-10) for c in conf]
        ax2.plot(entropy, "-", label=fw.upper(), color=color, linewidth=2)
        ax2.fill_between(range(len(entropy)), entropy, alpha=0.1, color=color)

    ax2.set_xlabel("Time Step", fontweight="bold")
    ax2.set_ylabel("Belief Uncertainty (-log₂ confidence)", fontweight="bold")
    ax2.set_title("Uncertainty Reduction Over Time", fontweight="bold", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        "Cross-Framework Confidence Analysis", fontsize=15, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    saved = safe_savefig(output_path, log=logger)
    if saved:
        logger.info(f"Generated confidence comparison: {output_path.name}")
        return [str(output_path)]
    return []


def generate_framework_radar(
    exec_summary_path: Path,
    framework_data: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> List[str]:
    """
    Generate a radar/spider chart comparing frameworks across multiple dimensions.

    Dimensions: Execution Speed, Data Richness, Belief Quality, Timesteps, Validation.

    Args:
        exec_summary_path: Path to execution_summary.json
        framework_data: Dictionary of framework data
        output_path: Path to save the visualization

    Returns:
        List of generated file paths
    """
    if not MATPLOTLIB_AVAILABLE:
        return []

    try:
        with open(exec_summary_path, "r") as f:
            exec_summary = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load execution summary: {e}")
        return []

    # Collect per-framework metrics
    fw_metrics: dict[Any, Any] = {}
    exec_details = exec_summary.get("execution_details", [])

    for detail in exec_details:
        fw = detail.get("framework", "unknown")
        fw_norm = _normalize_framework_name(fw)
        exec_time = detail.get("execution_time", detail.get("duration_seconds", 0))
        success = 1.0 if detail.get("success", False) else 0.0

        # Count data fields from framework_data
        data_richness = 0
        belief_quality = 0.0
        timesteps = 0
        for _key, data in framework_data.items():
            if data.get("framework") == fw_norm:
                sim = data.get("simulation_data", {})
                # Count non-empty data fields
                for field in [
                    "beliefs",
                    "actions",
                    "observations",
                    "true_states",
                    "free_energy",
                    "efe_history",
                ]:
                    if sim.get(field):
                        data_richness += 1
                # Check simulation_trace too
                trace = sim.get("simulation_trace", {})
                for field in ["beliefs", "actions", "efe_history", "belief_confidence"]:
                    if trace.get(field):
                        data_richness += 1
                # Metrics fields
                metrics = sim.get("metrics", {})
                for field in [
                    "expected_free_energy",
                    "belief_confidence",
                    "cumulative_preference",
                ]:
                    if metrics.get(field):
                        data_richness += 1
                # Belief quality: does beliefs sum to ~1?
                beliefs = sim.get("beliefs", [])
                if beliefs and isinstance(beliefs[0], list):
                    timesteps = len(beliefs)
                    final_conf = max(beliefs[-1]) if beliefs[-1] else 0
                    belief_quality = final_conf
                # Validation
                validation = sim.get("validation", {})
                if validation.get("all_beliefs_valid"):
                    belief_quality = max(belief_quality, 0.8)
                break

        fw_metrics[fw_norm] = {
            "speed": max(0, 1 - exec_time / 25),  # Normalized: 25s = 0, 0s = 1
            "data_richness": min(data_richness / 10, 1.0),  # Normalize to [0, 1]
            "belief_quality": belief_quality,
            "timesteps": min(timesteps / 20, 1.0),  # Normalize: 20 steps = 1.0
            "validation": success,
        }

    if len(fw_metrics) < 2:
        return []

    # Build radar chart
    categories: list[Any] = [
        "Speed",
        "Data Richness",
        "Belief Quality",
        "Timesteps",
        "Validation",
    ]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"polar": True})
    colors: dict[str, Any] = {
        "jax": "#E74C3C",
        "pymdp": "#3498DB",
        "rxinfer": "#2ECC71",
        "activeinference_jl": "#9B59B6",
        "discopy": "#F39C12",
    }

    for fw, metrics in fw_metrics.items():
        values: list[Any] = [
            metrics["speed"],
            metrics["data_richness"],
            metrics["belief_quality"],
            metrics["timesteps"],
            metrics["validation"],
        ]
        values += values[:1]  # Close the polygon
        color = colors.get(fw, "#7F8C8D")
        ax.plot(
            angles,
            values,
            "o-",
            linewidth=2.5,
            label=fw.upper(),
            color=color,
            markersize=7,
        )
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8, alpha=0.7)
    ax.set_title("Framework Capability Radar", fontsize=15, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    saved = safe_savefig(output_path, log=logger)
    if saved:
        logger.info(f"Generated framework radar: {output_path.name}")
        return [str(output_path)]
    return []
