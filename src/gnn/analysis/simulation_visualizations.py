#!/usr/bin/env python3
"""
Simulation-result and cross-framework metric visualizations for GNN Step 16 analysis.

Extracted from ``analysis.analyzer``.
"""

import importlib.util
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

import numpy as np

from gnn.visualization.compat.viz_compat import get_sns

from .result_adapter import categorical_trace, model_family, numeric_trace
from .viz_base import (
    MATPLOTLIB_AVAILABLE,
    plt,
)

logger = logging.getLogger(__name__)
sns: Any

SEABORN_AVAILABLE = importlib.util.find_spec("seaborn") is not None


def __getattr__(name: str) -> Any:
    if name == "sns":
        from gnn.visualization.compat.viz_compat import sns

        return sns
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def generate_matrix_visualizations(
    parsed_data: Dict[str, Any], output_dir: Path, model_name: str
) -> List[str]:
    """Generate heatmaps for model matrices."""
    visualizations: list[Any] = []
    if not MATPLOTLIB_AVAILABLE:
        return visualizations

    matrices = parsed_data.get("matrices", [])
    # Extract matrices from parsed data (A, B, C, D maps)
    for i, matrix_info in enumerate(matrices):
        matrix_data = matrix_info.get("data")
        if matrix_data is not None and isinstance(matrix_data, np.ndarray):
            matrix_name = matrix_info.get("name", f"matrix_{i}")
            plt.figure(figsize=(10, 8))

            sns = get_sns()
            if SEABORN_AVAILABLE and sns is not None:
                sns.heatmap(matrix_data, annot=matrix_data.size < 100, cmap="viridis")
            else:
                # Recovery to matplotlib imshow
                plt.imshow(matrix_data, cmap="viridis", aspect="auto")
                plt.colorbar()
                # Add annotations if small enough
                if matrix_data.size < 100:
                    for r in range(matrix_data.shape[0]):
                        for c in range(matrix_data.shape[1]):
                            plt.text(
                                c,
                                r,
                                f"{matrix_data[r, c]:.2g}",
                                ha="center",
                                va="center",
                                color="w",
                            )

            plt.title(f"{model_name} - {matrix_name}")
            plot_file = output_dir / f"{model_name}_{matrix_name}_heatmap.png"
            plt.savefig(plot_file, bbox_inches="tight")
            plt.close()
            visualizations.append(str(plot_file))
    return visualizations


def visualize_simulation_results(
    execution_results: Dict[str, Any], output_dir: Path
) -> List[str]:
    """Visualize actual simulation data from execution results."""
    visualizations: list[Any] = []
    if not MATPLOTLIB_AVAILABLE:
        return visualizations

    # Example: Visualize belief evolution if traces are present
    details = execution_results.get("execution_details", [])
    for detail in details:
        model_name = detail.get("model_name", "unknown")
        framework = detail.get("framework", "unknown")
        impl_dir = Path(detail.get("implementation_directory", ""))

        # Look for simulation data (e.g., traces.json or simulation_dump.json)
        trace_files = list(impl_dir.glob("**/traces.json")) + list(
            impl_dir.glob("**/simulation_data/*.json")
        )

        for trace_file in trace_files:
            try:
                with open(trace_file, "r") as f:
                    data = json.load(f)

                # Local Helper for attaching robust context
                def apply_chart_metadata(
                    framework: str = framework, data: dict[str, Any] = data
                ) -> None:
                    """Apply chart metadata."""
                    try:
                        meta_parts: list[Any] = [
                            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                            f"FW: {framework}",
                        ]
                        if "num_timesteps" in data:
                            meta_parts.append(f"T={data.get('num_timesteps')}")
                        elif "time_steps" in data:
                            meta_parts.append(f"T={data.get('time_steps')}")
                        if "model_parameters" in data and data["model_parameters"].get(
                            "num_states"
                        ):
                            meta_parts.append(
                                f"|S|={data['model_parameters']['num_states']}"
                            )

                        plt.figtext(
                            0.99,
                            0.01,
                            " | ".join(meta_parts),
                            ha="right",
                            va="bottom",
                            fontsize=8,
                            color="gray",
                            alpha=0.6,
                        )
                    except Exception as e:
                        logger.debug(f"Metadata application failed: {e}")

                # Plot traces (e.g., belief over time)
                if "beliefs" in data or "states" in data:
                    is_continuous = model_family(data) in {
                        "continuous",
                        "multi_agent_continuous",
                    }
                    is_posterior = "beliefs" in data
                    trace = data.get("beliefs", data.get("states", []))
                    beliefs = (
                        categorical_trace(trace, name="beliefs")
                        if is_posterior and not is_continuous
                        else numeric_trace(
                            trace, name="posterior means" if is_posterior else "states"
                        )
                    )
                    plt.figure(figsize=(10, 6))
                    if beliefs.ndim == 2:
                        for i in range(min(10, beliefs.shape[1])):
                            plt.plot(beliefs[:, i], label=f"State {i}")
                        plt.title(f"Belief Evolution - {model_name} ({framework})")
                        plt.legend()
                        fw_viz_dir = output_dir / framework
                        fw_viz_dir.mkdir(parents=True, exist_ok=True)
                        plot_file = (
                            fw_viz_dir / f"{model_name}_{framework}_belief_trace.png"
                        )
                        apply_chart_metadata()
                        plt.savefig(plot_file)
                        plt.close()
                        visualizations.append(str(plot_file))

                        # Posterior means are not categorical probabilities. This
                        # distance measures mean updates, not Gaussian uncertainty.
                        distances: list[float] = []
                        ylabel = (
                            "Posterior Mean Update (Euclidean Norm)"
                            if is_continuous and is_posterior
                            else "State Update (Euclidean Norm)"
                        )
                        if is_continuous or not is_posterior:
                            distances = np.linalg.norm(
                                np.diff(beliefs, axis=0), axis=1
                            ).tolist()
                        else:
                            try:
                                from scipy.spatial.distance import jensenshannon
                            except ImportError:
                                distances = np.linalg.norm(
                                    np.diff(beliefs, axis=0), axis=1
                                ).tolist()
                                ylabel = "Categorical Update (Euclidean Norm)"
                            else:
                                distances = [
                                    float(jensenshannon(p, q))
                                    for p, q in zip(
                                        beliefs[:-1], beliefs[1:], strict=True
                                    )
                                ]
                                ylabel = "Jensen-Shannon Distance"
                        if not np.isfinite(distances).all():
                            raise ValueError("Belief update distances must be finite")

                        if distances:
                            plt.figure(figsize=(10, 6))
                            plt.plot(
                                range(1, len(beliefs)),
                                distances,
                                label=(
                                    "Posterior Mean Update"
                                    if is_continuous and is_posterior
                                    else "Belief Update Magnitude"
                                    if is_posterior
                                    else "State Update"
                                ),
                                color="teal",
                            )
                            plt.title(
                                f"Belief Convergence Tracker - {model_name} ({framework})"
                            )
                            plt.xlabel("Timestep")
                            plt.ylabel(ylabel)
                            plt.legend()
                            fw_viz_dir = output_dir / framework
                            fw_viz_dir.mkdir(parents=True, exist_ok=True)
                            plot_file = (
                                fw_viz_dir
                                / f"{model_name}_{framework}_belief_convergence.png"
                            )
                            apply_chart_metadata()
                            plt.savefig(plot_file)
                            plt.close()
                            visualizations.append(str(plot_file))

                # Plot Free Energy trajectories
                if "free_energy" in data or "efe" in data or "F" in data:
                    plt.figure(figsize=(10, 6))
                    fe = data.get("free_energy", data.get("efe", data.get("F", [])))
                    if isinstance(fe, list) and len(fe) > 0:
                        plt.plot(
                            fe,
                            label="Free Energy",
                            color="purple",
                            marker="o",
                            markersize=4,
                        )
                        plt.title(
                            f"Free Energy Trajectory - {model_name} ({framework})"
                        )
                        plt.xlabel("Timestep")
                        plt.ylabel("Free Energy / Expected Free Energy")
                        plt.legend()
                        plot_file = (
                            output_dir / f"{model_name}_{framework}_free_energy.png"
                        )
                        apply_chart_metadata()
                        plt.savefig(plot_file)
                        plt.close()
                        visualizations.append(str(plot_file))

                # Plot Precision Dynamics
                if "precision" in data or "gamma" in data or "w" in data:
                    plt.figure(figsize=(10, 6))
                    prec = data.get("precision", data.get("gamma", data.get("w", [])))
                    if isinstance(prec, list) and len(prec) > 0:
                        plt.plot(
                            prec,
                            label="Precision Dynamics",
                            color="orange",
                            linestyle="--",
                            marker="x",
                        )
                        plt.title(f"Precision Dynamics - {model_name} ({framework})")
                        plt.xlabel("Timestep")
                        plt.ylabel("Precision (Gamma/w)")
                        plt.legend()
                        plot_file = (
                            output_dir / f"{model_name}_{framework}_precision.png"
                        )
                        apply_chart_metadata()
                        plt.savefig(plot_file)
                        plt.close()
                        visualizations.append(str(plot_file))

                # Plot Action History Frequency Maps
                if "actions" in data or "u" in data:
                    plt.figure(figsize=(8, 6))
                    actions = data.get("actions", data.get("u", []))
                    if isinstance(actions, list) and len(actions) > 0:
                        # flatten if nested
                        if isinstance(actions[0], list):
                            flat_actions = [a[0] if len(a) > 0 else 0 for a in actions]
                        else:
                            flat_actions = actions
                        unique_actions, counts = np.unique(
                            flat_actions, return_counts=True
                        )
                        plt.bar(
                            range(len(unique_actions)),
                            counts,
                            tick_label=[str(a) for a in unique_actions],
                            color="coral",
                            alpha=0.8,
                        )
                        plt.title(
                            f"Action Selection Frequencies - {model_name} ({framework})"
                        )
                        plt.xlabel("Action Variant")
                        plt.ylabel("Frequency")
                        fw_viz_dir = output_dir / framework
                        fw_viz_dir.mkdir(parents=True, exist_ok=True)
                        plot_file = (
                            fw_viz_dir
                            / f"{model_name}_{framework}_action_frequencies.png"
                        )
                        apply_chart_metadata()
                        plt.savefig(plot_file)
                        plt.close()
                        visualizations.append(str(plot_file))

            except Exception as e:
                logging.getLogger(__name__).debug(
                    f"Failed to visualize trace file {trace_file}: {e}"
                )
                continue

    return visualizations


def parse_matrix_data(matrix_str: str) -> Optional[np.ndarray]:
    """Parse matrix data from string representation."""
    try:
        # Simplified parsing logic for moving to analyzer
        import re

        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", matrix_str)
        if len(numbers) >= 1:
            return np.array([float(n) for n in numbers])
        return None
    except Exception as e:
        logger.debug(f"Failed to extract numeric array from value: {e}")
        return None


def visualize_cross_framework_metrics(
    comparison_data: Dict[str, Any],
    output_dir: Path,
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    """
    Generate visualizations comparing metrics across frameworks.

    Args:
        comparison_data: Output from analyze_framework_outputs()
        output_dir: Directory to save visualizations
        logger: Optional logger instance

    Returns:
        List of generated visualization file paths
    """
    import logging

    if logger is None:
        logger = logging.getLogger(__name__)

    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available, skipping visualizations")
        return []

    cross_fw_dir = output_dir / "cross_framework"
    cross_fw_dir.mkdir(parents=True, exist_ok=True)
    visualizations: list[Any] = []

    try:
        # Success rate comparison
        frameworks = list(comparison_data.get("frameworks", {}).keys())
        success_rates: list[Any] = []
        for framework in frameworks:
            data = comparison_data["frameworks"][framework]
            success_count = data.get("success_count", 0)
            total_count = data.get("total_count", 0)
            rate = (success_count / total_count * 100) if total_count > 0 else 0
            success_rates.append(rate)

        if frameworks and success_rates:
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(
                frameworks,
                success_rates,
                color=["#2ecc71" if r > 50 else "#e74c3c" for r in success_rates],
            )
            ax.set_ylabel("Success Rate (%)", fontweight="bold")
            ax.set_xlabel("Framework", fontweight="bold")
            ax.set_title(
                "Framework Execution Success Rates", fontweight="bold", fontsize=14
            )
            ax.set_ylim([0, 100])
            ax.grid(True, alpha=0.3, axis="y")

            # Fixed-point offsets keep labels inside the axis near its ceiling.
            for bar, rate in zip(bars, success_rates):
                height = bar.get_height()
                near_ceiling = height >= 90
                ax.annotate(
                    f"{rate:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2.0, height),
                    xytext=(0, -6 if near_ceiling else 6),
                    textcoords="offset points",
                    ha="center",
                    va="top" if near_ceiling else "bottom",
                    fontweight="bold",
                )

            plt.tight_layout()
            viz_file = cross_fw_dir / "framework_success_rates.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))

        # Performance comparison
        perf_comparison = comparison_data.get("comparisons", {}).get(
            "performance_comparison", {}
        )
        if perf_comparison:
            fig, ax = plt.subplots(figsize=(12, 6))
            frameworks = list(perf_comparison.keys())
            means = [perf_comparison[f]["mean"] for f in frameworks]
            stds = [perf_comparison[f]["std"] for f in frameworks]

            x_pos = np.arange(len(frameworks))
            bars = ax.bar(
                x_pos, means, yerr=stds, capsize=5, alpha=0.7, color="steelblue"
            )
            ax.set_ylabel("Execution Time (seconds)", fontweight="bold")
            ax.set_xlabel("Framework", fontweight="bold")
            ax.set_title(
                "Framework Execution Time Comparison", fontweight="bold", fontsize=14
            )
            ax.set_xticks(x_pos)
            ax.set_xticklabels(frameworks)
            ax.grid(True, alpha=0.3, axis="y")

            plt.tight_layout()
            viz_file = cross_fw_dir / "framework_performance_comparison.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))

    except Exception as e:
        logger.error(f"Error generating cross-framework visualizations: {e}")
        import traceback

        logger.debug(traceback.format_exc())

    return visualizations


# Explicit re-export surface (no_implicit_reexport).
__all__ = [
    "SEABORN_AVAILABLE",
    "generate_matrix_visualizations",
    "parse_matrix_data",
    "sns",
    "visualize_cross_framework_metrics",
    "visualize_simulation_results",
]
