"""
JAX Analysis Module

Per-framework analysis and visualization for JAX Active Inference simulations.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

logger = logging.getLogger(__name__)

# Import shared visualization utilities (centralized matplotlib setup)
from ..viz_base import MATPLOTLIB_AVAILABLE, np, plt


def parse_raw_output(raw_output: str) -> Dict[str, Any]:
    """
    Parse JAX simulation raw output text to extract additional data.

    The raw_output contains simulation details like:
    - Actions taken array
    - Final belief
    - Average EFE
    - EFE for all actions
    - Model shape information

    Args:
        raw_output: The raw stdout text from JAX execution

    Returns:
        Dictionary with extracted simulation data
    """
    extracted: dict[str, Any] = {
        "actions_from_output": [],
        "final_belief": [],
        "average_efe": None,
        "efe_all_actions": [],
        "model_shapes": {},
        "num_simulation_steps": 0,
    }

    if not raw_output:
        return extracted

    try:
        # Extract actions array: "Actions taken: [0 0 0 0 ... 0]"
        actions_match = re.search(r"Actions taken:\s*\[([^\]]+)\]", raw_output)
        if actions_match:
            actions_str = actions_match.group(1)
            extracted["actions_from_output"] = [int(x) for x in actions_str.split()]
            extracted["num_simulation_steps"] = len(extracted["actions_from_output"])

        # Extract final belief: "Final belief: [0. 0. 0.]"
        belief_match = re.search(r"Final belief:\s*\[([^\]]+)\]", raw_output)
        if belief_match:
            belief_str = belief_match.group(1)
            extracted["final_belief"] = [
                float(x.replace(".", "0.") if x == "." else x)
                for x in belief_str.split()
            ]

        # Extract average EFE: "Average EFE: 0.0003"
        avg_efe_match = re.search(r"Average EFE:\s*([\d.eE+-]+)", raw_output)
        if avg_efe_match:
            extracted["average_efe"] = float(avg_efe_match.group(1))

        # Extract EFE for all actions: "EFE for all actions: [0.00986223 0.00986223 0.00986223]"
        efe_all_match = re.search(r"EFE for all actions:\s*\[([^\]]+)\]", raw_output)
        if efe_all_match:
            efe_str = efe_all_match.group(1)
            extracted["efe_all_actions"] = [float(x) for x in efe_str.split()]

        # Extract model shapes
        shape_patterns: list[Any] = [
            (r"A matrix shape:\s*\((\d+),\s*(\d+)\)", "A_shape"),
            (r"B matrix shape:\s*\((\d+),\s*(\d+),\s*(\d+)\)", "B_shape"),
            (r"C vector shape:\s*\((\d+),\)", "C_shape"),
            (r"D vector shape:\s*\((\d+),\)", "D_shape"),
        ]
        for pattern, key in shape_patterns:
            match = re.search(pattern, raw_output)
            if match:
                extracted["model_shapes"][key] = tuple(int(x) for x in match.groups())

        # Extract number of states/actions/observations
        for key, pattern in [
            ("num_states", r"Number of states:\s*(\d+)"),
            ("num_observations", r"Number of observations:\s*(\d+)"),
            ("num_actions", r"Number of actions:\s*(\d+)"),
        ]:
            match = re.search(pattern, raw_output)
            if match:
                extracted["model_shapes"][key] = int(match.group(1))

    except Exception as e:
        logger.warning(f"Failed to parse JAX raw_output: {e}")

    return extracted


def load_simulation_results_json(jax_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Load structured simulation results JSON from JAX output directory.

    Args:
        jax_dir: Directory containing JAX execution outputs

    Returns:
        Parsed simulation results or None if not found
    """
    # Check multiple possible locations for simulation_results.json
    possible_paths: list[Any] = [
        jax_dir / "simulation_results.json",
        jax_dir / "simulation_data" / "simulation_results.json",
        jax_dir / "jax_outputs" / "simulation_results.json",
    ]

    for path in possible_paths:
        if path.exists():
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    logger.info(f"Loaded JAX simulation results from {path}")
                    return cast("dict[str, Any] | None", data)
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")

    return None


def generate_analysis_from_logs(
    execution_dir: Path, output_dir: Path, verbose: bool = False
) -> List[str]:
    """
    Generate analysis and visualizations from JAX execution logs.

    Args:
        execution_dir: Directory containing execution results
        output_dir: Directory to save visualizations
        verbose: Enable verbose logging

    Returns:
        List of generated visualization file paths
    """
    visualizations: list[Any] = []

    try:
        # Find JAX execution results
        jax_dirs = list(execution_dir.glob("*/jax"))

        for jax_dir in jax_dirs:
            model_name = jax_dir.parent.name

            # PRIORITY 1: Try to load structured simulation_results.json
            sim_results = load_simulation_results_json(jax_dir)
            if sim_results:
                logger.info(f"Using structured simulation results for {model_name}")
                viz_files = create_visualizations_from_structured_data(
                    sim_results, output_dir, model_name, verbose
                )
                visualizations.extend(viz_files)
                continue

            # PRIORITY 2: Fall back to parsing execution logs
            exec_logs_dir = jax_dir / "execution_logs"
            if exec_logs_dir.exists():
                # Load execution results
                results_files = list(exec_logs_dir.glob("*_results.json"))
                for results_file in results_files:
                    try:
                        with open(results_file, "r") as f:
                            data = json.load(f)

                        viz_files = create_jax_visualizations(
                            data, output_dir, model_name, verbose
                        )
                        visualizations.extend(viz_files)

                    except Exception as e:
                        logger.warning(f"Failed to process {results_file}: {e}")

    except Exception as e:
        logger.error(f"JAX analysis failed: {e}")

    return visualizations


def create_visualizations_from_structured_data(
    sim_results: Dict[str, Any],
    output_dir: Path,
    model_name: str,
    verbose: bool = False,
) -> List[str]:
    """
    Create rich visualizations from structured JAX simulation results JSON.

    Args:
        sim_results: Structured simulation results dictionary
        output_dir: Output directory
        model_name: Name of the model
        verbose: Enable verbose logging

    Returns:
        List of generated file paths
    """
    visualizations: list[Any] = []

    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available, skipping JAX visualizations")
        return visualizations

    output_dir.mkdir(parents=True, exist_ok=True)

    from gnn.analysis.result_adapter import model_family, result_views

    if model_family(sim_results) in {"continuous", "multi_agent_continuous"}:
        from gnn.analysis.rxinfer.family_visuals import (
            artifact_component,
            continuous_png,
        )

        return [
            continuous_png(
                view,
                output_dir
                / f"{model_name}_jax_{artifact_component(name)}_gaussian.png",
                f"{model_name}: {name}",
            )
            for name, view in result_views(sim_results).items()
        ]

    # Extract data from structured results
    trace = sim_results.get("simulation_trace", sim_results)
    beliefs = trace.get("beliefs", sim_results.get("beliefs", []))
    actions = trace.get("actions", sim_results.get("actions", []))
    efe_history = trace.get(
        "efe_history", sim_results.get("metrics", {}).get("expected_free_energy", [])
    )
    belief_confidence = trace.get(
        "belief_confidence", sim_results.get("metrics", {}).get("belief_confidence", [])
    )
    model_params = sim_results.get("model_parameters", {})

    # 1. Belief Evolution Heatmap
    if beliefs:
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

            beliefs_arr = np.array(beliefs)
            im = ax1.imshow(beliefs_arr.T, aspect="auto", cmap="Blues", origin="lower")
            ax1.set_xlabel("Time Step", fontweight="bold")
            ax1.set_ylabel("State", fontweight="bold")
            ax1.set_title(
                f"JAX Belief Evolution Heatmap - {model_name}",
                fontweight="bold",
                fontsize=14,
            )
            plt.colorbar(im, ax=ax1, label="Belief Probability")

            # Plot belief entropy
            entropies: list[Any] = []
            for b in beliefs:
                b_arr = np.array(b)
                entropy = -np.sum(b_arr * np.log(b_arr + 1e-10))
                entropies.append(entropy)
            ax2.plot(entropies, "o-", color="darkblue", linewidth=2, markersize=6)
            ax2.fill_between(range(len(entropies)), entropies, alpha=0.3, color="blue")
            ax2.set_xlabel("Time Step", fontweight="bold")
            ax2.set_ylabel("Belief Entropy (nats)", fontweight="bold")
            ax2.set_title("Belief Entropy Over Time", fontweight="bold")
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            viz_file = output_dir / f"{model_name}_jax_belief_evolution.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated belief evolution: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create belief evolution plot: {e}")

    # 2. Expected Free Energy Plot
    if efe_history:
        try:
            fig, ax = plt.subplots(figsize=(14, 6))

            # Collapse multi-dimensional EFE arrays (e.g. actions per timestep) into a 1D mean sequence
            efe_arr = np.array(efe_history)
            if efe_arr.ndim == 2:
                efe_1d = np.mean(efe_arr, axis=1)
            else:
                efe_1d = efe_arr

            x = range(len(efe_1d))
            ax.plot(x, efe_history, "o-", color="coral", linewidth=2, markersize=8)
            ax.fill_between(x, efe_1d, alpha=0.3, color="coral")
            ax.axhline(
                y=np.mean(efe_1d),
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean EFE: {np.mean(efe_1d):.4f}",
            )
            ax.set_xlabel("Time Step", fontweight="bold", fontsize=12)
            ax.set_ylabel("Expected Free Energy", fontweight="bold", fontsize=12)
            ax.set_title(
                f"JAX Expected Free Energy - {model_name}",
                fontweight="bold",
                fontsize=14,
            )
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)

            plt.tight_layout()
            viz_file = output_dir / f"{model_name}_jax_efe.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated EFE plot: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create EFE plot: {e}")

    # 3. Action Analysis (Distribution + Timeline)
    if actions:
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

            # Action distribution
            action_counts: dict[Any, Any] = {}
            for a in actions:
                action_counts[a] = action_counts.get(a, 0) + 1

            sorted_actions = sorted(action_counts.keys())
            counts = [action_counts[a] for a in sorted_actions]
            bars = ax1.bar(
                sorted_actions, counts, color="steelblue", alpha=0.8, edgecolor="navy"
            )
            ax1.set_xlabel("Action", fontweight="bold")
            ax1.set_ylabel("Count", fontweight="bold")
            ax1.set_title(
                f"Action Distribution ({len(actions)} steps)", fontweight="bold"
            )
            ax1.grid(True, alpha=0.3, axis="y")

            # Add count labels
            for bar, count in zip(bars, counts):
                # Offset in points, not data units, so the label gap is
                # independent of the count scale.
                ax1.annotate(
                    str(count),
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

            # Action timeline
            x = range(len(actions))
            ax2.step(x, actions, where="mid", linewidth=2, color="steelblue")
            ax2.scatter(x, actions, s=40, color="navy", zorder=5)
            ax2.set_xlabel("Time Step", fontweight="bold")
            ax2.set_ylabel("Action", fontweight="bold")
            ax2.set_title("Action Timeline", fontweight="bold")
            ax2.grid(True, alpha=0.3)
            ax2.set_yticks(sorted(set(actions)))

            plt.tight_layout()
            viz_file = output_dir / f"{model_name}_jax_actions.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated action analysis: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create action analysis: {e}")

    # 4. Belief Confidence Over Time
    if belief_confidence:
        try:
            fig, ax = plt.subplots(figsize=(14, 5))
            x = range(len(belief_confidence))
            ax.plot(
                x, belief_confidence, "o-", color="green", linewidth=2, markersize=6
            )
            ax.fill_between(x, belief_confidence, alpha=0.3, color="green")
            ax.set_xlabel("Time Step", fontweight="bold")
            ax.set_ylabel("Max Belief Probability", fontweight="bold")
            ax.set_title(
                f"JAX Belief Confidence - {model_name}", fontweight="bold", fontsize=14
            )
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            viz_file = output_dir / f"{model_name}_jax_confidence.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated confidence plot: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create confidence plot: {e}")

    # 5. Summary Dashboard
    try:
        fig = plt.figure(figsize=(16, 12))

        # Title
        fig.suptitle(
            f"JAX Active Inference Analysis - {model_name}",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        # Create 2x2 grid
        ax1 = fig.add_subplot(2, 2, 1)  # Beliefs heatmap
        ax2 = fig.add_subplot(2, 2, 2)  # EFE
        ax3 = fig.add_subplot(2, 2, 3)  # Actions
        ax4 = fig.add_subplot(2, 2, 4)  # Summary text

        # 1. Beliefs mini heatmap
        if beliefs:
            beliefs_arr = np.array(beliefs)
            im = ax1.imshow(beliefs_arr.T, aspect="auto", cmap="Blues", origin="lower")
            ax1.set_xlabel("Time Step")
            ax1.set_ylabel("State")
            ax1.set_title("Belief Evolution")
            plt.colorbar(im, ax=ax1)

        # 2. EFE mini plot
        if efe_history:
            ax2.plot(efe_history, "o-", color="coral", markersize=4)
            ax2.set_xlabel("Time Step")
            ax2.set_ylabel("EFE")
            ax2.set_title("Expected Free Energy")
            ax2.grid(True, alpha=0.3)

        # 3. Actions mini plot
        if actions:
            ax3.step(range(len(actions)), actions, where="mid", color="steelblue")
            ax3.set_xlabel("Time Step")
            ax3.set_ylabel("Action")
            ax3.set_title("Action Selection")
            ax3.grid(True, alpha=0.3)

        # 4. Summary statistics
        ax4.axis("off")
        # Pre-compute formatted values to avoid f-string issues
        avg_efe_str = f"{np.mean(efe_history):.6f}" if efe_history else "N/A"
        final_conf_str = f"{belief_confidence[-1]:.4f}" if belief_confidence else "N/A"
        timesteps_str = str(len(actions)) if actions else "N/A"
        diversity_str = f"{len(set(actions))} unique actions" if actions else "N/A"

        summary_text = f"""
JAX Active Inference Model Summary
{"=" * 40}

Model: {model_name}
Framework: JAX (Pure Functional)

Simulation Statistics:
  Timesteps: {timesteps_str}
  States: {model_params.get("num_states", "N/A")}
  Observations: {model_params.get("num_observations", "N/A")}
  Actions: {model_params.get("num_actions", "N/A")}

Performance Metrics:
  Avg EFE: {avg_efe_str}
  Final Confidence: {final_conf_str}
  Action Diversity: {diversity_str}

Validation:
  Beliefs Valid: {sim_results.get("validation", {}).get("all_beliefs_valid", "N/A")}
  Actions Valid: {sim_results.get("validation", {}).get("actions_in_range", "N/A")}
"""
        ax4.text(
            0.1,
            0.9,
            summary_text,
            transform=ax4.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox={"boxstyle": "round", "facecolor": "lightblue", "alpha": 0.3},
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        viz_file = output_dir / f"{model_name}_jax_dashboard.png"
        plt.savefig(viz_file, dpi=300, bbox_inches="tight")
        plt.close()
        visualizations.append(str(viz_file))
        logger.info(f"Generated summary dashboard: {viz_file.name}")
    except Exception as e:
        logger.warning(f"Failed to create summary dashboard: {e}")

    return visualizations


def create_jax_visualizations(
    data: Dict[str, Any], output_dir: Path, model_name: str, verbose: bool = False
) -> List[str]:
    """
    Create visualizations from JAX simulation data.

    Args:
        data: Execution results dictionary
        output_dir: Output directory
        model_name: Name of the model
        verbose: Enable verbose logging

    Returns:
        List of generated file paths
    """
    visualizations: list[Any] = []

    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available, skipping JAX visualizations")
        return visualizations

    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract simulation data from execution metadata
    sim_data = data.get("simulation_data", {})
    actions = sim_data.get("actions", [])
    free_energy = sim_data.get("free_energy", [])
    beliefs = sim_data.get("beliefs", [])

    # Parse raw_output for additional data
    raw_output = sim_data.get("raw_output", "")
    parsed = parse_raw_output(raw_output)

    # Use parsed actions if we have more data there
    if len(parsed["actions_from_output"]) > len(actions):
        actions = parsed["actions_from_output"]

    # 1. Free Energy Plot
    if free_energy:
        try:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(free_energy, "b-", linewidth=2, marker="o", markersize=8)
            ax.set_xlabel("Measurement Point", fontweight="bold")
            ax.set_ylabel("Expected Free Energy", fontweight="bold")
            ax.set_title(f"JAX Free Energy - {model_name}", fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.fill_between(range(len(free_energy)), free_energy, alpha=0.3)

            # Add average EFE annotation if available
            if parsed["average_efe"] is not None:
                ax.axhline(
                    y=parsed["average_efe"],
                    color="r",
                    linestyle="--",
                    label=f"Avg EFE: {parsed['average_efe']:.4f}",
                )
                ax.legend()

            viz_file = output_dir / f"{model_name}_jax_free_energy.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated free energy plot: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create free energy plot: {e}")

    # 2. Action Distribution (bar chart)
    if actions:
        try:
            fig, ax = plt.subplots(figsize=(10, 5))
            action_counts: dict[Any, Any] = {}
            for a in actions:
                action_counts[a] = action_counts.get(a, 0) + 1

            bars = ax.bar(
                list(action_counts.keys()),
                list(action_counts.values()),
                color="steelblue",
                alpha=0.7,
                edgecolor="navy",
            )
            ax.set_xlabel("Action", fontweight="bold")
            ax.set_ylabel("Count", fontweight="bold")
            ax.set_title(
                f"JAX Action Distribution ({len(actions)} steps) - {model_name}",
                fontweight="bold",
            )
            ax.grid(True, alpha=0.3, axis="y")

            # Add count labels on bars
            for bar, count in zip(bars, action_counts.values()):
                # Offset in points, not data units, so the label gap is
                # independent of the count scale.
                ax.annotate(
                    str(count),
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

            viz_file = output_dir / f"{model_name}_jax_action_dist.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated action distribution: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create action plot: {e}")

    # 3. Action Timeline (if we have step-by-step actions)
    if len(actions) > 1:
        try:
            fig, ax = plt.subplots(figsize=(14, 4))
            x = range(len(actions))
            ax.step(x, actions, where="mid", linewidth=2, color="steelblue")
            ax.scatter(x, actions, s=30, color="navy", zorder=5)
            ax.set_xlabel("Time Step", fontweight="bold")
            ax.set_ylabel("Action", fontweight="bold")
            ax.set_title(f"JAX Action Timeline - {model_name}", fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-0.5, len(actions) - 0.5)

            # Set y-ticks to integer actions
            unique_actions = sorted(set(actions))
            ax.set_yticks(unique_actions)

            viz_file = output_dir / f"{model_name}_jax_action_timeline.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated action timeline: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create action timeline: {e}")

    # 4. EFE Comparison (if we have EFE for all actions)
    if parsed["efe_all_actions"]:
        try:
            fig, ax = plt.subplots(figsize=(10, 5))
            efe_values = parsed["efe_all_actions"]
            x = range(len(efe_values))
            bars = ax.bar(x, efe_values, color="coral", alpha=0.7, edgecolor="darkred")
            ax.set_xlabel("Action", fontweight="bold")
            ax.set_ylabel("Expected Free Energy", fontweight="bold")
            ax.set_title(f"JAX EFE by Action - {model_name}", fontweight="bold")
            ax.grid(True, alpha=0.3, axis="y")
            ax.set_xticks(x)

            # Highlight minimum EFE action
            min_idx = np.argmin(efe_values)
            bars[min_idx].set_color("green")
            bars[min_idx].set_alpha(0.9)

            viz_file = output_dir / f"{model_name}_jax_efe_comparison.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated EFE comparison: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create EFE comparison: {e}")

    # 5. Model Summary Dashboard
    if parsed["model_shapes"]:
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.axis("off")

            # Build summary text
            shapes = parsed["model_shapes"]
            summary_lines: list[Any] = [
                "JAX Active Inference Model Summary",
                "─" * 40,
                "",
                f"Model: {model_name}",
                "Framework: JAX (Pure Functional)",
                "",
            ]

            if "num_states" in shapes:
                summary_lines.append(f"States: {shapes.get('num_states', 'N/A')}")
            if "num_observations" in shapes:
                summary_lines.append(
                    f"Observations: {shapes.get('num_observations', 'N/A')}"
                )
            if "num_actions" in shapes:
                summary_lines.append(f"Actions: {shapes.get('num_actions', 'N/A')}")

            summary_lines.append("")
            summary_lines.append("Matrix Shapes:")
            for key in ["A_shape", "B_shape", "C_shape", "D_shape"]:
                if key in shapes:
                    summary_lines.append(
                        f"  {key.replace('_shape', '')}: {shapes[key]}"
                    )

            if parsed["num_simulation_steps"] > 0:
                summary_lines.append("")
                summary_lines.append(
                    f"Simulation: {parsed['num_simulation_steps']} steps"
                )

            if parsed["average_efe"] is not None:
                summary_lines.append(f"Avg EFE: {parsed['average_efe']:.6f}")

            summary_text = "\n".join(summary_lines)
            ax.text(
                0.5,
                0.5,
                summary_text,
                ha="center",
                va="center",
                fontsize=12,
                fontfamily="monospace",
                transform=ax.transAxes,
                bbox={"boxstyle": "round", "facecolor": "lightblue", "alpha": 0.3},
            )

            viz_file = output_dir / f"{model_name}_jax_model_summary.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated model summary: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create model summary: {e}")

    # 6. Belief Trajectory (if available)
    if beliefs:
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            beliefs_arr = np.array(beliefs)

            if beliefs_arr.ndim == 2:
                for i in range(beliefs_arr.shape[1]):
                    ax.plot(beliefs_arr[:, i], label=f"State {i}", linewidth=2)
                ax.legend()
            else:
                ax.plot(beliefs_arr, linewidth=2)

            ax.set_xlabel("Time Step", fontweight="bold")
            ax.set_ylabel("Belief", fontweight="bold")
            ax.set_title(f"JAX Belief Trajectory - {model_name}", fontweight="bold")
            ax.grid(True, alpha=0.3)

            viz_file = output_dir / f"{model_name}_jax_beliefs.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated belief trajectory: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create belief plot: {e}")

    return visualizations


def extract_simulation_data(
    execution_dir: Path, logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Extract JAX simulation data from execution outputs.

    Args:
        execution_dir: Directory containing execution results
        logger: Logger instance

    Returns:
        Dictionary with extracted simulation data
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    data: dict[str, Any] = {
        "actions": [],
        "beliefs": [],
        "free_energy": [],
        "model_name": "",
        "framework": "jax",
    }

    try:
        exec_logs_dir = execution_dir / "execution_logs"
        if exec_logs_dir.exists():
            results_files = list(exec_logs_dir.glob("*_results.json"))
            if results_files:
                with open(results_files[0], "r") as f:
                    results = json.load(f)
                sim_data = results.get("simulation_data", {})
                data.update(sim_data)

    except Exception as e:
        logger.warning(f"Failed to extract JAX data: {e}")

    return data


__all__: list[Any] = [
    "generate_analysis_from_logs",
    "create_jax_visualizations",
    "extract_simulation_data",
    "parse_raw_output",
]
