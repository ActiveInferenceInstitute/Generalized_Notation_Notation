#!/usr/bin/env python3
"""
Per-model execution-output plot generation for GNN Step 16 analysis visualizations.

Extracted from ``analysis.visualizations``.
"""

import json
import logging
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from .viz_animations import generate_gridworld_animation_suite
from .viz_base import (
    np,
    plt,
    safe_savefig,
)
from .viz_dashboard import (
    generate_confidence_comparison,
    generate_cross_framework_comparison,
    generate_efe_convergence_comparison,
    generate_framework_radar,
)
from .viz_schema import (
    CURRENT_VISUALIZATION_SCHEMAS,
    _current_schema_visualization_data,
    _normalize_framework_name,
)

logger = logging.getLogger(__name__)


def visualize_all_framework_outputs(
    execution_dir: Path,
    output_dir: Path,
    logger_instance: Optional[logging.Logger] = None,
    allowed_frameworks: Optional[set[str]] = None,
    allowed_model_names: Optional[set[str]] = None,
    generate_animations: bool = True,
) -> List[str]:
    """
    Generate comprehensive visualizations for all raw execution outputs.

    This function creates a complete visualization suite for POMDP simulation outputs:
    - Belief trajectory plots per framework
    - Action selection histograms
    - Free energy evolution curves
    - State distribution heatmaps
    - Cross-framework metric comparisons

    Args:
        execution_dir: Directory containing execution results (e.g., output/12_execute_output)
        output_dir: Directory to save visualizations
        logger_instance: Optional logger instance
        allowed_frameworks: Optional normalized framework names from the current run
        allowed_model_names: Optional model directory names from the current run

    Returns:
        List of generated visualization file paths
    """
    log = logger_instance or logger
    generated_files: List[str] = []

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all execution data
    framework_data: Dict[str, Dict[str, Any]] = {}

    # Known framework directory names for path-based detection
    _FRAMEWORK_DIRS: set[Any] = {
        "pymdp",
        "rxinfer",
        "activeinference_jl",
        "jax",
        "discopy",
        "pytorch",
        "numpyro",
        "cpomdp",
    }

    # Search for execution result files
    for result_file in execution_dir.rglob("*_results.json"):
        if "simulation_data" in result_file.parts:
            continue
        try:
            with open(result_file, "r") as f:
                data = json.load(f)

            # Normalize framework name to canonical form
            raw_framework = data.get("framework", "unknown")
            framework = _normalize_framework_name(raw_framework)
            if (
                framework in {"pymdp", "rxinfer", "activeinference_jl"}
                and data.get("schema_version")
                and data.get("schema_version") not in CURRENT_VISUALIZATION_SCHEMAS
            ):
                continue
            model_name = data.get("model_name", result_file.parent.name)
            if allowed_frameworks and framework not in allowed_frameworks:
                continue
            if allowed_model_names and model_name not in allowed_model_names:
                model_slug = str(model_name).lower().replace(" ", "_").replace("-", "_")
                path_model = None
                for index, part in enumerate(result_file.parts):
                    if part in _FRAMEWORK_DIRS and index >= 1:
                        path_model = result_file.parts[index - 1]
                        break
                if (
                    model_slug not in allowed_model_names
                    and path_model not in allowed_model_names
                ):
                    continue

            key = f"{framework}_{model_name}"
            if key not in framework_data:
                framework_data[key] = {
                    "framework": framework,
                    "model_name": model_name,
                    "results": [],
                }
            framework_data[key]["results"].append(data)

        except Exception as e:
            log.warning(f"Failed to load {result_file}: {e}")

    # Also search for simulation_results.json files - MERGE into existing keys
    for sim_file in execution_dir.rglob("*simulation_results.json"):
        try:
            with open(sim_file, "r") as f:
                data = json.load(f)

            # Determine framework from path or file content
            path_parts = sim_file.parts
            framework = "unknown"
            for part in path_parts:
                if part in _FRAMEWORK_DIRS:
                    framework = part
                    break

            # Also check if framework is in the data itself
            if framework == "unknown" and "framework" in data:
                framework = data["framework"]

            # Normalize framework name to canonical form
            framework = _normalize_framework_name(framework)
            if (
                framework in {"pymdp", "rxinfer", "activeinference_jl"}
                and data.get("schema_version")
                and data.get("schema_version") not in CURRENT_VISUALIZATION_SCHEMAS
            ):
                continue
            if allowed_frameworks and framework not in allowed_frameworks:
                continue

            # Derive model_name: walk up directory tree to find the first
            # ancestor that isn't a framework directory or a subdirectory like
            # 'simulation_data'.  This prevents names like jax_jax_ or rxinfer_rxinfer_.
            model_name = "unknown"
            for ancestor in sim_file.parents:
                candidate = ancestor.name
                if (
                    candidate
                    and candidate not in _FRAMEWORK_DIRS
                    and candidate
                    not in {
                        "simulation_data",
                        "execution_logs",
                        "individual_outputs",
                        "12_execute_output",
                        "output",
                    }
                ):
                    model_name = candidate
                    break
            if allowed_model_names and model_name not in allowed_model_names:
                continue

            # Use the same key format as results (no _sim suffix) to merge data
            key = f"{framework}_{model_name}"
            if key not in framework_data:
                framework_data[key] = {
                    "framework": framework,
                    "model_name": model_name,
                    "simulation_data": data,
                    "raw_simulation_data": data,
                    "source_file": str(sim_file),
                }
            else:
                # Merge simulation data into existing entry
                framework_data[key]["simulation_data"] = data
                framework_data[key]["raw_simulation_data"] = data
                framework_data[key]["source_file"] = str(sim_file)

        except Exception as e:
            log.warning(f"Failed to load simulation file {sim_file}: {e}")

    if not framework_data:
        log.warning("No execution data found for visualization")
        return generated_files

    log.info(
        f"Found {len(framework_data)} framework/model combinations for visualization"
    )

    # Generate visualizations for each framework/model
    for _key, data in framework_data.items():
        framework = data["framework"]
        model_name = data["model_name"]

        try:
            # Extract simulation data
            sim_data = data.get("simulation_data", {})
            if not sim_data and data.get("results"):
                # Try to extract from first result
                result = data["results"][0]
                if framework in {"pymdp", "rxinfer", "activeinference_jl"}:
                    sim_data = _current_schema_visualization_data(result)
                else:
                    sim_data = result.get("simulation_data", {})

                # Also check implementation directory for files
                impl_dir = result.get("implementation_directory")
                if impl_dir:
                    impl_path = Path(impl_dir)
                    sim_data_dir = impl_path / "simulation_data"
                    if sim_data_dir.exists():
                        for json_file in sim_data_dir.glob("*.json"):
                            try:
                                with open(json_file, "r") as f:
                                    file_data = json.load(f)
                                if isinstance(file_data, dict):
                                    if framework in {
                                        "pymdp",
                                        "rxinfer",
                                        "activeinference_jl",
                                    }:
                                        sim_data.update(
                                            _current_schema_visualization_data(
                                                file_data
                                            )
                                        )
                                    else:
                                        sim_data.update(file_data)
                            except (json.JSONDecodeError, OSError) as e:
                                logger.debug(
                                    "Skipping unreadable simulation data file %s: %s",
                                    json_file,
                                    e,
                                )
            elif framework in {"pymdp", "rxinfer", "activeinference_jl"}:
                sim_data = _current_schema_visualization_data(sim_data)

                # For ActiveInference.jl: read CSV simulation data directly
                if framework == "activeinference_jl" and not sim_data.get("beliefs"):
                    if impl_dir:
                        import csv as csv_module

                        sim_data_path = Path(impl_dir) / "simulation_data"
                        csv_candidates: list[Any] = []
                        if sim_data_path.exists():
                            csv_candidates.append(
                                sim_data_path / "simulation_results.csv"
                            )
                            csv_candidates.extend(
                                sorted(sim_data_path.glob("*_simulation_results.csv"))
                            )

                        for csv_file in csv_candidates:
                            if csv_file.exists():
                                try:
                                    beliefs: list[Any] = []
                                    actions: list[Any] = []
                                    observations: list[Any] = []
                                    with open(csv_file, "r") as f:
                                        lines = [
                                            line
                                            for line in f
                                            if not line.startswith("#")
                                        ]
                                    if lines:
                                        reader = csv_module.reader(lines)
                                        for row in reader:
                                            if len(row) >= 3:
                                                try:
                                                    observations.append(
                                                        int(float(row[1]))
                                                    )
                                                    actions.append(int(float(row[2])))
                                                    if len(row) > 3:
                                                        beliefs.append(
                                                            [float(x) for x in row[3:]]
                                                        )
                                                except ValueError as e:
                                                    log.debug(
                                                        "Skipping non-numeric CSV row: %s",
                                                        e,
                                                    )
                                                    continue
                                    if beliefs:
                                        sim_data["beliefs"] = beliefs
                                    if actions:
                                        sim_data["actions"] = actions
                                    if observations:
                                        sim_data["observations"] = observations
                                    if beliefs or actions:
                                        log.info(
                                            f"Extracted {len(beliefs)} steps from ActiveInference.jl CSV"
                                        )
                                        break
                                except Exception as e:
                                    log.debug(
                                        f"Error reading ActiveInference.jl CSV: {e}"
                                    )

            # Route framework-specific visualizations to correct directories
            framework_viz_dir = output_dir.parent / framework
            framework_viz_dir.mkdir(parents=True, exist_ok=True)

            # NOTE: Belief heatmaps and action analysis are SKIPPED here
            # because the per-framework analyzers (jax/analyzer.py,
            # rxinfer/analyzer.py, etc.) already produce richer versions
            # of these plots. Generating them here would create duplicates.

            # Generate free energy plot (not produced by per-framework analyzers)
            free_energy = (
                sim_data.get("free_energy", [])
                or sim_data.get("efe_history", [])
                or sim_data.get("expected_free_energy", [])
            )
            vfe_energy = sim_data.get("variational_free_energy", []) or sim_data.get(
                "vfe_history", []
            )

            if free_energy:
                fe_file = (
                    framework_viz_dir / f"{model_name}_{framework}_free_energy.png"
                )
                try:
                    generate_free_energy_plots(
                        free_energy,
                        fe_file,
                        f"Free Energy - {model_name} ({framework})",
                    )
                    generated_files.append(str(fe_file))
                    log.info(f"Generated free energy plot: {fe_file.name}")
                except Exception as e:
                    log.warning(f"Failed to generate free energy plot for {key}: {e}")

            if free_energy and vfe_energy:
                fe_dual_file = (
                    framework_viz_dir / f"{model_name}_{framework}_vfe_vs_efe.png"
                )
                try:
                    generate_vfe_vs_efe_plot(
                        vfe_energy,
                        free_energy,
                        fe_dual_file,
                        f"Active Inference Energy Dynamics - {model_name} ({framework})",
                    )
                    generated_files.append(str(fe_dual_file))
                    log.info(f"Generated dual free energy plot: {fe_dual_file.name}")
                except Exception as e:
                    log.warning(
                        f"Failed to generate dual free energy plot for {key}: {e}"
                    )

            # Generate observation analysis (not produced by all per-framework analyzers)
            observations = sim_data.get("observations", [])
            if observations:
                obs_file = (
                    framework_viz_dir / f"{model_name}_{framework}_observations.png"
                )
                try:
                    generate_observation_analysis(
                        observations,
                        obs_file,
                        f"Observations - {model_name} ({framework})",
                    )
                    generated_files.append(str(obs_file))
                    log.info(f"Generated observation analysis: {obs_file.name}")
                except Exception as e:
                    log.warning(
                        f"Failed to generate observation analysis for {key}: {e}"
                    )

            # Update framework_data with enriched sim_data for comparison chart
            data["simulation_data"] = sim_data

        except Exception as e:
            log.error(f"Failed to generate visualizations for {key}: {e}")

    # Generate cross-framework comparison if multiple frameworks
    frameworks = {d["framework"] for d in framework_data.values()}
    if len(frameworks) > 1:
        # output_dir is already the cross-framework directory when called from
        # processor.py, so use it directly to avoid double-nesting.
        cross_fw_dir = output_dir
        cross_fw_dir.mkdir(parents=True, exist_ok=True)

        try:
            comparison_file = cross_fw_dir / "cross_framework_comparison.png"
            generate_cross_framework_comparison(framework_data, comparison_file)
            generated_files.append(str(comparison_file))
            log.info(f"Generated cross-framework comparison: {comparison_file.name}")
        except Exception as e:
            log.warning(f"Failed to generate cross-framework comparison: {e}")

        # EFE convergence overlay (JAX + PyMDP)
        try:
            efe_file = cross_fw_dir / "efe_convergence_comparison.png"
            files = generate_efe_convergence_comparison(framework_data, efe_file)
            if files:
                generated_files.extend(files)
        except Exception as e:
            log.warning(f"Failed to generate EFE convergence comparison: {e}")

        # Belief confidence comparison
        try:
            conf_file = cross_fw_dir / "confidence_comparison.png"
            files = generate_confidence_comparison(framework_data, conf_file)
            if files:
                generated_files.extend(files)
        except Exception as e:
            log.warning(f"Failed to generate confidence comparison: {e}")

        # Framework radar chart (from execution summary)
        try:
            exec_summary_path = execution_dir / "summaries" / "execution_summary.json"
            if not exec_summary_path.exists():
                exec_summary_path = execution_dir / "execution_summary.json"
            if exec_summary_path.exists():
                radar_file = cross_fw_dir / "framework_radar.png"
                files = generate_framework_radar(
                    exec_summary_path, framework_data, radar_file
                )
                if files:
                    generated_files.extend(files)
        except Exception as e:
            log.warning(f"Failed to generate framework radar: {e}")

        if generate_animations:
            try:
                animation_files = generate_gridworld_animation_suite(
                    framework_data, cross_fw_dir, log
                )
                generated_files.extend(animation_files)
            except Exception as e:
                log.warning(f"Failed to generate GridWorld animations: {e}")

    log.info(f"Generated {len(generated_files)} visualization files")
    return generated_files


def generate_belief_heatmaps(
    beliefs: List[List[float]],
    output_path: Path,
    title: str = "Belief State Evolution Heatmap",
) -> str:
    """
    Generate a heatmap visualization of belief state evolution over time.

    Args:
        beliefs: List of belief distributions at each timestep [[p1, p2, ...], ...]
        output_path: Path to save the visualization
        title: Title for the plot

    Returns:
        Path to the generated file
    """
    if not beliefs or len(beliefs) < 2:
        raise ValueError("Need at least 2 timesteps for heatmap")

    belief_array = np.array(beliefs)
    n_steps, n_states = belief_array.shape

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Heatmap
    ax1 = axes[0]
    im = ax1.imshow(belief_array.T, aspect="auto", cmap="viridis", origin="lower")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("State")
    ax1.set_title(f"{title}\n(Heatmap)")
    ax1.set_yticks(range(n_states))
    ax1.set_yticklabels([f"S{i + 1}" for i in range(n_states)])
    plt.colorbar(im, ax=ax1, label="Probability")

    # Line plot
    ax2 = axes[1]
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, n_states))
    for i in range(n_states):
        ax2.plot(
            range(n_steps),
            belief_array[:, i],
            label=f"State {i + 1}",
            color=colors[i],
            linewidth=2,
        )
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Probability")
    ax2.set_title(f"{title}\n(Trajectories)")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    saved = safe_savefig(output_path, log=logger)
    return saved or str(output_path)


def generate_action_analysis(
    actions: List[int], output_path: Path, title: str = "Action Selection Analysis"
) -> str:
    """
    Generate visualization of action selection patterns.

    Args:
        actions: List of action indices taken at each timestep
        output_path: Path to save the visualization
        title: Title for the plot

    Returns:
        Path to the generated file
    """
    if not actions:
        raise ValueError("No actions provided")

    actions_array = np.array(actions)
    unique_actions = sorted(set(actions))
    n_actions = len(unique_actions)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Action histogram
    ax1 = axes[0]
    action_counts = [np.sum(actions_array == a) for a in unique_actions]
    colors = plt.get_cmap("Set2")(np.linspace(0, 1, n_actions))
    bars = ax1.bar(unique_actions, action_counts, color=colors)
    ax1.set_xlabel("Action")
    ax1.set_ylabel("Count")
    ax1.set_title("Action Frequency Distribution")
    ax1.set_xticks(unique_actions)
    ax1.set_xticklabels([f"A{a}" for a in unique_actions])

    # Add count labels on bars
    for bar, count in zip(bars, action_counts):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(count),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Action sequence plot
    ax2 = axes[1]
    ax2.plot(range(len(actions)), actions, "o-", markersize=4, linewidth=1, alpha=0.7)
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Action")
    ax2.set_title("Action Sequence Over Time")
    ax2.set_yticks(unique_actions)
    ax2.set_yticklabels([f"A{a}" for a in unique_actions])
    ax2.grid(True, alpha=0.3)

    # Action transition matrix
    ax3 = axes[2]
    if len(actions) > 1:
        transition_matrix = np.zeros((n_actions, n_actions))
        action_to_idx = {a: i for i, a in enumerate(unique_actions)}
        for i in range(len(actions) - 1):
            from_idx = action_to_idx[actions[i]]
            to_idx = action_to_idx[actions[i + 1]]
            transition_matrix[from_idx, to_idx] += 1

        # Normalize rows
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(
            transition_matrix,
            row_sums,
            where=row_sums != 0,
            out=np.zeros_like(transition_matrix),
        )

        im = ax3.imshow(transition_matrix, cmap="Blues", vmin=0, vmax=1)
        ax3.set_xlabel("Next Action")
        ax3.set_ylabel("Current Action")
        ax3.set_title("Action Transition Probabilities")
        ax3.set_xticks(range(n_actions))
        ax3.set_yticks(range(n_actions))
        ax3.set_xticklabels([f"A{a}" for a in unique_actions])
        ax3.set_yticklabels([f"A{a}" for a in unique_actions])

        # Add text annotations
        for i in range(n_actions):
            for j in range(n_actions):
                _text = ax3.text(
                    j,
                    i,
                    f"{transition_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black" if transition_matrix[i, j] < 0.5 else "white",
                )

        plt.colorbar(im, ax=ax3, label="Probability")
    else:
        ax3.text(
            0.5,
            0.5,
            "Need > 1 action\nfor transitions",
            ha="center",
            va="center",
            transform=ax3.transAxes,
        )
        ax3.set_title("Action Transition Probabilities")

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    saved = safe_savefig(output_path, log=logger)
    return saved or str(output_path)


def generate_free_energy_plots(
    free_energy: List[float], output_path: Path, title: str = "Free Energy Dynamics"
) -> str:
    """
    Generate visualization of free energy evolution.

    Includes:
    - Free energy over time
    - Moving average trend
    - Convergence analysis

    Args:
        free_energy: List of free energy values over time
        output_path: Path to save the visualization
        title: Title for the plot

    Returns:
        Path to the generated file
    """
    if not free_energy:
        raise ValueError("No free energy values provided")

    fe_array = np.array(free_energy)
    n_steps = len(fe_array)
    is_per_policy = fe_array.ndim == 2

    if is_per_policy:
        n_policies = fe_array.shape[1]
        fe_summary = np.min(fe_array, axis=1)  # The EFE of the best policy at each step
    else:
        n_policies = 1
        fe_summary = fe_array

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Main free energy plot
    ax1 = axes[0, 0]
    if is_per_policy:
        if n_policies <= 10:
            for p in range(n_policies):
                # Only label first few to avoid legend clutter
                lbl = f"Policy {p + 1}" if p < 5 else None
                ax1.plot(
                    range(n_steps), fe_array[:, p], linewidth=1.0, alpha=0.6, label=lbl
                )
        else:
            # Add a heatmap background if there are many policies
            _im = ax1.imshow(
                fe_array.T,
                aspect="auto",
                cmap="viridis",
                interpolation="none",
                alpha=0.3,
            )
            ax1.set_ylabel("Policy Index / EFE")

        # Bold line for the selected/minimum EFE
        ax1.plot(
            range(n_steps), fe_summary, "k-", linewidth=2, label="Min EFE (Selected)"
        )
    else:
        ax1.plot(range(n_steps), fe_array, "b-", linewidth=1.5, label="Free Energy")

    # Add moving average if enough points (using summary EFE)
    if n_steps > 5:
        window = min(5, n_steps // 3)
        moving_avg = np.convolve(fe_summary, np.ones(window) / window, mode="valid")
        ax1.plot(
            range(window - 1, n_steps),
            moving_avg,
            "r--",
            linewidth=2,
            label=f"{window}-step Moving Average",
        )

    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Free Energy")
    ax1.set_title("Free Energy Evolution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Distribution of free energy values (using summary EFE)
    ax2 = axes[0, 1]

    # Check if data is constant to prevent "Too many bins for data range" errors
    fe_min = np.min(fe_summary)
    fe_max = np.max(fe_summary)

    if np.isclose(fe_min, fe_max) or fe_max - fe_min < 1e-10:
        # For constant or near-constant data, use a single explicit bin range centered around the value
        bins: int | list[Any] = [fe_min - 0.5, fe_min + 0.5]
    else:
        bins = min(20, max(1, n_steps))

    ax2.hist(fe_summary, bins=bins, color="steelblue", edgecolor="white", alpha=0.7)
    ax2.axvline(
        np.mean(fe_summary),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(fe_summary):.3f}",
    )
    ax2.axvline(
        np.median(fe_summary),
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {np.median(fe_summary):.3f}",
    )
    ax2.set_xlabel("Free Energy (Selected)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Selected Free Energy Distribution")
    ax2.legend()

    # Rate of change
    ax3 = axes[1, 0]
    if n_steps > 1:
        fe_diff = np.diff(fe_summary)
        ax3.bar(
            range(len(fe_diff)),
            fe_diff,
            color=["green" if d < 0 else "red" for d in fe_diff],
            alpha=0.7,
        )
        ax3.axhline(0, color="black", linestyle="-", linewidth=0.5)
        ax3.set_xlabel("Time Step")
        ax3.set_ylabel("\u0394FE")
        ax3.set_title("Free Energy Change per Step")

        # Add summary statistics
        positive_changes = np.sum(fe_diff > 0)
        negative_changes = np.sum(fe_diff < 0)
        ax3.text(
            0.02,
            0.98,
            f"\u2191 Increases: {positive_changes}\n\u2193 Decreases: {negative_changes}",
            transform=ax3.transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

    # Convergence analysis
    ax4 = axes[1, 1]
    if n_steps > 10:
        # Calculate rolling variance
        window = max(3, n_steps // 10)
        rolling_var = [
            np.var(fe_summary[max(0, i - window) : i]) for i in range(1, n_steps + 1)
        ]
        ax4.plot(range(1, n_steps + 1), rolling_var, "purple", linewidth=2)
        ax4.set_xlabel("Time Step")
        ax4.set_ylabel("Rolling Variance")
        ax4.set_title(f"Convergence Analysis ({window}-step variance)")
        ax4.grid(True, alpha=0.3)

        # Determine convergence status
        if rolling_var:
            final_var = rolling_var[-1]
            converged = final_var < 0.1
            status = "\u2713 Converged" if converged else "\u26a0 Not Converged"
            ax4.text(
                0.98,
                0.98,
                f"{status}\nFinal Variance: {final_var:.4f}",
                transform=ax4.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                fontsize=10,
                bbox={
                    "boxstyle": "round",
                    "facecolor": "lightgreen" if converged else "lightyellow",
                    "alpha": 0.7,
                },
            )
    else:
        ax4.text(
            0.5,
            0.5,
            "Need > 10 steps\nfor convergence analysis",
            ha="center",
            va="center",
            transform=ax4.transAxes,
        )
        ax4.set_title("Convergence Analysis")

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    saved = safe_savefig(output_path, log=logger)
    return saved or str(output_path)


def generate_vfe_vs_efe_plot(
    vfe: List[float],
    efe: List[Any],
    output_path: Path,
    title: str = "Variational vs Expected Free Energy",
) -> str:
    """
    Generate visualization comparing VFE and EFE over time.

    Args:
        vfe: List of variational free energy values (scalars)
        efe: List of expected free energy values (lists of scalars, one per policy)
        output_path: Path to save the visualization
        title: Title for the plot

    Returns:
        Path to the generated file
    """
    if not vfe or not efe:
        raise ValueError("Need both VFE and EFE data to generate plot")

    # Process EFE (take min across policies if it's a list)
    efe_summary: list[Any] = []
    for efe_t in efe:
        if hasattr(efe_t, "__iter__") and not isinstance(efe_t, str):
            efe_summary.append(min(efe_t) if len(efe_t) > 0 else 0)
        else:
            efe_summary.append(efe_t)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = "tab:blue"
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Variational Free Energy (VFE)", color=color1)
    ax1.plot(vfe, "o-", color=color1, linewidth=2, label="VFE (Belief Update Cost)")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color2 = "tab:orange"
    ax2.set_ylabel("Expected Free Energy (EFE)", color=color2)
    ax2.plot(
        efe_summary, "s--", color=color2, linewidth=2, label="Min EFE (Policy Cost)"
    )
    ax2.tick_params(axis="y", labelcolor=color2)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
    )
    plt.subplots_adjust(bottom=0.2)

    saved = safe_savefig(output_path, log=logger)
    return saved or str(output_path)


def generate_observation_analysis(
    observations: List[int], output_path: Path, title: str = "Observation Analysis"
) -> str:
    """
    Generate visualization of observation patterns.

    Args:
        observations: List of observation indices
        output_path: Path to save the visualization
        title: Title for the plot

    Returns:
        Path to the generated file
    """
    if not observations:
        raise ValueError("No observations provided")

    obs_array = np.array(observations)
    unique_obs = sorted(set(observations))
    n_obs = len(unique_obs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Observation frequency
    ax1 = axes[0]
    obs_counts = [np.sum(obs_array == o) for o in unique_obs]
    colors = plt.get_cmap("Pastel1")(np.linspace(0, 1, n_obs))
    ax1.bar(unique_obs, obs_counts, color=colors, edgecolor="black")
    ax1.set_xlabel("Observation")
    ax1.set_ylabel("Count")
    ax1.set_title("Observation Frequency")
    ax1.set_xticks(unique_obs)
    ax1.set_xticklabels([f"O{o}" for o in unique_obs])

    # Observation sequence
    ax2 = axes[1]
    ax2.scatter(
        range(len(observations)),
        observations,
        c=observations,
        cmap="tab10",
        s=30,
        alpha=0.7,
    )
    ax2.plot(range(len(observations)), observations, "gray", alpha=0.3, linewidth=0.5)
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Observation")
    ax2.set_title("Observation Sequence")
    ax2.set_yticks(unique_obs)
    ax2.set_yticklabels([f"O{o}" for o in unique_obs])
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    saved = safe_savefig(output_path, log=logger)
    return saved or str(output_path)
