"""
RxInfer.jl Analysis Module

Per-framework analysis and visualization for RxInfer.jl simulations.

This module is exercised for the pomdp_gridworld exemplar but is hardened to be
generic across ALL exemplars whose ``simulation_results.json`` conform to the
``rxinfer_simulation_v1`` schema. Different exemplars emit slightly different
mixtures of keys:

* ``beliefs`` / ``beliefs_by_factor`` (dict-shaped: ``{"joint_state": [...]}``)
* ``observations`` / ``observations_by_modality``
* ``true_states`` / ``hidden_states_by_factor``
* ``actions`` / ``actions_by_control_factor``
* ``expected_free_energy`` / ``variational_free_energy`` / ``efe_per_action``
* ``policy_posterior`` (optional)
* ``metrics``, ``validation`` (optional)

The analyzer normalises these into flat arrays and renders a consistent,
comprehensive visualization set for any result that carries the core belief /
observation arrays, skipping optional plots gracefully when their data is
absent rather than raising.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Import shared visualization utilities (centralized matplotlib setup)
from ..viz_base import MATPLOTLIB_AVAILABLE, np, plt
from .metrics import (
    _as_2d_list as _as_2d_list,
)
from .metrics import (
    _as_flat_list as _as_flat_list,
)
from .metrics import (
    _compute_convergence_diagnostics as _compute_convergence_diagnostics,
)
from .metrics import (
    _first_dict_value as _first_dict_value,
)
from .metrics import (
    _normalise_actions as _normalise_actions,
)
from .metrics import (
    _normalise_beliefs as _normalise_beliefs,
)
from .metrics import (
    _normalise_efe_per_action as _normalise_efe_per_action,
)
from .metrics import (
    _normalise_free_energy as _normalise_free_energy,
)
from .metrics import (
    _normalise_obs as _normalise_obs,
)
from .metrics import (
    _normalise_policy_posterior as _normalise_policy_posterior,
)
from .metrics import (
    _normalise_true_states as _normalise_true_states,
)
from .metrics import (
    compute_per_factor_beliefs as compute_per_factor_beliefs,
)
from .metrics import (
    summarize_strategy_validation as summarize_strategy_validation,
)
from .result_ingestion import (
    _current_rxinfer_models as _current_rxinfer_models,
)
from .result_ingestion import (
    _latest_current_results_file as _latest_current_results_file,
)
from .result_ingestion import (
    extract_simulation_data as extract_simulation_data,
)
from .result_ingestion import read_result_object


def _safe_close_figures(context: str) -> None:
    """Close open matplotlib figures, logging any cleanup failure (SC-17 tier 3).

    Replaces silent ``except: pass`` cleanup swallows so figure-close failures
    are observable instead of being discarded.
    """
    if plt is None:
        return
    try:
        plt.close()
    except (OSError, ValueError) as exc:
        logger.warning("Figure cleanup failed after %s: %s", context, exc)


# ---------------------------------------------------------------------------
# Data normalisation helpers
# ---------------------------------------------------------------------------
_CORE_PLOT_TYPES: List[str] = [
    "belief_evolution",
    "obs_vs_true",
    "belief_heatmap",
    "per_factor_beliefs",
    "belief_entropy",
    "accuracy",
    "action_frequencies",
    "belief_convergence",
    "belief_trace",
    "free_energy",
    "observations",
    "efe_per_action_heatmap",
    "convergence_diagnostics",
]


# ---------------------------------------------------------------------------
# Current-model helpers
# ---------------------------------------------------------------------------


def generate_analysis_from_logs(
    execution_dir: Path, output_dir: Path, verbose: bool = False
) -> List[str]:
    """
    Generate analysis and visualizations from RxInfer execution logs.

    Args:
        execution_dir: Directory containing execution results
        output_dir: Directory to save visualizations
        verbose: Enable verbose logging

    Returns:
        List of generated visualization file paths
    """
    visualizations: list[Any] = []

    try:
        # Find RxInfer execution results
        current_models = _current_rxinfer_models(execution_dir)
        rxinfer_dirs = list(execution_dir.glob("*/rxinfer"))

        for rxinfer_dir in rxinfer_dirs:
            model_name = rxinfer_dir.parent.name
            if current_models is not None and model_name not in current_models:
                continue
            sim_data_dir = rxinfer_dir / "simulation_data"
            if sim_data_dir.exists():
                # Load simulation results
                results_file = _latest_current_results_file(sim_data_dir)
                if results_file is not None:
                    try:
                        data = read_result_object(results_file)

                        viz_files = create_rxinfer_visualizations(
                            data, output_dir, model_name, verbose
                        )
                        visualizations.extend(viz_files)

                        # D7: also produce an animated GIF alongside the PNGs.
                        try:
                            from .gif_animator import generate_gif_animation

                            gif_path: Path = output_dir / (
                                f"{model_name}_rxinfer_animation.gif"
                            )
                            gif_file = generate_gif_animation(
                                data, gif_path, model_name=model_name
                            )
                            if gif_file:
                                visualizations.append(gif_file)
                        except Exception as e:
                            logger.warning(
                                f"GIF generation failed for {model_name}: {e}"
                            )

                    except Exception as e:
                        logger.warning(f"Failed to process {results_file}: {e}")

    except Exception as e:
        logger.error("RxInfer analysis failed (%s): %s", type(e).__name__, e)

    return visualizations


def create_rxinfer_visualizations(
    data: Dict[str, Any], output_dir: Path, model_name: str, verbose: bool = False
) -> List[str]:
    """
    Create visualizations from RxInfer simulation data.

    Produces a comprehensive, consistent visualization set for any
    ``rxinfer_simulation_v1`` result. Core plots (belief evolution,
    observation-vs-true, belief heatmap, belief entropy) are emitted whenever
    the required arrays exist; optional plots (accuracy, action frequencies,
    belief convergence, belief trace, free energy, observations) are skipped
    gracefully when their source data is missing — never raising.

    Args:
        data: Simulation results dictionary
        output_dir: Output directory
        model_name: Name of the model
        verbose: Enable verbose logging

    Returns:
        List of generated file paths
    """
    from gnn.analysis.result_adapter import model_family, result_views

    from .family_visuals import artifact_component, continuous_png

    views = result_views(data)
    if model_family(data) == "continuous":
        return [
            continuous_png(
                data,
                output_dir / f"{model_name}_rxinfer_gaussian_posterior.png",
                model_name,
            )
        ]
    if len(views) > 1:
        files = []
        for name, view in views.items():
            files.extend(
                create_rxinfer_visualizations(
                    view,
                    output_dir,
                    f"{model_name}_{artifact_component(name)}",
                    verbose,
                )
            )
        return files
    original_data = data
    if views:
        data = next(iter(views.values()))
    visualizations: list[Any] = []

    if not MATPLOTLIB_AVAILABLE or plt is None or np is None:
        logger.warning("Matplotlib unavailable, skipping RxInfer visualizations")
        return visualizations

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Normalise data into flat arrays (tolerates dict-shaped / missing keys)
    beliefs = _normalise_beliefs(data)
    observations = _normalise_obs(data)
    true_states = _normalise_true_states(data)
    actions = _normalise_actions(data)
    free_energy = _normalise_free_energy(data)
    efe_per_action = _normalise_efe_per_action(data)

    # --- Convergence diagnostics (D5): derived from the per-iteration VFE trace
    convergence_diagnostics = _compute_convergence_diagnostics(free_energy)
    # Store under a dedicated key so the diagnostics ride along in the results.
    data["convergence_diagnostics"] = convergence_diagnostics

    # --- Per-factor belief marginals (D4): empty for flat / single-factor models
    per_factor_beliefs = compute_per_factor_beliefs(data)
    data["per_factor_beliefs"] = per_factor_beliefs

    # --- Strategy-declared validation fields (FP-8): field -> value summary
    data["validation_summary"] = summarize_strategy_validation(data)
    for key in ("convergence_diagnostics", "per_factor_beliefs", "validation_summary"):
        original_data[key] = data[key]

    beliefs_arr = np.asarray(beliefs, dtype=float) if beliefs else np.zeros((0, 0))
    have_beliefs = beliefs_arr.ndim >= 1
    have_2d_beliefs = beliefs_arr.ndim == 2 and beliefs_arr.shape[0] > 0

    def _record(plot_type: str) -> Optional[str]:
        viz_file = output_dir / f"{model_name}_rxinfer_{plot_type}.png"
        if viz_file.exists() and viz_file.stat().st_size > 0:
            visualizations.append(str(viz_file))
            logger.info(f"Generated {plot_type}: {viz_file.name}")
            return str(viz_file)
        return None

    # 1. Belief Evolution Plot
    if have_beliefs:
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            if have_2d_beliefs:
                for i in range(beliefs_arr.shape[1]):
                    ax.plot(beliefs_arr[:, i], label=f"State {i + 1}", linewidth=2)
            else:
                ax.plot(beliefs_arr, label="Belief", linewidth=2)

            ax.set_xlabel("Time Step")
            ax.set_ylabel("Belief Probability")
            ax.set_title(f"RxInfer Belief Evolution - {model_name}", fontweight="bold")
            if ax.get_legend_handles_labels()[0]:
                ax.legend()
            ax.grid(True, alpha=0.3)

            viz_file = output_dir / f"{model_name}_rxinfer_belief_evolution.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated belief evolution: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create belief plot: {e}")
            _safe_close_figures(f"{model_name} belief plot")

    # 2. Observation vs True State Plot
    if observations and true_states:
        try:
            fig, ax = plt.subplots(figsize=(12, 4))
            x = range(len(observations))
            ax.scatter(x, observations, label="Observations", alpha=0.7, s=50)
            ax.scatter(x, true_states, label="True States", alpha=0.7, s=50, marker="x")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("State/Observation")
            ax.set_title(
                f"RxInfer Observations vs True States - {model_name}", fontweight="bold"
            )
            ax.legend()
            ax.grid(True, alpha=0.3)

            viz_file = output_dir / f"{model_name}_rxinfer_obs_vs_true.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated obs vs true: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create obs plot: {e}")
            _safe_close_figures(f"{model_name} obs plot")

    # 3. Belief Heatmap (2D visualization of beliefs over time)
    if have_2d_beliefs and beliefs_arr.shape[0] > 1:
        try:
            fig, ax = plt.subplots(figsize=(14, 5))
            im = ax.imshow(
                beliefs_arr.T,
                aspect="auto",
                cmap="viridis",
                origin="lower",
                interpolation="nearest",
            )
            ax.set_xlabel("Time Step")
            ax.set_ylabel("State")
            ax.set_title(f"RxInfer Belief Heatmap - {model_name}", fontweight="bold")
            ax.set_yticks(range(beliefs_arr.shape[1]))
            ax.set_yticklabels([f"State {i + 1}" for i in range(beliefs_arr.shape[1])])

            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Belief Probability", fontweight="bold")

            viz_file = output_dir / f"{model_name}_rxinfer_belief_heatmap.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated belief heatmap: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create belief heatmap: {e}")
            _safe_close_figures(f"{model_name} belief heatmap")

    # 3b. Per-Factor Belief Marginals (D4): one small-multiple panel per factor
    if per_factor_beliefs:
        factor_names = list(per_factor_beliefs)
        n_factors = len(factor_names)
        n_cols = min(3, n_factors)
        n_rows = (n_factors + n_cols - 1) // n_cols
        fig, panels = plt.subplots(
            n_rows,
            n_cols,
            figsize=(6.0 * n_cols, 3.5 * n_rows),
            squeeze=False,
            layout="constrained",
        )
        for index, name in enumerate(factor_names):
            ax = panels[index // n_cols][index % n_cols]
            trajectory = np.asarray(per_factor_beliefs[name], dtype=float)
            factor_size = trajectory.shape[1]
            for state_index in range(factor_size):
                ax.plot(
                    trajectory[:, state_index],
                    linewidth=2,
                    label=f"State {state_index + 1}",
                )
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Marginal Probability")
            ax.set_ylim(0, 1.05)
            ax.set_title(f"{name} ({factor_size} states)", fontweight="bold")
            ax.grid(True, alpha=0.3)
            # Timesteps are discrete — no fractional ticks.
            ax.locator_params(axis="x", integer=True)
            if factor_size <= 8:
                ax.legend(fontsize=8)
        for index in range(n_factors, n_rows * n_cols):
            panels[index // n_cols][index % n_cols].axis("off")

        fig.suptitle(
            f"RxInfer Per-Factor Belief Marginals - {model_name}", fontweight="bold"
        )
        viz_file = output_dir / f"{model_name}_rxinfer_per_factor_beliefs.png"
        plt.savefig(viz_file, dpi=300, bbox_inches="tight")
        plt.close()
        visualizations.append(str(viz_file))
        logger.info(f"Generated per-factor beliefs: {viz_file.name}")

    # 4. Belief Entropy (uncertainty tracking over time)
    if have_2d_beliefs:
        try:
            # Calculate entropy for each timestep: H = -sum(p * log(p))
            epsilon = 1e-10
            beliefs_clipped = np.clip(beliefs_arr, epsilon, 1.0)
            entropy = -np.sum(beliefs_clipped * np.log2(beliefs_clipped), axis=1)

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(entropy, "purple", linewidth=2, marker="o", markersize=3)
            ax.fill_between(range(len(entropy)), entropy, alpha=0.3, color="purple")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Belief Entropy (bits)")
            ax.set_title(
                f"RxInfer Belief Uncertainty - {model_name}", fontweight="bold"
            )
            ax.grid(True, alpha=0.3)

            max_entropy = np.log2(beliefs_arr.shape[1])
            ax.axhline(
                y=max_entropy,
                color="red",
                linestyle="--",
                alpha=0.5,
                label=f"Max Entropy ({max_entropy:.2f})",
            )
            ax.legend()

            viz_file = output_dir / f"{model_name}_rxinfer_belief_entropy.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated belief entropy: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create entropy plot: {e}")
            _safe_close_figures(f"{model_name} entropy plot")

    # 5. Inference Accuracy (if we have true states + 2D beliefs)
    if have_2d_beliefs and true_states:
        try:
            inferred_states = np.argmax(
                beliefs_arr, axis=1
            )  # 0-indexed (matching Julia output)
            true_arr = np.asarray(true_states[: len(inferred_states)], dtype=float)
            # true_states are 0-indexed from the generated script
            n = min(len(inferred_states), len(true_arr))

            matches = (inferred_states[:n] == true_arr[:n]).astype(int)
            cumulative_accuracy = np.cumsum(matches) / (np.arange(n) + 1)

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(cumulative_accuracy * 100, "green", linewidth=2)
            ax.fill_between(
                range(len(cumulative_accuracy)),
                cumulative_accuracy * 100,
                alpha=0.3,
                color="green",
            )
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Cumulative Accuracy (%)")
            ax.set_title(
                f"RxInfer Inference Accuracy - {model_name}", fontweight="bold"
            )
            ax.set_ylim(0, 105)
            ax.grid(True, alpha=0.3)

            if len(cumulative_accuracy) > 0:
                final_acc = cumulative_accuracy[-1] * 100
                ax.axhline(y=final_acc, color="navy", linestyle="--", alpha=0.5)
                ax.text(
                    len(cumulative_accuracy) - 1,
                    final_acc + 3,
                    f"Final: {final_acc:.1f}%",
                    ha="right",
                    fontweight="bold",
                )

            viz_file = output_dir / f"{model_name}_rxinfer_accuracy.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated inference accuracy: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create accuracy plot: {e}")
            _safe_close_figures(f"{model_name} accuracy plot")

    # 6. Action Frequencies (bar chart, if actions exist)
    if actions:
        try:
            actions_arr = np.asarray(actions, dtype=float)
            unique, counts = np.unique(actions_arr, return_counts=True)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(unique, counts, color="skyblue", edgecolor="navy")
            ax.set_xlabel("Action")
            ax.set_ylabel("Frequency")
            ax.set_title(
                f"RxInfer Action Frequencies - {model_name}", fontweight="bold"
            )
            ax.set_xticks(unique)
            ax.grid(True, alpha=0.3, axis="y")

            viz_file = output_dir / f"{model_name}_rxinfer_action_frequencies.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated action frequencies: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create action frequencies plot: {e}")
            _safe_close_figures(f"{model_name} action frequencies plot")

    # 6b. EFE Per Action Heatmap (structured EFE landscape over time)
    if efe_per_action:
        try:
            eaa_arr = np.asarray(efe_per_action, dtype=float)
            fig, ax = plt.subplots(figsize=(14, 5))
            im = ax.imshow(
                eaa_arr.T,
                aspect="auto",
                cmap="RdBu_r",
                origin="lower",
                interpolation="nearest",
            )
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Action")
            ax.set_title(f"RxInfer EFE per Action - {model_name}")
            if eaa_arr.shape[1] > 0:
                ax.set_yticks(range(eaa_arr.shape[1]))
                ax.set_yticklabels([f"A{i + 1}" for i in range(eaa_arr.shape[1])])

            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Expected Free Energy")

            viz_file = output_dir / f"{model_name}_rxinfer_efe_per_action_heatmap.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated EFE per action heatmap: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create EFE per action heatmap: {e}")
            _safe_close_figures(f"{model_name} EFE per action heatmap")

    # 7. Belief Convergence (max belief probability over time)
    if have_2d_beliefs:
        try:
            max_probabilities = np.max(beliefs_arr, axis=1)
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(max_probabilities, color="brown", linewidth=2)
            ax.fill_between(
                range(len(max_probabilities)),
                max_probabilities,
                alpha=0.3,
                color="brown",
            )
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Max State Probability")
            ax.set_title(
                f"RxInfer Belief Convergence - {model_name}", fontweight="bold"
            )
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)

            viz_file = output_dir / f"{model_name}_rxinfer_belief_convergence.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated belief convergence: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create belief convergence plot: {e}")
            _safe_close_figures(f"{model_name} belief convergence plot")

    # 8. Belief Trace (inferred most-likely state over time vs true state)
    if have_2d_beliefs:
        try:
            inferred_states = np.argmax(beliefs_arr, axis=1)
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(inferred_states, "o-", label="Inferred State", color="darkorange")
            if true_states:
                ax.plot(
                    np.asarray(true_states[: len(inferred_states)], dtype=float),
                    "s--",
                    label="True State",
                    color="steelblue",
                )
            ax.set_xlabel("Time Step")
            ax.set_ylabel("State Index")
            ax.set_title(f"RxInfer Belief Trace - {model_name}", fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)

            viz_file = output_dir / f"{model_name}_rxinfer_belief_trace.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated belief trace: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create belief trace plot: {e}")
            _safe_close_figures(f"{model_name} belief trace plot")

    # 9. Free Energy (per-iteration VFE or expected free energy over time)
    if free_energy:
        try:
            fe_arr = np.asarray(free_energy, dtype=float)
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(fe_arr, color="crimson", linewidth=2, marker="o", markersize=3)
            ax.fill_between(range(len(fe_arr)), fe_arr, alpha=0.3, color="crimson")
            # Label accurately: VFE is per-iteration when from variational_free_energy
            fe_key = (
                "vfe_per_iteration"
                if data.get("vfe_per_iteration") is not None
                else "variational_free_energy"
                if data.get("variational_free_energy") is not None
                else "expected_free_energy"
            )
            is_per_iteration = fe_key in (
                "vfe_per_iteration",
                "variational_free_energy",
            )
            xlabel = "Inference Iteration" if is_per_iteration else "Time Step"
            ylabel = "Variational Free Energy" if is_per_iteration else "Free Energy"
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            title_prefix = "VFE (per-iteration)" if is_per_iteration else "Free Energy"
            ax.set_title(f"RxInfer {title_prefix} - {model_name}", fontweight="bold")
            ax.grid(True, alpha=0.3)

            viz_file = output_dir / f"{model_name}_rxinfer_free_energy.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated free energy: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create free energy plot: {e}")
            _safe_close_figures(f"{model_name} free energy plot")

    # 9b. Convergence Diagnostics (D5): VFE slope / tail rate / iterations to converge
    if free_energy:
        try:
            diag = convergence_diagnostics
            fe_arr = np.asarray(free_energy, dtype=float)
            # Keep diagnostic labels in a reserved panel, clear of VFE and ticks.
            fig, (ax, diagnostics_ax) = plt.subplots(
                1,
                2,
                figsize=(14, 4.5),
                gridspec_kw={"width_ratios": [3, 1]},
                layout="constrained",
            )
            diagnostics_ax.set_axis_off()
            ax.plot(fe_arr, color="crimson", linewidth=2, marker="o", markersize=3)
            ax.fill_between(range(len(fe_arr)), fe_arr, alpha=0.3, color="crimson")
            ax.set_xlabel("Inference Iteration")
            ax.set_ylabel("Variational Free Energy")
            ax.set_title(
                f"RxInfer Convergence Diagnostics - {model_name}", fontweight="bold"
            )
            ax.grid(True, alpha=0.3)

            # Vertical marker at the iteration where VFE first settles
            itc = diag.get("iterations_to_convergence")
            if itc is not None and 1 <= itc <= len(fe_arr):
                ax.axvline(x=itc - 1, color="navy", linestyle="--", alpha=0.7)
                diagnostics_ax.text(
                    0.02,
                    0.98,
                    f"Converged @ iter {itc}",
                    transform=diagnostics_ax.transAxes,
                    va="top",
                    color="navy",
                    fontsize=9,
                    fontweight="bold",
                )

            # Annotate the derived diagnostics
            slope = diag.get("vfe_slope")
            rate = diag.get("convergence_rate")
            annotation_lines: List[str] = []
            if slope is not None:
                annotation_lines.append(f"VFE slope        : {slope:.4g}")
            if rate is not None:
                annotation_lines.append(f"Conv. rate (last10): {rate:.4g}")
            annotation_lines.append(
                f"Converged iter   : {itc if itc is not None else 'n/a'}"
            )
            diagnostics_ax.text(
                0.02,
                0.80,
                "\n".join(annotation_lines),
                transform=diagnostics_ax.transAxes,
                va="top",
                fontsize=9,
                fontfamily="monospace",
                bbox=dict(
                    boxstyle="round", facecolor="lightgoldenrodyellow", alpha=0.6
                ),
            )

            viz_file = output_dir / f"{model_name}_rxinfer_convergence_diagnostics.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated convergence diagnostics: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create convergence diagnostics plot: {e}")
            _safe_close_figures(f"{model_name} convergence diagnostics plot")

    # 10. Observations (raw observation trace over time)
    if observations:
        try:
            obs_arr = np.asarray(observations, dtype=float)
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(obs_arr, "o-", color="teal", linewidth=2, markersize=4)
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Observation")
            ax.set_title(f"RxInfer Observations - {model_name}", fontweight="bold")
            ax.grid(True, alpha=0.3)

            viz_file = output_dir / f"{model_name}_rxinfer_observations.png"
            plt.savefig(viz_file, dpi=300, bbox_inches="tight")
            plt.close()
            visualizations.append(str(viz_file))
            logger.info(f"Generated observations: {viz_file.name}")
        except Exception as e:
            logger.warning(f"Failed to create observations plot: {e}")
            _safe_close_figures(f"{model_name} observations plot")

    return visualizations


__all__: list[Any] = [
    "generate_analysis_from_logs",
    "create_rxinfer_visualizations",
    "compute_per_factor_beliefs",
    "summarize_strategy_validation",
    "extract_simulation_data",
]
