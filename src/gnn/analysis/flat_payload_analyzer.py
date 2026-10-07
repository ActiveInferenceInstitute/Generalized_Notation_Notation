"""Shared analyzer for flat-payload simulation results.

PyMDP-agnostic framework analyzers (PyTorch, NumPyro) share an identical
analysis pipeline: discover ``simulation_results.json`` files, compute belief
entropy / confidence / action distribution / EFE metrics, write an
``<framework>_analysis.json`` per model, and render belief/action/EFE plots.

This module is the single implementation; each framework's ``analyzer.py``
re-exports ``generate_analysis_from_logs`` and ``_generate_plots`` bound to a
``FlatPayloadSpec`` so the public call sites (and test pins) remain unchanged.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import numpy as np

from .viz_base import MATPLOTLIB_AVAILABLE, plt

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FlatPayloadSpec:
    """Per-framework configuration for the shared flat-payload analyzer."""

    framework: str
    # rglob patterns for discovering simulation_results.json files.
    file_patterns: tuple[str, ...]
    # Output JSON filename written under each model's output directory.
    analysis_filename: str
    # Plotting labels.
    title_prefix: str
    bar_color: str
    # Log-line label (e.g. "PyTorch", "NumPyro") — keeps existing log format.
    log_label: str


def discover_result_files(results_dir: Path, spec: FlatPayloadSpec) -> list[Path]:
    """Find all simulation_results.json files matching ``spec`` under ``results_dir``.

    Includes a root-level ``simulation_results.json`` recovery fallback.
    Deduplicates by path.
    """
    results_dir = Path(results_dir)
    found: list[Path] = []
    for pattern in spec.file_patterns:
        found.extend(results_dir.rglob(pattern))
    root_result = results_dir / "simulation_results.json"
    if root_result.exists() and root_result not in found:
        found.append(root_result)
    return found


def compute_flat_payload_metrics(
    beliefs: np.ndarray, actions: list[Any], efe: np.ndarray
) -> dict[str, Any]:
    """Compute the standard flat-payload metrics dict.

    Pure function — no I/O. Beliefs are a 2-D ``(timesteps, states)`` array.
    """
    metrics: dict[str, Any] = {}
    if beliefs.ndim == 2 and beliefs.shape[0] > 0:
        entropy = -np.sum(beliefs * np.log(beliefs + 1e-16), axis=1)
        metrics["mean_belief_entropy"] = float(np.mean(entropy))
        metrics["final_belief_entropy"] = float(entropy[-1])
        confidence = np.max(beliefs, axis=1)
        metrics["mean_confidence"] = float(np.mean(confidence))
        metrics["final_confidence"] = float(confidence[-1])
    if actions:
        unique, counts = np.unique(np.asarray(actions), return_counts=True)
        metrics["action_distribution"] = {
            int(a): int(c) for a, c in zip(unique, counts)
        }
    if efe.ndim == 2 and efe.shape[0] > 0:
        metrics["mean_efe"] = float(np.mean(efe))
    return metrics


def _generate_plots(
    spec: FlatPayloadSpec,
    beliefs: np.ndarray,
    actions: list[Any],
    observations: list[Any],
    efe: np.ndarray,
    output_dir: Path,
) -> bool:
    """Generate the standard three plots (belief, action, EFE).

    Returns True if at least one plot was written; False if matplotlib is
    unavailable or no plottable data.

    Raises:
        ValueError: The belief plot would exceed 64,000,000 canvas pixels
            at its saved 150 dpi. Working buffers remain subject to the watchdog.
    """
    if not MATPLOTLIB_AVAILABLE or plt is None:
        logger.warning("matplotlib not available — skipping plots")
        return False
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = False

    if beliefs.ndim == 2 and beliefs.shape[0] > 0:
        legend_columns = max(1, min(4, beliefs.shape[1]))
        legend_rows = (beliefs.shape[1] + legend_columns - 1) // legend_columns
        legend_height = 0.7 + 0.3 * legend_rows if beliefs.shape[1] > 8 else 0
        figure_size = (10, 4 + legend_height)
        # Admit the nominal 150-dpi canvas; RGBA and working buffers remain watchdog-owned.
        canvas_pixels = figure_size[0] * figure_size[1] * 150**2
        if canvas_pixels > 64_000_000:
            raise ValueError(
                f"Belief trajectory with {beliefs.shape[1]} states requires "
                f"{canvas_pixels:,.0f} canvas pixels; limit is 64,000,000"
            )
        if beliefs.shape[1] > 8:
            fig, (ax, legend_ax) = plt.subplots(
                2,
                1,
                figsize=figure_size,
                gridspec_kw={"height_ratios": (4, legend_height)},
            )
            legend_ax.set_axis_off()
        else:
            fig, ax = plt.subplots(figsize=figure_size)
            legend_ax = ax
        for s in range(beliefs.shape[1]):
            ax.plot(beliefs[:, s], label=f"State {s}", linewidth=1.5)
        ax.set_xlabel("Timestep", fontsize=14)
        ax.set_ylabel("Belief", fontsize=14)
        ax.set_title(f"{spec.title_prefix} — Belief Trajectory", fontsize=16)
        if legend_ax is ax:
            ax.legend(fontsize=12, ncol=legend_columns)
        else:
            handles, labels = ax.get_legend_handles_labels()
            legend_ax.legend(
                handles, labels, loc="center", fontsize=12, ncol=legend_columns
            )
        ax.tick_params(labelsize=12)
        fig.tight_layout()
        fig.savefig(output_dir / "belief_trajectory.png", dpi=150)
        plt.close(fig)
        saved = True

    if actions:
        fig, ax = plt.subplots(figsize=(6, 4))
        unique, counts = np.unique(np.asarray(actions), return_counts=True)
        ax.bar(unique.astype(str), counts, color=spec.bar_color)
        ax.set_xlabel("Action", fontsize=14)
        ax.set_ylabel("Count", fontsize=14)
        ax.set_title(f"{spec.title_prefix} — Action Distribution", fontsize=16)
        ax.tick_params(labelsize=12)
        fig.tight_layout()
        fig.savefig(output_dir / "action_distribution.png", dpi=150)
        plt.close(fig)
        saved = True

    if efe.ndim == 2 and efe.shape[0] > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        for a_idx in range(efe.shape[1]):
            ax.plot(efe[:, a_idx], label=f"Action {a_idx}", linewidth=1.5)
        ax.set_xlabel("Timestep", fontsize=14)
        ax.set_ylabel("EFE", fontsize=14)
        ax.set_title(f"{spec.title_prefix} — Expected Free Energy", fontsize=16)
        ax.legend(fontsize=12)
        ax.tick_params(labelsize=12)
        fig.tight_layout()
        fig.savefig(output_dir / "efe_history.png", dpi=150)
        plt.close(fig)
        saved = True

    if saved:
        logger.info(f"✅ {spec.log_label} analysis plots saved to: {output_dir}")
    return saved


def generate_analysis_from_logs(
    spec: FlatPayloadSpec,
    results_dir: Path,
    output_dir: Path | None = None,
    verbose: bool = False,
) -> List[str]:
    """Generate analysis from flat-payload simulation results.

    Discovers ``simulation_results.json`` files matching ``spec.file_patterns``,
    computes metrics, renders plots, and writes ``<framework>_analysis.json``
    per model. Returns the list of generated JSON file paths.
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_files: list[str] = []
    result_files = discover_result_files(results_dir, spec)

    if not result_files:
        if verbose:
            logger.debug(
                f"No {spec.log_label} simulation_results.json found under {results_dir}"
            )
        return generated_files

    for results_file in result_files:
        try:
            with open(results_file) as f:
                results = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read results: {e}")
            continue

        # Determine model name from path: the segment preceding the framework.
        path_parts = results_file.parts
        model_name = results.get("model_name", "unknown")
        for i, part in enumerate(path_parts):
            if part == spec.framework and i >= 1:
                model_name = path_parts[i - 1]
                break

        model_output_dir = output_dir / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        beliefs = np.array(results.get("beliefs", []))
        actions = results.get("actions", [])
        observations = results.get("observations", [])
        efe = np.array(results.get("efe_history", []))
        validation = results.get("validation", {})

        analysis: dict[str, Any] = {
            "framework": spec.framework,
            "model_name": model_name,
            "num_timesteps": len(actions),
            "num_states": beliefs.shape[1] if beliefs.ndim == 2 else 0,
            "validation": validation,
            "metrics": compute_flat_payload_metrics(beliefs, actions, efe),
        }

        try:
            plots_ok = _generate_plots(
                spec, beliefs, actions, observations, efe, model_output_dir
            )
            analysis["plots_generated"] = bool(plots_ok)
        except Exception as e:
            logger.warning(f"Plot generation failed for {model_name}: {e}")
            analysis["plots_generated"] = False

        analysis_file = model_output_dir / spec.analysis_filename
        with open(analysis_file, "w") as f:
            json.dump(analysis, f, indent=2)
        generated_files.append(str(analysis_file))
        logger.info(f"✅ {spec.log_label} analysis saved: {model_name}")

    return generated_files


__all__ = [
    "FlatPayloadSpec",
    "compute_flat_payload_metrics",
    "discover_result_files",
    "generate_analysis_from_logs",
]
