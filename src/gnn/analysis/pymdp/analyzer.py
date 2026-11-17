#!/usr/bin/env python3
"""
PyMDP analyzer for the pymdp_simulation_v1 execution schema.

Analysis is intentionally downstream of execution: Step 12 writes raw
``simulation_results.json`` files, and Step 16 reads only the current schema.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .visualizer import save_all_visualizations

logger = logging.getLogger("analysis.pymdp")


def _slug(name: str) -> str:
    """Handle slug for internal callers."""
    return name.replace(" ", "_").replace("/", "_")


def _first_sequence(mapping: Dict[str, Any], preferred_key: str) -> List[Any]:
    """Handle first sequence for internal callers."""
    preferred = mapping.get(preferred_key)
    if isinstance(preferred, list):
        return preferred
    for value in mapping.values():
        if isinstance(value, list):
            return value
    return []


def _load_json(path: Path) -> Dict[str, Any] | None:
    """Load json."""
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse %s: %s", path, exc)
        return None
    except OSError as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return None

    return data if isinstance(data, dict) else None


def _iter_execution_summary_paths(execution_results_dir: Path) -> Iterable[Path]:
    """Handle iter execution summary paths for internal callers."""
    yield execution_results_dir / "summaries" / "execution_summary.json"
    yield execution_results_dir / "execution_summary.json"


def _result_paths_from_execution_summary(execution_results_dir: Path) -> List[Path]:
    """Handle result paths from execution summary for internal callers."""
    for summary_path in _iter_execution_summary_paths(execution_results_dir):
        if not summary_path.exists():
            continue

        summary = _load_json(summary_path)
        if summary is None:
            return []

        paths: List[Path] = []
        for detail in summary.get("execution_details", []):
            if not isinstance(detail, dict):
                continue
            if str(detail.get("framework", "")).lower() != "pymdp":
                continue
            if not detail.get("success") or detail.get("skipped"):
                continue

            structured_result_file = detail.get("structured_result_file")
            if structured_result_file:
                structured_path = Path(structured_result_file)
                if not structured_path.is_absolute():
                    structured_path = Path.cwd() / structured_path
                structured_payload = (
                    _load_json(structured_path) if structured_path.exists() else None
                )
                for collected_path in (
                    (structured_payload or {})
                    .get("collected_outputs", {})
                    .get("simulation_data", [])
                ):
                    result_path = Path(collected_path)
                    if not result_path.is_absolute():
                        result_path = Path.cwd() / result_path
                    if result_path.exists():
                        paths.append(result_path)

            implementation_dir = detail.get("implementation_directory")
            if implementation_dir:
                impl_path = Path(implementation_dir)
                if not impl_path.is_absolute():
                    impl_path = Path.cwd() / impl_path
                sim_dir = impl_path / "simulation_data"
                if sim_dir.exists():
                    paths.extend(sim_dir.glob("*_simulation_results.json"))

        return _unique_existing_paths(paths)

    return []


def _unique_existing_paths(paths: Iterable[Path]) -> List[Path]:
    """Handle unique existing paths for internal callers."""
    unique: List[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen or not path.exists():
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _discover_pymdp_result_files(execution_results_dir: Path) -> List[Path]:
    """Discover pymdp result files."""
    summary_paths = _result_paths_from_execution_summary(execution_results_dir)
    if summary_paths:
        return summary_paths

    results: List[Path] = []
    for sim_dir in execution_results_dir.glob("**/pymdp/simulation_data"):
        named_results = list(sim_dir.glob("*_simulation_results.json"))
        current_named = [
            path
            for path in named_results
            if (_load_json(path) or {}).get("schema_version") == "pymdp_simulation_v1"
        ]
        if current_named:
            results.extend(current_named)
            continue

        bare_result = sim_dir / "simulation_results.json"
        if (
            bare_result.exists()
            and (_load_json(bare_result) or {}).get("schema_version")
            == "pymdp_simulation_v1"
        ):
            results.append(bare_result)

    if not results:
        for path in execution_results_dir.glob("**/simulation_results.json"):
            payload = _load_json(path) or {}
            if (
                str(payload.get("framework", "")).lower() == "pymdp"
                and payload.get("schema_version") == "pymdp_simulation_v1"
            ):
                results.append(path)

    return _unique_existing_paths(results)


def generate_analysis_from_logs(
    execution_results_dir: Path,
    output_dir: Path,
    verbose: bool = False,
) -> List[str]:
    """
    Generate PyMDP visualizations from pymdp_simulation_v1 result files.
    """
    generated_files: List[str] = []

    if not execution_results_dir.exists():
        logger.warning(
            "Execution results directory not found: %s", execution_results_dir
        )
        return generated_files

    output_dir.mkdir(parents=True, exist_ok=True)
    results_files = _discover_pymdp_result_files(execution_results_dir)
    if not results_files:
        logger.warning(
            "No pymdp_simulation_v1 result files found under %s", execution_results_dir
        )
        return generated_files

    for results_file in results_files:
        data = _load_json(results_file)
        if data is None:
            continue

        if str(data.get("framework", "")).lower() != "pymdp":
            if verbose:
                logger.debug("Skipping non-PyMDP result %s", results_file)
            continue
        if data.get("schema_version") != "pymdp_simulation_v1":
            logger.error(
                "Skipping %s: unsupported PyMDP schema %r",
                results_file,
                data.get("schema_version"),
            )
            continue

        model_name = data.get("model_name", results_file.parent.name)
        model_viz_dir = output_dir / _slug(str(model_name))
        model_viz_dir.mkdir(parents=True, exist_ok=True)

        beliefs = _first_sequence(
            data.get("beliefs_by_factor", {}) or {}, "joint_state"
        )
        true_states = _first_sequence(
            data.get("hidden_states_by_factor", {}) or {}, "joint_state"
        )
        observations = _first_sequence(
            data.get("observations_by_modality", {}) or {},
            "joint_observation",
        )
        actions = _first_sequence(
            data.get("actions_by_control_factor", {}) or {},
            "joint_action",
        )
        efe_history = data.get("expected_free_energy", [])
        vfe_history = data.get("variational_free_energy", [])
        params = data.get("model_parameters", {}) or {}

        if not beliefs:
            logger.error(
                "Skipping %s: pymdp_simulation_v1 payload has no beliefs", results_file
            )
            continue

        viz_results: dict[str, Any] = {
            "states": true_states,
            "beliefs": beliefs,
            "actions": actions,
            "observations": observations,
            "metrics": {
                "expected_free_energy": efe_history,
                "variational_free_energy": vfe_history,
                "belief_confidence": [max(b) for b in beliefs] if beliefs else [],
            },
            "num_states": int(params.get("num_states", len(beliefs[0]))),
        }

        viz_files_map = save_all_visualizations(
            simulation_results=viz_results,
            output_dir=model_viz_dir,
            config={"save_dir": model_viz_dir},
        )
        for filepath in viz_files_map.values():
            generated_files.append(str(filepath))

        cumulative_pref = (data.get("metrics", {}) or {}).get(
            "cumulative_preference", []
        )
        if cumulative_pref:
            pref_file = _plot_cumulative_preference(
                cumulative_pref, str(model_name), model_viz_dir
            )
            if pref_file:
                generated_files.append(str(pref_file))

    logger.info(
        "PyMDP analysis complete: generated %d visualization files",
        len(generated_files),
    )
    return generated_files


def _plot_cumulative_preference(
    cumulative_pref: List[float],
    model_name: str,
    model_viz_dir: Path,
) -> Path | None:
    """Plot cumulative preference."""
    try:
        from ..viz_base import MATPLOTLIB_AVAILABLE, np, plt

        if not MATPLOTLIB_AVAILABLE or plt is None:
            return None
        fig, ax = plt.subplots(figsize=(12, 5))
        x = range(len(cumulative_pref))
        ax.step(x, cumulative_pref, where="mid", linewidth=2, color="#2ECC71")
        ax.fill_between(x, cumulative_pref, step="mid", alpha=0.2, color="#2ECC71")
        ax2 = ax.twinx()
        ax2.plot(x, np.cumsum(cumulative_pref), "o-", color="#E74C3C", linewidth=2.5)
        ax.set_xlabel("Time Step", fontweight="bold")
        ax.set_ylabel("Per-Step Preference", fontweight="bold", color="#2ECC71")
        ax2.set_ylabel("Cumulative Preference", fontweight="bold", color="#E74C3C")
        ax.set_title(f"PyMDP Preference Accumulation - {model_name}", fontweight="bold")
        ax.grid(True, alpha=0.3)
        pref_file = model_viz_dir / "cumulative_preference.png"
        plt.savefig(str(pref_file), dpi=300, bbox_inches="tight")
        plt.close(fig)
        return pref_file
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to generate cumulative preference plot: %s", exc)
        return None
