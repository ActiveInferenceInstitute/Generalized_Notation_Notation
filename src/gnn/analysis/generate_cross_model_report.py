#!/usr/bin/env python3
"""
Cross-Model Comparison Report Generator

Generates a unified markdown report summarising how each POMDP model performs
across all frameworks (PyMDP, JAX, RxInfer, ActiveInference.jl, DisCoPy, PyTorch,
NumPyro, Stan, bnlearn).

Reads simulation_results.json files from 12_execute_output and
post_simulation_analysis.json files from 16_analysis_output/cross_framework.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

FRAMEWORK_ORDER: list[Any] = [
    "pymdp",
    "jax",
    "rxinfer",
    "activeinference_jl",
    "discopy",
    "pytorch",
    "numpyro",
    "cpomdp",
]
FRAMEWORK_LABELS: dict[str, Any] = {
    "pymdp": "PyMDP",
    "jax": "JAX",
    "rxinfer": "RxInfer",
    "activeinference_jl": "ActiveInf.jl",
    "discopy": "DisCoPy",
    "pytorch": "PyTorch",
    "numpyro": "NumPyro",
}


# ---------------------------------------------------------------------------
# Data collection helpers
# ---------------------------------------------------------------------------


def _collect_simulation_data(
    execution_dir: Path,
    allowed_frameworks: Optional[set[str]] = None,
    allowed_model_names: Optional[set[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Collect simulation results keyed by (model_name, framework)."""
    data: Dict[str, Dict[str, Any]] = {}  # model -> framework -> result

    for sim_file in sorted(execution_dir.rglob("*simulation_results.json")):
        try:
            with open(sim_file, "r") as f:
                result = json.load(f)

            # Determine framework from path parts
            parts = sim_file.parts
            framework = "unknown"
            for part in parts:
                if part in FRAMEWORK_ORDER:
                    framework = part
                    break
            if allowed_frameworks and framework not in allowed_frameworks:
                continue

            # Determine model name from path (parent directories above framework)
            model_name = None
            for i, part in enumerate(parts):
                if part in FRAMEWORK_ORDER:
                    # model is the directory just before the framework
                    if i >= 2:
                        model_name = parts[i - 1]
                    break
            if not model_name:
                model_name = result.get("model_name", sim_file.parent.parent.name)
            if allowed_model_names and model_name not in allowed_model_names:
                continue

            if model_name not in data:
                data[model_name] = {}
            if framework not in data[model_name]:
                data[model_name][framework] = result

        except Exception as e:
            logger.debug(f"Skipping {sim_file}: {e}")

    return data


def _collect_execution_times(
    execution_dir: Path,
    allowed_frameworks: Optional[set[str]] = None,
    allowed_model_names: Optional[set[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Collect execution times from execution_summary.json."""
    times: Dict[str, Dict[str, float]] = {}
    summary_file = execution_dir / "summaries" / "execution_summary.json"
    if not summary_file.exists():
        summary_file = execution_dir / "execution_summary.json"
    if not summary_file.exists():
        return times

    try:
        with open(summary_file, "r") as f:
            summary = json.load(f)

        for entry in summary.get("execution_results", summary.get("results", [])):
            script = entry.get("script", "")
            model_name = None
            framework = None

            # Derive model_name and framework from the script path
            for fw in FRAMEWORK_ORDER:
                if f"/{fw}/" in script or f"\\{fw}\\" in script:
                    framework = fw
                    break

            if framework:
                if allowed_frameworks and framework not in allowed_frameworks:
                    continue
                # model is the directory before the framework dir
                parts = Path(script).parts
                for i, part in enumerate(parts):
                    if part == framework and i >= 1:
                        model_name = parts[i - 1]
                        break

            if model_name and framework:
                if allowed_model_names and model_name not in allowed_model_names:
                    continue
                if model_name not in times:
                    times[model_name] = {}
                times[model_name][framework] = entry.get("execution_time", 0.0)

    except Exception as e:
        logger.debug(f"Failed to parse execution summary: {e}")

    return times


# ---------------------------------------------------------------------------
# Metric extractors
# ---------------------------------------------------------------------------


def _mean_belief_confidence(result: Dict[str, Any]) -> Optional[float]:
    """Compute mean of max-belief across timesteps."""
    beliefs = result.get("beliefs") or result.get("simulation_trace", {}).get("beliefs")
    if not beliefs:
        return None
    try:
        arr = np.array(beliefs, dtype=float)
        if arr.ndim == 2:
            return float(np.mean(np.max(arr, axis=1)))
        return None
    except (TypeError, ValueError) as exc:
        logger.debug("mean_belief_confidence metric failed: %s", exc)
        return None


def _mean_efe(result: Dict[str, Any]) -> Optional[float]:
    """Compute mean EFE (selected action's EFE) across timesteps."""
    efe = result.get("efe_history") or result.get("simulation_trace", {}).get(
        "efe_history"
    )
    metrics = result.get("metrics", {})
    if not efe and isinstance(metrics, dict):
        efe = metrics.get("expected_free_energy")
    if not efe:
        return None

    actions = result.get("actions") or result.get("simulation_trace", {}).get("actions")

    try:
        efe_arr = np.array(efe, dtype=float)
        if efe_arr.ndim == 1:
            return float(np.mean(efe_arr))
        elif efe_arr.ndim == 2 and actions:
            # Per-action EFE matrix: extract chosen action's EFE per step
            selected: list[Any] = []
            for t, a in enumerate(actions):
                if t < len(efe_arr):
                    a_idx = int(a) if int(a) < efe_arr.shape[1] else 0
                    selected.append(efe_arr[t, a_idx])
            return float(np.mean(selected)) if selected else float(np.mean(efe_arr))
        return None
    except (TypeError, ValueError, IndexError, KeyError, ZeroDivisionError) as exc:
        logger.debug("mean_efe metric failed: %s", exc)
        return None


def _mean_belief_entropy(result: Dict[str, Any]) -> Optional[float]:
    """Compute mean Shannon entropy of beliefs across timesteps."""
    beliefs = result.get("beliefs") or result.get("simulation_trace", {}).get("beliefs")
    if not beliefs:
        return None
    try:
        arr = np.array(beliefs, dtype=float)
        if arr.ndim != 2:
            return None
        # Shannon entropy per timestep
        arr = np.clip(arr, 1e-16, None)
        row_sums = arr.sum(axis=1, keepdims=True)
        arr = arr / row_sums
        entropies = -np.sum(arr * np.log(arr), axis=1)
        return float(np.mean(entropies))
    except (TypeError, ValueError, FloatingPointError) as exc:
        logger.debug("mean_belief_entropy metric failed: %s", exc)
        return None


def _action_diversity(result: Dict[str, Any]) -> Optional[float]:
    """Fraction of unique actions used."""
    actions = result.get("actions") or result.get("simulation_trace", {}).get("actions")
    if not actions:
        return None
    try:
        unique = len({int(a) for a in actions})
        return round(unique / len(actions), 3)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        logger.debug("action_diversity metric failed: %s", exc)
        return None


def _timestep_count(result: Dict[str, Any]) -> Optional[int]:
    """Number of simulation timesteps."""
    n = result.get("num_timesteps")
    if n:
        return int(n)
    beliefs = result.get("beliefs") or result.get("simulation_trace", {}).get("beliefs")
    if beliefs:
        return len(beliefs)
    return None


def _validation_status(result: Dict[str, Any]) -> str:
    """Return ✅ or ❌ based on validation dict."""
    v = result.get("validation", {})
    if not v:
        return "—"
    all_pass = all(val is True for val in v.values() if isinstance(val, bool))
    return "✅" if all_pass else "❌"


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_cross_model_report(
    execution_dir: Path,
    analysis_dir: Path,
    output_path: Path,
    allowed_frameworks: Optional[set[str]] = None,
    allowed_model_names: Optional[set[str]] = None,
) -> str:
    """
    Generate a unified cross-model comparison markdown report.

    Args:
        execution_dir: Path to output/12_execute_output
        analysis_dir: Path to output/16_analysis_output
        output_path: Path to write the markdown report
        allowed_frameworks: Optional normalized framework names from the current run
        allowed_model_names: Optional model directory names from the current run

    Returns:
        Path to the generated report file (as string)
    """
    logger.info("Generating cross-model comparison report...")
    sim_data = _collect_simulation_data(
        execution_dir,
        allowed_frameworks=allowed_frameworks,
        allowed_model_names=allowed_model_names,
    )
    exec_times = _collect_execution_times(
        execution_dir,
        allowed_frameworks=allowed_frameworks,
        allowed_model_names=allowed_model_names,
    )

    if not sim_data:
        logger.warning("No simulation data found — skipping cross-model report")
        return ""

    models = sorted(sim_data.keys())
    frameworks = [
        fw
        for fw in FRAMEWORK_ORDER
        if not allowed_frameworks or fw in allowed_frameworks
    ]

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: List[str] = []
    lines.append("# Cross-Model Comparison Report\n")
    lines.append(f"**Generated:** {timestamp}\n")
    lines.append(f"**Models:** {len(models)} | **Frameworks:** {len(frameworks)}\n")

    # ---- Summary table ----
    lines.append("## Summary Matrix\n")
    header = (
        "| Model |"
        + "|".join(f" {FRAMEWORK_LABELS.get(fw, fw)} " for fw in frameworks)
        + "|"
    )
    sep = "|" + "|".join("---" for _ in range(len(frameworks) + 1)) + "|"
    lines.append(header)
    lines.append(sep)

    for model in models:
        row = f"| **{model}** |"
        for fw in frameworks:
            result = sim_data.get(model, {}).get(fw)
            if result:
                conf = _mean_belief_confidence(result)
                val = _validation_status(result)
                conf_str = f"{conf:.3f}" if conf is not None else "—"
                row += f" {val} {conf_str} |"
            else:
                row += " — |"
        lines.append(row)

    lines.append("")
    lines.append(
        "> Values show validation status and mean belief confidence (max belief per timestep).\n"
    )

    # ---- EFE Comparison ----
    lines.append("## Expected Free Energy Comparison\n")
    header = (
        "| Model |"
        + "|".join(f" {FRAMEWORK_LABELS.get(fw, fw)} " for fw in frameworks)
        + "|"
    )
    lines.append(header)
    lines.append(sep)

    for model in models:
        row = f"| **{model}** |"
        for fw in frameworks:
            result = sim_data.get(model, {}).get(fw)
            if result:
                efe = _mean_efe(result)
                row += f" {efe:.4f} |" if efe is not None else " — |"
            else:
                row += " — |"
        lines.append(row)

    lines.append("")

    # ---- Belief Entropy Comparison ----
    lines.append("## Belief Entropy Comparison\n")
    lines.append("Mean Shannon entropy of posterior beliefs (lower = more certain).\n")
    header = (
        "| Model |"
        + "|".join(f" {FRAMEWORK_LABELS.get(fw, fw)} " for fw in frameworks)
        + "|"
    )
    lines.append(header)
    lines.append(sep)

    for model in models:
        row = f"| **{model}** |"
        for fw in frameworks:
            result = sim_data.get(model, {}).get(fw)
            if result:
                ent = _mean_belief_entropy(result)
                row += f" {ent:.4f} |" if ent is not None else " — |"
            else:
                row += " — |"
        lines.append(row)

    lines.append("")

    # ---- Performance Comparison ----
    lines.append("## Execution Time (seconds)\n")
    header = (
        "| Model |"
        + "|".join(f" {FRAMEWORK_LABELS.get(fw, fw)} " for fw in frameworks)
        + "|"
    )
    lines.append(header)
    lines.append(sep)

    for model in models:
        row = f"| **{model}** |"
        for fw in frameworks:
            t = exec_times.get(model, {}).get(fw)
            if t is not None:
                row += f" {t:.2f} |"
            else:
                # Try to infer from sim data if available
                row += " — |"
        lines.append(row)

    lines.append("")

    # ---- Per-model details ----
    lines.append("## Per-Model Details\n")

    for model in models:
        lines.append(f"### {model}\n")
        model_data = sim_data.get(model, {})

        # Quick stats table
        lines.append(
            "| Framework | Steps | Confidence | EFE (mean) | Entropy | Action Diversity | Validation |"
        )
        lines.append(
            "|-----------|-------|------------|------------|---------|------------------|------------|"
        )

        for fw in frameworks:
            result = model_data.get(fw)
            if not result:
                lines.append(
                    f"| {FRAMEWORK_LABELS.get(fw, fw)} | — | — | — | — | — | — |"
                )
                continue

            steps = _timestep_count(result)
            conf = _mean_belief_confidence(result)
            efe = _mean_efe(result)
            ent = _mean_belief_entropy(result)
            div = _action_diversity(result)
            val = _validation_status(result)

            parts_line: list[Any] = [f"| {FRAMEWORK_LABELS.get(fw, fw)}"]
            parts_line.append(f" {steps or '—'}")
            parts_line.append(f" {conf:.4f}" if conf is not None else " —")
            parts_line.append(f" {efe:.4f}" if efe is not None else " —")
            parts_line.append(f" {ent:.4f}" if ent is not None else " —")
            parts_line.append(f" {div:.3f}" if div is not None else " —")
            parts_line.append(f" {val}")
            lines.append(" |".join(parts_line) + " |")

        lines.append("")

        # Interpretation
        fw_with_data = [fw for fw in frameworks if fw in model_data]
        if fw_with_data:
            confs = {fw: _mean_belief_confidence(model_data[fw]) for fw in fw_with_data}
            confs_valid = {fw: c for fw, c in confs.items() if c is not None}
            if confs_valid:
                best_fw = max(confs_valid, key=lambda fw: confs_valid[fw])
                worst_fw = min(confs_valid, key=lambda fw: confs_valid[fw])
                lines.append(
                    f"**Highest confidence:** {FRAMEWORK_LABELS.get(best_fw, best_fw)} ({confs_valid[best_fw]:.4f}) | "
                    f"**Lowest:** {FRAMEWORK_LABELS.get(worst_fw, worst_fw)} ({confs_valid[worst_fw]:.4f})\n"
                )

    # ---- Cross-Model Observations ----
    lines.append("## Cross-Model Observations\n")

    # Find model with fastest convergence (highest confidence)
    model_confs: dict[Any, Any] = {}
    for model in models:
        all_confs: list[Any] = []
        for fw in frameworks:
            result = sim_data.get(model, {}).get(fw)
            if result:
                c = _mean_belief_confidence(result)
                if c is not None:
                    all_confs.append(c)
        if all_confs:
            model_confs[model] = np.mean(all_confs)

    if model_confs:
        best_model = max(model_confs, key=lambda model: model_confs[model])
        worst_model = min(model_confs, key=lambda model: model_confs[model])
        lines.append(
            f"- **Highest avg. confidence:** {best_model} ({model_confs[best_model]:.4f})"
        )
        lines.append(
            f"- **Lowest avg. confidence:** {worst_model} ({model_confs[worst_model]:.4f})"
        )

    # Framework speed comparison
    fw_speeds: Dict[str, List[float]] = {}
    for model in models:
        for fw in frameworks:
            t = exec_times.get(model, {}).get(fw)
            if t:
                fw_speeds.setdefault(fw, []).append(t)
    if fw_speeds:
        lines.append("")
        lines.append("**Average execution times across all models:**\n")
        for fw in frameworks:
            if fw in fw_speeds:
                avg = np.mean(fw_speeds[fw])
                lines.append(f"- {FRAMEWORK_LABELS.get(fw, fw)}: {avg:.2f}s")

    lines.append("")
    lines.append("---\n")
    lines.append(f"*Generated by GNN Analysis Pipeline — {timestamp}*\n")

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_content = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report_content)

    logger.info(f"Cross-model comparison report written to {output_path}")
    return str(output_path)
