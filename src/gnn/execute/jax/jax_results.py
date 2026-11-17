#!/usr/bin/env python3
"""
JAX Results Parser and Metrics Extractor for GNN Execute Pipeline.

Parses output from executed JAX POMDP scripts, extracts convergence metrics,
posterior distributions, and performance statistics for downstream analysis.
"""

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def parse_jax_output(output_path: Path) -> Optional[Dict[str, Any]]:
    """
    Parse JSON output produced by a JAX POMDP script.

    JAX scripts write a JSON file with this expected structure:
    {
        "model_name": "...",
        "timesteps": N,
        "beliefs": [[...], [...], ...],     # T x num_states
        "observations": [...],               # T observations
        "actions": [...],                    # T actions
        "free_energy": [...],               # T free energy values
        "elapsed_seconds": ...
    }

    Args:
        output_path: Path to the JSON output file from a JAX script

    Returns:
        Parsed and normalized results dict, or None if parsing fails
    """
    if not output_path.exists():
        logger.warning(f"JAX output file not found: {output_path}")
        return None

    try:
        with open(output_path) as f:
            raw = json.load(f)

        normalized: dict[str, Any] = {
            "model_name": raw.get("model_name", "unknown"),
            "framework": "JAX",
            "timesteps": raw.get("timesteps") or raw.get("T", 0),
            "beliefs": _extract_belief_trajectory(raw),
            "observations": raw.get("observations", []),
            "actions": raw.get("actions", []),
            "free_energy": _extract_free_energy_trace(raw),
            "elapsed_seconds": raw.get("elapsed_seconds") or raw.get("elapsed", 0.0),
            "device": raw.get("device", "cpu"),
            "parse_timestamp": datetime.now().isoformat(),
        }

        return normalized

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from {output_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing {output_path}: {e}")
        return None


def _extract_belief_trajectory(raw: Dict[str, Any]) -> List[List[float]]:
    """Extract belief (posterior) trajectory from raw JAX output."""
    beliefs = raw.get("beliefs") or raw.get("posteriors") or raw.get("q_s")

    if beliefs is None:
        return []

    if isinstance(beliefs, list):
        # Normalize: ensure it's a list of float lists
        result: list[Any] = []
        for belief in beliefs:
            if isinstance(belief, list):
                result.append([float(b) for b in belief])
            elif isinstance(belief, (int, float)):
                result.append([float(belief)])
        return result

    return []


def _extract_free_energy_trace(raw: Dict[str, Any]) -> List[float]:
    """Extract free energy trace from raw JAX output."""
    fe = raw.get("free_energy") or raw.get("F") or raw.get("elbo")

    if fe is None:
        return []
    if isinstance(fe, list):
        return [float(v) for v in fe if v is not None]
    try:
        return [float(fe)]
    except (TypeError, ValueError):
        return []


def compute_convergence_metrics(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute convergence metrics from parsed JAX results.

    For JAX Active Inference, convergence is assessed by:
    - Free energy stability (small changes between timesteps)
    - Belief entropy trajectory (should decrease as agent learns)
    - Prediction error magnitude over time

    Args:
        parsed: Output from parse_jax_output()

    Returns:
        Dict with convergence statistics
    """
    metrics: dict[str, Any] = {
        "total_timesteps": parsed.get("timesteps", 0),
        "final_free_energy": None,
        "mean_free_energy": None,
        "free_energy_std": None,
        "residual_last_10": None,  # Mean |ΔFE| over last 10 steps
        "converged": False,
        "belief_entropy_initial": None,
        "belief_entropy_final": None,
        "entropy_decreased": None,
    }

    fe_trace = parsed.get("free_energy", [])
    beliefs = parsed.get("beliefs", [])

    if fe_trace:
        metrics["final_free_energy"] = fe_trace[-1]
        metrics["mean_free_energy"] = sum(fe_trace) / len(fe_trace)

        if len(fe_trace) >= 2:
            mean = metrics["mean_free_energy"]
            variance = sum((v - mean) ** 2 for v in fe_trace) / len(fe_trace)
            metrics["free_energy_std"] = variance**0.5

            # Residual over last 10 timesteps
            last_10 = fe_trace[-10:] if len(fe_trace) >= 10 else fe_trace
            if len(last_10) >= 2:
                residuals = [
                    abs(last_10[i] - last_10[i - 1]) for i in range(1, len(last_10))
                ]
                metrics["residual_last_10"] = sum(residuals) / len(residuals)

                # Convergence: residual < 1% of |mean FE|
                threshold = (
                    0.01 * abs(metrics["mean_free_energy"])
                    if abs(metrics["mean_free_energy"]) > 1e-8
                    else 0.001
                )
                metrics["converged"] = metrics["residual_last_10"] < threshold

    if beliefs:
        # Compute belief entropy at first and last timestep
        if beliefs[0]:
            metrics["belief_entropy_initial"] = _categorical_entropy(beliefs[0])
        if beliefs[-1]:
            metrics["belief_entropy_final"] = _categorical_entropy(beliefs[-1])

        if (
            metrics["belief_entropy_initial"] is not None
            and metrics["belief_entropy_final"] is not None
        ):
            metrics["entropy_decreased"] = (
                metrics["belief_entropy_final"] < metrics["belief_entropy_initial"]
            )

    return metrics


def _categorical_entropy(probs: List[float]) -> float:
    """Compute entropy of a categorical distribution."""
    total = sum(probs)
    if total <= 0:
        return 0.0
    normalized = [p / total for p in probs]
    return -sum(p * math.log(p + 1e-10) for p in normalized)


def extract_array_statistics(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract summary statistics from array outputs in JAX results.

    JAX scripts may output arbitrary named arrays (belief trajectories,
    free energy curves, action sequences). This function summarizes them.

    Args:
        parsed: Output from parse_jax_output()

    Returns:
        Dict mapping array names to their statistics
    """
    stats: dict[Any, Any] = {}

    # Beliefs trajectory
    beliefs = parsed.get("beliefs", [])
    if beliefs:
        flat = [v for b in beliefs for v in b]
        if flat:
            stats["beliefs"] = {
                "shape": f"{len(beliefs)}x{len(beliefs[0]) if beliefs else 0}",
                "min": min(flat),
                "max": max(flat),
                "mean": sum(flat) / len(flat),
                "timesteps": len(beliefs),
            }

    # Free energy
    fe = parsed.get("free_energy", [])
    if fe:
        stats["free_energy"] = {
            "shape": f"{len(fe)}",
            "initial": fe[0],
            "final": fe[-1],
            "min": min(fe),
            "max": max(fe),
            "mean": sum(fe) / len(fe),
            "total_change": fe[-1] - fe[0],
        }

    # Actions
    actions = parsed.get("actions", [])
    if actions:
        action_counts: dict[Any, Any] = {}
        for a in actions:
            key = str(a)
            action_counts[key] = action_counts.get(key, 0) + 1
        stats["actions"] = {
            "count": len(actions),
            "distribution": action_counts,
            "unique_actions": len(action_counts),
        }

    # Observations
    obs = parsed.get("observations", [])
    if obs:
        obs_counts: dict[Any, Any] = {}
        for o in obs:
            key = str(o)
            obs_counts[key] = obs_counts.get(key, 0) + 1
        stats["observations"] = {"count": len(obs), "distribution": obs_counts}

    return stats


def collect_jax_results(
    output_dir: Path, model_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Collect and parse all JAX result JSON files from an output directory.

    Args:
        output_dir: Directory containing JAX output JSON files
        model_name: Optional filter by model name

    Returns:
        List of parsed result dicts with convergence metrics and array statistics
    """
    results: list[Any] = []

    if not output_dir.exists():
        logger.warning(f"Output directory not found: {output_dir}")
        return results

    # Find JAX output JSON files
    json_files = (
        list(output_dir.glob("**/*_jax*.json"))
        + list(output_dir.glob("**/*jax*.json"))
        + list(output_dir.glob("**/jax_output*.json"))
        + list(output_dir.glob("**/execution_log.json"))
    )

    # Deduplicate
    seen: set[Any] = set()
    unique_files: list[Any] = []
    for f in json_files:
        if str(f) not in seen:
            seen.add(str(f))
            unique_files.append(f)

    for json_file in unique_files:
        # Skip execution_log.json — it's metadata, not results
        if json_file.name == "execution_log.json":
            continue

        parsed = parse_jax_output(json_file)
        if parsed is None:
            continue

        if model_name and parsed.get("model_name") != model_name:
            continue

        # Enrich with computed metrics
        parsed["source_file"] = str(json_file)
        parsed["convergence_metrics"] = compute_convergence_metrics(parsed)
        parsed["array_statistics"] = extract_array_statistics(parsed)

        results.append(parsed)
        logger.info(
            f"Parsed JAX results from {json_file.name}: "
            f"timesteps={parsed['timesteps']}, "
            f"converged={parsed['convergence_metrics']['converged']}"
        )

    return results


def format_jax_report(results: List[Dict[str, Any]]) -> str:
    """
    Format JAX results as a markdown report.

    Args:
        results: List of parsed results from collect_jax_results()

    Returns:
        Markdown-formatted report string
    """
    if not results:
        return "# JAX Inference Results\n\nNo results found.\n"

    lines: list[Any] = [
        "# JAX Active Inference Results\n",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"**Models analyzed**: {len(results)}\n\n",
    ]

    for r in results:
        model = r.get("model_name", "unknown")
        conv = r.get("convergence_metrics", {})
        arr_stats = r.get("array_statistics", {})
        device = r.get("device", "cpu")
        elapsed = r.get("elapsed_seconds", 0)

        lines.append(f"## {model}\n")
        lines.append(f"- **Device**: {device}")
        lines.append(f"- **Elapsed**: {elapsed:.2f}s")
        lines.append(f"- **Timesteps**: {conv.get('total_timesteps', 'N/A')}")
        lines.append(f"- **Converged**: {'Yes' if conv.get('converged') else 'No'}")

        if conv.get("final_free_energy") is not None:
            lines.append(f"- **Final Free Energy**: {conv['final_free_energy']:.4f}")
        if conv.get("residual_last_10") is not None:
            lines.append(
                f"- **Residual (last 10 steps)**: {conv['residual_last_10']:.6f}"
            )
        if conv.get("belief_entropy_initial") is not None:
            h_i = conv["belief_entropy_initial"]
            h_f = conv.get("belief_entropy_final", "N/A")
            direction = "down" if conv.get("entropy_decreased") else "up"
            lines.append(f"- **Belief entropy**: {h_i:.4f} -> {h_f:.4f} ({direction})")

        # Array statistics summary
        fe_stats = arr_stats.get("free_energy")
        if fe_stats:
            lines.append("\n### Free Energy Trajectory")
            lines.append(f"- Initial: {fe_stats['initial']:.4f}")
            lines.append(f"- Final: {fe_stats['final']:.4f}")
            lines.append(f"- Total change: {fe_stats['total_change']:.4f}")

        action_stats = arr_stats.get("actions")
        if action_stats:
            lines.append("\n### Action Distribution")
            for action, count in sorted(action_stats["distribution"].items()):
                pct = 100 * count / action_stats["count"]
                lines.append(f"- Action {action}: {count} times ({pct:.1f}%)")

        lines.append("")

    return "\n".join(lines)
