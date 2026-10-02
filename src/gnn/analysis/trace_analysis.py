#!/usr/bin/env python3
"""
Trace Analysis Sub-module

Framework-agnostic analysis of simulation traces, free energy dynamics,
policy convergence, state distributions, and cross-framework comparison.

Extracted from post_simulation.py for maintainability.
"""

import logging
from typing import Any, Dict, List

import numpy as np

from .comparison_contract import (
    performance_admission,
    prepare_scientific_record,
    scientific_comparisons,
)
from .result_adapter import categorical_trace

logger = logging.getLogger(__name__)


def analyze_simulation_traces(
    traces: List[Any], framework: str, model_name: str
) -> Dict[str, Any]:
    """
    Analyze simulation traces (state/observation/action trajectories).

    Args:
        traces: List of trace data (format depends on framework)
        framework: Framework name
        model_name: Model name

    Returns:
        Dictionary with trace analysis results
    """
    try:
        analysis: dict[str, Any] = {
            "framework": framework,
            "model_name": model_name,
            "trace_count": len(traces),
            "trace_lengths": [],
            "state_entropy": [],
            "observation_diversity": [],
            "action_distribution": {},
            "convergence_metrics": {},
        }

        if not traces:
            return analysis

        # Extract trace lengths
        for trace in traces:
            if isinstance(trace, (list, tuple)):
                analysis["trace_lengths"].append(len(trace))
            elif isinstance(trace, dict):
                analysis["trace_lengths"].append(len(trace.get("states", [])))

        # Calculate statistics
        if analysis["trace_lengths"]:
            analysis["avg_trace_length"] = np.mean(analysis["trace_lengths"])
            analysis["max_trace_length"] = np.max(analysis["trace_lengths"])
            analysis["min_trace_length"] = np.min(analysis["trace_lengths"])

        return analysis

    except Exception as e:
        logger.error(f"Error analyzing simulation traces: {e}")
        return {
            "framework": framework,
            "model_name": model_name,
            "status": "failed",
            "error": str(e),
        }


def analyze_free_energy(
    free_energy_values: List[float], framework: str, model_name: str
) -> Dict[str, Any]:
    """
    Analyze free energy dynamics.

    Args:
        free_energy_values: List of free energy values over time
        framework: Framework name
        model_name: Model name

    Returns:
        Dictionary with free energy analysis results
    """
    try:
        analysis: dict[str, Any] = {
            "framework": framework,
            "model_name": model_name,
            "free_energy_count": len(free_energy_values),
            "free_energy_values": free_energy_values,
        }

        if not free_energy_values:
            return analysis

        fe_array = np.asarray(free_energy_values, dtype=float)
        if fe_array.ndim != 1 or not np.isfinite(fe_array).all():
            raise ValueError(
                "free energy must be a finite scalar series; policy arrays require explicit reduction semantics"
            )

        # Calculate statistics
        analysis["mean_free_energy"] = float(np.mean(fe_array))
        analysis["std_free_energy"] = float(np.std(fe_array))
        analysis["min_free_energy"] = float(np.min(fe_array))
        analysis["max_free_energy"] = float(np.max(fe_array))

        # Calculate trend (decreasing = good for Active Inference)
        if len(fe_array) > 1:
            trend = np.polyfit(range(len(fe_array)), fe_array, 1)[0]
            analysis["free_energy_trend"] = float(trend)
            analysis["free_energy_decreasing"] = trend < 0

        # Calculate convergence (variance in last 20% of values)
        if len(fe_array) > 5:
            last_portion = fe_array[int(0.8 * len(fe_array)) :]
            analysis["convergence_variance"] = float(np.var(last_portion))
            analysis["converged"] = analysis["convergence_variance"] < 0.1

        # Normalized convergence metric: relative entropy / scale-invariant decay
        if len(fe_array) > 1:
            f0 = abs(fe_array[0]) + 1e-8
            rel_decay = (fe_array[-1] - fe_array[0]) / f0
            analysis["relative_decay"] = float(rel_decay)
            analysis["normalized_converged"] = bool(
                abs(rel_decay) < 1.0 or analysis.get("converged", False)
            )

        return analysis

    except Exception as e:
        logger.error(f"Error analyzing free energy: {e}")
        return {
            "framework": framework,
            "model_name": model_name,
            "status": "failed",
            "error": str(e),
        }


def analyze_policy_convergence(
    policy_traces: List[Any], framework: str, model_name: str
) -> Dict[str, Any]:
    """
    Analyze policy evolution and convergence.

    Args:
        policy_traces: List of policy distributions over time
        framework: Framework name
        model_name: Model name

    Returns:
        Dictionary with policy convergence analysis
    """
    try:
        analysis: dict[str, Any] = {
            "framework": framework,
            "model_name": model_name,
            "policy_count": len(policy_traces),
            "policy_entropy": [],
            "policy_stability": {},
        }

        if not policy_traces:
            return analysis

        # Values must already be probabilities; analysis cannot repair them.
        for policy_array in categorical_trace(
            policy_traces, name="policy distributions"
        ):
            positive = policy_array[policy_array > 0]
            entropy = -np.sum(positive * np.log(positive))
            analysis["policy_entropy"].append(float(entropy))

        # Calculate stability (variance in policy entropy)
        if analysis["policy_entropy"]:
            analysis["policy_stability"]["entropy_mean"] = float(
                np.mean(analysis["policy_entropy"])
            )
            analysis["policy_stability"]["entropy_std"] = float(
                np.std(analysis["policy_entropy"])
            )
            analysis["policy_stability"]["stable"] = (
                analysis["policy_stability"]["entropy_std"] < 0.1
            )

        return analysis

    except Exception as e:
        logger.error(f"Error analyzing policy convergence: {e}")
        return {
            "framework": framework,
            "model_name": model_name,
            "status": "failed",
            "error": str(e),
        }


def analyze_state_distributions(
    state_traces: List[Any], framework: str, model_name: str
) -> Dict[str, Any]:
    """
    Analyze belief state distributions.

    Args:
        state_traces: List of state distributions over time
        framework: Framework name
        model_name: Model name

    Returns:
        Dictionary with state distribution analysis
    """
    try:
        analysis: dict[str, Any] = {
            "framework": framework,
            "model_name": model_name,
            "state_count": len(state_traces),
            "state_entropy": [],
            "state_diversity": {},
        }

        if not state_traces:
            return analysis

        for state_array in categorical_trace(state_traces, name="state distributions"):
            positive = state_array[state_array > 0]
            entropy = -np.sum(positive * np.log(positive))
            analysis["state_entropy"].append(float(entropy))

        # Calculate diversity metrics
        if analysis["state_entropy"]:
            analysis["state_diversity"]["mean_entropy"] = float(
                np.mean(analysis["state_entropy"])
            )
            analysis["state_diversity"]["std_entropy"] = float(
                np.std(analysis["state_entropy"])
            )

        return analysis

    except Exception as e:
        logger.error(f"Error analyzing state distributions: {e}")
        return {
            "framework": framework,
            "model_name": model_name,
            "status": "failed",
            "error": str(e),
        }


def compare_framework_results(
    framework_results: Dict[str, Dict[str, Any]], model_name: str
) -> Dict[str, Any]:
    """
    Compare results across different frameworks.

    Args:
        framework_results: Dictionary mapping framework names to their results
        model_name: Model name

    Returns:
        Dictionary with cross-framework comparison
    """
    try:
        comparison: dict[str, Any] = {
            "model_name": model_name,
            "frameworks_compared": list(framework_results.keys()),
            "framework_count": len(framework_results),
            "comparisons": {},
        }

        if len(framework_results) < 2:
            comparison["message"] = "Need at least 2 frameworks for comparison"
            return comparison

        records = {
            framework: [prepare_scientific_record(results)]
            for framework, results in framework_results.items()
        }
        comparison["comparisons"].update(scientific_comparisons(records))

        # Descriptive runtime observations remain useful without establishing
        # matched-corpus performance. A ranking requires explicit custody.
        exec_times = {
            framework: results["execution_time"]
            for framework, results in framework_results.items()
            if isinstance(results.get("execution_time"), (int, float))
            and not isinstance(results.get("execution_time"), bool)
            and np.isfinite(results["execution_time"])
            and results["execution_time"] >= 0
        }
        if exec_times:
            comparison["comparisons"]["execution_time"] = exec_times
            reasons = performance_admission(records)
            if reasons:
                comparison["comparisons"]["unavailable_metrics"] = {
                    "fastest_execution": reasons
                }
            elif len(exec_times) == len(framework_results):
                fastest_framework = min(exec_times.items(), key=lambda item: item[1])
                comparison["comparisons"]["fastest_execution"] = {
                    "framework": fastest_framework[0],
                    "time": fastest_framework[1],
                }

        # Compare success rates
        success_rates: dict[Any, Any] = {}
        for framework, results in framework_results.items():
            success_rates[framework] = results.get("success", False)

        if success_rates:
            comparison["comparisons"]["success_rates"] = success_rates

        return comparison

    except Exception as e:
        logger.error(f"Error comparing framework results: {e}")
        return {"model_name": model_name, "error": str(e)}
