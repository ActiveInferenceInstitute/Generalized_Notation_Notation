#!/usr/bin/env python3
"""
Post-Simulation Analysis Module

This module provides generic post-simulation analysis methods that work across
all frameworks (PyMDP, RxInfer.jl, ActiveInference.jl, JAX, DisCoPy, PyTorch,
NumPyro, Stan, bnlearn).

Implementation is split across sub-modules for maintainability:
- trace_analysis: Framework-agnostic trace, free energy, policy, state analysis
- framework_extractors: Framework-specific data extraction (extract_*_data)
- math_utils: Active Inference statistical functions (entropy, KL, VFE, EFE)
- visualizations: All plotting, animation, and dashboard generation

This file is the public facade for the analysis sub-modules listed above.
"""

from typing import Any

__all__: list[Any] = [
    # trace_analysis
    "analyze_simulation_traces",
    "analyze_free_energy",
    "analyze_policy_convergence",
    "analyze_state_distributions",
    "compare_framework_results",
    # framework_extractors
    "extract_pymdp_data",
    "extract_rxinfer_data",
    "extract_activeinference_jl_data",
    "extract_jax_data",
    "extract_jax_kronecker_data",
    "extract_numpyro_data",
    "extract_pytorch_data",
    "extract_bnlearn_data",
    "extract_discopy_data",
    # math_utils
    "compute_shannon_entropy",
    "compute_kl_divergence",
    "compute_variational_free_energy",
    "compute_expected_free_energy",
    "compute_information_gain",
    "analyze_active_inference_metrics",
    # visualizations
    "plot_belief_evolution",
    "animate_belief_evolution",
    "visualize_all_framework_outputs",
    "generate_belief_heatmaps",
    "generate_action_analysis",
    "generate_free_energy_plots",
    "generate_observation_analysis",
    "generate_cross_framework_comparison",
    "generate_unified_framework_dashboard",
    # local
    "analyze_execution_results",
]

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Re-export from trace_analysis
# Re-export from framework_extractors
from .framework_extractors import (
    extract_activeinference_jl_data,
    extract_bnlearn_data,
    extract_discopy_data,
    extract_jax_data,
    extract_jax_kronecker_data,
    extract_numpyro_data,
    extract_pymdp_data,
    extract_pytorch_data,
    extract_rxinfer_data,
)

# Re-export from math_utils
from .math_utils import (
    analyze_active_inference_metrics,
    compute_expected_free_energy,
    compute_information_gain,
    compute_kl_divergence,
    compute_shannon_entropy,
    compute_variational_free_energy,
)
from .trace_analysis import (
    analyze_free_energy,
    analyze_policy_convergence,
    analyze_simulation_traces,
    analyze_state_distributions,
    compare_framework_results,
)

# Re-export from visualizations
from .visualizations import (
    animate_belief_evolution,
    generate_action_analysis,
    generate_belief_heatmaps,
    generate_cross_framework_comparison,
    generate_free_energy_plots,
    generate_observation_analysis,
    generate_unified_framework_dashboard,
    plot_belief_evolution,
    visualize_all_framework_outputs,
)


def analyze_execution_results(
    execution_results_dir: Path,
    model_name: Optional[str] = None,
    allowed_frameworks: Optional[set[str]] = None,
) -> Dict[str, Any]:
    """
    Analyze execution results from the current execution scope.

    Args:
        execution_results_dir: Directory containing execution results
        model_name: Optional model name filter
        allowed_frameworks: Optional normalized framework names to include

    Returns:
        Dictionary with comprehensive analysis results
    """
    try:
        analysis_results: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "execution_results_dir": str(execution_results_dir),
            "framework_results": {},
            "cross_framework_comparison": {},
        }

        # Find all result JSON files
        result_files = list(execution_results_dir.rglob("*_results.json"))

        if not result_files:
            logger.warning(
                f"No execution result files found in {execution_results_dir}"
            )
            return analysis_results

        # Group by framework
        framework_data: dict[Any, Any] = {}

        for result_file in result_files:
            try:
                with open(result_file) as f:
                    result_data = json.load(f)

                framework = result_data.get("framework", "unknown")
                # Normalize framework name to lowercase for consistent grouping
                # Simulation JSONs may use "PyMDP"/"JAX" while execution logs use "pymdp"/"jax"
                framework = framework.lower().replace(".", "_").replace(" ", "_")
                if allowed_frameworks and framework not in allowed_frameworks:
                    continue
                file_model_name = result_data.get("model_name", "unknown")

                # Filter by model name if specified — use normalized comparison
                # because GNN file stems (e.g. "simple_mdp") differ from
                # display names in JSONs (e.g. "Simple MDP Agent", "GNNModel")
                if model_name and file_model_name != model_name:
                    file_slug = (
                        file_model_name.lower().replace(" ", "_").replace("-", "_")
                    )
                    param_slug = model_name.lower().replace(" ", "_").replace("-", "_")
                    if param_slug not in file_slug and file_slug not in param_slug:
                        continue

                if framework not in framework_data:
                    framework_data[framework] = []

                framework_data[framework].append(result_data)

            except Exception as e:
                logger.warning(f"Failed to load result file {result_file}: {e}")

        # Analyze each framework's results
        for framework, results in framework_data.items():
            try:
                framework_analysis: dict[str, Any] = {
                    "framework": framework,
                    "result_count": len(results),
                    "analyses": [],
                }

                for result in results:
                    # Extract framework-specific data
                    try:
                        if framework == "pymdp":
                            extracted = extract_pymdp_data(result)
                        elif framework == "rxinfer":
                            extracted = extract_rxinfer_data(result)
                        elif framework == "activeinference_jl":
                            extracted = extract_activeinference_jl_data(result)
                        elif framework == "pytorch":
                            extracted = extract_pytorch_data(result)
                        elif framework == "numpyro":
                            extracted = extract_numpyro_data(result)
                        elif framework == "jax":
                            extracted = extract_jax_data(result)
                        elif framework == "bnlearn":
                            extracted = extract_bnlearn_data(result)
                        elif framework == "discopy":
                            extracted = extract_discopy_data(result)
                        else:
                            extracted = result.get("simulation_data", {}) or {}

                        # Non-PyMDP frameworks still have heterogeneous outputs.
                        # PyMDP is strict and handled by extract_pymdp_data().
                        if (
                            framework not in {"pymdp", "bnlearn"}
                            and isinstance(extracted, dict)
                            and not extracted.get("beliefs")
                            and not extracted.get("observations")
                        ):
                            implementation_dir = result.get("implementation_directory")
                            if implementation_dir:
                                try:
                                    impl_path = Path(implementation_dir)
                                    sim_data_dir = impl_path / "simulation_data"
                                    if sim_data_dir.exists():
                                        results_files = list(
                                            sim_data_dir.glob("*.json")
                                        )
                                        for results_file in results_files:
                                            try:
                                                with open(results_file) as f:
                                                    file_data = json.load(f)
                                                    if isinstance(file_data, dict):
                                                        if (
                                                            "beliefs" in file_data
                                                            and not extracted.get(
                                                                "beliefs"
                                                            )
                                                        ):
                                                            extracted["beliefs"] = (
                                                                file_data["beliefs"]
                                                            )
                                                        if (
                                                            "actions" in file_data
                                                            and not extracted.get(
                                                                "actions"
                                                            )
                                                        ):
                                                            extracted["actions"] = (
                                                                file_data["actions"]
                                                            )
                                                        if (
                                                            "observations" in file_data
                                                            and not extracted.get(
                                                                "observations"
                                                            )
                                                        ):
                                                            extracted[
                                                                "observations"
                                                            ] = file_data[
                                                                "observations"
                                                            ]
                                            except Exception as e:
                                                logger.debug(
                                                    f"Failed to parse JSON file: {e}"
                                                )
                                except Exception as e:
                                    logger.debug(
                                        f"Error reading files for {framework}: {e}"
                                    )

                        # Run generic analyses
                        model_name_for_analysis = result.get("model_name", "unknown")

                        if isinstance(extracted, dict):
                            if framework != "pymdp" and not extracted.get(
                                "free_energy"
                            ):
                                efe = None
                                if result.get("efe_history"):
                                    efe = result["efe_history"]
                                elif isinstance(result.get("metrics"), dict) and result[
                                    "metrics"
                                ].get("expected_free_energy"):
                                    efe = result["metrics"]["expected_free_energy"]
                                elif isinstance(
                                    result.get("simulation_trace"), dict
                                ) and result["simulation_trace"].get("efe_history"):
                                    efe = result["simulation_trace"]["efe_history"]
                                if efe:
                                    extracted["free_energy"] = efe
                                    logger.debug(
                                        f"Recovery: extracted EFE for {framework} from result data ({len(efe)} entries)"
                                    )

                            if framework != "pymdp" and not extracted.get("beliefs"):
                                beliefs = result.get("beliefs") or result.get(
                                    "simulation_trace", {}
                                ).get("beliefs")
                                if beliefs:
                                    extracted["beliefs"] = beliefs

                            if framework != "pymdp" and not extracted.get("actions"):
                                actions = result.get("actions") or result.get(
                                    "simulation_trace", {}
                                ).get("actions")
                                if actions:
                                    extracted["actions"] = actions
                            if framework != "pymdp" and not extracted.get(
                                "observations"
                            ):
                                obs = result.get("observations") or result.get(
                                    "simulation_trace", {}
                                ).get("observations")
                                if obs:
                                    extracted["observations"] = obs

                            if extracted.get("free_energy"):
                                fe_analysis = analyze_free_energy(
                                    extracted["free_energy"],
                                    framework,
                                    model_name_for_analysis,
                                )
                                if extracted.get("schema_version"):
                                    fe_analysis["schema_version"] = extracted[
                                        "schema_version"
                                    ]
                                framework_analysis["analyses"].append(fe_analysis)

                            if extracted.get("traces"):
                                trace_analysis = analyze_simulation_traces(
                                    extracted["traces"],
                                    framework,
                                    model_name_for_analysis,
                                )
                                framework_analysis["analyses"].append(trace_analysis)

                            if extracted.get("policy"):
                                policy_analysis = analyze_policy_convergence(
                                    extracted["policy"],
                                    framework,
                                    model_name_for_analysis,
                                )
                                framework_analysis["analyses"].append(policy_analysis)
                    except Exception as e:
                        logger.warning(f"Error analyzing result for {framework}: {e}")
                        framework_analysis["analyses"].append({"error": str(e)})

                # Validate serializability
                def safe_json_default(obj: Any) -> Any:
                    """Provide safe json default behavior."""
                    if isinstance(obj, Path):
                        return str(obj)
                    if isinstance(obj, (np.integer, int)):
                        return int(obj)
                    if isinstance(obj, (np.floating, float)):
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, (set, frozenset)):
                        return list(obj)
                    if hasattr(obj, "__dict__"):
                        return str(obj)
                    return str(obj)

                try:
                    json.dumps(framework_analysis, default=safe_json_default)
                    analysis_results["framework_results"][framework] = (
                        framework_analysis
                    )
                except Exception as e:
                    logger.error(
                        f"Circular reference or serialization error in {framework} analysis: {e}"
                    )
                    analysis_results["framework_results"][framework] = {
                        "framework": framework,
                        "error": f"Serialization failed: {e}",
                    }

            except Exception as e:
                logger.error(f"Failed to analyze framework {framework}: {e}")
                analysis_results["framework_results"][framework] = {"error": str(e)}

        # Cross-framework comparison
        if len(framework_data) > 1:
            comparison_input: dict[Any, Any] = {}
            for framework, results in framework_data.items():
                if results:
                    comparison_input[framework] = results[0]

            comparison = compare_framework_results(
                comparison_input, model_name or "unknown"
            )
            analysis_results["cross_framework_comparison"] = comparison

        # Trigger visualizations and animations for the model results
        if model_name:
            results_dir = execution_results_dir.parent / "analysis_results"
            results_dir.mkdir(parents=True, exist_ok=True)

            for framework, data in analysis_results["framework_results"].items():
                for _, analysis in enumerate(data.get("analyses", [])):
                    if "beliefs" in analysis or (
                        isinstance(analysis, dict)
                        and "simulation_data" in analysis
                        and "beliefs" in analysis["simulation_data"]
                    ):
                        beliefs = analysis.get("beliefs") or analysis[
                            "simulation_data"
                        ].get("beliefs")
                        if beliefs:
                            plot_file = (
                                results_dir / f"{model_name}_{framework}_beliefs.png"
                            )
                            plot_belief_evolution(
                                beliefs,
                                plot_file,
                                title=f"Beliefs - {model_name} ({framework})",
                            )

                            anim_file = (
                                results_dir / f"{model_name}_{framework}_beliefs.gif"
                            )
                            try:
                                animate_belief_evolution(
                                    beliefs,
                                    anim_file,
                                    title=f"Evolution - {model_name} ({framework})",
                                )
                            except Exception as e:
                                logger.warning(f"Animation failed for {framework}: {e}")

                            analysis["plots"] = analysis.get("plots", [])
                            analysis["plots"].extend([str(plot_file), str(anim_file)])

        return analysis_results

    except Exception as e:
        logger.error(f"Error analyzing execution results: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return {"timestamp": datetime.now().isoformat(), "error": str(e)}
