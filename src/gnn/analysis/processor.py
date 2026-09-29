#!/usr/bin/env python3
"""
Analysis processor module for GNN analysis.
"""

import json
import logging
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import numpy as np

from gnn.processing.discovery import is_model_source_path
from gnn.utils.logging_utils import (
    log_step_error,
    log_step_start,
    log_step_success,
)

# Phase 6: analyzer submodule is in-tree with core-only dependencies.
# Unconditional import — any ImportError is a real bug, not a soft-dep miss.
from .analyzer import (
    calculate_complexity_metrics,
    generate_analysis_summary,
    generate_matrix_visualizations,
    perform_model_comparisons,
    perform_statistical_analysis,
    run_performance_benchmarks,
    visualize_simulation_results,
)
from .framework_common import (
    FRAMEWORK_DIR_NAMES,
    SCHEMA_GATED_FRAMEWORKS,
    framework_from_path,
    iter_current_schema_results,
    load_execution_summary,
    model_name_from_path,
    normalize_framework_name,
    resolve_execution_dir,
)

_FRAMEWORK_DIR_NAMES: frozenset[str] = FRAMEWORK_DIR_NAMES

# Internal alias — shared implementation lives in framework_common.
_normalize_framework_name = normalize_framework_name


def _coerce_bool_flag(value: Any, flag_name: str) -> bool:
    """Coerce bool-like CLI/config values without treating every string as truthy."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{flag_name} must be a boolean value, got {value!r}")


def _normalize_generate_animations(
    kwargs: Dict[str, Any], logger: logging.Logger
) -> bool:
    """Normalize Step 16 animation flags.

    ``generate_animations`` is the canonical contract. ``no_animations`` is
    accepted only as a compatibility inverse when the canonical key is absent.
    """
    has_canonical = kwargs.get("generate_animations") is not None
    has_legacy = kwargs.get("no_animations") is not None

    if has_canonical:
        generate_animations = _coerce_bool_flag(
            kwargs["generate_animations"], "generate_animations"
        )
        if has_legacy:
            legacy_generate = not _coerce_bool_flag(
                kwargs["no_animations"], "no_animations"
            )
            if legacy_generate != generate_animations:
                raise ValueError(
                    "Ambiguous animation flags: generate_animations conflicts with "
                    "compatibility no_animations; use generate_animations as source of truth"
                )
            logger.warning(
                "Both generate_animations and compatibility no_animations supplied; "
                "using generate_animations=%s",
                generate_animations,
            )
        logger.info("Step 16 GridWorld animations enabled: %s", generate_animations)
        return generate_animations

    if has_legacy:
        generate_animations = not _coerce_bool_flag(
            kwargs["no_animations"], "no_animations"
        )
        logger.warning(
            "Compatibility no_animations supplied; normalized to generate_animations=%s",
            generate_animations,
        )
        return generate_animations

    logger.info("Step 16 GridWorld animations enabled: True (default)")
    return True


def _scope_from_execution_summary(
    execution_summary: Dict[str, Any],
    target_model_names: Optional[set[str]] = None,
) -> Dict[str, Optional[set[str]]]:
    """Derive current-run framework/model scope from a Step 12 summary."""
    frameworks: set[str] = set()
    models: set[str] = set(target_model_names or set())

    requested = execution_summary.get("requested_frameworks")
    has_requested_frameworks = False
    if isinstance(requested, list):
        frameworks.update(_normalize_framework_name(item) for item in requested)
        has_requested_frameworks = bool(frameworks)

    details = execution_summary.get("execution_details")
    if not isinstance(details, list):
        details = execution_summary.get("execution_results", [])
    if not isinstance(details, list):
        details = []

    successful_detail_frameworks: set[str] = set()
    for detail in details:
        if not isinstance(detail, dict):
            continue
        framework = (
            _normalize_framework_name(detail["framework"])
            if detail.get("framework")
            else None
        )
        has_result_pointer = bool(
            detail.get("structured_result_file")
            or detail.get("simulation_data")
            or detail.get("output_file")
        )
        if (
            framework
            and detail.get("success", False)
            and not detail.get("skipped", False)
            and has_result_pointer
        ):
            successful_detail_frameworks.add(framework)
        if framework and not has_requested_frameworks:
            frameworks.add(framework)
        if detail.get("model_name"):
            models.add(str(detail["model_name"]))
        script_path = detail.get("script_path") or detail.get("script")
        if script_path:
            parts = Path(str(script_path)).parts
            for index, part in enumerate(parts):
                if part in _FRAMEWORK_DIR_NAMES and index >= 1:
                    models.add(parts[index - 1])
                    if not has_requested_frameworks:
                        frameworks.add(part)
                    break

    if successful_detail_frameworks:
        frameworks = successful_detail_frameworks

    return {
        "frameworks": frameworks or None,
        "models": models or None,
    }


def _filter_execution_summary(
    execution_summary: Dict[str, Any],
    allowed_frameworks: Optional[set[str]],
) -> Dict[str, Any]:
    """Return a copy of an execution summary limited to current-run frameworks."""
    if not allowed_frameworks:
        return execution_summary

    filtered = deepcopy(execution_summary)
    details = filtered.get("execution_details")
    if isinstance(details, list):
        filtered["execution_details"] = [
            detail
            for detail in details
            if isinstance(detail, dict)
            and _normalize_framework_name(detail.get("framework", ""))
            in allowed_frameworks
        ]

    framework_status = filtered.get("framework_status")
    if isinstance(framework_status, dict):
        filtered["framework_status"] = {
            framework: status
            for framework, status in framework_status.items()
            if _normalize_framework_name(framework) in allowed_frameworks
        }

    return filtered


def aggregate_simulation_results(results_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate results from multiple simulations.

    Args:
        results_list: List of simulation result dictionaries

    Returns:
        Aggregated results dictionary
    """
    aggregated: dict[str, Any] = {
        "count": len(results_list),
        "metrics": {},
        "frameworks_used": set(),
    }

    # Extract common metrics
    metrics_to_gather: list[str] = [
        "execution_time",
        "free_energy_final",
        "steps_completed",
    ]

    for res in results_list:
        if "framework" in res:
            aggregated["frameworks_used"].add(res["framework"])

        for metric in metrics_to_gather:
            if metric not in aggregated["metrics"]:
                aggregated["metrics"][metric] = []

            if metric in res:
                aggregated["metrics"][metric].append(res[metric])
            elif "metrics" in res and metric in res["metrics"]:
                aggregated["metrics"][metric].append(res["metrics"][metric])

    # Convert sets to lists
    aggregated["frameworks_used"] = list(aggregated["frameworks_used"])

    return aggregated


def generate_summary_statistics(aggregated_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate summary statistics from aggregated data.

    Args:
        aggregated_data: Aggregated data dictionary

    Returns:
        Dictionary with statistical summaries (mean, std, min, max)
    """
    stats: dict[Any, Any] = {}

    for metric, values in aggregated_data.get("metrics", {}).items():
        if not values:
            continue

        # Filter out None values
        valid_values = [
            v for v in values if v is not None and isinstance(v, (int, float))
        ]

        if not valid_values:
            continue

        stats[metric] = {
            "mean": float(np.mean(valid_values)),
            "std": float(np.std(valid_values)),
            "min": float(np.min(valid_values)),
            "max": float(np.max(valid_values)),
            "count": len(valid_values),
        }

    return stats


def process_analysis(
    target_dir: Path,
    output_dir: Path,
    verbose: bool = False,
    **kwargs: Any,
) -> Union[bool, int]:
    """
    Process GNN files with comprehensive analysis.

    Args:
        target_dir: Directory containing GNN files to process
        output_dir: Directory to save results
        verbose: Enable verbose output
        **kwargs: Additional arguments

    Returns:
        True if processing successful, False otherwise
    """
    logger = logging.getLogger("analysis")

    try:
        log_step_start(logger, "Processing analysis")
        logging.getLogger("matplotlib.category").setLevel(logging.WARNING)

        # Validate target_dir exists up-front (Phase 1.3 — Phase 0.3 utility).
        if not target_dir.exists():
            logger.warning(f"Target directory does not exist: {target_dir}")
            return 2

        results_dir = output_dir
        results_dir.mkdir(parents=True, exist_ok=True)

        results: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "processed_files": 0,
            "success": True,
            "errors": [],
            "statistical_analysis": [],
            "complexity_metrics": [],
            "performance_benchmarks": [],
            "model_comparisons": [],
            "visualization_files": [],
        }
        try:
            generate_animations = _normalize_generate_animations(kwargs, logger)
        except ValueError as e:
            log_step_error(logger, str(e))
            return False

        # Resolve execution results directory once at the top so all branches can reference it
        execution_dir = resolve_execution_dir(output_dir)

        # Find GNN files
        gnn_files = [
            path for path in target_dir.rglob("*.md") if is_model_source_path(path)
        ]
        if not gnn_files:
            logger.warning("No GNN files found for analysis")
            # Exit-code 2: no input is a warning, not an error. Per CLAUDE.md
            # "exit codes: 0=success, 1=error, 2=success with warnings/skipped" taxonomy, a missing
            # input dir is "nothing to do" — not a real failure.
            return 2
        else:
            results["processed_files"] = len(gnn_files)

            # Process each GNN file
            for gnn_file in gnn_files:
                try:
                    # Perform statistical analysis
                    stats_analysis = perform_statistical_analysis(gnn_file, verbose)
                    results["statistical_analysis"].append(stats_analysis)

                    # Calculate complexity metrics
                    complexity = calculate_complexity_metrics(gnn_file, verbose)
                    results["complexity_metrics"].append(complexity)

                    # Run performance benchmarks
                    benchmarks = run_performance_benchmarks(gnn_file, verbose)
                    results["performance_benchmarks"].append(benchmarks)

                    # Generate matrix visualizations (moved from Step 8)
                    matrix_viz = generate_matrix_visualizations(
                        {"matrices": stats_analysis.get("matrices", [])},
                        results_dir,
                        gnn_file.stem,
                    )
                    results["visualization_files"].extend(matrix_viz)

                except Exception as e:
                    error_info: dict[str, Any] = {
                        "file": str(gnn_file),
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                    results["errors"].append(error_info)
                    logger.error(f"Error processing {gnn_file}: {e}")

            # 2. Post-Simulation Analysis: Load execution results if available
            # (execution_dir resolved above, before the gnn_files check)
            # Log the exact path being searched for debugging
            logger.info(f"Looking for execution results in: {execution_dir}")
            target_model_names = {gnn_file.stem for gnn_file in gnn_files}
            execution_scope: Dict[str, Optional[set[str]]] = {
                "frameworks": None,
                "models": target_model_names or None,
            }

            if execution_dir.exists():
                logger.info(
                    "Found execution results directory, performing post-simulation analysis"
                )

                # Load the Step 12 execution summary (summaries/ first, then root).
                _, execution_results_data = load_execution_summary(execution_dir)
                if execution_results_data is not None:
                    try:
                        execution_scope = _scope_from_execution_summary(
                            execution_results_data,
                            target_model_names=target_model_names,
                        )

                        # Generate empirical visualizations from execution summary
                        scoped_execution_data = _filter_execution_summary(
                            execution_results_data,
                            execution_scope["frameworks"],
                        )
                        empirical_viz = visualize_simulation_results(
                            scoped_execution_data, results_dir
                        )
                        results["visualization_files"].extend(empirical_viz)
                        logger.info(
                            f"Generated {len(empirical_viz)} empirical visualizations from execution summary"
                        )
                    except (json.JSONDecodeError, OSError, ValueError, KeyError) as e:
                        logger.warning(f"Failed to load execution summary: {e}")
                else:
                    logger.warning("Execution summary not found for execution results")

                # Perform detailed post-simulation analysis for each model
                try:
                    from .post_simulation import analyze_execution_results

                    allowed_frameworks = execution_scope["frameworks"]
                    for gnn_file in gnn_files:
                        model_name = gnn_file.stem
                        logger.info(f"Analyzing execution results for {model_name}")

                        # Find execution results for this model
                        model_execution_dir = execution_dir / model_name
                        if model_execution_dir.exists():
                            post_sim_analysis = analyze_execution_results(
                                model_execution_dir,
                                model_name=model_name,
                                allowed_frameworks=allowed_frameworks,
                            )

                            analysis_cross_fw_dir = results_dir / "cross_framework"
                            analysis_cross_fw_dir.mkdir(parents=True, exist_ok=True)
                            analysis_file = (
                                analysis_cross_fw_dir
                                / f"{model_name}_post_simulation_analysis.json"
                            )
                            with open(analysis_file, "w") as f:
                                json.dump(
                                    post_sim_analysis,
                                    f,
                                    indent=2,
                                    default=convert_numpy_types,
                                )

                            results["post_simulation_analysis"] = results.get(
                                "post_simulation_analysis", []
                            )
                            results["post_simulation_analysis"].append(
                                {
                                    "model_name": model_name,
                                    "analysis_file": str(analysis_file),
                                    "framework_count": len(
                                        post_sim_analysis.get("framework_results", {})
                                    ),
                                    "has_comparison": "cross_framework_comparison"
                                    in post_sim_analysis,
                                }
                            )

                            logger.info(
                                f"Post-simulation analysis completed for {model_name}"
                            )
                        else:
                            logger.debug(
                                f"No model-specific execution directory for {model_name}"
                            )

                    # Aggregate results and generate statistics
                    if results.get("post_simulation_analysis"):
                        aggregated = aggregate_simulation_results(
                            results["post_simulation_analysis"]
                        )
                        stats = generate_summary_statistics(aggregated)
                        results["overall_statistics"] = stats
                        logger.info("Generated overall summary statistics")
                except Exception as e:
                    logger.error(f"Failed to process post-simulation analysis: {e}")
                    import traceback

                    logger.debug(traceback.format_exc())

                # 2.5-2.11: Generate per-framework visualizations from execution logs.
                # Only frameworks with a corresponding ``src/analysis/<framework>/analyzer.py``
                # are listed here; bnlearn is rendered and executed but has no analyzer.
                _FRAMEWORK_ANALYZERS: list[Any] = [
                    ("pymdp", "pymdp", "PyMDP"),
                    ("activeinference_jl", "activeinference_jl", "ActiveInference.jl"),
                    ("discopy", "discopy", "DisCoPy"),
                    ("jax", "jax", "JAX"),
                    ("rxinfer", "rxinfer", "RxInfer"),
                    ("pytorch", "pytorch", "PyTorch"),
                    ("numpyro", "numpyro", "NumPyro"),
                    ("cpomdp", "cpomdp", "cpomdp"),
                ]
                import importlib

                allowed_frameworks = execution_scope["frameworks"]
                for module_key, dir_name, display_name in _FRAMEWORK_ANALYZERS:
                    if allowed_frameworks and module_key not in allowed_frameworks:
                        logger.debug(
                            "Skipping %s analysis outside current execution scope",
                            display_name,
                        )
                        continue
                    try:
                        mod = importlib.import_module(
                            f".{module_key}.analyzer", package="analysis"
                        )
                        fw_output_dir = output_dir / dir_name
                        fw_output_dir.mkdir(parents=True, exist_ok=True)
                        logger.info(f"Generating {display_name} visualizations...")
                        fw_viz = mod.generate_analysis_from_logs(
                            execution_dir, fw_output_dir, verbose
                        )
                        if fw_viz:
                            results["visualization_files"].extend(fw_viz)
                            logger.info(
                                f"Generated {len(fw_viz)} {display_name} visualization files"
                            )
                    except ImportError as e:
                        logger.debug(f"{module_key} analyzer not available: {e}")
                    except Exception as e:  # Optional analysis — broad catch intentional for third-party failures
                        logger.warning(f"{display_name} analysis failed: {e}")
            else:
                logger.warning(
                    f"Execution directory not found at {execution_dir}. Skipping post-simulation analysis."
                )

            # Perform cross-model comparisons if multiple files
            if len(gnn_files) > 1:
                comparisons = perform_model_comparisons(
                    results["statistical_analysis"], verbose
                )
                results["model_comparisons"].append(comparisons)

        # Perform cross-framework analysis if execution results exist
        if execution_dir.exists():
            logger.info("Performing cross-framework analysis...")
            allowed_frameworks = execution_scope["frameworks"]
            allowed_model_names = execution_scope["models"]
            try:
                from .analyzer import (
                    analyze_framework_outputs,
                    generate_framework_comparison_report,
                    visualize_cross_framework_metrics,
                )

                framework_comparison = analyze_framework_outputs(
                    execution_dir,
                    logger,
                    allowed_frameworks=allowed_frameworks,
                )
                results["framework_comparison"] = framework_comparison

                # Generate comparison report
                report_file = generate_framework_comparison_report(
                    framework_comparison, results_dir, logger
                )
                results["framework_comparison_report"] = report_file

                # Generate comparison visualizations
                comparison_viz = visualize_cross_framework_metrics(
                    framework_comparison, results_dir, logger
                )
                results["visualization_files"].extend(comparison_viz)

                logger.info(
                    f"Generated {len(comparison_viz)} cross-framework comparison visualizations"
                )
            except Exception as e:
                logger.warning(f"Cross-framework analysis failed: {e}")
                import traceback

                logger.debug(traceback.format_exc())

            # 3. Comprehensive visualization of all execution outputs
            logger.info(
                "Generating comprehensive visualizations for all execution outputs..."
            )
            try:
                from .post_simulation import (
                    generate_unified_framework_dashboard,
                    visualize_all_framework_outputs,
                )
                from .visualizations import _current_schema_visualization_data

                # Use cross_framework folder for cross-implementation analysis
                viz_output_dir = results_dir / "cross_framework"
                comprehensive_viz = visualize_all_framework_outputs(
                    execution_dir,
                    viz_output_dir,
                    logger,
                    allowed_frameworks=allowed_frameworks,
                    allowed_model_names=allowed_model_names,
                    generate_animations=generate_animations,
                )
                results["comprehensive_visualizations"] = comprehensive_viz
                results["visualization_files"].extend(comprehensive_viz)

                logger.info(
                    f"Generated {len(comprehensive_viz)} comprehensive visualization files"
                )

                # Generate unified framework dashboard for direct comparison
                logger.info("Generating unified framework dashboard...")
                try:
                    # Build framework_data structure for unified dashboard from
                    # current-schema simulation results (shared loader).
                    framework_data_for_dashboard: dict[str, Any] = {}
                    for sim_file, raw_payload in iter_current_schema_results(
                        execution_dir, pattern="*simulation_results.json"
                    ):
                        framework = framework_from_path(sim_file)
                        if not framework:
                            continue
                        if allowed_frameworks and framework not in allowed_frameworks:
                            continue
                        if allowed_model_names:
                            model_from_path = model_name_from_path(sim_file)
                            if (
                                model_from_path
                                and model_from_path not in allowed_model_names
                            ):
                                continue
                        sim_data = raw_payload
                        if framework in SCHEMA_GATED_FRAMEWORKS:
                            sim_data = _current_schema_visualization_data(raw_payload)
                        if framework not in framework_data_for_dashboard:
                            framework_data_for_dashboard[framework] = {
                                "framework": framework,
                                "simulation_data": sim_data,
                            }

                    if len(framework_data_for_dashboard) >= 2:
                        model_name = (
                            gnn_files[0].stem if gnn_files else "Active Inference Model"
                        )
                        dashboard_viz = generate_unified_framework_dashboard(
                            framework_data_for_dashboard,
                            viz_output_dir / "unified_dashboard",
                            model_name=model_name,
                        )
                        results["visualization_files"].extend(dashboard_viz)
                        logger.info(
                            f"Generated {len(dashboard_viz)} unified dashboard visualizations"
                        )
                    else:
                        logger.info(
                            "Less than 2 frameworks with data - skipping unified dashboard"
                        )
                except Exception as e:
                    logger.warning(f"Unified dashboard generation failed: {e}")
                    import traceback

                    logger.debug(traceback.format_exc())
            except Exception as e:
                logger.warning(f"Comprehensive visualization generation failed: {e}")
                import traceback

                logger.debug(traceback.format_exc())

            # 4. Generate unified cross-model comparison report
            try:
                from .generate_cross_model_report import generate_cross_model_report

                report_path = results_dir / "cross_model_comparison_report.md"
                report_file = generate_cross_model_report(
                    execution_dir,
                    results_dir,
                    report_path,
                    allowed_frameworks=allowed_frameworks,
                    allowed_model_names=allowed_model_names,
                )
                if report_file:
                    results["cross_model_report"] = report_file
                    logger.info(
                        f"Generated cross-model comparison report: {report_file}"
                    )
            except Exception as e:
                logger.warning(f"Cross-model report generation failed: {e}")
                import traceback

                logger.debug(traceback.format_exc())

            if generate_animations:
                try:
                    from .visualizations import write_gridworld_analysis_manifest

                    manifest_file = write_gridworld_analysis_manifest(
                        execution_dir,
                        results_dir,
                        allowed_frameworks=allowed_frameworks,
                        allowed_model_names=allowed_model_names,
                        logger_instance=logger,
                    )
                    if manifest_file:
                        results["gridworld_analysis_manifest"] = manifest_file
                except Exception as e:
                    logger.warning(
                        f"GridWorld analysis manifest generation failed: {e}"
                    )
                    import traceback

                    logger.debug(traceback.format_exc())

        # Save detailed results
        results_file = results_dir / "analysis_results.json"
        try:
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2, default=convert_numpy_types)
        except (OSError, IOError) as e:
            logger.error(f"Failed to write analysis results: {e}")

        # Generate summary report
        summary = generate_analysis_summary(results)
        summary_file = results_dir / "analysis_summary.md"
        try:
            with open(summary_file, "w") as f:
                f.write(summary)
        except (OSError, IOError) as e:
            logger.error(f"Failed to write analysis summary: {e}")

        if results["success"]:
            log_step_success(logger, "Analysis processing completed successfully")
        else:
            log_step_error(logger, "Analysis processing failed")

        return cast("bool | int", results["success"])

    except Exception as e:
        # Use supported signature for log_step_error
        log_step_error(logger, f"Analysis processing failed: {e}")
        return False


def convert_numpy_types(obj: Any) -> Any:
    """
    Convert numpy types and other non-JSON-serializable types to native Python types.

    Handles:
    - numpy integers, floats, arrays
    - sets and frozensets
    - objects with __dict__ (to prevent circular reference issues)
    - Path objects
    """
    if obj is None:
        return obj
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (set, frozenset)):
        return list(obj)
    elif isinstance(obj, Path):
        return str(obj)
    elif hasattr(obj, "__dict__") and not isinstance(obj, dict):
        # Convert objects to dict representation to avoid circular references
        return f"<{type(obj).__name__}>"
    elif isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    # Use string representation for any other unknown types to ensure serializability
    # and prevent "Circular reference detected" or "not JSON serializable" errors
    return str(obj)
