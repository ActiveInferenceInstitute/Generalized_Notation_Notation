#!/usr/bin/env python3
"""
Report Analyzer Module

This module provides data collection and analysis functionality for pipeline reporting.
It analyzes pipeline outputs and extracts relevant metrics and statistics.
"""

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


STEP_OUTPUT_DIRS: tuple[str, ...] = (
    "0_template_output",
    "1_setup_output",
    "2_tests_output",
    "3_gnn_output",
    "4_model_registry_output",
    "5_type_checker_output",
    "6_validation_output",
    "7_export_output",
    "8_visualization_output",
    "9_advanced_viz_output",
    "10_ontology_output",
    "11_render_output",
    "12_execute_output",
    "13_llm_output",
    "14_ml_integration_output",
    "15_audio_output",
    "16_analysis_output",
    "17_integration_output",
    "18_security_output",
    "19_research_output",
    "20_website_output",
    "21_mcp_output",
    "22_gui_output",
    "23_report_output",
    "24_intelligent_analysis_output",
)
"""Canonical per-step output directories, in pipeline order.

Single source of truth for the step catalog; consumers should import this
tuple instead of redefining their own step lists.
"""

STEP_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "1_setup_output": ("0_template_output",),
    "2_tests_output": ("1_setup_output",),
    "3_gnn_output": ("1_setup_output",),
    "4_model_registry_output": ("3_gnn_output",),
    "5_type_checker_output": ("3_gnn_output",),
    "6_validation_output": ("3_gnn_output",),
    "7_export_output": ("5_type_checker_output",),
    "8_visualization_output": ("7_export_output",),
    "9_advanced_viz_output": ("8_visualization_output",),
    "10_ontology_output": ("7_export_output",),
    "11_render_output": ("7_export_output",),
    "12_execute_output": ("11_render_output",),
    "13_llm_output": ("3_gnn_output",),
    "14_ml_integration_output": ("3_gnn_output",),
    "15_audio_output": ("3_gnn_output",),
    "16_analysis_output": ("12_execute_output",),
    "17_integration_output": ("3_gnn_output",),
    "18_security_output": ("3_gnn_output",),
    "19_research_output": ("3_gnn_output",),
    "20_website_output": ("8_visualization_output", "10_ontology_output"),
    "21_mcp_output": ("3_gnn_output",),
    "22_gui_output": ("3_gnn_output",),
    "23_report_output": (
        "8_visualization_output",
        "10_ontology_output",
        "15_audio_output",
    ),
    "24_intelligent_analysis_output": ("23_report_output",),
}
"""Prerequisite step output directories per step (see analyze_step_dependencies)."""

KEY_FILE_PATTERNS: dict[str, tuple[str, ...]] = {
    "0_template_output": ("template_processing_summary.json",),
    "1_setup_output": ("setup_success.json", "directory_structure.json"),
    "2_tests_output": ("pytest_report.xml", "test_summary.json", "*.txt"),
    "3_gnn_output": ("gnn_processing_results.json", "*_parsed.json"),
    "4_model_registry_output": ("model_registry.json",),
    "5_type_checker_output": ("type_check_report.md", "type_check_summary.json"),
    "6_validation_output": ("validation_results.json",),
    "7_export_output": ("*.json", "*.xml", "*.graphml", "*.gexf"),
    "8_visualization_output": ("*.png", "*.svg", "*.html"),
    "9_advanced_viz_output": (
        "*.png",
        "*.svg",
        "*.html",
        "advanced_viz_summary.json",
    ),
    "10_ontology_output": ("ontology_results.json",),
    "11_render_output": ("*.py", "*.jl", "render_processing_summary.json"),
    "12_execute_output": (
        "summaries/execution_summary.json",
        "summaries/execution_report.md",
        "*.png",
    ),
    "13_llm_output": ("llm_results.json", "llm_summary.md"),
    "14_ml_integration_output": ("ml_integration_results.json",),
    "15_audio_output": ("*.wav", "*.mp3", "audio_processing_summary.json"),
    "16_analysis_output": ("analysis_results.json", "analysis_summary.md"),
    "17_integration_output": ("integration_processing_summary.json",),
    "18_security_output": ("security_results.json", "security_summary.md"),
    "19_research_output": (
        "research_results.json",
        "research_processing_summary.json",
        "research_report.md",
    ),
    "20_website_output": ("*.html", "*.css", "*.js"),
    "21_mcp_output": ("mcp_processing_summary.json", "registered_tools.json"),
    "22_gui_output": ("gui_processing_summary.json", "navigation.html"),
    "23_report_output": (
        "comprehensive_analysis_report.html",
        "report_summary.json",
    ),
    "24_intelligent_analysis_output": (
        "intelligent_analysis_report.md",
        "analysis_data.json",
        "intelligent_analysis_summary.json",
    ),
}
"""Key artifact name/glob patterns per step output directory (see is_key_file)."""


def _summary_timestamp(summary: Dict[str, Any]) -> str:
    """Use a timestamp from the evidence bundle, never the wall clock."""
    for key in ("end_time", "start_time", "timestamp"):
        value = summary.get(key)
        if isinstance(value, str) and value:
            return value
    return "unavailable"


def _step_output_name(script_name: str) -> str:
    """Map a canonical numbered script name to its output directory."""
    aliases = {"9_advanced_visualization": "9_advanced_viz_output"}
    stem = Path(script_name).stem
    return aliases.get(stem, f"{stem}_output")


def collect_pipeline_data(
    pipeline_output_dir: Path, logger: logging.Logger
) -> Dict[str, Any]:
    """
    Collect data from all pipeline outputs for analysis.

    Args:
        pipeline_output_dir: Directory containing pipeline outputs
        logger: Logger for this operation

    Returns:
        Dictionary containing collected pipeline data
    """
    pipeline_data: dict[str, Any] = {
        "report_generation_time": "unavailable",
        "report_timestamp_source": "unavailable",
        "pipeline_output_directory": str(pipeline_output_dir),
        "steps": {},
        "summary": {
            "total_files_processed": 0,
            "total_errors": 0,
            "total_warnings": 0,
            "processing_time": 0.0,
            "total_size_mb": 0.0,
            "success_rate": 0.0,
            "status_source": "artifacts_only",
        },
        "performance_metrics": {},
        "error_analysis": {},
        "file_type_analysis": {},
        "step_dependencies": {},
    }

    # Check for pipeline execution summary (main.py writes to 00_pipeline_summary/ subdir)
    pipeline_summary_candidates: list[Any] = [
        pipeline_output_dir / "00_pipeline_summary" / "pipeline_execution_summary.json",
        pipeline_output_dir / "pipeline_execution_summary.json",
    ]
    from gnn.pipeline.run_context import current_run_context

    context = current_run_context()
    if context is not None:
        pipeline_summary_candidates = [
            pipeline_output_dir / "00_pipeline_summary" / "current_summary.json"
        ]
    pipeline_summary_file = None
    for candidate in pipeline_summary_candidates:
        if candidate.exists():
            pipeline_summary_file = candidate
            break
    if context is not None and pipeline_summary_file is None:
        raise ValueError("Current-run report snapshot is unavailable")
    if pipeline_summary_file is not None:
        try:
            with open(pipeline_summary_file, "r", encoding="utf-8") as f:
                pipeline_data["pipeline_summary"] = json.load(f)
            if (
                context is not None
                and pipeline_data["pipeline_summary"].get("run_id") != context.run_id
            ):
                raise ValueError("Report snapshot belongs to a different run")

            # Extract performance metrics from summary
            summary = pipeline_data["pipeline_summary"]
            pipeline_data["report_generation_time"] = _summary_timestamp(summary)
            pipeline_data["report_timestamp_source"] = (
                "pipeline_execution_summary"
                if pipeline_data["report_generation_time"] != "unavailable"
                else "unavailable"
            )
            pipeline_data["overall_status"] = summary.get("overall_status", "UNKNOWN")
            if "performance_metrics" in summary:
                pipeline_data["performance_metrics"] = summary["performance_metrics"]
            elif "performance_summary" in summary:
                pipeline_data["performance_metrics"] = summary["performance_summary"]

            # Extract error analysis from summary
            if "errors" in summary:
                pipeline_data["error_analysis"] = analyze_errors(
                    summary["errors"], logger
                )

        except Exception as e:
            if context is not None:
                raise ValueError(f"Current-run report snapshot is invalid: {e}") from e
            logger.warning(f"Failed to read pipeline summary: {e}")

    # Collect data from each step directory
    step_directories = list(STEP_OUTPUT_DIRS)

    for step_dir in step_directories:
        step_path = pipeline_output_dir / step_dir
        if step_path.exists():
            step_data = analyze_step_directory(step_path, step_dir, logger)
            pipeline_data["steps"][step_dir] = step_data

            # Update summary statistics
            pipeline_data["summary"]["total_files_processed"] += step_data.get(
                "file_count", 0
            )
            pipeline_data["summary"]["total_size_mb"] += step_data.get(
                "total_size_mb", 0
            )
        else:
            pipeline_data["steps"][step_dir] = {
                "directory": str(step_path),
                "exists": False,
                "file_count": 0,
                "total_size_mb": 0.0,
                "file_types": {},
                "last_modified": None,
                "status": "missing",
            }

    # Fix self-referencing: step 23 (report) scans its own output directory
    # before files are written, resulting in 0 files. Annotate accordingly.
    report_step = pipeline_data["steps"].get("23_report_output")
    if (
        report_step
        and report_step.get("exists")
        and report_step.get("file_count", 0) == 0
    ):
        # The report step will generate report files (html, md, json + summaries)
        report_step["note"] = (
            "Report step output (self-referencing — files generated during this step)"
        )
        report_step["status"] = "generating"
        logger.debug("Annotated 23_report_output as self-referencing step")

    # Prefer execution receipts over the presence of output directories. A
    # failed step can leave a populated directory, so artifact presence alone
    # cannot establish successful execution.
    canonical_steps = pipeline_data.get("pipeline_summary", {}).get("steps", [])
    canonical_statuses: list[str] = []
    if isinstance(canonical_steps, list) and canonical_steps:
        for canonical_step in canonical_steps:
            if not isinstance(canonical_step, dict):
                continue
            status = str(canonical_step.get("status", "UNKNOWN")).upper()
            canonical_statuses.append(status)
            step_name = _step_output_name(str(canonical_step.get("script_name", "")))
            if step_name in pipeline_data["steps"]:
                step_record = pipeline_data["steps"][step_name]
                step_record["artifact_status"] = step_record.get("status")
                step_record["status"] = status.lower()
                step_record["execution_status"] = status
                step_record["status_source"] = "pipeline_execution_summary"

    total_steps = len(canonical_statuses) or len(step_directories)
    if canonical_statuses:
        successful_steps = sum(
            status.startswith("SUCCESS") for status in canonical_statuses
        )
        failed_steps = sum(status == "FAILED" for status in canonical_statuses)
        warning_steps = sum("WARNING" in status for status in canonical_statuses)
        unknown_steps = sum(
            not status.startswith("SUCCESS")
            and status != "FAILED"
            and "WARNING" not in status
            for status in canonical_statuses
        )
        pipeline_data["summary"].update(
            {
                "status_source": "pipeline_execution_summary",
                "successful_steps": successful_steps,
                "failed_steps": failed_steps,
                "warning_steps": warning_steps,
                "unknown_steps": unknown_steps,
                "processing_time": pipeline_data["pipeline_summary"].get(
                    "total_duration_seconds", 0.0
                ),
                "total_errors": failed_steps,
                "total_warnings": warning_steps,
            }
        )
        pipeline_data["health_score_basis"] = "pipeline_execution_summary"
    else:
        successful_steps = len(
            [
                step
                for step in pipeline_data["steps"].values()
                if step.get("exists", False)
            ]
        )
        pipeline_data["summary"].update(
            {
                "successful_steps": successful_steps,
                "failed_steps": 0,
                "warning_steps": 0,
                "unknown_steps": total_steps - successful_steps,
            }
        )
        pipeline_data["health_score_basis"] = "artifact_coverage_only"

    pipeline_data["summary"]["success_rate"] = (
        (successful_steps / total_steps) * 100 if total_steps > 0 else 0
    )

    # Analyze file types across all steps
    pipeline_data["file_type_analysis"] = analyze_file_types_across_steps(
        pipeline_data["steps"], logger
    )

    # Analyze step dependencies
    pipeline_data["step_dependencies"] = analyze_step_dependencies(
        pipeline_data["steps"], logger
    )

    # Collect visualizations from all pipeline outputs
    pipeline_data["visualizations"] = collect_visualizations(
        pipeline_output_dir, logger
    )

    return pipeline_data


def analyze_step_directory(
    step_path: Path, step_name: str, logger: logging.Logger
) -> Dict[str, Any]:
    """
    Analyze a specific step directory and extract relevant data.

    Args:
        step_path: Path to the step directory
        step_name: Name of the step
        logger: Logger for this operation

    Returns:
        Dictionary containing step analysis data
    """
    step_data: dict[str, Any] = {
        "directory": str(step_path),
        "exists": True,
        "file_count": 0,
        "total_size_mb": 0.0,
        "file_types": {},
        "last_modified": None,
        "status": "success",
        "performance_metrics": {},
        "error_logs": [],
        "key_files": [],
    }

    try:
        # Count files and calculate sizes
        from gnn.pipeline.artifact_ownership import current_artifact_files

        for file_path in current_artifact_files(step_path):
            if file_path.is_file():
                step_data["file_count"] += 1
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                step_data["total_size_mb"] += file_size_mb

                # Track file types
                file_ext = file_path.suffix.lower()
                if file_ext not in step_data["file_types"]:
                    step_data["file_types"][file_ext] = {
                        "count": 0,
                        "total_size_mb": 0.0,
                    }
                step_data["file_types"][file_ext]["count"] += 1
                step_data["file_types"][file_ext]["total_size_mb"] += file_size_mb

                # Track last modification
                mtime = file_path.stat().st_mtime
                if (
                    step_data["last_modified"] is None
                    or mtime > step_data["last_modified"]
                ):
                    step_data["last_modified"] = mtime

                # Identify key files
                if is_key_file(file_path, step_name):
                    step_data["key_files"].append(
                        {
                            "name": file_path.name,
                            "size_mb": round(file_size_mb, 2),
                            "type": file_ext,
                        }
                    )

        # Convert timestamp to ISO format
        if step_data["last_modified"]:
            step_data["last_modified"] = datetime.datetime.fromtimestamp(
                step_data["last_modified"], tz=datetime.timezone.utc
            ).isoformat()

        # Round size to 2 decimal places
        step_data["total_size_mb"] = round(step_data["total_size_mb"], 2)

        # Analyze step-specific performance and errors
        step_data.update(analyze_step_specific_data(step_path, step_name, logger))

    except Exception as e:
        logger.warning(f"Failed to analyze step directory {step_name}: {e}")
        step_data["error"] = str(e)
        step_data["status"] = "error"

    return step_data


def analyze_step_specific_data(
    step_path: Path, step_name: str, logger: logging.Logger
) -> Dict[str, Any]:
    """
    Analyze step-specific data like performance metrics and error logs.

    Args:
        step_path: Path to the step directory
        step_name: Name of the step
        logger: Logger for this operation

    Returns:
        Dictionary with step-specific analysis data
    """
    step_data: dict[Any, Any] = {}

    try:
        # Look for performance metrics files
        from gnn.pipeline.artifact_ownership import current_artifact_files

        perf_files = current_artifact_files(
            step_path, "*performance*.json"
        ) + current_artifact_files(step_path, "*metrics*.json")
        for perf_file in perf_files:
            try:
                with open(perf_file, "r", encoding="utf-8") as f:
                    step_data["performance_metrics"] = json.load(f)
                break
            except Exception as e:
                logger.debug(f"Failed to read performance file {perf_file}: {e}")

        # Look for error logs
        log_files = current_artifact_files(step_path, "*.log") + current_artifact_files(
            step_path, "*error*.json"
        )
        error_logs: list[Any] = []
        for log_file in log_files:
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    if "error" in content.lower() or "warning" in content.lower():
                        error_logs.append(
                            {
                                "file": log_file.name,
                                "size_mb": round(
                                    log_file.stat().st_size / (1024 * 1024), 2
                                ),
                            }
                        )
            except Exception as e:
                logger.debug(f"Failed to read log file {log_file}: {e}")

        if error_logs:
            step_data["error_logs"] = error_logs

    except Exception as e:
        logger.debug(f"Failed to analyze step-specific data for {step_name}: {e}")

    return step_data


def analyze_file_types_across_steps(
    steps: Dict[str, Any], logger: logging.Logger
) -> Dict[str, Any]:
    """
    Analyze file types across all pipeline steps.

    Args:
        steps: Dictionary of step data
        logger: Logger for this operation

    Returns:
        Dictionary with file type analysis
    """
    file_type_analysis: dict[str, Any] = {
        "total_by_type": {},
        "largest_files_by_type": {},
        "step_distribution": {},
    }

    try:
        for step_name, step_data in steps.items():
            if not step_data.get("exists", False):
                continue

            for file_ext, file_info in step_data.get("file_types", {}).items():
                # Track total counts and sizes
                if file_ext not in file_type_analysis["total_by_type"]:
                    file_type_analysis["total_by_type"][file_ext] = {
                        "count": 0,
                        "total_size_mb": 0.0,
                    }

                file_type_analysis["total_by_type"][file_ext]["count"] += file_info[
                    "count"
                ]
                file_type_analysis["total_by_type"][file_ext]["total_size_mb"] += (
                    file_info["total_size_mb"]
                )

                # Track step distribution
                if file_ext not in file_type_analysis["step_distribution"]:
                    file_type_analysis["step_distribution"][file_ext] = {}

                file_type_analysis["step_distribution"][file_ext][step_name] = {
                    "count": file_info["count"],
                    "size_mb": file_info["total_size_mb"],
                }

        # Round sizes
        for file_ext in file_type_analysis["total_by_type"]:
            file_type_analysis["total_by_type"][file_ext]["total_size_mb"] = round(
                file_type_analysis["total_by_type"][file_ext]["total_size_mb"], 2
            )

    except Exception as e:
        logger.warning(f"Failed to analyze file types: {e}")

    return file_type_analysis


def analyze_step_dependencies(
    steps: Dict[str, Any], logger: logging.Logger
) -> Dict[str, Any]:
    """
    Analyze dependencies between pipeline steps.

    Args:
        steps: Dictionary of step data
        logger: Logger for this operation

    Returns:
        Dictionary with dependency analysis
    """
    dependencies: dict[str, Any] = {
        "step_order": list(STEP_OUTPUT_DIRS),
        "dependency_chain": {},
        "missing_prerequisites": [],
    }

    try:
        # Check for missing prerequisites
        for step_name, prereqs in STEP_DEPENDENCIES.items():
            if step_name in steps and steps[step_name].get("exists", False):
                missing: list[Any] = []
                for prereq in prereqs:
                    if prereq not in steps or not steps[prereq].get("exists", False):
                        missing.append(prereq)

                if missing:
                    dependencies["missing_prerequisites"].append(
                        {"step": step_name, "missing": missing}
                    )

                dependencies["dependency_chain"][step_name] = {
                    "prerequisites": list(prereqs),
                    "missing_prerequisites": missing,
                    "status": "complete" if not missing else "incomplete",
                }

    except Exception as e:
        logger.warning(f"Failed to analyze step dependencies: {e}")

    return dependencies


def analyze_errors(
    errors: List[Dict[str, Any]], logger: logging.Logger
) -> Dict[str, Any]:
    """
    Analyze error patterns from pipeline execution.

    Args:
        errors: List of error dictionaries
        logger: Logger for this operation

    Returns:
        Dictionary with error analysis
    """
    error_analysis: dict[str, Any] = {
        "total_errors": len(errors),
        "error_types": {},
        "step_error_distribution": {},
        "critical_errors": [],
        "warnings": [],
    }

    try:
        for error in errors:
            error_type = error.get("type", "unknown")
            step_name = error.get("step", "unknown")

            # Count error types
            if error_type not in error_analysis["error_types"]:
                error_analysis["error_types"][error_type] = 0
            error_analysis["error_types"][error_type] += 1

            # Count errors by step
            if step_name not in error_analysis["step_error_distribution"]:
                error_analysis["step_error_distribution"][step_name] = 0
            error_analysis["step_error_distribution"][step_name] += 1

            # Categorize by severity
            if error.get("severity", "error") == "critical":
                error_analysis["critical_errors"].append(error)
            elif error.get("severity", "error") == "warning":
                error_analysis["warnings"].append(error)

    except Exception as e:
        logger.warning(f"Failed to analyze errors: {e}")

    return error_analysis


def is_key_file(file_path: Path, step_name: str) -> bool:
    """
    Determine if a file is a key file for a specific step.

    Args:
        file_path: Path to the file
        step_name: Name of the step

    Returns:
        True if the file is a key file for the step
    """
    if step_name not in KEY_FILE_PATTERNS:
        return False

    file_name = file_path.name
    for pattern in KEY_FILE_PATTERNS[step_name]:
        if pattern.startswith("*."):
            if file_name.endswith(pattern[1:]):
                return True
        elif file_name == pattern:
            return True

    return False


def collect_visualizations(
    pipeline_output_dir: Path, logger: logging.Logger
) -> Dict[str, Any]:
    """
    Collect all visualizations from pipeline output directories.

    Args:
        pipeline_output_dir: Directory containing pipeline outputs
        logger: Logger for this operation

    Returns:
        Dictionary with visualization catalog organized by step and type
    """
    visualizations: dict[str, Any] = {
        "total_count": 0,
        "by_step": {},
        "by_type": {},
        "all_visualizations": [],
    }

    # Visualization output directories to scan
    viz_directories: list[Any] = [
        ("8_visualization_output", ["*.png", "*.svg", "*.jpg", "*.jpeg", "*.html"]),
        ("9_advanced_viz_output", ["*.png", "*.svg", "*.jpg", "*.jpeg", "*.html"]),
        ("11_render_output", ["*.png", "*.svg", "*.jpg", "*.jpeg", "*.html"]),
        ("12_execute_output", ["*.png", "*.svg", "*.jpg", "*.jpeg", "*.html"]),
        ("20_website_output", ["*.html"]),
    ]

    try:
        for step_dir, patterns in viz_directories:
            step_path = pipeline_output_dir / step_dir
            if not step_path.exists():
                continue

            step_viz: dict[str, Any] = {"step": step_dir, "count": 0, "files": []}

            # Scan for visualization files
            for pattern in patterns:
                from gnn.pipeline.artifact_ownership import current_artifact_files

                for viz_file in current_artifact_files(step_path, pattern):
                    if viz_file.is_file():
                        try:
                            file_size = viz_file.stat().st_size
                            file_size_mb = file_size / (1024 * 1024)

                            # Determine file type
                            file_ext = viz_file.suffix.lower()
                            file_type = (
                                "image"
                                if file_ext in [".png", ".svg", ".jpg", ".jpeg", ".gif"]
                                else "html"
                            )

                            # Get relative path from pipeline output directory
                            rel_path = str(viz_file.relative_to(pipeline_output_dir))

                            viz_info: dict[str, Any] = {
                                "name": viz_file.name,
                                "path": str(viz_file),
                                "relative_path": rel_path,
                                "type": file_type,
                                "extension": file_ext,
                                "size_bytes": file_size,
                                "size_mb": round(file_size_mb, 2),
                                "step": step_dir,
                            }

                            step_viz["files"].append(viz_info)
                            step_viz["count"] += 1
                            visualizations["total_count"] += 1

                            # Track by type
                            if file_type not in visualizations["by_type"]:
                                visualizations["by_type"][file_type] = []
                            visualizations["by_type"][file_type].append(viz_info)

                            # Add to all visualizations
                            visualizations["all_visualizations"].append(viz_info)

                        except Exception as e:
                            logger.debug(
                                f"Error processing visualization file {viz_file}: {e}"
                            )

            if step_viz["count"] > 0:
                visualizations["by_step"][step_dir] = step_viz
                logger.info(f"Found {step_viz['count']} visualizations in {step_dir}")

        logger.info(f"Total visualizations discovered: {visualizations['total_count']}")

    except Exception as e:
        logger.warning(f"Failed to collect visualizations: {e}")

    return visualizations


def health_score_components(pipeline_data: Dict[str, Any]) -> Dict[str, float]:
    """Decompose the pipeline health score into its weighted components.

    The receipt-backed basis (``summary.status_source ==
    "pipeline_execution_summary"``) weighs four 0-100 subscores:
    ``step_completion`` (0.4), ``file_processing`` (0.3), ``error_margin``
    (0.2), and ``performance`` (0.1). The artifact-coverage basis degrades to
    the raw success rate and reports the neutral subscores.

    Args:
        pipeline_data: Pipeline analysis data

    Returns:
        Mapping of component name to 0-100 subscore, plus the resulting
        ``weighted_score``. See :func:`get_pipeline_health_score` for the
        scalar form.
    """
    summary = pipeline_data.get("summary", {})
    success_rate = float(summary.get("success_rate", 0))
    if summary.get("status_source") != "pipeline_execution_summary":
        return {
            "step_completion": success_rate,
            "file_processing": success_rate,
            "error_margin": 100.0,
            "performance": 50.0,
            "weighted_score": round(success_rate, 1),
        }

    # Error margin (20% weight): penalize 5 points per error, 10 per critical.
    error_analysis = pipeline_data.get("error_analysis", {})
    total_errors = error_analysis.get("total_errors", 0)
    critical_errors = len(error_analysis.get("critical_errors", []))
    error_penalty = min(total_errors * 5 + critical_errors * 10, 100)
    error_margin = 100.0 - error_penalty

    # Performance (10% weight): full credit under 10 minutes, decaying after.
    performance_metrics = pipeline_data.get("performance_metrics", {})
    if "execution_time" in performance_metrics:
        exec_time = performance_metrics["execution_time"]
        if exec_time < 600:
            performance = 100.0
        else:
            performance = max(0.0, 100.0 - (exec_time - 600) / 60)
    else:
        performance = 50.0  # Default subscore when no performance data exists

    subscores = {
        "step_completion": success_rate,
        "file_processing": success_rate,
        "error_margin": error_margin,
        "performance": performance,
    }
    weights = {
        "step_completion": 0.4,
        "file_processing": 0.3,
        "error_margin": 0.2,
        "performance": 0.1,
    }
    subscores["weighted_score"] = round(
        sum(subscores[name] * weight for name, weight in weights.items()), 1
    )
    return subscores


def get_pipeline_health_score(pipeline_data: Dict[str, Any]) -> float:
    """
    Calculate a health score for the pipeline based on various metrics.

    Args:
        pipeline_data: Pipeline analysis data

    Returns:
        Health score between 0 and 100

    See Also:
        :func:`health_score_components` for the per-component breakdown.
    """
    try:
        return float(health_score_components(pipeline_data)["weighted_score"])
    except Exception:
        logger.warning("health score computation failed", exc_info=True)
        return 0.0
