#!/usr/bin/env python3
"""
Report Generator Module

This module provides the main report generation functionality for the GNN pipeline.
It orchestrates the collection, analysis, and formatting of pipeline data into comprehensive reports.
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .analyzer import collect_pipeline_data, get_pipeline_health_score
from .formatters import generate_html_report, generate_markdown_report

DEFAULT_REPORT_FORMATS: list[str] = ["html", "markdown", "json"]
"""Formats :func:`generate_comprehensive_report` emits when none are given."""

REPORT_FILENAMES: dict[str, str] = {
    "html": "comprehensive_analysis_report.html",
    "markdown": "comprehensive_analysis_report.md",
    "json": "report_summary.json",
}
"""Output filename per report format (single source of truth for consumers)."""


def generate_comprehensive_report(
    pipeline_output_dir: Path,
    report_output_dir: Path,
    logger: logging.Logger,
    report_formats: Optional[List[str]] = None,
    include_performance: bool = True,
    include_errors: bool = True,
    include_dependencies: bool = True,
) -> bool:
    """
    Generate a comprehensive analysis report from all pipeline outputs.

    Args:
        pipeline_output_dir: Directory containing all pipeline outputs
        report_output_dir: Directory to save the generated report
        logger: Logger for this operation
        report_formats: List of report formats to generate (html, markdown, json)
        include_performance: Whether to include performance metrics
        include_errors: Whether to include error analysis
        include_dependencies: Whether to include dependency analysis

    Returns:
        True if report generation succeeded, False otherwise
    """
    try:
        logger.info("Starting comprehensive report generation from pipeline outputs")

        # Validate input directories
        if not pipeline_output_dir.exists():
            logger.error(
                f"Pipeline output directory does not exist: {pipeline_output_dir}"
            )
            return False
        if not pipeline_output_dir.is_dir():
            logger.error(
                f"Pipeline output path is not a directory: {pipeline_output_dir}"
            )
            return False

        # Create report output directory
        report_output_dir.mkdir(parents=True, exist_ok=True)

        # Set default report formats if not specified
        report_formats = list(
            DEFAULT_REPORT_FORMATS if report_formats is None else report_formats
        )

        # Collect data from all pipeline steps
        logger.info("Collecting pipeline data for analysis")
        pipeline_data = collect_pipeline_data(pipeline_output_dir, logger)

        # Calculate health score
        health_score = get_pipeline_health_score(pipeline_data)
        pipeline_data["health_score"] = health_score

        # Add generation metadata
        pipeline_data["report_metadata"] = {
            "generation_time": "unavailable",
            "evidence_as_of": pipeline_data["report_generation_time"],
            "timestamp_source": pipeline_data.get(
                "report_timestamp_source", "unavailable"
            ),
            "formats_generated": report_formats,
            "options": {
                "include_performance": include_performance,
                "include_errors": include_errors,
                "include_dependencies": include_dependencies,
            },
        }

        # Generate reports in requested formats
        generated_files: list[Any] = []

        writer_map: dict[
            str, Callable[[Dict[str, Any], Path, logging.Logger], bool]
        ] = {
            "html": generate_html_report_file,
            "markdown": generate_markdown_report_file,
            "json": generate_json_report_file,
        }
        for format_type in report_formats:
            try:
                writer = writer_map.get(format_type)
                if writer is None:
                    logger.warning(f"Unsupported report format: {format_type}")
                    continue
                if writer(pipeline_data, report_output_dir, logger):
                    generated_files.append(REPORT_FILENAMES[format_type])
            except Exception as e:
                logger.error(f"Failed to generate {format_type} report: {e}")

        # Generate summary report
        generate_summary_report(
            pipeline_data, report_output_dir, logger, generated_files
        )

        # Log success
        logger.info(f"Generated comprehensive report with {len(generated_files)} files")
        logger.info(f"Pipeline health score: {health_score}/100")
        logger.info(f"Report files: {', '.join(generated_files)}")

        return bool(generated_files)

    except Exception as e:
        logger.error(f"Failed to generate comprehensive report: {e}")
        return False


def generate_html_report_file(
    pipeline_data: Dict[str, Any], report_output_dir: Path, logger: logging.Logger
) -> bool:
    """
    Generate HTML report file.

    Args:
        pipeline_data: Pipeline analysis data
        report_output_dir: Output directory
        logger: Logger for this operation

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("Generating HTML report")

        # Generate HTML content
        html_content = generate_html_report(pipeline_data, logger)

        # Write HTML report
        report_file = report_output_dir / "comprehensive_analysis_report.html"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        file_size_mb = report_file.stat().st_size / (1024 * 1024)
        logger.info(f"HTML report generated: {report_file} ({file_size_mb:.2f} MB)")

        return True

    except Exception as e:
        logger.error(f"Failed to generate HTML report: {e}")
        return False


def generate_markdown_report_file(
    pipeline_data: Dict[str, Any], report_output_dir: Path, logger: logging.Logger
) -> bool:
    """
    Generate Markdown report file.

    Args:
        pipeline_data: Pipeline analysis data
        report_output_dir: Output directory
        logger: Logger for this operation

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("Generating Markdown report")

        # Generate Markdown content
        markdown_content = generate_markdown_report(pipeline_data, logger)

        # Write Markdown report
        report_file = report_output_dir / "comprehensive_analysis_report.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        file_size_mb = report_file.stat().st_size / (1024 * 1024)
        logger.info(f"Markdown report generated: {report_file} ({file_size_mb:.2f} MB)")

        return True

    except Exception as e:
        logger.error(f"Failed to generate Markdown report: {e}")
        return False


def generate_json_report_file(
    pipeline_data: Dict[str, Any], report_output_dir: Path, logger: logging.Logger
) -> bool:
    """
    Generate JSON report file.

    Args:
        pipeline_data: Pipeline analysis data
        report_output_dir: Output directory
        logger: Logger for this operation

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("Generating JSON report")

        # Write JSON report
        report_file = report_output_dir / "report_summary.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(pipeline_data, f, indent=2, default=str)

        file_size_mb = report_file.stat().st_size / (1024 * 1024)
        logger.info(f"JSON report generated: {report_file} ({file_size_mb:.2f} MB)")

        return True

    except Exception as e:
        logger.error(f"Failed to generate JSON report: {e}")
        return False


def generate_summary_report(
    pipeline_data: Dict[str, Any],
    report_output_dir: Path,
    logger: logging.Logger,
    generated_files: List[str],
) -> None:
    """
    Generate a summary report of the report generation process.

    Args:
        pipeline_data: Pipeline analysis data
        report_output_dir: Output directory
        logger: Logger for this operation
        generated_files: List of generated report files
    """
    try:
        summary: dict[str, Any] = {
            "report_generation_summary": {
                "generation_time": "unavailable",
                "evidence_as_of": pipeline_data.get(
                    "report_generation_time", "unavailable"
                ),
                "pipeline_directory": str(
                    pipeline_data.get("pipeline_output_directory", "")
                ),
                "report_directory": str(report_output_dir),
                "health_score": pipeline_data.get("health_score", 0),
                "generated_files": generated_files,
                "pipeline_summary": {
                    "total_steps": len(pipeline_data.get("steps", {})),
                    "successful_steps": pipeline_data.get("summary", {}).get(
                        "successful_steps", 0
                    ),
                    "total_files": pipeline_data.get("summary", {}).get(
                        "total_files_processed", 0
                    ),
                    "total_size_mb": pipeline_data.get("summary", {}).get(
                        "total_size_mb", 0
                    ),
                    "success_rate": pipeline_data.get("summary", {}).get(
                        "success_rate", 0
                    ),
                },
            }
        }

        summary_file = report_output_dir / "report_generation_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Report generation summary created: {summary_file}")

    except Exception as e:
        logger.error(f"Failed to generate summary report: {e}")


def generate_custom_report(
    pipeline_output_dir: Path,
    report_output_dir: Path,
    logger: logging.Logger,
    step_filter: Optional[List[str]] = None,
    exclude_steps: Optional[List[str]] = None,
    format_type: str = "html",
) -> bool:
    """
    Generate a custom report with specific step filtering.

    Args:
        pipeline_output_dir: Directory containing pipeline outputs
        report_output_dir: Directory to save generated reports
        logger: Logger for this operation
        step_filter: List of steps to include (optional)
        exclude_steps: List of steps to exclude (optional)
        format_type: Report format (html, markdown, json)

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("Generating custom report with step filtering")

        # Collect pipeline data
        pipeline_data = collect_pipeline_data(pipeline_output_dir, logger)

        # Filter steps if specified
        if step_filter or exclude_steps:
            filtered_steps: dict[Any, Any] = {}
            for step_name, step_data in pipeline_data.get("steps", {}).items():
                if step_filter and step_name not in step_filter:
                    continue
                if exclude_steps and step_name in exclude_steps:
                    continue
                filtered_steps[step_name] = step_data

            pipeline_data["steps"] = filtered_steps
            pipeline_data["filtering_applied"] = {
                "included_steps": step_filter,
                "excluded_steps": exclude_steps,
            }

        # Create output directory
        report_output_dir.mkdir(parents=True, exist_ok=True)

        # Generate report based on format
        if format_type == "html":
            return generate_html_report_file(pipeline_data, report_output_dir, logger)
        elif format_type == "markdown":
            return generate_markdown_report_file(
                pipeline_data, report_output_dir, logger
            )
        elif format_type == "json":
            return generate_json_report_file(pipeline_data, report_output_dir, logger)
        else:
            logger.error(f"Unsupported format type: {format_type}")
            return False

    except Exception as e:
        logger.error(f"Failed to generate custom report: {e}")
        return False


def validate_report_data(pipeline_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate pipeline data and return validation results.

    Args:
        pipeline_data: Pipeline analysis data

    Returns:
        Dictionary with validation results
    """
    validation_results: dict[str, Any] = {"valid": True, "warnings": [], "errors": []}

    try:
        # Check required fields
        required_fields: list[Any] = ["steps", "summary", "report_generation_time"]
        for field in required_fields:
            if field not in pipeline_data:
                validation_results["errors"].append(f"Missing required field: {field}")
                validation_results["valid"] = False

        # Check step data integrity
        steps = pipeline_data.get("steps", {})
        for step_name, step_data in steps.items():
            if not isinstance(step_data, dict):
                validation_results["errors"].append(
                    f"Invalid step data for {step_name}"
                )
                validation_results["valid"] = False
                continue

            # Check step data structure
            if "exists" not in step_data:
                validation_results["warnings"].append(
                    f"Missing 'exists' field for step {step_name}"
                )

            if step_data.get("exists", False):
                if "file_count" not in step_data:
                    validation_results["warnings"].append(
                        f"Missing 'file_count' for step {step_name}"
                    )

                if "total_size_mb" not in step_data:
                    validation_results["warnings"].append(
                        f"Missing 'total_size_mb' for step {step_name}"
                    )

        # Check summary data
        summary = pipeline_data.get("summary", {})
        if "total_files_processed" not in summary:
            validation_results["warnings"].append(
                "Missing 'total_files_processed' in summary"
            )

        if "success_rate" not in summary:
            validation_results["warnings"].append("Missing 'success_rate' in summary")

    except Exception as e:
        validation_results["errors"].append(f"Validation failed with exception: {e}")
        validation_results["valid"] = False

    return validation_results
