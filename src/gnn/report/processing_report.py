#!/usr/bin/env python3
"""
GNN Reporting Module

This module provides comprehensive reporting capabilities for GNN processing
results, including detailed analysis, performance metrics, and export formats.
"""

import json
import logging
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Comprehensive report generator for GNN processing results.

    Generates detailed reports in multiple formats with performance
    metrics, validation summaries, and processing insights.
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        self.report_formats = ["json", "markdown", "html"]

    def generate_processing_report(
        self,
        context: Any | None = None,
        output_dir: Path | None = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive processing report.

        Args:
            context: Processing context with results (None returns an empty
                success envelope for lightweight callers)
            output_dir: Directory for report output (defaults to the
                context's output directory)

        Returns:
            Dictionary with report metadata and file paths
        """
        if context is None:
            return {"status": "SUCCESS", "reports": []}
        if output_dir is None:
            output_dir = Path(context.output_dir)
        logger.info("Generating comprehensive processing report")

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Compile comprehensive report data
        report_data = self._compile_report_data(context)

        # Generate reports in multiple formats
        report_files: dict[Any, Any] = {}

        try:
            # JSON report (detailed data)
            json_file = output_dir / f"gnn_processing_report_{timestamp}.json"
            self._generate_json_report(report_data, json_file)
            report_files["json"] = str(json_file)

            # Markdown report (human-readable)
            md_file = output_dir / f"gnn_processing_report_{timestamp}.md"
            self._generate_markdown_report(report_data, md_file)
            report_files["markdown"] = str(md_file)

            # HTML report (web-viewable)
            html_file = output_dir / f"gnn_processing_report_{timestamp}.html"
            self._generate_html_report(report_data, html_file)
            report_files["html"] = str(html_file)

            logger.info(f"Reports generated successfully in {output_dir}")

        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            report_files["error"] = str(e)

        return {
            "timestamp": timestamp,
            "report_files": report_files,
            "report_data": report_data,
        }

    def generate_report(self, data: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Generate a lightweight report envelope from arbitrary data."""
        return {"status": "SUCCESS", "data": data or {}}

    def format_report(self, content: Any, fmt: str = "markdown") -> str:
        """Render report content as markdown or HTML text."""
        if fmt == "html":
            return f"<html><body><pre>{content}</pre></body></html>"
        return f"# Report\n\n{content}"

    def _compile_report_data(self, context: Any) -> Dict[str, Any]:
        """Compile comprehensive report data from processing context."""
        processing_time = context.get_processing_time()

        report_data: dict[str, Any] = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "target_directory": str(context.target_dir),
                "output_directory": str(context.output_dir),
                "processing_time": f"{processing_time:.2f}s",
                "validation_level": context.validation_level,
                "configuration": {
                    "recursive": context.recursive,
                    "round_trip_enabled": context.enable_round_trip,
                    "cross_format_enabled": context.enable_cross_format,
                    "test_subset": context.test_subset,
                    "reference_file": context.reference_file,
                },
            },
            "processing_summary": {
                "total_files_discovered": len(context.discovered_files),
                "valid_files_found": len(context.valid_files),
                "validation_success_rate": self._calculate_validation_success_rate(
                    context
                ),
                "phase_durations": {
                    phase.name.lower(): context.get_phase_duration(phase)
                    for phase in context.phase_times
                },
                "phase_logs": {
                    phase.name.lower(): message
                    for phase, message in context.phase_logs.items()
                },
            },
            "file_analysis": self._analyze_discovered_files(context),
            "validation_analysis": self._analyze_validation_results(context),
            "performance_metrics": self._calculate_performance_metrics(context),
            "recommendations": self._generate_recommendations(context),
        }

        # Add round-trip results if available
        if "round_trip_results" in context.processing_results:
            report_data["round_trip_analysis"] = self._analyze_round_trip_results(
                context.processing_results["round_trip_results"]
            )

        # Add cross-format results if available
        if "cross_format_results" in context.processing_results:
            report_data["cross_format_analysis"] = self._analyze_cross_format_results(
                context.processing_results["cross_format_results"]
            )

        return report_data

    def _calculate_validation_success_rate(self, context: Any) -> float:
        """Calculate validation success rate."""
        total_files = len(context.discovered_files)
        valid_files = len(context.valid_files)
        return (valid_files / total_files * 100) if total_files > 0 else 0.0

    def _analyze_discovered_files(self, context: Any) -> Dict[str, Any]:
        """Analyze discovered files for report."""
        files = context.discovered_files

        if not files:
            return {"total": 0, "formats": {}, "sizes": {}}

        # Format analysis
        formats: dict[Any, Any] = {}
        sizes: list[Any] = []

        for file_path in files:
            try:
                ext = file_path.suffix.lower()
                formats[ext] = formats.get(ext, 0) + 1
                sizes.append(file_path.stat().st_size)
            except Exception as e:
                logger.debug(f"Could not stat file {file_path}: {e}")
                continue

        return {
            "total": len(files),
            "formats": formats,
            "sizes": {
                "total_size": sum(sizes),
                "average_size": sum(sizes) / len(sizes) if sizes else 0,
                "size_range": (min(sizes), max(sizes)) if sizes else (0, 0),
            },
            "file_list": [str(f) for f in files],
        }

    def _analyze_validation_results(self, context: Any) -> Dict[str, Any]:
        """Analyze validation results for report."""
        validation_results = context.processing_results.get("validation_results", {})

        if not validation_results:
            return {"total": 0, "valid": 0, "invalid": 0, "error_patterns": {}}

        valid_count = 0
        invalid_count = 0
        error_patterns: dict[Any, Any] = {}
        warning_patterns: dict[Any, Any] = {}

        for file_path, result in validation_results.items():
            # Structural validation saves mappings; semantic validation saves
            # typed results. Required fields retain either producer's failures.
            if isinstance(result, Mapping):
                is_valid = result["valid"]
                errors = result["errors"]
                warnings = result["warnings"]
            else:
                is_valid = result.is_valid
                errors = result.errors
                warnings = result.warnings
            if not isinstance(is_valid, bool):
                raise TypeError(f"Validation validity must be a bool for {file_path}")
            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1

            # Analyze error patterns
            for error in errors:
                error_type = error.split(":")[0] if ":" in error else "General"
                error_patterns[error_type] = error_patterns.get(error_type, 0) + 1

            # Analyze warning patterns
            for warning in warnings:
                warning_type = warning.split(":")[0] if ":" in warning else "General"
                warning_patterns[warning_type] = (
                    warning_patterns.get(warning_type, 0) + 1
                )

        return {
            "total": len(validation_results),
            "valid": valid_count,
            "invalid": invalid_count,
            "success_rate": (valid_count / len(validation_results) * 100)
            if validation_results
            else 0,
            "error_patterns": error_patterns,
            "warning_patterns": warning_patterns,
        }

    def _analyze_round_trip_results(
        self, round_trip_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze round-trip test results."""
        if not round_trip_results or not round_trip_results.get("success"):
            return {
                "enabled": False,
                "error": round_trip_results.get("error", "Unknown error"),
            }

        summary = round_trip_results.get("summary", {})

        return {
            "enabled": True,
            "total_files_tested": summary.get("total_files", 0),
            "successful_files": summary.get("successful_files", 0),
            "failed_files": summary.get("failed_files", 0),
            "average_success_rate": summary.get("average_success_rate", 0.0),
            "format_performance": summary.get("format_performance", {}),
            "common_errors": summary.get("common_errors", []),
        }

    def _analyze_cross_format_results(
        self, cross_format_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze cross-format validation results."""
        if not cross_format_results or not cross_format_results.get("success"):
            return {
                "enabled": False,
                "error": cross_format_results.get("error", "Unknown error"),
            }

        summary = cross_format_results.get("summary", {})

        return {
            "enabled": True,
            "total_files_validated": summary.get("total_files", 0),
            "consistent_files": summary.get("consistent_files", 0),
            "inconsistent_files": summary.get("inconsistent_files", 0),
            "average_consistency_rate": summary.get("average_consistency_rate", 0.0),
            "common_inconsistencies": summary.get("common_inconsistencies", []),
        }

    def _calculate_performance_metrics(self, context: Any) -> Dict[str, Any]:
        """Calculate performance metrics for report."""
        total_time = context.get_processing_time()
        total_files = len(context.discovered_files)

        return {
            "total_processing_time": f"{total_time:.2f}s",
            "files_per_second": (total_files / total_time) if total_time > 0 else 0,
            "average_file_processing_time": (total_time / total_files)
            if total_files > 0
            else 0,
            "phase_performance": {
                phase.name.lower(): f"{context.get_phase_duration(phase):.2f}s"
                for phase in context.phase_times.keys()
            },
        }

    def _generate_recommendations(self, context: Any) -> List[str]:
        """Generate recommendations based on processing results."""
        recommendations: list[Any] = []

        # File discovery recommendations
        discovered_count = len(context.discovered_files)
        valid_count = len(context.valid_files)

        if discovered_count == 0:
            recommendations.append(
                "No GNN files found. Check target directory and file formats."
            )
        elif valid_count == 0:
            recommendations.append(
                "No valid GNN files found. Review file structure and syntax."
            )
        elif valid_count < discovered_count * 0.8:
            recommendations.append(
                "Many files failed validation. Consider reviewing file quality."
            )

        # Validation level recommendations
        if context.validation_level == "basic":
            recommendations.append(
                "Consider using 'standard' validation for better quality assurance."
            )
        elif context.validation_level == "standard" and valid_count == discovered_count:
            recommendations.append(
                "All files pass standard validation. Consider 'strict' for enhanced quality."
            )

        # Testing recommendations
        if not context.enable_round_trip:
            recommendations.append(
                "Enable round-trip testing for semantic preservation verification."
            )

        if not context.enable_cross_format:
            recommendations.append(
                "Enable cross-format validation for consistency checking."
            )

        # Performance recommendations
        total_time = context.get_processing_time()
        if total_time > 30 and discovered_count < 10:
            recommendations.append(
                "Processing time is high for few files. Check for performance issues."
            )

        return recommendations

    def _generate_json_report(
        self, report_data: Dict[str, Any], output_file: Path
    ) -> Any:
        """Generate detailed JSON report."""
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, default=str, ensure_ascii=False)
        logger.debug(f"JSON report saved: {output_file}")

    def _generate_markdown_report(
        self, report_data: Dict[str, Any], output_file: Path
    ) -> Any:
        """Generate human-readable Markdown report."""
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("# GNN Processing Report\n\n")

            # Metadata section
            metadata = report_data["metadata"]
            f.write("## Processing Summary\n\n")
            f.write(f"**Date:** {metadata['timestamp']}\n")
            f.write(f"**Target Directory:** `{metadata['target_directory']}`\n")
            f.write(f"**Output Directory:** `{metadata['output_directory']}`\n")
            f.write(f"**Processing Time:** {metadata['processing_time']}\n")
            f.write(f"**Validation Level:** {metadata['validation_level']}\n\n")

            # Configuration
            config = metadata["configuration"]
            f.write("### Configuration\n\n")
            f.write(f"- **Recursive Search:** {config['recursive']}\n")
            f.write(f"- **Round-Trip Testing:** {config['round_trip_enabled']}\n")
            f.write(
                f"- **Cross-Format Validation:** {config['cross_format_enabled']}\n\n"
            )

            # Results summary
            summary = report_data["processing_summary"]
            f.write("## Results Overview\n\n")
            f.write(f"- **Files Discovered:** {summary['total_files_discovered']}\n")
            f.write(f"- **Valid Files:** {summary['valid_files_found']}\n")
            f.write(
                f"- **Success Rate:** {summary['validation_success_rate']:.1f}%\n\n"
            )

            # File analysis
            file_analysis = report_data["file_analysis"]
            if file_analysis["formats"]:
                f.write("## File Format Distribution\n\n")
                for ext, count in file_analysis["formats"].items():
                    f.write(f"- **{ext}**: {count} files\n")
                f.write("\n")

            # Validation analysis
            validation_analysis = report_data["validation_analysis"]
            if validation_analysis.get("error_patterns"):
                f.write("## Common Validation Issues\n\n")
                for error_type, count in validation_analysis["error_patterns"].items():
                    f.write(f"- **{error_type}**: {count} occurrences\n")
                f.write("\n")

            # Round-trip results
            if "round_trip_analysis" in report_data:
                rt_analysis = report_data["round_trip_analysis"]
                if rt_analysis["enabled"]:
                    f.write("## Round-Trip Testing Results\n\n")
                    f.write(
                        f"- **Files Tested:** {rt_analysis['total_files_tested']}\n"
                    )
                    f.write(
                        f"- **Success Rate:** {rt_analysis['average_success_rate']:.1f}%\n"
                    )
                    f.write(f"- **Successful:** {rt_analysis['successful_files']}\n")
                    f.write(f"- **Failed:** {rt_analysis['failed_files']}\n\n")

            # Cross-format results
            if "cross_format_analysis" in report_data:
                cf_analysis = report_data["cross_format_analysis"]
                if cf_analysis["enabled"]:
                    f.write("## Cross-Format Validation Results\n\n")
                    f.write(
                        f"- **Files Validated:** {cf_analysis['total_files_validated']}\n"
                    )
                    f.write(
                        f"- **Consistency Rate:** {cf_analysis['average_consistency_rate']:.1f}%\n"
                    )
                    f.write(f"- **Consistent:** {cf_analysis['consistent_files']}\n")
                    f.write(
                        f"- **Inconsistent:** {cf_analysis['inconsistent_files']}\n\n"
                    )

            # Recommendations
            recommendations = report_data["recommendations"]
            if recommendations:
                f.write("## Recommendations\n\n")
                for rec in recommendations:
                    f.write(f"- {rec}\n")
                f.write("\n")

        logger.debug(f"Markdown report saved: {output_file}")

    def _generate_html_report(
        self, report_data: Dict[str, Any], output_file: Path
    ) -> Any:
        """Generate web-viewable HTML report."""
        html_content = self._create_html_template(report_data)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.debug(f"HTML report saved: {output_file}")

    def _create_html_template(self, report_data: Dict[str, Any]) -> str:
        """Create HTML template for report."""
        metadata = report_data["metadata"]
        summary = report_data["processing_summary"]

        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GNN Processing Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        h1, h2 {{ color: #333; }}
        .summary {{ background: #f4f4f4; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
        .success {{ color: #28a745; }}
        .warning {{ color: #ffc107; }}
        .error {{ color: #dc3545; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>GNN Processing Report</h1>
    
    <div class="summary">
        <h2>Processing Summary</h2>
        <div class="metric"><strong>Date:</strong> {metadata["timestamp"]}</div>
        <div class="metric"><strong>Processing Time:</strong> {metadata["processing_time"]}</div>
        <div class="metric"><strong>Files Discovered:</strong> {summary["total_files_discovered"]}</div>
        <div class="metric"><strong>Valid Files:</strong> {summary["valid_files_found"]}</div>
        <div class="metric"><strong>Success Rate:</strong> 
            <span class="{"success" if summary["validation_success_rate"] >= 80 else "warning" if summary["validation_success_rate"] >= 60 else "error"}">
                {summary["validation_success_rate"]:.1f}%
            </span>
        </div>
    </div>
    
    <h2>Configuration</h2>
    <ul>
        <li><strong>Target Directory:</strong> {metadata["target_directory"]}</li>
        <li><strong>Validation Level:</strong> {metadata["validation_level"]}</li>
        <li><strong>Recursive Search:</strong> {metadata["configuration"]["recursive"]}</li>
        <li><strong>Round-Trip Testing:</strong> {metadata["configuration"]["round_trip_enabled"]}</li>
        <li><strong>Cross-Format Validation:</strong> {metadata["configuration"]["cross_format_enabled"]}</li>
    </ul>
    
    <h2>Performance Metrics</h2>
    <p><strong>Processing Rate:</strong> {report_data["performance_metrics"]["files_per_second"]:.2f} files/second</p>
    <p><strong>Average File Time:</strong> {report_data["performance_metrics"]["average_file_processing_time"]:.3f} seconds</p>
    
    {self._generate_html_recommendations(report_data["recommendations"])}
    
    <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ccc; color: #666;">
        <p>Generated by GNN Processing Pipeline at {metadata["timestamp"]}</p>
    </footer>
</body>
</html>
        """

    def _generate_html_recommendations(self, recommendations: List[str]) -> str:
        """Generate HTML section for recommendations."""
        if not recommendations:
            return ""

        html = "<h2>Recommendations</h2>\n<ul>\n"
        for rec in recommendations:
            html += f"<li>{rec}</li>\n"
        html += "</ul>\n"

        return html
