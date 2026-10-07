#!/usr/bin/env python3
"""
Report Processor module for GNN Processing Pipeline.

This module provides report processing capabilities.
"""

import logging
from html import escape as html_escape
from pathlib import Path
from typing import Any, Dict, cast

from gnn.utils.observability.structured_logging import (
    log_step_error,
    log_step_start,
    log_step_success,
)

logger = logging.getLogger(__name__)


def process_report(
    target_dir: Path, output_dir: Path, verbose: bool = False, **kwargs: Any
) -> bool:
    """
    Process report for GNN files.

    Args:
        target_dir: Directory containing GNN files to process
        output_dir: Directory to save results
        verbose: Enable verbose output
        **kwargs: Additional arguments

    Returns:
        True if processing successful, False otherwise
    """
    logger = logging.getLogger("report")

    try:
        log_step_start(logger, "Processing report")

        results_dir = output_dir
        results_dir.mkdir(parents=True, exist_ok=True)

        # Basic report processing
        results: dict[str, Any] = {"processed_files": 0, "success": True, "errors": []}

        # Find GNN files
        from gnn.pipeline.run_context import selected_model_sources

        gnn_files = selected_model_sources(target_dir, 23)
        if gnn_files:
            results["processed_files"] = len(gnn_files)

        # Save results
        import json

        results_file = results_dir / "report_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, sort_keys=True)

        if results["success"]:
            log_step_success(logger, "report processing completed successfully")
        else:
            log_step_error(logger, "report processing failed")

        return cast("bool", results["success"])

    except Exception as e:
        log_step_error(logger, "report processing failed", error=str(e))
        return False


def generate_comprehensive_report(
    target_dir: Path, output_dir: Path, format: str = "json", **kwargs: Any
) -> Dict[str, Any]:
    """
    Generate a comprehensive report for GNN files.

    Args:
        target_dir: Directory containing GNN files to analyze
        output_dir: Directory to save the report
        format: Output format (json, html, markdown)
        **kwargs: Additional arguments

    Returns:
        Dictionary with report results
    """
    logger = logging.getLogger("report")

    try:
        log_step_start(logger, "Generating comprehensive report")

        supported_formats = {"html", "json", "markdown"}
        if format not in supported_formats:
            message = (
                f"Unsupported report format {format!r}; expected one of "
                f"{sorted(supported_formats)}"
            )
            log_step_error(logger, message)
            return {"success": False, "error": message, "format": format}

        # Create report directory
        report_dir = output_dir
        report_dir.mkdir(parents=True, exist_ok=True)

        # Analyze GNN files
        from gnn.pipeline.run_context import selected_model_sources

        gnn_files = selected_model_sources(target_dir, 23)

        report_data: dict[str, Any] = {
            "timestamp": "unavailable",
            "timestamp_source": "not_provided",
            "total_files": len(gnn_files),
            "files_analyzed": [],
            "summary": {"success": True, "successful_files": 0, "errors": []},
        }

        # Process each file
        for gnn_file in gnn_files:
            try:
                file_info = analyze_gnn_file(gnn_file)
                report_data["files_analyzed"].append(
                    {"file": str(gnn_file), "info": file_info}
                )
                if "error" in file_info:
                    report_data["summary"]["errors"].append(
                        {"file": str(gnn_file), "error": file_info["error"]}
                    )
                else:
                    report_data["summary"]["successful_files"] += 1
            except Exception as e:
                error_info: dict[str, Any] = {"file": str(gnn_file), "error": str(e)}
                report_data["summary"]["errors"].append(error_info)

        report_data["summary"]["success"] = not report_data["summary"]["errors"]

        # Generate report in specified format
        if format == "json":
            report_file = report_dir / "comprehensive_report.json"
            import json

            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2, sort_keys=True)
        elif format == "html":
            report_file = report_dir / "comprehensive_report.html"
            html_content = generate_html_report(report_data)
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(html_content)
        elif format == "markdown":
            report_file = report_dir / "comprehensive_report.md"
            markdown_content = generate_markdown_report(report_data)
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(markdown_content)

        if report_data["summary"]["success"]:
            log_step_success(
                logger, f"Comprehensive report generated in {format} format"
            )
        else:
            log_step_error(
                logger,
                "Comprehensive report generated with "
                f"{len(report_data['summary']['errors'])} analysis error(s)",
            )

        return {
            "success": report_data["summary"]["success"],
            "report_file": str(report_file),
            "format": format,
            "files_scanned": report_data["total_files"],
            "files_analyzed": report_data["summary"]["successful_files"],
            "error_count": len(report_data["summary"]["errors"]),
        }

    except Exception as e:
        log_step_error(logger, f"Failed to generate comprehensive report: {e}")
        return {"success": False, "error": str(e)}


def analyze_gnn_file(file_path: Path) -> Dict[str, Any]:
    """
    Analyze a GNN file for report generation.

    Args:
        file_path: Path to GNN file

    Returns:
        Dictionary with file analysis
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Basic analysis
        analysis: dict[str, Any] = {
            "file_size": len(content),
            "lines": len(content.split("\n")),
            "sections": [],
            "has_model_name": "ModelName:" in content,
            "has_state_space": "StateSpaceBlock:" in content,
            "has_gnn_version": "GNNVersionAndFlags:" in content,
        }

        # Extract sections
        lines = content.split("\n")
        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith("#"):
                current_section = line[1:].strip()
                analysis["sections"].append(current_section)

        return analysis

    except Exception as e:
        return {"error": str(e)}


def generate_html_report(report_data: Dict[str, Any]) -> str:
    """
    Generate HTML report.

    Args:
        report_data: Report data dictionary

    Returns:
        HTML content string
    """
    summary = report_data.get("summary", {})
    successful_files = summary.get(
        "successful_files",
        sum(
            "error" not in item.get("info", {})
            for item in report_data.get("files_analyzed", [])
        ),
    )
    evidence_as_of = html_escape(str(report_data.get("timestamp", "unavailable")))
    total_files = html_escape(str(report_data.get("total_files", 0)))
    successful_files_text = html_escape(str(successful_files))
    error_count = html_escape(str(len(summary.get("errors", []))))

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>GNN Comprehensive Analysis Report</title>
        <style>
            :root {{
                --bg-color: #fcfcfc;
                --text-main: #333333;
                --text-muted: #666666;
                --accent: #2c3e50;
                --border: #e2e8f0;
            }}
            body {{
                font-family: 'Merriweather', 'Georgia', serif;
                line-height: 1.8;
                max-width: 900px;
                margin: 0 auto;
                padding: 40px 20px;
                background-color: var(--bg-color);
                color: var(--text-main);
            }}
            .manuscript-header {{
                text-align: center;
                border-bottom: 2px solid var(--accent);
                padding-bottom: 20px;
                margin-bottom: 40px;
            }}
            h1 {{ font-size: 2.2em; color: var(--accent); margin-bottom: 10px; font-family: 'Inter', sans-serif; }}
            h2 {{ font-size: 1.5em; color: var(--accent); border-bottom: 1px solid var(--border); padding-bottom: 5px; margin-top: 40px; font-family: 'Inter', sans-serif; }}
            .metadata-block {{
                background: #ffffff;
                border: 1px solid var(--border);
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 30px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.02);
            }}
            .metadata-block p {{ margin: 5px 0; font-family: 'Inter', sans-serif; font-size: 0.9em; }}
            .file-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                gap: 15px;
                margin-top: 20px;
            }}
            .file-card {{
                background: #ffffff;
                border: 1px solid var(--border);
                padding: 15px;
                border-radius: 6px;
                font-family: 'Inter', sans-serif;
                font-size: 0.9em;
            }}
            .file-card code {{ color: #e53e3e; background: #fff5f5; padding: 2px 4px; border-radius: 4px; }}
            .mermaid-container {{ margin: 40px 0; text-align: center; }}
        </style>
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ startOnLoad: true, theme: 'neutral' }});
        </script>
    </head>
    <body>
        <div class="manuscript-header">
            <h1>GNN Structural Input Report</h1>
            <p style="font-style: italic; color: var(--text-muted);">Evidence as of: {evidence_as_of}</p>
        </div>
        
        <div class="mermaid-container">
            <div class="mermaid">
            graph LR
                A[GNN Input Files] --> B{{GNN Processor}}
                B --> C[Serialization]
                B --> D[Semantic Ontology]
                B --> E[LLM Inference]
                C --> F[Simulation Generation]
                D --> F
                E --> G[Final Analysis]
                F --> G
            </div>
            <p style="font-size: 0.8em; color: var(--text-muted);">Figure 1. Reference pipeline topology; this diagram is not execution evidence.</p>
        </div>

        <h2>I. Executive Summary</h2>
        <div class="metadata-block">
            <p><strong>Total Scanned Entities:</strong> {total_files}</p>
            <p><strong>Entities Successfully Evaluated:</strong> {successful_files_text}</p>
            <p><strong>Evaluation Errors:</strong> {error_count}</p>
        </div>
        
        <h2>II. Processed Models Validation</h2>
        <div class="file-grid">
    """

    for file_info in report_data.get("files_analyzed", []):
        info = file_info.get("info", {})
        name = html_escape(Path(file_info["file"]).name)
        size = html_escape(str(info.get("file_size", 0)))
        lines = html_escape(str(info.get("lines", 0)))
        html += f"""
            <div class="file-card">
                <strong>{name}</strong>
                <p>Size: {size} bytes | Lines: {lines}</p>
                <p>State Space Matrix: <code>{"Yes" if info.get("has_state_space") else "No"}</code></p>
            </div>
        """

    html += """
        </div>
    </body>
    </html>
    """

    return html


def generate_markdown_report(report_data: Dict[str, Any]) -> str:
    """
    Generate Markdown report.

    Args:
        report_data: Report data dictionary

    Returns:
        Markdown content string
    """
    summary = report_data.get("summary", {})
    successful_files = summary.get(
        "successful_files",
        sum(
            "error" not in item.get("info", {})
            for item in report_data.get("files_analyzed", [])
        ),
    )
    markdown = f"""# GNN Comprehensive Analysis Report

> **Evidence as of:** {report_data.get("timestamp", "unavailable")}
> **Purpose:** Top-level structural audit of GNN notation topologies.
> **Evidence note:** The topology below is a reference diagram, not execution proof.

## System Topology Flow

```mermaid
graph LR
    A[GNN Input Files] --> B{{GNN Processor}}
    B --> C[Serialization]
    B --> D[Semantic Ontology]
    B --> E[LLM Inference]
    C --> F[Simulation Generation]
    D --> F
    E --> G[Final Analysis]
    F --> G
```

## I. Executive Summary

- **Total Scanned Entities**: {report_data.get("total_files", 0)}
- **Entities Successfully Evaluated**: {successful_files}
- **Evaluation Errors**: {len(summary.get("errors", []))}

## II. Processed Models Validation

| Model Filename | Size (Bytes) | Lines | State Space Discovered |
|---|---|---|---|
"""

    for file_info in report_data.get("files_analyzed", []):
        info = file_info.get("info", {})
        name = Path(file_info["file"]).name.replace("|", "\\|").replace("\n", " ")
        size = info.get("file_size", 0)
        lines = info.get("lines", 0)
        has_state = "✅ Yes" if info.get("has_state_space") else "❌ No"
        markdown += f"| `{name}` | {size} | {lines} | {has_state} |\n"

    return markdown


# Explicit alias so __init__.py can import by this name without shadowing generator's generate_comprehensive_report
generate_report = generate_comprehensive_report
