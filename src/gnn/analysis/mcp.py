"""
Analysis MCP Integration

Exposes GNN analysis processing, complexity metrics, and result inspection
through the Model Context Protocol with full typed schemas.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from gnn.utils.mcp.dispatch import run_pipeline_step_mcp, run_tool_envelope

from . import process_analysis


def process_analysis_mcp(
    target_directory: str, output_directory: str, verbose: bool = False
) -> Dict[str, Any]:
    """
    Run GNN statistical and complexity analysis on a directory of GNN files.

    Args:
        target_directory: Directory containing GNN files to analyse
        output_directory: Directory to save analysis results (JSON)
        verbose: Enable verbose logging

    Returns:
        Dictionary with success status and summary of analysis performed.
    """
    return run_pipeline_step_mcp(
        process_analysis,
        wrapper_name="process_analysis_mcp",
        logger=logger,
        target_directory=target_directory,
        output_directory=output_directory,
        verbose=verbose,
        label="Analysis",
    )


def get_analysis_results_mcp(
    output_directory: str, model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Read and return saved analysis results from a previous analysis run.

    Searches the output directory for analysis_results.json or per-model JSON
    files and returns their contents as structured data.

    Args:
        output_directory: Directory where analysis results were saved
        model_name: Optional model name filter (returns only that model's results)

    Returns:
        Dictionary containing analysis results data.
    """

    def _build() -> Dict[str, Any]:
        out_dir = Path(output_directory)
        if not out_dir.exists():
            return {
                "success": False,
                "error": f"Output directory not found: {output_directory}",
            }

        results: List[Dict[str, Any]] = []
        for json_file in sorted(out_dir.rglob("*.json")):
            try:
                data = json.loads(json_file.read_text())
                if model_name and model_name not in json_file.name:
                    continue
                results.append({"file": json_file.name, "data": data})
            except (json.JSONDecodeError, OSError, ValueError) as e:
                logger.warning(f"Skipping {json_file.name}: {e}")

        return {
            "success": True,
            "output_directory": str(out_dir),
            "results_count": len(results),
            "results": results,
        }

    return run_tool_envelope(
        _build,
        wrapper_name="get_analysis_results_mcp",
        logger=logger,
    )


def compute_complexity_metrics_mcp(
    gnn_content: str, model_name: str = "model"
) -> Dict[str, Any]:
    """
    Compute complexity metrics for a GNN model provided as string content.

    Parses the GNN file and computes:
    - Number of state variables
    - Number of connections / edges
    - Parameter count
    - Estimated cyclomatic complexity
    - Section count

    Args:
        gnn_content: GNN model content as a string
        model_name: Human-readable name for this model

    Returns:
        Dictionary with computed complexity metrics.
    """
    try:
        lines = gnn_content.splitlines()
        sections = [l.strip("# ").strip() for l in lines if l.startswith("## ")]
        connections = [l for l in lines if "->" in l or "<->" in l or "→" in l]
        variables = [
            l for l in lines if "[" in l and "]" in l and not l.strip().startswith("#")
        ]
        parameters = [
            l
            for l in lines
            if "=" in l and not l.strip().startswith("#") and "->" not in l
        ]

        n_vars = len(variables)
        n_conns = len(connections)
        n_params = len(parameters)
        n_secs = len(sections)
        cyclomatic = max(1, n_conns - n_vars + 2)

        return {
            "success": True,
            "model_name": model_name,
            "sections": n_secs,
            "state_variables": n_vars,
            "connections": n_conns,
            "parameters": n_params,
            "total_lines": len(lines),
            "cyclomatic_complexity": cyclomatic,
            "complexity_rating": (
                "low" if cyclomatic < 5 else "medium" if cyclomatic < 15 else "high"
            ),
        }
    except Exception as e:
        logger.error(f"compute_complexity_metrics_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def list_analysis_tools_mcp() -> Dict[str, Any]:
    """
    Return information about available analysis tools and their capabilities.

    Returns:
        Dictionary with tool names, descriptions, and module metadata.
    """
    try:
        from . import check_analysis_tools

        tools_info = check_analysis_tools()
        return {"success": True, "tools": tools_info}
    except Exception as e:
        logger.error("list_analysis_tools_mcp error: %s", e, exc_info=True)
        return {
            "success": False,
            "error": str(e) if e else "Unknown error",
            "tools": {},
        }


def evaluate_torchlens_fidelity_mcp(results_path: str) -> Dict[str, Any]:
    """
    Compare declared GNN connections with TorchLens-traced PyTorch dataflow.

    Reads one PyTorch ``simulation_results.json`` (or every PyTorch result
    under a directory) and evaluates its recorded TorchLens capture: each
    declared connection is classified as direct, indirect, reverse_only,
    not_observed or unmapped, with undeclared traced dependencies and the
    recorded lesion witnesses. Read-only: nothing is executed or written.

    Args:
        results_path: A simulation_results.json file or a directory to search

    Returns:
        Dictionary with one fidelity report per result file.
    """

    def _build() -> Dict[str, Any]:
        from .flat_payload_analyzer import discover_result_files, result_model_name
        from .pytorch.analyzer import PYTORCH_SPEC
        from .pytorch.torchlens_fidelity import evaluate_capture

        path = Path(results_path)
        if path.is_file():
            files = [path]
            directory = None
        elif path.is_dir():
            files = sorted(discover_result_files(path, PYTORCH_SPEC))
            directory = path.resolve()
        else:
            return {"success": False, "error": f"Path not found: {results_path}"}

        reports: List[Dict[str, Any]] = []
        for results_file in files:
            try:
                if directory is not None and (
                    results_file.is_symlink()
                    or not results_file.resolve().is_relative_to(directory)
                ):
                    raise ValueError(
                        "result path must be inside the selected directory and not a symbolic link"
                    )
                results = json.loads(results_file.read_text(encoding="utf-8"))
                if not isinstance(results, dict):
                    raise ValueError("simulation result must be a JSON object")
            except (OSError, ValueError) as e:
                reports.append({"file": str(results_file), "error": str(e)})
                continue
            torchlens = results.get("torchlens")
            model_name = result_model_name(results_file, results, "pytorch")
            if torchlens is None:
                report: Dict[str, Any] = {
                    "status": "not_captured",
                    "reason": "result carries no TorchLens capture summary",
                }
            elif isinstance(torchlens, dict) and torchlens.get("status") == "disabled":
                report = {
                    "status": "not_captured",
                    "reason": torchlens.get("reason", "capture disabled"),
                }
            else:
                report = evaluate_capture(torchlens)
            reports.append(
                {"file": str(results_file), "model_name": model_name, "report": report}
            )
        return {"success": True, "reports_count": len(reports), "reports": reports}

    return run_tool_envelope(
        _build,
        wrapper_name="evaluate_torchlens_fidelity_mcp",
        logger=logger,
    )


# ── MCP Registration ────────────────────────────────────────────────────────


def register_tools(mcp_instance: Any) -> None:
    """Register analysis tools with the MCP server."""

    mcp_instance.register_tool(
        "process_analysis",
        process_analysis_mcp,
        {
            "type": "object",
            "properties": {
                "target_directory": {
                    "type": "string",
                    "description": "Directory containing GNN files to analyse",
                },
                "output_directory": {
                    "type": "string",
                    "description": "Directory to save analysis results",
                },
                "verbose": {
                    "type": "boolean",
                    "description": "Enable verbose logging",
                    "default": False,
                },
            },
            "required": ["target_directory", "output_directory"],
        },
        "Run statistical and complexity analysis on GNN files in a directory.",
        module=__package__,
        category="analysis",
    )

    mcp_instance.register_tool(
        "get_analysis_results",
        get_analysis_results_mcp,
        {
            "type": "object",
            "properties": {
                "output_directory": {
                    "type": "string",
                    "description": "Directory containing saved analysis JSON results",
                },
                "model_name": {
                    "type": "string",
                    "description": "Optional model name to filter results",
                },
            },
            "required": ["output_directory"],
        },
        "Read and return saved analysis results from a previous analysis run.",
        module=__package__,
        category="analysis",
    )

    mcp_instance.register_tool(
        "compute_complexity_metrics",
        compute_complexity_metrics_mcp,
        {
            "type": "object",
            "properties": {
                "gnn_content": {
                    "type": "string",
                    "description": "GNN model content as a string",
                },
                "model_name": {
                    "type": "string",
                    "description": "Human-readable model name",
                    "default": "model",
                },
            },
            "required": ["gnn_content"],
        },
        "Compute complexity metrics (variables, connections, cyclomatic complexity) for GNN content.",
        module=__package__,
        category="analysis",
    )

    mcp_instance.register_tool(
        "list_analysis_tools",
        list_analysis_tools_mcp,
        {},
        "Return information about available GNN analysis tools and capabilities.",
        module=__package__,
        category="analysis",
    )

    mcp_instance.register_tool(
        "evaluate_torchlens_fidelity",
        evaluate_torchlens_fidelity_mcp,
        {
            "type": "object",
            "properties": {
                "results_path": {
                    "type": "string",
                    "description": (
                        "PyTorch simulation_results.json file, or a directory "
                        "(e.g. 12_execute_output) to search for them"
                    ),
                },
            },
            "required": ["results_path"],
        },
        "Compare declared GNN connections with TorchLens-traced PyTorch dataflow (read-only).",
        module=__package__,
        category="analysis",
    )

    logger.info("analysis module MCP tools registered (5 tools).")
