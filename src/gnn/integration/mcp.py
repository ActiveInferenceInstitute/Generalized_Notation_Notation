"""
MCP integration for the integration module.

Exposes GNN third-party system integration tools: ActiveInference.jl export,
pymdp integration, Pyro/Stan adapters, and integration status through MCP.
"""

import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

from gnn.utils.mcp.dispatch import run_pipeline_step_mcp, run_tool_envelope

from . import process_integration


def process_integration_mcp(
    target_directory: str, output_directory: str, verbose: bool = False
) -> Dict[str, Any]:
    """
    Run third-party integration processing for GNN files.

    Exports GNN models to formats compatible with external systems such as
    ActiveInference.jl, pymdp, Pyro, Stan, and other probabilistic programming
    frameworks.

    Args:
        target_directory: Directory containing GNN files to process
        output_directory: Directory to save integration outputs
        verbose: Enable verbose logging

    Returns:
        Dictionary with success status and integration summary.
    """
    return run_pipeline_step_mcp(
        process_integration,
        wrapper_name="process_integration_mcp",
        logger=logger,
        target_directory=target_directory,
        output_directory=output_directory,
        verbose=verbose,
        label="Integration processing",
    )


def list_supported_integrations_mcp() -> Dict[str, Any]:
    """
    Return all supported third-party integration targets.

    Returns:
        Dictionary with integration names, descriptions, and availability.
    """
    integrations: dict[str, Any] = {
        "activeinference_jl": {
            "description": "ActiveInference.jl Julia package",
            "output_format": "julia",
            "available": True,
        },
        "pymdp": {
            "description": "pymdp Python Active Inference library",
            "output_format": "python",
            "available": True,
        },
        "rxinfer": {
            "description": "RxInfer.jl reactive Bayesian inference",
            "output_format": "julia",
            "available": True,
        },
        "pyro": {
            "description": "Uber Pyro probabilistic programming",
            "output_format": "python",
            "available": False,
        },
        "stan": {
            "description": "Stan statistical modelling language",
            "output_format": "stan",
            "available": False,
        },
    }
    return {"success": True, "integrations": integrations, "count": len(integrations)}


def get_integration_status_mcp(output_directory: str) -> Dict[str, Any]:
    """
    Check the status of a previous integration run.

    Args:
        output_directory: Directory where integration outputs were saved

    Returns:
        Dictionary with output file inventory and per-integration status.
    """

    def _build() -> Dict[str, Any]:
        out_dir = Path(output_directory)
        if not out_dir.exists():
            return {
                "success": False,
                "error": f"Directory not found: {output_directory}",
            }

        files = list(out_dir.rglob("*"))
        by_ext: Dict[str, int] = {}
        for f in files:
            if f.is_file():
                ext = f.suffix.lstrip(".")
                by_ext[ext] = by_ext.get(ext, 0) + 1

        return {
            "success": True,
            "directory": str(out_dir),
            "total_files": len([f for f in files if f.is_file()]),
            "by_extension": by_ext,
        }

    return run_tool_envelope(
        _build,
        wrapper_name="get_integration_status_mcp",
        logger=logger,
    )


def check_integration_dependencies_mcp() -> Dict[str, Any]:
    """
    Check which third-party integration dependencies are installed.

    Uses ``importlib.util.find_spec`` (lookup only, no execution) to avoid
    triggering heavy C-extension loading that can crash the process.

    Returns:
        Dictionary with dependency names and availability flags.
    """
    import importlib.util

    from gnn.utils.runtime_safety.julia_probe import julia_executable

    deps: Dict[str, Dict[str, Any]] = {}
    for pkg, label in [
        ("pymdp", "pymdp"),
        ("jax", "JAX"),
        ("torch", "PyTorch"),
        ("numpyro", "NumPyro"),
    ]:
        try:
            found = importlib.util.find_spec(pkg) is not None
        except (ModuleNotFoundError, ValueError):
            found = False
        deps[label] = {"available": found}

    deps["Julia"] = {"available": julia_executable() is not None}
    return {"success": True, "dependencies": deps}


# ── MCP Registration ────────────────────────────────────────────────────────


def register_tools(mcp_instance: Any) -> None:
    """Register integration tools with the MCP server."""

    mcp_instance.register_tool(
        "process_integration",
        process_integration_mcp,
        {
            "type": "object",
            "properties": {
                "target_directory": {
                    "type": "string",
                    "description": "Directory with GNN files",
                },
                "output_directory": {
                    "type": "string",
                    "description": "Integration output directory",
                },
                "verbose": {"type": "boolean", "default": False},
            },
            "required": ["target_directory", "output_directory"],
        },
        "Run GNN integration processing: export to ActiveInference.jl, pymdp, Pyro, Stan, etc.",
        module=__package__,
        category="integration",
    )

    mcp_instance.register_tool(
        "list_supported_integrations",
        list_supported_integrations_mcp,
        {},
        "Return all supported third-party integration targets and their availability.",
        module=__package__,
        category="integration",
    )

    mcp_instance.register_tool(
        "get_integration_status",
        get_integration_status_mcp,
        {
            "type": "object",
            "properties": {
                "output_directory": {
                    "type": "string",
                    "description": "Directory with integration outputs",
                },
            },
            "required": ["output_directory"],
        },
        "Check status and inventory of a previous integration run.",
        module=__package__,
        category="integration",
    )

    mcp_instance.register_tool(
        "check_integration_dependencies",
        check_integration_dependencies_mcp,
        {},
        "Check which third-party integration dependencies (pymdp, JAX, Julia, etc.) are installed.",
        module=__package__,
        category="integration",
    )

    logger.info("integration module MCP tools registered (4 tools).")
