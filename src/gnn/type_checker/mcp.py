"""
MCP (Model Context Protocol) integration for type_checker utilities.

This module exposes utility functions from the type_checker module through MCP.
"""

import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Import utilities from the type_checker module via class API exposed in __init__
from . import GNNTypeChecker

# MCP Tools for Type Checker Utilities Module


def validate_gnn_files_mcp(
    target_directory: str,
    output_directory: str,
    strict: bool = False,
    estimate_resources: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Validate GNN files for syntax and type correctness. Exposed via MCP.

    Args:
        target_directory: Directory containing GNN files to validate
        output_directory: Directory to save validation results
        strict: Enable strict type checking mode
        estimate_resources: Estimate computational resources
        verbose: Enable verbose output

    Returns:
        Dictionary with operation status and validation results. ``success``
        is a real bool — only the pipeline ``True`` sentinel counts — and the
        checker's raw exit sentinel is reported separately as ``exit_code``.
    """
    try:
        checker = GNNTypeChecker(strict_mode=strict)
        returned = checker.validate_gnn_files(
            target_dir=Path(target_directory),
            output_dir=Path(output_directory),
            verbose=verbose,
            strict=strict,
            estimate_resources=estimate_resources,
        )
        # ``validate_gnn_files`` returns the pipeline exit sentinel:
        # True (exit 0), 2 (exit 2, SUCCESS_WITH_WARNINGS), False (exit 1).
        # Only literal ``True`` is a pass. ``False == 0`` is true in Python,
        # so testing the sentinel against 0 would report a hard failure as
        # success; the returned value is echoed on refusal so the refusal is
        # diagnosable.
        if returned is True:
            exit_code, outcome = 0, "completed successfully"
        elif returned == 2:
            exit_code, outcome = (
                2,
                "completed with warnings (nothing to process or some files invalid)",
            )
        else:
            exit_code, outcome = 1, f"failed (checker returned {returned!r})"
        return {
            "success": returned is True,
            "exit_code": exit_code,
            "target_directory": target_directory,
            "output_directory": output_directory,
            "strict_mode": strict,
            "resource_estimation": estimate_resources,
            "message": f"GNN validation {outcome}",
        }
    except Exception as e:
        logger.error(
            f"Error in validate_gnn_files_mcp for {target_directory}: {e}",
            exc_info=True,
        )
        return {"success": False, "error": str(e)}


def validate_single_gnn_file_mcp(
    gnn_file_path: str, strict: bool = False, estimate_resources: bool = False
) -> Dict[str, Any]:
    """
    Validate a single GNN file. Exposed via MCP.

    Args:
        gnn_file_path: Path to the GNN file to validate
        strict: Enable strict validation mode
        estimate_resources: Estimate computational resources

    Returns:
        Dictionary with validation results.
    """
    try:
        checker = GNNTypeChecker(strict_mode=strict)
        result = checker.validate_single_gnn_file(
            file_path=Path(gnn_file_path),
            strict=strict,
        )
        return {
            "success": result["valid"],
            "file": gnn_file_path,
            "validation_result": result,
            "message": f"File validation {'passed' if result['valid'] else 'failed'}",
        }
    except Exception as e:
        logger.error(
            f"Error in validate_single_gnn_file_mcp for {gnn_file_path}: {e}",
            exc_info=True,
        )
        return {"success": False, "error": str(e)}


# MCP Registration Function
def register_tools(mcp_instance: Any) -> Any:
    """Register type_checker utility tools with the MCP."""

    mcp_instance.register_tool(
        "validate_gnn_files",
        validate_gnn_files_mcp,
        {
            "target_directory": {
                "type": "string",
                "description": "Directory containing GNN files to validate.",
            },
            "output_directory": {
                "type": "string",
                "description": "Directory to save validation results.",
            },
            "strict": {
                "type": "boolean",
                "description": "Enable strict type checking mode. Defaults to false.",
                "optional": True,
            },
            "estimate_resources": {
                "type": "boolean",
                "description": "Estimate computational resources. Defaults to false.",
                "optional": True,
            },
            "verbose": {
                "type": "boolean",
                "description": "Enable verbose output. Defaults to false.",
                "optional": True,
            },
        },
        "Validate GNN files for syntax and type correctness.",
    )

    mcp_instance.register_tool(
        "validate_single_gnn_file",
        validate_single_gnn_file_mcp,
        {
            "gnn_file_path": {
                "type": "string",
                "description": "Path to the GNN file to validate.",
            },
            "strict": {
                "type": "boolean",
                "description": "Enable strict validation mode. Defaults to false.",
                "optional": True,
            },
            "estimate_resources": {
                "type": "boolean",
                "description": "Estimate computational resources. Defaults to false.",
                "optional": True,
            },
        },
        "Validate a single GNN file for syntax and type correctness.",
    )

    logger.info("Type checker module MCP tools registered.")
