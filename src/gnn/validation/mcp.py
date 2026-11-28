"""
MCP integration for the validation module.

Exposes GNN validation tools: schema validation, semantic checks,
validation report retrieval, and configuration validation through MCP.
"""

import json
import logging
from pathlib import Path
from typing import Any

from gnn.schemas.section_contract import (
    OPTIONAL_SECTIONS,
    REQUIRED_SECTIONS,
)
from gnn.utils.mcp.dispatch import run_pipeline_step_mcp, run_tool_envelope

from . import process_validation
from .semantic_validator import validate_content

logger = logging.getLogger(__name__)


def process_validation_mcp(
    target_directory: str,
    output_directory: str,
    verbose: bool = False,
    transpose_b: bool = False,
    strict_probabilities: bool = False,
) -> dict[str, Any]:
    """
    Run full GNN validation on files in a directory.

    Performs schema validation, semantic checks, and produces a detailed
    validation report saved to the output directory.

    Args:
        target_directory: Directory containing GNN files to validate
        output_directory: Directory to save validation reports
        verbose: Enable verbose logging
        transpose_b: Opt-in canonical B-tensor transposition — textbook
            (row-stochastic) transition tensors are transposed in memory and
            the per-tensor transposition is recorded in the receipt
            (default: orientation warnings only, source files untouched)
        strict_probabilities: Check authored categorical probability values.

    Returns:
        Dictionary with success status and validation summary.
    """
    return run_pipeline_step_mcp(
        process_validation,
        wrapper_name="process_validation_mcp",
        logger=logger,
        target_directory=target_directory,
        output_directory=output_directory,
        verbose=verbose,
        extra_step_kwargs={
            "transpose_b": transpose_b,
            **({"strict_probabilities": True} if strict_probabilities else {}),
        },
        label="Validation",
    )


def validate_gnn_file_mcp(
    gnn_file_path: str,
    validation_level: str = "standard",
    strict_probabilities: bool = False,
) -> dict[str, Any]:
    """
    Validate a single GNN file at a specified validation level.

    Combines lightweight structural section checks with the module's full
    semantic validator, so callers get both a structural verdict and the
    rule-based semantic result in one response.

    Args:
        gnn_file_path:    Path to the GNN file to validate
        validation_level: Validation depth ('basic', 'standard', 'strict')
        strict_probabilities: Check authored categorical probability values.

    Returns:
        Dictionary with is_valid flag, errors, warnings, structural counts,
        and a ``semantic`` key carrying the full semantic validation result.
    """
    try:
        gnn_path = Path(gnn_file_path)
        if not gnn_path.exists():
            return {"success": False, "error": f"File not found: {gnn_file_path}"}

        content = gnn_path.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()

        # Structural checks
        section_headers = [l for l in lines if l.startswith("## ")]
        required_sections: list[Any] = ["ModelName", "StateSpaceBlock", "Connections"]
        missing = [
            s for s in required_sections if not any(s in h for h in section_headers)
        ]
        warnings: list[str] = []
        errors: list[str] = []

        if missing:
            if validation_level == "basic":
                warnings.extend([f"Missing recommended section: {s}" for s in missing])
            else:
                errors.extend([f"Missing required section: {s}" for s in missing])

        if not content.strip():
            errors.append("File is empty")

        # GNN syntax checks
        connections = [l for l in lines if "->" in l or "<->" in l]
        variables = [
            l for l in lines if "[" in l and "]" in l and not l.strip().startswith("#")
        ]
        if validation_level in ("strict",) and not connections:
            warnings.append("No connections defined in GNN model")

        semantic = validate_content(
            content,
            validation_level=validation_level,
            strict_probabilities=strict_probabilities,
        )
        errors.extend(error for error in semantic["errors"] if error not in errors)
        warnings.extend(
            warning for warning in semantic["warnings"] if warning not in warnings
        )
        is_valid = len(errors) == 0 and semantic["valid"]
        return {
            "success": True,
            "file": str(gnn_path),
            "is_valid": is_valid,
            "validation_level": validation_level,
            "errors": errors,
            "warnings": warnings,
            "sections_found": [h.lstrip("# ").strip() for h in section_headers],
            "variables_count": len(variables),
            "connections_count": len(connections),
            "semantic": semantic,
        }
    except Exception as e:
        logger.error(f"validate_gnn_file_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def get_validation_report_mcp(output_directory: str) -> dict[str, Any]:
    """
    Read and return the saved validation report from a previous validation run.

    Args:
        output_directory: Directory where validation results were saved

    Returns:
        Dictionary with validation report contents.
    """

    def _build() -> dict[str, Any]:
        out_dir = Path(output_directory)
        if not out_dir.exists():
            return {
                "success": False,
                "error": f"Directory not found: {output_directory}",
            }

        reports: list[Any] = []
        for jf in sorted(out_dir.rglob("*validation*.json"))[:10]:
            try:
                reports.append({"file": jf.name, "data": json.loads(jf.read_text())})
            except (json.JSONDecodeError, OSError) as e:
                logger.debug("Skipping unreadable validation report %s: %s", jf.name, e)
        txt_reports: list[Any] = []
        for tf in sorted(out_dir.rglob("*validation*.txt"))[:5]:
            try:
                txt_reports.append({"file": tf.name, "content": tf.read_text()[:2000]})
            except OSError as e:
                logger.debug("Skipping unreadable validation report %s: %s", tf.name, e)

        return {
            "success": True,
            "output_directory": str(out_dir),
            "json_reports": reports,
            "text_reports": txt_reports,
            "reports_found": len(reports) + len(txt_reports),
        }

    return run_tool_envelope(
        _build,
        wrapper_name="get_validation_report_mcp",
        logger=logger,
    )


def check_schema_compliance_mcp(gnn_content: str) -> dict[str, Any]:
    """
    Check a GNN model string against the canonical GNN schema requirements.

    Performs a lightweight structural compliance check without requiring file I/O.

    Args:
        gnn_content: GNN model content as a string

    Returns:
        Dictionary with compliance status, missing sections, and issue count.
    """
    try:
        lines = gnn_content.splitlines()
        section_headers = {l.lstrip("# ").strip() for l in lines if l.startswith("## ")}
        required: set[str] = set(REQUIRED_SECTIONS)
        optional: set[str] = set(OPTIONAL_SECTIONS)
        missing = required - section_headers
        extra = section_headers - required - optional

        return {
            "success": True,
            "is_compliant": len(missing) == 0,
            "sections_found": sorted(section_headers),
            "missing_required": sorted(missing),
            "unrecognised_sections": sorted(extra),
            "total_lines": len(lines),
        }
    except Exception as e:
        logger.error(f"check_schema_compliance_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


# ── MCP Registration ────────────────────────────────────────────────────────


def register_tools(mcp_instance: Any) -> None:
    """Register validation tools with the MCP server."""

    mcp_instance.register_tool(
        "process_validation",
        process_validation_mcp,
        {
            "type": "object",
            "properties": {
                "target_directory": {
                    "type": "string",
                    "description": "Directory with GNN files",
                },
                "output_directory": {
                    "type": "string",
                    "description": "Directory to save validation reports",
                },
                "verbose": {"type": "boolean", "default": False},
                "strict_probabilities": {
                    "type": "boolean",
                    "default": False,
                    "description": "Check authored categorical probability values",
                },
            },
            "required": ["target_directory", "output_directory"],
        },
        "Run full GNN validation pipeline on a directory of GNN files.",
        module=__package__,
        category="validation",
    )

    mcp_instance.register_tool(
        "validate_gnn_file",
        validate_gnn_file_mcp,
        {
            "type": "object",
            "properties": {
                "gnn_file_path": {"type": "string", "description": "Path to GNN file"},
                "validation_level": {
                    "type": "string",
                    "enum": ["basic", "standard", "strict"],
                    "default": "standard",
                },
                "strict_probabilities": {
                    "type": "boolean",
                    "default": False,
                    "description": "Check authored categorical probability values",
                },
            },
            "required": ["gnn_file_path"],
        },
        "Validate a single GNN file at a given level (basic/standard/strict).",
        module=__package__,
        category="validation",
    )

    mcp_instance.register_tool(
        "get_validation_report",
        get_validation_report_mcp,
        {
            "type": "object",
            "properties": {
                "output_directory": {
                    "type": "string",
                    "description": "Directory with saved validation results",
                },
            },
            "required": ["output_directory"],
        },
        "Read and return saved validation reports from a previous validation run.",
        module=__package__,
        category="validation",
    )

    mcp_instance.register_tool(
        "check_schema_compliance",
        check_schema_compliance_mcp,
        {
            "type": "object",
            "properties": {
                "gnn_content": {
                    "type": "string",
                    "description": "GNN model content as a string",
                },
            },
            "required": ["gnn_content"],
        },
        "Check a GNN model string against canonical GNN schema requirements.",
        module=__package__,
        category="validation",
    )

    logger.info("validation module MCP tools registered (4 tools).")
