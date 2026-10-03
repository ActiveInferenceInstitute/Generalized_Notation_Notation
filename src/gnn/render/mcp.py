"""
MCP (Model Context Protocol) integration for the render module.

Exposes GNN rendering tools: format listing, single-file rendering,
per-framework status, and batch processing through MCP.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

from gnn.utils.mcp.dispatch import run_pipeline_step_mcp

from . import process_render
from .framework_registry import get_available_renderers, get_supported_frameworks


def process_render_mcp(
    target_directory: str, output_directory: str, verbose: bool = False
) -> Dict[str, Any]:
    """
    Render GNN models in a directory to all supported code formats.

    Args:
        target_directory: Directory containing GNN files to render
        output_directory: Directory to write rendered outputs
        verbose: Enable verbose logging

    Returns:
        Dictionary with success status and render summary.
    """

    def _interpret(raw: Any) -> tuple[bool, Dict[str, Any], str | None]:
        # Phase 1.1 contract: process_render may return bool OR int (0/1/2).
        # Coerce to MCP bool envelope; surface "skipped" separately.
        if isinstance(raw, bool):
            success, skipped = raw, False
        else:  # int
            success, skipped = raw in (0, 2), raw == 2
        message = (
            "Render skipped (no GNN files found)"
            if skipped
            else f"Render {'completed successfully' if success else 'completed with issues'}"
        )
        return success, {"skipped": skipped}, message

    return run_pipeline_step_mcp(
        process_render,
        wrapper_name="process_render_mcp",
        logger=logger,
        target_directory=target_directory,
        output_directory=output_directory,
        verbose=verbose,
        interpret_result=_interpret,
    )


# Canonical framework descriptions used by both branches of
# ``list_render_frameworks_mcp`` so success/fallback share one shape.
_FRAMEWORK_DESCRIPTIONS: Dict[str, str] = {
    name: spec["description"] for name, spec in get_available_renderers().items()
}


def list_render_frameworks_mcp() -> Dict[str, Any]:
    """
    Return all supported rendering frameworks and their availability status.

    Shape:
        ``{"success": bool, "frameworks": {name: {"available": bool, "description": str}, ...}}``

    The fallback branch preserves this shape and sets every framework to
    ``available: False`` so consumers never need to branch on the result type.
    """
    frameworks: Dict[str, Dict[str, Any]] = {
        name: {"available": False, "description": desc}
        for name, desc in _FRAMEWORK_DESCRIPTIONS.items()
    }
    try:
        for name in get_supported_frameworks():
            entry = frameworks.setdefault(
                name, {"available": False, "description": name}
            )
            entry["available"] = True
        return {"success": True, "frameworks": frameworks}
    except Exception as e:
        return {
            "success": False,
            "frameworks": frameworks,
            "error": str(e),
        }


def render_gnn_to_format_mcp(
    gnn_file_path: str,
    output_directory: str,
    framework: str = "pymdp",
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Render a single GNN file to a specific target framework.

    Args:
        gnn_file_path:    Path to the GNN source file (.md)
        output_directory: Directory to write the rendered output
        framework:        Target framework name (best-effort hint; see note below)
        verbose:          Enable verbose logging

    Returns:
        Dictionary with success status, output file path, and any errors.
    """
    try:
        gnn_path = Path(gnn_file_path)
        out_dir = Path(output_directory)

        if not gnn_path.exists():
            return {"success": False, "error": f"GNN file not found: {gnn_file_path}"}

        # Use process_render on a single-file temp dir.
        #
        # NOTE: At present, this MCP tool does not filter to a single framework.
        # It runs Step 11 rendering and returns any artifacts written under the
        # requested output directory. The `framework` parameter is returned for
        # caller context and may be used for filtering in a future revision.
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_in = Path(tmp) / "input"
            tmp_in.mkdir()
            shutil.copy2(gnn_path, tmp_in / gnn_path.name)
            success = process_render(
                target_dir=tmp_in, output_dir=out_dir, verbose=verbose
            )

        output_files = list(out_dir.rglob(f"*{gnn_path.stem}*"))
        return {
            "success": success,
            "gnn_file": str(gnn_path),
            "framework": framework,
            "output_directory": str(out_dir),
            "output_files": [str(f) for f in output_files],
            "message": f"Rendering to {framework} {'succeeded' if success else 'failed'}",
        }
    except Exception as e:
        logger.error(f"render_gnn_to_format_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def render_spec_to_format_mcp(
    gnn_file_path: str,
    output_directory: str,
    framework: str = "pymdp",
    output_filename: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Render one GNN file to one framework through the canonical dispatch.

    Unlike :func:`render_gnn_to_format_mcp` (which runs the full Step 11
    directory flow and treats ``framework`` as a hint), this tool targets the
    exact framework via ``render.processor.render_gnn_spec``, so artifact
    lists contain only that framework's outputs.

    Args:
        gnn_file_path:    Path to the GNN source file (.md).
        output_directory: Directory to write the rendered output.
        framework:        Target framework accepted by ``render_gnn_spec``
                          (e.g. ``pymdp``, ``rxinfer``, ``jax``, ``stan``).
        output_filename:  Optional base filename for the output artifact
                          (without extension); defaults to the model name.

    Returns:
        Dictionary with success status, the render message, and artifact paths.
    """
    try:
        from gnn import parse_gnn_file

        from .processor import render_gnn_spec

        gnn_path = Path(gnn_file_path)
        if not gnn_path.exists():
            return {"success": False, "error": f"GNN file not found: {gnn_file_path}"}

        out_dir = Path(output_directory)
        parsed = parse_gnn_file(gnn_path)
        options = {
            **(options or {}),
            **({"output_filename": output_filename} if output_filename else {}),
        }
        success, message, artifacts = render_gnn_spec(
            parsed, framework, out_dir, options
        )
        return {
            "success": success,
            "gnn_file": str(gnn_path),
            "framework": framework,
            "output_directory": str(out_dir),
            "output_files": artifacts,
            "message": message,
        }
    except Exception as e:
        logger.error(f"render_spec_to_format_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def get_render_module_info_mcp() -> Dict[str, Any]:
    """
    Return metadata about the render module capabilities.

    Shape:
        ``{"success": bool, "module": str, "frameworks": [str, ...],
        "input_formats": [str, ...], "output_formats": [str, ...],
        "error": str}`` — ``error`` is present only on the failure branch,
        which keeps the same ``frameworks`` key (empty list) and reports
        ``success: False`` so a client can tell "registry broken" from
        "no frameworks".

    Returns:
        Dictionary with version, supported frameworks, and supported input formats.
    """
    try:
        from . import get_supported_frameworks

        frameworks: list[str] = get_supported_frameworks()
    except Exception as e:
        logger.error(
            f"get_render_module_info_mcp: framework registry failure: {e}",
            exc_info=True,
        )
        return {
            "success": False,
            "module": __package__,
            "frameworks": [],
            "error": f"framework registry failure: {e}",
            "input_formats": ["markdown", "gnn"],
            "output_formats": ["python", "julia", "julia_rxinfer"],
        }
    return {
        "success": True,
        "module": __package__,
        "frameworks": frameworks,
        "input_formats": ["markdown", "gnn"],
        "output_formats": ["python", "julia", "julia_rxinfer"],
    }


# ── MCP Registration ────────────────────────────────────────────────────────


def register_tools(mcp_instance: Any) -> None:
    """Register render tools with the MCP server."""

    mcp_instance.register_tool(
        "process_render",
        process_render_mcp,
        {
            "type": "object",
            "properties": {
                "target_directory": {
                    "type": "string",
                    "description": "Directory containing GNN files",
                },
                "output_directory": {
                    "type": "string",
                    "description": "Directory to write rendered outputs",
                },
                "verbose": {
                    "type": "boolean",
                    "description": "Enable verbose logging",
                    "default": False,
                },
            },
            "required": ["target_directory", "output_directory"],
        },
        "Render GNN models in a directory to all supported code frameworks.",
        module=__package__,
        category="render",
    )

    mcp_instance.register_tool(
        "render_spec_to_format",
        render_spec_to_format_mcp,
        {
            "type": "object",
            "properties": {
                "gnn_file_path": {
                    "type": "string",
                    "description": "Path to the GNN source file (.md)",
                },
                "output_directory": {
                    "type": "string",
                    "description": "Directory for rendered output",
                },
                "framework": {
                    "type": "string",
                    "description": "Target framework for render_gnn_spec dispatch",
                    "enum": get_supported_frameworks(),
                    "default": "pymdp",
                },
                "output_filename": {
                    "type": "string",
                    "description": "Optional base filename for the output artifact (without extension)",
                },
                "options": {
                    "type": "object",
                    "description": "Framework-specific validated render options",
                },
            },
            "required": ["gnn_file_path", "output_directory"],
        },
        "Render a single GNN file to exactly one framework via render_gnn_spec.",
        module=__package__,
        category="render",
    )

    mcp_instance.register_tool(
        "list_render_frameworks",
        list_render_frameworks_mcp,
        {},
        "Return supported render framework names and availability (best effort).",
        module=__package__,
        category="render",
    )

    mcp_instance.register_tool(
        "render_gnn_to_format",
        render_gnn_to_format_mcp,
        {
            "type": "object",
            "properties": {
                "gnn_file_path": {
                    "type": "string",
                    "description": "Path to the GNN source file (.md)",
                },
                "output_directory": {
                    "type": "string",
                    "description": "Directory for rendered output",
                },
                "framework": {
                    "type": "string",
                    "description": "Target framework",
                    "enum": get_supported_frameworks(),
                    "default": "pymdp",
                },
                "verbose": {
                    "type": "boolean",
                    "description": "Enable verbose logging",
                    "default": False,
                },
            },
            "required": ["gnn_file_path", "output_directory"],
        },
        "Render a single GNN file (runs Step 11 render; does not currently filter to one framework).",
        module=__package__,
        category="render",
    )

    mcp_instance.register_tool(
        "get_render_module_info",
        get_render_module_info_mcp,
        {},
        "Return metadata about the render module: supported frameworks and input/output formats.",
        module=__package__,
        category="render",
    )

    logger.info("render module MCP tools registered (4 tools).")
