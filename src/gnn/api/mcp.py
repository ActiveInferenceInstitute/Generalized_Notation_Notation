#!/usr/bin/env python3
"""
MCP API module for GNN pipeline job management.

This module integrates the GNN pipeline API capabilities with the MCP
(Multi-Agent Communication Protocol) framework, allowing AI assistants
to interact with the GNN pipeline for submitting, monitoring, and
managing processing jobs.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from .path_utils import PathValidationError
from .processor import (
    PIPELINE_STEPS,
    RunAmbiguousError,
    RunNotFoundError,
    cancel_job,
    create_job,
    delete_run,
    get_job,
    list_jobs,
)

logger = logging.getLogger(__name__)

# Basic module metadata (version re-exported from the canonical package constant)
from gnn import __version__ as __version__
from gnn.api import DEFAULT_API_HOST, DEFAULT_API_PORT

__description__ = "API module MCP integration for GNN pipeline job management."
__dependencies__: list[str] = []


def gnn_submit_job_mcp(
    target_dir: str,
    steps: List[int] | None = None,
    skip_steps: List[int] | None = None,
    verbose: bool = False,
    strict: bool = False,
    parallel: bool = False,
    consolidated_steps: bool = False,
    output_dir: str | None = None,
) -> Dict[str, Any]:
    """Create a pending GNN pipeline job record via MCP.

    Contract: this tool creates the job record and returns immediately; it
    does NOT start pipeline execution, and no built-in watcher executes
    externally created records. Execution happens only where the caller
    also starts it: POST /api/v1/process or POST /api/v1/tools/{step} on
    the job/tool API server (``gnn.api.server``), which create and execute
    their own jobs, or a direct call to
    ``gnn.api.processor.execute_job_async(job_id)``. ``gnn serve`` serves
    the Runs surface (``gnn.api.app``) and never consumes these records.
    """
    try:
        job_id = create_job(
            target_dir=target_dir,
            steps=steps,
            skip_steps=skip_steps,
            verbose=verbose,
            strict=strict,
            parallel=parallel,
            consolidated_steps=consolidated_steps,
            output_dir=output_dir,
        )
        return {
            "status": "success",
            "job_id": job_id,
            "message": (
                "Job created but not started: execution runs only where a "
                "caller also starts it (POST /api/v1/process on the API "
                "server, gnn.api.server, or a direct execute_job_async call)."
            ),
        }
    except (PathValidationError, ValueError) as e:
        logger.warning("Rejected MCP job submission: %s", e)
        return {"status": "error", "message": str(e)}
    except Exception as e:
        logger.error("Failed to submit job via MCP: %s", e, exc_info=True)
        return {"status": "error", "message": str(e)}


def gnn_get_job_status_mcp(job_id: str) -> Dict[str, Any]:
    """Retrieve the status of a GNN pipeline job via MCP."""
    try:
        job = get_job(job_id)
        if job:
            return {"status": "success", "job": job}
        return {"status": "error", "message": f"Job {job_id} not found."}
    except Exception as e:
        logger.error(f"Failed to get job status via MCP: {e}")
        return {"status": "error", "message": str(e)}


def gnn_cancel_job_mcp(job_id: str) -> Dict[str, Any]:
    """Cancel a GNN pipeline job via MCP."""
    try:
        success = cancel_job(job_id)
        if success:
            return {
                "status": "success",
                "message": f"Job {job_id} cancelled successfully.",
            }
        return {
            "status": "error",
            "message": f"Failed to cancel job {job_id}. It may not exist or is already terminal.",
        }
    except Exception as e:
        logger.error(f"Failed to cancel job via MCP: {e}")
        return {"status": "error", "message": str(e)}


def gnn_delete_run_mcp(run_hash: str) -> Dict[str, Any]:
    """Delete a GNN pipeline run via MCP, cancelling it first when active."""
    try:
        result = delete_run(run_hash)
    except RunNotFoundError as e:
        logger.info(f"Run {run_hash} not found via MCP delete: {e}")
        return {"status": "error", "message": f"Run {run_hash} not found."}
    except RunAmbiguousError as e:
        return {"status": "error", "message": f"Ambiguous run hash prefix: {e}"}
    except Exception as e:
        logger.error(f"Failed to delete run via MCP: {e}")
        return {"status": "error", "message": str(e)}
    return {
        "status": "success",
        "message": f"Run {result['deleted']} deleted successfully.",
        **result,
    }


def gnn_list_jobs_mcp(limit: int = 50) -> Dict[str, Any]:
    """List recent GNN pipeline jobs via MCP."""
    try:
        jobs = list_jobs(limit=limit)
        return {"status": "success", "jobs": jobs, "total": len(jobs)}
    except Exception as e:
        logger.error(f"Failed to list jobs via MCP: {e}")
        return {"status": "error", "message": str(e)}


def gnn_get_pipeline_tools_mcp() -> Dict[str, Any]:
    """List available pipeline steps via MCP."""
    try:
        tools = [
            {"step_number": step, "name": name, "description": desc}
            for step, (name, desc) in PIPELINE_STEPS.items()
        ]
        return {"status": "success", "tools": tools, "total": len(tools)}
    except Exception as e:
        logger.error(f"Failed to list pipeline tools via MCP: {e}")
        return {"status": "error", "message": str(e)}


_MCP_TOOL_DEFINITIONS: tuple[Dict[str, Any], ...] = (
    {
        "name": "gnn_submit_job",
        "handler": gnn_submit_job_mcp,
        "input_schema": {
            "type": "object",
            "properties": {
                "target_dir": {
                    "type": "string",
                    "description": "Target directory containing GNN files",
                },
                "output_dir": {
                    "type": "string",
                    "description": "Workspace-local output directory (defaults to output)",
                },
                "steps": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 1,
                    "uniqueItems": True,
                    "description": "Specific steps to run (optional)",
                },
                "skip_steps": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Steps to skip (optional)",
                },
                "verbose": {"type": "boolean", "default": False},
                "strict": {"type": "boolean", "default": False},
                "parallel": {"type": "boolean", "default": False},
                "consolidated_steps": {"type": "boolean", "default": False},
            },
            "required": ["target_dir"],
            "additionalProperties": False,
        },
        "description": (
            "Create a GNN pipeline job record (pending; not executed). "
            "Execution happens only via POST /api/v1/process or "
            "POST /api/v1/tools/{step} on the job/tool API server "
            "(gnn.api.server), or a direct execute_job_async call; "
            "gnn serve (Runs surface) never consumes these records."
        ),
    },
    {
        "name": "gnn_get_job_status",
        "handler": gnn_get_job_status_mcp,
        "input_schema": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "The ID of the job to query",
                }
            },
            "required": ["job_id"],
        },
        "description": "Retrieve the status of a GNN pipeline job.",
    },
    {
        "name": "gnn_cancel_job",
        "handler": gnn_cancel_job_mcp,
        "input_schema": {
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "The ID of the job to cancel",
                }
            },
            "required": ["job_id"],
        },
        "description": "Cancel a GNN pipeline job.",
    },
    {
        "name": "gnn_delete_run",
        "handler": gnn_delete_run_mcp,
        "input_schema": {
            "type": "object",
            "properties": {
                "run_hash": {
                    "type": "string",
                    "description": "The run hash (or unique prefix) to delete",
                }
            },
            "required": ["run_hash"],
        },
        "description": "Delete a pipeline run, cancelling it first when active.",
    },
    {
        "name": "gnn_list_jobs",
        "handler": gnn_list_jobs_mcp,
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 50,
                    "description": "Maximum number of jobs to return",
                }
            },
        },
        "description": "List recent GNN pipeline jobs.",
    },
    {
        "name": "gnn_get_pipeline_tools",
        "handler": gnn_get_pipeline_tools_mcp,
        "input_schema": {"type": "object", "properties": {}},
        "description": "List available pipeline steps.",
    },
)


def register_tools(mcp_instance: Any) -> None:
    """Register the same API tools advertised by the serialized manifest."""
    for tool in _MCP_TOOL_DEFINITIONS:
        mcp_instance.register_tool(
            tool["name"],
            tool["handler"],
            tool["input_schema"],
            tool["description"],
            module=__package__,
            category="api",
        )

    logger.info("api module MCP tools registered (%d tools).", len(MCP_TOOLS))


MCP_TOOLS: list[Dict[str, Any]] = [
    {
        "name": tool["name"],
        "description": tool["description"],
        "inputSchema": tool["input_schema"],
    }
    for tool in _MCP_TOOL_DEFINITIONS
]


def register_mcp_tools() -> Dict[str, Any]:
    """Return MCP tool registration manifest."""
    return {
        "module": "api",
        "tools": MCP_TOOLS,
        "endpoint": f"http://{DEFAULT_API_HOST}:{DEFAULT_API_PORT}/api/v1",
        "version": "1.0.0",
    }


def save_mcp_manifest(output_dir: Path) -> bool:
    """Save MCP tool manifest to output directory."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = register_mcp_tools()
        manifest_path = output_dir / "api_mcp_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"MCP manifest saved to {manifest_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save MCP manifest: {e}")
        return False
