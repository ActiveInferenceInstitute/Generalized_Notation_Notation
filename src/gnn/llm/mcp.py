"""
MCP integration for the llm module.

Exposes GNN LLM tools: pipeline driver, per-file LLM analysis,
documentation generation, provider listing, and module introspection
through MCP.
"""

from __future__ import annotations

import inspect
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)
from gnn.api.path_utils import PathValidationError, resolve_repo_path
from gnn.utils.mcp.dispatch import run_pipeline_step_mcp, run_tool_envelope

from . import (
    analyze_gnn_file_with_llm,
    generate_documentation,
    process_llm,
)
from .defaults import DEFAULT_OLLAMA_MODEL
from .sync_bridge import run_async

_GNN_MODEL_SUFFIXES = {".md", ".json", ".yaml", ".yml"}
_ANALYSIS_TYPES = ("comprehensive", "summary", "complexity", "connections")


def _resolve_gnn_model_path(path_value: str, *, purpose: str) -> Path:
    path = resolve_repo_path(
        path_value,
        purpose=purpose,
        must_exist=True,
        must_be_dir=False,
    )
    if not path.is_file():
        raise PathValidationError(f"{purpose} must be a file: {path_value}")
    if path.suffix.lower() not in _GNN_MODEL_SUFFIXES:
        allowed = ", ".join(sorted(_GNN_MODEL_SUFFIXES))
        raise PathValidationError(
            f"{purpose} must be a GNN source file ({allowed}): {path_value}"
        )
    return path


def _resolve_input_directory(path_value: str, *, purpose: str) -> Path:
    return resolve_repo_path(
        path_value,
        purpose=purpose,
        must_exist=True,
        must_be_dir=True,
    )


def _resolve_output_directory(path_value: str, *, purpose: str) -> Path:
    return resolve_repo_path(
        path_value,
        purpose=purpose,
        must_be_dir=True,
        create=True,
    )


def _resolve_output_file(path_value: str, *, purpose: str) -> Path:
    path = resolve_repo_path(
        path_value,
        purpose=purpose,
        must_be_dir=False,
    )
    if path.exists() and path.is_dir():
        raise PathValidationError(f"{purpose} must be a file path: {path_value}")
    return path


# ── Domain tools ─────────────────────────────────────────────────────────────


def process_llm_mcp(
    target_directory: str,
    output_directory: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run LLM analysis pipeline for all GNN files in a directory.

    Discovers all `.md` GNN files, generates LLM-based summaries and
    documentation for each, and saves structured JSON results.

    Args:
        target_directory: Directory containing GNN model files.
        output_directory: Directory to save LLM analysis outputs.
        verbose: Enable verbose logging.

    Returns:
        Dictionary with success flag and processing summary.
    """

    def _resolve(target: str, output: str) -> tuple[Path, Path]:
        return (
            _resolve_input_directory(target, purpose="LLM target directory"),
            _resolve_output_directory(output, purpose="LLM output directory"),
        )

    return run_pipeline_step_mcp(
        process_llm,
        wrapper_name="process_llm_mcp",
        logger=logger,
        target_directory=target_directory,
        output_directory=output_directory,
        verbose=verbose,
        resolve_paths=_resolve,
        echo_resolved=True,
        label="LLM processing",
        success_wording="completed",
        failure_wording="failed",
    )


def analyze_gnn_with_llm_mcp(
    gnn_file_path: str,
    analysis_type: str = "comprehensive",
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run LLM-based analysis on a single GNN model file.

    Sends the GNN content to an LLM provider (configured via env vars or
    `provider` override) and returns structured analysis: summary, insights,
    complexity assessment, and suggested improvements.

    Args:
        gnn_file_path: Path to the GNN `.md` model file.
        analysis_type: One of "comprehensive", "summary", "complexity", "connections".
        provider: Optional override; "ollama" pins the configured local
            tag for the summary call. None uses default routing;
            other explicit values are rejected.

    Returns:
        Dictionary with success flag, analysis type, and LLM output.
    """
    try:
        if provider not in (None, "ollama"):
            return {
                "success": False,
                "error": "This tool supports explicit Ollama or default routing; use LLMProcessor for other explicit providers",
            }
        if analysis_type not in _ANALYSIS_TYPES:
            return {
                "success": False,
                "error": (
                    f"Unknown analysis_type '{analysis_type}'; expected one of "
                    f"{', '.join(_ANALYSIS_TYPES)}"
                ),
            }
        gnn_path = _resolve_gnn_model_path(
            gnn_file_path,
            purpose="GNN model file",
        )
        # Honor the provider override where the analyzer supports it: "ollama"
        # pins the pipeline's default local tag for the summary call; other
        # names use the standard multi-provider routing.
        ollama_model = (
            (os.getenv("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL)
            if provider == "ollama"
            else None
        )
        result = analyze_gnn_file_with_llm(gnn_path, ollama_model=ollama_model)
        if inspect.iscoroutine(result):
            result = run_async(result)
        if isinstance(result, dict):
            return {
                "success": result.get("llm_summary_status") != "unavailable",
                "analysis_type": analysis_type,
                "provider": provider,
                **result,
            }
        return {
            "success": False,
            "analysis_type": analysis_type,
            "provider": provider,
            "error": "LLM analyzer did not return an analysis dictionary",
        }
    except Exception as e:
        logger.error(f"analyze_gnn_with_llm_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def _format_structural_documentation(result: Dict[str, Any], format: str) -> str:
    """Render every structural section as the requested human-readable format."""
    overview = str(result.get("model_overview", "")).strip()
    if format == "text":
        import re

        overview = re.sub(r"(?m)^#{1,6}\s+", "", overview).replace("**", "")
    sections = [overview]
    for title, key in (
        ("Variables", "variable_documentation"),
        ("Connections", "connection_documentation"),
        ("Usage Examples", "usage_examples"),
    ):
        sections.append(f"### {title}" if format == "markdown" else title)
        for item in result.get(key, []):
            label = item.get("name") or item.get("title") or item.get("connection")
            sections.append(f"- {label}: {item.get('description', '')}")
            if item.get("definition"):
                sections.append(str(item["definition"]))
            if item.get("code"):
                sections.append(
                    f"```text\n{item['code']}\n```"
                    if format == "markdown"
                    else str(item["code"])
                )
    return "\n\n".join(sections).strip() + "\n"


def generate_llm_documentation_mcp(
    gnn_file_path: str,
    output_path: Optional[str] = None,
    format: str = "markdown",
) -> Dict[str, Any]:
    """
    Generate deterministic structural documentation for a GNN model.

    Produces a model overview, extracted variable and connection descriptions,
    and generated usage examples, formatted as Markdown or plain text.
    This operation does not call an LLM provider.

    Args:
        gnn_file_path: Path to the GNN `.md` model file.
        output_path: Optional path to write the generated documentation.
        format: Output format — "markdown" (default) or "text".

    Returns:
        Dictionary with success flag and generated documentation string.
    """
    try:
        if format not in ("markdown", "text"):
            raise ValueError("format must be 'markdown' or 'text'")
        # Read the GNN file and build a lightweight analysis dict
        # that matches the signature of generate_documentation(file_analysis).
        gnn_path = _resolve_gnn_model_path(
            gnn_file_path,
            purpose="GNN model file",
        )
        resolved_output_path = (
            _resolve_output_file(
                output_path,
                purpose="LLM documentation output path",
            )
            if output_path
            else None
        )
        content = (
            gnn_path.read_text(encoding="utf-8", errors="replace")
            if gnn_path.exists()
            else ""
        )

        # Build a minimal file_analysis structure expected by generate_documentation
        file_analysis: Dict[str, Any] = {
            "file_path": str(gnn_path),
            "file_name": gnn_path.name,
            "variables": [],
            "connections": [],
            "complexity_metrics": {},
            "patterns": {},
        }

        # If the analyzer module is available, extract richer data
        try:
            from .analyzer import (
                calculate_complexity_metrics,
                extract_connections,
                extract_variables,
                identify_patterns,
            )

            file_analysis["variables"] = extract_variables(content)
            file_analysis["connections"] = extract_connections(content)
            file_analysis["patterns"] = identify_patterns(
                content, file_analysis["variables"], file_analysis["connections"]
            )
            file_analysis["complexity_metrics"] = calculate_complexity_metrics(
                file_analysis["variables"], file_analysis["connections"]
            )
        except Exception as exc:
            logger.warning(
                "LLM analyzer not available for documentation generation; "
                "using lightweight analysis: %s",
                exc,
            )

        result: object = generate_documentation(file_analysis)

        if isinstance(result, dict):
            # Optionally write to disk if output_path was provided
            if resolved_output_path:
                out = resolved_output_path
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(
                    _format_structural_documentation(result, format), encoding="utf-8"
                )
            return {"success": True, "analysis_source": "structural", **result}
        return {"success": True, "documentation": str(result)}
    except Exception as e:
        logger.error(f"generate_llm_documentation_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def get_llm_providers_mcp() -> Dict[str, Any]:
    """
    Return available LLM providers and their configuration status.

    Checks environment variables (OPENAI_API_KEY, ANTHROPIC_API_KEY,
    OLLAMA_BASE_URL, etc.) and reports which providers are configured.

    Returns:
        Dictionary with provider names and availability/configuration status.
    """
    try:
        import importlib.util
        import os

        providers: Dict[str, Dict[str, Any]] = {}

        # OpenAI
        providers["openai"] = {
            "available": importlib.util.find_spec("openai") is not None,
            "configured": bool(os.getenv("OPENAI_API_KEY")),
        }
        # Anthropic
        providers["anthropic"] = {
            "available": False,
            "implemented": False,
            "configured": bool(os.getenv("ANTHROPIC_API_KEY")),
        }
        # OpenRouter and Perplexity ride the openai client library, so their
        # "available" flag tracks that dependency rather than a distinct lib.
        providers["openrouter"] = {
            "available": importlib.util.find_spec("openai") is not None,
            "configured": bool(os.getenv("OPENROUTER_API_KEY")),
        }
        providers["perplexity"] = {
            "available": importlib.util.find_spec("openai") is not None,
            "configured": bool(os.getenv("PERPLEXITY_API_KEY")),
        }
        # Ollama (local)
        from .provider_identity import resolved_endpoint
        from .providers.base_provider import ProviderType

        ollama_endpoint, _ = resolved_endpoint(ProviderType.OLLAMA)
        providers["ollama"] = {
            "available": True,  # Maintained HTTP transport uses the standard library.
            "configured": os.getenv("OLLAMA_DISABLED", "").lower() not in {"1", "true"},
            "base_url": ollama_endpoint,
            "readiness": "not_probed",  # Configuration is not live inference evidence.
        }
        # Google Gemini
        providers["google"] = {
            "available": False,
            "implemented": False,
            "configured": bool(os.getenv("GOOGLE_API_KEY")),
        }

        configured = [p for p, info in providers.items() if info["configured"]]
        return {
            "success": True,
            "providers": providers,
            "configured": configured,
            "count": len(providers),
        }
    except Exception as e:
        logger.error(f"get_llm_providers_mcp error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def get_llm_module_info_mcp() -> Dict[str, Any]:
    """
    Return version, feature flags, and API surface of the LLM module.

    Includes: module version, supported analysis types, available providers,
    and list of exported public tools.

    Returns:
        Dictionary with module metadata and feature inventory.
    """

    def _build() -> Dict[str, Any]:
        import importlib

        mod = importlib.import_module(__package__)
        return {
            "success": True,
            "module": __package__,
            "version": getattr(mod, "__version__", "unknown"),
            "analysis_types": ["comprehensive", "summary", "complexity", "connections"],
            "output_formats": ["markdown", "text"],
            "tools": [
                "process_llm",
                "analyze_gnn_with_llm",
                "generate_llm_documentation",
                "get_llm_providers",
                "get_llm_module_info",
            ],
        }

    return run_tool_envelope(
        _build,
        wrapper_name="get_llm_module_info_mcp",
        logger=logger,
    )


# ── MCP Registration ──────────────────────────────────────────────────────────


def initialize_llm_module(mcp_instance: Any) -> None:
    """
    Initialize the LLM module prior to tool registration.
    Loads API keys and configures the default LLM processor.
    """
    try:
        import asyncio

        from gnn.llm import create_processor_from_env

        logger.info("Initializing LLM module prior to MCP registration...")

        # create_processor_from_env is a coroutine; must be awaited
        processor: Any
        try:
            loop = asyncio.get_running_loop()
            # If there's an active loop, wrap in a task
            processor = loop.create_task(create_processor_from_env())
        except RuntimeError:
            processor = asyncio.run(create_processor_from_env())

        if processor:
            logger.info("LLM processor initialized successfully.")
        else:
            logger.warning("LLM processor initialized but returned False/None.")
    except Exception as e:
        logger.error(f"Failed to initialize LLM module: {e}")


def register_tools(mcp_instance: Any) -> None:
    """Register LLM domain tools with the MCP server."""

    # Single source of truth for tool registration: (name, handler, schema,
    # description). All llm tools share module + category wiring.
    tool_specs: tuple[tuple[str, Any, dict[str, Any], str], ...] = (
        (
            "process_llm",
            process_llm_mcp,
            {
                "type": "object",
                "properties": {
                    "target_directory": {
                        "type": "string",
                        "description": "Directory with GNN files",
                    },
                    "output_directory": {
                        "type": "string",
                        "description": "LLM output directory",
                    },
                    "verbose": {"type": "boolean", "default": False},
                },
                "required": ["target_directory", "output_directory"],
            },
            "Run LLM analysis pipeline for all GNN files in a directory.",
        ),
        (
            "analyze_gnn_with_llm",
            analyze_gnn_with_llm_mcp,
            {
                "type": "object",
                "properties": {
                    "gnn_file_path": {
                        "type": "string",
                        "description": "Path to the GNN model file",
                    },
                    "analysis_type": {
                        "type": "string",
                        "enum": list(_ANALYSIS_TYPES),
                        "default": "comprehensive",
                    },
                    "provider": {
                        "type": "string",
                        "enum": ["ollama"],
                        "description": "Explicit local Ollama, or null for default routing",
                        "nullable": True,
                    },
                },
                "required": ["gnn_file_path"],
            },
            "Run LLM-based analysis on a single GNN model file: summary, complexity, connections.",
        ),
        (
            "generate_llm_documentation",
            generate_llm_documentation_mcp,
            {
                "type": "object",
                "properties": {
                    "gnn_file_path": {
                        "type": "string",
                        "description": "Path to the GNN model file",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Optional path to write documentation",
                        "nullable": True,
                    },
                    "format": {
                        "type": "string",
                        "enum": ["markdown", "text"],
                        "default": "markdown",
                    },
                },
                "required": ["gnn_file_path"],
            },
            "Generate Markdown or text documentation from deterministic GNN structural analysis.",
        ),
        (
            "get_llm_providers",
            get_llm_providers_mcp,
            {},
            "Return available LLM providers and their API key / configuration status.",
        ),
        (
            "get_llm_module_info",
            get_llm_module_info_mcp,
            {},
            "Return version, analysis types, output formats, and tool list of the GNN LLM module.",
        ),
    )

    for name, handler, schema, description in tool_specs:
        mcp_instance.register_tool(
            name,
            handler,
            schema,
            description,
            module=__package__,
            category="llm",
        )

    logger.info("llm module MCP tools registered (5 real domain tools).")
