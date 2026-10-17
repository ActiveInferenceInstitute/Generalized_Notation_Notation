#!/usr/bin/env python3
"""
LLM analyzer module for GNN file analysis.
"""

import asyncio
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Coroutine, Dict, List, Optional, Union

from gnn.analysis.analyzer import extract_sections

from .llm_operations import LLMOperations
from .providers.openai_provider import OpenAIProvider  # for patching in tests

logger = logging.getLogger(__name__)


async def _analyze_gnn_file_with_llm(
    file_path: Path,
    verbose: bool = False,
    ollama_model: Optional[str] = None,
    attempt_llm: bool = True,
) -> Dict[str, Any]:
    """
    Analyze a GNN file using LLM-enhanced techniques.

    Args:
        file_path: Path to the GNN file
        verbose: Enable verbose output
        ollama_model: When set (e.g. pipeline-selected tag), pass to summarization on Ollama.
        attempt_llm: Whether a live provider call should be attempted. Set false
            when upstream provider initialization has already failed.

    Returns:
        Dictionary containing analysis results
    """
    try:
        with open(file_path, "r") as f:
            content = f.read()

        # Extract GNN structure
        variables = extract_variables(content)
        connections = extract_connections(content)
        sections = extract_sections(content)

        # Perform semantic analysis
        semantic_analysis = perform_semantic_analysis(content, variables, connections)

        # Generate model complexity metrics
        complexity_metrics = calculate_complexity_metrics(variables, connections)

        # Identify patterns and anti-patterns
        patterns = identify_patterns(content, variables, connections)

        result: Dict[str, Any] = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_size": file_path.stat().st_size,
            "line_count": len(content.splitlines()),
            "variables": variables,
            "connections": connections,
            "sections": sections,
            "semantic_analysis": semantic_analysis,
            "complexity_metrics": complexity_metrics,
            "patterns": patterns,
            "analysis_timestamp": datetime.now().isoformat(),
            "analysis": "Deterministic structural analysis complete",
            "analysis_method": "structural",
            "llm_summary_status": "not_attempted",
        }

        # Attempt LLM-based summary using multi-provider system (with Ollama recovery)
        if not attempt_llm:
            result["llm_summary_status"] = "unavailable"
            result["llm_summary_error"] = (
                "Live LLM analysis was skipped because no provider initialized"
            )
        else:
            try:
                # Use async-aware API when available
                ops = LLMOperations()
                summary_candidate = ops.summarize_gnn(
                    content, max_length=500, ollama_model=ollama_model
                )
                if hasattr(summary_candidate, "__await__"):
                    summary_text = await summary_candidate
                else:
                    summary_text = summary_candidate
                if not isinstance(summary_text, str) or not summary_text.strip():
                    raise RuntimeError("LLM provider returned an empty summary")
                if summary_text.lstrip().lower().startswith("error:"):
                    raise RuntimeError(summary_text)
                result["llm_summary"] = summary_text
                result["llm_summary_status"] = "available"
                result["analysis_method"] = "structural_with_llm_summary"
                result["analysis"] = "Structural and LLM-assisted analysis complete"
            except Exception as e:
                # Recovery: tests and earlier integrations may patch this provider.
                try:
                    if ollama_model is not None:
                        raise RuntimeError(
                            "Pinned local analysis cannot substitute a cloud provider"
                        )
                    provider = OpenAIProvider()
                    candidate = provider.analyze(content, "summary")
                    if hasattr(candidate, "__await__"):
                        summary_text = await candidate
                    else:
                        summary_text = candidate
                    if not isinstance(summary_text, str) or not summary_text.strip():
                        raise RuntimeError(
                            "fallback provider returned an empty summary"
                        )
                    result["llm_summary"] = summary_text
                    result["llm_summary_status"] = "available"
                    result["analysis_method"] = "structural_with_llm_summary"
                    result["analysis"] = "Structural and LLM-assisted analysis complete"
                except Exception:
                    # Do not fail structural analysis if every provider is unavailable.
                    logger.warning(
                        f"LLM summary generation failed for {file_path.name}: {e}"
                    )
                    result["llm_summary_error"] = str(e)
                    result["llm_summary_status"] = "unavailable"

        result.setdefault("status", "SUCCESS")
        # Provide minimal documentation block expected by tests
        result.setdefault(
            "documentation", {"file_path": str(file_path), "model_overview": ""}
        )
        return result

    except Exception as e:
        raise Exception(f"Failed to analyze {file_path}: {e}") from e


def analyze_gnn_file_with_llm(
    file_path: Path,
    verbose: bool = False,
    ollama_model: Optional[str] = None,
    attempt_llm: bool = True,
) -> Union[Dict[str, Any], Coroutine[Any, Any, Dict[str, Any]]]:
    """
    Compatibility wrapper for the async analyzer.

    - If called from an active event loop, returns a coroutine which can be awaited.
    - If called synchronously, runs the async analyzer to completion and returns its result.
    """
    coro = _analyze_gnn_file_with_llm(
        file_path, verbose, ollama_model, attempt_llm=attempt_llm
    )
    try:
        # If an event loop is running, return the coroutine for the caller to await
        asyncio.get_running_loop()
        return coro
    except RuntimeError:
        # No running loop; execute synchronously
        return asyncio.run(coro)


def extract_variables(content: str) -> List[Dict[str, Any]]:
    """Extract variables from GNN content."""
    variables: list[Any] = []

    # Look for variable definitions
    var_patterns: list[Any] = [
        r"(\w+)\s*:\s*(\w+)",  # name: type
        r"(\w+)\s*=\s*([^;\n]+)",  # name = value
        r"(\w+)\s*\[([^\]]+)\]",  # name[dimensions]
    ]

    for pattern in var_patterns:
        matches = re.finditer(pattern, content)
        for match in matches:
            variables.append(
                {
                    "name": match.group(1),
                    "definition": match.group(0),
                    "line": content[: match.start()].count("\n") + 1,
                }
            )

    return variables


def extract_connections(content: str) -> List[Dict[str, Any]]:
    """Extract connections from GNN content.

    Parses the ## Connections section for GNN operators (>, -, <)
    and falls back to previous patterns (->. →, connects).
    """
    connections: list[Any] = []

    # Primary: parse ## Connections section for GNN-specific operators
    conn_section = re.search(
        r"##\s*Connections\s*\n(.*?)(?=\n##\s|\Z)", content, re.DOTALL
    )
    if conn_section:
        section_text = conn_section.group(1)
        for line in section_text.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # GNN connection operators: > (directional), - (bidirectional), < (reverse)
            conn_match = re.match(r"(\w+)\s*([>\-<])\s*(\w+)", line)
            if conn_match:
                src, op, tgt = (
                    conn_match.group(1),
                    conn_match.group(2),
                    conn_match.group(3),
                )
                conn_type = {
                    ">": "directional",
                    "-": "bidirectional",
                    "<": "reverse",
                }.get(op, "unknown")
                connections.append(
                    {
                        "source": src,
                        "target": tgt,
                        "connection": f"{src} {op} {tgt}",
                        "connection_type": conn_type,
                        "line": content[
                            : conn_match.start() + conn_section.start()
                        ].count("\n")
                        + 1,
                    }
                )

    # Recovery: previous patterns throughout the entire file
    if not connections:
        conn_patterns: list[Any] = [
            r"(\w+)\s*->\s*(\w+)",  # source -> target
            r"(\w+)\s*→\s*(\w+)",  # source → target
            r"(\w+)\s*connects\s*(\w+)",  # source connects target
        ]
        for pattern in conn_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                connections.append(
                    {
                        "source": match.group(1),
                        "target": match.group(2),
                        "connection": match.group(0),
                        "line": content[: match.start()].count("\n") + 1,
                    }
                )

    return connections


def variable_type_counts(variables: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count variables by declared type parsed from their definition.

    Shared by :func:`perform_semantic_analysis` and
    :mod:`llm.generator` so both report identical type distributions.
    Definitions without a ``name: type`` shape count as ``"unknown"``.
    """
    counts: dict[str, int] = {}
    for var in variables:
        definition = var.get("definition", "")
        var_type = definition.split(":")[-1].strip() if ":" in definition else "unknown"
        counts[var_type] = counts.get(var_type, 0) + 1
    return counts


def perform_semantic_analysis(
    content: str, variables: List[Dict], connections: List[Dict]
) -> Dict[str, Any]:
    """Perform semantic analysis of GNN content."""
    analysis: dict[str, Any] = {
        "variable_count": len(variables),
        "connection_count": len(connections),
        "complexity_score": len(variables) + len(connections),
        "semantic_patterns": [],
    }

    analysis["variable_types"] = variable_type_counts(variables)

    # Analyze connection patterns
    connection_types: dict[Any, Any] = {}
    for conn in connections:
        conn_type = (
            conn.get("connection", "").split()[1]
            if len(conn.get("connection", "").split()) > 1
            else "unknown"
        )
        connection_types[conn_type] = connection_types.get(conn_type, 0) + 1

    analysis["connection_types"] = connection_types

    return analysis


def calculate_complexity_metrics(
    variables: List[Dict], connections: List[Dict]
) -> Dict[str, Any]:
    """Calculate complexity metrics for the GNN model."""
    metrics: dict[str, Any] = {
        "total_elements": len(variables) + len(connections),
        "variable_complexity": len(variables),
        "connection_complexity": len(connections),
        "density": len(connections) / max(len(variables), 1),
        "cyclomatic_complexity": len(connections) - len(variables) + 2,
    }

    return metrics


def identify_patterns(
    content: str, variables: List[Dict], connections: List[Dict]
) -> Dict[str, Any]:
    """Identify patterns and anti-patterns in GNN content."""
    patterns: dict[str, Any] = {"patterns": [], "anti_patterns": [], "suggestions": []}

    # Check for common patterns
    if len(variables) > 10:
        patterns["patterns"].append("High variable count - complex model")

    if len(connections) > len(variables) * 2:
        patterns["patterns"].append("High connectivity - dense graph")

    # Check for anti-patterns
    if len(variables) == 0:
        patterns["anti_patterns"].append("No variables defined")

    if len(connections) == 0:
        patterns["anti_patterns"].append("No connections defined")

    # Generate suggestions
    if len(variables) > 20:
        patterns["suggestions"].append("Consider breaking down into smaller modules")

    if len(connections) > 50:
        patterns["suggestions"].append("Consider simplifying the model structure")

    return patterns
