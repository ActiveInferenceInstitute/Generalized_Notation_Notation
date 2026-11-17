#!/usr/bin/env python3
"""Shared command semantics for the ``gnn`` CLI, HTTP API and MCP adapters.

The CLI dispatcher (:mod:`gnn.cli`) and the API parity adapters
(:mod:`gnn.api.parity`) are transports over the same operations: parsing,
validation, extraction, preflight, and report publication live here, below
both adapters, and each adapter only maps outcomes onto its own contract
(CLI exit codes ``0``/``2``/``1``; HTTP envelopes). The render helpers
delegate to the renderer, which owns single-framework result and artifact
selection for all callers. :mod:`gnn.cli.commands` remains a re-export
facade for existing importers.

Import policy: module scope imports only the standard library, and every
heavy backend module (``gnn.schema``, ``gnn.validation``, ``gnn.extract``,
``gnn.parsers``, ``gnn.report``, ``gnn.pipeline.preflight``) is imported
lazily inside the functions. This keeps CLI boot fast and lets
``gnn.api.parity`` import this module at module scope. This module must
never import ``gnn.cli`` or ``gnn.api``.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gnn.pipeline.preflight import PreflightReport
    from gnn.schema import GNNConnectionEdge, GNNParseError, GNNVariable

__all__ = [
    "EXTRACT_DEFECT_EMPTY_SKELETON",
    "EXTRACT_DEFECT_ERROR_STATUS",
    "ValidationOutcome",
    "build_parse_payload",
    "extract_payload_defect",
    "find_render_artifact",
    "preflight_severities",
    "publish_pipeline_report",
    "run_validation_checks",
    "render_processing_succeeded",
]

logger = logging.getLogger(__name__)


def render_processing_succeeded(value: bool | int) -> bool:
    """Classify the renderer result through the compatibility import."""
    from gnn.render import render_processing_succeeded as succeeded

    return succeeded(value)


#: ``extract_payload_defect`` verdict: the extractor returned its failure
#: envelope (``{"status": "error", ...}``).
EXTRACT_DEFECT_ERROR_STATUS = "error-status"

#: ``extract_payload_defect`` verdict: lenient extraction fabricated a default
#: skeleton because the file had no GNN state-space content at all.
EXTRACT_DEFECT_EMPTY_SKELETON = "empty-skeleton"


@dataclass
class ValidationOutcome:
    """Result of the shared ``gnn validate`` check chain."""

    errors: list[GNNParseError]
    variables: list[GNNVariable]
    connections: list[GNNConnectionEdge]
    semantic: dict[str, Any]


def run_validation_checks(content: str, file_name: str) -> ValidationOutcome:
    """Run the ``gnn validate`` check chain over one GNN document.

    Sections → state space → connections (against the parsed variables) →
    matrix dimensions → semantic validation, with ``GNN-SEMANTIC`` findings
    deduplicated against earlier errors by message.
    """
    from gnn.schema import (
        GNNParseError,
        parse_connections,
        parse_state_space,
        validate_matrix_dimensions,
        validate_required_sections,
    )
    from gnn.validation import validate_content

    errors: list[GNNParseError] = []
    errors.extend(validate_required_sections(content, file_path=file_name))
    variables, variable_errors = parse_state_space(content, file_path=file_name)
    errors.extend(variable_errors)
    connections, connection_errors = parse_connections(
        content,
        known_variables={variable.name for variable in variables},
        file_path=file_name,
    )
    errors.extend(connection_errors)
    errors.extend(validate_matrix_dimensions(content, variables, file_path=file_name))

    semantic = validate_content(content)
    known_messages = {error.message for error in errors}
    errors.extend(
        GNNParseError(code="GNN-SEMANTIC", message=message, file=file_name)
        for message in semantic["errors"]
        if message not in known_messages
    )
    return ValidationOutcome(
        errors=errors, variables=variables, connections=connections, semantic=semantic
    )


def build_parse_payload(gnn_file: Path) -> tuple[dict[str, Any], list[GNNParseError]]:
    """Parse one GNN file into the CLI-shaped result dict.

    Mirrors the shared ``gnn parse`` body: state-space and connection parsing,
    best-effort frontmatter, and best-effort POMDP extraction. Non-POMDP files
    or extraction failures omit the ``pomdp`` key and append a warning note
    (last) to ``warnings``. The extraction must identify a parsed model parameter
    declared in the same document snapshot. Adapter defaults alone cannot
    turn an interface into a probabilistic model. ``errors`` entries are plain dicts with
    ``code``/``message``/``line``/``file`` keys.

    Returns ``(result, parse_errors)`` — the raw parse errors power exit-code
    and status decisions at both call sites.
    """
    from gnn.schema import parse_connections, parse_state_space

    content = gnn_file.read_text(encoding="utf-8")
    file_name = str(gnn_file)

    variables, variable_errors = parse_state_space(content, file_path=file_name)
    connections, connection_errors = parse_connections(
        content,
        known_variables={variable.name for variable in variables},
        file_path=file_name,
    )
    parse_errors = [*variable_errors, *connection_errors]

    metadata: dict[str, Any] = {}
    try:
        from gnn.parsers.frontmatter import parse_frontmatter

        metadata, _ = parse_frontmatter(content)
    except ImportError as exc:
        logger.debug("Frontmatter parsing not available: %s", exc)

    # POMDP extraction is best-effort, exactly like the CLI: non-POMDP files
    # or extraction failures omit the "pomdp" key and add a warning note.
    pomdp_note: str | None = None
    pomdp_data: dict[str, Any] | None = None
    if not variables:
        pomdp_note = "File declares no StateSpaceBlock variables; 'pomdp' key omitted"
    else:
        try:
            from gnn.extract.pomdp_extractor import extract_pomdp_from_content

            pomdp_space = extract_pomdp_from_content(content, strict_validation=False)
            if pomdp_space is None:
                pomdp_note = "File does not parse as a POMDP; 'pomdp' key omitted"
            elif any(
                evidence.get("source") == "InitialParameterization"
                for evidence in (pomdp_space.matrix_provenance or {}).values()
            ):
                pomdp_data = pomdp_space.to_dict()
            else:
                pomdp_note = (
                    "No parsed source-declared POMDP parameters; 'pomdp' key omitted"
                )
        except Exception as exc:
            logger.debug("POMDP extraction unavailable: %s", exc)
            pomdp_note = f"POMDP extraction unavailable: {exc}"

    warnings_list = [str(error) for error in parse_errors]
    if pomdp_note is not None:
        warnings_list.append(pomdp_note)

    variables_payload: list[dict[str, Any]] = [
        {
            "name": variable.name,
            "dimensions": variable.dimensions,
            "dtype": variable.dtype,
            "default": variable.default,
        }
        for variable in variables
    ]
    connections_payload: list[dict[str, Any]] = [
        {
            "source": connection.source,
            "target": connection.target,
            "directed": connection.directed,
            "label": connection.label,
            "line": connection.line,
        }
        for connection in connections
    ]
    serialized_errors: list[dict[str, Any]] = [
        {
            "code": error.code,
            "message": error.message,
            "line": error.line,
            "file": error.file,
        }
        for error in parse_errors
    ]
    result: dict[str, Any] = {
        "file": file_name,
        "metadata": metadata,
        "variables": variables_payload,
        "connections": connections_payload,
        "warnings": warnings_list,
        "errors": serialized_errors,
    }
    if pomdp_data is not None:
        result["pomdp"] = pomdp_data
    return result, parse_errors


def extract_payload_defect(payload_obj: Any) -> str | None:
    """Classify an ``extract`` JSON payload's user-visible defect, if any.

    Returns ``EXTRACT_DEFECT_ERROR_STATUS`` for a failure envelope,
    ``EXTRACT_DEFECT_EMPTY_SKELETON`` for the fabricated empty skeleton, and
    ``None`` otherwise. Non-dict payloads yield ``None``: the CLI treats
    unparseable output as success, while the API layer maps non-dict output
    to its own 400 — that divergence lives at the call sites.
    """
    if not isinstance(payload_obj, dict):
        return None
    if payload_obj.get("status") == "error":
        return EXTRACT_DEFECT_ERROR_STATUS
    if (
        not payload_obj.get("state_variables")
        and not payload_obj.get("matrices")
        and payload_obj.get("model_name") is None
    ):
        return EXTRACT_DEFECT_EMPTY_SKELETON
    return None


def preflight_severities(report: PreflightReport) -> tuple[bool, bool]:
    """Return ``(has_errors, has_warnings)`` for one preflight report."""
    return (
        any(issue.severity == "error" for issue in report.issues),
        any(issue.severity == "warning" for issue in report.issues),
    )


def publish_pipeline_report(output_dir: Path) -> tuple[str, Path]:
    """Generate and atomically publish ``PIPELINE_REPORT.md`` in ``output_dir``.

    Writes through a same-directory temporary file and ``os.replace`` so
    readers never observe a partial report. Returns the report text and the
    published path.
    """
    from gnn.report.pipeline_report import generate_pipeline_report

    report = generate_pipeline_report(output_dir)
    report_path = output_dir / "PIPELINE_REPORT.md"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=report_path.parent, delete=False
    ) as tmp_file:
        tmp_file.write(report)
    os.replace(tmp_file.name, str(report_path))
    return report, report_path


def find_render_artifact(
    render_dir: Path, framework: str, *, require_current: bool = False
) -> Path | None:
    """Locate a render artifact through the compatibility import."""
    from gnn.render import find_render_artifact as find_artifact

    return find_artifact(render_dir, framework, require_current=require_current)
