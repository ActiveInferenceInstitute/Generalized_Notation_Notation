#!/usr/bin/env python3
"""
GNN Schema Module  — JSON Schema definitions and validation for parsed GNN objects.

Provides:
  - GNNParseError: structured error with code, line number, and message
  - GNN_MODEL_SCHEMA: JSON Schema dict for a single parsed GNN model
  - validate_gnn_object(): validates a dict against the schema
  - parse_connections(): extracts edges with optional annotations
  - validate_matrix_dimensions(): cross-checks parameterization vs declarations
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .parameter_dimensions import (
    validate_matrix_dimensions as validate_matrix_dimensions,
)
from .types import GNNConnectionEdge, GNNParseError, GNNVariable

logger = logging.getLogger(__name__)

# ─── Connection Parsing ─────────────────────────────────────────────────────────

# Matches: A>B, A-B, A>B:label, A-B:label (names may contain '+', e.g. s_t+1)
_CONNECTION_RE = re.compile(
    r"^(?P<source>[A-Za-z_π'][A-Za-z0-9_π'+]*)"
    r"(?P<op>[>\-])"
    r"(?P<target>[A-Za-z_π'][A-Za-z0-9_π'+]*)"
    r"(?::(?P<label>[A-Za-z0-9_]+))?$"
)


def parse_connections(
    content: str,
    *,
    known_variables: Optional[set] = None,
    file_path: Optional[str] = None,
) -> Tuple[List[GNNConnectionEdge], List[GNNParseError]]:
    """
    Parse the Connections section of a GNN file.

    Args:
        content: Full GNN file content (or just the Connections section).
        known_variables: Set of declared variable names for cross-validation.
        file_path: Optional source file for error reporting.

    Returns:
        (connections, errors) tuple.
    """
    connections: List[GNNConnectionEdge] = []
    errors: List[GNNParseError] = []

    # Find the ## Connections section
    in_connections = False
    for line_no, raw_line in enumerate(content.splitlines(), start=1):
        line = raw_line.strip()

        # Detect section boundaries
        if line.startswith("## "):
            section_name = line[3:].strip()
            in_connections = section_name == "Connections"
            continue

        if not in_connections:
            continue
        if not line or line.startswith("#"):
            continue

        m = _CONNECTION_RE.match(line)
        if m:
            conn = GNNConnectionEdge(
                source=m.group("source"),
                target=m.group("target"),
                directed=(m.group("op") == ">"),
                label=m.group("label"),
                line=line_no,
            )
            connections.append(conn)

            # Cross-validate against declared variables
            if known_variables is not None:
                if conn.source not in known_variables:
                    errors.append(
                        GNNParseError(
                            code="GNN-W002",
                            message=f"Connection references undeclared variable '{conn.source}'",
                            line=line_no,
                            file=file_path,
                            severity="warning",
                        )
                    )
                if conn.target not in known_variables:
                    errors.append(
                        GNNParseError(
                            code="GNN-W002",
                            message=f"Connection references undeclared variable '{conn.target}'",
                            line=line_no,
                            file=file_path,
                            severity="warning",
                        )
                    )
        else:
            errors.append(
                GNNParseError(
                    code="GNN-E005",
                    message=f"Unparseable connection: '{line}'",
                    line=line_no,
                    file=file_path,
                )
            )

    return connections, errors


# ─── Variable Parsing (enhanced) ────────────────────────────────────────────────

_VAR_RE = re.compile(
    r"^(?P<name>[A-Za-z_π'][A-Za-z0-9_π'+]*)"
    r"\[(?P<dims>[^\]]+)\]"
)


def parse_state_space(
    content: str,
    *,
    file_path: Optional[str] = None,
) -> Tuple[List[GNNVariable], List[GNNParseError]]:
    """
    Parse the StateSpaceBlock section of a GNN file.

    Returns:
        (variables, errors) tuple.
    """
    variables: List[GNNVariable] = []
    errors: List[GNNParseError] = []
    seen_names: set = set()

    in_ssb = False
    for line_no, raw_line in enumerate(content.splitlines(), start=1):
        line = raw_line.strip()

        if line.startswith("## "):
            section_name = line[3:].strip()
            in_ssb = section_name == "StateSpaceBlock"
            continue

        if not in_ssb:
            continue
        if not line or line.startswith("#"):
            continue

        # Strip inline comment
        if "#" in line:
            line = line[: line.index("#")].strip()

        m = _VAR_RE.match(line)
        if not m:
            continue

        name = m.group("name")
        raw_dims = m.group("dims")

        # Parse comma-separated dimension entries
        parts = [p.strip() for p in raw_dims.split(",")]
        dims: list[Any] = []
        dtype = "float"
        default = None
        for part in parts:
            if "=" in part:
                key, val = part.split("=", 1)
                key, val = key.strip(), val.strip()
                if key == "type":
                    dtype = val
                elif key == "default":
                    default = val
            else:
                dims.append(part)

        if name in seen_names:
            errors.append(
                GNNParseError(
                    code="GNN-E004",
                    message=f"Duplicate variable declaration: '{name}'",
                    line=line_no,
                    file=file_path,
                )
            )
        seen_names.add(name)

        variables.append(
            GNNVariable(
                name=name,
                dimensions=dims,
                dtype=dtype,
                default=default,
                line=line_no,
            )
        )

    return variables, errors


# ─── Matrix Dimension Validation ─────────────────────────────────────────────────


# ─── JSON Schema ─────────────────────────────────────────────────────────────────

GNN_MODEL_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://activeinference.org/gnn/model.schema.json",
    "title": "GNN Model",
    "description": "Schema for a parsed GNN (Generalized Notation Notation) model object.",
    "type": "object",
    "required": [
        "gnn_section",
        "gnn_version",
        "model_name",
        "state_space",
        "connections",
    ],
    "properties": {
        "gnn_section": {
            "type": "string",
            "description": "Short model identifier (no spaces).",
            "pattern": r"^[A-Za-z][A-Za-z0-9_]*$",
        },
        "gnn_version": {
            "type": "string",
            "description": "GNN language version.",
            "enum": ["1", "1.0", "1.1"],
        },
        "model_name": {
            "type": "string",
            "description": "Human-readable model name.",
            "minLength": 1,
        },
        "model_annotation": {
            "type": "string",
            "description": "Optional free-text model description.",
        },
        "state_space": {
            "type": "array",
            "description": "List of variable declarations.",
            "items": {
                "type": "object",
                "required": ["name", "dimensions", "dtype"],
                "properties": {
                    "name": {"type": "string"},
                    "dimensions": {
                        "type": "array",
                        "items": {"type": ["string", "integer"]},
                    },
                    "dtype": {"type": "string", "enum": ["float", "int", "bool"]},
                    "default": {"type": ["string", "null"]},
                },
            },
        },
        "connections": {
            "type": "array",
            "description": "List of edges between state-space variables.",
            "items": {
                "type": "object",
                "required": ["source", "target", "directed"],
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "directed": {"type": "boolean"},
                    "label": {"type": ["string", "null"]},
                },
            },
        },
        "initial_parameterization": {
            "type": "object",
            "description": "Variable name → matrix values (as nested lists).",
            "additionalProperties": True,
        },
    },
}


def validate_gnn_object(obj: Dict[str, Any]) -> List[str]:
    """
    Validate a parsed GNN model dict against the JSON Schema.

    Uses jsonschema if available, otherwise performs basic key checks.

    Returns:
        List of validation error messages (empty = valid).
    """
    try:
        import jsonschema

        validator = jsonschema.Draft202012Validator(GNN_MODEL_SCHEMA)
        return [e.message for e in sorted(validator.iter_errors(obj), key=str)]
    except ImportError:
        logger.debug("jsonschema not installed — using basic validation")
        missing = [k for k in GNN_MODEL_SCHEMA["required"] if k not in obj]
        return [f"Missing required key: '{k}'" for k in missing]


# ─── Required Sections Validation ────────────────────────────────────────────────

REQUIRED_SECTIONS: set[Any] = {
    "GNNSection",
    "GNNVersionAndFlags",
    "ModelName",
    "StateSpaceBlock",
    "Connections",
}


def validate_required_sections(
    content: str,
    *,
    file_path: Optional[str] = None,
) -> List[GNNParseError]:
    """Check that all required sections are present."""
    found: set[Any] = set()
    for line in content.splitlines():
        if line.strip().startswith("## "):
            found.add(line.strip()[3:].strip())

    errors: list[Any] = []
    for section in REQUIRED_SECTIONS:
        if section not in found:
            errors.append(
                GNNParseError(
                    code="GNN-E001",
                    message=f"Missing required section: '## {section}'",
                    file=file_path,
                )
            )
    return errors
