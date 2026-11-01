"""Markdown GNN parsing fallback when step-3 parsed JSON is unavailable."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from gnn.parsers.common import canonical_section_heading

from ..matrix.compat import parse_matrix_data

logger = logging.getLogger(__name__)


def _is_complete_parameter(value_str: str) -> bool:
    """Return whether complete parameter."""
    open_braces = value_str.count("{")
    close_braces = value_str.count("}")
    open_parens = value_str.count("(")
    close_parens = value_str.count(")")
    return (
        open_braces == close_braces
        and open_parens == close_parens
        and (open_braces > 0 or open_parens > 0)
    )


def _parse_parameter_value(value_str: str) -> Any:
    """Parse parameter value."""
    from gnn.utils.runtime_safety.safe_eval import safe_literal_eval

    try:
        cleaned = value_str.strip()
        cleaned = cleaned.replace("{", "[").replace("}", "]")
        cleaned = cleaned.replace("(", "[").replace(")", "]")
        return safe_literal_eval(cleaned)
    except Exception:
        return None


def _save_parameter(
    parsed: Dict[str, Any], param_name: str, param_lines: List[str]
) -> None:
    """Save parameter."""
    try:
        full_value = " ".join(param_lines)
        parsed_value = _parse_parameter_value(full_value)
        if parsed_value is not None:
            parsed["parameters"].append(
                {"name": param_name, "value": parsed_value, "raw": full_value}
            )
    except Exception as e:
        parsed["parameters"].append(
            {
                "name": param_name,
                "value": None,
                "raw": " ".join(param_lines),
                "parse_error": str(e),
            }
        )


def parse_gnn_content(content: str) -> Dict[str, Any]:
    """
    Parse GNN markdown into structured data for visualization.

    Prefer load_visualization_model (JSON-first) when step-3 output exists.
    """
    try:
        parsed: Dict[str, Any] = {
            "sections": {},
            "raw_sections": {},
            "variables": [],
            "connections": [],
            "matrices": [],
            "parameters": [],
            "metadata": {},
        }

        lines = content.split("\n")
        current_section = None
        current_param_name = None
        current_param_lines: List[str] = []
        in_multiline_param = False

        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:
                continue

            if stripped_line.startswith("##") and not stripped_line.startswith("###"):
                if current_param_name and current_param_lines:
                    _save_parameter(parsed, current_param_name, current_param_lines)
                    current_param_name = None
                    current_param_lines = []
                    in_multiline_param = False

                current_section = canonical_section_heading(
                    stripped_line.lstrip("#").strip()
                )
                parsed["sections"][current_section] = []
                parsed["raw_sections"][current_section] = ""
            elif current_section:
                parsed["sections"][current_section].append(stripped_line)
                if parsed["raw_sections"][current_section]:
                    parsed["raw_sections"][current_section] += "\n" + line
                else:
                    parsed["raw_sections"][current_section] = line

                if current_section == "InitialParameterization":
                    if "=" in stripped_line and not stripped_line.startswith("#"):
                        if current_param_name and current_param_lines:
                            _save_parameter(
                                parsed, current_param_name, current_param_lines
                            )
                            current_param_lines = []

                        eq_pos = stripped_line.find("=")
                        current_param_name = stripped_line[:eq_pos].strip()
                        param_value_part = stripped_line[eq_pos + 1 :].strip()

                        if param_value_part:
                            current_param_lines = [param_value_part]
                            if _is_complete_parameter(param_value_part):
                                _save_parameter(
                                    parsed, current_param_name, current_param_lines
                                )
                                current_param_name = None
                                current_param_lines = []
                                in_multiline_param = False
                            else:
                                in_multiline_param = True
                    elif in_multiline_param and current_param_name:
                        current_param_lines.append(stripped_line)
                        full_value = " ".join(current_param_lines)
                        if _is_complete_parameter(full_value):
                            _save_parameter(
                                parsed, current_param_name, current_param_lines
                            )
                            current_param_name = None
                            current_param_lines = []
                            in_multiline_param = False

                if ":" in stripped_line and "=" not in stripped_line:
                    var_parts = stripped_line.split(":", 1)
                    if len(var_parts) == 2:
                        var_name = var_parts[0].strip()
                        var_type = var_parts[1].strip()
                        parsed["variables"].append({"name": var_name, "type": var_type})
                elif "[" in stripped_line and "type=" in stripped_line:
                    bracket_pos = stripped_line.find("[")
                    if bracket_pos != -1:
                        var_name = stripped_line[:bracket_pos].strip()
                        type_start = stripped_line.find("type=", bracket_pos)
                        if type_start != -1:
                            type_end = (
                                stripped_line.find(",", type_start)
                                if "," in stripped_line[type_start:]
                                else stripped_line.find("]", type_start)
                            )
                            if type_end == -1:
                                type_end = len(stripped_line)
                            var_type = stripped_line[type_start:type_end].strip()
                            parsed["variables"].append(
                                {"name": var_name, "type": var_type}
                            )
                elif (
                    "->" in stripped_line
                    or "\u2192" in stripped_line
                    or ">" in stripped_line
                    or ("-" in stripped_line and current_section == "Connections")
                ):
                    if "->" in stripped_line:
                        conn_parts = stripped_line.split("->", 1)
                    elif "\u2192" in stripped_line:
                        conn_parts = stripped_line.split("\u2192", 1)
                    elif ">" in stripped_line:
                        conn_parts = stripped_line.split(">", 1)
                    else:
                        conn_parts = stripped_line.split("-", 1)

                    if len(conn_parts) == 2:
                        source = conn_parts[0].strip()
                        target = conn_parts[1].strip()
                        if (
                            source
                            and target
                            and (
                                source.replace("_", "").replace("-", "").isalnum()
                                or source in ["s", "o", "\u03c0", "u"]
                            )
                            and (
                                target.replace("_", "").replace("-", "").isalnum()
                                or target in ["s", "o", "\u03c0", "u"]
                            )
                        ):
                            parsed["connections"].append(
                                {"source": source, "target": target}
                            )
                elif ("{" in stripped_line and "}" in stripped_line) or (
                    "[" in stripped_line and "]" in stripped_line
                ):
                    try:
                        matrix_data = parse_matrix_data(stripped_line)
                        if matrix_data is not None:
                            parsed["matrices"].append(
                                {"data": matrix_data, "definition": stripped_line}
                            )
                    except (ValueError, TypeError, SyntaxError):
                        logger.debug(
                            "Best-effort matrix parsing failed for line: %s",
                            stripped_line[:80],
                        )

        if current_param_name and current_param_lines:
            _save_parameter(parsed, current_param_name, current_param_lines)

        return parsed

    except Exception as e:
        return {
            "error": str(e),
            "sections": {},
            "raw_sections": {},
            "variables": [],
            "connections": [],
            "matrices": [],
            "parameters": [],
            "metadata": {},
        }
