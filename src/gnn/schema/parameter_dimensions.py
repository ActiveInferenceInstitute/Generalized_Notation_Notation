"""Parameter assignment parsing and declaration-shape diagnostics.

This lightweight boundary preserves symbolic and action-major layout semantics;
it validates declared shapes, not scientific probability normalization.
"""

import logging
import re
from collections.abc import Sequence
from typing import Any, List, Optional

from .types import GNNParseError, GNNVariable

logger = logging.getLogger(__name__)


_PARAM_ASSIGN_RE = re.compile(
    r"^(?P<name>[A-Za-z_π'][A-Za-z0-9_π']*)\s*=\s*(?P<value>.*)$"
)


def _strip_parameter_comment(line: str) -> str:
    """Remove a GNN inline comment from one parameterization line."""
    if "#" not in line:
        return line.strip()
    return line[: line.index("#")].strip()


def _parse_parameter_value(raw_value: str) -> Any:
    """Parse GNN tuple/braced parameter syntax as a Python literal structure."""
    from gnn.utils.runtime_safety.safe_eval import safe_literal_eval

    literal_text = raw_value.strip().replace("{", "[").replace("}", "]")
    return safe_literal_eval(literal_text)


def _parameter_shape(value: Any) -> tuple[list[int], bool]:
    """Return the nested sequence shape and whether any nested row is ragged."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not value:
            return [0], False
        child_shapes: list[list[int]] = []
        ragged = False
        for child in value:
            child_shape, child_ragged = _parameter_shape(child)
            child_shapes.append(child_shape)
            ragged = ragged or child_ragged
        first_shape = child_shapes[0]
        if any(shape != first_shape for shape in child_shapes[1:]):
            ragged = True
        return [len(value), *first_shape], ragged
    return [], False


def _numeric_dimensions(dimensions: list[str]) -> list[int] | None:
    """Return numeric dimensions or None when any dimension is symbolic."""
    try:
        return [int(dim) for dim in dimensions]
    except (TypeError, ValueError):
        return None


def _shape_for_comparison(
    actual_shape: list[int], expected_shape: list[int]
) -> list[int]:
    """Normalize established GNN parameter layout conventions for comparison."""
    if len(expected_shape) == 1 and len(actual_shape) == 2 and actual_shape[0] == 1:
        return actual_shape[1:]
    if len(expected_shape) == 2 and expected_shape[1] == 1:
        if len(actual_shape) == 1 and actual_shape[0] == expected_shape[0]:
            return expected_shape
        if len(actual_shape) == 2 and actual_shape == [1, expected_shape[0]]:
            return expected_shape
    if len(expected_shape) == 3 and len(actual_shape) == 3:
        action_major_shape = [expected_shape[2], expected_shape[0], expected_shape[1]]
        if actual_shape == action_major_shape:
            return expected_shape
    return actual_shape


def _format_shape(shape: list[int]) -> str:
    """Format a numeric shape for diagnostics."""
    return "scalar" if not shape else "x".join(str(part) for part in shape)


def _validate_parameter_assignment(
    *,
    variable_name: str,
    raw_value: str,
    end_line: int,
    var_map: dict[str, GNNVariable],
    file_path: Optional[str],
) -> list[GNNParseError]:
    """Validate one InitialParameterization assignment against declarations."""
    if variable_name not in var_map:
        return [
            GNNParseError(
                code="GNN-W003",
                message=f"Parameterization for undeclared variable '{variable_name}'",
                line=end_line,
                file=file_path,
                severity="warning",
            )
        ]

    decl = var_map[variable_name]
    expected_shape = _numeric_dimensions(decl.dimensions)
    if expected_shape is None:
        logger.debug(
            "Symbolic dimensions for '%s', skipping numeric check: %s",
            variable_name,
            decl.dimensions,
        )
        return []

    try:
        parsed_value = _parse_parameter_value(raw_value)
    except (SyntaxError, ValueError, TypeError) as e:
        logger.debug(
            "Could not parse parameterization for '%s', skipping shape check: %s",
            variable_name,
            e,
        )
        return []

    actual_shape, ragged = _parameter_shape(parsed_value)
    comparable_shape = _shape_for_comparison(actual_shape, expected_shape)
    if comparable_shape == expected_shape and not ragged:
        return []

    ragged_suffix = " (ragged)" if ragged else ""
    return [
        GNNParseError(
            code="GNN-E002",
            message=(
                f"Matrix '{variable_name}': declared shape "
                f"{_format_shape(expected_shape)} but parameterization has shape "
                f"{_format_shape(comparable_shape)}{ragged_suffix}"
            ),
            line=end_line,
            file=file_path,
        )
    ]


def validate_matrix_dimensions(
    content: str,
    variables: List[GNNVariable],
    *,
    file_path: Optional[str] = None,
) -> List[GNNParseError]:
    """
    Cross-validate InitialParameterization dimensions against StateSpaceBlock
    declarations.

    Returns list of GNNParseError for any mismatches.
    """
    errors: List[GNNParseError] = []
    var_map = {v.name: v for v in variables}

    in_param = False
    current_var: Optional[str] = None
    brace_depth = 0
    value_lines: list[str] = []

    for line_no, raw_line in enumerate(content.splitlines(), start=1):
        line = raw_line.strip()

        if line.startswith("## "):
            if current_var is not None:
                errors.extend(
                    _validate_parameter_assignment(
                        variable_name=current_var,
                        raw_value="\n".join(value_lines),
                        end_line=line_no - 1,
                        var_map=var_map,
                        file_path=file_path,
                    )
                )
                current_var = None
                value_lines = []
                brace_depth = 0
            section_name = line[3:].strip()
            in_param = section_name == "InitialParameterization"
            continue

        if not in_param:
            continue
        if not line or line.startswith("#"):
            continue

        value_line = _strip_parameter_comment(line)
        if not value_line:
            continue

        if current_var is not None:
            value_lines.append(value_line)
            brace_depth += value_line.count("{") - value_line.count("}")
        else:
            assign_match = _PARAM_ASSIGN_RE.match(value_line)
            if not assign_match:
                continue
            current_var = assign_match.group("name")
            assignment_value = assign_match.group("value").strip()
            value_lines = [assignment_value]
            brace_depth = assignment_value.count("{") - assignment_value.count("}")

        if current_var is not None and brace_depth <= 0:
            errors.extend(
                _validate_parameter_assignment(
                    variable_name=current_var,
                    raw_value="\n".join(value_lines),
                    end_line=line_no,
                    var_map=var_map,
                    file_path=file_path,
                )
            )
            current_var = None
            value_lines = []
            brace_depth = 0

    if current_var is not None:
        errors.extend(
            _validate_parameter_assignment(
                variable_name=current_var,
                raw_value="\n".join(value_lines),
                end_line=len(content.splitlines()),
                var_map=var_map,
                file_path=file_path,
            )
        )

    return errors
