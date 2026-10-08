"""Model parsing and serialization helpers for GUI 2's matrix editor."""

from __future__ import annotations

import copy
import math
import re
from typing import Any, Dict, List, Sequence

_MATRIX_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "A": {"shape": [3, 3], "description": "Likelihood matrix"},
    "B": {"shape": [3, 3, 3], "description": "Transition matrices"},
    "C": {"shape": [3], "description": "Preference vector"},
    "D": {"shape": [3], "description": "Prior vector"},
}
_NUMBER_RE = re.compile(
    r"(?<![\w.])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?![\w.])"
)
_STATE_DECLARATION_RE = re.compile(
    r"^\s*([^\W\d]\w*)\s*\[([^\]]+)\](?:\s*#\s*(.*))?$", re.UNICODE
)


def get_pomdp_template() -> str:
    """Return the structurally valid POMDP starter used by GUI 2."""
    return """# GNN Visual Model Editor
# GNN Version: 1.0
# This model is being constructed using the Visual Matrix Editor (GUI 2)

## GNNSection
VisualPOMDP

## GNNVersionAndFlags
GNN v1

## ModelName
Visual Active Inference POMDP Agent

## ModelAnnotation
This model is constructed using the visual matrix editor:
- Interactive matrix editing via drag-and-drop
- Real-time state space visualization
- Live GNN markdown synchronization

## StateSpaceBlock
# Likelihood matrix: A[observation_outcomes, hidden_states]
A[3,3,type=float]   # Observation likelihood matrix

# Transition matrix: B[states_next, states_previous, actions]
B[3,3,3,type=float]   # State transition matrices

# Preference vector: C[observation_outcomes]
C[3,type=float]       # Log-preferences over observations

# Prior vector: D[states]
D[3,type=float]       # Prior over initial hidden states

# Hidden State
s[3,1,type=float]     # Current hidden state distribution

# Observation
o[3,1,type=int]       # Current observation

# Policy and Control
π[3,type=float]       # Policy distribution
u[1,type=int]         # Action taken

# Time
t[1,type=int]         # Discrete time step

## Connections
D>s
s-A
A-o
s>B
B>u
π>u

## InitialParameterization
# Visual editing will populate these values
A={
  (0.9, 0.05, 0.05),
  (0.05, 0.9, 0.05),
  (0.05, 0.05, 0.9)
}

B={
  ( (1.0,0.0,0.0), (0.0,1.0,0.0), (0.0,0.0,1.0) ),
  ( (0.0,1.0,0.0), (1.0,0.0,0.0), (0.0,0.0,1.0) ),
  ( (0.0,0.0,1.0), (0.0,1.0,0.0), (1.0,0.0,0.0) )
}

C={(0.1, 0.1, 1.0)}
D={(0.33, 0.33, 0.33)}

## Footer
Visual Active Inference POMDP Agent - GUI 2 Visual Editor
"""


def _section_content(gnn_text: str, section_name: str) -> str:
    pattern = (
        rf"(?ms)^##[ \t]+{re.escape(section_name)}[ \t]*\r?\n(.*?)(?=^##[ \t]+|\Z)"
    )
    match = re.search(pattern, gnn_text)
    return match.group(1) if match else ""


def _numeric_dimensions(raw_dimensions: str) -> List[int]:
    dimensions: List[int] = []
    for raw_part in raw_dimensions.split(","):
        part = raw_part.strip()
        if not part or "=" in part:
            continue
        if part.isdigit():
            dimensions.append(int(part))
    return dimensions


def _parse_state_spaces(gnn_text: str) -> Dict[str, Dict[str, Any]]:
    state_spaces: Dict[str, Dict[str, Any]] = {}
    for raw_line in _section_content(gnn_text, "StateSpaceBlock").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _STATE_DECLARATION_RE.match(line)
        if match is None:
            continue
        raw_dimensions = match.group(2)
        type_match = re.search(r"(?:^|,)\s*type\s*=\s*([^,\]]+)", raw_dimensions)
        state_spaces[match.group(1)] = {
            "shape": _numeric_dimensions(raw_dimensions),
            "type": type_match.group(1).strip() if type_match else None,
            "description": (match.group(3) or "").strip(),
        }
    return state_spaces


def _braced_assignment_span(text: str, name: str) -> tuple[int, int] | None:
    match = re.search(rf"(?m)^[ \t]*{re.escape(name)}[ \t]*=[ \t]*\{{", text)
    if match is None:
        return None
    opening = match.end() - 1
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return match.start(), index + 1
    return None


def _reshape_flat(values: Sequence[float], shape: Sequence[int]) -> Any:
    if not shape:
        return list(values)
    if len(shape) == 1:
        return list(values[: shape[0]])
    stride = math.prod(shape[1:])
    return [
        _reshape_flat(values[index * stride : (index + 1) * stride], shape[1:])
        for index in range(shape[0])
    ]


def _parse_parameter_values(
    parameter_block: str, name: str, shape: Sequence[int]
) -> Any | None:
    span = _braced_assignment_span(parameter_block, name)
    if span is None or not shape or any(dimension < 0 for dimension in shape):
        return None
    assignment = parameter_block[span[0] : span[1]]
    opening = assignment.find("{")
    numeric_values = [
        float(value) for value in _NUMBER_RE.findall(assignment[opening + 1 : -1])
    ]
    if len(numeric_values) != math.prod(shape):
        return None
    if name == "B" and len(shape) == 3:
        rows, columns, depth = shape
        canonical = _reshape_flat(numeric_values, shape)
        # GNN declares B[next_state, previous_state, action]; the editor
        # displays one action plane as a two-dimensional table.
        return [
            [
                [canonical[row][column][action] for column in range(columns)]
                for row in range(rows)
            ]
            for action in range(depth)
        ]
    return _reshape_flat(numeric_values, shape)


def parse_matrix_from_gnn(gnn_text: str) -> Dict[str, Any]:
    """Parse editable matrices, state spaces, connections, and metadata."""
    text = str(gnn_text or "")
    state_spaces = _parse_state_spaces(text)
    parameter_block = _section_content(text, "InitialParameterization")
    matrices: Dict[str, Dict[str, Any]] = {}

    for name, defaults in _MATRIX_DEFAULTS.items():
        declaration = state_spaces.get(name)
        declared = declaration is not None and bool(declaration.get("shape"))
        shape = (
            list(declaration["shape"])
            if declaration is not None and declared
            else list(defaults["shape"])
        )
        description = (
            declaration.get("description") if declaration else None
        ) or defaults["description"]
        matrices[name] = {
            "shape": shape,
            "values": _parse_parameter_values(parameter_block, name, shape)
            if declared
            else None,
            "description": description,
            "declared": declared,
        }

    # Undeclared optional vectors get useful display sizes without becoming
    # part of a later export (``declared`` remains false).
    if matrices["A"]["declared"] and len(matrices["A"]["shape"]) == 2:
        observations, states = matrices["A"]["shape"]
        if not matrices["C"]["declared"]:
            matrices["C"]["shape"] = [observations]
        if not matrices["D"]["declared"]:
            matrices["D"]["shape"] = [states]

    return {
        "matrices": matrices,
        "state_spaces": state_spaces,
        "connections": _parse_connections(text),
        "metadata": _parse_metadata(text),
    }


def _zeros(shape: Sequence[int]) -> Any:
    if len(shape) == 1:
        return [0.0 for _ in range(shape[0])]
    return [_zeros(shape[1:]) for _ in range(shape[0])]


def create_matrix_from_gnn(gnn_text: str) -> Dict[str, Any]:
    """Create the GUI's visual representation without discarding real values."""
    parsed = parse_matrix_from_gnn(gnn_text)
    visual_matrices: Dict[str, Dict[str, Any]] = {}

    for name, info in parsed["matrices"].items():
        shape = info["shape"]
        values = copy.deepcopy(info["values"])
        common = {
            "description": info["description"],
            "editable": True,
            "declared": info["declared"],
        }
        if len(shape) == 1:
            visual_matrices[name] = {
                **common,
                "type": "vector",
                "size": shape[0],
                "values": values if values is not None else _zeros(shape),
            }
        elif len(shape) == 2:
            rows, columns = shape
            visual_matrices[name] = {
                **common,
                "type": "matrix",
                "rows": rows,
                "cols": columns,
                "values": values if values is not None else _zeros(shape),
            }
        elif len(shape) == 3:
            rows, columns, depth = shape
            tensor_shape = [depth, rows, columns]
            visual_matrices[name] = {
                **common,
                "type": "tensor",
                "depth": depth,
                "rows": rows,
                "cols": columns,
                "values": values if values is not None else _zeros(tensor_shape),
                "current_slice": 0,
            }

    return {
        "visual_matrices": visual_matrices,
        "state_spaces": parsed["state_spaces"],
        "connections": parsed["connections"],
        "metadata": parsed["metadata"],
    }


def _finite_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Matrix values must be numeric, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Matrix values must be finite, got {value!r}")
    return number


def _format_parameter(name: str, matrix: Dict[str, Any]) -> str:
    matrix_type = matrix.get("type")
    if matrix_type == "vector":
        values = [_finite_float(value) for value in matrix.get("values", [])]
        return f"{name}={{(" + ", ".join(f"{value:.6g}" for value in values) + ")}"
    if matrix_type == "matrix":
        rows = [
            "(" + ", ".join(f"{_finite_float(value):.6g}" for value in row) + ")"
            for row in matrix.get("values", [])
        ]
        return f"{name}={{\n  " + ",\n  ".join(rows) + "\n}"
    if matrix_type == "tensor":
        tensor_values = matrix.get("values", [])
        if name == "B":
            # Invert the named transition tensor's UI representation only.
            # Other tensors retain their literal axis order.
            tensor_values = [
                [
                    [
                        tensor_values[action][row][column]
                        for action in range(matrix["depth"])
                    ]
                    for column in range(matrix["cols"])
                ]
                for row in range(matrix["rows"])
            ]
        slices: List[str] = []
        for slice_data in tensor_values:
            rows = [
                "(" + ", ".join(f"{_finite_float(value):.6g}" for value in row) + ")"
                for row in slice_data
            ]
            slices.append("( " + ", ".join(rows) + " )")
        return f"{name}={{\n  " + ",\n  ".join(slices) + "\n}"
    raise ValueError(f"Unsupported matrix type for {name}: {matrix_type!r}")


def _declaration_dimensions(matrix: Dict[str, Any]) -> str:
    matrix_type = matrix.get("type")
    if matrix_type == "vector":
        return str(matrix["size"])
    if matrix_type == "matrix":
        return f"{matrix['rows']},{matrix['cols']}"
    if matrix_type == "tensor":
        return f"{matrix['rows']},{matrix['cols']},{matrix['depth']}"
    raise ValueError(f"Unsupported matrix type: {matrix_type!r}")


def _replace_declaration(section: str, name: str, dimensions: str) -> str:
    pattern = re.compile(rf"(?m)^(\s*{re.escape(name)}\s*)\[[^\]]+\]")
    return pattern.sub(rf"\g<1>[{dimensions},type=float]", section, count=1)


def _replace_assignment(block: str, name: str, assignment: str) -> tuple[str, bool]:
    span = _braced_assignment_span(block, name)
    if span is None:
        return block, False
    return block[: span[0]] + assignment + block[span[1] :], True


def update_gnn_from_matrix(visual_data: Dict[str, Any], template: str) -> str:
    """Update editable declarations and assignments while preserving the model."""
    errors = validate_visual_matrix_dimensions(visual_data)
    if errors:
        raise ValueError("; ".join(errors))

    matrices = visual_data.get("visual_matrices", {})
    active = {
        name: matrix
        for name, matrix in matrices.items()
        if isinstance(matrix, dict) and matrix.get("declared", True)
    }
    if not active:
        return template

    updated = template
    state_match = re.search(
        r"(?ms)(^##[ \t]+StateSpaceBlock[ \t]*\r?\n)(.*?)(?=^##[ \t]+|\Z)",
        updated,
    )
    if state_match:
        state_block = state_match.group(2)
        for name, matrix in active.items():
            state_block = _replace_declaration(
                state_block, name, _declaration_dimensions(matrix)
            )
        updated = (
            updated[: state_match.start(2)]
            + state_block
            + updated[state_match.end(2) :]
        )

    parameter_match = re.search(
        r"(?ms)(^##[ \t]+InitialParameterization[ \t]*\r?\n)(.*?)(?=^##[ \t]+|\Z)",
        updated,
    )
    assignments = {
        name: _format_parameter(name, matrix) for name, matrix in active.items()
    }
    if parameter_match:
        parameter_block = parameter_match.group(2)
        missing: List[str] = []
        for name, assignment in assignments.items():
            parameter_block, replaced = _replace_assignment(
                parameter_block, name, assignment
            )
            if not replaced:
                missing.append(assignment)
        if missing:
            parameter_block = (
                parameter_block.rstrip() + "\n\n" + "\n\n".join(missing) + "\n"
            )
        updated = (
            updated[: parameter_match.start(2)]
            + parameter_block
            + updated[parameter_match.end(2) :]
        )
    else:
        parameter_section = (
            "\n\n## InitialParameterization\n"
            + "\n\n".join(assignments.values())
            + "\n"
        )
        footer_match = re.search(r"(?m)^##[ \t]+Footer\b", updated)
        insertion = footer_match.start() if footer_match else len(updated)
        updated = (
            updated[:insertion].rstrip()
            + parameter_section
            + "\n"
            + updated[insertion:]
        )

    return updated


def _parse_connections(gnn_text: str) -> List[str]:
    """Parse connection lines without interpreting their operators."""
    connections: List[str] = []
    for raw_line in _section_content(str(gnn_text or ""), "Connections").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            connections.append(line)
    return connections


def _parse_metadata(gnn_text: str) -> Dict[str, str]:
    """Parse model name and annotation sections."""
    metadata: Dict[str, str] = {}
    model_name = _section_content(str(gnn_text or ""), "ModelName").strip()
    annotation = _section_content(str(gnn_text or ""), "ModelAnnotation").strip()
    if model_name:
        metadata["model_name"] = model_name
    if annotation:
        metadata["annotation"] = annotation
    return metadata


def _value_shape_errors(name: str, matrix: Dict[str, Any]) -> List[str]:
    matrix_type = matrix.get("type")
    values = matrix.get("values")
    if matrix_type == "vector":
        size = matrix.get("size")
        if not isinstance(size, int) or size < 1:
            return [f"{name} vector size must be a positive integer"]
        if not isinstance(values, list) or len(values) != size:
            return [f"{name} vector values must contain {size} entries"]
        candidates = values
    elif matrix_type == "matrix":
        rows, columns = matrix.get("rows"), matrix.get("cols")
        if (
            not isinstance(rows, int)
            or rows < 1
            or not isinstance(columns, int)
            or columns < 1
        ):
            return [f"{name} matrix dimensions must be positive integers"]
        if (
            not isinstance(values, list)
            or len(values) != rows
            or any(not isinstance(row, list) or len(row) != columns for row in values)
        ):
            return [f"{name} matrix values must have shape {rows}x{columns}"]
        candidates = [value for row in values for value in row]
    elif matrix_type == "tensor":
        depth = matrix.get("depth")
        rows = matrix.get("rows")
        columns = matrix.get("cols")
        if any(
            not isinstance(value, int) or value < 1 for value in (depth, rows, columns)
        ):
            return [f"{name} tensor dimensions must be positive integers"]
        if (
            not isinstance(values, list)
            or len(values) != depth
            or any(
                not isinstance(slice_data, list)
                or len(slice_data) != rows
                or any(
                    not isinstance(row, list) or len(row) != columns
                    for row in slice_data
                )
                for slice_data in values
            )
        ):
            return [
                f"{name} tensor values must have {depth} slices of shape {rows}x{columns}"
            ]
        candidates = [
            value for slice_data in values for row in slice_data for value in row
        ]
    else:
        return [f"{name} has unsupported matrix type {matrix_type!r}"]

    try:
        for value in candidates:
            _finite_float(value)
    except ValueError as exc:
        return [f"{name}: {exc}"]
    return []


def validate_visual_matrix_dimensions(visual_data: Dict[str, Any]) -> List[str]:
    """Validate shapes, finite values, and POMDP cross-matrix dimensions."""
    matrices = visual_data.get("visual_matrices", {})
    if not isinstance(matrices, dict):
        return ["visual_matrices must be a mapping"]

    active = {
        name: matrix
        for name, matrix in matrices.items()
        if isinstance(matrix, dict) and matrix.get("declared", True)
    }
    errors: List[str] = []
    for name, matrix in active.items():
        errors.extend(_value_shape_errors(name, matrix))

    a_matrix = active.get("A")
    c_vector = active.get("C")
    d_vector = active.get("D")
    b_matrix = active.get("B")
    if a_matrix and a_matrix.get("type") == "matrix":
        if (
            c_vector
            and c_vector.get("type") == "vector"
            and a_matrix.get("rows") != c_vector.get("size")
        ):
            errors.append(
                f"A matrix rows ({a_matrix.get('rows')}) must match C vector size ({c_vector.get('size')})"
            )
        if (
            d_vector
            and d_vector.get("type") == "vector"
            and a_matrix.get("cols") != d_vector.get("size")
        ):
            errors.append(
                f"A matrix columns ({a_matrix.get('cols')}) must match D vector size ({d_vector.get('size')})"
            )
    if b_matrix and d_vector and d_vector.get("type") == "vector":
        state_size = d_vector.get("size")
        if b_matrix.get("type") in {"matrix", "tensor"}:
            if b_matrix.get("rows") != state_size:
                errors.append(
                    f"B matrix rows ({b_matrix.get('rows')}) must match D vector size ({state_size})"
                )
            if b_matrix.get("cols") != state_size:
                errors.append(
                    f"B matrix columns ({b_matrix.get('cols')}) must match D vector size ({state_size})"
                )

    state_spaces = visual_data.get("state_spaces", {})
    if isinstance(state_spaces, dict):
        for variable, matrix_name in (("s", "D"), ("o", "C")):
            declaration = state_spaces.get(variable)
            linked_matrix = active.get(matrix_name)
            shape = declaration.get("shape") if isinstance(declaration, dict) else None
            if shape and linked_matrix and linked_matrix.get("size") != shape[0]:
                errors.append(
                    f"{matrix_name} vector size ({linked_matrix.get('size')}) must match {variable} size ({shape[0]})"
                )

    return errors
