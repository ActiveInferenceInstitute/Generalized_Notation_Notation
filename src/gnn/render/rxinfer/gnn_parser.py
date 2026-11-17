"""
Module for parsing GNN files specifically for RxInfer.jl configuration generation.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)


def parse_gnn_section(content: str, section_name: str) -> str:
    """
    Parse a specific section from GNN content.

    Args:
        content: The full GNN file content
        section_name: The section name to extract

    Returns:
        The extracted section content as a string
    """
    pattern = rf"## {section_name}\n(.*?)(?=\n## |$)"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def parse_initial_value(value_str: str) -> Any:
    """
    Parse GNN initial value notation into Python objects.

    Args:
        value_str: String representation of value from GNN file

    Returns:
        The parsed value (string, number, boolean, tuple, etc.)
    """
    value_str = value_str.strip()

    # Boolean
    if value_str.lower() == "true":
        return True
    if value_str.lower() == "false":
        return False

    # String (already a string, just remove quotes if present)
    if value_str.startswith('"') and value_str.endswith('"'):
        return value_str[1:-1]

    # Try parsing as number
    try:
        # Integer
        if "." not in value_str:
            return int(value_str)
        # Float
        return float(value_str)
    except ValueError:
        logger.debug(
            "Value '%s' is not a simple numeric, trying structured formats",
            value_str[:40],
        )

    # Check for tuple/matrix notation {(a,b), (c,d)}
    if value_str.startswith("{") and value_str.endswith("}"):
        inner_content = value_str[1:-1].strip()

        # Check for tuple notation (a, b)
        if inner_content.startswith("(") and inner_content.endswith(")"):
            # Single tuple
            return parse_tuple(inner_content)

        # Check for list of tuples notation (a,b),(c,d)
        if "," in inner_content and "(" in inner_content and ")" in inner_content:
            # Try to split by "),(" which is the separator between tuples
            parts = inner_content.split("),(")
            if len(parts) > 1:
                tuples: list[Any] = []
                for i, part in enumerate(parts):
                    # Clean up outer parentheses
                    if i == 0:
                        part = part[1:] if part.startswith("(") else part
                    if i == len(parts) - 1:
                        part = part[:-1] if part.endswith(")") else part

                    # Parse tuple elements
                    elements = [parse_initial_value(e.strip()) for e in part.split(",")]
                    tuples.append(tuple(elements))
                return tuples

        # Simple tuple inside braces {(a, b)}
        if inner_content.count("(") == 1 and inner_content.count(")") == 1:
            return parse_tuple(inner_content)

    # Default to returning the string as is
    return value_str


def parse_tuple(tuple_str: str) -> Tuple:
    """
    Parse a GNN tuple string into a Python tuple.

    Args:
        tuple_str: String representation of a tuple

    Returns:
        A Python tuple with parsed values
    """
    # Remove outer parentheses if present
    tuple_str = tuple_str.strip()
    if tuple_str.startswith("(") and tuple_str.endswith(")"):
        tuple_str = tuple_str[1:-1]

    # Split by comma and parse each element
    elements = [parse_initial_value(e.strip()) for e in tuple_str.split(",")]
    return tuple(elements)


def parse_initial_parameterization(content: str) -> Dict[str, Any]:
    """
    Parse the InitialParameterization section of a GNN file.

    Args:
        content: The full GNN file content

    Returns:
        Dictionary of parameter names to values
    """
    section = parse_gnn_section(content, "InitialParameterization")
    if not section:
        return {}

    params: dict[Any, Any] = {}
    # Split by lines, ignoring comments
    lines = [
        line.strip()
        for line in section.split("\n")
        if line.strip() and not line.strip().startswith("#")
    ]

    for line in lines:
        # Find parameter assignments (param=value)
        if "=" in line:
            name, value_str = line.split("=", 1)
            name = name.strip()
            params[name] = parse_initial_value(value_str)

    return params


def parse_state_space_block(content: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse the StateSpaceBlock section of a GNN file.

    Args:
        content: The full GNN file content

    Returns:
        Dictionary of variable definitions
    """
    section = parse_gnn_section(content, "StateSpaceBlock")
    if not section:
        return {}

    variables: dict[Any, Any] = {}
    # Split by lines, ignoring comments
    lines = [
        line.strip()
        for line in section.split("\n")
        if line.strip() and not line.strip().startswith("#")
    ]

    for line in lines:
        # Parse variable definitions like: var_name[dim1,dim2,type=type_name]
        match = re.match(r"(\w+)\[([^\]]+)\]\s*(#.*)?", line)
        if match:
            var_name = match.group(1)
            dimensions_type = match.group(2)

            var_info: dict[str, Any] = {"name": var_name}

            # Parse dimensions and type
            if "type=" in dimensions_type:
                dims_part, type_part = dimensions_type.split("type=", 1)
                var_info["type"] = type_part.strip()
                dims = [d.strip() for d in dims_part.strip().rstrip(",").split(",")]
            else:
                dims = [d.strip() for d in dimensions_type.strip().split(",")]
                var_info["type"] = "float"  # Default type

            # Convert dimensions to integers if possible
            var_info["dimensions"] = []
            for dim in dims:
                try:
                    var_info["dimensions"].append(int(dim))
                except ValueError:
                    var_info["dimensions"].append(dim)  # Keep as string if not a number

            variables[var_name] = var_info

    return variables


def parse_model_name(content: str) -> str:
    """
    Parse the ModelName section of a GNN file.

    Args:
        content: The full GNN file content

    Returns:
        The model name as a string
    """
    return parse_gnn_section(content, "ModelName")


def parse_model_parameters(content: str) -> Dict[str, Any]:
    """
    Parse the ModelParameters section of a GNN file.

    Args:
        content: The full GNN file content

    Returns:
        Dictionary of model parameters
    """
    section = parse_gnn_section(content, "ModelParameters")
    if not section:
        return {}

    params: dict[Any, Any] = {}
    # Split by lines, ignoring comments
    lines = [
        line.strip()
        for line in section.split("\n")
        if line.strip() and not line.strip().startswith("#")
    ]

    for line in lines:
        # Find parameter assignments (param=value or param: value)
        if "=" in line:
            name, value_str = line.split("=", 1)
            name = name.strip()
            params[name] = parse_initial_value(value_str)
        elif ":" in line:
            name, value_str = line.split(":", 1)
            name = name.strip()
            params[name] = parse_initial_value(value_str)

    return params


def parse_gnn_file(file_path: Path) -> Dict[str, Any]:
    """
    Parse a GNN file into a structured dictionary.

    Args:
        file_path: Path to the GNN file

    Returns:
        Dictionary containing parsed GNN content
    """
    if not file_path.exists():
        logger.error(f"GNN file not found: {file_path}")
        raise FileNotFoundError(f"GNN file not found: {file_path}")

    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    # Parse the various sections
    model_name = parse_model_name(content)
    initial_params = parse_initial_parameterization(content)
    variables = parse_state_space_block(content)
    model_params = parse_model_parameters(content)

    # Combine into a structured dictionary
    return {
        "model_name": model_name,
        "variables": variables,
        "initial_parameters": initial_params,
        "model_parameters": model_params,
        "raw_content": content,
    }


if __name__ == "__main__":
    # Simple test if run directly
    import sys

    if len(sys.argv) > 1:
        test_file = Path(sys.argv[1])
        result = parse_gnn_file(test_file)
        import json

        print(json.dumps(result, indent=2, default=str))
