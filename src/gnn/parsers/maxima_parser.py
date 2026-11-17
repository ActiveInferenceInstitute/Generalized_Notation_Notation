"""
Maxima Parser for GNN

This module provides parsing capabilities for Maxima symbolic computation
specifications of GNN models.

Author: @docxology
Date: 2025-01-11
"""

import re
from typing import Any, ClassVar, Dict, List, Optional, cast

from .common import (
    BaseGNNParser,
    Connection,
    ConnectionType,
    DataType,
    Equation,
    Parameter,
    ParseResult,
    Variable,
    VariableType,
)


class MaximaParser(BaseGNNParser):
    """Parser for Maxima symbolic computation specifications with embedded data support."""

    def __init__(self) -> None:
        """Initialize the instance."""
        super().__init__()
        self.assignment_pattern = re.compile(r"(\w+)\s*:\s*(.+?);", re.MULTILINE)
        self.function_pattern = re.compile(
            r"(\w+)\s*\(\s*([^)]*)\s*\)\s*:=\s*(.+?);", re.MULTILINE
        )
        self.matrix_pattern = re.compile(
            r"matrix\s*\(\s*(.+?)\s*\)", re.IGNORECASE | re.DOTALL
        )
        self.solve_pattern = re.compile(
            r"solve\s*\(\s*(.+?)\s*,\s*(.+?)\s*\)", re.IGNORECASE
        )

    EMBEDDED_JSON_PATTERNS: ClassVar[list[str]] = [
        r"/\*\s*MODEL_DATA:\s*(\{.+?\})\s*\*/",  # /* MODEL_DATA: {...} */
    ]
    EMBEDDED_LENIENT_MODEL_NAME: ClassVar[str] = "MaximaGNNModel"

    def get_supported_extensions(self) -> List[str]:
        """Get file extensions supported by this parser."""
        return [".mac", ".max", ".wxm"]

    def parse_file(self, file_path: str) -> ParseResult:
        """Parse a Maxima file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
            return self.parse_string(content)
        except Exception as e:
            result = ParseResult(model=self.create_empty_model())
            result.add_error(f"Failed to read file: {e}")
            return result

    def parse_string(self, content: str) -> ParseResult:
        """Parse Maxima content string."""
        result = ParseResult(model=self.create_empty_model())

        # First, try to extract embedded JSON data for perfect round-trip
        embedded_data = self._extract_embedded_json_data(content)
        if embedded_data:
            return self._parse_embedded_data_lenient(embedded_data, result)

        # Recovery to standard parsing
        try:
            # Extract model name from comments or filename
            result.model.model_name = self._extract_model_name(content)

            # Parse assignments (parameters)
            for match in self.assignment_pattern.finditer(content):
                var_name = match.group(1)
                var_value = match.group(2).strip()

                if self._is_matrix_definition(var_value):
                    # Create variable for matrix
                    dims = self._parse_matrix_dimensions(var_value)
                    variable = Variable(
                        name=var_name,
                        var_type=self._infer_variable_type(var_name),
                        dimensions=dims,
                        data_type=DataType.FLOAT,
                        description=f"Maxima matrix assignment: {var_value[:50]}...",
                    )
                    result.model.variables.append(variable)
                else:
                    # Create parameter
                    parameter = Parameter(
                        name=var_name,
                        value=self._parse_maxima_value(var_value),
                        description=f"Maxima assignment: {var_value}",
                    )
                    result.model.parameters.append(parameter)

            # Parse function definitions
            for match in self.function_pattern.finditer(content):
                func_name = match.group(1)
                func_args = match.group(2)
                func_body = match.group(3)

                equation = Equation(
                    label=func_name,
                    content=f"{func_name}({func_args}) := {func_body}",
                    format="maxima",
                    description=f"Maxima function with args: {func_args}",
                )
                result.model.equations.append(equation)

                # Extract connections from function dependencies
                connections = self._extract_function_connections(
                    func_name, func_args, func_body
                )
                result.model.connections.extend(connections)

            # Parse solve statements as equations
            for match in self.solve_pattern.finditer(content):
                equation_expr = match.group(1)
                variables = match.group(2)

                equation = Equation(
                    label=f"solve_{len(result.model.equations)}",
                    content=f"solve({equation_expr}, {variables})",
                    format="maxima",
                    description="Maxima solve statement",
                )
                result.model.equations.append(equation)

            result.model.annotation = "Parsed from Maxima symbolic computation"

        except Exception as e:
            result.add_error(f"Parsing error: {e}")

        return result

    def _extract_model_name(self, content: str) -> str:
        """Extract model name from comments."""
        comment_patterns: list[Any] = [
            re.compile(r"/\*\s*Model:\s*(\w+)", re.IGNORECASE),
            re.compile(r"/\*\s*(\w+)\s*Model", re.IGNORECASE),
            re.compile(r"/\*.*?(\w+).*?\*/", re.IGNORECASE | re.DOTALL),
        ]

        for pattern in comment_patterns:
            match = pattern.search(content)
            if match:
                return cast("str", match.group(1))

        return "MaximaModel"

    def _is_matrix_definition(self, value: str) -> bool:
        """Check if value is a matrix definition."""
        return (
            "matrix(" in value.lower()
            or "[" in value
            and "]" in value
            or "transpose(" in value.lower()
        )

    def _parse_matrix_dimensions(self, matrix_def: str) -> List[int]:
        """Parse matrix dimensions from definition."""
        # Simple heuristic for common matrix patterns
        if "matrix(" in matrix_def.lower():
            # Count comma-separated rows
            matrix_match = self.matrix_pattern.search(matrix_def)
            if matrix_match:
                content = matrix_match.group(1)
                rows = content.split("],[")
                if len(rows) > 1:
                    cols = len(rows[0].split(","))
                    return [len(rows), cols]

        # Default assumption
        return [1, 1]

    def _parse_maxima_value(self, value: str) -> Any:
        """Parse Maxima value to Python type."""
        value_clean = value.strip()

        # Try to evaluate simple numeric expressions
        try:
            if value_clean.replace(".", "").replace("-", "").isdigit():
                return float(value_clean) if "." in value_clean else int(value_clean)
            elif "%pi" in value_clean:
                return f"π-expression: {value_clean}"
            elif "%e" in value_clean:
                return f"e-expression: {value_clean}"
            else:
                return value_clean
        except (ValueError, TypeError):
            return value_clean

    def _infer_variable_type(self, name: str) -> VariableType:
        """Infer variable type from name."""
        name_lower = name.lower()
        if any(prefix in name_lower for prefix in ["a_", "likelihood"]):
            return VariableType.LIKELIHOOD_MATRIX
        elif any(prefix in name_lower for prefix in ["b_", "transition"]):
            return VariableType.TRANSITION_MATRIX
        elif any(prefix in name_lower for prefix in ["c_", "preference"]):
            return VariableType.PREFERENCE_VECTOR
        elif any(prefix in name_lower for prefix in ["d_", "prior"]):
            return VariableType.PRIOR_VECTOR
        elif "s_" in name_lower or "state" in name_lower:
            return VariableType.HIDDEN_STATE
        elif "o_" in name_lower or "obs" in name_lower:
            return VariableType.OBSERVATION
        elif "u_" in name_lower or "action" in name_lower:
            return VariableType.ACTION
        else:
            return VariableType.HIDDEN_STATE

    def _extract_function_connections(
        self, func_name: str, args: str, body: str
    ) -> List[Connection]:
        """Extract connections from function dependencies."""
        connections: list[Any] = []

        # Parse function arguments as source variables
        arg_vars = [arg.strip() for arg in args.split(",") if arg.strip()]

        # Look for variable references in function body
        var_pattern = re.compile(r"\b([a-zA-Z_]\w*)\b")
        var_pattern.findall(body)

        # Create connections from arguments to function
        for arg_var in arg_vars:
            if arg_var and arg_var != func_name:
                connection = Connection(
                    source_variables=[arg_var],
                    target_variables=[func_name],
                    connection_type=ConnectionType.DIRECTED,
                    description=f"Maxima function dependency: {arg_var} -> {func_name}",
                )
                connections.append(connection)

        return connections
