"""
Temporal Logic Parsers for GNN

This module provides parsing capabilities for temporal logic specifications:
- TLA+ (Temporal Logic of Actions)
- Agda (Dependently typed functional programming language)

Author: @docxology
Date: 2025-01-11
License: MIT
"""

import logging
import re
from typing import Any, ClassVar, List, Optional

logger = logging.getLogger(__name__)

from .common import (
    BaseGNNParser,
    Connection,
    ConnectionType,
    DataType,
    Equation,
    GNNInternalRepresentation,
    Parameter,
    ParseResult,
    TimeSpecification,
    Variable,
    VariableType,
)


class TLAParser(BaseGNNParser):
    """Parser for TLA+ (Temporal Logic of Actions) specifications."""

    # Regex patterns locating embedded MODEL_DATA JSON in TLA+ comments
    EMBEDDED_JSON_PATTERNS: ClassVar[list[str]] = [
        r"\\\*\s*MODEL_DATA:\s*(\{.+\})",  # \* MODEL_DATA: {...}
        r"\(\*\s*MODEL_DATA:\s*(\{.+?\})\s*\*\)",  # (* MODEL_DATA: {...} *)
    ]

    def __init__(self) -> None:
        """Initialize the instance."""
        super().__init__()
        self.module_pattern = re.compile(r"MODULE\s+(\w+)", re.IGNORECASE)
        self.variable_pattern = re.compile(r"VARIABLES?\s+([\w\s,]+)", re.IGNORECASE)
        self.constant_pattern = re.compile(r"CONSTANTS?\s+([\w\s,]+)", re.IGNORECASE)
        self.operator_pattern = re.compile(
            r"(\w+)\s*==\s*(.+?)(?=\n\w+\s*==|\n\n|$)", re.DOTALL
        )
        self.temporal_pattern = re.compile(
            r"(ALWAYS|EVENTUALLY|NEXT|LEADS_TO|UNTIL)\s*\(([^)]+)\)", re.IGNORECASE
        )

    def get_supported_extensions(self) -> List[str]:
        """Return supported extensions."""
        return [".tla"]

    def parse_file(self, file_path: str) -> ParseResult:
        """Parse file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return self.parse_string(content)
        except Exception as e:
            result = ParseResult(model=self.create_empty_model())
            result.add_error(f"Failed to read TLA+ file: {e}")
            return result

    def parse_string(self, content: str) -> ParseResult:
        # First try to extract embedded JSON model data
        """Parse string."""
        embedded_data = self._extract_embedded_json_data(content)
        if embedded_data:
            result = ParseResult(model=self.create_empty_model())
            return self._parse_from_embedded_data(embedded_data, result)

        result = ParseResult(model=self.create_empty_model())

        try:
            # Parse module name
            module_match = self.module_pattern.search(content)
            if module_match:
                result.model.model_name = module_match.group(1)
            else:
                result.model.model_name = "TLAModel"

            # Parse variables
            var_match = self.variable_pattern.search(content)
            if var_match:
                var_names = [name.strip() for name in var_match.group(1).split(",")]
                for var_name in var_names:
                    if var_name:
                        variable = Variable(
                            name=var_name,
                            var_type=self._infer_variable_type(var_name),
                            dimensions=[1],
                            data_type=DataType.CATEGORICAL,
                            description="TLA+ variable",
                        )
                        result.model.variables.append(variable)

            # Parse constants as parameters
            const_match = self.constant_pattern.search(content)
            if const_match:
                const_names = [name.strip() for name in const_match.group(1).split(",")]
                for const_name in const_names:
                    if const_name:
                        parameter = Parameter(
                            name=const_name, value=None, description="TLA+ constant"
                        )
                        result.model.parameters.append(parameter)

            # Parse operators as equations
            for match in self.operator_pattern.finditer(content):
                op_name = match.group(1)
                op_def = match.group(2).strip()

                equation = Equation(
                    label=op_name,
                    content=op_def,
                    format="tla",
                    description="TLA+ operator definition",
                )
                result.model.equations.append(equation)

                # Extract variable dependencies
                connections = self._extract_tla_connections(
                    op_name, op_def, result.model.variables
                )
                result.model.connections.extend(connections)

            # Parse temporal formulas
            for match in self.temporal_pattern.finditer(content):
                temporal_op = match.group(1)
                formula = match.group(2)

                equation = Equation(
                    label=f"temporal_{temporal_op.lower()}_{len(result.model.equations)}",
                    content=f"{temporal_op}({formula})",
                    format="tla_temporal",
                    description=f"TLA+ temporal formula using {temporal_op}",
                )
                result.model.equations.append(equation)

            # Set time specification for TLA+ (always dynamic)
            result.model.time_specification = TimeSpecification(
                time_type="Dynamic", discretization="DiscreteTime", horizon="Unbounded"
            )

            result.model.annotation = "Parsed from TLA+ temporal logic specification"

        except Exception as e:
            result.add_error(f"Parsing error: {e}")

        return result

    def _extract_tla_connections(
        self, op_name: str, op_def: str, variables: List[Variable]
    ) -> List[Connection]:
        """Extract variable dependencies from TLA+ operator definition."""
        connections: list[Any] = []
        var_names = {var.name for var in variables}

        # Find variable references in the operator definition
        var_pattern = re.compile(r"\b([a-zA-Z_]\w*)\b")
        referenced_vars: list[Any] = []

        for match in var_pattern.finditer(op_def):
            var_ref = match.group(1)
            if var_ref in var_names and var_ref != op_name:
                referenced_vars.append(var_ref)

        # Create connections from referenced variables to this operator
        for var_ref in set(referenced_vars):
            connection = Connection(
                source_variables=[var_ref],
                target_variables=[op_name],
                connection_type=ConnectionType.DIRECTED,
                description=f"TLA+ operator dependency: {var_ref} -> {op_name}",
            )
            connections.append(connection)

        return connections

    def _infer_variable_type(self, name: str) -> VariableType:
        """Infer variable type from name."""
        name_lower = name.lower()
        if "state" in name_lower or "s_" in name_lower:
            return VariableType.HIDDEN_STATE
        elif "observation" in name_lower or "obs" in name_lower or "o_" in name_lower:
            return VariableType.OBSERVATION
        elif "action" in name_lower or "u_" in name_lower:
            return VariableType.ACTION
        elif "policy" in name_lower or "pi_" in name_lower:
            return VariableType.POLICY
        else:
            return VariableType.HIDDEN_STATE


class AgdaParser(BaseGNNParser):
    """Parser for Agda dependently typed functional programming language."""

    # Regex patterns locating embedded MODEL_DATA JSON in Agda comments
    EMBEDDED_JSON_PATTERNS: ClassVar[list[str]] = [
        r"--\s*MODEL_DATA:\s*(\{.+\})",  # -- MODEL_DATA: {...}
        r"\{-\s*MODEL_DATA:\s*(\{.+?\})\s*-\}",  # {- MODEL_DATA: {...} -}
    ]

    def __init__(self) -> None:
        """Initialize the instance."""
        super().__init__()
        self.module_pattern = re.compile(r"module\s+([\w.]+)\s+where", re.IGNORECASE)
        self.data_pattern = re.compile(
            r"data\s+(\w+).*?where\s*(.*?)(?=\ndata|\n\w+\s*:|\Z)",
            re.DOTALL | re.IGNORECASE,
        )
        self.function_pattern = re.compile(
            r"^[ \t]*(\w+)[ \t]*:[ \t]*([^=\n]+)"
            r"(?:\n[ \t]*\1\b[ \t]*(.+?))?"
            r"(?=\n(?:[ \t]*\n)*[ \t]*(?:\w+[ \t]*:|data[ \t]|postulate\b)|\s*\Z)",
            re.DOTALL | re.MULTILINE,
        )
        self.import_pattern = re.compile(r"import\s+([\w.]+)", re.IGNORECASE)

    def get_supported_extensions(self) -> List[str]:
        """Return supported extensions."""
        return [".agda"]

    def parse_file(self, file_path: str) -> ParseResult:
        """Parse file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return self.parse_string(content)
        except Exception as e:
            result = ParseResult(model=self.create_empty_model())
            result.add_error(f"Failed to read Agda file: {e}")
            return result

    def parse_string(self, content: str) -> ParseResult:
        # First try to extract embedded JSON model data
        """Parse string."""
        embedded_data = self._extract_embedded_json_data(content)
        if embedded_data:
            result = ParseResult(model=self.create_empty_model())
            return self._parse_from_embedded_data(embedded_data, result)

        result = ParseResult(model=self.create_empty_model())

        try:
            # Parse module name
            module_match = self.module_pattern.search(content)
            if module_match:
                result.model.model_name = module_match.group(1).replace(".", "_")
            else:
                result.model.model_name = "AgdaModel"

            # Parse data types (variables)
            for match in self.data_pattern.finditer(content):
                data_name = match.group(1)
                data_body = match.group(2) if match.group(2) else ""

                variable = Variable(
                    name=data_name,
                    var_type=self._infer_variable_type(data_name),
                    dimensions=self._parse_agda_dimensions(data_body),
                    data_type=self._infer_agda_data_type(data_body),
                    description=f"Agda data type: {data_body.strip()[:50]}...",
                )
                result.model.variables.append(variable)

            # Parse function definitions
            for match in self.function_pattern.finditer(content):
                func_name = match.group(1)
                func_type = match.group(2).strip()
                func_body = match.group(3).strip() if match.group(3) else ""

                if self._is_agda_parameter(func_type, func_body):
                    # Create parameter
                    parameter = Parameter(
                        name=func_name,
                        value=self._extract_agda_value(func_body),
                        type_hint=func_type,
                        description=f"Agda definition: {func_type}",
                    )
                    result.model.parameters.append(parameter)
                else:
                    # Create equation
                    equation = Equation(
                        label=func_name,
                        content=f"{func_name} : {func_type}"
                        + (f"\n{func_name} {func_body}" if func_body else ""),
                        format="agda",
                        description=f"Agda function with type {func_type}",
                    )
                    result.model.equations.append(equation)

                    # Extract dependencies
                    connections = self._extract_agda_connections(
                        func_name, func_type, func_body
                    )
                    result.model.connections.extend(connections)

            result.model.annotation = "Parsed from Agda dependently typed specification"

        except Exception as e:
            result.add_error(f"Parsing error: {e}")

        return result

    def _parse_agda_dimensions(self, data_body: str) -> List[int]:
        """Parse dimensions from Agda data type body."""
        if not data_body.strip():
            return [1]

        # Count constructors as dimension
        constructors = re.findall(r"\n\s*(\w+)", data_body)
        return [len(constructors)] if constructors else [1]

    def _infer_agda_data_type(self, data_body: str) -> DataType:
        """Infer data type from Agda data body."""
        if "ℕ" in data_body or "Nat" in data_body:
            return DataType.INTEGER
        elif "Bool" in data_body:
            return DataType.BINARY
        elif "ℝ" in data_body or "Real" in data_body:
            return DataType.FLOAT
        else:
            return DataType.CATEGORICAL

    def _is_agda_parameter(self, func_type: str, func_body: str) -> bool:
        """Check if function definition represents a parameter."""
        # Simple heuristics for parameter detection
        return bool(func_body) and (
            func_body.replace(" ", "").isdigit()
            or any(const in func_body for const in ["true", "false", "zero", "suc"])
            or func_type in ["ℕ", "Nat", "Bool", "ℝ"]
        )

    def _extract_agda_value(self, func_body: str) -> Any:
        """Extract value from Agda function body."""
        body_clean = func_body.strip()
        if body_clean == "true":
            return True
        elif body_clean == "false":
            return False
        elif body_clean.isdigit():
            return int(body_clean)
        elif body_clean == "zero":
            return 0
        elif body_clean.startswith("suc"):
            # Count successor applications
            return body_clean.count("suc")
        else:
            return body_clean

    def _extract_agda_connections(
        self, func_name: str, func_type: str, func_body: str
    ) -> List[Connection]:
        """Extract dependencies from Agda function."""
        connections: list[Any] = []

        # Simple pattern to find type/function references
        ref_pattern = re.compile(r"\b([A-Z][a-zA-Z0-9]*)\b")

        # Check type signature for dependencies
        type_refs = ref_pattern.findall(func_type)
        for ref in type_refs:
            if ref != func_name:
                connection = Connection(
                    source_variables=[ref],
                    target_variables=[func_name],
                    connection_type=ConnectionType.DIRECTED,
                    description=f"Agda type dependency: {ref} -> {func_name}",
                )
                connections.append(connection)

        return connections

    def _infer_variable_type(self, name: str) -> VariableType:
        """Infer variable type from name."""
        name_lower = name.lower()
        if "state" in name_lower:
            return VariableType.HIDDEN_STATE
        elif "observation" in name_lower or "obs" in name_lower:
            return VariableType.OBSERVATION
        elif "action" in name_lower:
            return VariableType.ACTION
        else:
            return VariableType.HIDDEN_STATE
