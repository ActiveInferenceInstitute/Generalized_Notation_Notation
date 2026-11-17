"""
Coq Parser for GNN Formal Verification

This module provides parsing capabilities for Coq files that specify
GNN models using formal verification constructs.

Author: @docxology
Date: 2025-01-11
License: MIT
"""

import logging
import re
from typing import Any, ClassVar, List

from .common import (
    BaseGNNParser,
    DataType,
    GNNInternalRepresentation,
    Parameter,
    ParseResult,
    Variable,
    VariableType,
)

logger = logging.getLogger(__name__)


class CoqGNNParser(BaseGNNParser):
    """Parser for Coq formal verification specifications."""

    EMBEDDED_JSON_PATTERNS: ClassVar[list[str]] = [
        r"\(\*\s*MODEL_DATA:\s*(\{.+?\})\s*\*\)",  # (* MODEL_DATA: {...} *)
    ]

    def __init__(self) -> None:
        """Initialize the Coq parser."""
        super().__init__()
        self.require_pattern = re.compile(r"Require\s+Import\s+([^\n.]+)\.")
        self.module_pattern = re.compile(r"Module\s+(\w+)\.")
        self.parameter_pattern = re.compile(r"Parameter\s+(\w+)\s*:\s*([^.]+)\.")
        self.definition_pattern = re.compile(
            r"Definition\s+(\w+)[^:]*:\s*([^:=]+)(?::=)?"
        )
        self.variable_pattern = re.compile(r"Variable\s+(\w+)\s*:\s*([^.]+)\.")
        self.theorem_pattern = re.compile(r"Theorem\s+(\w+)[^:]*:\s*([^.]+)\.")

    def parse_file(self, file_path: str) -> ParseResult:
        """Parse a Coq file containing GNN formal specifications."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            return self.parse_string(content)

        except Exception as e:
            logger.error(f"Error parsing Coq file {file_path}: {e}")
            result = ParseResult(
                model=self.create_empty_model("Failed Coq Parse"), success=False
            )
            result.add_error(f"Failed to parse Coq file: {e}")
            return result

    def parse_string(self, content: str) -> ParseResult:
        """Parse Coq content from string."""
        # First try to extract embedded JSON model data
        embedded_data = self._extract_embedded_json_data(content)
        if embedded_data:
            result = ParseResult(model=self.create_empty_model())
            return self._parse_from_embedded_data(embedded_data, result)

        try:
            model = self._parse_coq_content(content)
            return ParseResult(model=model, success=True)

        except Exception as e:
            logger.error(f"Error parsing Coq content: {e}")
            result = ParseResult(
                model=self.create_empty_model("Failed Coq Parse"), success=False
            )
            result.add_error(f"Failed to parse Coq content: {e}")
            return result

    def _parse_coq_content(self, content: str) -> GNNInternalRepresentation:
        """Parse the main Coq content."""
        # Extract model name from module
        model_name = self._extract_model_name(content)

        # Create base model
        model = GNNInternalRepresentation(
            model_name=model_name,
            annotation="Parsed from Coq formal verification specification",
        )

        # Parse requires
        self._parse_requires(content, model)

        # Parse parameters and variables
        self._parse_parameters(content, model)
        self._parse_variables(content, model)
        self._parse_definitions(content, model)

        # Parse theorems as constraints
        self._parse_theorems(content, model)

        return model

    def _extract_model_name(self, content: str) -> str:
        """Extract model name from Coq content."""
        module_match = self.module_pattern.search(content)
        if module_match:
            return module_match.group(1)

        return "CoqGNNModel"

    def _parse_requires(self, content: str, model: GNNInternalRepresentation) -> Any:
        """Parse Require Import statements."""
        requires = self.require_pattern.findall(content)
        model.extensions["coq_requires"] = requires

        # Check for relevant libraries
        has_reals = any("Reals" in req for req in requires)
        has_lists = any("Lists" in req for req in requires)

        model.extensions["uses_reals"] = has_reals
        model.extensions["uses_lists"] = has_lists

    def _parse_parameters(self, content: str, model: GNNInternalRepresentation) -> Any:
        """Parse Parameter declarations."""
        parameter_matches = self.parameter_pattern.findall(content)

        for param_name, param_type in parameter_matches:
            # Infer variable type
            var_type = self._infer_variable_type(param_name, param_type)
            data_type = self._infer_data_type(param_type)

            variable = Variable(
                name=param_name,
                var_type=var_type,
                dimensions=self._infer_dimensions(param_type),
                data_type=data_type,
                description=f"Parameter: {param_type.strip()}",
            )

            model.variables.append(variable)

    def _parse_variables(self, content: str, model: GNNInternalRepresentation) -> Any:
        """Parse Variable declarations."""
        variable_matches = self.variable_pattern.findall(content)

        for var_name, var_type in variable_matches:
            # Skip if already parsed as parameter
            if any(var.name == var_name for var in model.variables):
                continue

            var_type_enum = self._infer_variable_type(var_name, var_type)
            data_type = self._infer_data_type(var_type)

            variable = Variable(
                name=var_name,
                var_type=var_type_enum,
                dimensions=self._infer_dimensions(var_type),
                data_type=data_type,
                description=f"Variable: {var_type.strip()}",
            )

            model.variables.append(variable)

    def _parse_definitions(self, content: str, model: GNNInternalRepresentation) -> Any:
        """Parse Definition statements."""
        def_matches = self.definition_pattern.findall(content)

        for def_name, def_type in def_matches:
            # Skip if already parsed
            if any(var.name == def_name for var in model.variables):
                continue

            var_type = self._infer_variable_type(def_name, def_type)
            data_type = self._infer_data_type(def_type)

            variable = Variable(
                name=def_name,
                var_type=var_type,
                dimensions=self._infer_dimensions(def_type),
                data_type=data_type,
                description=f"Definition: {def_type.strip()}",
            )

            model.variables.append(variable)

    def _parse_theorems(self, content: str, model: GNNInternalRepresentation) -> Any:
        """Parse Theorem statements as model constraints."""
        theorem_matches = self.theorem_pattern.findall(content)

        for theorem_name, theorem_statement in theorem_matches:
            # Store theorem as parameter/constraint
            parameter = Parameter(
                name=f"theorem_{theorem_name}",
                value=theorem_statement.strip(),
                type_hint="theorem",
                description=f"Theorem: {theorem_name}",
            )

            model.parameters.append(parameter)

    def _infer_variable_type(self, name: str, type_def: str) -> VariableType:
        """Infer GNN variable type from Coq name and type."""
        name_lower = name.lower()
        type_lower = type_def.lower()

        if any(keyword in name_lower for keyword in ["state", "hidden", "s_"]):
            return VariableType.HIDDEN_STATE
        elif any(keyword in name_lower for keyword in ["obs", "observation", "o_"]):
            return VariableType.OBSERVATION
        elif any(keyword in name_lower for keyword in ["action", "control", "u_"]):
            return VariableType.ACTION
        elif any(keyword in name_lower for keyword in ["policy", "pi_"]):
            return VariableType.POLICY
        elif "matrix" in type_lower or name_lower in ["a", "a_matrix"]:
            return VariableType.LIKELIHOOD_MATRIX
        elif "matrix" in type_lower or name_lower in ["b", "b_matrix"]:
            return VariableType.TRANSITION_MATRIX
        elif "list" in type_lower and name_lower in ["c", "c_vector"]:
            return VariableType.PREFERENCE_VECTOR
        elif "list" in type_lower and name_lower in ["d", "d_vector"]:
            return VariableType.PRIOR_VECTOR

        return VariableType.HIDDEN_STATE

    def _infer_data_type(self, type_def: str) -> DataType:
        """Infer data type from Coq type definition."""
        type_lower = type_def.lower()

        if any(keyword in type_lower for keyword in ["r", "real"]):
            return DataType.CONTINUOUS
        elif any(keyword in type_lower for keyword in ["nat", "z"]):
            return DataType.INTEGER
        elif "bool" in type_lower:
            return DataType.BINARY
        elif "list" in type_lower:
            return DataType.CATEGORICAL

        return DataType.CONTINUOUS

    def _infer_dimensions(self, type_def: str) -> List[int]:
        """Infer dimensions from Coq type definition."""
        # Simple heuristic for matrix/vector types
        if "matrix" in type_def.lower():
            return [3, 3]  # Default matrix size
        elif "list" in type_def.lower():
            return [5]  # Default vector size

        return []

    def get_supported_extensions(self) -> List[str]:
        """Get supported file extensions."""
        return [".v"]
