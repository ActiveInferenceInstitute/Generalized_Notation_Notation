"""
Lean Parser for GNN Category Theory Proofs

This module provides parsing capabilities for Lean files that specify
GNN models using category theory constructs.

Author: @docxology
Date: 2025-01-11
License: MIT
"""

import logging
import re
from typing import Any, ClassVar, Dict, List, Optional

from .common import (
    BaseGNNParser,
    DataType,
    GNNInternalRepresentation,
    ParseResult,
    Variable,
    VariableType,
)

logger = logging.getLogger(__name__)


class LeanGNNParser(BaseGNNParser):
    """Parser for Lean category theory specifications."""

    def __init__(self) -> None:
        """Initialize the Lean parser."""
        super().__init__()
        self.namespace_pattern = re.compile(r"namespace\s+(\w+)")
        self.structure_pattern = re.compile(r"structure\s+(\w+)\s+where")
        self.import_pattern = re.compile(r"import\s+([^\n]+)")
        self.def_pattern = re.compile(r"def\s+(\w+)[^:]*:\s*([^:=]+)(?::=)?")
        self.variable_pattern = re.compile(r"(\w+)\s*:\s*([^\n]+)")

    EMBEDDED_JSON_PATTERNS: ClassVar[list[str]] = [
        r"--\s*MODEL_DATA:\s*(\{.+\})",  # -- MODEL_DATA: {...}
        r"/\*\s*MODEL_DATA:\s*(\{.+?\})\s*\*/",  # /* MODEL_DATA: {...} */
    ]

    def parse_file(self, file_path: str) -> ParseResult:
        """Parse a Lean file containing GNN category theory specifications."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            return self.parse_string(content)

        except Exception as e:
            logger.error(f"Error parsing Lean file {file_path}: {e}")
            result = ParseResult(
                model=self.create_empty_model("Failed Lean Parse"), success=False
            )
            result.add_error(f"Failed to parse Lean file: {e}")
            return result

    def parse_string(self, content: str) -> ParseResult:
        """Parse Lean content from string."""
        # First try to extract embedded JSON model data
        embedded_data = self._extract_embedded_json_data(content)
        if embedded_data:
            result = ParseResult(model=self.create_empty_model())
            return self._parse_from_embedded_data(embedded_data, result)

        try:
            model = self._parse_lean_content(content)
            return ParseResult(model=model, success=True)

        except Exception as e:
            logger.error(f"Error parsing Lean content: {e}")
            result = ParseResult(
                model=self.create_empty_model("Failed Lean Parse"), success=False
            )
            result.add_error(f"Failed to parse Lean content: {e}")
            return result

    def _parse_lean_content(self, content: str) -> GNNInternalRepresentation:
        """Parse the main Lean content."""
        content.split("\n")

        # Extract model name from structure or namespace
        model_name = self._extract_model_name(content)

        # Create base model
        model = GNNInternalRepresentation(
            model_name=model_name,
            annotation="Parsed from Lean category theory specification",
        )

        # Parse imports
        self._parse_imports(content, model)

        # Parse structures and definitions
        self._parse_structures(content, model)

        # Parse variable definitions
        self._parse_variables(content, model)

        return model

    def _extract_model_name(self, content: str) -> str:
        """Extract model name from Lean content."""
        # Try structure name first
        structure_match = self.structure_pattern.search(content)
        if structure_match:
            return structure_match.group(1)

        # Try namespace
        namespace_match = self.namespace_pattern.search(content)
        if namespace_match:
            return namespace_match.group(1)

        # Default
        return "LeanGNNModel"

    def _parse_imports(self, content: str, model: GNNInternalRepresentation) -> Any:
        """Parse import statements to understand dependencies."""
        imports = self.import_pattern.findall(content)

        # Store import information in metadata
        model.extensions["lean_imports"] = imports

        # Check for category theory imports
        has_category_theory = any("CategoryTheory" in imp for imp in imports)
        if has_category_theory:
            model.extensions["uses_category_theory"] = True

    def _parse_structures(self, content: str, model: GNNInternalRepresentation) -> Any:
        """Parse Lean structures to extract GNN components."""
        structure_matches = list(self.structure_pattern.finditer(content))

        for match in structure_matches:
            structure_name = match.group(1)

            # Find the structure body
            start_pos = match.end()
            structure_body = self._extract_structure_body(content, start_pos)

            # Parse structure fields as variables
            self._parse_structure_fields(structure_body, model, structure_name)

    def _extract_structure_body(self, content: str, start_pos: int) -> str:
        """Extract the body of a structure definition."""
        # Simple approach: find until next top-level definition or end
        remaining = content[start_pos:]
        lines = remaining.split("\n")

        body_lines: list[Any] = []
        indent_level = None

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Determine base indentation from first non-empty line
            if indent_level is None and stripped:
                indent_level = len(line) - len(line.lstrip())

            # If we hit a line with less indentation, we've left the structure
            current_indent = len(line) - len(line.lstrip())
            if (
                indent_level is not None
                and current_indent <= indent_level
                and stripped
                and not stripped.startswith("--")
            ):
                break

            body_lines.append(line)

        return "\n".join(body_lines)

    def _parse_structure_fields(
        self, structure_body: str, model: GNNInternalRepresentation, structure_name: str
    ) -> Any:
        """Parse fields within a structure as variables."""
        field_matches = self.variable_pattern.findall(structure_body)

        for field_name, field_type in field_matches:
            # Skip comments and imports
            if field_name.startswith("--") or field_name.startswith("import"):
                continue

            # Infer variable type and data type
            var_type = self._infer_variable_type(field_name, field_type)
            data_type = self._infer_data_type(field_type)

            variable = Variable(
                name=field_name,
                var_type=var_type,
                dimensions=[],  # Lean types are abstract
                data_type=data_type,
                description=f"Field from structure {structure_name}",
            )

            model.variables.append(variable)

    def _parse_variables(self, content: str, model: GNNInternalRepresentation) -> Any:
        """Parse standalone variable definitions."""
        def_matches = self.def_pattern.findall(content)

        for def_name, def_type in def_matches:
            # Skip if already parsed as structure field
            if any(var.name == def_name for var in model.variables):
                continue

            var_type = self._infer_variable_type(def_name, def_type)
            data_type = self._infer_data_type(def_type)

            variable = Variable(
                name=def_name,
                var_type=var_type,
                dimensions=[],
                data_type=data_type,
                description=f"Definition: {def_type.strip()}",
            )

            model.variables.append(variable)

    def _infer_variable_type(self, name: str, type_def: str) -> VariableType:
        """Infer GNN variable type from Lean name and type."""
        name_lower = name.lower()
        type_lower = type_def.lower()

        if any(keyword in name_lower for keyword in ["state", "hidden"]):
            return VariableType.HIDDEN_STATE
        elif any(keyword in name_lower for keyword in ["obs", "observation"]):
            return VariableType.OBSERVATION
        elif any(keyword in name_lower for keyword in ["action", "control"]):
            return VariableType.ACTION
        elif any(keyword in name_lower for keyword in ["policy", "pi"]):
            return VariableType.POLICY
        elif "matrix" in type_lower or "mat" in type_lower:
            if "a" in name_lower:
                return VariableType.LIKELIHOOD_MATRIX
            elif "b" in name_lower:
                return VariableType.TRANSITION_MATRIX
            else:
                return VariableType.LIKELIHOOD_MATRIX
        elif "vector" in type_lower or "vec" in type_lower:
            if "c" in name_lower:
                return VariableType.PREFERENCE_VECTOR
            elif "d" in name_lower:
                return VariableType.PRIOR_VECTOR
            else:
                return VariableType.PREFERENCE_VECTOR

        # Default based on name patterns
        return VariableType.HIDDEN_STATE

    def _infer_data_type(self, type_def: str) -> DataType:
        """Infer data type from Lean type definition."""
        type_lower = type_def.lower()

        if any(keyword in type_lower for keyword in ["real", "ℝ", "float"]):
            return DataType.CONTINUOUS
        elif any(keyword in type_lower for keyword in ["nat", "ℕ", "int", "integer"]):
            return DataType.INTEGER
        elif any(keyword in type_lower for keyword in ["bool", "prop"]):
            return DataType.BINARY
        elif any(keyword in type_lower for keyword in ["list", "vector", "array"]):
            return DataType.CATEGORICAL

        # Default for abstract types
        return DataType.CONTINUOUS

    def get_supported_extensions(self) -> List[str]:
        """Get supported file extensions."""
        return [".lean"]
