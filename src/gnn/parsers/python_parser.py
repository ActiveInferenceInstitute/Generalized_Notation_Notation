"""
Python Parser for GNN Geometric/Neural Implementations

This module provides parsing capabilities for Python files that implement
GNN models using neural networks and geometric approaches.

Author: @docxology
Date: 2025-01-11
License: MIT
"""

import ast
import logging
import re
from typing import Any, ClassVar, Dict, List, Optional, cast

from .common import (
    BaseGNNParser,
    DataType,
    GNNInternalRepresentation,
    ParseResult,
    Variable,
    VariableType,
)

logger = logging.getLogger(__name__)


class PythonGNNParser(BaseGNNParser):
    """Parser for Python geometric/neural implementations."""

    EMBEDDED_JSON_PATTERNS: ClassVar[list[str]] = [
        r"#\s*MODEL_DATA:\s*(\{.+\})",  # # MODEL_DATA: {...}
        r'"""\s*MODEL_DATA:\s*(\{.+?\})\s*"""',  # """ MODEL_DATA: {...} """
        r"'''\s*MODEL_DATA:\s*(\{.+?\})\s*'''",  # ''' MODEL_DATA: {...} '''
    ]

    def __init__(self) -> None:
        """Initialize the Python parser."""
        super().__init__()
        self.class_pattern = re.compile(r"class\s+(\w+).*:")
        self.function_pattern = re.compile(r"def\s+(\w+)\s*\([^)]*\):")
        self.import_pattern = re.compile(r"(?:from\s+([^\s]+)\s+)?import\s+([^\n]+)")

    def parse_file(self, file_path: str) -> ParseResult:
        """Parse a Python file containing GNN implementations."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            return self.parse_string(content)

        except Exception as e:
            logger.error(f"Error parsing Python file {file_path}: {e}")
            result = ParseResult(
                model=self.create_empty_model("Failed Python Parse"), success=False
            )
            result.add_error(f"Failed to parse Python file: {e}")
            return result

    def parse_string(self, content: str) -> ParseResult:
        """Parse Python content from string."""
        # First try to extract embedded JSON model data
        embedded_data = self._extract_embedded_json_data(content)
        if embedded_data:
            result = ParseResult(model=self.create_empty_model())
            return self._parse_from_embedded_data(embedded_data, result)

        try:
            model = self._parse_python_content(content)
            return ParseResult(model=model, success=True)

        except Exception as e:
            logger.error(f"Error parsing Python content: {e}")
            result = ParseResult(
                model=self.create_empty_model("Failed Python Parse"), success=False
            )
            result.add_error(f"Failed to parse Python content: {e}")
            return result

    def _parse_python_content(self, content: str) -> GNNInternalRepresentation:
        """Parse the main Python content."""
        # Try to parse as AST first
        try:
            tree = ast.parse(content)
            return self._parse_ast(tree, content)
        except SyntaxError:
            # Recovery to regex parsing
            return self._parse_regex(content)

    def _parse_ast(self, tree: ast.AST, content: str) -> GNNInternalRepresentation:
        """Parse using AST analysis."""
        model_name = self._extract_model_name_ast(tree) or "PythonGNNModel"

        model = GNNInternalRepresentation(
            model_name=model_name, annotation="Parsed from Python implementation"
        )

        # Parse imports
        self._parse_imports_ast(tree, model)

        # Parse classes and functions
        self._parse_classes_ast(tree, model)
        self._parse_functions_ast(tree, model)

        return model

    def _parse_regex(self, content: str) -> GNNInternalRepresentation:
        """Recovery regex parsing."""
        model_name = self._extract_model_name_regex(content)

        model = GNNInternalRepresentation(
            model_name=model_name,
            annotation="Parsed from Python implementation (regex recovery)",
        )

        # Parse imports
        self._parse_imports_regex(content, model)

        # Parse classes and functions
        self._parse_classes_regex(content, model)

        return model

    def _extract_model_name_ast(self, tree: ast.AST) -> Optional[str]:
        """Extract model name from AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if any(
                    keyword in node.name.lower()
                    for keyword in ["model", "gnn", "active", "inference"]
                ):
                    return node.name

        # Try module docstring
        if isinstance(tree, ast.Module) and tree.body:
            first = tree.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                docstring = first.value.value
                if isinstance(docstring, str) and "model" in docstring.lower():
                    # Extract potential model name from docstring
                    lines = docstring.split("\n")
                    for line in lines:
                        if "model" in line.lower():
                            words = line.split()
                            for word in words:
                                if "model" in word.lower():
                                    return word.replace(":", "")

        return None

    def _extract_model_name_regex(self, content: str) -> str:
        """Extract model name using regex."""
        class_matches = self.class_pattern.findall(content)

        for class_name in class_matches:
            if any(
                keyword in class_name.lower()
                for keyword in ["model", "gnn", "active", "inference"]
            ):
                return cast("str", class_name)

        if class_matches:
            return cast("str", class_matches[0])

        return "PythonGNNModel"

    def _parse_imports_ast(
        self, tree: ast.AST, model: GNNInternalRepresentation
    ) -> Any:
        """Parse imports from AST."""
        imports: list[Any] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")

        model.extensions["python_imports"] = imports

        # Check for relevant libraries
        ml_libraries: list[Any] = ["torch", "jax", "tensorflow", "numpy", "scipy"]
        ai_libraries: list[Any] = ["pymdp", "active_inference"]

        model.extensions["uses_ml_libraries"] = any(
            any(lib in imp for lib in ml_libraries) for imp in imports
        )
        model.extensions["uses_ai_libraries"] = any(
            any(lib in imp for lib in ai_libraries) for imp in imports
        )

    def _parse_imports_regex(
        self, content: str, model: GNNInternalRepresentation
    ) -> Any:
        """Parse imports using regex."""
        import_matches = self.import_pattern.findall(content)
        imports: list[Any] = []

        for from_module, import_items in import_matches:
            if from_module:
                for item in import_items.split(","):
                    imports.append(f"{from_module}.{item.strip()}")
            else:
                imports.extend([item.strip() for item in import_items.split(",")])

        model.extensions["python_imports"] = imports

    def _parse_classes_ast(
        self, tree: ast.AST, model: GNNInternalRepresentation
    ) -> Any:
        """Parse class definitions from AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                self._parse_class_ast(node, model)

    def _parse_class_ast(
        self, class_node: ast.ClassDef, model: GNNInternalRepresentation
    ) -> Any:
        """Parse a single class definition."""
        class_name = class_node.name

        # Parse methods to extract variables
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                self._parse_method_ast(node, model, class_name)
            elif isinstance(node, ast.Assign):
                self._parse_class_assignment_ast(node, model, class_name)

    def _parse_method_ast(
        self,
        method_node: ast.FunctionDef,
        model: GNNInternalRepresentation,
        class_name: str,
    ) -> Any:
        """Parse method to extract variables and connections."""
        method_name = method_node.name

        # Look for variable assignments and operations
        for node in ast.walk(method_node):
            if isinstance(node, ast.Assign):
                self._parse_assignment_ast(node, model, f"{class_name}.{method_name}")

    def _parse_assignment_ast(
        self, assign_node: ast.Assign, model: GNNInternalRepresentation, context: str
    ) -> Any:
        """Parse assignment to extract variables."""
        for target in assign_node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id

                # Infer variable properties
                var_type = self._infer_variable_type_from_name(var_name)
                data_type = self._infer_data_type_from_assignment(assign_node)

                # Skip if already exists
                if any(var.name == var_name for var in model.variables):
                    continue

                variable = Variable(
                    name=var_name,
                    var_type=var_type,
                    dimensions=self._infer_dimensions_from_assignment(assign_node),
                    data_type=data_type,
                    description=f"Variable from {context}",
                )

                model.variables.append(variable)

    def _parse_class_assignment_ast(
        self, assign_node: ast.Assign, model: GNNInternalRepresentation, class_name: str
    ) -> Any:
        """Parse class-level assignments."""
        self._parse_assignment_ast(assign_node, model, f"class {class_name}")

    def _parse_functions_ast(
        self, tree: ast.AST, model: GNNInternalRepresentation
    ) -> Any:
        """Parse standalone function definitions."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and self._is_top_level_function(
                node, tree
            ):
                # Parse function body for variables
                for sub_node in ast.walk(node):
                    if isinstance(sub_node, ast.Assign):
                        self._parse_assignment_ast(
                            sub_node, model, f"function {node.name}"
                        )

    def _is_top_level_function(self, func_node: ast.FunctionDef, tree: ast.AST) -> bool:
        """Check if function is at module level (not in a class)."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if func_node in ast.walk(node):
                    return False
        return True

    def _parse_classes_regex(
        self, content: str, model: GNNInternalRepresentation
    ) -> Any:
        """Parse classes using regex."""
        class_matches = self.class_pattern.findall(content)

        for class_name in class_matches:
            # Extract class body (simplified)
            class_pattern = (
                rf"class\s+{re.escape(class_name)}.*?(?=\nclass|\n\w+\s*=|\Z)"
            )
            class_match = re.search(class_pattern, content, re.DOTALL)

            if class_match:
                class_body = class_match.group(0)
                self._extract_variables_from_text(
                    class_body, model, f"class {class_name}"
                )

    def _extract_variables_from_text(
        self, text: str, model: GNNInternalRepresentation, context: str
    ) -> Any:
        """Extract variables from text using heuristics."""
        # Look for common variable patterns
        patterns: list[Any] = [
            r"self\.(\w+)\s*=",  # Instance variables
            r"(\w+)\s*=\s*(?:torch|jax|np)\.",  # ML library assignments
            r"(\w+)\s*=\s*.*(?:state|action|observation|policy)",  # AI variables
        ]

        found_vars: set[Any] = set()

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                var_name = match if isinstance(match, str) else match[0]

                if var_name not in found_vars and not var_name.startswith("_"):
                    found_vars.add(var_name)

                    var_type = self._infer_variable_type_from_name(var_name)
                    data_type = DataType.CONTINUOUS  # Default for Python

                    variable = Variable(
                        name=var_name,
                        var_type=var_type,
                        dimensions=[],
                        data_type=data_type,
                        description=f"Variable from {context}",
                    )

                    model.variables.append(variable)

    def _infer_variable_type_from_name(self, name: str) -> VariableType:
        """Infer variable type from name."""
        name_lower = name.lower()

        if any(keyword in name_lower for keyword in ["state", "hidden", "s_"]):
            return VariableType.HIDDEN_STATE
        elif any(keyword in name_lower for keyword in ["obs", "observation", "o_"]):
            return VariableType.OBSERVATION
        elif any(keyword in name_lower for keyword in ["action", "control", "u_"]):
            return VariableType.ACTION
        elif any(keyword in name_lower for keyword in ["policy", "pi_"]):
            return VariableType.POLICY
        elif name_lower in ["a", "a_matrix", "likelihood"]:
            return VariableType.LIKELIHOOD_MATRIX
        elif name_lower in ["b", "b_matrix", "transition"]:
            return VariableType.TRANSITION_MATRIX
        elif name_lower in ["c", "c_vector", "preference"]:
            return VariableType.PREFERENCE_VECTOR
        elif name_lower in ["d", "d_vector", "prior"]:
            return VariableType.PRIOR_VECTOR

        return VariableType.HIDDEN_STATE

    def _infer_data_type_from_assignment(self, assign_node: ast.Assign) -> DataType:
        """Infer data type from assignment AST."""
        if isinstance(assign_node.value, ast.Constant):
            value = assign_node.value.value
            if isinstance(value, bool):
                return DataType.BINARY
            elif isinstance(value, int):
                return DataType.INTEGER
            elif isinstance(value, float):
                return DataType.CONTINUOUS

        # Check for function calls that might indicate type
        if isinstance(assign_node.value, ast.Call):
            if isinstance(assign_node.value.func, ast.Attribute):
                func_name = assign_node.value.func.attr
                if func_name in ["zeros", "ones", "randn", "random"]:
                    return DataType.CONTINUOUS
                elif func_name in ["randint", "arange"]:
                    return DataType.INTEGER

        return DataType.CONTINUOUS

    def _infer_dimensions_from_assignment(self, assign_node: ast.Assign) -> List[int]:
        """Infer dimensions from assignment."""
        if isinstance(assign_node.value, ast.Call):
            if assign_node.value.args:
                # Try to extract shape from function arguments
                for arg in assign_node.value.args:
                    if isinstance(arg, (ast.Tuple, ast.List)):
                        try:
                            dims: list[Any] = []
                            for elt in arg.elts:
                                if isinstance(elt, ast.Constant) and isinstance(
                                    elt.value, int
                                ):
                                    dims.append(elt.value)
                            if dims:
                                return dims
                        except (ValueError, TypeError, AttributeError) as e:
                            logger.debug(
                                "AST dimension extraction failed (best-effort): %s", e
                            )

        return []

    def get_supported_extensions(self) -> List[str]:
        """Get supported file extensions."""
        return [".py"]
