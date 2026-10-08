"""
Markdown GNN Parser

This module provides parsing for the standard GNN Markdown format.
This is the reference parser implementation that other formats can use as a model.

Author: @docxology
Date: 2025-01-11
License: MIT
"""

import logging
import re
from typing import Any, cast

TRACE_LEVEL = 5
logging.addLevelName(TRACE_LEVEL, "TRACE")
from pathlib import Path
from typing import Dict, List, Optional

from .common import (
    BaseGNNParser,
    Connection,
    DataType,
    Equation,
    GNNInternalRepresentation,
    OntologyMapping,
    ParseError,
    ParseResult,
    TimeSpecification,
    Variable,
    infer_variable_type,
    normalize_variable_name,
    parse_connection_operator,
    parse_dimensions,
)
from .markdown_parser_parameter import ParameterParsingMixin

logger = logging.getLogger(__name__)


class MarkdownGNNParser(ParameterParsingMixin, BaseGNNParser):
    """
    Parser for GNN Markdown format.

    This parser handles the standard GNN format as defined in the specification,
    including all sections and their contents.
    """

    def __init__(self) -> None:
        """Initialize the instance."""
        super().__init__()
        self.section_parsers = {
            "GNNSection": self._parse_gnn_section,
            "GNNVersionAndFlags": self._parse_version_section,
            "ModelName": self._parse_model_name,
            "ModelAnnotation": self._parse_annotation,
            "StateSpaceBlock": self._parse_state_space,
            "Connections": self._parse_connections,
            "InitialParameterization": self._parse_parameters,
            "Equations": self._parse_equations,
            "Time": self._parse_time,
            "ActInfOntologyAnnotation": self._parse_ontology,
            "ModelParameters": self._parse_model_parameters,
            "Footer": self._parse_footer,
            "Signature": self._parse_signature,
        }

    def get_supported_extensions(self) -> List[str]:
        """Get supported file extensions."""
        return [".md", ".markdown"]

    def parse_file(self, file_path: str) -> ParseResult:
        """Parse a GNN Markdown file."""
        self.current_file = file_path
        path = Path(file_path)

        try:
            content = path.read_text(encoding="utf-8")
            return self.parse_string(content)
        except Exception as e:
            raise ParseError(f"Failed to read file {path}: {e}") from e

    def parse_string(self, content: str) -> ParseResult:
        """Parse GNN Markdown content from string."""
        try:
            # Handle empty or whitespace-only content as invalid
            if content is None or not str(content).strip():
                result = ParseResult(model=self.create_empty_model())
                result.success = False
                result.add_error("Empty content")
                return result
            # Create result container
            result = ParseResult(model=self.create_empty_model())

            # Split content into sections
            sections = self._split_into_sections(content)

            # If no headers/sections found, treat as invalid
            if not sections:
                result.success = False
                result.add_error("No sections found in markdown content")
                return result

            # Parse each section
            for section_name, section_content in sections.items():
                try:
                    self._parse_section(section_name, section_content, result.model)
                except Exception as e:
                    error_msg = f"Error parsing section '{section_name}': {e}"
                    result.add_warning(error_msg)
                    logger.warning(error_msg)

            # Post-processing and validation
            self._post_process_model(result.model)

            # Set model name if not found
            if (
                not result.model.model_name
                or result.model.model_name == "Unnamed Model"
            ):
                result.model.model_name = self._extract_model_name_fallback(content)

            result.model.raw_sections = dict(sections.items())

            return result

        except Exception as e:
            result = ParseResult(
                model=self.create_empty_model("Parse Error"), success=False
            )
            result.add_error(str(e))
            return result

    def _split_into_sections(self, content: str) -> Dict[str, str]:
        """Split content into named sections based on ## headers."""
        sections: dict[Any, Any] = {}
        current_section = None
        current_content: list[Any] = []

        lines = content.split("\n")

        for i, line in enumerate(lines):
            self.current_line = i + 1

            # Check for section header
            if line.strip().startswith("## "):
                # Save previous section
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()

                # Start new section
                current_section = line.strip()[3:].strip()
                current_content = []

            elif current_section:
                current_content.append(line)

        # Save last section
        if current_section:
            sections[current_section] = "\n".join(current_content).strip()

        return sections

    def _parse_section(
        self, section_name: str, content: str, model: GNNInternalRepresentation
    ) -> Any:
        """Parse a specific section."""
        if section_name in self.section_parsers:
            self.section_parsers[section_name](content, model)
        else:
            # Handle unknown sections as extensions
            model.extensions[section_name] = content
            logger.debug(f"Unknown section '{section_name}' stored as extension")

    def _parse_gnn_section(self, content: str, model: GNNInternalRepresentation) -> Any:
        """Parse the GNNSection."""
        # Usually contains the model type or identifier
        model.extensions["gnn_section"] = content.strip()

    def _parse_version_section(
        self, content: str, model: GNNInternalRepresentation
    ) -> Any:
        """Parse the GNNVersionAndFlags section."""
        lines = content.strip().split("\n")

        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                if "GNN v" in line or "v" in line:
                    # Extract version
                    version_match = re.search(r"v?(\d+\.\d+(?:\.\d+)?)", line)
                    if version_match:
                        model.version = version_match.group(1)
                else:
                    # Handle flags
                    model.extensions.setdefault("flags", []).append(line)

    def _parse_model_name(self, content: str, model: GNNInternalRepresentation) -> Any:
        """Parse the ModelName section."""
        model.model_name = content.strip()

    def _parse_annotation(self, content: str, model: GNNInternalRepresentation) -> Any:
        """Parse the ModelAnnotation section."""
        model.annotation = content.strip()

    def _parse_state_space(self, content: str, model: GNNInternalRepresentation) -> Any:
        """Parse the StateSpaceBlock section."""
        lines = content.strip().split("\n")

        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                # Check if line contains variable definition pattern
                if "[" in line and "]" in line:
                    variable = self._parse_variable_definition(line)
                    if variable:
                        model.variables.append(variable)
                        logger.log(
                            TRACE_LEVEL,
                            "Parsed variable: %s with dimensions %s",
                            variable.name,
                            variable.dimensions,
                        )

    def _parse_variable_definition(self, line: str) -> Optional[Variable]:
        """Parse a single variable definition line."""
        try:
            # Handle lines like: A[3,3,type=float]   # Comment
            comment = None
            if "#" in line:
                line, comment = line.split("#", 1)
                comment = comment.strip()
                line = line.strip()

            # Parse variable pattern: name[dimensions,type=datatype]
            bracket_start = line.find("[")
            bracket_end = line.find("]")
            if bracket_start == -1 or bracket_end == -1:
                return None

            name = line[:bracket_start].strip()
            dimensions_str = line[bracket_start + 1 : bracket_end].strip()

            # Extract type specification
            type_spec = None
            dimension_parts = dimensions_str.split(",")
            for index, part in enumerate(dimension_parts):
                token = part.strip()
                if token.startswith("type="):
                    type_spec = token.split("=", 1)[1].strip()
                    dimensions_str = ",".join(dimension_parts[:index])
                    break

            # Parse dimensions
            dimensions = parse_dimensions("[" + dimensions_str + "]")

            # Determine variable type and data type
            var_type = infer_variable_type(name)
            data_type = (
                self._parse_data_type(type_spec) if type_spec else DataType.FLOAT
            )

            logger.log(
                TRACE_LEVEL,
                "Parsed variable '%s' with dimensions %s, type=%s",
                name,
                dimensions,
                type_spec,
            )

            return Variable(
                name=normalize_variable_name(name),
                var_type=var_type,
                dimensions=dimensions,
                data_type=data_type,
                description=comment,
            )

        except Exception as e:
            logger.warning(f"Failed to parse variable definition '{line}': {e}")
            return None

    def _parse_data_type(self, type_str: str) -> DataType:
        """Parse data type string."""
        type_str = type_str.lower().strip()

        type_map: dict[str, Any] = {
            "categorical": DataType.CATEGORICAL,
            "continuous": DataType.CONTINUOUS,
            "binary": DataType.BINARY,
            "bool": DataType.BINARY,
            "integer": DataType.INTEGER,
            "int": DataType.INTEGER,
            "float": DataType.FLOAT,
            "real": DataType.FLOAT,
            "complex": DataType.COMPLEX,
        }

        return cast("DataType", type_map.get(type_str, DataType.CATEGORICAL))

    def _parse_connections(self, content: str, model: GNNInternalRepresentation) -> Any:
        """Parse the Connections section."""
        lines = content.strip().split("\n")

        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                connection = self._parse_connection_definition(line)
                if connection:
                    model.connections.append(connection)
                    logger.log(
                        TRACE_LEVEL,
                        "Parsed connection: %s -> %s",
                        connection.source_variables,
                        connection.target_variables,
                    )

    def _parse_connection_definition(self, line: str) -> Optional[Connection]:
        """Parse a single connection definition line."""
        try:
            # Extract comment
            comment = None
            if "#" in line:
                line, comment = line.split("#", 1)
                comment = comment.strip()
                line = line.strip()

            # Parse connection pattern: source>target, source-target, (s1,s2)>target
            # Find connection operators
            operators: list[Any] = ["<->", "->", ">", "-", "|"]
            found_op = None
            op_pos = -1

            for op in operators:
                pos = line.find(op)
                if pos != -1:
                    found_op = op
                    op_pos = pos
                    break

            if not found_op:
                logger.warning(f"No connection operator found in: {line}")
                return None

            # Split source and target
            source_part = line[:op_pos].strip()
            target_part = line[op_pos + len(found_op) :].strip()

            # v1.1 connection annotation: an ':annotation' suffix on the
            # target side (docs/gnn/gnn_syntax.md section 3, "Annotated
            # directed edge" `A>B:label`). Annotations are labels for
            # rendering/documentation; parsers must accept and preserve
            # them but not treat them as part of the target variable name.
            # Only the FIRST annotation suffix is stripped; the target
            # variable group itself never contains ':' in v1 syntax.
            annotation: Optional[str] = None
            if ":" in target_part:
                target_part, annotation = target_part.split(":", 1)
                annotation = annotation.strip() or None
                target_part = target_part.strip()

            # Parse variable groups
            source_vars = self._parse_variable_group(source_part)
            target_vars = self._parse_variable_group(target_part)

            # Parse connection type
            conn_type = parse_connection_operator(found_op)

            return Connection(
                source_variables=source_vars,
                target_variables=target_vars,
                connection_type=conn_type,
                annotation=annotation,
                description=comment,
            )

        except Exception as e:
            logger.warning(f"Failed to parse connection definition '{line}': {e}")
            return None

    def _parse_variable_group(self, group_str: str) -> List[str]:
        """Parse a variable group like 'X' or '(X,Y,Z)'."""
        group_str = group_str.strip()

        if group_str.startswith("(") and group_str.endswith(")"):
            # Multiple variables
            inner = group_str[1:-1]
            variables = [var.strip() for var in inner.split(",")]
            return [normalize_variable_name(var) for var in variables if var.strip()]
        else:
            # Single variable
            return [normalize_variable_name(group_str)]

    def _parse_parameters(self, content: str, model: GNNInternalRepresentation) -> Any:
        """Parse the InitialParameterization section."""
        lines = content.strip().split("\n")

        current_parameter = None
        current_value_lines: list[Any] = []
        in_matrix = False
        matrix_brace_count = 0

        for line in lines:
            line = line.strip()

            if line and not line.startswith("#"):
                # Check if this is a new parameter definition (contains '=' or ':' and not inside a matrix)
                has_assignment = "=" in line or ":" in line
                if has_assignment and not in_matrix and not line.startswith("#"):
                    # Save previous parameter if exists
                    if current_parameter and current_value_lines:
                        value_str = "\n".join(current_value_lines)
                        parameter = self._parse_parameter_assignment(
                            f"{current_parameter}={value_str}"
                        )
                        if parameter:
                            model.parameters.append(parameter)

                    # Start new parameter
                    if "=" in line:
                        name, value = line.split("=", 1)
                    else:
                        name, value = line.split(":", 1)
                    current_parameter = name.strip()
                    current_value_lines = [value.strip()]

                    # Check if this parameter value starts a matrix definition
                    value_trimmed = value.strip()
                    if value_trimmed.startswith("{"):
                        in_matrix = True
                        matrix_brace_count = value_trimmed.count(
                            "{"
                        ) - value_trimmed.count("}")
                        if matrix_brace_count <= 0:
                            in_matrix = False  # Matrix completed on the same line
                else:
                    # Continue current parameter value
                    if current_parameter:
                        current_value_lines.append(line)

                        # Track matrix braces if we're in a matrix
                        if in_matrix:
                            matrix_brace_count += line.count("{") - line.count("}")
                            if matrix_brace_count <= 0:
                                in_matrix = False  # Matrix completed
            elif line.startswith("#"):
                # Skip comments unless we're inside a matrix
                if not in_matrix:
                    continue
                # If inside matrix, include comments as they might be part of the value
                if current_parameter:
                    current_value_lines.append(line)

        # Handle last parameter
        if current_parameter and current_value_lines:
            value_str = "\n".join(current_value_lines)
            parameter = self._parse_parameter_assignment(
                f"{current_parameter}={value_str}"
            )
            if parameter:
                model.parameters.append(parameter)

    # _parse_parameter_assignment, _parse_parameter_value, _parse_matrix_rows,
    # _parse_matrix_row, _has_nested_tuples, _extract_tuples, and _parse_single_value
    # are inherited from ParameterParsingMixin (markdown_parser_parameter.py)

    def _parse_equations(self, content: str, model: GNNInternalRepresentation) -> Any:
        """Parse the Equations section."""
        lines = content.strip().split("\n")
        current_equation: list[Any] = []
        current_label = None

        for line in lines:
            line = line.strip()

            if line and not line.startswith("#"):
                # Check for equation label
                if (
                    ":" in line
                    and not line.startswith("$")
                    and "=" not in line.split(":")[0]
                ):
                    label_part, eq_part = line.split(":", 1)
                    current_label = label_part.strip()
                    if eq_part.strip():
                        current_equation.append(eq_part.strip())
                else:
                    current_equation.append(line)

            # Check if equation is complete (empty line or new label)
            if (not line or line.startswith("#")) and current_equation:
                equation_content = " ".join(current_equation).strip()
                if equation_content:
                    equation = Equation(
                        label=current_label, content=equation_content, format="latex"
                    )
                    model.equations.append(equation)

                current_equation = []
                current_label = None

        # Handle last equation
        if current_equation:
            equation_content = " ".join(current_equation).strip()
            if equation_content:
                equation = Equation(
                    label=current_label, content=equation_content, format="latex"
                )
                model.equations.append(equation)

    def _parse_time(self, content: str, model: GNNInternalRepresentation) -> Any:
        """Parse the Time section."""
        lines = content.strip().split("\n")

        time_spec = TimeSpecification(time_type="Static")

        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                line_lower = line.lower()

                if "static" in line_lower:
                    time_spec.time_type = "Static"
                elif "dynamic" in line_lower:
                    time_spec.time_type = "Dynamic"
                elif "discretetime" in line_lower:
                    time_spec.discretization = "DiscreteTime"
                elif "continuoustime" in line_lower:
                    time_spec.discretization = "ContinuousTime"
                elif "modeltimehorizon" in line_lower and "=" in line:
                    _, horizon_str = line.split("=", 1)
                    try:
                        time_spec.horizon = int(horizon_str.strip())
                    except ValueError:
                        time_spec.horizon = horizon_str.strip()

        model.time_specification = time_spec

    def _parse_ontology(self, content: str, model: GNNInternalRepresentation) -> Any:
        """Parse the ActInfOntologyAnnotation section."""
        lines = content.strip().split("\n")

        for line in lines:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                mapping = self._parse_ontology_mapping(line)
                if mapping:
                    model.ontology_mappings.append(mapping)

    def _parse_ontology_mapping(self, line: str) -> Optional[OntologyMapping]:
        """Parse a single ontology mapping line."""
        try:
            # Extract comment
            comment = None
            if "###" in line:
                line, comment = line.split("###", 1)
                comment = comment.strip()
                line = line.strip()

            # Split by '='
            if "=" not in line:
                return None

            var_name, term = line.split("=", 1)
            var_name = var_name.strip()
            term = term.strip()

            return OntologyMapping(
                variable_name=normalize_variable_name(var_name),
                ontology_term=term,
                description=comment,
            )

        except Exception as e:
            logger.warning(f"Failed to parse ontology mapping '{line}': {e}")
            return None

    def _parse_model_parameters(
        self, content: str, model: GNNInternalRepresentation
    ) -> Any:
        """Parse the ModelParameters section."""
        # Similar to InitialParameterization but for model-level parameters
        self._parse_parameters(content, model)

    def _parse_footer(self, content: str, model: GNNInternalRepresentation) -> Any:
        """Parse the Footer section."""
        model.extensions["footer"] = content.strip()

    def _parse_signature(self, content: str, model: GNNInternalRepresentation) -> Any:
        """Parse the Signature section."""
        model.extensions["signature"] = content.strip()

    def _extract_model_name_fallback(self, content: str) -> str:
        """Extract model name from content if not found in ModelName section."""
        lines = content.split("\n")

        # Look for title (first # header)
        for line in lines:
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()

        # Look for filename pattern
        if self.current_file:
            return Path(self.current_file).stem

        return "Unnamed Model"

    def _post_process_model(self, model: GNNInternalRepresentation) -> Any:
        """Post-process the parsed model for consistency."""
        # Ensure variable names in connections exist
        all_var_names = {var.name for var in model.variables}

        for connection in model.connections:
            # Check source variables
            missing_sources = [
                var for var in connection.source_variables if var not in all_var_names
            ]
            if missing_sources:
                logger.warning(
                    f"Connection references unknown source variables: {missing_sources}"
                )

            # Check target variables
            missing_targets = [
                var for var in connection.target_variables if var not in all_var_names
            ]
            if missing_targets:
                logger.warning(
                    f"Connection references unknown target variables: {missing_targets}"
                )

        # Ensure ontology mappings reference existing variables or are standard Active Inference variables
        standard_ai_vars: set[Any] = {
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "Q",
            "R",
            "π",
            "u",
            "s",
            "o",
            "t",
        }
        for mapping in model.ontology_mappings:
            if (
                mapping.variable_name not in all_var_names
                and mapping.variable_name not in standard_ai_vars
            ):
                logger.warning(
                    f"Ontology mapping references unknown variable: {mapping.variable_name}"
                )


# Export the parser class
__all__: list[Any] = ["MarkdownGNNParser"]
