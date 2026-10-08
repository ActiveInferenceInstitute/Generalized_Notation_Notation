"""Section-level GNN markdown parsers for GNNParser.

Holds SectionParsersMixin: the per-section parsing methods (markdown
dispatch, state space, connections, parameters, equations, time, ontology
mappings, signature), split out of gnn/schema_validator/syntax.py.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from gnn.types import GNNConnection, GNNVariable, ParsedGNN

logger = logging.getLogger(__name__)


class SectionParsersMixin:
    """Mixin with GNN section parsing methods."""

    SECTION_PATTERN: re.Pattern[str]
    VARIABLE_PATTERN: re.Pattern[str]
    CONNECTION_PATTERN: re.Pattern[str]
    PARAMETER_PATTERN: re.Pattern[str]
    ONTOLOGY_PATTERN: re.Pattern[str]
    COMMENT_PATTERN: re.Pattern[str]
    line_number: int
    current_section: Optional[str]

    def _parse_markdown_content(self, content: str, source_name: str) -> ParsedGNN:
        """Parse GNN content from string (markdown format).

        Side effects: resets self.line_number to 0 and self.current_section to None
        at the start of each call, then mutates both as parsing progresses.
        """
        lines = content.split("\n")
        self.line_number = 0
        self.current_section = None

        # Initialize parsed structure
        parsed = ParsedGNN(
            gnn_section="",
            version="",
            model_name="",
            model_annotation="",
            variables={},
            connections=[],
            parameters={},
            equations=[],
            time_config={},
            ontology_mappings={},
            model_parameters={},
            footer="",
        )

        current_content: list[Any] = []

        for line in lines:
            self.line_number += 1
            line = line.rstrip()

            # Check for section headers
            section_match = self.SECTION_PATTERN.match(line)
            if section_match:
                # Process previous section content
                if self.current_section and current_content:
                    self._process_section_content(
                        parsed, self.current_section, current_content
                    )

                # Start new section
                self.current_section = section_match.group(1)
                current_content = []
                continue

            # Skip empty lines at section boundaries
            if not line.strip() and not current_content:
                continue

            current_content.append(line)

        # Process final section
        if self.current_section and current_content:
            self._process_section_content(parsed, self.current_section, current_content)

        return parsed

    def _process_section_content(
        self, parsed: ParsedGNN, section: str, content: List[str]
    ) -> Any:
        """Process content for a specific section."""
        content_text = "\n".join(content).strip()

        if section == "GNNSection":
            parsed.gnn_section = content_text
        elif section == "GNNVersionAndFlags":
            parsed.version = content_text
        elif section == "ModelName":
            parsed.model_name = content_text
        elif section == "ModelAnnotation":
            parsed.model_annotation = content_text
        elif section == "StateSpaceBlock":
            self._parse_state_space_block(parsed, content)
        elif section == "Connections":
            self._parse_connections(parsed, content)
        elif section == "InitialParameterization":
            self._parse_parameters(parsed, content, "parameters")
        elif section == "Equations":
            self._parse_equations(parsed, content)
        elif section == "Time":
            self._parse_time_config(parsed, content)
        elif section == "ActInfOntologyAnnotation":
            self._parse_ontology_mappings(parsed, content)
        elif section == "ModelParameters":
            self._parse_parameters(parsed, content, "model_parameters")
        elif section == "Footer":
            parsed.footer = content_text
        elif section == "Signature":
            parsed.signature = self._parse_signature(content_text)

    def _parse_state_space_block(self, parsed: ParsedGNN, content: List[str]) -> Any:
        """Parse StateSpaceBlock section."""
        for i, line in enumerate(content):
            line = line.strip()
            if not line or self.COMMENT_PATTERN.match(line):
                continue

            match = self.VARIABLE_PATTERN.match(line)
            if match:
                name = match.group(1)
                dims_str = match.group(3)
                data_type = match.group(4)
                description = match.group(5)

                # Parse dimensions
                dimensions: list[Any] = []
                if dims_str:
                    for dim in dims_str.split(","):
                        dim = dim.strip()
                        # Bind the bracket field; an explicit existing suffix wins.
                        if dim.startswith("type="):
                            if data_type is None:
                                data_type = dim.split("=", 1)[1].strip()
                            continue
                        # Only count numeric dimensions for tests that expect numeric length
                        if dim.isdigit():
                            dimensions.append(int(dim))
                        else:
                            # Keep string dimensions but do not count them as numeric dims
                            dimensions.append(dim)

                variable = GNNVariable(
                    name=name,
                    dimensions=dimensions,
                    data_type=data_type or "float",
                    description=description,
                    line_number=self.line_number - len(content) + i,
                )

                parsed.variables[name] = variable
            else:
                logger.warning(f"Could not parse variable definition: {line}")

    def _parse_connections(self, parsed: ParsedGNN, content: List[str]) -> Any:
        """Parse Connections section."""
        for i, line in enumerate(content):
            line = line.strip()
            if not line or self.COMMENT_PATTERN.match(line):
                continue

            match = self.CONNECTION_PATTERN.match(line)
            if match:
                source_str = match.group(1).strip()
                symbol = match.group(2)
                target_str = match.group(3).strip()
                description = match.group(4)

                # Parse variable groups
                source = self._parse_variable_group(source_str)
                target = self._parse_variable_group(target_str)

                # Determine connection type
                connection_type = self._get_connection_type(symbol)

                connection = GNNConnection(
                    source=source,
                    target=target,
                    connection_type=connection_type,
                    symbol=symbol,
                    description=description,
                    line_number=self.line_number - len(content) + i,
                )

                parsed.connections.append(connection)
            else:
                logger.warning(f"Could not parse connection: {line}")

    def _parse_variable_group(self, group_str: str) -> Union[str, List[str]]:
        """Parse a variable group (single variable or parenthesized list)."""
        group_str = group_str.strip()

        if group_str.startswith("(") and group_str.endswith(")"):
            # Parse comma-separated list
            inner = group_str[1:-1]
            variables = [v.strip() for v in inner.split(",")]
            return variables if len(variables) > 1 else variables[0]
        else:
            return group_str

    def _get_connection_type(self, symbol: str) -> str:
        """Determine connection type from symbol."""
        if symbol in [">", "->"]:
            return "directed"
        elif symbol == "-":
            return "undirected"
        elif symbol == "|":
            return "conditional"
        else:
            return "unknown"

    def _parse_parameters(
        self, parsed: ParsedGNN, content: List[str], param_type: str
    ) -> Any:
        """Parse parameter sections."""
        target_dict = getattr(parsed, param_type)

        for line in content:
            line = line.strip()
            if not line or self.COMMENT_PATTERN.match(line):
                continue

            match = self.PARAMETER_PATTERN.match(line)
            if match:
                name = match.group(1)
                value_str = match.group(3)
                match.group(4)

                # Try to parse the value
                try:
                    value = self._parse_parameter_value(value_str)
                    target_dict[name] = value
                except Exception as e:
                    logger.warning(f"Could not parse parameter value for {name}: {e}")
                    target_dict[name] = value_str

    def _parse_parameter_value(self, value_str: str) -> Any:
        """Parse a parameter value string."""
        value_str = value_str.strip()

        # Try JSON parsing first
        try:
            return json.loads(value_str)
        except json.JSONDecodeError:
            logger.debug(
                "Value is not valid JSON, trying GNN-specific formats: %s",
                value_str[:80],
            )

        # Try specific GNN formats
        if value_str.startswith("{") and value_str.endswith("}"):
            # Matrix or tuple format
            return self._parse_matrix_or_tuple(value_str)
        elif value_str.startswith("(") and value_str.endswith(")"):
            # Tuple format
            return self._parse_tuple(value_str)
        elif value_str.replace(".", "").replace("-", "").isdigit():
            # Numeric value
            return float(value_str) if "." in value_str else int(value_str)
        elif value_str.lower() in ["true", "false"]:
            # Boolean value
            return value_str.lower() == "true"
        else:
            # String value
            return value_str

    def _parse_matrix_or_tuple(self, value_str: str) -> Any:
        """Parse matrix or tuple notation with full Active Inference support."""
        value_str = value_str.strip()

        # Handle nested tuple/matrix structures like {((0.8,0.1,0.1),(0.1,0.8,0.1))}
        if value_str.startswith("{") and value_str.endswith("}"):
            inner = value_str[1:-1].strip()

            # Check for matrix structure with nested tuples
            if inner.startswith("(") and inner.count("(") > 1:
                return self._parse_nested_matrix(inner)
            else:
                return self._parse_tuple(inner)

        # Handle simple tuple like (0.5,0.5)
        elif value_str.startswith("(") and value_str.endswith(")"):
            return self._parse_tuple(value_str)

        # Handle list/array notation
        elif value_str.startswith("[") and value_str.endswith("]"):
            return self._parse_array(value_str)

        return value_str

    def _parse_nested_matrix(self, inner: str) -> List[List[float]]:
        """Parse nested matrix structure like ((0.8,0.1),(0.1,0.8))."""
        matrix: list[Any] = []
        depth = 0
        current_tuple = ""

        for char in inner:
            if char == "(":
                depth += 1
                current_tuple += char
            elif char == ")":
                depth -= 1
                current_tuple += char
                if depth == 0:
                    # Parse complete tuple
                    tuple_values = self._parse_tuple(current_tuple)
                    if isinstance(tuple_values, (list, tuple)):
                        matrix.append(list(tuple_values))
                    current_tuple = ""
            elif depth > 0:
                current_tuple += char
            elif char == "," and depth == 0:
                continue  # Skip commas between tuples

        return matrix

    def _parse_array(self, value_str: str) -> List[Any]:
        """Parse array notation [1,2,3] or [[1,2],[3,4]]."""
        from gnn.utils.runtime_safety.safe_eval import safe_literal_eval

        try:
            return cast("list[Any]", safe_literal_eval(value_str))
        except (ValueError, SyntaxError):
            # Recovery parsing
            inner = value_str[1:-1].strip()
            if not inner:
                return []

            elements: list[Any] = []
            depth = 0
            current = ""

            for char in inner:
                if char in "[(":
                    depth += 1
                elif char in "])":
                    depth -= 1

                if char == "," and depth == 0:
                    elements.append(self._parse_scalar_value(current.strip()))
                    current = ""
                else:
                    current += char

            if current.strip():
                elements.append(self._parse_scalar_value(current.strip()))

            return elements

    def _parse_scalar_value(self, value_str: str) -> Union[float, int, bool, str]:
        """Parse a scalar value with proper type conversion."""
        value_str = value_str.strip()

        # Boolean values
        if value_str.lower() in ["true", "false"]:
            return value_str.lower() == "true"

        # Numeric values
        try:
            if "." in value_str or "e" in value_str.lower():
                return float(value_str)
            else:
                return int(value_str)
        except ValueError:
            logger.debug("Value is not numeric, treating as string: %s", value_str[:80])

        # String values
        if value_str.startswith('"') and value_str.endswith('"'):
            return value_str[1:-1]
        elif value_str.startswith("'") and value_str.endswith("'"):
            return value_str[1:-1]

        return value_str

    def _parse_tuple(self, value_str: str) -> Tuple[Any, ...]:
        """Parse tuple notation with proper type conversion."""
        if value_str.startswith("(") and value_str.endswith(")"):
            inner = value_str[1:-1].strip()
        else:
            inner = value_str.strip()

        if not inner:
            return ()

        # Split on commas, handling nested structures
        elements: list[Any] = []
        depth = 0
        current = ""

        for char in inner:
            if char in "([{":
                depth += 1
                current += char
            elif char in ")]}":
                depth -= 1
                current += char
            elif char == "," and depth == 0:
                elements.append(self._parse_scalar_value(current.strip()))
                current = ""
            else:
                current += char

        if current.strip():
            elements.append(self._parse_scalar_value(current.strip()))

        return tuple(elements)

    def _parse_equations(self, parsed: ParsedGNN, content: List[str]) -> Any:
        """Parse Equations section."""
        current_equation: dict[Any, Any] = {}

        for line in content:
            line = line.strip()
            if not line:
                if current_equation:
                    parsed.equations.append(current_equation)
                    current_equation = {}
                continue

            comment_match = self.COMMENT_PATTERN.match(line)
            if comment_match is not None:
                if current_equation:
                    current_equation["description"] = comment_match.group(1)
            else:
                if not current_equation:
                    current_equation = {"latex": line}
                else:
                    current_equation["latex"] += " " + line

        if current_equation:
            parsed.equations.append(current_equation)

    def _parse_time_config(self, parsed: ParsedGNN, content: List[str]) -> Any:
        """Parse Time section."""
        for line in content:
            line = line.strip()
            if not line or self.COMMENT_PATTERN.match(line):
                continue

            if "=" in line:
                key, value = line.split("=", 1)
                parsed.time_config[key.strip()] = value.strip()
            else:
                # Simple time type specification
                parsed.time_config["type"] = line

    def _parse_ontology_mappings(self, parsed: ParsedGNN, content: List[str]) -> Any:
        """Parse ActInfOntologyAnnotation section."""
        for line in content:
            line = line.strip()
            if not line or self.COMMENT_PATTERN.match(line):
                continue

            match = self.ONTOLOGY_PATTERN.match(line)
            if match:
                variable = match.group(1)
                ontology_term = match.group(3)
                parsed.ontology_mappings[variable] = ontology_term

    def _parse_signature(self, content: str) -> Dict[str, str]:
        """Parse Signature section."""
        signature: dict[Any, Any] = {}
        for line in content.split("\n"):
            line = line.strip()
            if ":" in line:
                key, value = line.split(":", 1)
                signature[key.strip()] = value.strip()
        return signature
