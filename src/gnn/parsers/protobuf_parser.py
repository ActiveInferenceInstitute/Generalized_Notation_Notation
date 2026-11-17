"""
GNN Protobuf Parser - Enhanced with Complete Model Extraction

This module provides enhanced parsing capabilities for Protocol Buffer files
that contain GNN model specifications with complete round-trip fidelity.

Author: @docxology
Date: 2025-01-11
License: MIT
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

logger = logging.getLogger(__name__)

from .common import (
    BaseGNNParser,
    Connection,
    ConnectionType,
    DataType,
    GNNInternalRepresentation,
    OntologyMapping,
    Parameter,
    ParseResult,
    TimeSpecification,
    Variable,
    VariableType,
)


class ProtobufGNNParser(BaseGNNParser):
    """Enhanced parser for Protocol Buffer (.proto) files containing GNN models."""

    def __init__(self) -> None:
        """Initialize the instance."""
        super().__init__()
        self.model_cache: dict[str, GNNInternalRepresentation] = {}

    def get_supported_extensions(self) -> List[str]:
        """Return supported extensions."""
        return [".proto"]

    def parse_file(self, file_path: str) -> ParseResult:
        """Parse Protocol Buffer file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
            return self.parse_string(content)
        except Exception as e:
            result = ParseResult(model=self.create_empty_model())
            result.add_error(f"Failed to read protobuf file: {e}")
            return result

    def parse_string(self, content: str) -> ParseResult:
        """Parse Protocol Buffer content string."""
        return self.parse_content(content)

    def parse_content(
        self, content: str, file_path: Optional[Path] = None
    ) -> ParseResult:
        """Parse Protocol Buffer content with enhanced model extraction."""
        try:
            # First try to extract embedded JSON model data for perfect round-trip
            embedded_data = self._extract_embedded_model_data(content)
            if embedded_data:
                result = ParseResult(model=self.create_empty_model())
                return self._parse_from_embedded_data(embedded_data, result)

            model = GNNInternalRepresentation(model_name="ProtobufGNNModel")
            errors: list[Any] = []
            warnings: list[Any] = []

            # Parse model metadata
            model_name = self._extract_model_name(content)
            model.model_name = model_name or "ProtobufGNNModel"

            # Parse model annotation/documentation
            model.annotation = self._extract_model_annotation(content)

            # Parse variables with complete information
            variables = self._parse_variables_enhanced(content)
            model.variables = variables

            # Parse connections with proper relationship mapping
            connections = self._parse_connections_enhanced(content)
            model.connections = connections

            # Parse parameters with value preservation
            parameters = self._parse_parameters_enhanced(content)
            model.parameters = parameters

            # Parse equations if present
            equations = self._parse_equations(content)
            model.equations = equations

            # Parse time specification
            time_spec = self._parse_time_specification(content)
            model.time_specification = time_spec

            # Parse ontology mappings
            ontology_mappings = self._parse_ontology_mappings(content)
            model.ontology_mappings = ontology_mappings

            # Add model validation
            validation_errors = self._validate_model_completeness(model, content)
            errors.extend(validation_errors)

            success = len(errors) == 0

            result = ParseResult(
                model=model,
                success=success,
                errors=errors,
                warnings=warnings,
                source_file=str(file_path) if file_path else None,
            )
            return result

        except Exception as e:
            import traceback

            error_msg = f"Protobuf parsing failed: {str(e)}"
            traceback_msg = traceback.format_exc()

            return ParseResult(
                model=self.create_empty_model(),
                success=False,
                errors=[error_msg, f"Traceback: {traceback_msg}"],
                warnings=[],
            )

    def _extract_model_name(self, content: str) -> Optional[str]:
        """Extract model name from protobuf content."""
        # Look for package name
        package_match = re.search(r"package\s+([^;]+);", content)
        if package_match:
            return package_match.group(1).replace(".", "_")

        # Look for service name
        service_match = re.search(r"service\s+(\w+)", content)
        if service_match:
            return service_match.group(1)

        # Look for main message name
        message_match = re.search(r"message\s+(\w+Model)", content)
        if message_match:
            return message_match.group(1).replace("Model", "")

        # Look for any message
        any_message_match = re.search(r"message\s+(\w+)", content)
        if any_message_match:
            return any_message_match.group(1)

        return None

    def _extract_model_annotation(self, content: str) -> str:
        """Extract model annotation from comments."""
        annotations: list[Any] = []

        # Look for file-level comments
        file_comments = re.findall(r"//\s*(.+)", content)
        if file_comments:
            annotations.extend(file_comments[:3])  # Take first 3 comments

        # Look for block comments
        block_comments = re.findall(r"/\*\s*(.*?)\s*\*/", content, re.DOTALL)
        for comment in block_comments:
            clean_comment = re.sub(r"\s+", " ", comment.strip())
            if clean_comment and len(clean_comment) > 10:
                annotations.append(clean_comment)

        return " ".join(annotations) if annotations else ""

    def _parse_variables_enhanced(self, content: str) -> List[Variable]:
        """Parse variables with complete information preservation."""
        variables: list[Any] = []

        # Parse from Variable messages
        variable_messages = re.findall(
            r"message\s+Variable\s*{([^}]+)}", content, re.DOTALL
        )
        for msg_content in variable_messages:
            var_fields = re.findall(r"(\w+)\s+(\w+)\s*=\s*\d+;", msg_content)
            for field_type, field_name in var_fields:
                if field_name in ["name", "type", "dimensions", "data_type"]:
                    continue  # Skip metadata fields

                var = Variable(
                    name=field_name,
                    var_type=self._map_protobuf_type_to_variable_type(field_type),
                    data_type=self._map_protobuf_type_to_data_type(field_type),
                    dimensions=[],
                )
                variables.append(var)

        # Parse from individual field declarations
        field_declarations = re.findall(
            r"(repeated\s+)?(\w+)\s+(\w+)\s*=\s*\d+;", content
        )
        for is_repeated, field_type, field_name in field_declarations:
            if field_name in [
                "variables",
                "connections",
                "parameters",
                "name",
                "annotation",
            ]:
                continue  # Skip container fields

            # Skip if already found in Variable message
            if any(var.name == field_name for var in variables):
                continue

            var = Variable(
                name=field_name,
                var_type=self._map_protobuf_type_to_variable_type(field_type),
                data_type=self._map_protobuf_type_to_data_type(field_type),
                dimensions=[0] if is_repeated else [],
            )
            variables.append(var)

        # Parse from embedded structure if available
        model_content = self._extract_embedded_model_data(content)
        if model_content and "variables" in model_content:
            for var_data in model_content["variables"]:
                if isinstance(var_data, dict) and "name" in var_data:
                    from .common import safe_enum_convert

                    var = Variable(
                        name=var_data["name"],
                        var_type=safe_enum_convert(
                            VariableType,
                            var_data.get("var_type", "hidden_state"),
                            VariableType.HIDDEN_STATE,
                        )
                        or VariableType.HIDDEN_STATE,
                        data_type=safe_enum_convert(
                            DataType,
                            var_data.get("data_type", "categorical"),
                            DataType.CATEGORICAL,
                        )
                        or DataType.CATEGORICAL,
                        dimensions=var_data.get("dimensions", []),
                    )
                    # Avoid duplicates
                    if not any(v.name == var.name for v in variables):
                        variables.append(var)

        return variables

    def _parse_connections_enhanced(self, content: str) -> List[Connection]:
        """Parse connections with complete relationship preservation."""
        connections: list[Any] = []

        # Parse from Connection messages
        connection_messages = re.findall(
            r"message\s+Connection\s*{([^}]+)}", content, re.DOTALL
        )
        for msg_content in connection_messages:
            # Extract connection fields
            source_match = re.search(
                r"repeated\s+string\s+source_variables", msg_content
            )
            target_match = re.search(
                r"repeated\s+string\s+target_variables", msg_content
            )
            re.search(r"string\s+connection_type", msg_content)

            if source_match and target_match:
                # This indicates the structure exists
                logger.debug("Detected protobuf connection field structure")

        # Parse from embedded model data
        model_content = self._extract_embedded_model_data(content)
        if model_content and "connections" in model_content:
            for conn_data in model_content["connections"]:
                if isinstance(conn_data, dict):
                    connection = Connection(
                        annotation=conn_data.get("annotation"),
                        source_variables=conn_data.get("source_variables", []),
                        target_variables=conn_data.get("target_variables", []),
                        connection_type=ConnectionType(
                            conn_data.get("connection_type", "directed")
                        ),
                    )
                    connections.append(connection)

        # Parse from comment annotations if available
        connection_comments = re.findall(r"//\s*Connection:\s*(.+)", content)
        for comment in connection_comments:
            # Parse connection strings like "A --> B" or "s --directed--> s_prime"
            conn_match = re.match(r"(\w+)\s*--(\w+)?-->\s*(\w+)", comment)
            if conn_match:
                source, conn_type, target = conn_match.groups()
                connection = Connection(
                    source_variables=[source],
                    target_variables=[target],
                    connection_type=ConnectionType(conn_type or "directed"),
                )
                connections.append(connection)

        return connections

    def _parse_parameters_enhanced(self, content: str) -> List[Parameter]:
        """Parse parameters with complete value preservation."""
        parameters: list[Any] = []

        # Parse from Parameter messages
        re.findall(r"message\s+Parameter\s*{([^}]+)}", content, re.DOTALL)

        # Parse from embedded model data
        model_content = self._extract_embedded_model_data(content)
        if model_content and "parameters" in model_content:
            for param_data in model_content["parameters"]:
                if isinstance(param_data, dict) and "name" in param_data:
                    parameter = Parameter(
                        name=param_data["name"],
                        value=param_data.get("value"),
                        type_hint=param_data.get("param_type", "constant"),
                        description="Protobuf parameter",
                    )
                    parameters.append(parameter)

        # Parse from comment annotations
        param_comments = re.findall(r"//\s*Parameter:\s*(\w+)\s*=\s*(.+)", content)
        for name, value in param_comments:
            parameter = Parameter(
                name=name,
                value=self._parse_parameter_value(value.strip()),
                type_hint="constant",
                description="Protobuf comment parameter",
            )
            parameters.append(parameter)

        return parameters

    def _parse_equations(self, content: str) -> List:
        """Parse equations from protobuf content."""
        equations: list[Any] = []

        # Parse equation comments
        eq_comments = re.findall(r"//\s*Equation:\s*(.+)", content)
        for eq_text in eq_comments:
            equations.append({"equation": eq_text.strip()})

        return equations

    def _parse_time_specification(self, content: str) -> Optional[TimeSpecification]:
        """Parse time specification from protobuf content."""
        # Look for time-related messages or comments
        time_messages = re.findall(
            r"message\s+TimeSpecification\s*{([^}]+)}", content, re.DOTALL
        )
        if time_messages:
            return TimeSpecification(time_type="discrete", horizon=1)

        time_comments = re.findall(r"//\s*Time:\s*(.+)", content)
        if time_comments:
            return TimeSpecification(time_type="discrete")

        return None

    def _parse_ontology_mappings(self, content: str) -> List:
        """Parse ontology mappings from protobuf content."""
        mappings: list[Any] = []

        # Parse ontology comments
        onto_comments = re.findall(r"//\s*Ontology:\s*(\w+)\s*->\s*(.+)", content)
        for var_name, ontology_term in onto_comments:
            mappings.append(
                {"variable_name": var_name, "ontology_term": ontology_term.strip()}
            )

        return mappings

    def _extract_embedded_model_data(self, content: str) -> Optional[Dict]:
        """Extract embedded JSON model data from protobuf comments."""
        # Look for JSON data in comments
        json_matches = re.findall(r"//\s*JSON:\s*({.+?})", content, re.DOTALL)
        for json_str in json_matches:
            try:
                return cast("dict[Any, Any] | None", json.loads(json_str))
            except json.JSONDecodeError as e:
                logger.debug("Malformed JSON in comment, trying next: %s", e)
                continue

        # Look for embedded data structure in comments
        data_matches = re.findall(
            r"/\*\s*MODEL_DATA:\s*({.+?})\s*\*/", content, re.DOTALL
        )
        for data_str in data_matches:
            try:
                return cast("dict[Any, Any] | None", json.loads(data_str))
            except json.JSONDecodeError as e:
                logger.debug("Malformed MODEL_DATA JSON, trying next: %s", e)
                continue

        return None

    def _parse_from_embedded_data(
        self, data: Dict[str, Any], result: ParseResult
    ) -> ParseResult:
        """Parse model from embedded JSON data for perfect round-trip."""
        try:
            # Restore original model data
            result.model.model_name = data.get("model_name", "ProtobufModel")
            result.model.annotation = data.get("annotation", "")
            result.model.version = data.get("version", "1.0")

            # Restore variables
            for var_data in data.get("variables", []):
                from .common import safe_enum_convert

                variable = Variable(
                    name=var_data["name"],
                    var_type=safe_enum_convert(
                        VariableType,
                        var_data.get("var_type", "hidden_state"),
                        VariableType.HIDDEN_STATE,
                    )
                    or VariableType.HIDDEN_STATE,
                    data_type=safe_enum_convert(
                        DataType,
                        var_data.get("data_type", "categorical"),
                        DataType.CATEGORICAL,
                    )
                    or DataType.CATEGORICAL,
                    dimensions=var_data.get("dimensions", []),
                    description=var_data.get("description", ""),
                )
                result.model.variables.append(variable)

            # Restore connections
            for conn_data in data.get("connections", []):
                connection = Connection(
                    annotation=conn_data.get("annotation"),
                    source_variables=conn_data.get("source_variables", []),
                    target_variables=conn_data.get("target_variables", []),
                    connection_type=ConnectionType(
                        conn_data.get("connection_type", "directed")
                    ),
                    description=conn_data.get("description", ""),
                )
                result.model.connections.append(connection)

            # Restore parameters
            for param_data in data.get("parameters", []):
                parameter = Parameter(
                    name=param_data["name"],
                    value=param_data["value"],
                    type_hint=param_data.get("param_type", "constant"),
                    description=param_data.get("description", "Protobuf parameter"),
                )
                result.model.parameters.append(parameter)

            # Restore time specification for Protobuf parser
            if "time_specification" in data and data["time_specification"]:
                time_spec_data = data["time_specification"]
                if isinstance(time_spec_data, dict):
                    result.model.time_specification = TimeSpecification(
                        time_type=str(
                            time_spec_data.get(
                                "time_type", time_spec_data.get("type", "Static")
                            )
                        ),
                        discretization=(
                            str(time_spec_data["discretization"])
                            if time_spec_data.get("discretization") is not None
                            else None
                        ),
                        horizon=cast(
                            Optional[int | str], time_spec_data.get("horizon")
                        ),
                        step_size=(
                            float(time_spec_data["step_size"])
                            if time_spec_data.get("step_size") is not None
                            else None
                        ),
                    )

            # Restore ontology mappings for Protobuf parser
            if "ontology_mappings" in data and data["ontology_mappings"]:
                ontology_data = data["ontology_mappings"]
                if isinstance(ontology_data, list):
                    result.model.ontology_mappings = [
                        OntologyMapping(
                            variable_name=str(mapping.get("variable_name", "")),
                            ontology_term=str(mapping.get("ontology_term", "")),
                            description=mapping.get("description"),
                        )
                        for mapping in ontology_data
                        if isinstance(mapping, dict)
                    ]

            # Keep the original annotation without modification for perfect round-trip
            result.success = True

        except Exception as e:
            result.add_error(f"Failed to parse embedded data: {e}")

        return result

    def _map_protobuf_type_to_variable_type(self, proto_type: str) -> VariableType:
        """Map protobuf types to GNN variable types."""
        mapping: dict[str, Any] = {
            "string": VariableType.OBSERVATION,
            "int32": VariableType.HIDDEN_STATE,
            "int64": VariableType.HIDDEN_STATE,
            "float": VariableType.HIDDEN_STATE,  # Changed from PARAMETER to more appropriate default
            "double": VariableType.HIDDEN_STATE,  # Changed from PARAMETER to more appropriate default
            "bool": VariableType.POLICY,
            "bytes": VariableType.OBSERVATION,
        }
        return cast(
            "VariableType", mapping.get(proto_type.lower(), VariableType.HIDDEN_STATE)
        )

    def _map_protobuf_type_to_data_type(self, proto_type: str) -> DataType:
        """Map protobuf types to GNN data types."""
        mapping: dict[str, Any] = {
            "string": DataType.CATEGORICAL,
            "int32": DataType.INTEGER,
            "int64": DataType.INTEGER,
            "float": DataType.FLOAT,
            "double": DataType.FLOAT,
            "bool": DataType.BINARY,
            "bytes": DataType.COMPLEX,
        }
        return cast("DataType", mapping.get(proto_type.lower(), DataType.CATEGORICAL))

    def _parse_parameter_value(self, value_str: str) -> Any:
        """Parse parameter value from string."""
        value_str = value_str.strip()

        # Try to parse as number
        try:
            if "." in value_str:
                return float(value_str)
            else:
                return int(value_str)
        except ValueError as e:
            logger.debug(
                "Value not numeric, falling through to boolean/string check: %s", e
            )

        # Try to parse as boolean
        if value_str.lower() in ["true", "false"]:
            return value_str.lower() == "true"

        # Return as string
        return value_str

    def _validate_model_completeness(
        self, model: GNNInternalRepresentation, content: str
    ) -> List[str]:
        """Validate that the model was completely parsed."""
        errors: list[Any] = []

        if not model.model_name:
            errors.append("Model name not found in protobuf content")

        if not model.variables:
            errors.append("No variables found in protobuf content")

        # Check for presence of expected sections
        expected_sections: list[Any] = ["message", "package", "syntax"]
        found_sections = [
            section for section in expected_sections if section in content.lower()
        ]

        if not found_sections:
            errors.append("No valid protobuf structure found")

        return errors
