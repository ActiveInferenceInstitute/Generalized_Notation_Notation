"""
JSON Parser for GNN Data Interchange

This module provides parsing capabilities for JSON files that specify
GNN models in structured data format.

Author: @docxology
Date: 2025-01-11
License: MIT
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .common import (
    BaseGNNParser,
    Connection,
    ConnectionType,
    DataType,
    Equation,
    GNNInternalRepresentation,
    OntologyMapping,
    Parameter,
    ParseResult,
    TimeSpecification,
    Variable,
    VariableType,
)

logger = logging.getLogger(__name__)


class JSONGNNParser(BaseGNNParser):
    """Parser for JSON data interchange format."""

    def __init__(self) -> None:
        """Initialize the JSON parser."""
        super().__init__()

    def parse_file(self, file_path: str) -> ParseResult:
        """Parse a JSON file containing GNN specifications."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            return self._parse_json_data(data)

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {file_path}: {e}")
            result = ParseResult(
                model=self.create_empty_model("Failed JSON Parse"), success=False
            )
            result.add_error(f"Invalid JSON format: {e}")
            return result

        except Exception as e:
            logger.error(f"Error parsing JSON file {file_path}: {e}")
            result = ParseResult(
                model=self.create_empty_model("Failed JSON Parse"), success=False
            )
            result.add_error(f"Failed to parse JSON file: {e}")
            return result

    def parse_string(self, content: str) -> ParseResult:
        """Parse JSON content from string."""
        try:
            # Check if this content is likely JSON
            content = content.strip()
            if not (content.startswith("{") or content.startswith("[")):
                # This doesn't look like JSON, might be markdown or another format
                result = ParseResult(
                    model=self.create_empty_model("Failed JSON Parse"), success=False
                )
                result.add_error(
                    "Content doesn't appear to be valid JSON (missing opening brace or bracket)"
                )
                return result

            data = json.loads(content)
            return self._parse_json_data(data)

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            result = ParseResult(
                model=self.create_empty_model("Failed JSON Parse"), success=False
            )
            result.add_error(f"Invalid JSON format: {e}")
            return result

        except Exception as e:
            logger.error(f"Error parsing JSON content: {str(e)}")
            result = ParseResult(
                model=self.create_empty_model("Failed JSON Parse"), success=False
            )
            result.add_error(f"Failed to parse JSON content: {str(e)}")
            return result

    def _parse_json_data(self, data: Dict[str, Any]) -> ParseResult:
        """Parse JSON data into GNN internal representation."""
        try:
            model = self._convert_json_to_model(data)
            return ParseResult(model=model, success=True)

        except Exception as e:
            logger.error(f"Error converting JSON to model: {e}")
            result = ParseResult(
                model=self.create_empty_model("Failed JSON Conversion"), success=False
            )
            result.add_error(f"Failed to convert JSON to model: {e}")
            return result

    def _convert_json_to_model(self, data: Dict[str, Any]) -> GNNInternalRepresentation:
        """Convert JSON data to GNN internal representation."""

        # Extract basic information
        model_name = data.get("model_name", "JSONGNNModel")
        version = data.get("version", "1.0")
        annotation = data.get("annotation", "")

        # Create model
        model = GNNInternalRepresentation(
            model_name=model_name, version=version, annotation=annotation
        )

        # Parse variables
        if "variables" in data:
            model.variables = self._parse_variables(data["variables"])

        # Parse connections
        if "connections" in data:
            model.connections = self._parse_connections(data["connections"])

        # Parse parameters
        if "parameters" in data:
            model.parameters = self._parse_parameters(data["parameters"])

        # Parse equations
        if "equations" in data:
            model.equations = self._parse_equations(data["equations"])

        # Parse time specification
        if "time_specification" in data:
            model.time_specification = self._parse_time_specification(
                data["time_specification"]
            )

        # Parse ontology mappings
        if "ontology_mappings" in data:
            model.ontology_mappings = self._parse_ontology_mappings(
                data["ontology_mappings"]
            )

        # Parse metadata
        if "created_at" in data:
            try:
                model.created_at = datetime.fromisoformat(data["created_at"])
            except (ValueError, TypeError) as e:
                logger.debug(f"Skipping malformed created_at metadata: {e}")

        if "modified_at" in data:
            try:
                model.modified_at = datetime.fromisoformat(data["modified_at"])
            except (ValueError, TypeError) as e:
                logger.debug(f"Skipping malformed modified_at metadata: {e}")

        model.checksum = data.get("checksum")

        # Parse extensions
        if "extensions" in data:
            model.extensions = data["extensions"]

        # Parse raw sections
        if "raw_sections" in data:
            model.raw_sections = data["raw_sections"]

        return model

    def _parse_variables(self, variables_data: List[Dict[str, Any]]) -> List[Variable]:
        """Parse variables from JSON data."""
        variables: list[Any] = []

        for var_data in variables_data:
            try:
                # Handle both dict format and direct Variable format
                if isinstance(var_data, dict):
                    name = var_data.get("name", "")
                    var_type_str = var_data.get("var_type", "hidden_state")
                    data_type_str = var_data.get("data_type", "continuous")
                    dimensions = var_data.get("dimensions", [])
                    description = var_data.get("description", "")
                    constraints = var_data.get("constraints", {})

                    # Convert string enums to enum values
                    try:
                        var_type = VariableType(var_type_str)
                    except ValueError:
                        var_type = VariableType.HIDDEN_STATE

                    try:
                        data_type = DataType(data_type_str)
                    except ValueError:
                        data_type = DataType.CONTINUOUS

                    variable = Variable(
                        name=name,
                        var_type=var_type,
                        dimensions=dimensions,
                        data_type=data_type,
                        description=description,
                        constraints=constraints,
                    )

                    variables.append(variable)

            except Exception as e:
                logger.warning(f"Failed to parse variable {var_data}: {e}")
                continue

        return variables

    def _parse_connections(
        self, connections_data: List[Dict[str, Any]]
    ) -> List[Connection]:
        """Parse connections from JSON data."""
        connections: list[Any] = []

        for conn_data in connections_data:
            try:
                source_variables = conn_data.get("source_variables", [])
                target_variables = conn_data.get("target_variables", [])
                connection_type_str = conn_data.get("connection_type", "directed")
                weight = conn_data.get("weight")
                description = conn_data.get("description", "")

                try:
                    connection_type = ConnectionType(connection_type_str)
                except ValueError:
                    connection_type = ConnectionType.DIRECTED

                connection = Connection(
                    source_variables=source_variables,
                    target_variables=target_variables,
                    connection_type=connection_type,
                    weight=weight,
                    annotation=conn_data.get("annotation"),
                    description=description,
                )

                connections.append(connection)

            except Exception as e:
                logger.warning(f"Failed to parse connection {conn_data}: {e}")
                continue

        return connections

    def _parse_parameters(
        self, parameters_data: List[Dict[str, Any]]
    ) -> List[Parameter]:
        """Parse parameters from JSON data."""
        parameters: list[Any] = []

        for param_data in parameters_data:
            try:
                name = param_data.get("name", "")
                value = param_data.get("value")
                type_hint = param_data.get("type_hint")
                description = param_data.get("description", "")

                parameter = Parameter(
                    name=name, value=value, type_hint=type_hint, description=description
                )

                parameters.append(parameter)

            except Exception as e:
                logger.warning(f"Failed to parse parameter {param_data}: {e}")
                continue

        return parameters

    def _parse_equations(self, equations_data: List[Dict[str, Any]]) -> List[Equation]:
        """Parse equations from JSON data."""
        equations: list[Any] = []

        for eq_data in equations_data:
            try:
                label = eq_data.get("label")
                content = eq_data.get("content", "")
                format_type = eq_data.get("format", "latex")
                description = eq_data.get("description", "")

                equation = Equation(
                    label=label,
                    content=content,
                    format=format_type,
                    description=description,
                )

                equations.append(equation)

            except Exception as e:
                logger.warning(f"Failed to parse equation {eq_data}: {e}")
                continue

        return equations

    def _parse_time_specification(
        self, time_data: Dict[str, Any]
    ) -> Optional[TimeSpecification]:
        """Parse time specification from JSON data."""
        try:
            time_type = time_data.get("time_type", "Static")
            discretization = time_data.get("discretization")
            horizon = time_data.get("horizon")

            return TimeSpecification(
                time_type=time_type,
                discretization=discretization,
                horizon=horizon,
                step_size=time_data.get("step_size"),
            )

        except Exception as e:
            logger.warning(f"Failed to parse time specification {time_data}: {e}")
            return None

    def _parse_ontology_mappings(
        self, mappings_data: List[Dict[str, Any]]
    ) -> List[OntologyMapping]:
        """Parse ontology mappings from JSON data."""
        mappings: list[Any] = []

        for mapping_data in mappings_data:
            try:
                variable_name = mapping_data.get("variable_name", "")
                ontology_term = mapping_data.get("ontology_term", "")
                description = mapping_data.get("description", "")

                mapping = OntologyMapping(
                    variable_name=variable_name,
                    ontology_term=ontology_term,
                    description=description,
                )

                mappings.append(mapping)

            except Exception as e:
                logger.warning(f"Failed to parse ontology mapping {mapping_data}: {e}")
                continue

        return mappings

    def get_supported_extensions(self) -> List[str]:
        """Get supported file extensions."""
        return [".json"]
