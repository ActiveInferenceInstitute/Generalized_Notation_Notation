"""
YAML Parser for GNN Configuration Format

This module provides parsing capabilities for YAML files that specify
GNN models in human-readable configuration format.

Author: @docxology
Date: 2025-01-11
License: MIT
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

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


class YAMLGNNParser(BaseGNNParser):
    """Parser for YAML configuration format."""

    def __init__(self) -> None:
        """Initialize the YAML parser."""
        super().__init__()
        if not HAS_YAML:
            logger.warning("PyYAML not available, using recovery YAML parser")

    def parse_file(self, file_path: str) -> ParseResult:
        """Parse a YAML file containing GNN specifications."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            return self.parse_string(content)

        except Exception as e:
            logger.error(f"Error parsing YAML file {file_path}: {e}")
            result = ParseResult(
                model=self.create_empty_model("Failed YAML Parse"), success=False
            )
            result.add_error(f"Failed to parse YAML file: {e}")
            return result

    def parse_string(self, content: str) -> ParseResult:
        """Parse YAML content from string."""
        content = content.strip()

        try:
            # Try to parse with PyYAML if available
            if HAS_YAML:
                data = yaml.safe_load(content)
                if not isinstance(data, dict):
                    raise ValueError("YAML content must produce a dictionary")
                return self._parse_yaml_data(data)
            else:
                # Use simplified parsing
                data = self._fallback_yaml_parse(content)
                return self._parse_yaml_data(data)

        except Exception as e:
            logger.error(f"Error parsing YAML content: {e}")
            result = ParseResult(
                model=self.create_empty_model("Failed YAML Parse"), success=False
            )
            result.add_error(f"Failed to parse YAML content: {str(e)}")
            return result

    def _fallback_yaml_parse(self, content: str) -> Dict[str, Any]:
        """Recovery YAML parser when PyYAML is not available."""
        logger.info("Using recovery YAML parser")

        data: dict[Any, Any] = {}
        current_section = None
        lines = content.split("\n")

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Handle top-level keys
            if ":" in line and not line.startswith(" ") and not line.startswith("-"):
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()

                if value:
                    # Try to parse simple values
                    data[key] = self._parse_yaml_value(value)
                else:
                    # This is a section header
                    current_section = key
                    data[key] = []

            # Handle list items
            elif line.startswith("-") and current_section:
                item_content = line[1:].strip()
                if ":" in item_content:
                    # Dictionary item
                    item_dict: dict[Any, Any] = {}
                    parts = item_content.split(",")
                    for part in parts:
                        if ":" in part:
                            k, v = part.split(":", 1)
                            item_dict[k.strip()] = self._parse_yaml_value(v.strip())
                    data[current_section].append(item_dict)
                else:
                    data[current_section].append(self._parse_yaml_value(item_content))

        return data

    def _parse_yaml_value(self, value: str) -> Any:
        """Parse a YAML value from string."""
        value = value.strip()

        # Handle quoted strings
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            return value[1:-1]

        # Handle booleans
        if value.lower() in ["true", "yes", "on"]:
            return True
        elif value.lower() in ["false", "no", "off"]:
            return False

        # Handle null
        if value.lower() in ["null", "none", "~"]:
            return None

        # Handle numbers
        try:
            if "." in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            logger.debug(
                "Value is not numeric, treating as string: %s",
                value[:80] if value else value,
            )

        # Handle lists
        if value.startswith("[") and value.endswith("]"):
            items = value[1:-1].split(",")
            return [
                self._parse_yaml_value(item.strip()) for item in items if item.strip()
            ]

        # Return as string
        return value

    def _parse_yaml_data(self, data: Dict[str, Any]) -> ParseResult:
        """Parse YAML data into GNN internal representation."""
        try:
            model = self._convert_yaml_to_model(data)
            return ParseResult(model=model, success=True)

        except Exception as e:
            logger.error(f"Error converting YAML to model: {e}")
            result = ParseResult(
                model=self.create_empty_model("Failed YAML Conversion"), success=False
            )
            result.add_error(f"Failed to convert YAML to model: {e}")
            return result

    def _convert_yaml_to_model(self, data: Dict[str, Any]) -> GNNInternalRepresentation:
        """Convert YAML data to GNN internal representation."""

        # Extract basic information
        model_name = data.get("model_name", data.get("name", "YAMLGNNModel"))
        version = data.get("version", "1.0")
        annotation = data.get("annotation", data.get("description", ""))

        # Create model
        model = GNNInternalRepresentation(
            model_name=model_name, version=version, annotation=annotation
        )

        # Parse variables
        if "variables" in data:
            model.variables = self._parse_yaml_variables(data["variables"])
        elif "state_space" in data:
            model.variables = self._parse_yaml_variables(data["state_space"])

        # Parse connections
        if "connections" in data:
            model.connections = self._parse_yaml_connections(data["connections"])

        # Parse parameters
        if "parameters" in data:
            model.parameters = self._parse_yaml_parameters(data["parameters"])
        elif "initial_parameterization" in data:
            model.parameters = self._parse_yaml_parameters(
                data["initial_parameterization"]
            )

        # Parse equations
        if "equations" in data:
            model.equations = self._parse_yaml_equations(data["equations"])

        # Parse time specification
        if "time" in data:
            model.time_specification = self._parse_yaml_time_specification(data["time"])
        elif "time_specification" in data:
            model.time_specification = self._parse_yaml_time_specification(
                data["time_specification"]
            )

        # Parse ontology mappings
        if "ontology_mappings" in data:
            model.ontology_mappings = self._parse_yaml_ontology_mappings(
                data["ontology_mappings"]
            )
        elif "actinf_ontology_annotation" in data:
            model.ontology_mappings = self._parse_yaml_ontology_mappings(
                data["actinf_ontology_annotation"]
            )

        # Parse metadata
        if "created_at" in data:
            try:
                if isinstance(data["created_at"], str):
                    model.created_at = datetime.fromisoformat(data["created_at"])
                else:
                    model.created_at = data["created_at"]
            except (ValueError, TypeError) as e:
                logger.debug(f"Could not parse created_at timestamp: {e}")

        if "modified_at" in data:
            try:
                if isinstance(data["modified_at"], str):
                    model.modified_at = datetime.fromisoformat(data["modified_at"])
                else:
                    model.modified_at = data["modified_at"]
            except (ValueError, TypeError) as e:
                logger.debug(f"Could not parse modified_at timestamp: {e}")

        model.checksum = data.get("checksum")

        # Parse extensions
        if "extensions" in data:
            model.extensions = data["extensions"]

        # Parse raw sections
        if "raw_sections" in data:
            model.raw_sections = data["raw_sections"]

        return model

    def _parse_yaml_variables(
        self, variables_data: Union[List, Dict]
    ) -> List[Variable]:
        """Parse variables from YAML data."""
        variables: list[Any] = []

        if isinstance(variables_data, list):
            # List format
            for var_data in variables_data:
                variable = self._parse_yaml_variable(var_data)
                if variable:
                    variables.append(variable)
        elif isinstance(variables_data, dict):
            # Dictionary format
            for var_name, var_spec in variables_data.items():
                variable = self._parse_yaml_variable_dict(var_name, var_spec)
                if variable:
                    variables.append(variable)

        return variables

    def _parse_yaml_variable(
        self, var_data: Union[Dict[str, Any], str]
    ) -> Optional[Variable]:
        """Parse a single variable from YAML data."""
        try:
            data: Any = var_data
            if isinstance(data, str):
                # Simple string format: "name[dimensions],type"
                return self._parse_variable_string(data)

            elif isinstance(data, dict):
                name = data.get("name", "")
                var_type_str = data.get("type", data.get("var_type", "hidden_state"))
                data_type_str = data.get("data_type", "continuous")
                dimensions = data.get("dimensions", [])
                description = data.get("description", "")
                constraints = data.get("constraints", {})

                # Convert string enums to enum values
                try:
                    var_type = VariableType(var_type_str)
                except ValueError:
                    var_type = VariableType.HIDDEN_STATE

                try:
                    data_type = DataType(data_type_str)
                except ValueError:
                    data_type = DataType.CONTINUOUS

                return Variable(
                    name=name,
                    var_type=var_type,
                    dimensions=dimensions,
                    data_type=data_type,
                    description=description,
                    constraints=constraints,
                )
            return None

        except Exception as e:
            logger.warning(f"Failed to parse YAML variable {var_data}: {e}")
            return None

    def _parse_yaml_variable_dict(self, name: str, spec: Any) -> Optional[Variable]:
        """Parse variable from dictionary format."""
        try:
            if isinstance(spec, str):
                # Simple format: name: "type" or name: "[dim],type"
                return self._parse_variable_string(f"{name},{spec}")

            elif isinstance(spec, dict):
                var_type_str = spec.get("type", "hidden_state")
                data_type_str = spec.get("data_type", "continuous")
                dimensions = spec.get("dimensions", [])
                description = spec.get("description", "")

                try:
                    var_type = VariableType(var_type_str)
                except ValueError:
                    var_type = VariableType.HIDDEN_STATE

                try:
                    data_type = DataType(data_type_str)
                except ValueError:
                    data_type = DataType.CONTINUOUS

                return Variable(
                    name=name,
                    var_type=var_type,
                    dimensions=dimensions,
                    data_type=data_type,
                    description=description,
                )
            return None

        except Exception as e:
            logger.warning(f"Failed to parse YAML variable {name}: {spec}, {e}")
            return None

    def _parse_variable_string(self, var_str: str) -> Optional[Variable]:
        """Parse variable from string format like 'name[dims],type'."""
        try:
            from .common import (
                infer_variable_type,
                normalize_variable_name,
                parse_dimensions,
            )

            parts = var_str.split(",")
            if len(parts) < 2:
                return None

            name_part = parts[0].strip()
            type_part = parts[1].strip()

            # Extract dimensions if present
            dimensions: list[Any] = []
            if "[" in name_part and "]" in name_part:
                name, dim_str = name_part.split("[", 1)
                dim_str = dim_str.rstrip("]")
                dimensions = parse_dimensions(dim_str)
            else:
                name = name_part

            name = normalize_variable_name(name)

            # Infer variable type
            var_type = infer_variable_type(name)

            # Parse data type
            try:
                data_type = DataType(type_part)
            except ValueError:
                data_type = DataType.CONTINUOUS

            return Variable(
                name=name,
                var_type=var_type,
                dimensions=dimensions,
                data_type=data_type,
                description=f"Parsed from YAML: {var_str}",
            )

        except Exception as e:
            logger.warning(f"Failed to parse variable string {var_str}: {e}")
            return None

    def _parse_yaml_connections(
        self, connections_data: List[Union[Dict, str]]
    ) -> List[Connection]:
        """Parse connections from YAML data."""
        connections: list[Any] = []

        for conn_data in connections_data:
            connection = self._parse_yaml_connection(conn_data)
            if connection:
                connections.append(connection)

        return connections

    def _parse_yaml_connection(
        self, conn_data: Union[Dict[str, Any], str]
    ) -> Optional[Connection]:
        """Parse a single connection from YAML data."""
        try:
            if isinstance(conn_data, str):
                # Simple string format: "source>target" or "source-target"
                return self._parse_connection_string(conn_data)

            elif isinstance(conn_data, dict):
                # Try multiple field names for source and target
                source_variables = (
                    conn_data.get("source_variables")
                    or conn_data.get("source")
                    or conn_data.get("from")
                    or []
                )
                target_variables = (
                    conn_data.get("target_variables")
                    or conn_data.get("target")
                    or conn_data.get("to")
                    or []
                )
                connection_type_str = conn_data.get(
                    "connection_type", conn_data.get("type", "directed")
                )
                weight = conn_data.get("weight")
                description = conn_data.get("description", "")

                if isinstance(source_variables, str):
                    source_variables = [source_variables]
                if isinstance(target_variables, str):
                    target_variables = [target_variables]

                try:
                    connection_type = ConnectionType(connection_type_str)
                except ValueError:
                    connection_type = ConnectionType.DIRECTED

                return Connection(
                    source_variables=source_variables,
                    target_variables=target_variables,
                    connection_type=connection_type,
                    weight=weight,
                    annotation=conn_data.get("annotation"),
                    description=description,
                )

        except Exception as e:
            logger.warning(f"Failed to parse YAML connection {conn_data}: {e}")
            return None

    def _parse_connection_string(self, conn_str: str) -> Optional[Connection]:
        """Parse connection from string format like 'A>B' or 'A-B'."""
        try:
            from .common import parse_connection_operator

            # Find the operator
            operators: list[Any] = [">", "->", "-"]
            operator = None
            source = None
            target = None

            for op in operators:
                if op in conn_str:
                    operator = op
                    parts = conn_str.split(op, 1)
                    source = parts[0].strip()
                    target = parts[1].strip()
                    break

            if not operator or not source or not target:
                return None

            connection_type = parse_connection_operator(operator)

            return Connection(
                source_variables=[source],
                target_variables=[target],
                connection_type=connection_type,
                description=f"Parsed from YAML: {conn_str}",
            )

        except Exception as e:
            logger.warning(f"Failed to parse connection string {conn_str}: {e}")
            return None

    def _parse_yaml_parameters(self, params_data: Union[List, Dict]) -> List[Parameter]:
        """Parse parameters from YAML data."""
        parameters: list[Any] = []

        if isinstance(params_data, list):
            for param_data in params_data:
                parameter = self._parse_yaml_parameter(param_data)
                if parameter:
                    parameters.append(parameter)
        elif isinstance(params_data, dict):
            for param_name, param_value in params_data.items():
                parameter = Parameter(
                    name=param_name,
                    value=param_value,
                    description=f"YAML parameter: {param_name}",
                )
                parameters.append(parameter)

        return parameters

    def _parse_yaml_parameter(self, param_data: Dict[str, Any]) -> Optional[Parameter]:
        """Parse a single parameter from YAML data."""
        try:
            name = param_data.get("name", "")
            value = param_data.get("value")
            type_hint = param_data.get("type")
            description = param_data.get("description", "")

            return Parameter(
                name=name, value=value, type_hint=type_hint, description=description
            )

        except Exception as e:
            logger.warning(f"Failed to parse YAML parameter {param_data}: {e}")
            return None

    def _parse_yaml_equations(
        self, equations_data: List[Union[Dict, str]]
    ) -> List[Equation]:
        """Parse equations from YAML data."""
        equations: list[Any] = []

        for eq_data in equations_data:
            equation = self._parse_yaml_equation(eq_data)
            if equation:
                equations.append(equation)

        return equations

    def _parse_yaml_equation(
        self, eq_data: Union[Dict[str, Any], str]
    ) -> Optional[Equation]:
        """Parse a single equation from YAML data."""
        try:
            if isinstance(eq_data, str):
                return Equation(
                    label=None,
                    content=eq_data,
                    format="latex",
                    description="YAML equation",
                )

            elif isinstance(eq_data, dict):
                label = eq_data.get("label")
                content = eq_data.get("content", eq_data.get("equation", ""))
                format_type = eq_data.get("format", "latex")
                description = eq_data.get("description", "")

                return Equation(
                    label=label,
                    content=str(content or ""),
                    format=str(format_type),
                    description=str(description) if description is not None else None,
                )

        except Exception as e:
            logger.warning(f"Failed to parse YAML equation {eq_data}: {e}")
            return None

    def _parse_yaml_time_specification(
        self, time_data: Union[Dict, str]
    ) -> Optional[TimeSpecification]:
        """Parse time specification from YAML data."""
        try:
            if isinstance(time_data, str):
                return TimeSpecification(
                    time_type=time_data, discretization=None, horizon=None
                )

            elif isinstance(time_data, dict):
                time_type = time_data.get("type", time_data.get("time_type", "Static"))
                discretization = time_data.get("discretization")
                horizon = time_data.get("horizon", time_data.get("model_time_horizon"))

                return TimeSpecification(
                    time_type=time_type,
                    discretization=discretization,
                    horizon=horizon,
                    step_size=time_data.get("step_size"),
                )

        except Exception as e:
            logger.warning(f"Failed to parse YAML time specification {time_data}: {e}")
            return None

    def _parse_yaml_ontology_mappings(
        self, mappings_data: Union[List, Dict]
    ) -> List[OntologyMapping]:
        """Parse ontology mappings from YAML data."""
        mappings: list[Any] = []

        if isinstance(mappings_data, list):
            for mapping_data in mappings_data:
                mapping = self._parse_yaml_ontology_mapping(mapping_data)
                if mapping:
                    mappings.append(mapping)
        elif isinstance(mappings_data, dict):
            for var_name, ontology_term in mappings_data.items():
                mapping = OntologyMapping(
                    variable_name=var_name,
                    ontology_term=str(ontology_term),
                    description=f"YAML ontology mapping: {var_name}",
                )
                mappings.append(mapping)

        return mappings

    def _parse_yaml_ontology_mapping(
        self, mapping_data: Dict[str, Any]
    ) -> Optional[OntologyMapping]:
        """Parse a single ontology mapping from YAML data."""
        try:
            variable_name = mapping_data.get(
                "variable", mapping_data.get("variable_name", "")
            )
            ontology_term = mapping_data.get(
                "term", mapping_data.get("ontology_term", "")
            )
            description = mapping_data.get("description", "")

            return OntologyMapping(
                variable_name=variable_name,
                ontology_term=ontology_term,
                description=description,
            )

        except Exception as e:
            logger.warning(f"Failed to parse YAML ontology mapping {mapping_data}: {e}")
            return None

    def get_supported_extensions(self) -> List[str]:
        """Get supported file extensions."""
        return [".yaml", ".yml"]
