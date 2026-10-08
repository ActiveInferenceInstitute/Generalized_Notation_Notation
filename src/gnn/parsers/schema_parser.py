"""
Schema Parser for GNN XML Schema, ASN.1, PKL, Alloy, and Z Notation

This module provides parsing capabilities for formal schema definition languages
that specify GNN models using various schema formats.

Author: @docxology
Date: 2025-01-11
License: MIT
"""

import json
import logging
import re
import shutil
from typing import Any, cast

try:
    import defusedxml.ElementTree as ET
except ImportError as _exc:  # pragma: no cover - broken environment only
    raise RuntimeError(
        "defusedxml is a required dependency of gnn.parsers.schema_parser but "
        "could not be imported; the environment is broken. Reinstall with "
        "'uv sync' (see pyproject.toml)."
    ) from _exc
from pathlib import Path
from typing import Dict, List, Optional, Union

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
    extract_embedded_json_data,
)

logger = logging.getLogger(__name__)


def _extract_line_model_data(
    content: str, comment_prefix: str
) -> Optional[Dict[str, Any]]:
    """Read one native line comment without consuming later declarations.

    A present MODEL_DATA comment is an explicit interchange payload. Invalid
    JSON or a non-object payload must fail rather than silently admit a native
    declaration with a different identity.
    """
    prefix_pattern = r"%+" if comment_prefix == "%" else re.escape(comment_prefix)
    match = re.search(
        rf"{prefix_pattern}[ \t]*MODEL_DATA:[ \t]*([^\r\n]*)",
        content,
        re.MULTILINE,
    )
    if match is None:
        return None
    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed MODEL_DATA line comment: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("MODEL_DATA line comment requires a JSON object")
    return data


def _time_specification_from_mapping(data: Dict[str, Any]) -> TimeSpecification:
    """Build a typed time specification from embedded schema data."""
    raw_step_size = data.get("step_size")
    step_size = float(raw_step_size) if raw_step_size is not None else None
    return TimeSpecification(
        time_type=str(data.get("time_type", data.get("type", "Static"))),
        discretization=(
            str(data["discretization"])
            if data.get("discretization") is not None
            else None
        ),
        horizon=cast(Optional[Union[int, str]], data.get("horizon")),
        step_size=step_size,
    )


class XSDParser(BaseGNNParser):
    """Parser for XML Schema Definition (XSD) files."""

    def __init__(self) -> None:
        """Initialize the instance."""
        super().__init__()

    def _extract_embedded_json_data(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract embedded JSON model data from schema comments."""
        return extract_embedded_json_data(
            content,
            [
                r"/\*\s*MODEL_DATA:\s*(\{.*?\})\s*\*/",  # /* MODEL_DATA: {...} */
                r"<!--\s*MODEL_DATA:\s*(\{.*?\})\s*-->",  # <!-- MODEL_DATA: {...} -->
                r"#\s*MODEL_DATA:\s*(\{.*?\})",  # # MODEL_DATA: {...}
                r"//\s*MODEL_DATA:\s*(\{.*?\})",  # // MODEL_DATA: {...}
            ],
        )

    def _parse_from_embedded_data(
        self, data: Dict[str, Any], result: ParseResult
    ) -> ParseResult:
        """Parse model from embedded JSON data for perfect round-trip."""
        try:
            # Restore original model data
            result.model.model_name = data.get("model_name", "SchemaModel")
            result.model.annotation = data.get("annotation", "")
            result.model.version = data.get("version", "1.0")

            # Restore variables
            for var_data in data.get("variables", []):
                variable = Variable(
                    name=var_data["name"],
                    var_type=self._parse_enum_value(
                        VariableType, var_data.get("var_type", "hidden_state")
                    ),
                    data_type=self._parse_enum_value(
                        DataType, var_data.get("data_type", "categorical")
                    ),
                    dimensions=var_data.get("dimensions", [1]),
                    description=var_data.get("description", ""),
                )
                result.model.variables.append(variable)

            # Restore connections
            for conn_data in data.get("connections", []):
                connection = Connection(
                    annotation=conn_data.get("annotation"),
                    source_variables=conn_data.get("source_variables", []),
                    target_variables=conn_data.get("target_variables", []),
                    connection_type=self._parse_enum_value(
                        ConnectionType, conn_data.get("connection_type", "directed")
                    ),
                    description=conn_data.get("description", ""),
                )
                result.model.connections.append(connection)

            # Restore parameters
            for param_data in data.get("parameters", []):
                parameter = Parameter(
                    name=param_data["name"],
                    value=param_data["value"],
                    description=param_data.get("description", ""),
                )
                result.model.parameters.append(parameter)

            # Restore other fields including time specification and ontology mappings
            if "time_specification" in data and data["time_specification"]:
                time_spec_data = data["time_specification"]
                if isinstance(time_spec_data, dict):
                    result.model.time_specification = _time_specification_from_mapping(
                        time_spec_data
                    )

            # Restore ontology mappings for XSD parser
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

        except Exception as e:
            result.add_error(f"Failed to parse embedded data: {e}")

        return result

    def _parse_enum_value(self, enum_class: Any, value_str: str) -> Any:
        """Parse enum value from string."""
        try:
            # Try to get enum by value
            for enum_val in enum_class:
                if enum_val.value == value_str:
                    return enum_val
            # Recovery to first enum value
            return list(enum_class)[0]
        except (TypeError, AttributeError, ValueError, IndexError):
            return list(enum_class)[0]

    def get_supported_extensions(self) -> List[str]:
        """Return supported extensions."""
        return [".xsd"]

    def parse_file(self, file_path: str) -> ParseResult:
        """Parse file."""
        try:
            # Read the file content first to check for embedded data
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return self.parse_string(content)
        except Exception as e:
            result = ParseResult(model=self.create_empty_model())
            result.add_error(f"Failed to parse XSD file: {e}")
            return result

    def parse_string(self, content: str) -> ParseResult:
        # First try to extract embedded JSON model data from original content
        """Parse string."""
        embedded_data = self._extract_embedded_json_data(content)
        if embedded_data:
            result = ParseResult(model=self.create_empty_model())
            return self._parse_from_embedded_data(embedded_data, result)

        try:
            root = ET.fromstring(content)  # nosec B314
            return self.parse_xml_element(root)
        except Exception as e:
            result = ParseResult(model=self.create_empty_model())
            result.add_error(f"Failed to parse XSD content: {e}")
            return result

    def parse_xml_element(self, root: Any) -> ParseResult:
        """Parse xml element."""
        result = ParseResult(model=self.create_empty_model())

        # Convert XML element back to string to look for embedded data
        content = ET.tostring(root, encoding="unicode")

        # First try to extract embedded JSON model data
        embedded_data = self._extract_embedded_json_data(content)
        if embedded_data:
            return self._parse_from_embedded_data(embedded_data, result)

        # Recovery to basic XSD parsing
        result.model.model_name = root.get("targetNamespace", "XSDModel").split("/")[-1]

        # Parse element definitions
        for elem in root.findall(".//{http://www.w3.org/2001/XMLSchema}element"):
            name = elem.get("name")
            type_attr = elem.get("type", "string")

            if name:
                variable = Variable(
                    name=name,
                    var_type=self._infer_variable_type(name),
                    dimensions=[1],
                    data_type=self._map_xsd_type(type_attr),
                    description=f"XSD element of type {type_attr}",
                )
                result.model.variables.append(variable)

        return result

    def _map_xsd_type(self, xsd_type: str) -> DataType:
        """Handle map xsd type for internal callers."""
        type_mapping: dict[str, Any] = {
            "string": DataType.CATEGORICAL,
            "int": DataType.INTEGER,
            "integer": DataType.INTEGER,
            "float": DataType.FLOAT,
            "double": DataType.FLOAT,
            "boolean": DataType.BINARY,
            "decimal": DataType.FLOAT,
        }
        return cast(
            "DataType", type_mapping.get(xsd_type.split(":")[-1], DataType.CATEGORICAL)
        )

    def _infer_variable_type(self, name: str) -> VariableType:
        """Handle infer variable type for internal callers."""
        name_lower = name.lower()
        if "state" in name_lower:
            return VariableType.HIDDEN_STATE
        elif "observation" in name_lower:
            return VariableType.OBSERVATION
        else:
            return VariableType.HIDDEN_STATE


class ASN1Parser(BaseGNNParser):
    """Parser for ASN.1 schema definition files."""

    def __init__(self) -> None:
        """Initialize the instance."""
        super().__init__()
        self.module_pattern = re.compile(
            r"(\w+)\s+DEFINITIONS.*::=\s*BEGIN", re.IGNORECASE
        )
        self.sequence_pattern = re.compile(
            r"(\w+)\s*::=\s*SEQUENCE\s*\{([^}]+)\}", re.IGNORECASE | re.DOTALL
        )
        self.field_pattern = re.compile(
            r"(\w+)\s+([A-Z][A-Z0-9]*(?:\s+[A-Z0-9]*)*)", re.IGNORECASE
        )

    def _extract_embedded_json_data(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract embedded JSON model data from schema comments."""
        block_data = extract_embedded_json_data(
            content,
            [
                r"/\*\s*MODEL_DATA:\s*(\{.*?\})\s*\*/",  # /* MODEL_DATA: {...} */
                r"<!--\s*MODEL_DATA:\s*(\{.*?\})\s*-->",  # <!-- MODEL_DATA: {...} -->
                r"#\s*MODEL_DATA:\s*(\{.*?\})",  # # MODEL_DATA: {...}
                r"//\s*MODEL_DATA:\s*(\{.*?\})",  # // MODEL_DATA: {...}
            ],
        )
        if block_data is not None:
            return block_data
        return _extract_line_model_data(content, "--")

    def _parse_from_embedded_data(
        self, data: Dict[str, Any], result: ParseResult
    ) -> ParseResult:
        """Parse model from embedded JSON data for perfect round-trip."""
        try:
            # Restore original model data
            result.model.model_name = data.get("model_name", "SchemaModel")
            result.model.annotation = data.get("annotation", "")
            result.model.version = data.get("version", "1.0")

            # Restore variables
            for var_data in data.get("variables", []):
                variable = Variable(
                    name=var_data["name"],
                    var_type=self._parse_enum_value(
                        VariableType, var_data.get("var_type", "hidden_state")
                    ),
                    data_type=self._parse_enum_value(
                        DataType, var_data.get("data_type", "categorical")
                    ),
                    dimensions=var_data.get("dimensions", [1]),
                    description=var_data.get("description", ""),
                )
                result.model.variables.append(variable)

            # Restore connections
            for conn_data in data.get("connections", []):
                connection = Connection(
                    annotation=conn_data.get("annotation"),
                    source_variables=conn_data.get("source_variables", []),
                    target_variables=conn_data.get("target_variables", []),
                    connection_type=self._parse_enum_value(
                        ConnectionType, conn_data.get("connection_type", "directed")
                    ),
                    description=conn_data.get("description", ""),
                )
                result.model.connections.append(connection)

            # Restore parameters
            for param_data in data.get("parameters", []):
                parameter = Parameter(
                    name=param_data["name"],
                    value=param_data["value"],
                    description=param_data.get("description", ""),
                )
                result.model.parameters.append(parameter)

            # Restore time specification for ASN1 parser
            if "time_specification" in data and data["time_specification"]:
                time_spec_data = data["time_specification"]
                if isinstance(time_spec_data, dict):
                    result.model.time_specification = _time_specification_from_mapping(
                        time_spec_data
                    )

            # Restore ontology mappings for ASN1 parser
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

        except Exception as e:
            result.add_error(f"Failed to parse embedded data: {e}")

        return result

    def _parse_enum_value(self, enum_class: Any, value_str: str) -> Any:
        """Parse enum value from string."""
        try:
            # Try to get enum by value
            for enum_val in enum_class:
                if enum_val.value == value_str:
                    return enum_val
            # Recovery to first enum value
            return list(enum_class)[0]
        except (TypeError, AttributeError, ValueError, IndexError):
            return list(enum_class)[0]

    def get_supported_extensions(self) -> List[str]:
        """Return supported extensions."""
        return [".asn1", ".asn"]

    def parse_file(self, file_path: str) -> ParseResult:
        """Parse file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return self.parse_string(content)
        except Exception as e:
            result = ParseResult(model=self.create_empty_model())
            result.add_error(f"Failed to read ASN.1 file: {e}")
            return result

    def parse_string(self, content: str) -> ParseResult:
        """Parse string."""
        result = ParseResult(model=self.create_empty_model())

        try:
            # First try to extract embedded JSON model data
            embedded_data = self._extract_embedded_json_data(content)
            if embedded_data:
                return self._parse_from_embedded_data(embedded_data, result)

            # Extract module name
            module_match = self.module_pattern.search(content)
            if module_match:
                result.model.model_name = f"{module_match.group(1)}Model"
            else:
                result.model.model_name = "ASN1Model"

            # Parse SEQUENCE definitions
            for match in self.sequence_pattern.finditer(content):
                seq_name = match.group(1)
                seq_body = match.group(2)

                # Parse fields within the sequence
                for field_match in self.field_pattern.finditer(seq_body):
                    field_name = field_match.group(1)
                    field_type = field_match.group(2).strip()

                    variable = Variable(
                        name=field_name,
                        var_type=self._infer_variable_type(field_name),
                        dimensions=self._parse_asn1_dimensions(field_type),
                        data_type=self._map_asn1_type(field_type),
                        description=f"ASN.1 {seq_name} field of type {field_type}",
                    )
                    result.model.variables.append(variable)

            result.model.annotation = "Parsed from ASN.1 schema definition"

        except Exception as e:
            result.add_error(f"ASN.1 parsing error: {e}")

        return result

    def _parse_asn1_dimensions(self, type_def: str) -> List[int]:
        """Parse asn1 dimensions."""
        if "SEQUENCE OF" in type_def.upper():
            return [1]  # Array/list type
        elif "CHOICE" in type_def.upper():
            return [1]  # Union type
        return []

    def _map_asn1_type(self, type_def: str) -> DataType:
        """Handle map asn1 type for internal callers."""
        type_upper = type_def.upper()
        if "INTEGER" in type_upper:
            return DataType.INTEGER
        elif "REAL" in type_upper:
            return DataType.FLOAT
        elif "BOOLEAN" in type_upper:
            return DataType.BINARY
        elif "UTF8STRING" in type_upper or "OCTET STRING" in type_upper:
            return DataType.CATEGORICAL
        elif "SEQUENCE" in type_upper:
            return DataType.CATEGORICAL
        return DataType.CATEGORICAL

    def _infer_variable_type(self, name: str) -> VariableType:
        """Handle infer variable type for internal callers."""
        name_lower = name.lower()
        if "state" in name_lower or "hidden" in name_lower:
            return VariableType.HIDDEN_STATE
        elif "observation" in name_lower or "obs" in name_lower:
            return VariableType.OBSERVATION
        elif "action" in name_lower or "control" in name_lower:
            return VariableType.ACTION
        else:
            return VariableType.HIDDEN_STATE


class PKLParser(BaseGNNParser):
    """Parser for Apple PKL configuration files."""

    def __init__(self) -> None:
        """Initialize the instance."""
        super().__init__()
        self.class_pattern = re.compile(r"class\s+(\w+)\s*\{([^}]+)\}", re.DOTALL)
        self.property_pattern = re.compile(
            r"^[ \t]*(\w+)[ \t]*:[ \t]*([^=\n]+?)(?:=[ \t]*([^\n]+))?[ \t]*$",
            re.MULTILINE,
        )
        self.mapping_pattern = re.compile(
            r"(\w+)\s*:\s*Mapping<[^>]+>\s*=\s*new\s+Mapping\s*\{([^}]+)\}", re.DOTALL
        )
        self.mapping_entry_pattern = re.compile(
            r'\["([^"]+)"\]\s*=\s*([^\n]+)', re.MULTILINE
        )

    def _extract_embedded_json_data(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract embedded JSON model data from schema comments."""
        return extract_embedded_json_data(
            content,
            [
                r"/\*\s*MODEL_DATA:\s*(\{.*?\})\s*\*/",  # /* MODEL_DATA: {...} */
                r"<!--\s*MODEL_DATA:\s*(\{.*?\})\s*-->",  # <!-- MODEL_DATA: {...} -->
                r"#\s*MODEL_DATA:\s*(\{.*?\})",  # # MODEL_DATA: {...}
                r"//\s*MODEL_DATA:\s*(\{.*?\})",  # // MODEL_DATA: {...}
            ],
        )

    def _parse_from_embedded_data(
        self, data: Dict[str, Any], result: ParseResult
    ) -> ParseResult:
        """Parse model from embedded JSON data for perfect round-trip."""
        try:
            # Restore original model data
            result.model.model_name = data.get("model_name", "SchemaModel")
            result.model.annotation = data.get("annotation", "")
            result.model.version = data.get("version", "1.0")

            # Restore variables
            for var_data in data.get("variables", []):
                variable = Variable(
                    name=var_data["name"],
                    var_type=self._parse_enum_value(
                        VariableType, var_data.get("var_type", "hidden_state")
                    ),
                    data_type=self._parse_enum_value(
                        DataType, var_data.get("data_type", "categorical")
                    ),
                    dimensions=var_data.get("dimensions", [1]),
                    description=var_data.get("description", ""),
                )
                result.model.variables.append(variable)

            # Restore connections
            for conn_data in data.get("connections", []):
                connection = Connection(
                    annotation=conn_data.get("annotation"),
                    source_variables=conn_data.get("source_variables", []),
                    target_variables=conn_data.get("target_variables", []),
                    connection_type=self._parse_enum_value(
                        ConnectionType, conn_data.get("connection_type", "directed")
                    ),
                    description=conn_data.get("description", ""),
                )
                result.model.connections.append(connection)

            # Restore parameters
            for param_data in data.get("parameters", []):
                parameter = Parameter(
                    name=param_data["name"],
                    value=param_data["value"],
                    description=param_data.get("description", ""),
                )
                result.model.parameters.append(parameter)

            # Restore time specification for PKL parser
            if "time_specification" in data and data["time_specification"]:
                time_spec_data = data["time_specification"]
                if isinstance(time_spec_data, dict):
                    result.model.time_specification = _time_specification_from_mapping(
                        time_spec_data
                    )

            # Restore ontology mappings for PKL parser
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
            # result.model.annotation is already set above

        except Exception as e:
            result.add_error(f"Failed to parse embedded data: {e}")

        return result

    def _parse_enum_value(self, enum_class: Any, value_str: str) -> Any:
        """Parse enum value from string."""
        try:
            # Try to get enum by value
            for enum_val in enum_class:
                if enum_val.value == value_str:
                    return enum_val
            # Recovery to first enum value
            return list(enum_class)[0]
        except (TypeError, AttributeError, ValueError, IndexError):
            return list(enum_class)[0]

    def get_supported_extensions(self) -> List[str]:
        """Return supported extensions."""
        return [".pkl"]

    def parse_file(self, file_path: str) -> ParseResult:
        """Parse file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return self.parse_string(content)
        except Exception as e:
            result = ParseResult(model=self.create_empty_model())
            result.add_error(f"Failed to read PKL file: {e}")
            return result

    def parse_string(self, content: str) -> ParseResult:
        """Parse string."""
        result = ParseResult(model=self.create_empty_model())

        try:
            # First try to extract embedded JSON model data
            embedded_data = self._extract_embedded_json_data(content)
            if embedded_data:
                return self._parse_from_embedded_data(embedded_data, result)

            # Find main model class
            model_class_match = re.search(
                r"class\s+(\w*Model|\w*GNN\w*)\s*\{", content, re.IGNORECASE
            )
            if model_class_match:
                result.model.model_name = model_class_match.group(1)
            else:
                result.model.model_name = "PKLModel"

            # Parse class definitions
            for class_match in self.class_pattern.finditer(content):
                class_match.group(1)
                class_body = class_match.group(2)

                # Parse properties within the class
                for prop_match in self.property_pattern.finditer(class_body):
                    prop_name = prop_match.group(1)
                    prop_type = prop_match.group(2).strip()
                    prop_value = (
                        prop_match.group(3).strip() if prop_match.group(3) else None
                    )

                    # Skip special PKL properties
                    if prop_name in ["name", "version", "annotation"]:
                        continue

                    # Determine if this is a variable or parameter
                    if self._is_variable_property(prop_name, prop_type):
                        variable = Variable(
                            name=prop_name,
                            var_type=self._infer_variable_type(prop_name),
                            dimensions=self._parse_pkl_dimensions(prop_type),
                            data_type=self._map_pkl_type(prop_type),
                            description=f"PKL property of type {prop_type}",
                        )
                        result.model.variables.append(variable)
                    else:
                        parameter = Parameter(
                            name=prop_name,
                            value=prop_value or f"<{prop_type}>",
                            type_hint=prop_type,
                            description="PKL property",
                        )
                        result.model.parameters.append(parameter)

            # Parse mappings (like variables: Mapping<String, Variable>)
            for mapping_match in self.mapping_pattern.finditer(content):
                mapping_name = mapping_match.group(1)
                mapping_body = mapping_match.group(2)

                # Parse mapping entries
                for entry_match in self.mapping_entry_pattern.finditer(mapping_body):
                    entry_key = entry_match.group(1)
                    entry_value = entry_match.group(2).strip()

                    if mapping_name == "variables":
                        parsed_variable = self._parse_pkl_variable_entry(
                            entry_key, entry_value
                        )
                        if parsed_variable:
                            result.model.variables.append(parsed_variable)
                    elif mapping_name == "parameters":
                        parameter = Parameter(
                            name=entry_key,
                            value=entry_value,
                            description="PKL parameter",
                        )
                        result.model.parameters.append(parameter)

            result.model.annotation = "Parsed from Apple PKL configuration"
            result.metadata["format"] = "pkl"

        except Exception as e:
            result.add_error(f"PKL parsing error: {e}")

        return result

    def _is_variable_property(self, name: str, type_str: str) -> bool:
        """Determine if a property represents a variable."""
        return (
            "Variable" in type_str
            or "Mapping" in type_str
            and "Variable" in type_str
            or any(
                keyword in name.lower() for keyword in ["var", "state", "obs", "action"]
            )
        )

    def _parse_pkl_dimensions(self, type_str: str) -> List[int]:
        """Parse dimensions from PKL type definition."""
        if "List<" in type_str:
            return [1]  # List type
        elif "Mapping<" in type_str:
            return [1]  # Map type
        return []

    def _map_pkl_type(self, type_str: str) -> DataType:
        """Map PKL types to GNN data types."""
        if "Int" in type_str:
            return DataType.INTEGER
        elif "Float" in type_str or "Double" in type_str:
            return DataType.FLOAT
        elif "Boolean" in type_str:
            return DataType.BINARY
        elif "String" in type_str:
            return DataType.CATEGORICAL
        elif "List" in type_str:
            return DataType.CATEGORICAL
        return DataType.CATEGORICAL

    def _parse_pkl_variable_entry(self, key: str, value: str) -> Optional[Variable]:
        """Parse a PKL variable entry."""
        try:
            # Simple parsing for PKL variable structure
            variable = Variable(
                name=key,
                var_type=self._infer_variable_type(key),
                dimensions=[],
                data_type=DataType.CATEGORICAL,
                description="PKL variable definition",
            )
            return variable
        except Exception as e:
            # The Variable construction is infallible for well-formed input;
            # reaching here means a genuinely unexpected failure, so surface
            # it before honouring the silent-skip (None) contract.
            logger.warning("PKL variable entry %r could not be parsed: %s", key, e)
            return None

    def _infer_variable_type(self, name: str) -> VariableType:
        """Infer variable type from name."""
        name_lower = name.lower()
        if "state" in name_lower or "s_" in name_lower:
            return VariableType.HIDDEN_STATE
        elif "obs" in name_lower or "o_" in name_lower:
            return VariableType.OBSERVATION
        elif "action" in name_lower or "u_" in name_lower:
            return VariableType.ACTION
        elif "policy" in name_lower or "pi_" in name_lower:
            return VariableType.POLICY
        else:
            return VariableType.HIDDEN_STATE

    def parse_content(
        self, content: str, file_path: Optional[Path] = None
    ) -> ParseResult:
        """Parse PKL content with enhanced model extraction."""
        try:
            model = GNNInternalRepresentation(model_name="PKLGNNModel")
            errors: list[Any] = []
            warnings: list[Any] = []
            metadata: dict[str, Any] = {
                "format": "pkl",
                "source_file": str(file_path) if file_path else None,
            }

            # Parse model metadata
            model_name = self._extract_pkl_model_name(content)
            model.model_name = model_name or "PKLGNNModel"

            # Attempt native pkl compiler evaluation if pkl binary is available
            native_eval = self._eval_pkl_native(content, file_path)
            if native_eval and isinstance(native_eval, dict):
                metadata["pkl_native_eval"] = True
                if "model_name" in native_eval:
                    model.model_name = str(native_eval["model_name"])

            # Parse model annotation from comments
            model.annotation = self._extract_pkl_annotation(content)

            # Parse variables with complete information
            variables = self._parse_pkl_variables_enhanced(content)
            model.variables = variables

            # Parse connections with proper relationship mapping
            connections = self._parse_pkl_connections_enhanced(content)
            model.connections = connections

            # Parse parameters with value preservation
            parameters = self._parse_pkl_parameters_enhanced(content)
            model.parameters = parameters

            # Parse equations if present
            equations = self._parse_pkl_equations(content)
            model.equations = equations

            # Parse time specification
            time_spec = self._parse_pkl_time_specification(content)
            model.time_specification = time_spec

            # Parse ontology mappings
            ontology_mappings = self._parse_pkl_ontology_mappings(content)
            model.ontology_mappings = ontology_mappings

            # Parse embedded JSON data if present
            embedded_data = self._extract_pkl_embedded_data(content)
            if embedded_data:
                self._apply_embedded_data_to_model(model, embedded_data)

            # Add model validation
            validation_errors = self._validate_pkl_model_completeness(model, content)
            errors.extend(validation_errors)

            success = len(errors) == 0

            return ParseResult(
                model=model,
                success=success,
                errors=errors,
                warnings=warnings,
                metadata=metadata,
            )

        except Exception as e:
            return ParseResult(
                model=self.create_empty_model(),
                success=False,
                errors=[f"PKL parsing failed: {str(e)}"],
                warnings=[],
                metadata={"format": "pkl"},
            )

    def _eval_pkl_native(
        self, content: str, file_path: Optional[Path] = None
    ) -> Optional[Dict[str, Any]]:
        """Evaluate PKL content via the external ``pkl`` compiler (JSON output).

        Security posture: this evaluates *untrusted* PKL source. There is no
        Python ``eval`` here — the ``pkl`` binary is a full evaluation engine,
        so a "restricted module namespace / builtins whitelist" does not
        apply; the residual risk is documented instead. Mitigations in place:

        - The external ``pkl`` binary (resolved by ``shutil.which``, never
          attacker-controlled) is invoked as a fixed argv list — no shell.
        - The call is bounded by a hard ``timeout=10``; a hostile document
          that spins the evaluator is killed and the parse falls back to the
          textual path (returns ``None``).
        - Only the JSON result is parsed; a failure yields ``None``, never an
          exception escaping to the caller.

        Residual risk: PKL is a programmable language — a hostile document
        may use ``read("file://...")``/``import``/``io`` to touch files within
        the evaluator's privileges, or exhaust memory before the timeout.
        Treat ``pkl``-parsed input as best-effort metadata only; do not run
        this against untrusted sources in a privileged environment.
        """
        pkl_bin = shutil.which("pkl")
        if not pkl_bin:
            return None
        try:
            import json
            import subprocess
            import tempfile

            if file_path and file_path.exists():
                cmd = [pkl_bin, "eval", "-f", "json", str(file_path)]
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                if res.returncode == 0 and res.stdout.strip():
                    return cast("Dict[str, Any]", json.loads(res.stdout))
            else:
                with tempfile.NamedTemporaryFile(
                    "w", suffix=".pkl", delete=False
                ) as tf:
                    tf.write(content)
                    tf_path = tf.name
                try:
                    cmd = [pkl_bin, "eval", "-f", "json", tf_path]
                    res = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=10
                    )
                    if res.returncode == 0 and res.stdout.strip():
                        return cast("Dict[str, Any]", json.loads(res.stdout))
                finally:
                    Path(tf_path).unlink(missing_ok=True)
        except Exception as e:
            logger.debug(f"Pkl native eval exception: {e}")
        return None

    def _extract_pkl_model_name(self, content: str) -> Optional[str]:
        """Extract model name from PKL content."""
        # Look for class name
        class_match = re.search(r"class\s+(\w+)\s*{", content)
        if class_match:
            return class_match.group(1)

        # Look for amends clause
        amends_match = re.search(r'amends\s+"([^"]+)"', content)
        if amends_match:
            return amends_match.group(1).replace("/", "_")

        # Look for module clause
        module_match = re.search(r"module\s+(\w+)", content)
        if module_match:
            return module_match.group(1)

        return None

    def _extract_pkl_annotation(self, content: str) -> str:
        """Extract model annotation from PKL comments."""
        annotations: list[Any] = []

        # Look for /// comments (doc comments)
        doc_comments = re.findall(r"///\s*(.+)", content)
        if doc_comments:
            annotations.extend(doc_comments[:3])

        # Look for // comments
        line_comments = re.findall(r"//\s*(.+)", content)
        if line_comments:
            annotations.extend([c for c in line_comments[:3] if not c.startswith("/")])

        # Look for /* */ comments
        block_comments = re.findall(r"/\*\s*(.*?)\s*\*/", content, re.DOTALL)
        for comment in block_comments:
            clean_comment = re.sub(r"\s+", " ", comment.strip())
            if (
                clean_comment
                and len(clean_comment) > 10
                and "MODEL_DATA" not in clean_comment
            ):
                annotations.append(clean_comment)

        return " ".join(annotations) if annotations else ""

    def _parse_pkl_variables_enhanced(self, content: str) -> List[Variable]:
        """Parse variables with complete information preservation."""
        variables: list[Any] = []

        # Parse property declarations
        prop_declarations = re.findall(r"(\w+):\s*(\w+)(?:\s*=\s*([^;\n]+))?", content)
        for name, type_hint, _ in prop_declarations:
            if name in ["name", "annotation", "variables", "connections", "parameters"]:
                continue  # Skip model metadata fields

            var = Variable(
                name=name,
                var_type=self._map_pkl_type_to_variable_type(type_hint),
                data_type=self._map_pkl_type_to_data_type(type_hint),
                dimensions=[],
            )
            variables.append(var)

        # Parse embedded data
        embedded_data = self._extract_pkl_embedded_data(content)
        if embedded_data and "variables" in embedded_data:
            for var_data in embedded_data["variables"]:
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

    def _parse_pkl_connections_enhanced(self, content: str) -> List[Connection]:
        """Parse connections with complete relationship preservation."""
        connections: list[Any] = []

        # Parse from embedded data
        embedded_data = self._extract_pkl_embedded_data(content)
        if embedded_data and "connections" in embedded_data:
            for conn_data in embedded_data["connections"]:
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

        # Parse from comment annotations
        connection_comments = re.findall(r"//\s*Connection:\s*(.+)", content)
        for comment in connection_comments:
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

    def _parse_pkl_parameters_enhanced(self, content: str) -> List[Parameter]:
        """Parse parameters with complete value preservation."""
        parameters: list[Any] = []

        # Parse from property declarations with default values
        prop_declarations = re.findall(r"(\w+):\s*\w+\s*=\s*([^;\n]+)", content)
        for name, value_str in prop_declarations:
            if name in ["name", "annotation", "variables", "connections", "parameters"]:
                continue

            parameter = Parameter(
                name=name,
                value=self._parse_pkl_value(value_str.strip()),
                type_hint="constant",
            )
            parameters.append(parameter)

        # Parse from embedded data
        embedded_data = self._extract_pkl_embedded_data(content)
        if embedded_data and "parameters" in embedded_data:
            for param_data in embedded_data["parameters"]:
                if isinstance(param_data, dict) and "name" in param_data:
                    parameter = Parameter(
                        name=param_data["name"],
                        value=param_data.get("value"),
                        type_hint=param_data.get("param_type", "constant"),
                        description="PKL parameter",
                    )
                    # Avoid duplicates
                    if not any(p.name == parameter.name for p in parameters):
                        parameters.append(parameter)

        return parameters

    def _parse_pkl_equations(self, content: str) -> List:
        """Parse equations from PKL content."""
        equations: list[Any] = []

        # Parse equation comments
        eq_comments = re.findall(r"//\s*Equation:\s*(.+)", content)
        for eq_text in eq_comments:
            equations.append({"equation": eq_text.strip()})

        return equations

    def _parse_pkl_time_specification(
        self, content: str
    ) -> Optional[TimeSpecification]:
        """Parse time specification from PKL content."""
        time_comments = re.findall(r"//\s*Time:\s*(.+)", content)
        if time_comments:
            return TimeSpecification(time_type="discrete")

        return None

    def _parse_pkl_ontology_mappings(self, content: str) -> List:
        """Parse ontology mappings from PKL content."""
        mappings: list[Any] = []

        onto_comments = re.findall(r"//\s*Ontology:\s*(\w+)\s*->\s*(.+)", content)
        for var_name, ontology_term in onto_comments:
            mappings.append(
                {"variable_name": var_name, "ontology_term": ontology_term.strip()}
            )

        return mappings

    def _extract_pkl_embedded_data(self, content: str) -> Optional[Dict]:
        """Extract embedded JSON model data from PKL comments."""
        import json

        # Look for embedded JSON in comments
        json_matches = re.findall(r"//\s*JSON:\s*({.+?})", content, re.DOTALL)
        for json_str in json_matches:
            try:
                return cast("dict[Any, Any] | None", json.loads(json_str))
            except json.JSONDecodeError as e:
                logger.debug("Malformed JSON in embedded data, trying next: %s", e)
                continue

        # Look for MODEL_DATA in block comments
        data_matches = re.findall(
            r"/\*\s*MODEL_DATA:\s*({.+?})\s*\*/", content, re.DOTALL
        )
        for data_str in data_matches:
            try:
                return cast("dict[Any, Any] | None", json.loads(data_str))
            except json.JSONDecodeError as e:
                logger.debug("Malformed JSON in embedded data, trying next: %s", e)
                continue

        return None

    def _apply_embedded_data_to_model(
        self, model: GNNInternalRepresentation, data: Dict
    ) -> Any:
        """Apply embedded data to enhance the model."""
        if "model_name" in data and data["model_name"]:
            model.model_name = data["model_name"]

        if "annotation" in data and data["annotation"]:
            model.annotation = data["annotation"]

        # Restore time specification
        if "time_specification" in data and data["time_specification"]:
            time_spec_data = data["time_specification"]
            if isinstance(time_spec_data, dict):
                model.time_specification = _time_specification_from_mapping(
                    time_spec_data
                )

        # Restore ontology mappings
        if "ontology_mappings" in data and data["ontology_mappings"]:
            ontology_data = data["ontology_mappings"]
            if isinstance(ontology_data, list):
                model.ontology_mappings = [
                    OntologyMapping(
                        variable_name=str(mapping.get("variable_name", "")),
                        ontology_term=str(mapping.get("ontology_term", "")),
                        description=mapping.get("description"),
                    )
                    for mapping in ontology_data
                    if isinstance(mapping, dict)
                ]

    def _map_pkl_type_to_variable_type(self, pkl_type: str) -> VariableType:
        """Map PKL types to GNN variable types."""
        mapping: dict[str, Any] = {
            "String": VariableType.OBSERVATION,
            "Int": VariableType.HIDDEN_STATE,
            "Float": VariableType.HIDDEN_STATE,  # Changed from PARAMETER to more appropriate default
            "Boolean": VariableType.POLICY,
            "List": VariableType.OBSERVATION,
            "Map": VariableType.OBSERVATION,
        }
        return cast("VariableType", mapping.get(pkl_type, VariableType.HIDDEN_STATE))

    def _map_pkl_type_to_data_type(self, pkl_type: str) -> DataType:
        """Map PKL types to GNN data types."""
        mapping: dict[str, Any] = {
            "String": DataType.CATEGORICAL,
            "Int": DataType.INTEGER,
            "Float": DataType.FLOAT,
            "Boolean": DataType.BINARY,
            "List": DataType.CATEGORICAL,
            "Map": DataType.COMPLEX,
        }
        return cast("DataType", mapping.get(pkl_type, DataType.CATEGORICAL))

    def _parse_pkl_value(self, value_str: str) -> Any:
        """Parse PKL value from string."""
        value_str = value_str.strip()

        # Remove quotes if string
        if value_str.startswith('"') and value_str.endswith('"'):
            return value_str[1:-1]

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

        return value_str

    def _validate_pkl_model_completeness(
        self, model: GNNInternalRepresentation, content: str
    ) -> List[str]:
        """Validate that the PKL model was completely parsed."""
        errors: list[Any] = []

        if not model.model_name:
            errors.append("Model name not found in PKL content")

        # Check for PKL structure
        if not re.search(r'class\s+\w+|amends\s+"[^"]+"|module\s+\w+', content):
            errors.append("No valid PKL structure found")

        return errors


class AlloyParser(BaseGNNParser):
    """Parser for Alloy model specification files with embedded data support."""

    def __init__(self) -> None:
        """Initialize the instance."""
        super().__init__()
        self.sig_pattern = re.compile(
            r"sig\s+(\w+)\s*\{([^}]*)\}", re.IGNORECASE | re.DOTALL
        )
        self.field_pattern = re.compile(r"(\w+)\s*:\s*([^\n,]+)", re.IGNORECASE)

    def get_supported_extensions(self) -> List[str]:
        """Return supported extensions."""
        return [".als"]

    def _extract_embedded_json_data(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract embedded JSON model data from Alloy comments."""
        return extract_embedded_json_data(
            content,
            [
                r"/\*\s*MODEL_DATA:\s*(\{.+?\})\s*\*/",  # /* MODEL_DATA: {...} */
            ],
        )

    def _parse_from_embedded_data(
        self, embedded_data: Dict[str, Any], result: ParseResult
    ) -> ParseResult:
        """Parse model from embedded JSON data for perfect round-trip fidelity."""
        from .common import (
            Connection,
            ConnectionType,
            DataType,
            Parameter,
            Variable,
            VariableType,
        )

        try:
            result.model.model_name = embedded_data.get("model_name", "AlloyModel")
            result.model.annotation = embedded_data.get("annotation", "")

            # Restore variables
            for var_data in embedded_data.get("variables", []):
                var = Variable(
                    name=var_data["name"],
                    var_type=VariableType(var_data.get("var_type", "hidden_state")),
                    data_type=DataType(var_data.get("data_type", "categorical")),
                    dimensions=var_data.get("dimensions", []),
                )
                result.model.variables.append(var)

            # Restore connections
            for conn_data in embedded_data.get("connections", []):
                conn = Connection(
                    annotation=conn_data.get("annotation"),
                    source_variables=conn_data.get("source_variables", []),
                    target_variables=conn_data.get("target_variables", []),
                    connection_type=ConnectionType(
                        conn_data.get("connection_type", "directed")
                    ),
                )
                result.model.connections.append(conn)

            # Restore parameters
            for param_data in embedded_data.get("parameters", []):
                param = Parameter(
                    name=param_data["name"],
                    value=param_data["value"],
                    description=param_data.get("description"),
                )
                result.model.parameters.append(param)

            # Restore time specification
            if embedded_data.get("time_specification"):
                from .common import TimeSpecification

                time_data = embedded_data["time_specification"]
                result.model.time_specification = TimeSpecification(
                    time_type=time_data.get("time_type", "dynamic"),
                    discretization=time_data.get("discretization"),
                    horizon=time_data.get("horizon"),
                    step_size=time_data.get("step_size"),
                )

            # Restore ontology mappings
            for mapping_data in embedded_data.get("ontology_mappings", []):
                from .common import OntologyMapping

                mapping = OntologyMapping(
                    variable_name=mapping_data.get("variable_name", ""),
                    ontology_term=mapping_data.get("ontology_term", ""),
                    description=mapping_data.get("description"),
                )
                result.model.ontology_mappings.append(mapping)

            return result

        except Exception as e:
            result.add_error(f"Failed to parse embedded data: {e}")
            return result

    def parse_file(self, file_path: str) -> ParseResult:
        """Parse file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return self.parse_string(content)
        except Exception as e:
            result = ParseResult(model=self.create_empty_model())
            result.add_error(f"Failed to read Alloy file: {e}")
            return result

    def parse_string(self, content: str) -> ParseResult:
        """Parse string."""
        result = ParseResult(model=self.create_empty_model())

        # First, try to extract embedded JSON data for perfect round-trip
        embedded_data = self._extract_embedded_json_data(content)
        if embedded_data:
            return self._parse_from_embedded_data(embedded_data, result)

        # Recovery to standard parsing
        result.model.model_name = "AlloyModel"

        try:
            # Parse signature definitions
            for sig_match in self.sig_pattern.finditer(content):
                sig_name = sig_match.group(1)
                sig_body = sig_match.group(2)

                # Parse fields within the signature
                for field_match in self.field_pattern.finditer(sig_body):
                    field_name = field_match.group(1)
                    field_type = field_match.group(2).strip()

                    variable = Variable(
                        name=f"{sig_name}_{field_name}",
                        var_type=self._infer_variable_type(field_name),
                        dimensions=[1],
                        data_type=DataType.CATEGORICAL,
                        description=f"Alloy signature {sig_name} field {field_name} : {field_type}",
                    )
                    result.model.variables.append(variable)

            result.model.annotation = "Parsed from Alloy model specification"

        except Exception as e:
            result.add_error(f"Alloy parsing error: {e}")

        return result

    def _infer_variable_type(self, name: str) -> VariableType:
        """Handle infer variable type for internal callers."""
        name_lower = name.lower()
        if "state" in name_lower:
            return VariableType.HIDDEN_STATE
        elif "observation" in name_lower:
            return VariableType.OBSERVATION
        else:
            return VariableType.HIDDEN_STATE


class ZNotationParser(BaseGNNParser):
    """Parser for Z notation formal specification files with embedded data support."""

    def __init__(self) -> None:
        """Initialize the instance."""
        super().__init__()
        self.schema_pattern = re.compile(r"┌─\s*(\w+)\s*─+┐\s*([^└]+)└─+┘", re.DOTALL)
        self.declaration_pattern = re.compile(r"(\w+)\s*:\s*([^\n]+)", re.MULTILINE)

    def get_supported_extensions(self) -> List[str]:
        """Return supported extensions."""
        return [".zed", ".z"]

    def _extract_embedded_json_data(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract embedded JSON model data from Z notation comments."""
        return _extract_line_model_data(content, "%")

    def _parse_from_embedded_data(
        self, embedded_data: Dict[str, Any], result: ParseResult
    ) -> ParseResult:
        """Parse model from embedded JSON data for perfect round-trip fidelity."""
        from .common import (
            Connection,
            ConnectionType,
            DataType,
            Parameter,
            Variable,
            VariableType,
        )

        try:
            result.model.model_name = embedded_data.get("model_name", "ZNotationModel")
            result.model.annotation = embedded_data.get("annotation", "")

            # Restore variables
            for var_data in embedded_data.get("variables", []):
                var = Variable(
                    name=var_data["name"],
                    var_type=VariableType(var_data.get("var_type", "hidden_state")),
                    data_type=DataType(var_data.get("data_type", "categorical")),
                    dimensions=var_data.get("dimensions", []),
                )
                result.model.variables.append(var)

            # Restore connections
            for conn_data in embedded_data.get("connections", []):
                conn = Connection(
                    annotation=conn_data.get("annotation"),
                    source_variables=conn_data.get("source_variables", []),
                    target_variables=conn_data.get("target_variables", []),
                    connection_type=ConnectionType(
                        conn_data.get("connection_type", "directed")
                    ),
                )
                result.model.connections.append(conn)

            # Restore parameters
            for param_data in embedded_data.get("parameters", []):
                param = Parameter(
                    name=param_data["name"],
                    value=param_data["value"],
                    description=param_data.get("description"),
                )
                result.model.parameters.append(param)

            # Restore time specification
            if embedded_data.get("time_specification"):
                from .common import TimeSpecification

                time_data = embedded_data["time_specification"]
                result.model.time_specification = TimeSpecification(
                    time_type=time_data.get("time_type", "dynamic"),
                    discretization=time_data.get("discretization"),
                    horizon=time_data.get("horizon"),
                    step_size=time_data.get("step_size"),
                )

            # Restore ontology mappings
            for mapping_data in embedded_data.get("ontology_mappings", []):
                from .common import OntologyMapping

                mapping = OntologyMapping(
                    variable_name=mapping_data.get("variable_name", ""),
                    ontology_term=mapping_data.get("ontology_term", ""),
                    description=mapping_data.get("description"),
                )
                result.model.ontology_mappings.append(mapping)

            return result

        except Exception as e:
            result.add_error(f"Failed to parse embedded data: {e}")
            return result

    def parse_file(self, file_path: str) -> ParseResult:
        """Parse file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return self.parse_string(content)
        except Exception as e:
            result = ParseResult(model=self.create_empty_model())
            result.add_error(f"Failed to read Z notation file: {e}")
            return result

    def parse_string(self, content: str) -> ParseResult:
        """Parse string."""
        result = ParseResult(model=self.create_empty_model())

        try:
            # A present invalid payload is a failed interchange, not a native
            # declaration with a different identity.
            embedded_data = self._extract_embedded_json_data(content)
            if embedded_data:
                return self._parse_from_embedded_data(embedded_data, result)

            result.model.model_name = "ZNotationModel"
            # Parse schema definitions
            for schema_match in self.schema_pattern.finditer(content):
                schema_name = schema_match.group(1)
                schema_body = schema_match.group(2)

                # Parse declarations within the schema
                for decl_match in self.declaration_pattern.finditer(schema_body):
                    var_name = decl_match.group(1)
                    var_type = decl_match.group(2).strip()

                    variable = Variable(
                        name=var_name,
                        var_type=self._infer_variable_type(var_name),
                        dimensions=[1],
                        data_type=self._map_z_type(var_type),
                        description=f"Z notation schema {schema_name} variable : {var_type}",
                    )
                    result.model.variables.append(variable)

            result.model.annotation = "Parsed from Z notation formal specification"

        except Exception as e:
            result.add_error(f"Z notation parsing error: {e}")

        return result

    def _map_z_type(self, z_type: str) -> DataType:
        """Handle map z type for internal callers."""
        if "ℕ" in z_type or "NAT" in z_type:
            return DataType.INTEGER
        elif "ℝ" in z_type or "REAL" in z_type:
            return DataType.FLOAT
        elif "𝔹" in z_type or "BOOL" in z_type:
            return DataType.BINARY
        else:
            return DataType.CATEGORICAL

    def _infer_variable_type(self, name: str) -> VariableType:
        """Handle infer variable type for internal callers."""
        name_lower = name.lower()
        if "state" in name_lower:
            return VariableType.HIDDEN_STATE
        elif "observation" in name_lower:
            return VariableType.OBSERVATION
        else:
            return VariableType.HIDDEN_STATE


AlloyGNNParser = AlloyParser
ZNotationGNNParser = ZNotationParser
