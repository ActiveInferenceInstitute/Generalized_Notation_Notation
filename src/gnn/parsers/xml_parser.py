"""
XML Parser for GNN XML and PNML Formats

This module provides parsing capabilities for XML and PNML files that specify
GNN models using XML-based representations including Petri nets.

Author: @docxology
Date: 2025-01-11
License: MIT
"""

from typing import Any, cast

try:
    import defusedxml.ElementTree as ET
except ImportError as _exc:  # pragma: no cover - broken environment only
    raise RuntimeError(
        "defusedxml is a required dependency of gnn.parsers.xml_parser but "
        "could not be imported; the environment is broken. Reinstall with "
        "'uv sync' (see pyproject.toml)."
    ) from _exc

import logging
from typing import Dict, Iterator, List, Optional

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


def _unique_xml_elements(root: Any, paths: tuple[str, ...]) -> Iterator[Any]:
    """Yield each physical element once, preserving XPath discovery priority."""
    seen: set[int] = set()
    for path in paths:
        for element in root.findall(path):
            identity = id(element)
            if identity not in seen:
                seen.add(identity)
                yield element


class XMLGNNParser(BaseGNNParser):
    """Parser for XML format GNN specifications."""

    def __init__(self) -> None:
        """Initialize the XML parser."""
        super().__init__()

    def parse_file(self, file_path: str) -> ParseResult:
        """Parse an XML file containing GNN specifications."""
        try:
            # Read file content first to check for embedded data
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # First try to extract embedded JSON data for perfect round-trip
            embedded_data = self._extract_embedded_json_data(content)
            if embedded_data:
                # Use embedded data for perfect round-trip fidelity
                model = self._build_model_from_embedded_data(embedded_data)
                return ParseResult(model=model, success=True)

            # Only parse XML structure if no embedded data is available
            tree = ET.parse(file_path)  # nosec B314
            root = tree.getroot()
            return self._parse_xml_root(root)

        except ET.ParseError as e:
            logger.error(f"XML parse error in {file_path}: {e}")
            result = ParseResult(
                model=self.create_empty_model("Failed XML Parse"), success=False
            )
            result.add_error(f"Invalid XML format: {e}")
            return result

        except Exception as e:
            logger.error(f"Error parsing XML file {file_path}: {e}")
            result = ParseResult(
                model=self.create_empty_model("Failed XML Parse"), success=False
            )
            result.add_error(f"Failed to parse XML file: {e}")
            return result

    def parse_string(self, content: str) -> ParseResult:
        """Parse XML content from string."""
        try:
            # Quick check if this even looks like XML
            content = content.strip()
            if not content.startswith("<") or (
                "<?xml" not in content and "<gnn" not in content
            ):
                result = ParseResult(
                    model=self.create_empty_model("Invalid XML"), success=False
                )
                result.add_error(
                    "Content doesn't appear to be valid XML (missing XML declaration or root element)"
                )
                return result

            # First try to extract embedded JSON data for perfect round-trip (before parsing XML)
            embedded_data = self._extract_embedded_json_data(content)
            if embedded_data:
                # Use embedded data for perfect round-trip fidelity
                model = self._build_model_from_embedded_data(embedded_data)
                return ParseResult(model=model, success=True)

            # Only parse XML structure if no embedded data is available
            root = ET.fromstring(content)  # nosec B314
            return self._parse_xml_root(root)

        except ET.ParseError as e:
            logger.error(f"XML parse error: {e}")
            result = ParseResult(
                model=self.create_empty_model("Failed XML Parse"), success=False
            )
            result.add_error(f"Invalid XML format: {e}")
            return result

        except Exception as e:
            logger.error(f"Error parsing XML content: {e}")
            result = ParseResult(
                model=self.create_empty_model("Failed XML Parse"), success=False
            )
            result.add_error(f"Failed to parse XML content: {e}")
            return result

    def _parse_xml_root(self, root: Any) -> ParseResult:
        """Parse XML root element."""
        try:
            model = self._convert_xml_to_model(root)
            return ParseResult(model=model, success=True)

        except Exception as e:
            logger.error(f"Error converting XML to model: {e}")
            result = ParseResult(
                model=self.create_empty_model("Failed XML Conversion"), success=False
            )
            result.add_error(f"Failed to convert XML to model: {e}")
            return result

    def _convert_xml_to_model(self, root: Any) -> GNNInternalRepresentation:
        """Convert XML element to GNN internal representation."""

        # Extract model name from root element or attributes
        model_name = (
            root.get("name")
            or root.get("model_name")
            or root.tag.replace("_", " ").title()
            or "XMLGNNModel"
        )

        version = root.get("version", "1.0")

        model = GNNInternalRepresentation(
            model_name=model_name,
            version=version,
            annotation="Parsed from XML specification",
        )

        # Parse metadata
        self._parse_metadata(root, model)

        # Parse variables
        self._parse_xml_variables(root, model)

        # Parse connections
        self._parse_xml_connections(root, model)

        # Parse parameters
        self._parse_xml_parameters(root, model)

        # Parse equations
        self._parse_xml_equations(root, model)

        # Parse time specification
        self._parse_xml_time_specification(root, model)

        # Parse ontology mappings
        self._parse_xml_ontology_mappings(root, model)

        return model

    def _parse_metadata(self, root: Any, model: GNNInternalRepresentation) -> Any:
        """Parse metadata from XML."""
        metadata_elem = root.find(".//metadata")
        if metadata_elem is not None:
            annotation_elem = metadata_elem.find("annotation")
            if annotation_elem is not None and annotation_elem.text:
                model.annotation = annotation_elem.text

    def _parse_xml_variables(self, root: Any, model: GNNInternalRepresentation) -> Any:
        """Parse variables from each distinct physical element."""
        for var_elem in _unique_xml_elements(
            root, (".//variables/variable", ".//variable", ".//state", ".//node")
        ):
            variable = self._parse_xml_variable(var_elem)
            if variable:
                model.variables.append(variable)

    def _parse_xml_variable(self, var_elem: Any) -> Optional[Variable]:
        """Parse a single variable from XML element."""
        try:
            name = var_elem.get("name") or var_elem.get("id") or var_elem.tag

            # Get variable type
            var_type_str = var_elem.get("type", "hidden_state")
            try:
                var_type = VariableType(var_type_str)
            except ValueError:
                var_type = self._infer_variable_type_from_name(name)

            # Get data type
            data_type_str = var_elem.get("data_type", "continuous")
            try:
                data_type = DataType(data_type_str)
            except ValueError:
                data_type = DataType.CONTINUOUS

            # Get dimensions
            dimensions: list[Any] = []
            dims_str = var_elem.get("dimensions")
            if dims_str:
                try:
                    dimensions = [int(d.strip()) for d in dims_str.split(",")]
                except ValueError:
                    logger.debug(
                        "Non-integer dimension in '%s', keeping as string: %s",
                        var_elem.get("name", "?"),
                        dims_str,
                    )

            # Get description
            description = var_elem.get("description", "")
            if not description:
                desc_elem = var_elem.find("description")
                if desc_elem is not None and desc_elem.text:
                    description = desc_elem.text

            return Variable(
                name=name,
                var_type=var_type,
                dimensions=dimensions,
                data_type=data_type,
                description=description,
            )

        except Exception as e:
            logger.warning(f"Failed to parse XML variable {var_elem.tag}: {e}")
            return None

    def _parse_xml_connections(
        self, root: Any, model: GNNInternalRepresentation
    ) -> Any:
        """Parse connections from XML."""
        # Look for connections in prioritized order to avoid duplicates
        connection_elements: list[Any] = []

        # First try structured format
        connections_in_container = root.findall(".//connections/connection")
        if connections_in_container:
            connection_elements = connections_in_container
        else:
            # Fall back to other formats if no structured format found
            connection_elements = (
                root.findall(".//connection")
                + root.findall(".//edge")
                + root.findall(".//arc")
            )

        for conn_elem in connection_elements:
            connection = self._parse_xml_connection(conn_elem)
            if connection:
                model.connections.append(connection)

    def _parse_xml_connection(self, conn_elem: Any) -> Optional[Connection]:
        """Parse a single connection from XML element."""
        try:
            # Get source and target variables
            source_vars: list[Any] = []
            target_vars: list[Any] = []

            # Multiple ways to specify sources and targets
            source_attr = conn_elem.get("source") or conn_elem.get("from")
            target_attr = conn_elem.get("target") or conn_elem.get("to")

            if source_attr:
                source_vars = [s.strip() for s in source_attr.split(",")]
            if target_attr:
                target_vars = [t.strip() for t in target_attr.split(",")]

            # Try subelements
            sources_elem = conn_elem.find("sources")
            if sources_elem is not None and sources_elem.text:
                source_vars = [s.strip() for s in sources_elem.text.split(",")]

            targets_elem = conn_elem.find("targets")
            if targets_elem is not None and targets_elem.text:
                target_vars = [t.strip() for t in targets_elem.text.split(",")]

            if not source_vars or not target_vars:
                return None

            # Get connection type
            conn_type_str = conn_elem.get("type", "directed")
            try:
                conn_type = ConnectionType(conn_type_str)
            except ValueError:
                conn_type = ConnectionType.DIRECTED

            # Get weight
            weight = None
            weight_str = conn_elem.get("weight")
            if weight_str:
                try:
                    weight = float(weight_str)
                except ValueError:
                    logger.debug("Non-numeric weight value ignored: %s", weight_str)

            # Get description
            description = conn_elem.get("description", "")

            return Connection(
                source_variables=source_vars,
                target_variables=target_vars,
                connection_type=conn_type,
                weight=weight,
                description=description,
            )

        except Exception as e:
            logger.warning(f"Failed to parse XML connection {conn_elem.tag}: {e}")
            return None

    def _parse_xml_parameters(self, root: Any, model: GNNInternalRepresentation) -> Any:
        """Parse parameters from each distinct physical element."""
        for param_elem in _unique_xml_elements(
            root, (".//parameters/parameter", ".//parameter", ".//param")
        ):
            parameter = self._parse_xml_parameter(param_elem)
            if parameter:
                model.parameters.append(parameter)

    def _parse_xml_parameter(self, param_elem: Any) -> Optional[Parameter]:
        """Parse a single parameter from XML element."""
        try:
            name = param_elem.get("name") or param_elem.get("id")
            if not name:
                return None

            value = param_elem.get("value") or param_elem.text
            type_hint = param_elem.get("type")
            description = param_elem.get("description", "")

            return Parameter(
                name=name, value=value, type_hint=type_hint, description=description
            )

        except Exception as e:
            logger.warning(f"Failed to parse XML parameter {param_elem.tag}: {e}")
            return None

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

        return VariableType.HIDDEN_STATE

    def _parse_xml_equations(self, root: Any, model: GNNInternalRepresentation) -> Any:
        """Parse equations from each distinct physical element."""
        for eq_elem in _unique_xml_elements(
            root, (".//equations/equation", ".//equation")
        ):
            equation = self._parse_xml_equation(eq_elem)
            if equation:
                model.equations.append(equation)

    def _parse_xml_equation(self, eq_elem: Any) -> Optional[Equation]:
        """Parse a single equation from XML element."""
        try:
            content = eq_elem.text or ""
            label = eq_elem.get("label")
            format_type = eq_elem.get("format", "latex")
            description = eq_elem.get("description", "")

            return Equation(
                content=content,
                label=label,
                format=format_type,
                description=description,
            )

        except Exception as e:
            logger.warning(f"Failed to parse XML equation {eq_elem.tag}: {e}")
            return None

    def _parse_xml_time_specification(
        self, root: Any, model: GNNInternalRepresentation
    ) -> Any:
        """Parse time specification from XML."""
        time_elem = root.find(".//time_specification")
        if time_elem is not None:
            time_type = time_elem.get("type", "Static")
            discretization = time_elem.get("discretization")
            horizon = time_elem.get("horizon")

            # Convert horizon to appropriate type
            if horizon:
                try:
                    if horizon.isdigit():
                        horizon = int(horizon)
                except Exception as e:
                    logger.debug(
                        f"Could not convert horizon to int, keeping as string: {e}"
                    )

            model.time_specification = TimeSpecification(
                time_type=time_type, discretization=discretization, horizon=horizon
            )

    def _parse_xml_ontology_mappings(
        self, root: Any, model: GNNInternalRepresentation
    ) -> Any:
        """Parse ontology mappings from each distinct physical element."""
        for mapping_elem in _unique_xml_elements(
            root, (".//ontology_mappings/mapping", ".//ontology/mapping", ".//mapping")
        ):
            mapping = self._parse_xml_ontology_mapping(mapping_elem)
            if mapping:
                model.ontology_mappings.append(mapping)

    def _parse_xml_ontology_mapping(
        self, mapping_elem: Any
    ) -> Optional[OntologyMapping]:
        """Parse a single ontology mapping from XML element."""
        try:
            variable_name = mapping_elem.get("variable")
            ontology_term = mapping_elem.get("term")
            description = mapping_elem.get("description", "")

            if not variable_name or not ontology_term:
                return None

            return OntologyMapping(
                variable_name=variable_name,
                ontology_term=ontology_term,
                description=description,
            )

        except Exception as e:
            logger.warning(
                f"Failed to parse XML ontology mapping {mapping_elem.tag}: {e}"
            )
            return None

    def get_supported_extensions(self) -> List[str]:
        """Get supported file extensions."""
        return [".xml"]

    def _extract_embedded_json_data(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract embedded JSON model data from XML comments."""
        import json
        import re

        # Look for JSON data in <!-- MODEL_DATA: {...} --> comments
        pattern = r"<!--\s*MODEL_DATA:\s*(\{.*\})\s*-->"
        match = re.search(pattern, content, re.DOTALL | re.MULTILINE)
        if match:
            try:
                json_data = match.group(1)
                return cast("dict[str, Any] | None", json.loads(json_data))
            except json.JSONDecodeError:
                logger.debug(
                    "Embedded MODEL_DATA JSON is malformed, falling back to XML parsing"
                )
        return None

    def _build_model_from_embedded_data(
        self, embedded_data: Dict[str, Any]
    ) -> GNNInternalRepresentation:
        """Parse model from embedded JSON data for perfect round-trip fidelity."""

        model = GNNInternalRepresentation(
            model_name=embedded_data.get("model_name", "XMLGNNModel"),
            annotation=embedded_data.get("annotation", ""),
            version=embedded_data.get("version", "1.0"),
        )

        # Restore variables
        for var_data in embedded_data.get("variables", []):
            var = Variable(
                name=var_data["name"],
                var_type=VariableType(var_data.get("var_type", "hidden_state")),
                data_type=DataType(var_data.get("data_type", "categorical")),
                dimensions=var_data.get("dimensions", []),
                description=var_data.get("description"),
            )
            model.variables.append(var)

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
            model.connections.append(conn)

        # Restore parameters
        for param_data in embedded_data.get("parameters", []):
            param = Parameter(name=param_data["name"], value=param_data["value"])
            model.parameters.append(param)

        # Restore time specification
        if embedded_data.get("time_specification"):
            time_data = embedded_data["time_specification"]
            model.time_specification = TimeSpecification(
                time_type=time_data.get("time_type", "dynamic"),
                discretization=time_data.get("discretization"),
                horizon=time_data.get("horizon"),
                step_size=time_data.get("step_size"),
            )

        # Restore ontology mappings
        for mapping_data in embedded_data.get("ontology_mappings", []):
            mapping = OntologyMapping(
                variable_name=mapping_data.get("variable_name", ""),
                ontology_term=mapping_data.get("ontology_term", ""),
                description=mapping_data.get("description"),
            )
            model.ontology_mappings.append(mapping)

        return model


class PNMLParser(XMLGNNParser):
    """Parser for PNML (Petri Net Markup Language) format."""

    def __init__(self) -> None:
        """Initialize the PNML parser."""
        super().__init__()

    def _convert_xml_to_model(self, root: Any) -> GNNInternalRepresentation:
        """Convert PNML to GNN internal representation."""

        # PNML has specific structure
        model_name = root.get("id", "PNMLGNNModel")

        model = GNNInternalRepresentation(
            model_name=model_name,
            annotation="Parsed from PNML (Petri Net) specification",
        )

        # Parse nets
        nets = root.findall(".//net")
        for net in nets:
            self._parse_pnml_net(net, model)

        return model

    def _parse_pnml_net(self, net_elem: Any, model: GNNInternalRepresentation) -> Any:
        """Parse a PNML net element."""
        # Parse places as variables
        places = net_elem.findall(".//place")
        for place in places:
            variable = self._parse_pnml_place(place)
            if variable:
                model.variables.append(variable)

        # Parse transitions as variables
        transitions = net_elem.findall(".//transition")
        for transition in transitions:
            variable = self._parse_pnml_transition(transition)
            if variable:
                model.variables.append(variable)

        # Parse arcs as connections
        arcs = net_elem.findall(".//arc")
        for arc in arcs:
            connection = self._parse_pnml_arc(arc)
            if connection:
                model.connections.append(connection)

    def _parse_pnml_place(self, place_elem: Any) -> Optional[Variable]:
        """Parse a PNML place as a variable."""
        place_id = place_elem.get("id")
        if not place_id:
            return None

        # Get name from text element
        name_elem = place_elem.find(".//text")
        name = name_elem.text if name_elem is not None and name_elem.text else place_id

        return Variable(
            name=name,
            var_type=VariableType.HIDDEN_STATE,  # Places represent states
            dimensions=[],
            data_type=DataType.INTEGER,  # Petri net places typically hold tokens (integers)
            description=f"PNML place: {place_id}",
        )

    def _parse_pnml_transition(self, trans_elem: Any) -> Optional[Variable]:
        """Parse a PNML transition as a variable."""
        trans_id = trans_elem.get("id")
        if not trans_id:
            return None

        # Get name from text element
        name_elem = trans_elem.find(".//text")
        name = name_elem.text if name_elem is not None and name_elem.text else trans_id

        return Variable(
            name=name,
            var_type=VariableType.ACTION,  # Transitions represent actions
            dimensions=[],
            data_type=DataType.BINARY,  # Transitions fire or don't fire
            description=f"PNML transition: {trans_id}",
        )

    def _parse_pnml_arc(self, arc_elem: Any) -> Optional[Connection]:
        """Parse a PNML arc as a connection."""
        source = arc_elem.get("source")
        target = arc_elem.get("target")

        if not source or not target:
            return None

        return Connection(
            source_variables=[source],
            target_variables=[target],
            connection_type=ConnectionType.DIRECTED,
            description=f"PNML arc: {source} -> {target}",
        )

    def get_supported_extensions(self) -> List[str]:
        """Get supported file extensions."""
        return [".pnml"]

    # _extract_embedded_json_data and _build_model_from_embedded_data defined on XMLGNNParser.
