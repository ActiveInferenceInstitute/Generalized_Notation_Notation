"""GNN XML serializer.

Serializes GNN internal representations to XML format.
"""

import json
import xml.etree.ElementTree as ET  # nosec B405
from typing import Any
from xml.dom import (
    minidom,  # nosec B408
)

from .base_serializer import BaseGNNSerializer
from .common import GNNInternalRepresentation


class XMLSerializer(BaseGNNSerializer):
    """Serializer for XML format with deterministic output."""

    def serialize(self, model: GNNInternalRepresentation) -> str:
        """Convert GNN model to XML format with consistent ordering."""
        root = ET.Element("gnn_model")

        # Set attributes in alphabetical order for consistency
        root.set("name", model.model_name)
        root.set("version", model.version)

        # Metadata - always first
        metadata = ET.SubElement(root, "metadata")
        ET.SubElement(metadata, "annotation").text = model.annotation or ""
        ET.SubElement(metadata, "created_at").text = model.created_at.isoformat()
        ET.SubElement(metadata, "modified_at").text = model.modified_at.isoformat()

        # Variables - sorted by name for consistency
        if model.variables:
            variables = ET.SubElement(root, "variables")
            sorted_vars = sorted(model.variables, key=lambda v: v.name)
            for var in sorted_vars:
                var_elem = ET.SubElement(variables, "variable")
                # Set attributes in alphabetical order
                var_elem.set("data_type", var.data_type.value)
                if var.dimensions:
                    var_elem.set("dimensions", ",".join(map(str, var.dimensions)))
                var_elem.set("name", var.name)
                var_elem.set("type", var.var_type.value)
                if var.description:
                    var_elem.set("description", var.description)

        # Connections - sorted by source then target for consistency
        if model.connections:
            connections = ET.SubElement(root, "connections")
            sorted_conns = sorted(
                model.connections,
                key=lambda c: (
                    ",".join(sorted(c.source_variables)),
                    ",".join(sorted(c.target_variables)),
                ),
            )
            for conn in sorted_conns:
                conn_elem = ET.SubElement(connections, "connection")
                conn_elem.set("type", conn.connection_type.value)
                if conn.weight is not None:
                    conn_elem.set("weight", str(conn.weight))

                # Create child elements in consistent order
                sources_elem = ET.SubElement(conn_elem, "sources")
                sources_elem.text = ",".join(sorted(conn.source_variables))

                targets_elem = ET.SubElement(conn_elem, "targets")
                targets_elem.text = ",".join(sorted(conn.target_variables))

                if conn.description:
                    desc_elem = ET.SubElement(conn_elem, "description")
                    desc_elem.text = conn.description

        # Parameters - sorted by name
        if model.parameters:
            parameters = ET.SubElement(root, "parameters")
            sorted_params = sorted(model.parameters, key=lambda p: p.name)
            for param in sorted_params:
                param_elem = ET.SubElement(parameters, "parameter")
                param_elem.set("name", param.name)
                if param.type_hint:
                    param_elem.set("type", param.type_hint)
                param_elem.text = str(param.value)
                if param.description:
                    param_elem.set("description", param.description)

        # Equations - sorted by label
        if model.equations:
            equations = ET.SubElement(root, "equations")
            sorted_eqs = sorted(model.equations, key=lambda e: e.label or "")
            for eq in sorted_eqs:
                eq_elem = ET.SubElement(equations, "equation")
                if eq.format:
                    eq_elem.set("format", eq.format)
                if eq.label:
                    eq_elem.set("label", eq.label)
                eq_elem.text = eq.content
                if eq.description:
                    eq_elem.set("description", eq.description)

        # Time Specification
        if model.time_specification:
            time_elem = ET.SubElement(root, "time_specification")
            if model.time_specification.discretization:
                time_elem.set("discretization", model.time_specification.discretization)
            if model.time_specification.horizon:
                time_elem.set("horizon", str(model.time_specification.horizon))
            if model.time_specification.step_size:
                time_elem.set("step_size", str(model.time_specification.step_size))
            time_elem.set("type", model.time_specification.time_type)

        # Ontology Mappings - sorted by variable name
        if model.ontology_mappings:
            ontology = ET.SubElement(root, "ontology_mappings")
            sorted_mappings = sorted(
                model.ontology_mappings, key=lambda m: m.variable_name
            )
            for mapping in sorted_mappings:
                mapping_elem = ET.SubElement(ontology, "mapping")
                mapping_elem.set("term", mapping.ontology_term)
                mapping_elem.set("variable", mapping.variable_name)
                if mapping.description:
                    mapping_elem.set("description", mapping.description)

        # Add embedded complete model data as XML comment for perfect round-trip
        model_data: dict[str, Any] = {
            "model_name": model.model_name,
            "annotation": model.annotation,
            "variables": [
                {
                    "name": var.name,
                    "var_type": var.var_type.value
                    if hasattr(var, "var_type")
                    else "hidden_state",
                    "data_type": var.data_type.value
                    if hasattr(var, "data_type")
                    else "categorical",
                    "dimensions": var.dimensions if hasattr(var, "dimensions") else [],
                    "description": var.description,
                }
                for var in model.variables
            ],
            "connections": [
                {
                    "annotation": getattr(conn, "annotation", None),
                    "source_variables": conn.source_variables
                    if hasattr(conn, "source_variables")
                    else [],
                    "target_variables": conn.target_variables
                    if hasattr(conn, "target_variables")
                    else [],
                    "connection_type": conn.connection_type.value
                    if hasattr(conn, "connection_type")
                    else "directed",
                }
                for conn in model.connections
            ],
            "parameters": [
                {
                    "name": param.name,
                    "value": param.value,
                    "param_type": getattr(param, "param_type", "constant"),
                }
                for param in model.parameters
            ],
            "equations": [
                str(eq)
                for eq in (model.equations if hasattr(model, "equations") else [])
            ],
            "time_specification": self._serialize_time_spec(model.time_specification)
            if hasattr(model, "time_specification") and model.time_specification
            else None,
            "ontology_mappings": self._serialize_ontology_mappings(
                model.ontology_mappings
            )
            if hasattr(model, "ontology_mappings")
            else [],
        }

        # Convert to string with consistent formatting
        rough_string = ET.tostring(root, encoding="unicode")
        reparsed = minidom.parseString(rough_string)  # nosec B318

        # Get pretty XML and normalize whitespace
        pretty_xml = reparsed.toprettyxml(indent="  ")

        # Remove empty lines and normalize
        lines = [line for line in pretty_xml.split("\n") if line.strip()]

        # Add embedded data as XML comment before the closing tag
        if lines and lines[-1].strip() == "</gnn_model>":
            lines.insert(
                -1,
                f"<!-- MODEL_DATA: {json.dumps(model_data, separators=(',', ':'))} -->",
            )

        return "\n".join(lines)

    def _serialize_time_spec(self, time_spec: Any) -> Any:
        """Serialize time specification object."""
        if not time_spec:
            return None
        return {
            "time_type": getattr(time_spec, "time_type", "dynamic"),
            "discretization": getattr(time_spec, "discretization", None),
            "horizon": getattr(time_spec, "horizon", None),
            "step_size": getattr(time_spec, "step_size", None),
        }

    def _serialize_ontology_mappings(self, mappings: Any) -> Any:
        """Serialize ontology mappings."""
        if not mappings:
            return []
        return [
            {
                "variable_name": mapping.variable_name
                if hasattr(mapping, "variable_name")
                else str(mapping),
                "ontology_term": mapping.ontology_term
                if hasattr(mapping, "ontology_term")
                else "",
                "description": getattr(mapping, "description", None),
            }
            for mapping in mappings
        ]
