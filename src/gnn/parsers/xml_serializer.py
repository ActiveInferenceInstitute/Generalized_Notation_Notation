"""GNN XML serializer.

Serializes GNN internal representations to XML format.
"""

import json
import xml.etree.ElementTree as ET  # nosec B405
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
        if model.extensions:
            ET.SubElement(metadata, "extensions").text = json.dumps(
                model.extensions, ensure_ascii=False
            )

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
                if var.constraints:
                    var_elem.set(
                        "constraints", json.dumps(var.constraints, ensure_ascii=False)
                    )

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
                if conn.annotation is not None:
                    conn_elem.set("annotation", conn.annotation)

                # Create child elements in consistent order
                sources_elem = ET.SubElement(conn_elem, "sources")
                sources_elem.text = ",".join(conn.source_variables)

                targets_elem = ET.SubElement(conn_elem, "targets")
                targets_elem.text = ",".join(conn.target_variables)

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
                param_elem.set("value_encoding", "json")
                param_elem.text = json.dumps(param.value, ensure_ascii=False)
                if param.description:
                    param_elem.set("description", param.description)
                if param.section:
                    param_elem.set("section", param.section)
                if param.metadata:
                    param_elem.set(
                        "metadata", json.dumps(param.metadata, ensure_ascii=False)
                    )

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
            if model.time_specification.horizon is not None:
                time_elem.set("horizon", str(model.time_specification.horizon))
                if isinstance(model.time_specification.horizon, str):
                    time_elem.set("horizon_encoding", "string")
            if model.time_specification.step_size is not None:
                time_elem.set("step_size", str(model.time_specification.step_size))
            if model.time_specification.metadata:
                time_elem.set(
                    "metadata",
                    json.dumps(model.time_specification.metadata, ensure_ascii=False),
                )
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

        # Convert to string with consistent formatting
        rough_string = ET.tostring(root, encoding="unicode")
        reparsed = minidom.parseString(rough_string)  # nosec B318

        # Trim the document terminator without removing authored blank lines.
        return reparsed.toprettyxml(indent="  ").rstrip("\n")
