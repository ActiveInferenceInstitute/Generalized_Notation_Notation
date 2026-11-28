"""GNN Markdown serializer.

Serializes GNN internal representations to Markdown format.
"""

from datetime import datetime
from typing import Any

from .base_serializer import BaseGNNSerializer
from .common import GNNInternalRepresentation


class MarkdownSerializer(BaseGNNSerializer):
    """Serializer for GNN Markdown format."""

    def serialize(self, model: GNNInternalRepresentation) -> str:
        """Convert GNN model back to Markdown format."""
        sections: list[Any] = []

        # GNNVersionAndFlags
        sections.append("## GNNVersionAndFlags")
        sections.append("Version: 1.0")
        sections.append("")

        # ModelName
        sections.append("## ModelName")
        sections.append(model.model_name)
        sections.append("")

        # ModelAnnotation
        if model.annotation:
            sections.append("## ModelAnnotation")
            sections.append(model.annotation)
            sections.append("")

        if model.boundary_annotations:
            sections.append("## BoundaryAnnotation")
            for variable, fields in model.boundary_annotations.items():
                values = "; ".join(f"{key}={value}" for key, value in fields.items())
                sections.append(f"{variable}: {values}")
            sections.append("")

        # StateSpaceBlock
        if model.variables:
            sections.append("## StateSpaceBlock")
            for var in model.variables:
                dims_str = (
                    f"[{','.join(map(str, var.dimensions))}]" if var.dimensions else ""
                )
                sections.append(f"{var.name}{dims_str},{var.data_type.value}")
            sections.append("")

        # Connections
        if model.connections:
            sections.append("## Connections")
            for conn in model.connections:
                if conn.connection_type.value == "directed":
                    op = ">"
                elif conn.connection_type.value == "undirected":
                    op = "-"
                else:
                    op = "->"

                for src in conn.source_variables:
                    for tgt in conn.target_variables:
                        suffix = f":{conn.annotation}" if conn.annotation else ""
                        sections.append(f"{src}{op}{tgt}{suffix}")
            sections.append("")

        # InitialParameterization
        if model.parameters:
            sections.append("## InitialParameterization")
            for param in model.parameters:
                sections.append(f"{param.name} = {param.value}")
            sections.append("")

        # Equations
        if model.equations:
            sections.append("## Equations")
            for eq in model.equations:
                if eq.label:
                    sections.append(f"**{eq.label}:**")
                sections.append(f"$${eq.content}$$")
                sections.append("")

        # Time
        if model.time_specification:
            sections.append("## Time")
            sections.append(model.time_specification.time_type)
            if model.time_specification.discretization:
                sections.append(model.time_specification.discretization)
            if model.time_specification.horizon:
                sections.append(
                    f"ModelTimeHorizon = {model.time_specification.horizon}"
                )
            sections.append("")

        # ActInfOntologyAnnotation
        if model.ontology_mappings:
            sections.append("## ActInfOntologyAnnotation")
            for mapping in model.ontology_mappings:
                sections.append(f"{mapping.variable_name} = {mapping.ontology_term}")
            sections.append("")

        # Footer
        sections.append("## Footer")
        sections.append(f"Generated: {datetime.now().isoformat()}")
        sections.append("")

        # Signature
        sections.append("## Signature")
        if model.checksum:
            sections.append(f"Checksum: {model.checksum}")
        sections.append("")

        return "\n".join(sections)
