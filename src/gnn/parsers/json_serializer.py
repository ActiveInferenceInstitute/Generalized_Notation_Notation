"""GNN JSON serializer.

Serializes GNN internal representations to JSON format.
"""

import json
from typing import Any

from .base_serializer import BaseGNNSerializer
from .common import GNNInternalRepresentation


class JSONSerializer(BaseGNNSerializer):
    """Serializer for JSON data interchange format."""

    def serialize(self, model: GNNInternalRepresentation) -> str:
        """Convert GNN model to JSON format."""
        # Manually construct the data dictionary to handle dynamic objects properly
        data: dict[str, Any] = {
            "model_name": model.model_name,
            "annotation": model.annotation,
            "version": getattr(model, "version", "1.0"),
            "checksum": getattr(model, "checksum", None),
            "created_at": model.created_at.isoformat()
            if hasattr(model, "created_at")
            else None,
            "modified_at": model.modified_at.isoformat()
            if hasattr(model, "modified_at")
            else None,
            "extensions": getattr(model, "extensions", {}),
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
                    "description": getattr(var, "description", ""),
                    **({"constraints": var.constraints} if var.constraints else {}),
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
                    "weight": getattr(conn, "weight", None),
                    "description": getattr(conn, "description", ""),
                }
                for conn in model.connections
            ],
            "parameters": [
                {
                    "name": param.name,
                    "value": param.value,
                    "type_hint": getattr(param, "type_hint", "constant"),
                    "description": getattr(param, "description", ""),
                    **({"metadata": param.metadata} if param.metadata else {}),
                    **({"section": param.section} if param.section else {}),
                }
                for param in model.parameters
            ],
            "equations": [
                {
                    "label": getattr(eq, "label", None),
                    "content": getattr(eq, "content", ""),
                    "format": getattr(eq, "format", "latex"),
                    "description": getattr(eq, "description", ""),
                }
                for eq in (model.equations if hasattr(model, "equations") else [])
            ],
            "time_specification": {
                **self._serialize_time_spec(model.time_specification),
                **(
                    {"metadata": model.time_specification.metadata}
                    if model.time_specification.metadata
                    else {}
                ),
            }
            if hasattr(model, "time_specification") and model.time_specification
            else None,
            "ontology_mappings": [
                {
                    "variable_name": mapping.variable_name
                    if hasattr(mapping, "variable_name")
                    else str(mapping),
                    "ontology_term": mapping.ontology_term
                    if hasattr(mapping, "ontology_term")
                    else "",
                    "description": getattr(mapping, "description", ""),
                }
                for mapping in (
                    model.ontology_mappings
                    if hasattr(model, "ontology_mappings")
                    else []
                )
            ],
        }

        return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)
