"""GNN YAML serializer.

Serializes GNN internal representations to YAML format.
"""

from typing import Any, Dict, cast

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False
from .base_serializer import BaseGNNSerializer
from .common import GNNInternalRepresentation


class YAMLSerializer(BaseGNNSerializer):
    """Serializer for YAML configuration format."""

    def serialize(self, model: GNNInternalRepresentation) -> str:
        """Convert GNN model to YAML format."""
        # Create a cleaner structure for YAML with consistent ordering
        data: dict[str, Any] = {
            "model_name": model.model_name,
            "version": model.version,
            "annotation": model.annotation,
            **({"extensions": model.extensions} if model.extensions else {}),
            "variables": [
                {
                    "name": var.name,
                    "type": var.var_type.value,
                    "data_type": var.data_type.value,
                    "dimensions": var.dimensions,
                    "description": var.description,
                    **({"constraints": var.constraints} if var.constraints else {}),
                }
                for var in sorted(model.variables, key=lambda v: v.name)
            ],
            "connections": [
                {
                    "annotation": conn.annotation,
                    "source_variables": conn.source_variables,
                    "target_variables": conn.target_variables,
                    "connection_type": conn.connection_type.value,
                    "weight": conn.weight,
                    "description": conn.description,
                }
                for conn in sorted(
                    model.connections,
                    key=lambda c: (
                        ",".join(sorted(c.source_variables)),
                        ",".join(sorted(c.target_variables)),
                    ),
                )
            ],
            "parameters": [
                {
                    "name": param.name,
                    "value": param.value,
                    "type_hint": param.type_hint,
                    "description": param.description,
                    **({"metadata": param.metadata} if param.metadata else {}),
                    **({"section": param.section} if param.section else {}),
                }
                for param in sorted(model.parameters, key=lambda p: p.name)
            ],
            "equations": [
                {
                    "content": eq.content,
                    "label": eq.label,
                    "format": eq.format,
                    "description": eq.description,
                }
                for eq in sorted(model.equations, key=lambda e: e.label or "")
            ],
            "time_specification": {
                "time_type": model.time_specification.time_type,
                "discretization": model.time_specification.discretization,
                "horizon": model.time_specification.horizon,
                "step_size": model.time_specification.step_size,
                **(
                    {"metadata": model.time_specification.metadata}
                    if model.time_specification.metadata
                    else {}
                ),
            }
            if model.time_specification
            else None,
            "ontology_mappings": [
                {
                    "variable_name": mapping.variable_name,
                    "ontology_term": mapping.ontology_term,
                    "description": mapping.description,
                }
                for mapping in sorted(
                    model.ontology_mappings, key=lambda m: m.variable_name
                )
            ],
            "created_at": model.created_at.isoformat(),
            "modified_at": model.modified_at.isoformat(),
        }

        if not HAS_YAML:
            return self._dict_to_yaml_like(data)

        return cast(
            "str",
            yaml.safe_dump(
                data, default_flow_style=False, allow_unicode=True, sort_keys=True
            ),
        )

    def _dict_to_yaml_like(self, data: Dict[str, Any], indent: int = 0) -> str:
        """Convert dict to YAML-like format when PyYAML is not available."""
        lines: list[Any] = []
        spaces = "  " * indent

        for key, value in (
            sorted(data.items()) if isinstance(data, dict) else data.items()
        ):
            if isinstance(value, dict):
                lines.append(f"{spaces}{key}:")
                lines.append(self._dict_to_yaml_like(value, indent + 1))
            elif isinstance(value, list):
                lines.append(f"{spaces}{key}:")
                for item in value:
                    if isinstance(item, dict):
                        lines.append(f"{spaces}- ")
                        lines.append(self._dict_to_yaml_like(item, indent + 1))
                    else:
                        lines.append(f"{spaces}- {item}")
            else:
                lines.append(f"{spaces}{key}: {value}")

        return "\n".join(lines)
