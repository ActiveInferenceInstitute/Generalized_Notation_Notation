#!/usr/bin/env python3
"""
Dependency-free direct markdown parser for the GNN round-trip test suite.

Extracted from the former ``testing.test_round_trip`` monolith.
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
)

from gnn.parsers.common import canonical_section_heading

from .round_trip_availability import GNNInternalRepresentation

logger = logging.getLogger(__name__)


class _DirectMarkdownParser:
    """A simple, robust markdown parser that doesn't rely on complex validation."""

    def parse_file(self, file_path: Path) -> "GNNInternalRepresentation":
        """Parse a GNN markdown file directly."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return self.parse_content(content)

    def parse_content(self, content: str) -> "GNNInternalRepresentation":
        """Parse GNN markdown content."""
        sections = self._extract_sections(content)
        model = GNNInternalRepresentation(
            model_name=sections.get("ModelName", "Unknown Model"),
            annotation=sections.get("ModelAnnotation", ""),
        )
        model.version = sections.get("GNNVersionAndFlags", "1.0")
        model.created_at = datetime.now()
        model.modified_at = datetime.now()
        model.checksum = None
        model.extensions = {}
        model.raw_sections = sections
        model.equations = []
        if "StateSpaceBlock" in sections:
            model.variables = self._parse_variables(sections["StateSpaceBlock"])
        if "Connections" in sections:
            model.connections = self._parse_connections(sections["Connections"])
        if "InitialParameterization" in sections:
            model.parameters = self._parse_parameters(
                sections["InitialParameterization"]
            )
        if "Time" in sections:
            time_data = self._parse_time_spec(sections["Time"])
            model.time_specification = (
                type(
                    "TimeSpecification",
                    (),
                    {
                        "time_type": time_data.get("time_type", "dynamic"),
                        "discretization": time_data.get("discretization", None),
                        "horizon": time_data.get("horizon", None),
                        "step_size": time_data.get("step_size", None),
                    },
                )()
                if time_data
                else None
            )
        if "ActInfOntologyAnnotation" in sections:
            model.ontology_mappings = self._parse_ontology(
                sections["ActInfOntologyAnnotation"]
            )
        return model

    def _extract_sections(self, content: str) -> Dict[str, str]:
        """Extract sections from GNN markdown content."""
        sections: Dict[str, str] = {}
        current_section = None
        current_content: List[str] = []
        for line in content.split("\n"):
            if line.startswith("## "):
                if current_section:
                    sections[current_section] = "\n".join(current_content).strip()
                current_section = canonical_section_heading(line[3:].strip())
                current_content = []
            elif current_section:
                current_content.append(line)
        if current_section:
            sections[current_section] = "\n".join(current_content).strip()
        return sections

    def _parse_variables(self, content: str) -> List[Any]:
        """Parse variables from StateSpaceBlock content."""
        variables: list[Any] = []
        var_pattern = re.compile(r"(\w+)\[([^\]]+)\](?:\s*#\s*(.*))?")
        for line in content.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = var_pattern.match(line)
            if match:
                name = match.group(1)
                dims_str = match.group(2)
                description = match.group(3) or ""
                dims_parts = [p.strip() for p in dims_str.split(",")]
                dimensions: List[int] = []
                data_type = "float"
                for part in dims_parts:
                    if part.startswith("type="):
                        raw_type = part[5:]
                        type_mapping: dict[str, Any] = {
                            "int": "integer",
                            "float": "float",
                            "bool": "binary",
                            "str": "categorical",
                            "string": "categorical",
                        }
                        data_type = type_mapping.get(raw_type, raw_type)
                    else:
                        try:
                            dimensions.append(int(part))
                        except ValueError as e:
                            logger.debug(
                                "Ignoring non-dimension token %r: %s",
                                part,
                                e,
                            )
                var_type = "hidden_state"
                if name in ["A", "B", "C", "D"]:
                    var_type = (
                        "likelihood_matrix"
                        if name == "A"
                        else "transition_matrix"
                        if name == "B"
                        else "preference_vector"
                        if name == "C"
                        else "prior_vector"
                    )
                elif name in ["o", "u"]:
                    var_type = "observation" if name == "o" else "action"
                elif name in ["s", "s_prime"]:
                    var_type = "hidden_state"
                elif name in ["π", "G"]:
                    var_type = "policy"
                var = type(
                    "Variable",
                    (),
                    {
                        "name": name,
                        "dimensions": dimensions,
                        "var_type": type("VarType", (), {"value": var_type})(),
                        "data_type": type("DataType", (), {"value": data_type})(),
                        "description": description,
                    },
                )()
                variables.append(var)
        return variables

    def _parse_connections(self, content: str) -> List[Any]:
        """Parse connections from Connections content."""
        connections: list[Any] = []
        for line in content.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ">" in line:
                parts = line.split(">")
                if len(parts) == 2:
                    conn = type(
                        "Connection",
                        (),
                        {
                            "source_variables": [parts[0].strip()],
                            "target_variables": [parts[1].strip()],
                            "connection_type": type(
                                "ConnType", (), {"value": "directed"}
                            )(),
                            "weight": None,
                            "description": "",
                        },
                    )()
                    connections.append(conn)
        return connections

    def _parse_parameters(self, content: str) -> List[Any]:
        """Parse parameters from InitialParameterization content."""
        parameters: list[Any] = []
        for line in content.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                parts = line.split("=", 1)
                if len(parts) == 2:
                    param = type(
                        "Parameter",
                        (),
                        {
                            "name": parts[0].strip(),
                            "value": parts[1].strip(),
                            "type_hint": "constant",
                            "description": "",
                        },
                    )()
                    parameters.append(param)
        return parameters

    def _parse_time_spec(self, content: str) -> Dict[str, str]:
        """Parse time specification."""
        time_spec: Dict[str, str] = {}
        for line in content.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                time_spec[key.strip().lower()] = value.strip()
            else:
                time_spec["time_type"] = line
        return time_spec

    def _parse_ontology(self, content: str) -> List[Any]:
        """Parse ontology mappings."""
        mappings: list[Any] = []
        for line in content.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                parts = line.split("=", 1)
                if len(parts) == 2:
                    mapping = type(
                        "OntologyMapping",
                        (),
                        {
                            "variable_name": parts[0].strip(),
                            "ontology_term": parts[1].strip(),
                            "description": "",
                        },
                    )()
                    mappings.append(mapping)
        return mappings
