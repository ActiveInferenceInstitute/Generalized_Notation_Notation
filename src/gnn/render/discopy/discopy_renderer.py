#!/usr/bin/env python3
"""
DisCoPy Renderer

Renders GNN specifications to DisCoPy categorical diagram code for compositional models.
This renderer creates executable DisCoPy visualizations configured from parsed GNN POMDP specifications.

The supported generator encodes the specification's declared ``Connections``
section: each connection becomes one monoidal box whose domain is the tensor of
its source wires and whose codomain is the tensor of its target wires. The
exact connection record is exported beside the diagram so a consumer can
compare the diagram to the source without reverse-engineering the drawing.

Features:
- GNN-to-DisCoPy connection extraction
- One box per declared connection with source/target wiring
- String diagram visualization
- Pipeline integration support

Author: GNN DisCoPy Integration
Date: 2024
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from gnn.parsers.common import canonical_section_heading
from gnn.render.naming import atomic_write_text

from .gnn_parsing import parse_gnn_content
from .symmetry import build_matrix_permutation_metadata

_RELATION_ALIASES = {
    "->": ">",
    ">": ">",
    "→": ">",
    "-": "-",
    "—": "-",
    "=": "=",
}
_RELATION_LABELS = {">": "generates", "-": "couples", "=": "defines"}


def _split_endpoint(value: Any) -> List[str]:
    """Split a declared connection endpoint into individual wire names."""
    if isinstance(value, (list, tuple)):
        parts = [str(item) for item in value]
    else:
        text = str(value).strip()
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1]
        parts = text.split(",")
    return [part.strip() for part in parts if part.strip()]


def _parse_connection_entry(entry: Any, index: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return the literal metadata record and the normalised wiring record."""
    if isinstance(entry, dict):
        if "source" not in entry or "target" not in entry:
            raise ValueError(f"connection #{index} is missing a source or target")
        source_value = entry["source"]
        target_value = entry["target"]
        relation_value = entry.get("relation", ">")
        metadata = {
            "source": source_value,
            "relation": relation_value,
            "target": target_value,
        }
    elif isinstance(entry, str):
        text = entry.strip()
        match = re.match(r"^(.*?)\s*(->|>|→|-|—|=)\s*(.*?)$", text)
        if not match:
            raise ValueError(
                f"connection #{index} {text!r} is not a 'source > target' declaration"
            )
        source_value, relation_value, target_value = match.groups()
        metadata = {
            "source": source_value.strip(),
            "relation": relation_value.strip(),
            "target": target_value.strip(),
        }
    else:
        raise ValueError(f"connection #{index} is not a mapping or declaration string")

    relation = _RELATION_ALIASES.get(str(relation_value).strip())
    if relation is None:
        raise ValueError(
            f"connection #{index} declares unsupported relation "
            f"{relation_value!r}; expected '>', '-', '->' or '='"
        )
    sources = _split_endpoint(source_value)
    targets = _split_endpoint(target_value)
    if not sources:
        raise ValueError(f"connection #{index} has no source wire")
    if not targets:
        raise ValueError(f"connection #{index} has no target wire")
    wiring = {
        "source": sources,
        "relation": relation,
        "target": targets,
    }
    return metadata, wiring


def _normalise_connections(
    connections: Any,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Validate declared connections and return literal and wiring records."""
    if connections is None:
        return [], []
    if not isinstance(connections, (list, tuple)):
        raise ValueError("connections must be a sequence of declarations")
    metadata: List[Dict[str, Any]] = []
    wiring: List[Dict[str, Any]] = []
    for index, entry in enumerate(connections):
        literal, edge = _parse_connection_entry(entry, index)
        metadata.append(literal)
        wiring.append(edge)
    return metadata, wiring


def _variable_dimensions(gnn_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Return declared wire dimensions where the specification provides them."""
    dimensions: Dict[str, Any] = {}
    for variable in gnn_spec.get("variables", []) or []:
        if not isinstance(variable, dict):
            continue
        name = variable.get("name")
        if name:
            dimensions[str(name)] = variable.get("dimensions", [])
    model_parameters = gnn_spec.get("model_parameters") or {}
    if isinstance(model_parameters, dict):
        for key in (
            "state_factors",
            "observation_modalities",
            "control_factors",
        ):
            for factor in model_parameters.get(key, []) or []:
                if not isinstance(factor, dict):
                    continue
                name = factor.get("name")
                if name and str(name) not in dimensions:
                    dimensions[str(name)] = factor.get("dimensions", [])
    return dimensions


class DisCoPyRenderer:
    """
    DisCoPy renderer for generating categorical diagram code from GNN specifications.
    """

    def __init__(self, options: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize DisCoPy renderer.

        Args:
            options: Optional configuration options
        """
        self.options = options or {}
        self.logger = logging.getLogger(__name__)

    def render_file(self, gnn_file_path: Path, output_path: Path) -> Tuple[bool, str]:
        """
        Render a single GNN file to DisCoPy categorical diagram code.

        Args:
            gnn_file_path: Path to GNN file
            output_path: Path for output DisCoPy script

        Returns:
            Tuple of (success, message)
        """
        try:
            # Read GNN file
            with open(gnn_file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse GNN content (simplified for now)
            gnn_spec = self._parse_gnn_content(content, gnn_file_path.stem)
            gnn_spec["source_file"] = gnn_file_path.name

            # Generate DisCoPy categorical diagram code
            discopy_code = self._generate_discopy_diagram_code(
                gnn_spec, gnn_file_path.stem
            )

            # Write output file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(output_path, discopy_code)

            self.logger.info(f"Generated DisCoPy diagram: {output_path}")
            return True, "Successfully generated DisCoPy categorical diagram code"

        except Exception as e:
            error_msg = f"Error rendering {gnn_file_path}: {e}"
            self.logger.error(error_msg)
            return False, error_msg

    def _parse_gnn_content(self, content: str, model_name: str) -> Dict[str, Any]:
        """Parse GNN content into a structured dictionary (simplified parser)."""
        gnn_spec: dict[str, Any] = {
            "model_name": model_name,
            "variables": [],
            "model_parameters": {},
            "initial_parameterization": {},
            "connections": [],
        }

        self.logger.info(f"Parsing GNN content for model: {model_name}")

        # Simple parser for key sections
        lines = content.split("\n")
        current_section = None
        sections_found: list[Any] = []

        for line in lines:
            line = line.strip()
            if line.startswith("## "):
                current_section = canonical_section_heading(line[3:].strip())
                sections_found.append(current_section)
                self.logger.debug(f"Found section: {current_section}")
            elif current_section == "ModelParameters" and ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                try:
                    if "." in value:
                        gnn_spec["model_parameters"][key] = float(value)
                    else:
                        gnn_spec["model_parameters"][key] = int(value)
                    self.logger.debug(
                        f"Extracted parameter: {key}={gnn_spec['model_parameters'][key]}"
                    )
                except ValueError:
                    gnn_spec["model_parameters"][key] = value
                    self.logger.debug(f"Extracted parameter (string): {key}={value}")

        parsed_sections = parse_gnn_content(content)
        for raw in parsed_sections.get("StateSpaceBlock", []) or []:
            match = re.match(r"^([^\s\[#]+)\s*(?:\[([^\]]*)\])?", str(raw))
            if not match:
                continue
            name, dims = match.groups()
            dimensions = [
                int(part.strip())
                for part in (dims or "").split(",")
                if part.strip().isdigit()
            ]
            gnn_spec["variables"].append(
                {"name": name, "dimensions": dimensions or [1]}
            )
        for raw in parsed_sections.get("Connections", []) or []:
            declaration = str(raw).strip()
            if declaration and not declaration.startswith("#"):
                gnn_spec["connections"].append(declaration)

        self.logger.info(
            f"Parsed {len(sections_found)} sections, "
            f"{len(gnn_spec['model_parameters'])} model parameters, "
            f"{len(gnn_spec['connections'])} connections"
        )
        return gnn_spec

    def _generate_discopy_diagram_code(
        self, gnn_spec: Dict[str, Any], model_name: str
    ) -> str:
        """
        Generate executable DisCoPy categorical diagram code from GNN specification.

        Args:
            gnn_spec: Parsed GNN specification
            model_name: Name of the model

        Returns:
            Generated Python code string
        """
        # Extract key information from GNN spec
        model_display_name = gnn_spec.get("model_name", model_name)
        source_file = (
            gnn_spec.get("source_file")
            or gnn_spec.get("file_name")
            or gnn_spec.get("file_path")
            or "unknown"
        )
        source_file = Path(str(source_file)).name

        # Extract dimensions from model parameters
        model_params = gnn_spec.get("model_parameters", {})
        num_states = model_params.get("num_hidden_states", 3)
        num_observations = model_params.get("num_obs", 3)
        num_actions = model_params.get("num_actions", 3)

        # Try to extract from variables if available
        variables = gnn_spec.get("variables", [])
        for var in variables:
            if var.get("name") == "A" and "dimensions" in var:
                dims = var["dimensions"]
                if len(dims) >= 2:
                    num_observations = dims[0]
                    num_states = dims[1]
            elif var.get("name") == "B" and "dimensions" in var:
                dims = var["dimensions"]
                if len(dims) >= 3:
                    num_actions = dims[2]

        connections, _wiring = _normalise_connections(gnn_spec.get("connections"))
        variable_dimensions = _variable_dimensions(gnn_spec)
        symmetry_metadata = build_matrix_permutation_metadata(
            gnn_spec, self.options.get("matrix_permutations")
        )
        symmetry_metadata_json = json.dumps(symmetry_metadata, indent=2)
        connections_json = json.dumps(connections, ensure_ascii=False, indent=4)
        variables_json = json.dumps(
            variable_dimensions, ensure_ascii=False, indent=4, sort_keys=True
        )
        module_docstring = f"""
DisCoPy Categorical Diagram Generation
Generated from GNN specification: {source_file}
Generated from GNN Model: {model_display_name}

This script creates categorical diagrams representing the Active Inference model
structure using DisCoPy's compositional framework.
"""

        # Generate the Python code
        code = f'''#!/usr/bin/env python3
{module_docstring!r}

import sys

try:
    import discopy
except ImportError:
    print("DisCoPy is not installed. Install the project render extras before running this script.")
    sys.exit(1)

from discopy import *
from discopy.monoidal import Ty, Box, Id
from discopy.drawing import Equation
import numpy as np
# Note: Visualizations are generated by the analysis step (16_analysis.py),
# not by this script. This script focuses on circuit construction and data export.
from pathlib import Path
import json
import logging
from datetime import datetime

# Model parameters extracted from GNN specification
MODEL_NAME = {model_display_name!r}
NUM_STATES = {num_states}
NUM_OBSERVATIONS = {num_observations}
NUM_ACTIONS = {num_actions}
CONNECTIONS = {connections_json}
VARIABLES = {variables_json}
MATRIX_PERMUTATION_METADATA = {symmetry_metadata_json}
MATRIX_PERMUTATION_APPLIED_TO_DIAGRAM = False

def validate_matrix_permutation_metadata(metadata):
    """Validate generated matrix permutation metadata before exporting diagrams."""
    if not isinstance(metadata, dict):
        raise ValueError("Matrix permutation metadata must be a dictionary")
    for matrix_name, record in metadata.items():
        if record.get("axis") != "rows":
            raise ValueError(f"Unsupported permutation axis for {{matrix_name}}")
        shape = record.get("shape")
        permutation = record.get("permutation")
        if not isinstance(shape, list) or not isinstance(permutation, list):
            raise ValueError(f"Invalid permutation metadata for {{matrix_name}}")
        if not shape:
            raise ValueError(f"Missing matrix shape for {{matrix_name}}")
        if len(permutation) != int(shape[0]):
            raise ValueError(f"Permutation length mismatch for {{matrix_name}}")
        if sorted(int(item) for item in permutation) != list(range(int(shape[0]))):
            raise ValueError(f"Permutation indices mismatch for {{matrix_name}}")

validate_matrix_permutation_metadata(MATRIX_PERMUTATION_METADATA)

_logger = logging.getLogger(__name__)

_RELATION_LABELS = {{
    ">": "generates",
    "-": "couples",
    "=": "defines",
}}

def split_endpoint(value):
    """Split a declared connection endpoint into individual wire names."""
    if isinstance(value, (list, tuple)):
        parts = [str(item) for item in value]
    else:
        text = str(value).strip()
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1]
        parts = text.split(",")
    return [part.strip() for part in parts if part.strip()]

def endpoint_type(names):
    """Build a DisCoPy type from a list of wire names."""
    return Ty(*names) if names else Ty()

def build_diagram():
    """Build one box per declared connection, wired by its endpoints.

    Sequential composition is used only when the previous codomain exactly
    matches the next domain; otherwise boxes are juxtaposed. No implicit
    copying, merging, or invented components are introduced.
    """
    boxes = []
    signatures = []
    for index, connection in enumerate(CONNECTIONS):
        sources = split_endpoint(connection.get("source", ""))
        targets = split_endpoint(connection.get("target", ""))
        if not sources or not targets:
            raise ValueError(f"connection {{index}} has an empty source or target")
        relation = str(connection.get("relation", ">")).strip()
        label = _RELATION_LABELS.get(relation)
        if label is None:
            raise ValueError(f"connection {{index}} has unsupported relation {{relation!r}}")
        name = f"{{'+'.join(sources)}}__{{label}}__{{'+'.join(targets)}}"
        box = Box(name, endpoint_type(sources), endpoint_type(targets))
        boxes.append(box)
        signatures.append({{
            "index": index,
            "name": name,
            "source": sources,
            "relation": relation,
            "target": targets,
            "dom": str(box.dom),
            "cod": str(box.cod),
        }})
    if not boxes:
        return Id(), [], []
    diagram = boxes[0]
    for box in boxes[1:]:
        if diagram.cod == box.dom:
            diagram = diagram >> box
        else:
            diagram = diagram @ box
    print(f"Built {{len(boxes)}} boxes from {{len(CONNECTIONS)}} declared connections")
    return diagram, boxes, signatures

def analyze_diagram(diagram, boxes, signatures):
    """Describe the constructed diagram and its declared wiring."""
    return {{
        "num_connections": len(CONNECTIONS),
        "num_boxes": len(boxes),
        "box_names": [box.name for box in boxes],
        "diagram_domain": str(diagram.dom),
        "diagram_codomain": str(diagram.cod),
        "connections": CONNECTIONS,
        "box_signatures": signatures,
    }}

def export_circuit_data(analysis, output_dir="discopy_diagrams"):
    """Export circuit data and analysis results."""
    print(f"Exporting circuit data to {{output_dir}}/...")
    Path(output_dir).mkdir(exist_ok=True)

    analysis_file = Path(output_dir) / "circuit_analysis.json"
    with open(analysis_file, "w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2)
    print(f"Exported analysis: {{analysis_file}}")

    circuit_info = {{
        "model_name": MODEL_NAME,
        "timestamp": datetime.now().isoformat(),
        "parameters": {{
            "num_states": NUM_STATES,
            "num_observations": NUM_OBSERVATIONS,
            "num_actions": NUM_ACTIONS,
        }},
        "connections": CONNECTIONS,
        "variables": VARIABLES,
        "components": analysis["box_names"],
        "box_signatures": analysis["box_signatures"],
        "analysis": analysis,
        "matrix_permutation_metadata": MATRIX_PERMUTATION_METADATA,
        "matrix_permutation_applied_to_diagram": MATRIX_PERMUTATION_APPLIED_TO_DIAGRAM,
    }}
    circuit_file = Path(output_dir) / "circuit_info.json"
    with open(circuit_file, "w", encoding="utf-8") as handle:
        json.dump(circuit_info, handle, indent=2)
    print(f"Exported circuit info: {{circuit_file}}")

def main():
    """Main execution function."""
    print("=" * 60)
    print("DisCoPy Categorical Diagrams - GNN Generated")
    print("Model:", MODEL_NAME)
    print("=" * 60)
    try:
        diagram, boxes, signatures = build_diagram()
        analysis_results = analyze_diagram(diagram, boxes, signatures)
        export_circuit_data(analysis_results)
        print("DisCoPy categorical diagram generation completed successfully!")
        return 0
    except Exception as error:
        print(f"Diagram generation failed: {{error}}")
        import traceback

        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
'''

        return code


def render_gnn_to_discopy(
    gnn_spec: Dict[str, Any],
    output_path: Path,
    options: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, List[str]]:
    """
    Render GNN specification to DisCoPy categorical diagram script.

    The supported generator requires the specification's declared
    ``Connections`` section. A specification without connections is refused
    rather than drawn as a generic Active Inference circuit.

    Args:
        gnn_spec: Parsed GNN specification dictionary
        output_path: Path for output DisCoPy script
        options: Optional rendering options

    Returns:
        Tuple of (success, message, warnings: List[str])
    """
    from gnn.render.execution_contracts import unsupported_contract

    refusal = unsupported_contract(gnn_spec, "discopy")
    if refusal:
        return False, refusal, []

    try:
        connections, _wiring = _normalise_connections(gnn_spec.get("connections"))
    except ValueError as error:
        return False, f"unsupported-discopy-connections: {error}", []
    if not connections:
        return (
            False,
            "unsupported-discopy-connections: the spec declares no Connections "
            "entries; DisCoPy renders the declared GNN connection graph and will "
            "not substitute a generic Active Inference circuit",
            [],
        )

    try:
        renderer = DisCoPyRenderer(options)

        # Generate simulation code directly from spec
        model_name = gnn_spec.get("model_name", "GNN_Model")
        discopy_code = renderer._generate_discopy_diagram_code(gnn_spec, model_name)

        # Write output file only after successful generation.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(output_path, discopy_code)

        message = f"Generated DisCoPy categorical diagram script: {output_path}"
        warnings: list[Any] = []

        # Check for potential issues
        initial_parameterization = gnn_spec.get(
            "initial_parameterization"
        ) or gnn_spec.get("initialparameterization")
        if not initial_parameterization:
            warnings.append("No initial parameterization found - using defaults")

        if not gnn_spec.get("model_parameters"):
            warnings.append("No model parameters found - using inferred dimensions")

        return True, message, warnings

    except Exception as e:
        return False, f"Error generating DisCoPy script: {e}", []
