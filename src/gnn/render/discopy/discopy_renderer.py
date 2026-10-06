#!/usr/bin/env python3
"""
DisCoPy Renderer

Renders GNN specifications to DisCoPy categorical diagram code for compositional models.
This renderer creates executable DisCoPy visualizations configured from parsed GNN POMDP specifications.

Features:
- GNN-to-DisCoPy parameter extraction
- Categorical diagram generation
- String diagram visualization
- Pipeline integration support

Author: GNN DisCoPy Integration
Date: 2024
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .symmetry import build_matrix_permutation_metadata


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

            # Generate DisCoPy categorical diagram code
            discopy_code = self._generate_discopy_diagram_code(
                gnn_spec, gnn_file_path.stem
            )

            # Write output file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(discopy_code)

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
                current_section = line[3:].strip()
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

        self.logger.info(
            f"Parsed {len(sections_found)} sections, {len(gnn_spec['model_parameters'])} model parameters"
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

        # Extract dimensions from model parameters
        model_params = gnn_spec.get("model_parameters", {})
        num_states = model_params.get("num_hidden_states", 3)
        num_observations = model_params.get("num_obs", 3)
        num_actions = model_params.get("num_actions", 3)

        # Try to extract from variables if available
        variables = gnn_spec.get("variables", [])
        variable_names: list[Any] = []
        for var in variables:
            var_name = var.get("name", "")
            if var_name:
                variable_names.append(var_name)

            if var.get("name") == "A" and "dimensions" in var:
                dims = var["dimensions"]
                if len(dims) >= 2:
                    num_observations = dims[0]
                    num_states = dims[1]
            elif var.get("name") == "B" and "dimensions" in var:
                dims = var["dimensions"]
                if len(dims) >= 3:
                    num_actions = dims[2]

        # Get initial parameterization if available
        gnn_spec.get("initial_parameterization", {})

        # Extract connections if available
        gnn_spec.get("connections", [])
        symmetry_metadata = build_matrix_permutation_metadata(
            gnn_spec, self.options.get("matrix_permutations")
        )
        symmetry_metadata_json = json.dumps(symmetry_metadata, indent=2)

        # Generate the Python code
        code = f'''#!/usr/bin/env python3
"""
DisCoPy Categorical Diagram Generation
Generated from GNN Model: {model_display_name}

This script creates categorical diagrams representing the Active Inference model
structure using DisCoPy's compositional framework.
"""

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
NUM_STATES = {num_states}
NUM_OBSERVATIONS = {num_observations}
NUM_ACTIONS = {num_actions}
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

# Define basic types for Active Inference
def define_types():
    """Define the basic types for our Active Inference model."""
    
    # Basic types
    S = Ty('S')  # Hidden states
    O = Ty('O')  # Observations  
    A = Ty('A')  # Actions
    P = Ty('P')  # Probabilities
    
    print("✓ Defined basic types: S (states), O (observations), A (actions), P (probabilities)")
    return S, O, A, P

# Define morphisms (boxes) for model components
def define_model_components(S, O, A, P):
    """Define the morphisms representing Active Inference model components."""
    
    components = {{}}
    
    # A matrix: Observation model P(o|s)
    components['A_matrix'] = Box('A', S, O @ P)

    # B matrix: Transition model P(s'|s,a)
    components['B_matrix'] = Box('B', S @ A, S @ P)

    # C vector: Preference vector over observations
    components['C_vector'] = Box('C', Ty(), O @ P)

    # D vector: Prior beliefs over states
    components['D_vector'] = Box('D', Ty(), S @ P)

    # E vector: Policy priors (if applicable)
    components['E_vector'] = Box('E', Ty(), A @ P)

    # Inference processes
    components['state_inference'] = Box('StateInf', O, S @ P)
    components['policy_inference'] = Box('PolicyInf', S @ P, A @ P)
    components['action_selection'] = Box('ActionSel', A @ P, A)
    
    print(f"✓ Defined {{len(components)}} model components as morphisms")
    return components

# Create Active Inference circuit
def create_active_inference_circuit(S, O, A, P, components):
    """Create the full Active Inference circuit as a categorical diagram."""
    
    print("\\n🏗️  Building Active Inference circuit...")
    
    # Extract components
    A_matrix = components['A_matrix']
    B_matrix = components['B_matrix'] 
    C_vector = components['C_vector']
    D_vector = components['D_vector']
    state_inf = components['state_inference']
    policy_inf = components['policy_inference']
    action_sel = components['action_selection']
    
    # Basic perception-action loop
    # Observation -> State Inference -> Policy Inference -> Action Selection
    perception_action_loop = (
        state_inf 
        >> policy_inf 
        >> action_sel
    )
    
    # Full generative model - compose priors alongside the inference loop (D-3 fix)
    # D_vector and C_vector enter as parallel tensor products
    try:
        # Compose: (D_vector @ C_vector) provides initial state and preference priors
        prior_block = D_vector @ C_vector
        generative_model = prior_block >> (state_inf @ Id(P)) >> policy_inf >> action_sel
        print("\u2713 Created generative model with composed priors")
    except Exception as e:
        # Recovery: type mismatch means priors cannot directly compose
        print(f"\u26a0\ufe0f  Prior composition type mismatch ({{e}}), using perception-action loop")
        generative_model = perception_action_loop
    
    print("\u2713 Created perception-action loop")
    
    return {{
        'perception_action_loop': perception_action_loop,
        'generative_model': generative_model,
        'components': components
    }}



# Analysis functions
def analyze_circuit_structure(circuit_dict):
    """Analyze the structure of the categorical diagrams."""
    
    print("\\n🔍 Analyzing circuit structure...")
    
    components = circuit_dict['components']
    loop = circuit_dict['perception_action_loop']
    model = circuit_dict['generative_model']
    
    # Basic structural analysis
    print(f"📊 Circuit Analysis:")
    print(f"  - Number of components: {{len(components)}}")
    print(f"  - Perception-action loop domain: {{loop.dom}}")
    print(f"  - Perception-action loop codomain: {{loop.cod}}")
    print(f"  - Generative model domain: {{model.dom}}")
    print(f"  - Generative model codomain: {{model.cod}}")
    
    # Component analysis
    print(f"\\n🧩 Component Details:")
    for name, component in components.items():
        print(f"  - {{name}}: {{component.dom}} → {{component.cod}}")
    
    # Compositional structure
    print(f"\\n🔗 Compositional Structure:")
    print(f"  - Loop composition depth: {{len(str(loop).split('>>'))}}")
    print(f"  - Model composition depth: {{len(str(model).split('>>'))}}")
    
    return {{
        'num_components': len(components),
        'loop_domain': str(loop.dom),
        'loop_codomain': str(loop.cod),
        'model_domain': str(model.dom),
        'model_codomain': str(model.cod)
    }}

# Export functions
def export_circuit_data(circuit_dict, analysis_results, output_dir="discopy_diagrams"):
    """Export circuit data and analysis results."""
    
    print(f"\\n💾 Exporting circuit data to {{output_dir}}/...")
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # Export analysis results
    analysis_file = Path(output_dir) / "circuit_analysis.json"
    with open(analysis_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"✓ Exported analysis: {{analysis_file}}")
    
    # Export circuit information
    circuit_info = {{
        'model_name': '{model_display_name}',
        'timestamp': datetime.now().isoformat(),
        'parameters': {{
            'num_states': NUM_STATES,
            'num_observations': NUM_OBSERVATIONS, 
            'num_actions': NUM_ACTIONS
        }},
        'components': list(circuit_dict['components'].keys()),
        'matrix_permutation_metadata': MATRIX_PERMUTATION_METADATA,
        'matrix_permutation_applied_to_diagram': MATRIX_PERMUTATION_APPLIED_TO_DIAGRAM,
        'analysis': analysis_results
    }}
    
    circuit_file = Path(output_dir) / "circuit_info.json"
    with open(circuit_file, 'w') as f:
        json.dump(circuit_info, f, indent=2)
    
    print(f"✓ Exported circuit info: {{circuit_file}}")

# Main execution
def main():
    """Main execution function."""
    
    print("="*60)
    print("DisCoPy Categorical Diagrams - GNN Generated")
    print(f"Model: {model_display_name}")
    print("="*60)
    
    try:
        # Define types
        S, O, A, P = define_types()
        
        # Define model components
        components = define_model_components(S, O, A, P)
        
        # Create Active Inference circuit
        circuit_dict = create_active_inference_circuit(S, O, A, P, components)
        
        # Analyze circuit structure
        analysis_results = analyze_circuit_structure(circuit_dict)
        
        # Export data
        export_circuit_data(circuit_dict, analysis_results)
        
        print("\\n✅ DisCoPy categorical diagram generation completed successfully!")
        print("📊 Circuit analysis and data export successful")
        
        print("\\n🎉 Categorical representation of Active Inference model complete!")
        
        return 0
        
    except Exception as e:
        print(f"❌ Diagram generation failed: {{e}}")
        print("🔍 Stack trace:")
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
        renderer = DisCoPyRenderer(options)

        # Generate simulation code directly from spec
        model_name = gnn_spec.get("model_name", "GNN_Model")
        discopy_code = renderer._generate_discopy_diagram_code(gnn_spec, model_name)

        # Write output file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(discopy_code)

        message = f"Generated DisCoPy categorical diagram script: {output_path}"
        warnings: list[Any] = []

        # Check for potential issues
        if not gnn_spec.get("initial_parameterization"):
            warnings.append("No initial parameterization found - using defaults")

        if not gnn_spec.get("model_parameters"):
            warnings.append("No model parameters found - using inferred dimensions")

        if not gnn_spec.get("connections"):
            warnings.append(
                "No explicit connections found - using default Active Inference structure"
            )

        return True, message, warnings

    except Exception as e:
        return False, f"Error generating DisCoPy script: {e}", []
