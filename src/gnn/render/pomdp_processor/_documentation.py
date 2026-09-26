#!/usr/bin/env python3
"""Documentation mixin for the POMDP render processor."""

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import numpy as np

from ._support import _POMDPProcessorSupportMixin

if TYPE_CHECKING:
    from gnn.extract.pomdp_extractor import POMDPStateSpace


class _DocumentationMixin(_POMDPProcessorSupportMixin):
    def _create_framework_documentation(
        self,
        framework: str,
        pomdp_space: "POMDPStateSpace",
        output_dir: Path,
        render_result: Dict[str, Any],
    ) -> None:
        """
        Create framework-specific documentation.

        Args:
            framework: Framework name
            pomdp_space: POMDP state space
            output_dir: Output directory
            render_result: Rendering result
        """
        try:
            doc_file = output_dir / "README.md"

            # Get model annotation safely
            model_annotation = getattr(pomdp_space, "model_annotation", None) or "N/A"

            def _shape_text(value: Any) -> str | None:
                """Handle shape text for internal callers."""
                if value is None:
                    return None
                try:
                    array = np.asarray(value)
                    if array.size == 0:
                        return None
                    return "×".join(str(dim) for dim in array.shape)
                except (TypeError, ValueError):
                    if isinstance(value, (list, tuple)) and value:
                        nested_shapes: list[str] = []
                        for item in value:
                            nested_shape = _shape_text(item)
                            if nested_shape is not None:
                                nested_shapes.append(nested_shape)
                        if nested_shapes:
                            unique_shapes = sorted(set(nested_shapes))
                            return f"{len(value)} blocks ({', '.join(unique_shapes)})"
                    return None

            def _vector_length(value: Any) -> int | None:
                """Handle vector length for internal callers."""
                if value is None:
                    return None
                array = np.asarray(value)
                if array.size == 0:
                    return None
                return int(array.size)

            doc_content = f"""# {framework.upper()} Rendering Results

Generated from GNN POMDP Model: **{pomdp_space.model_name}**

## Model Information

- **Model Name**: {pomdp_space.model_name}
- **Model Description**: {model_annotation}
- **Generation Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## POMDP Dimensions

- **Number of States**: {pomdp_space.num_states}
- **Number of Observations**: {pomdp_space.num_observations}
- **Number of Actions**: {pomdp_space.num_actions}

## Active Inference Matrices

### Available Matrices/Vectors:
"""

            # Safely check for matrices/vectors
            A_matrix = getattr(pomdp_space, "A_matrix", None)
            A_shape = _shape_text(A_matrix)
            if A_shape is not None:
                doc_content += f"- **A Matrix (Likelihood)**: {A_shape} - Maps hidden states to observations\n"

            B_matrix = getattr(pomdp_space, "B_matrix", None)
            B_shape = _shape_text(B_matrix)
            if B_shape is not None:
                doc_content += f"- **B Matrix (Transition)**: {B_shape} - State transitions given actions\n"

            C_vector = getattr(pomdp_space, "C_vector", None)
            C_length = _vector_length(C_vector)
            if C_length is not None:
                doc_content += f"- **C Vector (Preferences)**: Length {C_length} - Preferences over observations\n"

            D_vector = getattr(pomdp_space, "D_vector", None)
            D_length = _vector_length(D_vector)
            if D_length is not None:
                doc_content += f"- **D Vector (Prior)**: Length {D_length} - Prior beliefs over states\n"

            E_vector = getattr(pomdp_space, "E_vector", None)
            E_length = _vector_length(E_vector)
            if E_length is not None:
                doc_content += (
                    f"- **E Vector (Habits)**: Length {E_length} - Policy priors\n"
                )

            doc_content += """

## Generated Files

"""

            for artifact in render_result.get("artifacts", []):
                artifact_path = Path(artifact)
                doc_content += (
                    f"- `{artifact_path.name}` - {framework} simulation script\n"
                )

            if render_result.get("warnings"):
                doc_content += """

## Warnings

"""
                for warning in render_result["warnings"]:
                    doc_content += f"- ⚠️ {warning}\n"

            doc_content += f"""

## Usage

Refer to the main {framework} documentation for information on how to run the generated simulation scripts.

## Framework-Specific Information

- **Framework**: {framework}
- **File Extension**: {self.framework_configs[framework]["file_extension"]}
- **Multi-Modality Support**: {"✅" if self.framework_configs[framework]["supports_multi_modality"] else "❌"}
- **Multi-Factor Support**: {"✅" if self.framework_configs[framework]["supports_multi_factor"] else "❌"}
"""

            with open(doc_file, "w") as f:
                f.write(doc_content)

            self.logger.info(f"Created documentation: {doc_file}")

        except Exception as e:
            self.logger.warning(f"Failed to create documentation for {framework}: {e}")
