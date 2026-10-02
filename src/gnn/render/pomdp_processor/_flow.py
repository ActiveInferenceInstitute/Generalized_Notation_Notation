#!/usr/bin/env python3
"""Process-flow mixin for the POMDP render processor."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from gnn.render.framework_registry import get_pomdp_framework_configs
from gnn.utils.config_io.code_metrics import count_code_metrics

from ._support import _POMDPProcessorSupportMixin

if TYPE_CHECKING:
    from gnn.extract.pomdp_extractor import POMDPStateSpace


class _ProcessFlowMixin(_POMDPProcessorSupportMixin):
    def __init__(self, base_output_dir: Path) -> None:
        """
        Initialize POMDP render processor.

        Args:
            base_output_dir: Base output directory for all renderers
        """
        self.base_output_dir = Path(base_output_dir)
        self.logger = logging.getLogger(__name__)

        self.framework_configs = get_pomdp_framework_configs()

    def process_pomdp_for_all_frameworks(
        self,
        pomdp_space: "POMDPStateSpace",
        gnn_file_path: Optional[Path] = None,
        frameworks: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Process POMDP state space for all or specified frameworks.

        Args:
            pomdp_space: Extracted POMDP state space
            gnn_file_path: Original GNN file path (for reference)
            frameworks: List of frameworks to render for (default: all)
            **kwargs: Additional processing options

        Returns:
            Dictionary with processing results for each framework
        """
        if frameworks is None:
            frameworks = [
                name
                for name, spec in self.framework_configs.items()
                if not spec.get("experimental", False)
            ]

        results: dict[Any, Any] = {}
        overall_success = True

        # Create base output directory
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

        # Create processing summary
        processing_summary: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "source_file": str(gnn_file_path) if gnn_file_path else None,
            "model_name": pomdp_space.model_name,
            "pomdp_dimensions": {
                "num_states": pomdp_space.num_states,
                "num_observations": pomdp_space.num_observations,
                "num_actions": pomdp_space.num_actions,
            },
            "frameworks_requested": frameworks,
            "frameworks_processed": [],
            "frameworks_failed": [],
            "frameworks_unsupported": [],
            "model_kind": getattr(pomdp_space, "model_kind", "discrete"),
        }

        self.logger.info(
            f"Processing POMDP '{pomdp_space.model_name}' for frameworks: {frameworks}"
        )

        for framework in frameworks:
            try:
                self.logger.info(f"Processing framework: {framework}")
                framework_result = self._process_single_framework(
                    pomdp_space, framework, gnn_file_path, **kwargs
                )

                results[framework] = framework_result

                if framework_result["success"]:
                    processing_summary["frameworks_processed"].append(framework)
                    self.logger.info(f"✅ {framework}: {framework_result['message']}")
                elif framework_result.get("unsupported"):
                    processing_summary["frameworks_unsupported"].append(framework)
                    self.logger.info(
                        f"⏭️ {framework}: unsupported — {framework_result['message']}"
                    )
                else:
                    processing_summary["frameworks_failed"].append(framework)
                    self.logger.error(f"❌ {framework}: {framework_result['message']}")

            except Exception as e:
                error_msg = f"Unexpected error processing {framework}: {e}"
                self.logger.error(error_msg)
                results[framework] = {
                    "success": False,
                    "status": "failed",
                    "error_type": type(e).__name__,
                    "message": error_msg,
                    "output_files": [],
                    "warnings": [],
                }
                processing_summary["frameworks_failed"].append(framework)

        # Frameworks that cannot represent this model kind (e.g. a categorical
        # backend given a continuous-state model) are reported as unsupported
        # and excluded from the success denominator: neither rendered nor failed.
        unsupported_count = len(processing_summary["frameworks_unsupported"])
        total_frameworks = len(frameworks) - unsupported_count
        successful_frameworks = len(processing_summary["frameworks_processed"])
        success_rate = (
            successful_frameworks / total_frameworks if total_frameworks > 0 else 0
        )
        overall_success = successful_frameworks == total_frameworks

        if not overall_success:
            self.logger.warning(
                "Framework rendering incomplete: %d/%d succeeded (%.1f%%)",
                successful_frameworks,
                total_frameworks,
                success_rate * 100,
            )

        # Save processing summary
        summary_file = self.base_output_dir / "processing_summary.json"
        with open(summary_file, "w") as f:
            json.dump(processing_summary, f, indent=2)

        return {
            "overall_success": overall_success,
            "framework_results": results,
            "summary_file": str(summary_file),
            "output_directory": str(self.base_output_dir),
        }

    def _process_single_framework(
        self,
        pomdp_space: "POMDPStateSpace",
        framework: str,
        gnn_file_path: Optional[Path] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Process POMDP state space for a single framework.

        Args:
            pomdp_space: POMDP state space data
            framework: Target framework name
            gnn_file_path: Original GNN file path
            **kwargs: Additional options

        Returns:
            Processing result dictionary
        """
        if framework not in self.framework_configs:
            return {
                "success": False,
                "message": f"Unknown framework: {framework}",
                "output_files": [],
                "warnings": [],
            }

        config = self.framework_configs[framework]

        # Validate POMDP compatibility with framework
        validation_result = self._validate_pomdp_framework_compatibility(
            pomdp_space, framework
        )
        if not validation_result["compatible"]:
            unsupported = bool(validation_result.get("unsupported"))
            return {
                "success": False,
                "unsupported": unsupported,
                "status": "unsupported" if unsupported else "failed",
                "message": (
                    validation_result["reason"]
                    if unsupported
                    else f"POMDP not compatible with {framework}: {validation_result['reason']}"
                ),
                "output_files": [],
                "warnings": validation_result.get("warnings", []),
            }

        # Create framework-specific output directory
        framework_output_dir = self.base_output_dir / str(config["output_subdir"])
        framework_output_dir.mkdir(parents=True, exist_ok=True)

        # Convert POMDP to GNN spec format expected by renderers
        spec_options = {
            **kwargs,
            "native_agents": framework in {"rxinfer", "activeinference_jl"},
            "preserve_discrete_structure": framework == "thrml",
        }
        gnn_spec = self._pomdp_to_gnn_spec(pomdp_space, **spec_options)

        # Get framework-specific renderer
        try:
            renderer_result = self._call_framework_renderer(
                framework, gnn_spec, framework_output_dir, **kwargs
            )

            if renderer_result["success"]:
                # Create framework-specific documentation
                self._create_framework_documentation(
                    framework, pomdp_space, framework_output_dir, renderer_result
                )

                # Calculate code metrics for generated files
                code_metrics: dict[Any, Any] = {}
                for output_file in renderer_result.get("artifacts", []):
                    file_path = Path(output_file)
                    if file_path.exists():
                        code_metrics = count_code_metrics(file_path)
                        break  # Use first file's metrics

                return {
                    "success": True,
                    "message": renderer_result["message"],
                    "output_files": renderer_result.get("artifacts", []),
                    "output_directory": str(framework_output_dir),
                    "warnings": validation_result.get("warnings", []),
                    "code_metrics": code_metrics,
                }
            else:
                return {
                    "success": False,
                    "unsupported": bool(renderer_result.get("unsupported")),
                    "status": renderer_result.get("status", "failed"),
                    "message": renderer_result["message"],
                    "output_files": [],
                    "warnings": validation_result.get("warnings", []),
                }

        except Exception as e:
            return {
                "success": False,
                "message": f"{framework} renderer failed: {e}",
                "output_files": [],
                "warnings": validation_result.get("warnings", []),
            }
