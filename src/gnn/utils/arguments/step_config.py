"""
Step configuration for the GNN processing pipeline.

Provides StepConfiguration, the declarative configuration for individual
pipeline steps (criticality, required/optional arguments, defaults,
descriptions) plus step config lookup and argument validation.
"""

import argparse
import sys
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, List, Optional


# Add enhanced validation and step configuration
class StepConfiguration:
    """Configuration for individual pipeline steps."""

    # Define step-specific requirements and defaults
    STEP_CONFIGS = MappingProxyType(
        {
            "0_template": {
                "critical": True,
                "required_args": ["target_dir", "output_dir"],
                "optional_args": ["recursive", "verbose", "simulate_error"],
                "defaults": {
                    "recursive": True,
                    "verbose": False,
                    "simulate_error": False,
                },
                "description": "Standardized pipeline step template",
            },
            "1_setup": {
                "critical": True,
                "required_args": ["target_dir", "output_dir"],
                "optional_args": [
                    "recursive",
                    "verbose",
                    "recreate_venv",
                    "dev",
                    "install_all_extras",
                    "setup_core_only",
                    "install_optional",
                    "optional_groups",
                ],
                "defaults": {
                    "verbose": False,
                    "recreate_venv": False,
                    "dev": False,
                    "install_all_extras": False,
                    "setup_core_only": False,
                    "install_optional": False,
                    "optional_groups": None,
                },
                "description": "Project Setup & Environment Configuration",
            },
            "2_tests": {
                "critical": False,
                "required_args": [],
                "optional_args": [
                    "target_dir",
                    "output_dir",
                    "verbose",
                    "fast_only",
                    "include_performance",
                    "comprehensive",
                ],
                "defaults": {
                    "verbose": False,
                    "fast_only": True,
                    "include_performance": False,
                    "comprehensive": False,
                },
                "description": "Test Execution & Validation",
            },
            "3_gnn": {
                "critical": True,
                "required_args": ["target_dir", "output_dir"],
                "optional_args": [
                    "recursive",
                    "verbose",
                    "enable_round_trip",
                    "enable_cross_format",
                    "serialize_preset",
                ],
                "defaults": {
                    "recursive": True,
                    "verbose": False,
                    "enable_round_trip": True,
                    "enable_cross_format": True,
                    "serialize_preset": "full",
                },
                "description": "GNN Discovery & Basic Parse",
            },
            "4_model_registry": {
                "critical": False,
                "required_args": ["target_dir", "output_dir"],
                "optional_args": [
                    "recursive",
                    "verbose",
                    "registry_path",
                    "query_ontology",
                ],
                "defaults": {
                    "recursive": True,
                    "verbose": False,
                    "registry_path": None,
                    "query_ontology": None,
                },
                "description": "Model Registry & Versioning",
            },
            "5_type_checker": {
                "critical": True,
                "required_args": ["target_dir", "output_dir"],
                "optional_args": [
                    "recursive",
                    "verbose",
                    "strict",
                    "estimate_resources",
                ],
                "defaults": {
                    "recursive": True,
                    "verbose": False,
                    "strict": False,
                    "estimate_resources": True,
                },
                "description": "Type Checking & Resource Estimation",
            },
            "6_validation": {
                "critical": True,
                "required_args": ["target_dir", "output_dir"],
                "optional_args": [
                    "recursive",
                    "verbose",
                    "strict",
                    "profile",
                    "transpose_b",
                ],
                "defaults": {
                    "recursive": True,
                    "verbose": False,
                    "strict": False,
                    "profile": False,
                    "transpose_b": False,
                },
                "description": "Validation & Quality Assurance",
            },
            "7_export": {
                "critical": True,
                "required_args": ["target_dir", "output_dir"],
                "optional_args": [
                    "recursive",
                    "verbose",
                    "geo_step_seconds",
                    "geo_state_ids",
                    "geo_space_kind",
                    "geo_derive_metadata",
                    "formats",
                    "geo_infer_options_file",
                ],
                "defaults": {"recursive": True, "verbose": False},
                "description": "GNN Export & Format Conversion",
            },
            "8_visualization": {
                "critical": False,
                "required_args": ["target_dir", "output_dir"],
                "optional_args": ["recursive", "verbose"],
                "defaults": {"recursive": True, "verbose": False},
                "description": "Basic Visualization Generation",
            },
            "9_advanced_viz": {
                "critical": False,
                "required_args": ["target_dir", "output_dir"],
                "optional_args": [
                    "recursive",
                    "verbose",
                    "viz_type",
                    "interactive",
                    "export_formats",
                ],
                "defaults": {
                    "recursive": True,
                    "verbose": False,
                    "viz_type": "all",
                    "interactive": True,
                    "export_formats": ["html", "json"],
                },
                "description": "Advanced Visualization & Exploration",
            },
            "10_ontology": {
                "critical": False,
                "required_args": ["target_dir", "output_dir"],
                "optional_args": ["recursive", "verbose", "ontology_terms_file"],
                "defaults": {"recursive": True, "verbose": False},
                "description": "Ontology Processing & Validation",
            },
            "11_render": {
                "critical": True,
                "required_args": ["target_dir", "output_dir"],
                "optional_args": [
                    "recursive",
                    "verbose",
                    "timesteps",
                    "simulation_params",
                    "frameworks",
                    "strict_framework_success",
                ],
                "defaults": {"recursive": True, "verbose": False},
                "description": "Simulator Code Generation",
            },
            "12_execute": {
                "critical": False,
                "required_args": ["target_dir", "output_dir"],
                "optional_args": [
                    "recursive",
                    "verbose",
                    "frameworks",
                    "timeout",
                    "render_output_dir",
                    "distributed",
                    "execution_workers",
                    "backend",
                    "execution_benchmark_repeats",
                    "execution_summary_detail",
                ],
                "defaults": {
                    "recursive": True,
                    "verbose": False,
                    "execution_benchmark_repeats": 1,
                    "execution_summary_detail": False,
                },
                "description": "Simulator Execution",
            },
            "13_llm": {
                "critical": False,
                "required_args": ["target_dir", "output_dir"],
                "optional_args": ["recursive", "verbose", "llm_tasks", "llm_timeout"],
                "defaults": {
                    "recursive": True,
                    "verbose": False,
                    "llm_tasks": "all",
                    "llm_timeout": None,
                },
                "description": "LLM Analysis & Processing",
            },
            "14_ml_integration": {
                "critical": False,
                "required_args": ["target_dir", "output_dir"],
                "optional_args": ["recursive", "verbose"],
                "defaults": {"recursive": True, "verbose": False},
                "description": "Machine Learning Integration",
            },
            "15_audio": {
                "critical": False,
                "required_args": ["target_dir", "output_dir"],
                "optional_args": [
                    "recursive",
                    "verbose",
                    "duration",
                    "audio_backend",
                    "sonification",
                    "full_analysis",
                ],
                "defaults": {
                    "recursive": True,
                    "verbose": False,
                    "duration": 30.0,
                    "audio_backend": "auto",
                    "sonification": True,
                    "full_analysis": False,
                },
                "description": "Audio Generation & Processing",
            },
            "16_analysis": {
                "critical": False,
                "required_args": ["target_dir", "output_dir"],
                "optional_args": [
                    "recursive",
                    "verbose",
                    "advanced_stats",
                    "generate_animations",
                ],
                "defaults": {
                    "recursive": True,
                    "verbose": False,
                    "advanced_stats": False,
                    "generate_animations": True,
                },
                "description": "Advanced Analysis & Reporting",
            },
            "17_integration": {
                "critical": False,
                "required_args": ["target_dir", "output_dir"],
                "optional_args": ["recursive", "verbose"],
                "defaults": {"recursive": True, "verbose": False},
                "description": "API Gateway & Plugin System",
            },
            "18_security": {
                "critical": False,
                "required_args": ["target_dir", "output_dir"],
                "optional_args": ["recursive", "verbose"],
                "defaults": {"recursive": True, "verbose": False},
                "description": "Security & Compliance Features",
            },
            "19_research": {
                "critical": False,
                "required_args": ["target_dir", "output_dir"],
                "optional_args": ["recursive", "verbose"],
                "defaults": {"recursive": True, "verbose": False},
                "description": "Research Workflow Enhancement",
            },
            "20_website": {
                "critical": False,
                "required_args": ["target_dir", "output_dir"],
                "optional_args": ["recursive", "verbose"],
                "defaults": {"recursive": True, "verbose": False},
                "description": "HTML Website Generation",
            },
            "21_mcp": {
                "critical": False,
                "required_args": ["target_dir", "output_dir"],
                "optional_args": [
                    "recursive",
                    "verbose",
                    "performance_mode",
                    "mcp_strict_validation",
                    "mcp_cache_ttl",
                    "mcp_per_module_timeout",
                    "mcp_overall_timeout",
                    "mcp_modules_allowlist",
                ],
                "defaults": {
                    "recursive": True,
                    "verbose": False,
                    "performance_mode": "low",
                },
                "description": "Model Context Protocol Processing",
            },
            "22_gui": {
                "critical": False,
                "required_args": ["target_dir", "output_dir"],
                "optional_args": [
                    "recursive",
                    "verbose",
                    "headless",
                    "interactive",
                    "gui_types",
                    "open_browser",
                    "launch_editor",
                ],
                "defaults": {
                    "recursive": True,
                    "verbose": False,
                    "headless": False,
                    "interactive": False,
                    "gui_types": "gui_1,gui_2",
                    "open_browser": False,
                    "launch_editor": False,
                },
                "description": "Interactive GUI for Constructing/Editing GNN Models",
            },
            "23_report": {
                "critical": False,
                "required_args": ["target_dir", "output_dir"],
                "optional_args": ["recursive", "verbose"],
                "defaults": {"recursive": True, "verbose": False},
                "description": "Comprehensive Analysis Report Generation",
            },
            "24_intelligent_analysis": {
                "critical": False,
                "required_args": ["target_dir", "output_dir"],
                "optional_args": [
                    "verbose",
                    "analysis_model",
                    "skip_llm",
                    "bottleneck_threshold",
                ],
                "defaults": {
                    "verbose": False,
                    "analysis_model": None,
                    "skip_llm": False,
                    "bottleneck_threshold": 60.0,
                },
                "description": "AI-powered Pipeline Analysis and Optimization Recommendations",
            },
        }
    )

    @classmethod
    def get_step_config(cls, step_name: str) -> Dict[str, Any]:
        """Get configuration for a specific step."""
        return cls.STEP_CONFIGS.get(step_name, {})

    @classmethod
    def validate_step_args(
        cls,
        step_name: str,
        args: argparse.Namespace,
        project_root: Optional[Path] = None,
    ) -> List[str]:
        """Validate arguments for a specific step.

        Args:
            step_name: Step identifier (e.g. ``"16_analysis"``).
            args: Parsed argument namespace to validate. Missing relative
                input paths are repaired in place when a project-root-relative
                location exists (historical behavior).
            project_root: Explicit project root used to resolve relative
                input paths that do not exist as given. When omitted, the
                caller's frame is inspected for a file directly inside a
                ``src/`` directory and its grandparent is used (existing
                behavior, kept for existing callers).

        Returns:
            Human-readable validation errors; empty when the arguments are valid.
        """
        errors: list[str] = []
        config = cls.get_step_config(step_name)

        if not config:
            errors.append(f"Unknown step: {step_name}")
            return errors

        # Check required arguments
        for req_arg in config.get("required_args", []):
            if not hasattr(args, req_arg) or getattr(args, req_arg) is None:
                errors.append(
                    f"Missing required argument for {step_name}: --{req_arg.replace('_', '-')}"
                )

        # Validate path arguments exist if they should
        path_args: list[str] = ["target_dir", "output_dir", "ontology_terms_file"]
        caller_file: Optional[Path] = None
        if project_root is None and hasattr(sys, "_getframe"):
            try:
                caller_file = Path(sys._getframe(1).f_code.co_filename)
            except (ValueError, AttributeError):
                caller_file = None
        for arg_name in path_args:
            if hasattr(args, arg_name):
                arg_value = getattr(args, arg_name)
                if arg_value and isinstance(arg_value, Path):
                    # Only validate existence for input paths, not output paths
                    if arg_name in ["target_dir", "ontology_terms_file"]:
                        if not arg_value.exists():
                            resolved = cls._resolve_missing_input_path(
                                arg_value,
                                project_root=project_root,
                                caller_file=caller_file,
                            )
                            if resolved is not None:
                                setattr(args, arg_name, resolved)
                            else:
                                errors.append(
                                    f"Path does not exist for {step_name}: {arg_value}"
                                )

        return errors

    @staticmethod
    def _resolve_missing_input_path(
        arg_value: Path,
        *,
        project_root: Optional[Path],
        caller_file: Optional[Path],
    ) -> Optional[Path]:
        """Resolve a nonexistent relative input path against a project root.

        Returns the corrected path when it exists, else ``None``. With an
        explicit *project_root* the candidate is ``project_root / name``;
        caller-frame resolution instead requires *caller_file* to live directly inside a
        ``src/`` directory and tries its grandparent.
        """
        if project_root is not None:
            candidate = project_root / arg_value.name
            return candidate if candidate.exists() else None
        if caller_file is None:
            return None
        if not (caller_file.name.endswith(".py") and caller_file.parent.name == "src"):
            return None
        project_path = caller_file.parent.parent / arg_value.name
        return project_path if project_path.exists() else None
