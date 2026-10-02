"""Immutable argument definitions and step exposure catalog.

Parser behavior lives in arg_parsing; this module contains only declarative
metadata. ArgumentParser retains its public class attributes as aliases.
"""

import argparse
from pathlib import Path
from types import MappingProxyType

from gnn.frameworks import ALL_FRAMEWORKS

from .arg_definitions import ArgumentDefinition

ARGUMENT_DEFINITIONS = MappingProxyType(
    {
        "formats": ArgumentDefinition(
            flag="--formats",
            nargs="+",
            default=None,
            help_text="Step 7 export formats (default: five pipeline formats)",
        ),
        "geo_infer_options_file": ArgumentDefinition(
            flag="--geo-infer-options-file",
            arg_type=Path,
            default=None,
            help_text="Step 7 JSON mapping source filenames to GEO metadata",
        ),
        "target_dir": ArgumentDefinition(
            flag="--target-dir",
            arg_type=Path,
            default=Path("input/gnn_files"),
            help_text="Target directory for GNN files",
        ),
        "output_dir": ArgumentDefinition(
            flag="--output-dir",
            arg_type=Path,
            default=Path("output"),
            help_text="Directory to save outputs",
        ),
        "recursive": ArgumentDefinition(
            flag="--recursive",
            action=argparse.BooleanOptionalAction,
            default=True,
            help_text="Recursively process directories",
        ),
        "verbose": ArgumentDefinition(
            flag="--verbose",
            action=argparse.BooleanOptionalAction,
            default=False,
            help_text="Enable verbose output",
        ),
        "log_format": ArgumentDefinition(
            flag="--log-format",
            arg_type=str,
            choices=["human", "json"],
            default="human",
            help_text="Output format for pipeline logs",
        ),
        "enable_round_trip": ArgumentDefinition(
            flag="--enable-round-trip",
            action="store_true",
            help_text="Enable comprehensive round-trip testing across all 21 formats",
        ),
        "enable_cross_format": ArgumentDefinition(
            flag="--enable-cross-format",
            action="store_true",
            help_text="Enable cross-format consistency validation",
        ),
        "skip_steps": ArgumentDefinition(
            flag="--skip-steps",
            default=None,
            help_text="Comma-separated list of steps to skip",
        ),
        "only_steps": ArgumentDefinition(
            flag="--only-steps",
            default=None,
            help_text="Comma-separated list of steps to run exclusively",
        ),
        "parallel": ArgumentDefinition(
            flag="--parallel",
            action="store_true",
            default=False,
            help_text="Execute independent steps within topological tiers in parallel",
        ),
        "autonomous": ArgumentDefinition(
            flag="--autonomous",
            action="store_true",
            default=False,
            help_text=(
                "Write bounded autonomous proposal artifacts under output/ "
                "without editing source files"
            ),
        ),
        "consolidated_steps": ArgumentDefinition(
            flag="--consolidated-steps",
            action="store_true",
            help_text=(
                "Opt-in: run the whitelisted discovery/schema steps "
                "(0, 3, 5) in-process via the shared step executor "
                "instead of one subprocess per step "
                "(docs/decisions/0001-consolidated-pipeline-execution.md)"
            ),
        ),
        "skip_llm": ArgumentDefinition(
            flag="--skip-llm",
            action="store_true",
            help_text="Skip LLM-powered processing where supported",
        ),
        "strict": ArgumentDefinition(
            flag="--strict", action="store_true", help_text="Enable strict mode"
        ),
        "profile": ArgumentDefinition(
            flag="--profile",
            action="store_true",
            help_text="Enable performance profiling",
        ),
        "transpose_b": ArgumentDefinition(
            flag="--transpose-b",
            action="store_true",
            help_text=(
                "Validation (step 6): opt-in canonical B-tensor "
                "transposition — textbook (row-stochastic) transition "
                "tensors are transposed in memory and recorded in the "
                "validation receipt (default: warnings only)"
            ),
        ),
        "simulate_error": ArgumentDefinition(
            flag="--simulate-error",
            action="store_true",
            use_suppress=True,
            help_text="Simulate an error for testing",
        ),
        "registry_path": ArgumentDefinition(
            flag="--registry-path",
            arg_type=Path,
            default=None,
            help_text="Path to model registry file",
        ),
        "query_ontology": ArgumentDefinition(
            flag="--query-ontology",
            arg_type=str,
            default=None,
            help_text="Filter registered models by ontology concept substring",
        ),
        "estimate_resources": ArgumentDefinition(
            flag="--estimate-resources",
            action=argparse.BooleanOptionalAction,
            default=False,
            help_text="Estimate computational resources",
        ),
        "ontology_terms_file": ArgumentDefinition(
            flag="--ontology-terms-file",
            arg_type=Path,
            help_text="Path to ontology terms file",
        ),
        "pipeline_summary_file": ArgumentDefinition(
            flag="--pipeline-summary-file",
            arg_type=Path,
            help_text="Path to save pipeline summary",
        ),
        "llm_tasks": ArgumentDefinition(
            flag="--llm-tasks", help_text="Comma-separated list of LLM tasks"
        ),
        "llm_timeout": ArgumentDefinition(
            flag="--llm-timeout",
            arg_type=int,
            help_text="Timeout for LLM processing in seconds",
        ),
        "performance_mode": ArgumentDefinition(
            flag="--performance-mode",
            arg_type=str,
            default="low",
            help_text="Performance mode for applicable steps (low, medium, high)",
            choices=["low", "medium", "high"],
        ),
        "mcp_strict_validation": ArgumentDefinition(
            flag="--mcp-strict-validation",
            action="store_true",
            help_text="MCP (step 21): enforce JSON-schema validation on every tool call",
        ),
        "mcp_cache_ttl": ArgumentDefinition(
            flag="--mcp-cache-ttl",
            arg_type=float,
            help_text="MCP (step 21): result-cache TTL in seconds (default 300)",
        ),
        "mcp_per_module_timeout": ArgumentDefinition(
            flag="--mcp-per-module-timeout",
            arg_type=float,
            help_text="MCP (step 21): max seconds to wait per module during discovery (default 30)",
        ),
        "mcp_overall_timeout": ArgumentDefinition(
            flag="--mcp-overall-timeout",
            arg_type=float,
            help_text="MCP (step 21): overall wall-clock budget for parallel discovery (default 120)",
        ),
        "mcp_modules_allowlist": ArgumentDefinition(
            flag="--mcp-modules-allowlist",
            arg_type=str,
            help_text="MCP (step 21): comma-separated module names to restrict discovery to",
        ),
        "frameworks": ArgumentDefinition(
            flag="--frameworks",
            arg_type=str,
            default="all",
            help_text=(
                "Frameworks to execute/render (all, lite, or comma-separated list: "
                + ", ".join(ALL_FRAMEWORKS)
                + ")"
            ),
        ),
        "strict_framework_success": ArgumentDefinition(
            flag="--strict-framework-success",
            action="store_true",
            help_text="Render step: fail if any requested framework render fails",
        ),
        "render_output_dir": ArgumentDefinition(
            flag="--render-output-dir",
            arg_type=Path,
            default=None,
            help_text="Explicit path to 11_render_output directory (avoids filesystem heuristics)",
        ),
        "distributed": ArgumentDefinition(
            flag="--distributed",
            action="store_true",
            default=False,
            help_text="Enable distributed execution for step 12 (if supported)",
        ),
        "execution_workers": ArgumentDefinition(
            flag="--execution-workers",
            arg_type=int,
            default=1,
            help_text="Number of local or distributed workers for step 12 execution",
        ),
        "backend": ArgumentDefinition(
            flag="--backend",
            arg_type=str,
            default="ray",
            choices=["ray", "dask"],
            help_text="Distributed backend for step 12 (ray or dask)",
        ),
        "serialize_preset": ArgumentDefinition(
            flag="--serialize-preset",
            arg_type=str,
            default="full",
            choices=["full", "minimal"],
            help_text="Step 3: serialization preset (full=all formats; minimal=markdown+json+python)",
        ),
        "execution_benchmark_repeats": ArgumentDefinition(
            flag="--execution-benchmark-repeats",
            arg_type=int,
            default=1,
            help_text="Step 12: sequential benchmark repeats per script; median runtime when >1",
        ),
        "execution_summary_detail": ArgumentDefinition(
            flag="--execution-summary-detail",
            action=argparse.BooleanOptionalAction,
            default=False,
            help_text="Step 12: also write execution_summary_detail.json with full per-script payloads",
        ),
        "recreate_venv": ArgumentDefinition(
            flag="--recreate-uv-env",
            action="store_true",
            use_suppress=True,
            help_text="Recreate UV virtual environment",
        ),
        "dev": ArgumentDefinition(
            flag="--dev",
            action="store_true",
            use_suppress=True,
            help_text="Install development dependencies (uv sync --extra dev)",
        ),
        "install_all_extras": ArgumentDefinition(
            flag="--install-all-extras",
            action="store_true",
            help_text="Install all optional dependency groups (uv sync --all-extras)",
        ),
        "setup_core_only": ArgumentDefinition(
            flag="--setup-core-only",
            action="store_true",
            help_text=(
                "Step 1: skip the post-sync JAX/PyMDP self-test after installing core dependencies"
            ),
        ),
        "duration": ArgumentDefinition(
            flag="--duration",
            arg_type=float,
            default=30.0,
            help_text="Audio duration in seconds for audio generation",
        ),
        "audio_backend": ArgumentDefinition(
            flag="--audio-backend",
            arg_type=str,
            default="auto",
            help_text="Audio backend to use (auto, sapf, pedalboard, default: auto)",
        ),
        "sonification": ArgumentDefinition(
            flag="--sonification",
            action=argparse.BooleanOptionalAction,
            default=True,
            help_text="Generate model sonification",
        ),
        "full_analysis": ArgumentDefinition(
            flag="--full-analysis",
            action="store_true",
            default=False,
            help_text="Run full audio analysis",
        ),
        "fast_only": ArgumentDefinition(
            flag="--fast-only",
            action="store_true",
            use_suppress=True,
            help_text="Run only fast tests, skip slow and performance tests",
        ),
        "include_performance": ArgumentDefinition(
            flag="--include-performance",
            action="store_true",
            help_text="Include performance test categories",
        ),
        "comprehensive": ArgumentDefinition(
            flag="--comprehensive",
            action="store_true",
            use_suppress=True,
            help_text="Run all test categories including comprehensive suite",
        ),
        "install_optional": ArgumentDefinition(
            flag="--install-optional",
            action="store_true",
            help_text="Install optional dependency groups",
        ),
        "optional_groups": ArgumentDefinition(
            flag="--optional-groups",
            default=None,
            help_text="Comma-separated optional dependency groups to install, e.g. gui,audio",
        ),
        "viz_type": ArgumentDefinition(
            flag="--viz-type",
            arg_type=str,
            default="all",
            choices=[
                "all",
                "3d",
                "interactive",
                "dashboard",
                "d2",
                "diagrams",
                "pipeline",
                "statistical",
                "pomdp",
                "network",
            ],
            help_text="Step 9: type of advanced visualization to generate",
        ),
        "interactive": ArgumentDefinition(
            flag="--interactive",
            action=argparse.BooleanOptionalAction,
            default=False,
            help_text="Enable interactive mode where supported",
        ),
        "export_formats": ArgumentDefinition(
            flag="--export-formats",
            arg_type=str,
            default=["html", "json"],
            nargs="+",
            help_text="Step 9: visualization export formats",
        ),
        "headless": ArgumentDefinition(
            flag="--headless",
            action="store_true",
            default=False,
            help_text="Step 22: run GUI processors in headless artifact mode",
        ),
        "gui_types": ArgumentDefinition(
            flag="--gui-types",
            arg_type=str,
            default="gui_1,gui_2",
            help_text="Step 22: comma-separated GUI processors to run",
        ),
        "open_browser": ArgumentDefinition(
            flag="--open-browser",
            action="store_true",
            default=False,
            help_text="Step 22: open browser for interactive GUIs",
        ),
        "launch_editor": ArgumentDefinition(
            flag="--launch-editor",
            action="store_true",
            default=False,
            help_text="Step 22: launch the oxdraw editor (interactive oxdraw GUI type)",
        ),
        "analysis_model": ArgumentDefinition(
            flag="--analysis-model",
            arg_type=str,
            default=None,
            help_text="Step 24: LLM model tag for intelligent analysis",
        ),
        "bottleneck_threshold": ArgumentDefinition(
            flag="--bottleneck-threshold",
            arg_type=float,
            default=60.0,
            help_text="Step 24: duration threshold in seconds for bottleneck detection",
        ),
        "timesteps": ArgumentDefinition(
            flag="--timesteps",
            arg_type=int,
            default=None,
            help_text="Number of timesteps for simulation",
        ),
        "simulation_params": ArgumentDefinition(
            flag="--simulation-params",
            arg_type=str,
            default="{}",
            help_text="JSON string containing simulation parameters",
        ),
        "timeout": ArgumentDefinition(
            flag="--timeout",
            arg_type=int,
            default=300,
            help_text="Timeout for execution in seconds",
        ),
        "advanced_stats": ArgumentDefinition(
            flag="--advanced-stats",
            action="store_true",
            help_text="Include advanced statistical analysis",
        ),
        "generate_animations": ArgumentDefinition(
            flag="--no-animations",
            action="store_false",
            default=True,
            dest="generate_animations",
            help_text=(
                "Disable Step 16 GridWorld GIF animation artifacts (enabled by default)"
            ),
        ),
        "geo_step_seconds": ArgumentDefinition(
            flag="--geo-step-seconds",
            arg_type=float,
            help_text="GEO-INFER export (step 7): explicit step seconds",
        ),
        "geo_state_ids": ArgumentDefinition(
            flag="--geo-state-ids",
            arg_type=str,
            help_text="GEO-INFER export (step 7): path to state IDs file",
        ),
        "geo_space_kind": ArgumentDefinition(
            flag="--geo-space-kind",
            arg_type=str,
            choices=["categorical", "h3"],
            help_text="GEO-INFER export (step 7): space kind",
        ),
        "geo_derive_metadata": ArgumentDefinition(
            flag="--geo-derive-metadata",
            action="store_true",
            help_text=(
                "GEO-INFER export (step 7): derive missing metadata "
                "(step seconds) from explicit notation declarations"
            ),
        ),
    }
)


STEP_ARGUMENTS = MappingProxyType(
    {
        "0_template.py": [
            "target_dir",
            "output_dir",
            "recursive",
            "verbose",
            "simulate_error",
        ],
        "1_setup.py": [
            "target_dir",
            "output_dir",
            "recursive",
            "verbose",
            "recreate_venv",
            "dev",
            "install_all_extras",
            "setup_core_only",
            "install_optional",
            "optional_groups",
        ],
        "2_tests.py": [
            "target_dir",
            "output_dir",
            "verbose",
            "fast_only",
            "include_performance",
            "comprehensive",
        ],
        "3_gnn.py": [
            "target_dir",
            "output_dir",
            "recursive",
            "verbose",
            "enable_round_trip",
            "enable_cross_format",
            "serialize_preset",
        ],
        "4_model_registry.py": [
            "target_dir",
            "output_dir",
            "recursive",
            "verbose",
            "registry_path",
            "query_ontology",
        ],
        "5_type_checker.py": [
            "target_dir",
            "output_dir",
            "recursive",
            "verbose",
            "strict",
            "estimate_resources",
        ],
        "6_validation.py": [
            "target_dir",
            "output_dir",
            "recursive",
            "verbose",
            "strict",
            "profile",
            "transpose_b",
        ],
        "7_export.py": [
            "target_dir",
            "output_dir",
            "recursive",
            "verbose",
            "geo_step_seconds",
            "geo_state_ids",
            "geo_space_kind",
            "geo_derive_metadata",
            "formats",
            "geo_infer_options_file",
        ],
        "8_visualization.py": ["target_dir", "output_dir", "recursive", "verbose"],
        "9_advanced_viz.py": [
            "target_dir",
            "output_dir",
            "recursive",
            "verbose",
            "viz_type",
            "interactive",
            "export_formats",
        ],
        "10_ontology.py": [
            "target_dir",
            "output_dir",
            "recursive",
            "verbose",
            "ontology_terms_file",
        ],
        "11_render.py": [
            "target_dir",
            "output_dir",
            "recursive",
            "verbose",
            "timesteps",
            "simulation_params",
            "frameworks",
            "strict_framework_success",
        ],
        "12_execute.py": [
            "target_dir",
            "output_dir",
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
        "13_llm.py": [
            "target_dir",
            "output_dir",
            "recursive",
            "verbose",
            "llm_tasks",
            "llm_timeout",
        ],
        "14_ml_integration.py": [
            "target_dir",
            "output_dir",
            "recursive",
            "verbose",
        ],
        "15_audio.py": [
            "target_dir",
            "output_dir",
            "recursive",
            "verbose",
            "duration",
            "audio_backend",
            "sonification",
            "full_analysis",
        ],
        "16_analysis.py": [
            "target_dir",
            "output_dir",
            "recursive",
            "verbose",
            "advanced_stats",
            "generate_animations",
        ],
        "17_integration.py": ["target_dir", "output_dir", "recursive", "verbose"],
        "18_security.py": ["target_dir", "output_dir", "recursive", "verbose"],
        "19_research.py": ["target_dir", "output_dir", "recursive", "verbose"],
        "20_website.py": [
            "target_dir",
            "output_dir",
            "recursive",
            "verbose",
        ],
        "21_mcp.py": [
            "target_dir",
            "output_dir",
            "recursive",
            "verbose",
            "performance_mode",
            "mcp_strict_validation",
            "mcp_cache_ttl",
            "mcp_per_module_timeout",
            "mcp_overall_timeout",
            "mcp_modules_allowlist",
        ],
        "22_gui.py": [
            "target_dir",
            "output_dir",
            "recursive",
            "verbose",
            "headless",
            "interactive",
            "gui_types",
            "open_browser",
            "launch_editor",
        ],
        "23_report.py": ["target_dir", "output_dir", "recursive", "verbose"],
        "24_intelligent_analysis.py": [
            "target_dir",
            "output_dir",
            "verbose",
            "analysis_model",
            "skip_llm",
            "bottleneck_threshold",
        ],
        "main.py": list(ARGUMENT_DEFINITIONS.keys()),
    }
)
