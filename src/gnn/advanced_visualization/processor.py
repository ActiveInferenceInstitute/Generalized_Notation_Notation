"""
Advanced Visualization Processor - Core Processing Logic

This module provides advanced visualization capabilities including:
- 3D network visualizations
- Interactive dashboards
- Statistical analysis plots
- Multi-format export support
- Comprehensive error handling and recovery mechanisms

Implementation is split across sub-modules for maintainability:
- _shared: Dataclasses, validation, and utility functions (no circular imports)
- network_viz: 3D visualization, interactive dashboards, network metrics, D2 diagrams
- statistical_viz: Statistical plots, matrix correlations
- interactive_viz: Plotly dashboard generation

This file is the public processing facade for the sub-modules listed above.
"""

import importlib.util
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from gnn.utils.observability.performance_tracking import PerformanceTracker

# Availability flags, guarded imports, and the shared attempt/result types
# all live in ._shared; this module only adds the Step 9 orchestration logic.
from ._shared import (
    MATPLOTLIB_AVAILABLE,
    SEABORN_AVAILABLE,
    AdvancedVisualizationResults,
    np,
    record_attempt,
)


class SafeAdvancedVisualizationManager:
    """Context manager for safe advanced visualization with automatic cleanup"""

    def __init__(self, logger: logging.Logger) -> None:
        """Initialize the instance."""
        self.logger = logger
        self.tracker = PerformanceTracker()
        self.start_time: float | None = None

    def __enter__(self) -> Any:
        """Enter the context manager."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        """Exit the context manager and release resources."""
        if exc_type is not None:
            self.logger.warning(f"Advanced visualization encountered error: {exc_val}")
        return False  # Don't suppress exceptions


# Re-export from interactive_viz / network_viz / statistical_viz (isort: i < n < s)
from .interactive_viz import _generate_interactive_plotly_dashboard
from .network_viz import (
    _generate_3d_visualization,
    _generate_d2_visualizations_safe,
    _generate_interactive_dashboard,
    _generate_network_metrics,
    _generate_pipeline_d2_diagrams_safe,
    _generate_policy_visualization,
    _generate_pomdp_transition_analysis,
)
from .statistical_viz import (
    _generate_matrix_correlations,
    _generate_statistical_plots,
)

SEABORN_AVAILABLE = (
    SEABORN_AVAILABLE  # re-exported for back-compat (module-level symbol)
)


def _check_dependencies(logger: logging.Logger) -> Dict[str, bool]:
    """Check availability of visualization dependencies"""
    dependencies: dict[str, Any] = {
        "matplotlib": MATPLOTLIB_AVAILABLE,
        "plotly": False,
        "seaborn": SEABORN_AVAILABLE,
        "bokeh": False,
        "numpy": np is not None,
    }
    if importlib.util.find_spec("plotly") is not None:
        dependencies["plotly"] = True
    else:
        logger.info("plotly not available - interactive visualizations will be limited")
    if importlib.util.find_spec("bokeh") is not None:
        dependencies["bokeh"] = True
    else:
        logger.debug("bokeh not available - will use plotly recovery")
    return dependencies


def _load_gnn_models(
    target_dir: Path, logger: logging.Logger, base_output_dir: Optional[Path] = None
) -> Dict[str, Dict[str, Any]]:
    """Load GNN models from processing results"""
    from gnn.pipeline.config import get_output_dir_for_script

    # Get GNN output directory
    if base_output_dir is None:
        base_output_dir = Path("output")
    gnn_output_dir = get_output_dir_for_script("3_gnn.py", base_output_dir)

    logger.info(f"Looking for GNN output in: {gnn_output_dir}")

    # Check for double-nested directory structure
    results_file = gnn_output_dir / "gnn_processing_results.json"
    logger.info(
        f"Looking for results file: {results_file} (exists: {results_file.exists()})"
    )

    if not results_file.exists():
        # Try nested structure
        nested_results_file = (
            gnn_output_dir / "3_gnn_output" / "gnn_processing_results.json"
        )
        logger.info(
            f"Looking for nested results file: {nested_results_file} (exists: {nested_results_file.exists()})"
        )
        if nested_results_file.exists():
            results_file = nested_results_file
            gnn_output_dir = gnn_output_dir / "3_gnn_output"

    if not results_file.exists():
        logger.warning(f"GNN processing results not found at {results_file}")
        from gnn.pipeline.run_context import current_run_context

        if current_run_context() is not None:
            raise ValueError(
                "Current-run Step 3 receipt is required for advanced visualization"
            )
        # Try to find any parsed JSON files in the GNN output directory
        parsed_files = sorted(gnn_output_dir.glob("**/*_parsed.json"))
        logger.info(f"Found {len(parsed_files)} parsed files in {gnn_output_dir}")
        if parsed_files:
            logger.info(f"Found {len(parsed_files)} parsed files, loading directly")
            models: dict[Any, Any] = {}
            for parsed_file in parsed_files:
                logger.info(f"Processing parsed file: {parsed_file}")
                try:
                    with open(parsed_file) as f:
                        model_data = json.load(f)
                    model_name = parsed_file.stem.replace("_parsed", "")
                    models[model_name] = model_data
                    logger.info(f"Loaded model: {model_name}")
                except Exception as e:
                    logger.warning(f"Failed to load {parsed_file}: {e}")
            return models
        return {}

    # Load results
    try:
        with open(results_file) as f:
            processing_results = json.load(f)
        from gnn.pipeline.run_context import current_run_context

        context = current_run_context()
        if context is not None and processing_results.get("run_id") != context.run_id:
            raise ValueError("Step 3 model inventory belongs to a different run")
        allowed_stems = (
            {m.artifact_stem for m in context.selected_models(9)} if context else None
        )

        models = {}
        # The results file uses "processed_files" not "results"
        processed_files = processing_results.get("processed_files", [])
        logger.info(f"Found {len(processed_files)} processed files in results")

        for result in sorted(
            processed_files,
            key=lambda item: str(
                item.get("parsed_model_file") or item.get("file_name") or ""
            ),
        ):
            if result.get(
                "parse_success"
            ):  # Note: it's "parse_success" not "parsing_success"
                parsed_model_file = result.get("parsed_model_file")
                if parsed_model_file and parsed_model_file.endswith("_parsed.json"):
                    # Extract model name from file path
                    model_name = parsed_model_file.split("/")[-1].replace(
                        "_parsed.json", ""
                    )
                    if allowed_stems is not None and model_name not in allowed_stems:
                        continue

                    # Construct full path to parsed file
                    parsed_file = Path(parsed_model_file)

                    if parsed_file.exists():
                        try:
                            with open(parsed_file) as f:
                                model_data = json.load(f)
                            models[model_name] = model_data
                            logger.info(f"Loaded parsed model: {model_name}")
                        except Exception as e:
                            logger.warning(
                                f"Failed to load model {model_name} from {parsed_file}: {e}"
                            )
                    else:
                        # Try to resolve relative to gnn_output_dir
                        # The JSON contains "output_directory" which is the root for these files usually
                        json_out_dir = processing_results.get("output_directory")
                        if json_out_dir and str(parsed_model_file).startswith(
                            str(json_out_dir)
                        ):
                            rel_path = str(parsed_model_file)[
                                len(str(json_out_dir)) :
                            ].lstrip("/\\")
                            parsed_file = gnn_output_dir / rel_path

                        if parsed_file.exists():
                            try:
                                with open(parsed_file) as f:
                                    model_data = json.load(f)
                                models[model_name] = model_data
                                logger.info(
                                    f"Loaded parsed model (path resolved): {model_name}"
                                )
                            except Exception as e:
                                logger.warning(
                                    f"Failed to load model {model_name} from {parsed_file}: {e}"
                                )
                        else:
                            logger.warning(
                                f"Parsed model file not found: {parsed_file}"
                            )
            else:
                logger.warning(
                    f"Skipping failed parse result: {result.get('file_name', 'unknown')}"
                )

        return models

    except Exception as e:
        logger.error(f"Failed to load GNN models: {e}")
        return {}


def _save_results(
    output_dir: Path, results: AdvancedVisualizationResults, logger: logging.Logger
) -> Any:
    """Save visualization results to JSON with detailed skipped feature tracking"""
    # Categorize skipped visualizations by reason
    skipped_by_reason: dict[Any, Any] = {}
    for attempt in results.attempts:
        if attempt.status == "skipped":
            reason = attempt.error_message or "Unknown reason"
            if reason not in skipped_by_reason:
                skipped_by_reason[reason] = []
            skipped_by_reason[reason].append(f"{attempt.viz_type}:{attempt.model_name}")

    # Build the summary
    summary: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "total_attempts": results.total_attempts,
        "successful": results.successful,
        "failed": results.failed,
        "skipped": results.skipped,
        "total_duration_ms": results.total_duration_ms,
        "output_files": results.output_files,
        "warnings": results.warnings,
        "errors": results.errors,
        "skipped_features": {
            "count": results.skipped,
            "by_reason": skipped_by_reason,
            "details": [
                {
                    "feature": f"{a.viz_type}:{a.model_name}",
                    "reason": a.error_message or "Unknown",
                    "fallback_available": a.fallback_used,
                }
                for a in results.attempts
                if a.status == "skipped"
            ],
        },
        "attempts": [
            {
                "viz_type": a.viz_type,
                "model_name": a.model_name,
                "status": a.status,
                "duration_ms": a.duration_ms,
                "output_files": a.output_files,
                "error_message": a.error_message,
                "fallback_used": a.fallback_used,
            }
            for a in results.attempts
        ],
    }
    from gnn.pipeline.run_context import current_run_context

    context = current_run_context()
    if context is not None:
        summary["run_id"] = context.run_id
        summary["model_selection"] = [m.__dict__ for m in context.selected_models(9)]

    output_file = output_dir / "advanced_viz_summary.json"
    try:
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved advanced visualization summary: {output_file}")
    except OSError as e:
        logger.warning(f"Failed to save advanced visualization summary: {e}")

    # Log detailed skipped feature report if there are skipped items
    if skipped_by_reason:
        logger.info(f"Skipped visualization features ({results.skipped} total):")
        for reason, features in skipped_by_reason.items():
            logger.info(f"  - {reason}: {len(features)} feature(s)")
            for feature in features[:3]:  # Show first 3 examples
                logger.debug(f"    * {feature}")
            if len(features) > 3:
                logger.debug(f"    ... and {len(features) - 3} more")


def process_advanced_viz(
    target_dir: Path,
    output_dir: Path,
    logger: Optional[logging.Logger] = None,
    viz_type: str = "all",
    interactive: bool = True,
    export_formats: Optional[List[str]] = None,
    **kwargs: Any,
) -> Union[bool, int]:
    """
    Main advanced visualization processing function.

    Args:
        target_dir: Directory containing GNN files
        output_dir: Output directory for visualizations
        logger: Optional logger instance; falls back to module logger
        viz_type: Type of visualization ("all", "3d", "interactive", "dashboard")
        interactive: Enable interactive visualizations
        export_formats: List of export formats ["html", "json", "png"]
        **kwargs: Additional arguments

    Returns:
        True if artifacts were produced, 2 for warning-only/no-data recovery,
        False for hard failures.
    """
    logger = logger or logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("ADVANCED VISUALIZATION PROCESSING")
    logger.info("=" * 80)

    results = AdvancedVisualizationResults()

    # Set default export formats
    if export_formats is None:
        export_formats = ["html", "json"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for required dependencies
    dependencies_available = _check_dependencies(logger)

    try:
        with SafeAdvancedVisualizationManager(logger):
            # Load GNN processing results
            gnn_models = _load_gnn_models(
                target_dir,
                logger,
                output_dir.parent
                if output_dir.name.endswith("_output")
                else output_dir,
            )

            if not gnn_models:
                logger.warning("No GNN models found for advanced visualization")
                results.warnings.append("No GNN models found")
                _save_results(output_dir, results, logger)
                return 2  # Not a hard failure, but no artifacts were possible.

            # Process each model
            for model_name, model_data in sorted(gnn_models.items()):
                logger.info(f"Processing advanced visualizations for: {model_name}")

                # Generate visualizations based on type
                if viz_type in ["all", "3d"]:
                    record_attempt(
                        results,
                        _generate_3d_visualization(
                            model_name,
                            model_data,
                            output_dir,
                            export_formats,
                            dependencies_available,
                            logger,
                        ),
                    )

                if viz_type in ["all", "statistical"]:
                    record_attempt(
                        results,
                        _generate_statistical_plots(
                            model_name,
                            model_data,
                            output_dir,
                            dependencies_available,
                            logger,
                        ),
                    )

                if viz_type in ["all", "statistical"]:
                    record_attempt(
                        results,
                        _generate_matrix_correlations(
                            model_name,
                            model_data,
                            output_dir,
                            dependencies_available,
                            logger,
                        ),
                    )

                if viz_type in ["all", "pomdp"]:
                    record_attempt(
                        results,
                        _generate_pomdp_transition_analysis(
                            model_name,
                            model_data,
                            output_dir,
                            dependencies_available,
                            logger,
                        ),
                    )

                if viz_type in ["all", "pomdp"]:
                    record_attempt(
                        results,
                        _generate_policy_visualization(
                            model_name,
                            model_data,
                            output_dir,
                            dependencies_available,
                            logger,
                        ),
                    )

                if viz_type in ["all", "interactive"] and interactive:
                    record_attempt(
                        results,
                        _generate_interactive_plotly_dashboard(
                            model_name,
                            model_data,
                            output_dir,
                            export_formats,
                            dependencies_available,
                            logger,
                        ),
                    )

                if viz_type in ["all", "dashboard"] and interactive:
                    record_attempt(
                        results,
                        _generate_interactive_dashboard(
                            model_name,
                            model_data,
                            output_dir,
                            export_formats,
                            dependencies_available,
                            logger,
                        ),
                    )

                if viz_type in ["all", "network"]:
                    record_attempt(
                        results,
                        _generate_network_metrics(
                            model_name,
                            model_data,
                            output_dir,
                            dependencies_available,
                            logger,
                        ),
                    )

                # Generate D2 diagrams for each model
                if viz_type in ["all", "d2", "diagrams"]:
                    record_attempt(
                        results,
                        _generate_d2_visualizations_safe(
                            model_data, output_dir, logger
                        ),
                        optional_message_filter="D2 CLI",
                    )

            # Generate pipeline-level D2 diagrams (once for all models)
            if viz_type in ["all", "d2", "diagrams", "pipeline"]:
                record_attempt(
                    results,
                    _generate_pipeline_d2_diagrams_safe(output_dir, logger),
                    optional_message_filter="D2 CLI",
                )

        # Save results
        _save_results(output_dir, results, logger)

        # Log summary
        logger.info("Advanced visualization complete:")
        logger.info(f"  Total attempts: {results.total_attempts}")
        logger.info(f"  Successful: {results.successful}")
        logger.info(f"  Failed: {results.failed}")
        logger.info(f"  Skipped: {results.skipped}")
        logger.info(f"  Output files: {len(results.output_files)}")

        # Return success if:
        # 1. At least some visualizations succeeded, OR
        # 2. No attempts were made (no data), OR
        # 3. Only failures are skipped optional features (no actual errors)
        if results.successful > 0:
            return True
        if results.total_attempts == 0 or (results.failed == 0 and results.skipped > 0):
            return 2
        return False

    except Exception as e:
        logger.error(f"Advanced visualization processing failed: {e}")
        results.errors.append(str(e))
        _save_results(output_dir, results, logger)
        return False
