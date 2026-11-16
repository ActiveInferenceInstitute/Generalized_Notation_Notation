"""
Shared utilities for advanced visualization sub-modules.

Contains dataclasses, validation, and helper functions used by both
processor.py, network_viz.py, and statistical_viz.py. Exists to
avoid circular imports between processor and sub-modules.
"""

import importlib.util
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

logger = logging.getLogger(__name__)

_SNS_MODULE: Any = None
_SNS_LOAD_FAILED: bool = False

sns: Any

FORCE_LAYOUT_SEED = 42

__all__ = [
    "FORCE_LAYOUT_SEED",
    "LAYOUT_SEED",
    "LAYOUT_SPAN",
    "LAYOUT_ITERATIONS",
    "LAYOUT_STEP",
    "NUMPY_AVAILABLE",
    "MATPLOTLIB_AVAILABLE",
    "SEABORN_AVAILABLE",
    "np",
    "plt",
    "sns",
    "VAR_TYPE_COLORS",
    "VAR_TYPE_UNKNOWN_COLOR",
    "AdvancedVisualizationAttempt",
    "AdvancedVisualizationResults",
    "record_attempt",
    "normalize_connection_format",
    "_conn_endpoints",
    "_calculate_semantic_positions",
    "validate_visualization_data",
    "_generate_fallback_report",
    "_MatrixVisualizer",
]

LAYOUT_SEED = FORCE_LAYOUT_SEED
LAYOUT_SPAN = 10.0
LAYOUT_ITERATIONS = 50
LAYOUT_STEP = 0.01

from gnn.visualization.connection_format import (
    conn_endpoints as _conn_endpoints,
)
from gnn.visualization.connection_format import (
    normalize_connection_format as normalize_connection_format,
)
from gnn.visualization.theme import VAR_TYPE_COLORS_3D as _THEME_3D_PALETTE

# Single-source palette: the 3-D subset of the canonical visualization theme.
# Do not inline hex values here; extend VAR_TYPE_COLORS_3D in
# ``gnn/visualization/theme.py`` instead.
VAR_TYPE_COLORS: dict[str, str] = {
    key: _THEME_3D_PALETTE[key]
    for key in (
        "likelihood_matrix",
        "transition_matrix",
        "preference_vector",
        "prior_vector",
        "hidden_state",
        "observation",
        "policy",
        "action",
    )
}
VAR_TYPE_UNKNOWN_COLOR = _THEME_3D_PALETTE["unknown"]


try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    np = cast(Any, None)
    NUMPY_AVAILABLE = False

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = cast(Any, None)
SEABORN_AVAILABLE = importlib.util.find_spec("seaborn") is not None


def get_sns() -> Any:
    """Return the seaborn module (loading it on first call), or ``None``."""
    global _SNS_MODULE, _SNS_LOAD_FAILED
    if _SNS_MODULE is None and not _SNS_LOAD_FAILED:
        try:
            import seaborn as _sns

            _SNS_MODULE = _sns
        except ImportError:
            logger.debug("seaborn unavailable; seaborn-based plots will be skipped")
            _SNS_LOAD_FAILED = True
    return _SNS_MODULE


def __getattr__(name: str) -> Any:
    if name == "sns":
        return get_sns()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class _LazyMatrixVisualizer:
    """Defer MatrixVisualizer import until advanced visualization actually needs it."""

    def __call__(self) -> Any:
        """Return a MatrixVisualizer, or ``None`` when the optional dependency
        ``visualization.matrix_visualizer`` is unavailable (callers treat a
        ``None`` result as the documented "MatrixVisualizer not available" skip
        instead of surfacing a raw ImportError)."""
        try:
            from gnn.visualization.matrix_visualizer import MatrixVisualizer
        except ImportError:
            return None
        return MatrixVisualizer()


_MatrixVisualizer = _LazyMatrixVisualizer()


@dataclass
class AdvancedVisualizationAttempt:
    """Track individual visualization attempts"""

    viz_type: str
    model_name: str
    status: str  # "success", "failed", "skipped"
    duration_ms: float = 0.0
    output_files: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    fallback_used: bool = False


@dataclass
class AdvancedVisualizationResults:
    """Aggregate results for advanced visualization processing"""

    total_attempts: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    total_duration_ms: float = 0.0
    attempts: List[AdvancedVisualizationAttempt] = field(default_factory=list)
    output_files: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    parsed_model_schema: Optional[str] = None
    parsed_inputs: Dict[str, Dict[str, Any]] = field(default_factory=dict)


def record_attempt(
    results: "AdvancedVisualizationResults",
    attempt: AdvancedVisualizationAttempt,
    *,
    optional_message_filter: Optional[str] = None,
) -> None:
    """Record an :class:`AdvancedVisualizationAttempt` onto aggregate results.

    Pure aggregate bookkeeping used by :func:`process_advanced_viz`. Counts the
    attempt, extends ``output_files``/``errors``/``warnings`` by status, and
    (for ``skipped``) optionally suppresses warning entries whose message
    mentions the optional dependency marker (e.g. ``"D2 CLI"``) so optional
    CLI absence is not surfaced as a hard warning.
    """
    results.attempts.append(attempt)
    results.total_attempts += 1
    if attempt.status == "success":
        results.successful += 1
        results.output_files.extend(attempt.output_files)
    elif attempt.status == "failed":
        results.failed += 1
        if attempt.error_message:
            results.errors.append(attempt.error_message)
    else:  # skipped
        results.skipped += 1
        message = attempt.error_message
        if message and (
            optional_message_filter is None or optional_message_filter not in message
        ):
            results.warnings.append(message)


def _calculate_semantic_positions(
    variables: List[Dict[str, Any]], connections: List[Dict[str, Any]]
) -> Any:
    """
    Calculate meaningful 3D positions for variables based on semantic relationships.

    Args:
        variables: List of variable dictionaries
        connections: List of connection dictionaries

    Returns:
        Array of 3D positions for each variable
    """
    if not NUMPY_AVAILABLE or np is None:
        return []

    if not variables:
        return np.array([])

    n_vars = len(variables)

    rng = np.random.default_rng(LAYOUT_SEED)
    positions = rng.random((n_vars, 3)) * LAYOUT_SPAN

    var_names = [var.get("name", f"var_{i}") for i, var in enumerate(variables)]
    connection_matrix = np.zeros((n_vars, n_vars))

    for conn_info in connections:
        source_vars, target_vars = _conn_endpoints(conn_info)

        for source_var in source_vars:
            for target_var in target_vars:
                if source_var != target_var:
                    source_idx = None
                    target_idx = None

                    for idx, name in enumerate(var_names):
                        if name == source_var:
                            source_idx = idx
                        if name == target_var:
                            target_idx = idx

                    if source_idx is not None and target_idx is not None:
                        connection_matrix[source_idx, target_idx] = 1

    for _ in range(LAYOUT_ITERATIONS):
        forces = np.zeros_like(positions)

        for i in range(n_vars):
            for j in range(n_vars):
                if i != j:
                    diff = positions[i] - positions[j]
                    distance = np.linalg.norm(diff)
                    if distance > 0:
                        forces[i] += (diff / distance) * (1 / distance)

        for i in range(n_vars):
            for j in range(n_vars):
                if connection_matrix[i, j] > 0:
                    diff = positions[j] - positions[i]
                    distance = np.linalg.norm(diff)
                    if distance > 0:
                        forces[i] += diff * (distance / LAYOUT_SPAN)

        positions += forces * LAYOUT_STEP

    positions = (
        (positions - positions.min())
        / (positions.max() - positions.min())
        * LAYOUT_SPAN
    )

    return positions


def _generate_fallback_report(
    model_name: str,
    viz_type: str,
    output_dir: Path,
    model_data: Dict[str, Any],
    logger: logging.Logger,
) -> Any:
    """Generate recovery HTML report when advanced libraries unavailable"""
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{model_name} - {viz_type.upper()} Visualization (Recovery)</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .info {{ background: #f0f0f0; padding: 10px; margin: 10px 0; }}
        .data {{ background: #fff; border: 1px solid #ddd; padding: 10px; }}
        pre {{ background: #f5f5f5; padding: 10px; overflow-x: auto; }}
    </style>
</head>
<body>
    <h1>{model_name} - {viz_type.upper()} Visualization</h1>
    <div class="info">
        <p><strong>Note:</strong> Advanced visualization libraries not available.
        Showing basic model information instead.</p>
    </div>
    <div class="data">
        <h2>Model Structure</h2>
        <pre>{json.dumps(model_data, indent=2)}</pre>
    </div>
</body>
</html>"""

    output_file = output_dir / f"{model_name}_{viz_type}_fallback.html"
    with open(output_file, "w") as f:
        f.write(html_content)

    logger.info(f"Generated recovery report: {output_file}")


def validate_visualization_data(
    model_data: Dict[str, Any], logger: logging.Logger
) -> Dict[str, Any]:
    """
    Validate that visualization data is complete and meaningful.

    Args:
        model_data: Parsed model data
        logger: Logger instance

    Returns:
        Validation results dictionary
    """
    validation_results: dict[str, Any] = {
        "overall_valid": True,
        "warnings": [],
        "errors": [],
        "data_quality": {},
        "recommendations": [],
    }

    try:
        model_data_obj: object = model_data
        if not isinstance(model_data_obj, dict):
            validation_results["errors"].append("Model data is not a dictionary")
            validation_results["overall_valid"] = False
            return validation_results

        required_keys: list[Any] = ["variables", "connections"]
        for key in required_keys:
            if key not in model_data:
                validation_results["warnings"].append(f"Missing key: {key}")
            elif not model_data[key]:
                validation_results["warnings"].append(f"Empty data for key: {key}")

        variables = model_data.get("variables", [])
        if not variables:
            validation_results["errors"].append("No variables found in model")
            validation_results["overall_valid"] = False
        else:
            validation_results["data_quality"]["total_variables"] = len(variables)

            valid_vars = 0
            for var in variables:
                if isinstance(var, dict) and "name" in var and "var_type" in var:
                    valid_vars += 1
                else:
                    validation_results["warnings"].append(
                        f"Invalid variable structure: {var}"
                    )

            validation_results["data_quality"]["valid_variables"] = valid_vars
            validation_results["data_quality"]["variable_validity_rate"] = (
                valid_vars / len(variables)
            )

            if valid_vars < len(variables) * 0.8:
                validation_results["warnings"].append("Low variable validity rate")

        connections = model_data.get("connections", [])
        if not connections:
            validation_results["warnings"].append("No connections found in model")
        else:
            validation_results["data_quality"]["total_connections"] = len(connections)

            valid_connections = 0
            for conn in connections:
                if isinstance(conn, dict) and (
                    "source_variables" in conn or "target_variables" in conn
                ):
                    valid_connections += 1
                else:
                    validation_results["warnings"].append(
                        f"Invalid connection structure: {conn}"
                    )

            validation_results["data_quality"]["valid_connections"] = valid_connections
            validation_results["data_quality"]["connection_validity_rate"] = (
                valid_connections / len(connections)
            )

        pomdp_indicators: dict[str, Any] = {
            "likelihood_matrix": 0,
            "transition_matrix": 0,
            "preference_vector": 0,
            "prior_vector": 0,
            "hidden_state": 0,
            "observation": 0,
            "policy": 0,
        }

        for var in variables:
            if isinstance(var, dict):
                var_type = var.get("var_type", "")
                for indicator in pomdp_indicators:
                    if indicator in var_type:
                        pomdp_indicators[indicator] += 1

        validation_results["data_quality"]["pomdp_indicators"] = pomdp_indicators

        pomdp_score = sum(pomdp_indicators.values())
        if pomdp_score >= 3:
            validation_results["data_quality"]["is_pomdp_model"] = True
            validation_results["data_quality"]["pomdp_completeness"] = (
                pomdp_score / len(pomdp_indicators)
            )
        else:
            validation_results["data_quality"]["is_pomdp_model"] = False
            validation_results["warnings"].append(
                "Model does not appear to be a complete POMDP"
            )

        if validation_results["data_quality"].get("variable_validity_rate", 1) < 0.9:
            validation_results["recommendations"].append(
                "Review variable parsing - high invalidity rate"
            )

        if validation_results["data_quality"].get("connection_validity_rate", 1) < 0.9:
            validation_results["recommendations"].append(
                "Review connection parsing - high invalidity rate"
            )

        if pomdp_score < 3:
            validation_results["recommendations"].append(
                "Model may not be a complete POMDP - check GNN structure"
            )

        if validation_results["errors"]:
            validation_results["overall_valid"] = False
        elif len(validation_results["warnings"]) > 2:
            validation_results["overall_valid"] = False
            validation_results["warnings"].append(
                "Too many warnings - data quality may be poor"
            )

        if logger:
            logger.info(
                f"Validation completed: {validation_results['overall_valid']} (errors: {len(validation_results['errors'])}, warnings: {len(validation_results['warnings'])})"
            )

    except Exception as e:
        validation_results["errors"].append(f"Validation error: {e}")
        validation_results["overall_valid"] = False
        if logger:
            logger.error(f"Validation failed: {e}")

    return validation_results
