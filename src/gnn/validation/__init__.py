"""
Validation Module

This module provides comprehensive validation capabilities for GNN models,
including semantic validation, performance profiling, and consistency checking.
"""

from pathlib import Path
from typing import Any

from gnn import __version__

FEATURES: dict[str, Any] = {
    "semantic_validation": True,
    "performance_profiling": True,
    "consistency_checking": True,
    "multi_model_validation": True,
    "b_orientation_checks": True,
    "mcp_integration": True,
}

from .consistency_checker import ConsistencyChecker, check_consistency
from .orientation import check_b_orientation
from .performance_profiler import PerformanceProfiler, profile_performance
from .semantic_validator import (
    SemanticValidator,
    process_semantic_validation,
    validate_content,
)
from .simple import SimpleValidator
from .workflow import StageServices, validate_directory


def process_validation(
    target_dir: Path, output_dir: Path, verbose: bool = False, **kwargs: Any
) -> bool:
    """
    Main validation processing function for GNN models.

    This function orchestrates the complete validation workflow including:
    - Semantic validation
    - Performance profiling
    - Consistency checking

    Args:
        target_dir: Directory containing GNN files to validate
        output_dir: Output directory for validation results
        verbose: Whether to enable verbose logging
        **kwargs: Additional processing options:
            - ``validation_level`` (str): Semantic validation depth
              ("basic", "standard", "strict", "research"; default "standard").
            - ``strict`` (bool): Shorthand that raises ``validation_level`` to
              "strict" when True (wired to the orchestrator's --strict flag).
            - ``strict_probabilities`` (bool): Apply the strict categorical
              probability contract to the authored model (default False).
            - ``run_id`` (str): Stable identity for intentional accumulation
              across multiple step-3 manifests in one run.
            - ``transpose_b`` (bool): Opt-in canonical B-tensor transposition
              for the orientation stage; textbook (row-stochastic) transition
              tensors are transposed in memory and the per-tensor
              transposition is recorded in the receipt (default warnings-only;
              source files are never modified).
            - ``logger``, ``recursive``, and ``profile`` are accepted for the
              standardized pipeline-script contract and do not alter behavior.

    Returns:
        True for a nonempty current pass with no failed file validations.
    """
    strict = bool(kwargs.get("strict", False))
    default_level = "strict" if strict else "standard"
    validation_level = str(kwargs.get("validation_level", default_level))
    return validate_directory(
        target_dir,
        output_dir,
        services=StageServices(
            semantic=process_semantic_validation,
            performance=profile_performance,
            consistency=check_consistency,
            orientation=check_b_orientation,
        ),
        verbose=verbose,
        validation_level=validation_level,
        run_id=kwargs.get("run_id"),
        transpose_b=bool(kwargs.get("transpose_b", False)),
        strict_probabilities=bool(kwargs.get("strict_probabilities", False)),
    )


# Re-export main classes and functions
__all__: list[str] = [
    "__version__",
    "FEATURES",
    "SemanticValidator",
    "SimpleValidator",
    "PerformanceProfiler",
    "check_b_orientation",
    "validate_content",
    "profile_performance",
    "check_consistency",
    "process_validation",
    "validate_directory",
    "StageServices",
]


def get_module_info() -> dict:
    """Return module metadata for composability and MCP discovery."""
    return {
        "name": "validation",
        "version": __version__,
        "description": "Advanced validation and consistency checking",
        "features": FEATURES,
    }
