#!/usr/bin/env python3
"""Parsing and normalization helpers for the GNN render processor.

Output-stem sanitization, typed-node/spec mapping adaptation, file-backed
parse-summary rehydration, initial-vector flattening, POMDP render
validation, and probability-matrix normalization shared by the render
pipeline.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

from gnn.render.naming import safe_output_stem

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    np = cast(Any, None)
    NUMPY_AVAILABLE = False


def _safe_output_stem(value: Any, fallback: str = "model") -> str:
    """Sanitize a value into a filesystem-safe output stem (see render.naming)."""
    return safe_output_stem(value, fallback)


def _node_to_mapping(node: Any) -> Dict[str, Any]:
    """Convert one typed GNN node to the mapping renderers consume."""
    if isinstance(node, dict):
        return dict(node)
    to_dict = getattr(node, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if isinstance(value, dict):
            return value
    return {
        key: getattr(node, key)
        for key in ("name", "value", "id", "dimensions", "type", "description")
        if hasattr(node, key)
    }


def _internal_representation_to_mapping(gnn_spec: Any) -> Dict[str, Any]:
    """Adapt ``GNNInternalRepresentation`` without leaking typed nodes.

    Framework renderers use a serialisable mapping contract. Passing the
    representation's ``Variable``/``Parameter`` instances through directly
    previously caused JAX to call ``.get`` on those objects and silently emit
    a degraded fallback. Keep the public object input accepted, but lower it to
    the same concrete matrix contract as dictionary callers.
    """
    parameter_nodes = [_node_to_mapping(item) for item in gnn_spec.parameters]
    parameter_values = {
        str(item["name"]): item.get("value")
        for item in parameter_nodes
        if item.get("name")
    }
    model_parameters = {
        key: value
        for key, value in parameter_values.items()
        if key.startswith("num_")
        or key
        in {
            "seed",
            "random_seed",
            "action_precision",
            "inference_iterations",
        }
    }
    return {
        "name": gnn_spec.model_name or "model",
        "model_name": gnn_spec.model_name or "model",
        "variables": [_node_to_mapping(item) for item in gnn_spec.variables],
        "connections": [_node_to_mapping(item) for item in gnn_spec.connections],
        "parameters": parameter_nodes,
        "model_parameters": model_parameters,
        "initialparameterization": parameter_values,
    }


def _rehydrate_file_backed_parse_summary(
    gnn_spec: Dict[str, Any],
) -> Dict[str, Any]:
    """Turn the lightweight public parser summary back into a renderable spec.

    ``gnn.parse_gnn_file`` intentionally returns discovery metadata, including
    ``file_path`` and string variable names. The standalone renderer CLI used
    that summary as if it contained matrices; JAX then reported success after
    emitting a degraded fallback. When this exact summary shape is supplied, parse
    its source through the maintained POMDP extractor and use the canonical
    renderer mapping. Other dictionary contracts pass through unchanged.
    """
    if not (
        gnn_spec.get("success") is True
        and isinstance(gnn_spec.get("file_path"), str)
        and isinstance(gnn_spec.get("sections"), list)
    ):
        return gnn_spec

    source_path = Path(gnn_spec["file_path"])
    if not source_path.is_file():
        raise ValueError(f"Parsed GNN source file does not exist: {source_path}")

    from gnn.extract.pomdp_extractor import extract_pomdp_from_file

    from gnn.render.pomdp_processor import pomdp_to_gnn_spec

    pomdp_space = extract_pomdp_from_file(source_path, strict_validation=True)
    if pomdp_space is None:
        raise ValueError(
            f"Parsed GNN source is not a renderable POMDP specification: {source_path}"
        )
    return pomdp_to_gnn_spec(pomdp_space)


def _normalize_initial_vectors(gnn_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten singleton-row/column C/D/E vectors before dimension inference.

    The structured Markdown parser preserves braces around a tuple as a
    one-row vector. Canonical inference must count its entries, not that row.
    Genuine multidimensional arrays remain intact for normal validation.
    """
    initial = gnn_spec.get("initialparameterization") or gnn_spec.get(
        "initial_parameterization"
    )
    if not isinstance(initial, dict):
        return gnn_spec
    normalized = dict(initial)
    for name in ("C", "D", "E"):
        if name in normalized:
            vector = np.asarray(normalized[name])
            if vector.ndim == 2 and 1 in vector.shape:
                normalized[name] = vector.reshape(-1).tolist()
    return {**gnn_spec, "initialparameterization": normalized}


def _render_succeeded(
    *,
    success_count: int,
    total_files: int,
    total_framework_successes: int,
    total_framework_attempts: int,
    strict_framework_success: bool = False,
) -> bool | int:
    """Evaluate render success while preserving partial success for normal pipelines."""
    if total_files == 0:
        return 2
    if total_framework_attempts > 0:
        if strict_framework_success:
            return (
                success_count == total_files
                and total_framework_successes == total_framework_attempts
            )
        framework_success_rate = (
            total_framework_successes / total_framework_attempts
        ) * 100
        return framework_success_rate >= 80.0 or success_count > 0
    return success_count == total_files


def validate_pomdp_for_rendering(pomdp_space: Any) -> Tuple[bool, List[str]]:
    """
    Validate POMDP space for rendering requirements.

    Args:
        pomdp_space: Extracted POMDP space object

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors: list[Any] = []

    # Check basic dimensions
    if not hasattr(pomdp_space, "num_states") or pomdp_space.num_states is None:
        errors.append("Missing number of states")
    elif isinstance(pomdp_space.num_states, list):
        if any(n <= 0 for n in pomdp_space.num_states):
            errors.append("Invalid state dimensions (must be positive)")
    elif pomdp_space.num_states <= 0:
        errors.append("Invalid number of states (must be positive)")

    if (
        not hasattr(pomdp_space, "num_observations")
        or pomdp_space.num_observations is None
    ):
        errors.append("Missing number of observations")

    if not hasattr(pomdp_space, "num_actions") or pomdp_space.num_actions is None:
        errors.append("Missing number of actions")

    return len(errors) == 0, errors


def normalize_matrices(pomdp_space: Any, logger: logging.Logger) -> Any:
    """
    Normalize probability matrices in POMDP space.

    Ensures that:
    - A matrix (observation model): columns sum to 1 over the observation dimension
    - B matrix (transition model): columns sum to 1 over the next-state dimension (per action)

    Handles 2D arrays, 3D arrays, and list-of-arrays (factorial) structures.

    Args:
        pomdp_space: POMDP space object with optional A_matrix and B_matrix attributes
        logger: Logger instance

    Returns:
        POMDP space with normalized matrices, or the original object unchanged
        if normalization fails (normalization error logged as warning)
    """

    def _normalize_columns(matrix: np.ndarray) -> np.ndarray:
        """Normalize columns so each sums to 1. Works on 2D arrays."""
        m = np.asarray(matrix, dtype=np.float64)
        if m.ndim != 2:
            return m
        col_sums = m.sum(axis=0, keepdims=True)
        # Avoid division by zero: replace zero-sum columns with uniform
        zero_mask = col_sums == 0
        col_sums[zero_mask] = 1.0
        m = m / col_sums
        # Fill zero-sum columns with uniform distribution
        if np.any(zero_mask):
            uniform = np.ones(m.shape[0], dtype=np.float64) / m.shape[0]
            for col_idx in np.where(zero_mask.flatten())[0]:
                m[:, col_idx] = uniform
        return cast(np.ndarray, m)

    try:
        # Normalize A matrix (Observation model)
        # Columns should sum to 1.0 over the observation dimension
        if hasattr(pomdp_space, "A_matrix") and pomdp_space.A_matrix is not None:
            A = pomdp_space.A_matrix
            if isinstance(A, list) and len(A) > 0 and isinstance(A[0], np.ndarray):
                # Factorial structure: list of per-modality arrays
                pomdp_space.A_matrix = [_normalize_columns(a) for a in A]
                logger.debug(f"Normalized {len(A)} A-matrix modalities")
            else:
                A = np.asarray(A, dtype=np.float64)
                if A.ndim == 2:
                    pomdp_space.A_matrix = _normalize_columns(A)
                elif A.ndim == 3:
                    # 3D: (obs, states, factors) — normalize over axis 0 per slice
                    for i in range(A.shape[2]):
                        A[:, :, i] = _normalize_columns(A[:, :, i])
                    pomdp_space.A_matrix = A
                logger.debug("Normalized A matrix")

        # Normalize B matrix (Transition model)
        # Columns should sum to 1.0 over next-state dimension, per action
        if hasattr(pomdp_space, "B_matrix") and pomdp_space.B_matrix is not None:
            B = pomdp_space.B_matrix
            if isinstance(B, list) and len(B) > 0 and isinstance(B[0], np.ndarray):
                # Factorial structure: list of per-factor transition arrays
                normalized_B: list[Any] = []
                for b in B:
                    b = np.asarray(b, dtype=np.float64)
                    if b.ndim == 3:
                        # (next_state, current_state, action)
                        for a_idx in range(b.shape[2]):
                            b[:, :, a_idx] = _normalize_columns(b[:, :, a_idx])
                    elif b.ndim == 2:
                        b = _normalize_columns(b)
                    normalized_B.append(b)
                pomdp_space.B_matrix = normalized_B
                logger.debug(f"Normalized {len(B)} B-matrix factors")
            else:
                B = np.asarray(B, dtype=np.float64)
                if B.ndim == 3:
                    # (next_state, current_state, action)
                    for a_idx in range(B.shape[2]):
                        B[:, :, a_idx] = _normalize_columns(B[:, :, a_idx])
                elif B.ndim == 2:
                    B = _normalize_columns(B)
                pomdp_space.B_matrix = B
                logger.debug("Normalized B matrix")

    except Exception as e:
        logger.warning(f"Matrix normalization failed: {e}")

    return pomdp_space
