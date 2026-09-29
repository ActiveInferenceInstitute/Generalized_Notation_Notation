#!/usr/bin/env python3
"""Shared discrete A/B/C/D extraction for lightweight framework renderers.

Single source of truth for the parameter-source fallback chain and matrix
parsing used identically by the PyTorch and NumPyro renderers (previously a
verbatim ~50-line duplicate in each). The extraction is intentionally
permissive: missing matrices fall back to neutral defaults so the emitted
simulation scripts remain runnable for demo/specs without full
``initialparameterization`` blocks.

Parameter-source precedence (first non-empty wins):
    1. ``stateSpace.parameters``
    2. ``initialparameterization``
    3. ``parameters``

Array literals embedded as strings are parsed with
``gnn.utils.runtime_safety.safe_eval.safe_literal_eval`` (bounded), never ``eval``.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from gnn.render.matrix_utils import normalize_columns


def parse_gnn_matrix_value(raw: Any, default: Any) -> Any:
    """Parse one raw matrix/vector value from a GNN spec.

    ``None``/unparseable values return *default* unchanged; lists and
    arrays convert to ``float`` ndarrays; strings go through the safe
    literal evaluator (falling back to *default* on failure).
    """
    if raw is None:
        return default
    if isinstance(raw, (list, np.ndarray)):
        return np.array(raw, dtype=float)
    if isinstance(raw, str):
        try:
            from gnn.utils.runtime_safety.safe_eval import (
                MATRIX_MAX_LEN,
                safe_literal_eval,
            )

            parsed = safe_literal_eval(raw, max_len=MATRIX_MAX_LEN)
            return np.array(parsed, dtype=float)
        except Exception:
            return default
    return default


def extract_abcd_matrices(
    gnn_spec: dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract A, B, C, D matrices from a discrete GNN spec.

    Applies the shared fallback dimensions (2 hidden states unless declared),
    neutral defaults (identity A/B, one-hot C, uniform D), column-normalizes
    A (and 2-D B), and normalizes D to a probability vector.

    Returns:
        Tuple ``(A, B, C, D)`` as numpy arrays.
    """
    params = gnn_spec.get("stateSpace", {}).get("parameters", {})
    if not params:
        params = gnn_spec.get("initialparameterization", {})
    if not params:
        params = gnn_spec.get("parameters", {})

    num_states = gnn_spec.get("stateSpace", {}).get("size", None)
    if num_states is None:
        num_states = gnn_spec.get("model_parameters", {}).get("num_hidden_states", 2)
    num_obs = gnn_spec.get("observationSpace", {}).get("size", None)
    if num_obs is None:
        num_obs = gnn_spec.get("model_parameters", {}).get("num_obs", num_states)

    default_a = np.eye(num_obs, num_states)
    default_b = np.eye(num_states)
    default_c = np.zeros(num_obs)
    default_c[0] = 1.0
    default_d = np.ones(num_states) / num_states

    a = parse_gnn_matrix_value(params.get("A"), default_a)
    b = parse_gnn_matrix_value(params.get("B"), default_b)
    c = parse_gnn_matrix_value(params.get("C"), default_c)
    d = parse_gnn_matrix_value(params.get("D"), default_d)

    a = normalize_columns(a)
    if b.ndim == 2:
        b = normalize_columns(b)
    d = d / d.sum() if d.sum() > 0 else d

    return a, b, c, d


def format_array_literal(
    arr: np.ndarray, *, prefix: str, suffix: str = "", indent: int = 4
) -> str:
    """Format a numpy array as a ``<prefix>(...)`` code literal.

    1-D and 2-D arrays render with round-trip (``repr``) floats and matching
    indentation, so a normalized distribution stays normalized in the generated
    code (6-decimal rounding turned a uniform ``1/3`` prior into a vector summing
    to 0.999999, which NumPyro's simplex validation rejects); higher-rank arrays
    fall back to ``arr.tolist()``. *suffix* is appended
    inside the parentheses (e.g. ``", dtype=torch.float64"``).
    """
    body_prefix = " " * indent
    if arr.ndim == 1:
        vals = ", ".join(repr(float(v)) for v in arr)
        return f"{prefix}([{vals}]{suffix})"
    if arr.ndim == 2:
        rows = []
        for row in arr:
            vals = ", ".join(repr(float(v)) for v in row)
            rows.append(f"{body_prefix}    [{vals}]")
        inner = ",\n".join(rows)
        return f"{prefix}([\n{inner}\n{body_prefix}]{suffix})"
    return f"{prefix}({arr.tolist()}{suffix})"


__all__ = ["extract_abcd_matrices", "format_array_literal", "parse_gnn_matrix_value"]
