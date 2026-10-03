#!/usr/bin/env python3
"""
POMDP math helpers for the render module.

Numeric normalization helpers (probability vectors, column-stochastic
matrices) and Kronecker factored-spec detection / joint-action decoding used
by ``pomdp_processor`` when composing factored POMDPs into joint PyMDP-style
matrices.
"""

import re
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np

if TYPE_CHECKING:
    from gnn.extract.pomdp_extractor import POMDPStateSpace


def _is_kronecker_factorized_spec(pomdp_space: "POMDPStateSpace") -> bool:
    """Detect specs with independent per-factor action spaces (MAJ-02).

    The Kronecker-factorized generator (``scripts/pymdp_spec_generator.py::
    generate_factorized_gnn_file``) declares one action variable ``u_fN`` per
    factor and does **not** surface shared control factors, whereas
    shared-control factored specs (gridworld) and multi-agent specs populate
    ``control_factors``. For the former, the joint action space is the
    *product* of the per-factor action counts and the joint model is the
    Kronecker composition; for the latter, one joint action index is shared
    across all factors.
    Kronecker factoring is discrete-B only — it composes categorical
    per-factor A/B/C/D matrices and never applies to the per-factor
    continuous LGSSM path (``F_fN``/``H_fN``/``Q_fN``/``R_fN``), which
    renders independent linear-Gaussian blocks (``render.continuous_script``).
    """
    matrices = getattr(pomdp_space, "matrices", None) or {}
    b_factor_keys = [key for key in matrices if re.match(r"^B_f\d+$", str(key))]
    if len(b_factor_keys) < 2:
        return False
    if getattr(pomdp_space, "control_factors", None):
        return False
    return True


def _factor_action_counts(matrices: Dict[str, Any], b_keys: List[str]) -> List[int]:
    """Per-factor action counts from action-major ``B_fN`` tensors."""
    counts: List[int] = []
    for key in b_keys:
        raw = np.asarray(matrices[key], dtype=np.float64)
        if raw.ndim == 2:
            counts.append(1)
        elif raw.ndim == 3 and raw.shape[1] == raw.shape[2]:
            counts.append(max(1, int(raw.shape[0])))
        elif raw.ndim == 3 and raw.shape[0] == raw.shape[1]:
            counts.append(max(1, int(raw.shape[2])))
        else:
            counts.append(max(1, int(raw.shape[-1])))
    return counts


def _mixed_radix_digit(action: int, radices: List[int], index: int) -> int:
    """Decode a flat joint action into one factor's action index (LSB-first)."""
    for factor_index, radix in enumerate(radices):
        digit = action % radix
        if factor_index == index:
            return digit
        action //= radix
    return 0


def _normalise_prob_vector(
    values: np.ndarray, *, allow_weights: bool = False
) -> np.ndarray:
    """Validate source probabilities before bounded decimal mass correction."""
    from gnn.render.pomdp_contract import normalise_vector

    return np.asarray(
        normalise_vector(values, name="source prior", allow_weights=allow_weights),
        dtype=np.float64,
    )


def _normalise_columns(
    matrix: np.ndarray, *, allow_weights: bool = False
) -> np.ndarray:
    """Preserve source likelihood semantics; invalid columns are errors."""
    from gnn.render.pomdp_contract import normalise_matrix_columns

    return np.asarray(
        normalise_matrix_columns(
            matrix, name="source likelihood", allow_weights=allow_weights
        ),
        dtype=np.float64,
    )
