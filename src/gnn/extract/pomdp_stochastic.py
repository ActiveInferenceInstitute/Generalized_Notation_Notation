#!/usr/bin/env python3
"""
Strict probability and dimension diagnostics for extracted discrete POMDPs.

``stochastic_diagnostics`` inspects the as-written Active Inference tensors of
a discrete :class:`~gnn.extract.pomdp_state.POMDPStateSpace` and returns one
:class:`StochasticDiagnostic` per violated contract. It never re-orients,
normalizes or otherwise mutates the payload; :class:`POMDPExtractor` decides
whether a diagnostic is raised, collected or only logged.

Contracts (``atol`` defaults to ``1e-6``):

- ``A`` / ``A_*`` likelihoods: every entry is finite and non-negative, and the
  sum over the leading observation axis is ``1`` for every conditioning index
  (``A[obs, state, ...]``; factorized slices follow the same rule).
- ``B`` transitions: finite non-negative entries; the sum over the declared
  next-state axis is ``1`` for every (previous state, action) pair. Without a
  machine-readable order, any compatible supported layout with unit mass is
  admitted. Unspecified higher-rank factored transitions retain their
  square-state-axis diagnostic.
- ``D`` / ``D_*`` priors: finite non-negative probability vectors summing to
  ``1``. ``D`` additionally matches ``num_states``.
- ``C`` / ``C_*`` preferences: finite entries (log-preferences may be
  negative). ``C`` additionally matches ``num_observations`` on its leading
  axis.

Dimension violations use ``GNN-E002`` (shape contradiction); probability
violations use ``GNN-E007`` (non-stochastic, negative or non-finite tensor).

This module is stdlib-only at import time.
"""

import math
import re
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from .pomdp_axes import b_permutation, b_shape_orders, declared_b_order, parse_b_order
from .pomdp_state import POMDPStateSpace, _shape_of

DEFAULT_ATOL = 1e-6
DIMENSION_CODE = "GNN-E002"
PROBABILITY_CODE = "GNN-E007"
_PHASE_TRANSITION = re.compile(r"^B_(?:t|regime)(?:\d*|_\w+)$", re.IGNORECASE)

__all__ = [
    "DEFAULT_ATOL",
    "DIMENSION_CODE",
    "PROBABILITY_CODE",
    "StochasticDiagnostic",
    "stochastic_diagnostics",
]


@dataclass(frozen=True)
class StochasticDiagnostic:
    """One violated probability or dimension contract."""

    code: str
    tensor: str
    message: str


def _family(key: str) -> Optional[str]:
    """Return 'A'/'B'/'C'/'D' for ``X`` or ``X_<suffix>`` keys, else None."""
    for prefix in ("A", "B", "C", "D"):
        if key == prefix or key.startswith(f"{prefix}_"):
            return prefix
    return None


def _flatten(value: Any) -> Iterator[Any]:
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten(item)
    else:
        yield value


def _entry_fault(value: Any) -> Optional[str]:
    """Describe the first non-numeric, non-finite or ragged entry fault."""
    for entry in _flatten(value):
        if isinstance(entry, bool) or not isinstance(entry, (int, float)):
            return f"non-numeric entry {entry!r}"
        try:
            finite = math.isfinite(float(entry))
        except OverflowError:
            finite = False
        if not finite:
            return f"non-finite entry {entry!r}"
    return None


def _negative_entry(value: Any) -> Optional[float]:
    for entry in _flatten(value):
        if float(entry) < 0.0:
            return float(entry)
    return None


def _is_rectangular(value: Any, shape: Sequence[int]) -> bool:
    """True when every nesting level matches the inferred shape."""
    if not shape:
        return not isinstance(value, (list, tuple))
    if not isinstance(value, (list, tuple)) or len(value) != shape[0]:
        return False
    return all(_is_rectangular(item, shape[1:]) for item in value)


def _at(value: Any, index: Tuple[int, ...]) -> float:
    for position in index:
        value = value[position]
    return float(value)


def _worst_axis_sum(
    value: Any, shape: Sequence[int], axis: int, atol: float
) -> Optional[Tuple[Tuple[int, ...], float]]:
    """First conditioning index whose sum over ``axis`` is not 1 within atol."""
    others = [range(size) for position, size in enumerate(shape) if position != axis]
    for rest in product(*others):
        total = 0.0
        for k in range(shape[axis]):
            index = rest[:axis] + (k,) + rest[axis:]
            total += _at(value, index)
        if abs(total - 1.0) > atol:
            return tuple(rest), total
    return None


def _b_next_axes(key: str, shape: Sequence[int], space: POMDPStateSpace) -> List[int]:
    """Candidate next-state axes for a transition tensor."""
    if len(shape) < 3 or (key == "B" and len(shape) != 3):
        return [0]
    phase_offset = 1 if len(shape) == 4 and _PHASE_TRANSITION.fullmatch(key) else 0
    if len(shape) == 3 or phase_offset:
        provenance = (space.matrix_provenance or {}).get(key) or {}
        order = (
            parse_b_order(provenance.get("stored_order"))
            or declared_b_order(space.model_parameters or {}, name=key)
            or parse_b_order(provenance.get("detected_order"))
        )
        candidates = b_shape_orders(shape[phase_offset:])
        if order:
            if order not in candidates:
                raise ValueError(f"tensor shape {list(shape)} does not match {order}")
            return [phase_offset + b_permutation(order)[0]]
        if candidates:
            return [phase_offset + b_permutation(order)[0] for order in candidates]
        if phase_offset:
            raise ValueError(
                f"phase tensor shape {list(shape)} has no supported layout"
            )
    paired = sorted(
        {
            axis
            for axis in range(len(shape))
            for other in range(len(shape))
            if axis != other and shape[axis] == shape[other]
        }
    )
    return paired or list(range(len(shape)))


def _check_tensor(
    key: str,
    family: str,
    value: Any,
    space: POMDPStateSpace,
    atol: float,
) -> List[StochasticDiagnostic]:
    diagnostics: List[StochasticDiagnostic] = []

    def add(code: str, message: str) -> None:
        diagnostics.append(StochasticDiagnostic(code, key, f"{key}: {message}"))

    if not isinstance(value, (list, tuple)):
        if key == family:
            add(DIMENSION_CODE, "expected a numeric tensor or vector")
        return diagnostics
    shape = _shape_of(value)
    if not shape or 0 in shape or not _is_rectangular(value, shape):
        add(DIMENSION_CODE, f"ragged or empty tensor with inferred shape {shape}")
        return diagnostics

    fault = _entry_fault(value)
    if fault is not None:
        add(PROBABILITY_CODE, fault)
        return diagnostics

    if key == "C" and space.num_observations and shape[0] != space.num_observations:
        add(
            DIMENSION_CODE,
            f"length {shape[0]} does not match num_observations "
            f"{space.num_observations}",
        )
    if key == "D" and space.num_states and shape[0] != space.num_states:
        add(
            DIMENSION_CODE,
            f"length {shape[0]} does not match num_states {space.num_states}",
        )
    if family == "C":
        return diagnostics  # log-preferences: any finite real is admissible

    negative = _negative_entry(value)
    if negative is not None:
        add(PROBABILITY_CODE, f"negative probability entry {negative!r}")
        return diagnostics

    if family == "D":
        rows = [value] if len(shape) == 1 else list(value)
        for row_index, row in enumerate(rows):
            total = sum(float(entry) for entry in _flatten(row))
            if abs(total - 1.0) > atol:
                where = "" if len(shape) == 1 else f" row {row_index}"
                add(
                    PROBABILITY_CODE,
                    f"prior{where} sums to {total:.6g}, expected 1 (atol={atol:g})",
                )
                break
        return diagnostics

    if family == "A":
        if len(shape) < 2:
            add(
                DIMENSION_CODE,
                f"likelihood must have an observation and a state axis, got {shape}",
            )
            return diagnostics
        worst = _worst_axis_sum(value, shape, 0, atol)
        if worst is not None:
            index, total = worst
            add(
                PROBABILITY_CODE,
                f"column {list(index)} sums to {total:.6g} over the observation "
                f"axis, expected 1 (atol={atol:g})",
            )
        return diagnostics

    # family == "B"
    try:
        candidates = _b_next_axes(key, shape, space)
    except ValueError as error:
        add(DIMENSION_CODE, str(error))
        return diagnostics
    failures: Dict[int, Tuple[Tuple[int, ...], float]] = {}
    for axis in candidates:
        worst = _worst_axis_sum(value, shape, axis, atol)
        if worst is None:
            return diagnostics
        failures[axis] = worst
    axis = candidates[0]
    index, total = failures[axis]
    axes_text = (
        f"next-state axis {axis}"
        if len(candidates) == 1
        else f"any candidate next-state axis {candidates} (axis {axis} shown)"
    )
    message = (
        f"conditional {list(index)} sums to {total:.6g} over {axes_text}, "
        f"expected 1 (atol={atol:g})"
    )
    unsupported = [
        other
        for other in range(len(shape))
        if other not in candidates
        and _worst_axis_sum(value, shape, other, atol) is None
    ]
    if unsupported:
        message += (
            f"; axis {unsupported[0]} is stochastic, but that stored layout "
            "matches no supported next-state order"
        )
    add(PROBABILITY_CODE, message)
    return diagnostics


def stochastic_diagnostics(
    space: POMDPStateSpace, *, atol: float = DEFAULT_ATOL
) -> List[StochasticDiagnostic]:
    """Return every probability/dimension violation in a discrete payload.

    Continuous (linear-Gaussian) payloads carry no categorical tensors and
    always return an empty list. Keys are visited in payload order so the
    first diagnostic is deterministic.
    """
    if space.model_kind != "discrete":
        return []
    tensors: Dict[str, Any] = dict(space.matrices or {})
    for key, attribute in (
        ("A", space.A_matrix),
        ("B", space.B_matrix),
        ("C", space.C_vector),
        ("D", space.D_vector),
    ):
        if attribute is not None:
            tensors.setdefault(key, attribute)
    diagnostics: List[StochasticDiagnostic] = []
    for key, value in tensors.items():
        family = _family(key)
        if family is None:
            continue
        diagnostics.extend(_check_tensor(key, family, value, space, atol))
    return diagnostics
