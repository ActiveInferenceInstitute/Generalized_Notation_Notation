"""Shared axis declarations for stored and canonical POMDP transitions.

The resolver changes no values and uses only the standard library, so
headless extraction and numerical consumers can use the same contract.
"""

from collections.abc import Mapping, Sequence
from itertools import product
from typing import Any

CANONICAL = "next_state_previous_state_action"
ACTION_NEXT_PREVIOUS = "action_next_state_previous_state"
ACTION_PREVIOUS_NEXT = "action_previous_state_next_state"

_ALIASES = {
    CANONICAL: CANONICAL,
    "next_previous_action": CANONICAL,
    "next_prev_action": CANONICAL,
    "states_next_states_previous_actions": CANONICAL,
    ACTION_NEXT_PREVIOUS: ACTION_NEXT_PREVIOUS,
    "action_next_previous": ACTION_NEXT_PREVIOUS,
    "actions_next_previous": ACTION_NEXT_PREVIOUS,
    ACTION_PREVIOUS_NEXT: ACTION_PREVIOUS_NEXT,
    "action_previous_next": ACTION_PREVIOUS_NEXT,
    "actions_previous_next": ACTION_PREVIOUS_NEXT,
}

_AXES = {
    CANONICAL: ("next_state", "previous_state", "action"),
    ACTION_NEXT_PREVIOUS: ("action", "next_state", "previous_state"),
    ACTION_PREVIOUS_NEXT: ("action", "previous_state", "next_state"),
}

_PERMUTATIONS = {
    CANONICAL: (0, 1, 2),
    ACTION_NEXT_PREVIOUS: (1, 2, 0),
    ACTION_PREVIOUS_NEXT: (2, 1, 0),
}

_SLICE_ORDERS = {
    "rows_previous_columns_next": ACTION_PREVIOUS_NEXT,
    "rows_next_columns_previous": ACTION_NEXT_PREVIOUS,
}


def parse_b_order(raw: object) -> str:
    """Resolve a string alias or a provenance axis sequence."""
    if isinstance(raw, (list, tuple)):
        raw = "_".join(str(axis) for axis in raw)
    value = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
    if value and value not in _ALIASES:
        raise ValueError(f"Unsupported declared B tensor order: {value}")
    return _ALIASES.get(value, "")


def declared_b_order(parameters: Mapping[str, Any], *, name: str = "B") -> str:
    """Return the declared order, allowing a specific B family member first."""
    aliases = ("b_tensor_order", "B_tensor_order", "transition_tensor_order")
    suffix = name[1:]
    if suffix:
        for key in aliases:
            value = parameters.get(key + suffix)
            if value is not None and value != "":
                return parse_b_order(value)
    for key in aliases:
        value = parameters.get(key)
        if value is not None and value != "":
            return parse_b_order(value)
    return ""


def b_axes(order: str) -> tuple[str, str, str]:
    return _AXES[order]


def b_permutation(order: str) -> tuple[int, int, int]:
    """Index the stored axes in canonical (next, previous, action) order."""
    return _PERMUTATIONS[order]


def stored_b_order(declared: str, provenance: Mapping[str, Any]) -> str:
    """Use canonical custody markers before interpreting raw declarations."""
    if provenance.get("source_order") and (
        provenance.get("canonical_order") == CANONICAL
        or provenance.get("stored_order") == CANONICAL
    ):
        return CANONICAL
    if provenance.get("stored_order"):
        return parse_b_order(provenance["stored_order"])
    if declared:
        return declared
    if provenance.get("declared_order_explicit"):
        return parse_b_order(provenance.get("declared_order"))
    claimed = provenance.get("claimed_slice_convention")
    return parse_b_order(provenance.get("detected_order")) or (
        _SLICE_ORDERS.get(claimed, "") if isinstance(claimed, str) else ""
    )


def b_shape_orders(
    shape: Sequence[int], *, states: int | None = None, actions: int | None = None
) -> list[str]:
    """Layouts compatible with the complete shape and available dimensions."""
    if len(shape) != 3 or not all(shape):
        return []
    orders = []
    for order, (next_axis, previous_axis, action_axis) in _PERMUTATIONS.items():
        n, p, a = shape[next_axis], shape[previous_axis], shape[action_axis]
        if (
            n == p
            and (states is None or n == states)
            and (actions is None or a in (1, actions))
        ):
            orders.append(order)
    return orders


def stochastic_b_orders(
    value: Any, shape: Sequence[int], orders: Sequence[str], *, atol: float = 1e-6
) -> list[str]:
    """Return candidate orders with unit mass over their next-state axis."""
    result = []
    for order in orders:
        axis = b_permutation(order)[0]
        other = [range(size) for i, size in enumerate(shape) if i != axis]
        for rest in product(*other):
            total = 0.0
            for k in range(shape[axis]):
                index = rest[:axis] + (k,) + rest[axis:]
                total += float(value[index[0]][index[1]][index[2]])
            if not abs(total - 1.0) <= atol:
                break
        else:
            result.append(order)
    return result


def resolve_b_order(
    value: Any,
    shape: Sequence[int],
    *,
    declared: str = "",
    provenance: Mapping[str, Any] | None = None,
    states: int | None = None,
    actions: int | None = None,
    default: str = ACTION_PREVIOUS_NEXT,
) -> str:
    """Choose an explicit order, unique stochastic evidence, or a legacy fallback."""
    metadata = provenance or {}
    order = stored_b_order(declared, metadata)
    bound = (
        declared
        or metadata.get("stored_order")
        or (
            metadata.get("source_order")
            and metadata.get("canonical_order") == CANONICAL
        )
    )
    if order and bound:
        if order not in b_shape_orders(shape, states=states):
            raise ValueError(f"B tensor shape {tuple(shape)} does not match {order}")
        action_count = shape[b_permutation(order)[2]]
        if actions is not None and action_count not in (1, actions):
            raise ValueError(
                f"Canonical B action dimension {action_count} does not match {actions}"
            )
        return order
    candidates = b_shape_orders(shape, states=states, actions=actions)
    if not candidates:
        raise ValueError(f"B tensor shape {tuple(shape)} has no supported axis order")
    stochastic = stochastic_b_orders(value, shape, candidates)
    possible = stochastic or candidates
    if order in possible:
        return order
    return default if default in possible else possible[0]
