"""Strict source-declared finite controls; no inference dependency imports."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

from .admission import MAX_ESTIMATED_BYTES, MAX_POLICIES, MAX_STEP_EVALUATIONS

FINITE_CONTROL_CONTRACT = "gnn-cpomdp-finite-actions-v1"
CANDIDATE_SEARCH_CONTRACT = "gnn-cpomdp-candidate-sequences-v1"


def _label(value: object, source: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 128:
        raise ValueError(
            f"{source} labels must be nonempty strings of at most 128 characters"
        )
    return value


def finite_controls(raw: object, n: int) -> dict[str, Any]:
    """Validate ordered actions and an explicit state-by-control linear map.

    Duplicate vectors retain distinct action IDs and authored tie order. Units
    are declarations, not a dimensional-analysis or extraction proof.
    """
    keys = {
        "contract",
        "version",
        "action_ids",
        "control_units",
        "state_units",
        "actions",
        "control_matrix",
    }
    if not isinstance(raw, dict) or set(raw) != keys:
        raise ValueError(f"cpomdp_control requires exactly {sorted(keys)}")
    if raw["contract"] != FINITE_CONTROL_CONTRACT:
        raise ValueError("unsupported cpomdp_control contract")

    version = _label(raw["version"], "cpomdp_control")
    actions, matrix = raw["actions"], raw["control_matrix"]
    if not isinstance(actions, (list, tuple)) or not 2 <= len(actions) <= MAX_POLICIES:
        raise ValueError("cpomdp_control requires 2..4096 ordered actions")
    if not isinstance(matrix, (list, tuple)) or len(matrix) != n:
        raise ValueError("cpomdp_control control_matrix must have n state rows")
    if not isinstance(matrix[0], (list, tuple)) or not matrix[0]:
        raise ValueError("cpomdp_control control_matrix needs a nonempty control axis")
    p = len(matrix[0])
    if 64 * (len(actions) * p + n * p) > MAX_ESTIMATED_BYTES:
        raise ValueError(
            "cpomdp_control declared allocation exceeds max_estimated_bytes"
        )

    def labels(value: object, count: int) -> list[str]:
        if not isinstance(value, (list, tuple)) or len(value) != count:
            raise ValueError(
                "cpomdp_control label dimensions must match their declared axes"
            )
        return [_label(v, "cpomdp_control") for v in value]

    ids = labels(raw["action_ids"], len(actions))
    if len(set(ids)) != len(ids):
        raise ValueError("cpomdp_control action_ids must be unique")

    def rows(value: Any) -> list[list[float]]:
        result = []
        for row in value:
            if not isinstance(row, (list, tuple)) or len(row) != p:
                raise ValueError("cpomdp_control rows must match the control dimension")
            converted = []
            for v in row:
                if isinstance(v, bool) or not isinstance(v, Real):
                    raise ValueError(
                        "cpomdp_control values must be finite nonboolean numbers"
                    )
                try:
                    number = float(v)
                except OverflowError as exc:
                    raise ValueError(
                        "cpomdp_control values must be finite nonboolean numbers"
                    ) from exc
                if not math.isfinite(number):
                    raise ValueError(
                        "cpomdp_control values must be finite nonboolean numbers"
                    )
                converted.append(number)
            result.append(converted)
        return result

    return {
        "contract": FINITE_CONTROL_CONTRACT,
        "version": version,
        "action_ids": ids,
        "control_units": labels(raw["control_units"], p),
        "state_units": labels(raw["state_units"], n),
        "actions": rows(actions),
        "control_matrix": rows(matrix),
    }


def candidate_search(raw: object, action_ids: list[str]) -> dict[str, Any]:
    """Validate a bounded source-authored list of open-loop action sequences."""
    keys = {"contract", "version", "horizon", "policies"}
    if not isinstance(raw, dict) or set(raw) != keys:
        raise ValueError(f"cpomdp_search requires exactly {sorted(keys)}")
    if raw["contract"] != CANDIDATE_SEARCH_CONTRACT:
        raise ValueError("unsupported cpomdp_search contract")
    version = _label(raw["version"], "cpomdp_search")
    horizon, policies = raw["horizon"], raw["policies"]
    if type(horizon) is not int or horizon < 1:
        raise ValueError("cpomdp_search horizon must be a positive integer")
    if (
        not isinstance(policies, (list, tuple))
        or not 1 <= len(policies) <= MAX_POLICIES
    ):
        raise ValueError("cpomdp_search candidate count must be within max_policies")
    if horizon > MAX_STEP_EVALUATIONS // len(policies):
        raise ValueError("cpomdp_search exceeds max_step_evaluations")

    valid_actions = set(action_ids)
    sequences: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for policy in policies:
        if not isinstance(policy, (list, tuple)) or len(policy) != horizon:
            raise ValueError("cpomdp_search policies must match the declared horizon")
        if any(
            not isinstance(action, str) or action not in valid_actions
            for action in policy
        ):
            raise ValueError(
                "cpomdp_search policies must reference declared action_ids"
            )
        key = tuple(policy)
        if key in seen:
            raise ValueError(
                "cpomdp_search policies must be unique action-ID sequences"
            )
        seen.add(key)
        sequences.append(list(key))

    return {
        "contract": CANDIDATE_SEARCH_CONTRACT,
        "version": version,
        "horizon": horizon,
        "policies": sequences,
    }
