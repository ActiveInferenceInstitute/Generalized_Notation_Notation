"""Conservative admission before allocating an exhaustive policy search."""

from __future__ import annotations

MAX_POLICIES = 4096
MAX_STEP_EVALUATIONS = 32768
MAX_ESTIMATED_BYTES = 128 * 1024 * 1024


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def admit_search(
    n: int,
    m: int,
    p: int,
    timesteps: int,
    horizon: int,
    controlled: bool,
    max_policies: int = MAX_POLICIES,
    max_step_evaluations: int = MAX_STEP_EVALUATIONS,
    max_estimated_bytes: int = MAX_ESTIMATED_BYTES,
) -> dict:
    """Validate and estimate search allocation without computing an unbounded power.

    This is an estimate of algorithm allocation, excluding runtime/JIT overhead.
    A process watchdog remains authoritative for execution containment.
    """
    for name, value in (
        ("n", n),
        ("m", m),
        ("p", p),
        ("timesteps", timesteps),
        ("horizon", horizon),
    ):
        _positive_int(value, name)
    for name, value, ceiling in (
        ("max_policies", max_policies, MAX_POLICIES),
        ("max_step_evaluations", max_step_evaluations, MAX_STEP_EVALUATIONS),
        ("max_estimated_bytes", max_estimated_bytes, MAX_ESTIMATED_BYTES),
    ):
        _positive_int(value, name)
        if value > ceiling:
            raise ValueError(
                f"{name} exceeds the maintained admission ceiling {ceiling}"
            )
    action_count = 9 if p == 2 else 2 * p + 1
    policies = 0
    if controlled:
        policies = 1
        for _ in range(horizon):
            if policies > max_policies // action_count:
                raise ValueError("cpomdp policy enumeration exceeds max_policies")
            policies *= action_count
    step_evaluations = policies * horizon
    if step_evaluations > max_step_evaluations:
        raise ValueError("cpomdp search exceeds max_step_evaluations")
    estimate = 64 * policies * (
        horizon * (p + 1) + n * n + m * m + 4 * (n + m) + 16
    ) + 64 * timesteps * (policies + n * n + 3 * n + m + 8)
    if estimate > max_estimated_bytes:
        raise ValueError("cpomdp estimated allocation exceeds max_estimated_bytes")
    return {
        "policy_count": policies,
        "step_evaluations_per_cycle": step_evaluations,
        "action_count": action_count if controlled else 0,
        "estimated_bytes": estimate,
        "estimate_kind": "conservative algorithm allocation estimate; excludes runtime/JIT overhead",
        "limits": {
            "policies": max_policies,
            "step_evaluations": max_step_evaluations,
            "estimated_bytes": max_estimated_bytes,
        },
    }
