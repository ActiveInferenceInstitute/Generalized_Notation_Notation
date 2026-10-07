"""Resolve supervised work and cleanup inside an optional absolute deadline."""

from __future__ import annotations

import math
import time


def resolve_process_deadlines(
    local_timeout: float | None,
    *,
    deadline_monotonic: float | None = None,
    cleanup_seconds: float = 1.0,
) -> tuple[float | None, float, float | None]:
    """Return work deadline, reserved cleanup seconds, and cleanup ceiling.

    An absolute invocation/request deadline includes cleanup. Reserve at most
    one cleanup allowance or a quarter of its remaining budget, so a short
    deadline still permits supervised work. Standalone local timeouts retain
    their separate bounded cleanup allowance.
    """
    if local_timeout is not None and (
        isinstance(local_timeout, bool)
        or not isinstance(local_timeout, (int, float))
        or not math.isfinite(local_timeout)
        or local_timeout < 0
    ):
        raise ValueError("Process timeout must be finite and nonnegative or null")
    if (
        isinstance(cleanup_seconds, bool)
        or not isinstance(cleanup_seconds, (int, float))
        or not math.isfinite(cleanup_seconds)
        or cleanup_seconds < 0
    ):
        raise ValueError("Cleanup allowance must be finite and nonnegative")
    if deadline_monotonic is not None and (
        isinstance(deadline_monotonic, bool)
        or not isinstance(deadline_monotonic, (int, float))
        or not math.isfinite(deadline_monotonic)
    ):
        raise ValueError("Absolute process deadline must be finite or null")
    now = time.monotonic()
    work_deadline = None if local_timeout is None else now + local_timeout
    allowance = cleanup_seconds
    if deadline_monotonic is not None:
        allowance = min(cleanup_seconds, max(0.0, deadline_monotonic - now) * 0.25)
        work_ceiling = deadline_monotonic - allowance
        work_deadline = (
            work_ceiling if work_deadline is None else min(work_deadline, work_ceiling)
        )
    return work_deadline, allowance, deadline_monotonic
