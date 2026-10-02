"""Shared invocation preconditions before probes, cache reads and dispatch."""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

from gnn.utils.runtime_safety.framework_availability import FrameworkStatus

from .subprocess_envelope import CancelToken, _resolve_timeout

_SCRIPT_DEADLINE: ContextVar[float | None] = ContextVar(
    "gnn_script_deadline", default=None
)


def current_script_deadline() -> float | None:
    """Return the ceiling of this thread's synchronous execution scope."""
    return _SCRIPT_DEADLINE.get()


@contextmanager
def script_execution_scope(timeout: float | None) -> Iterator[float]:
    """Scope a deadline to nested calls and restore it even when they raise.

    Ordinary new threads and remote/process workers must receive explicit
    deadline data; this scope supplies no implicit cross-worker propagation.
    """
    deadline = script_deadline(timeout)
    token = _SCRIPT_DEADLINE.set(deadline)
    try:
        yield deadline
    finally:
        _SCRIPT_DEADLINE.reset(token)


def execution_precondition(
    timeout: float | None,
    cancel_token: CancelToken | None,
    *,
    deadline_monotonic: float | None = None,
) -> dict[str, Any] | None:
    """Reject cancelled, invalid or exhausted work before any cache lookup."""
    envelope: dict[str, Any] = {
        "success": False,
        "return_code": -1,
        "stdout": "",
        "stderr": "",
        "duration_seconds": 0.0,
        "cancelled": False,
        "containment": "not_started",
        "cleanup_verified": None,
        "streams_drained": None,
    }
    if cancel_token is not None and cancel_token.cancelled:
        envelope.update(
            cancelled=True,
            status="cancelled",
            error_type="Cancelled",
            error=cancel_token.reason or "Execution cancelled",
        )
        return envelope
    try:
        _resolve_timeout(timeout)
    except (TypeError, ValueError) as error:
        envelope.update(
            status="failed", error_type="InvalidExecutionTimeout", error=str(error)
        )
        return envelope
    if deadline_monotonic is not None:
        if (
            isinstance(deadline_monotonic, bool)
            or not isinstance(deadline_monotonic, (int, float))
            or not math.isfinite(deadline_monotonic)
        ):
            envelope.update(
                status="failed",
                error_type="InvalidExecutionDeadline",
                error="Execution deadline must be finite and numeric",
            )
            return envelope
        if deadline_monotonic <= time.monotonic():
            envelope.update(
                status="timed_out",
                error_type="TimeoutExpired",
                error="Script deadline exhausted",
            )
            return envelope
    scoped_deadline = current_script_deadline()
    if scoped_deadline is not None and scoped_deadline <= time.monotonic():
        envelope.update(
            status="timed_out",
            error_type="TimeoutExpired",
            error="Scoped execution deadline exhausted",
        )
        return envelope
    from gnn.pipeline.run_context import current_run_context

    context = current_run_context()
    if context is not None and context.remaining_seconds() == 0:
        envelope.update(
            status="timed_out",
            error_type="TimeoutExpired",
            error="Invocation deadline exhausted",
        )
        return envelope
    return None


def unavailable_framework_result(diagnosis: FrameworkStatus) -> dict[str, Any]:
    """Preserve probe uncertainty and cleanup failure as unsuccessful work."""
    skipped = diagnosis.reason_code in {
        "missing_module",
        "missing_toolchain",
        "executor_unavailable",
        "unsupported_python",
        "unsupported_version",
    }
    cleanup_failed = (
        diagnosis.cleanup_verified is False or diagnosis.streams_drained is False
    )
    cancelled = (
        diagnosis.reason_code == "probe_cancelled"
        or diagnosis.execution_error_type == "Cancelled"
    )
    return {
        "success": False,
        "skipped": skipped and not cleanup_failed and not cancelled,
        "cancelled": cancelled,
        "status": "failed"
        if cleanup_failed
        else "cancelled"
        if cancelled
        else "skipped"
        if skipped
        else "timed_out"
        if diagnosis.reason_code == "probe_timeout"
        else "failed",
        "error_type": "ProcessCleanupFailure"
        if cleanup_failed
        else "Cancelled"
        if cancelled
        else diagnosis.execution_error_type
        or ("DependencyUnavailable" if skipped else "FrameworkProbeFailure"),
        "execution_error_type": diagnosis.execution_error_type,
        "error": diagnosis.reason,
        "reason": diagnosis.reason,
        "reason_code": diagnosis.reason_code,
        "install_hint": diagnosis.install_hint,
        "cleanup_verified": diagnosis.cleanup_verified,
        "streams_drained": diagnosis.streams_drained,
    }


def script_deadline(
    timeout: float | None, deadline_monotonic: float | None = None
) -> float:
    """One local ceiling shared by readiness, dispatch and process cleanup."""
    local_deadline = time.monotonic() + _resolve_timeout(timeout)
    scoped_deadline = current_script_deadline()
    if scoped_deadline is not None:
        local_deadline = min(local_deadline, scoped_deadline)
    from gnn.pipeline.run_context import current_run_context

    context = current_run_context()
    if context is not None and context.deadline_monotonic is not None:
        local_deadline = min(local_deadline, context.deadline_monotonic)
    if deadline_monotonic is None:
        return local_deadline
    if (
        isinstance(deadline_monotonic, bool)
        or not isinstance(deadline_monotonic, (int, float))
        or not math.isfinite(deadline_monotonic)
    ):
        raise ValueError("Execution deadline must be finite and numeric")
    return min(local_deadline, deadline_monotonic)


def script_readiness(
    framework: str, timeout: float | None
) -> tuple[float | None, dict[str, Any] | None]:
    """Probe a standalone Python runner without importing optional modules here."""
    precondition = execution_precondition(timeout, None)
    if precondition is not None:
        return None, precondition
    from gnn.utils.runtime_safety.framework_availability import (
        DEFAULT_PROBE_TIMEOUT_SECONDS,
        check_framework,
    )

    deadline = script_deadline(timeout)
    diagnosis = check_framework(
        framework,
        timeout=min(DEFAULT_PROBE_TIMEOUT_SECONDS, _resolve_timeout(timeout)),
        deadline_monotonic=deadline,
    )
    if not diagnosis.available:
        return deadline, unavailable_framework_result(diagnosis)
    if deadline <= time.monotonic():
        return deadline, {
            "success": False,
            "status": "timed_out",
            "error_type": "TimeoutExpired",
            "error": "Script deadline exhausted after framework probe",
        }
    return deadline, None
