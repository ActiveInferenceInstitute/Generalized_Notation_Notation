#!/usr/bin/env python3
"""
Uniform subprocess execution envelope for GNN execution backends.

Every GNN execution path (script runners, MCP executors, tool
probes) converts ``subprocess`` outcomes into the same structured
envelope so callers never have to differentiate between a timeout, an
``OSError``, a cancellation, and a non-zero exit code.

The child leads a fresh process group and the poll loop watches a
:class:`CancelToken` plus the resolved deadline in 0.25s slices. Cleanup
terminates the process group and descendants observed while the leader
was alive, including observed children that create a separate session.
Termination, reaping, and stream draining share one additional second.
The receipt records whether this cleanup was verified; incomplete
observation or pipes that remain open produce a failure. This boundary
does not establish containment of a hostile child that detaches before
observation; that requires an operating-system sandbox.
A ``timeout=None`` run is still bounded: ``_resolve_timeout`` applies
``DEFAULT_TIMEOUT_SECONDS``, overridable per process via
``GNN_EXECUTE_DEFAULT_TIMEOUT``.

Extracted from ``execute.executor`` (the canonical
``execute_script_safely`` envelope) into a leaf module so the
per-framework runners can share one implementation without importing the
executor module (which imports the runner modules at module scope).
"""

from __future__ import annotations

import logging
import math
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Union, cast

from gnn.utils.runtime_safety.process_budget import resolve_process_deadlines
from gnn.utils.runtime_safety.process_tree import (
    DescendantTracker,
    terminate_process_tree,
)

logger = logging.getLogger(__name__)

# Sentinel exit codes shared across GNN execution surfaces (this envelope,
# the step-12 processor, per-framework runners). A sentinel alone is
# ambiguous — always pair it with ``error_type`` when interpreting failures.
NEVER_STARTED = -1  # the process never ran: OSError or caller-side timeout
INTERNAL_ERROR = -2  # the harness itself failed while orchestrating the run
UNKNOWN_STATE = -3  # no execution record exists at all

# Default wall-clock bound applied when a caller passes ``timeout=None``.
# Overridable per process via ``GNN_EXECUTE_DEFAULT_TIMEOUT`` (positive int).
DEFAULT_TIMEOUT_SECONDS: int = 3600

# Cancellation/deadline poll cadence; never check faster than this.
_POLL_INTERVAL_SECONDS = 0.25
_CLEANUP_TIMEOUT_SECONDS = 1.0

# Invalid ``GNN_EXECUTE_DEFAULT_TIMEOUT`` values that already warned (warn-once).
_INVALID_DEFAULT_TIMEOUTS_WARNED: set[str] = set()


# psutil is a core dependency, but the envelope's result contract must not
# depend on it: sample the child tree when importable, degrade to explicit
# null keys when not (execute/validator.py optional-import precedent).
try:
    import psutil as _psutil_module
except ImportError:  # pragma: no cover - psutil is core; belt-and-braces
    _psutil_module = cast(Any, None)


class CancelToken:
    """Thread-safe cooperative cancellation flag for in-flight subprocess runs.

    Pass a token to :func:`run_subprocess_envelope` (or the executor helpers
    built on it) and call :meth:`cancel` from any thread: the in-flight run's
    poll loop observes the flag within one poll slice (<=0.25s), then runs
    bounded cleanup and reports ``error_type="Cancelled"`` without raising.
    A cleanup failure takes precedence and retains cancellation evidence.
    ``cancel()`` is idempotent; the first ``reason`` wins.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._reason: Optional[str] = None

    def cancel(self, reason: Optional[str] = None) -> None:
        """Request cancellation (idempotent); the first ``reason`` wins."""
        if not self._event.is_set():
            self._reason = reason
        self._event.set()

    @property
    def cancelled(self) -> bool:
        """True once :meth:`cancel` has been called from any thread."""
        return self._event.is_set()

    @property
    def reason(self) -> Optional[str]:
        """Reason recorded by the first :meth:`cancel` call (may be None)."""
        return self._reason


def _warn_invalid_default_timeout(raw: str) -> None:
    """Warn once per distinct invalid ``GNN_EXECUTE_DEFAULT_TIMEOUT`` value."""
    if raw in _INVALID_DEFAULT_TIMEOUTS_WARNED:
        return
    _INVALID_DEFAULT_TIMEOUTS_WARNED.add(raw)
    logger.warning(
        "GNN_EXECUTE_DEFAULT_TIMEOUT=%r is not a positive int; using "
        "DEFAULT_TIMEOUT_SECONDS=%s instead",
        raw,
        DEFAULT_TIMEOUT_SECONDS,
    )


def _resolve_timeout(timeout: float | None) -> float:
    """Return the effective wall-clock timeout for an envelope run.

    An explicit ``timeout`` must be positive and finite. ``None`` resolves to
    ``GNN_EXECUTE_DEFAULT_TIMEOUT`` when that env var parses to a positive
    int; an unset, invalid, or non-positive value warns once and falls back
    to :data:`DEFAULT_TIMEOUT_SECONDS`.
    """
    if timeout is not None:
        if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Execution timeout must be positive and finite")
        return timeout
    raw = os.environ.get("GNN_EXECUTE_DEFAULT_TIMEOUT")
    if raw is None:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        resolved = int(raw)
    except ValueError:
        _warn_invalid_default_timeout(raw)
        return DEFAULT_TIMEOUT_SECONDS
    if resolved <= 0:
        _warn_invalid_default_timeout(raw)
        return DEFAULT_TIMEOUT_SECONDS
    return resolved


def _as_text(value: Any) -> str:
    """Normalize stream captures to text (subprocess mixes str/bytes)."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


__all__ = [
    "run_subprocess_envelope",
    "CancelToken",
    "DEFAULT_TIMEOUT_SECONDS",
    "NEVER_STARTED",
    "INTERNAL_ERROR",
    "UNKNOWN_STATE",
]


def _sandbox_prefix_and_mode() -> tuple[List[str], str, Optional[str]]:
    """Return ``(prefix, effective_mode, blocked_reason)`` from ``GNN_SANDBOX``.

    Mirrors the Step-12 semantics (``execute.processor._sandbox_mode`` /
    ``_sandbox_command_prefix``): ``off`` runs unsandboxed, ``prefer``
    falls back to unsandboxed execution (with a warning) when no backend
    exists, and ``require`` without a backend yields a non-None
    ``blocked_reason`` so the caller can refuse to execute.
    """
    from gnn.execute.sandbox import _resolve_mode, detect_sandbox

    mode = _resolve_mode(None)
    if mode == "off":
        return [], mode, None
    spec = detect_sandbox()
    if spec is None:
        if mode == "require":
            return (
                [],
                mode,
                (
                    "GNN_SANDBOX=require but no sandbox backend "
                    "(firejail/bwrap/nsjail) is installed"
                ),
            )
        logger.warning(
            "GNN_SANDBOX=%s but no sandbox backend found; running unsandboxed",
            mode,
        )
        return [], mode, None
    return list(spec.prefix), mode, None


def _spawn_kwargs() -> Dict[str, Any]:
    """Popen kwargs making the child the leader of a fresh process group."""
    if hasattr(os, "setsid"):
        return {"start_new_session": True}
    # CREATE_NEW_PROCESS_GROUP exists only on Windows; getattr keeps the
    # attribute access runtime- and mypy-safe on POSIX checkouts.
    windows_process_group = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    return {"creationflags": windows_process_group}


def _kill_process_group(proc: subprocess.Popen[Any]) -> None:
    """Compatibility wrapper for bounded group cleanup without a tracker."""
    terminate_process_tree(proc, cleanup_timeout=_CLEANUP_TIMEOUT_SECONDS)


def _mark_cancelled(envelope: Dict[str, Any], reason: Optional[str]) -> None:
    """Stamp a run-cancelled envelope (partial streams are the caller's job)."""
    envelope["cancelled"] = True
    envelope["success"] = False
    envelope["error"] = (
        "Execution cancelled" if reason is None else f"Execution cancelled: {reason}"
    )
    envelope["error_type"] = "Cancelled"
    envelope["return_code"] = NEVER_STARTED


def _mark_cleanup_failed(envelope: Dict[str, Any], error: object) -> None:
    """Keep cleanup failures fatal while retaining any earlier cleanup evidence."""
    prior = envelope.get("cleanup_error")
    detail = f"{prior}; {error}" if prior else str(error)
    envelope.update(
        success=False,
        cleanup_verified=False,
        cleanup_error=detail,
        error=f"Process cleanup could not be verified: {detail}",
        error_type="ProcessCleanupFailure",
        return_code=INTERNAL_ERROR,
    )


class _ChildRssSampler:
    """Best-effort peak-RSS sampler for the envelope's child process tree.

    One :meth:`sample` per poll slice sums ``memory_info().rss`` over the
    direct child (``pid``) and descendants already observed by its tracker,
    within an advisory 10ms sampling budget. Unobserved children or a large
    observed tree can be omitted; this is a sampled estimate. Every failure
    mode degrades without touching the
    envelope contract: a missing psutil module disables sampling entirely;
    a vanished root process (``NoSuchProcess``/zombie) or access error stops
    further sampling while keeping an already-observed peak; a grandchild
    that exits mid-poll is skipped for that sample. Sampling rides the
    envelope's existing poll cadence (``_POLL_INTERVAL_SECONDS``) — no new
    control flow, no extra wake-ups.

    macOS note: psutil RSS counts shared pages, so the sum estimates
    resident footprint rather than exact peak; receipts must never claim
    exact peak memory from these keys.
    """

    def __init__(
        self,
        pid: int,
        tracker: DescendantTracker | None = None,
        deadline_monotonic: float | None = None,
    ) -> None:
        self._pid = pid
        self._tracker = tracker
        self._deadline_monotonic = deadline_monotonic
        self._peak_rss_bytes = 0.0
        self._samples = 0
        self._stopped = False

    def sample(self) -> None:
        """Take one poll-slice sample; never raises."""
        psutil = _psutil_module
        if psutil is None or self._stopped:
            return
        # Resource sampling is advisory and must not monopolize a short
        # execution budget when an owned worker spawned a large process tree.
        sample_deadline = time.monotonic() + 0.01
        if self._deadline_monotonic is not None:
            sample_deadline = min(sample_deadline, self._deadline_monotonic)
        if time.monotonic() >= sample_deadline:
            return
        try:
            root = psutil.Process(self._pid)
            rss = float(root.memory_info().rss)
        except (psutil.Error, OSError):  # vanished (NoSuchProcess/zombie)/access
            # The tree vanished between polls (or psutil lost access): stop
            # sampling; an already-observed peak survives.
            self._stopped = True
            return
        # Reuse scoped observation; a host-wide psutil.children() walk can
        # block deadline polling for seconds on a busy macOS machine.
        children = self._tracker.observed_processes if self._tracker else []
        for child in children:
            if time.monotonic() >= sample_deadline:
                break
            try:
                if child.is_running():
                    rss += float(child.memory_info().rss)
            except (psutil.Error, OSError):  # grandchild exited mid-poll
                continue
        self._peak_rss_bytes = max(self._peak_rss_bytes, rss)
        self._samples += 1

    @property
    def peak_rss_mb(self) -> Optional[float]:
        """Observed peak MB (2dp), or None when no sample ever succeeded."""
        if self._samples == 0:
            return None
        return round(self._peak_rss_bytes / (1024 * 1024), 2)

    @property
    def samples(self) -> int:
        """Number of successful samples taken for this run."""
        return self._samples


def run_subprocess_envelope(
    command: List[str],
    *,
    timeout: float | None = None,
    env: Optional[Dict[str, str]] = None,
    capture_output: bool = True,
    cwd: Optional[Union[str, Path]] = None,
    input: Optional[str] = None,
    sandbox: bool = True,
    cancel_token: Optional[CancelToken] = None,
    deadline_monotonic: float | None = None,
) -> Dict[str, Any]:
    """Run ``command`` in a fresh process group and return a structured envelope.

    Args:
        command:        Argument vector (no shell).
        timeout:        Wall-clock timeout in seconds. ``None`` resolves via
                        ``_resolve_timeout``: ``GNN_EXECUTE_DEFAULT_TIMEOUT``
                        when it parses to a positive int, else
                        ``DEFAULT_TIMEOUT_SECONDS`` — runs are never
                        unbounded.
        cwd:            Working directory for the subprocess.
        env:            Environment variable overrides, merged over
                        ``os.environ`` (``None`` inherits the parent env).
        capture_output: If True, capture stdout/stderr; otherwise stream to
                        the parent process.
        input:          Text supplied through an owned temporary stdin
                        descriptor; ``None`` leaves stdin attached to the parent.
        sandbox:        Apply the ``GNN_SANDBOX`` env-prefix pattern
                        (Step-12 semantics). ``True`` prefixes the command
                        with the configured sandbox backend's argument
                        vector; ``False`` runs unsandboxed unconditionally.
        cancel_token:   Optional cooperative :class:`CancelToken`; checked
                        before spawning and every poll slice. A cancel
                        (pre-spawn or mid-flight) requests bounded cleanup
                        and reports ``error_type="Cancelled"`` unless cleanup
                        itself fails.
        deadline_monotonic: Optional absolute deadline including cleanup.
                        Intersected with an active invocation deadline. Up to
                        one second (25% of remaining time for short budgets)
                        is reserved for observation, termination, and draining.

    Returns:
        Dict with keys:
            - ``success`` (bool): True iff the process exited with code 0
              and cleanup was verified.
            - ``return_code`` (int): Exit code (``NEVER_STARTED`` if never
              started — OSError, timeout, or cancellation).
            - ``stdout`` (str): Captured stdout (empty when not capturing).
              On timeout/cancel the child's partial stdout survives
              when draining finishes or reaches its bound.
            - ``stderr`` (str): Captured stderr (empty when not capturing).
              On timeout/cancel the child's partial stderr survives
              when draining finishes or reaches its bound.
            - ``duration_seconds`` (float): Wall-clock execution time.
            - ``cancelled`` (bool): Always present; True iff the run was
              cancelled via ``cancel_token`` (pre-spawn or mid-flight).
            - ``child_peak_rss_mb`` (float, optional): Peak sampled summed RSS
              (MB, rounded to 2dp) across the child and observed descendants,
              with a 10ms sampling-work ceiling. Unobserved children or large
              trees can be omitted. ``None`` when
              unmeasurable (psutil unavailable, or the process vanished
              before the first sample). psutil RSS counts shared pages, so
              this is an estimate of resident footprint, never an exact
              peak; receipts must not claim exact peak memory.
            - ``rss_sample_interval_seconds`` (float): The sampling cadence,
              equal to the poll interval (0.25s).
            - ``rss_samples_count`` (int): Successful samples taken; 0 when
              unmeasured.
            - ``sandbox_mode`` (str): Effective ``GNN_SANDBOX`` mode
              (``"off"`` when ``sandbox=False``).
            - ``sandboxed`` (bool): True iff the command was actually
              wrapped with a sandbox backend prefix.
            - ``containment`` (str): ``not_started``, ``process_group_only``,
              or ``observed_descendants``; an observation boundary, not an
              operating-system isolation claim.
            - ``cleanup_verified`` (bool, optional): Whether group and
              observed descendant cleanup was verified; None before launch.
            - ``observed_descendant_count`` (int): Retained process identities;
              zero does not establish that the child created no descendants.
            - ``streams_drained`` (bool, optional): Whether output pipes
              closed within cleanup's shared one-second bound.
            - ``cleanup_error`` (str, optional): Reason cleanup failed;
              ``execution_error_type`` retains an earlier timeout or cancel.
            - ``error`` (str, optional): Populated on failure.
            - ``error_type`` (str, optional): Exception class name on
              failure; ``"SandboxUnavailable"`` when ``GNN_SANDBOX=require``
              found no backend and the run was refused;
              ``"TimeoutExpired"`` on timeout; ``"Cancelled"`` when the
              ``cancel_token`` fired.

    The child runs as the leader of a fresh process group
    (``start_new_session`` on POSIX, ``CREATE_NEW_PROCESS_GROUP`` on
    Windows). Every exit path terminates the group and observed descendants,
    then drains streams within a shared cleanup bound. An absolute caller or
    run-context deadline reserves cleanup inside that total budget. Standalone
    calls without an absolute deadline retain a separate one-second allowance.
    ``KeyboardInterrupt`` runs this cleanup and re-raises. A child that
    detaches before observation requires stronger OS containment; this
    envelope does not claim that guarantee.

    When ``sandbox`` is True and ``GNN_SANDBOX`` is ``off`` (the default),
    a ``sandbox_disabled_receipt`` warning is emitted but the command runs
    unsandboxed. When ``sandbox`` is False, the caller is asserting that it
    manages sandboxing itself — the same receipt is emitted and the env
    prefix is never applied.
    """
    envelope: Dict[str, Any] = {
        "success": False,
        "return_code": NEVER_STARTED,
        "stdout": "",
        "stderr": "",
        "duration_seconds": 0.0,
        "child_peak_rss_mb": None,
        "rss_sample_interval_seconds": _POLL_INTERVAL_SECONDS,
        "rss_samples_count": 0,
        "cancelled": False,
        "sandbox_mode": "off",
        "sandboxed": False,
        "containment": "not_started",
        "cleanup_verified": None,
        "streams_drained": None,
        "observed_descendant_count": 0,
    }

    subject = command[0] if command else "<empty-command>"
    if not sandbox:
        logger.warning(
            "sandbox_disabled_receipt: %s executed WITHOUT a sandbox "
            "(sandbox=False). The command runs with operator privileges; "
            "set GNN_SANDBOX=prefer/require and sandbox=True for isolation.",
            subject,
            extra={
                "event": "sandbox_disabled_receipt",
                "sandbox_mode": "off",
                "command": subject,
            },
        )
    else:
        sandbox_prefix, mode, blocked = _sandbox_prefix_and_mode()
        envelope["sandbox_mode"] = mode
        if blocked is not None:
            # Mirrors the Step-12 processor: refuse to run unsandboxed.
            envelope["error"] = blocked
            envelope["error_type"] = "SandboxUnavailable"
            logger.error(blocked)
            return envelope
        if sandbox_prefix:
            envelope["sandboxed"] = True
            logger.info(
                "sandbox_active_receipt: %s will run under sandbox "
                "isolation (GNN_SANDBOX=%s)",
                subject,
                mode,
                extra={
                    "event": "sandbox_active_receipt",
                    "sandbox_mode": mode,
                },
            )
            command = [*sandbox_prefix, *command]
        else:
            logger.warning(
                "sandbox_disabled_receipt: %s executed WITHOUT a sandbox "
                "(GNN_SANDBOX=off, the default). The command runs with "
                "operator privileges; set GNN_SANDBOX=prefer/require for "
                "isolation.",
                subject,
                extra={
                    "event": "sandbox_disabled_receipt",
                    "sandbox_mode": mode,
                    "command": subject,
                },
            )

    merged_env: Optional[Dict[str, str]] = None
    if env is not None:
        merged_env = dict(os.environ)
        merged_env.update(env)

    start = time.monotonic()
    try:
        resolved_timeout = _resolve_timeout(timeout)
    except (TypeError, ValueError) as exc:
        envelope.update(error=str(exc), error_type="InvalidExecutionTimeout")
        return envelope
    from gnn.pipeline.run_context import current_run_context

    context = current_run_context()
    if deadline_monotonic is not None and (
        isinstance(deadline_monotonic, bool)
        or not isinstance(deadline_monotonic, (int, float))
        or not math.isfinite(deadline_monotonic)
    ):
        envelope.update(
            error="Absolute execution deadline must be finite",
            error_type="InvalidExecutionDeadline",
        )
        return envelope
    from .preconditions import current_script_deadline

    scoped_deadline = current_script_deadline()
    if scoped_deadline is not None:
        deadline_monotonic = (
            scoped_deadline
            if deadline_monotonic is None
            else min(deadline_monotonic, scoped_deadline)
        )
    if context is not None and context.deadline_monotonic is not None:
        deadline_monotonic = (
            context.deadline_monotonic
            if deadline_monotonic is None
            else min(deadline_monotonic, context.deadline_monotonic)
        )
    deadline, cleanup_allowance, cleanup_ceiling = resolve_process_deadlines(
        resolved_timeout,
        deadline_monotonic=deadline_monotonic,
        cleanup_seconds=_CLEANUP_TIMEOUT_SECONDS,
    )
    assert deadline is not None  # a resolved local timeout always supplies one
    total_budget_limited = (
        cleanup_ceiling is not None and deadline >= cleanup_ceiling - cleanup_allowance
    )
    # A large stdin pipe can fill before the child finishes importing. After
    # communicate(input) times out, CPython retries with communicate(None)
    # stop registering unfinished stdin writes; repeating input is forbidden.
    # An owned TemporaryFile descriptor supplies every byte and EOF without
    # tying delivery to the output/deadline polling loop. It is unlinked on
    # POSIX, never passed in argv and closed on every return/failure path.
    stdin_file: BinaryIO | None = None
    stdout_bytes: Optional[bytes] = None
    stderr_bytes: Optional[bytes] = None
    timed_out = False
    proc: Optional[subprocess.Popen[Any]] = None
    rss_sampler: Optional[_ChildRssSampler] = None
    tracker: DescendantTracker | None = None
    try:
        if cancel_token is not None and cancel_token.cancelled:
            # Pre-spawn cancel: identical envelope, no process is spawned.
            _mark_cancelled(envelope, cancel_token.reason)
            return envelope

        if time.monotonic() >= deadline:
            envelope.update(
                error="Execution budget exhausted before launch",
                error_type="TimeoutExpired",
            )
            return envelope

        if input is not None:
            stdin_file = tempfile.TemporaryFile(mode="w+b")
            payload = input.encode("utf-8")
            if stdin_file.write(payload) != len(payload):
                raise OSError("Could not stage complete subprocess input")
            stdin_file.seek(0)
            # Staging belongs to the original budget, including cancellation.
            if cancel_token is not None and cancel_token.cancelled:
                _mark_cancelled(envelope, cancel_token.reason)
                return envelope
            if time.monotonic() >= deadline:
                envelope.update(
                    error="Execution budget exhausted before launch",
                    error_type="TimeoutExpired",
                )
                return envelope

        proc = subprocess.Popen(  # nosec B603 — argument vector, no shell
            command,
            stdin=stdin_file,
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.PIPE if capture_output else None,
            cwd=cwd,
            env=merged_env,
            **_spawn_kwargs(),
        )
        tracker = DescendantTracker(proc.pid)
        tracker.start()
        rss_sampler = _ChildRssSampler(proc.pid, tracker, deadline)
        rss_sampler.sample()  # immediate post-spawn footprint, pre-poll
        while True:
            try:
                stdout_bytes, stderr_bytes = proc.communicate(
                    timeout=min(
                        _POLL_INTERVAL_SECONDS, max(0.001, deadline - time.monotonic())
                    ),
                )
                envelope["return_code"] = proc.returncode
                envelope["success"] = proc.returncode == 0
                break  # process exited; final streams collected
            except subprocess.TimeoutExpired as exc:
                stdout_bytes, stderr_bytes = exc.output, exc.stderr
                if cancel_token is not None and cancel_token.cancelled:
                    _mark_cancelled(envelope, cancel_token.reason)
                    break
                if time.monotonic() >= deadline:
                    timed_out = True
                    break
                rss_sampler.sample()
    except KeyboardInterrupt:
        # The finally block uses the same bounded cleanup before re-raising.
        raise
    except Exception as exc:  # noqa: BLE001 — convert any failure to envelope
        envelope["error"] = str(exc)
        envelope["error_type"] = type(exc).__name__
    finally:
        # Popen duplicated this descriptor; closing our copy cannot interrupt
        # the child's read. Do this before cleanup so even a cleanup exception
        # cannot leave request bytes or an owned file descriptor behind.
        if stdin_file is not None:
            try:
                stdin_file.close()
            except OSError as close_exc:
                _mark_cleanup_failed(envelope, close_exc)
        if proc is not None:
            cleanup_deadline = time.monotonic() + cleanup_allowance
            if cleanup_ceiling is not None:
                cleanup_deadline = min(cleanup_deadline, cleanup_ceiling)
            envelope["cleanup_timeout_seconds"] = cleanup_allowance
            tracked = tracker.observed_processes if tracker else []
            try:
                if tracker:
                    tracked = tracker.stop(
                        timeout=max(0.0, cleanup_deadline - time.monotonic())
                    )
            except (OSError, RuntimeError, subprocess.SubprocessError) as cleanup_exc:
                _mark_cleanup_failed(envelope, cleanup_exc)
                tracked = tracker.observed_processes if tracker else []
            # Stopping an observer must never prevent a kill attempt. In
            # particular, a slow process-table scan can exhaust its allowance.
            try:
                terminate_process_tree(
                    proc,
                    tracked,
                    cleanup_timeout=max(0.0, cleanup_deadline - time.monotonic()),
                )
            except (OSError, RuntimeError, subprocess.SubprocessError) as cleanup_exc:
                _mark_cleanup_failed(envelope, cleanup_exc)
            if tracker and tracker.errors:
                _mark_cleanup_failed(envelope, "Descendant observation was incomplete")
            if envelope["cleanup_verified"] is not False:
                envelope["cleanup_verified"] = True
            envelope["containment"] = (
                tracker.boundary if tracker else "process_group_only"
            )
            envelope["observed_descendant_count"] = (
                tracker.observed_count if tracker else 0
            )
            try:
                stdout_bytes, stderr_bytes = proc.communicate(
                    timeout=max(0.001, cleanup_deadline - time.monotonic())
                )
                envelope["streams_drained"] = True
            except subprocess.TimeoutExpired as exc:
                stdout_bytes, stderr_bytes = (
                    exc.output or stdout_bytes,
                    exc.stderr or stderr_bytes,
                )
                envelope["streams_drained"] = False
                _mark_cleanup_failed(
                    envelope,
                    "Inherited output pipes did not close within cleanup budget",
                )
                for stream in (proc.stdin, proc.stdout, proc.stderr):
                    if stream is not None:
                        stream.close()
            if cleanup_ceiling is not None and time.monotonic() >= cleanup_ceiling:
                _mark_cleanup_failed(envelope, "Absolute process deadline exhausted")
        envelope["duration_seconds"] = time.monotonic() - start
        if rss_sampler is not None:
            envelope["child_peak_rss_mb"] = rss_sampler.peak_rss_mb
            envelope["rss_samples_count"] = rss_sampler.samples

    if timed_out:
        if envelope.get("cleanup_verified") is False:
            envelope["execution_error_type"] = "TimeoutExpired"
        else:
            envelope["error"] = (
                "Execution exceeded its total process budget after reserving cleanup"
                if total_budget_limited
                else f"Execution timed out after {resolved_timeout}s"
            )
            envelope["error_type"] = "TimeoutExpired"
            envelope["return_code"] = NEVER_STARTED
    elif envelope.get("cancelled") and envelope.get("cleanup_verified") is False:
        envelope["execution_error_type"] = "Cancelled"
    envelope["stdout"] = _as_text(stdout_bytes) if capture_output else ""
    envelope["stderr"] = _as_text(stderr_bytes) if capture_output else ""

    return envelope
