"""Bounded native supervision shared by the API job and run executors."""

from __future__ import annotations

import asyncio
import os
import signal
import time
from collections.abc import Callable
from typing import Any

from gnn.utils.runtime_safety.process_tree import DescendantTracker


class ProcessCleanupError(RuntimeError):
    """The API worker could not be stopped and drained within its cleanup budget."""

    def __init__(self, message: str, receipt: dict[str, Any]) -> None:
        super().__init__(message)
        self.receipt = receipt


def request_process_stop(process: Any, *, force: bool = False) -> None:
    """Request stop of the owned native group, or direct Windows worker.

    Use the group established at spawn, not getpgid on a potentially exited PID.
    The direct-child fallback also supports the API's existing process adapters.
    This requests termination; only the async supervisor verifies its completion.
    """
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
            return
        except PermissionError:
            if getattr(process, "returncode", None) is None:
                (getattr(process, "kill", None) or process.terminate)()
            raise
        except (ProcessLookupError, AttributeError):
            pass
    if getattr(process, "returncode", None) is None:
        stop = (getattr(process, "kill", None) or process.terminate) if force else process.terminate
        stop()


async def supervise_api_process(
    process: Any,
    *,
    cancelled: Callable[[], bool],
    cleanup_timeout: float = 1.0,
) -> tuple[bytes, bytes, dict[str, Any]]:
    """Observe while running, then stop and drain within one shared deadline.

    The direct/group boundary has the same native limits as the execute envelope.
    Completion includes killing observed detached descendants even after a normal
    leader exit. Cancellation is a request, not proof that cleanup succeeded.
    """
    import psutil

    tracker = DescendantTracker(process.pid)
    task = asyncio.create_task(process.communicate())
    original_error: BaseException | None = None
    errors: list[str] = []
    try:
        tracker.start()
        while not task.done() and not cancelled() and process.returncode is None:
            await asyncio.wait({task}, timeout=.05)
    except BaseException as error:
        original_error = error

    deadline = time.monotonic() + max(0., cleanup_timeout)
    try:
        tracked = tracker.stop(timeout=min(.1, max(0., deadline - time.monotonic())))
    except RuntimeError as error:
        errors.append(str(error))
        tracked = tracker.observed_processes
    try:
        request_process_stop(process)
    except (OSError, AttributeError) as error:
        errors.append(f"Worker termination failed: {error}")
    # Preserve ordinary graceful SIGTERM cancellation, then escalate inside the
    # same ceiling. A SIGTERM-ignoring worker cannot keep the API awaiting pipes.
    if not task.done():
        await asyncio.wait({task}, timeout=min(.15, max(0., deadline - time.monotonic())))
    try:
        request_process_stop(process, force=True)
    except (OSError, AttributeError) as error:
        errors.append(f"Worker forced termination failed: {error}")
    for child in tracked:
        if time.monotonic() >= deadline:
            errors.append("Descendant termination exceeded cleanup budget")
            break
        try:
            if child.is_running():
                child.kill()  # psutil verifies retained birth-time identity
        except psutil.NoSuchProcess:
            pass
        except psutil.Error as error:
            errors.append(f"Descendant termination failed: {error}")

    stdout, stderr = b"", b""
    streams_drained = False
    try:
        stdout, stderr = await asyncio.wait_for(
            asyncio.shield(task), timeout=max(.001, deadline - time.monotonic())
        )
        streams_drained = True
    except BaseException as error:
        errors.append(f"Worker stream drain failed: {type(error).__name__}: {error}")
        task.cancel()
    while tracked:
        alive = []
        for child in tracked:
            if time.monotonic() >= deadline:
                errors.append("Descendant verification exceeded cleanup budget")
                break
            try:
                if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                    alive.append(child)
            except psutil.NoSuchProcess:
                pass
            except psutil.Error as error:
                errors.append(f"Descendant verification failed: {error}")
        if time.monotonic() >= deadline:
            break
        tracked = alive
        if tracked:
            await asyncio.sleep(min(.01, max(0., deadline - time.monotonic())))
    errors.extend(tracker.errors)
    if process.returncode is None:
        errors.append("Worker exit could not be verified")
    if time.monotonic() > deadline:
        errors.append("Cleanup exceeded its shared deadline")
    receipt: dict[str, Any] = {
        "containment": tracker.boundary,
        "observed_descendant_count": tracker.observed_count,
        "cleanup_verified": not errors,
        "streams_drained": streams_drained,
        "cleanup_timeout_seconds": cleanup_timeout,
    }
    if errors:
        receipt["cleanup_error"] = "; ".join(dict.fromkeys(errors))
        raise ProcessCleanupError(receipt["cleanup_error"], receipt)
    if original_error is not None:
        raise original_error
    return stdout, stderr, receipt
