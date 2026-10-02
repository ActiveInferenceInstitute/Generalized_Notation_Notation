"""
Execution Utilities
==================

This module provides utilities for executing external commands with real-time output streaming.
It allows long-running processes (like test suites) to display their progress immediately
rather than buffering all output until completion.
"""

import logging
import os
import subprocess  # nosec B404
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from gnn.utils.runtime_safety.process_tree import (
    DescendantTracker,
)
from gnn.utils.runtime_safety.process_tree import (
    terminate_process_tree as _terminate_process_tree,
)

logger = logging.getLogger(__name__)


def execute_command_streaming(
    cmd: List[str],
    cwd: Optional[Union[str, Path]] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = None,
    print_stdout: bool = True,
    print_stderr: bool = True,
    capture_output: bool = True,
) -> Dict[str, Any]:
    """Stream a supervised worker and always clean up within a shared ceiling.

    Ordinary descendants use the worker process group; observations retain
    detached backend workers. Observation uncertainty and surviving streams
    prevent success. Cleanup runs even if process setup, waiting or observation
    raises, so an exception cannot release output ownership before a kill attempt.
    """
    import time

    from gnn.pipeline.run_context import current_run_context
    from gnn.utils.runtime_safety.process_budget import resolve_process_deadlines

    process_env = os.environ.copy()
    if env:
        process_env.update(env)
    process_env["PYTHONUNBUFFERED"] = "1"
    stdout_captured: list[str] = []
    stderr_captured: list[str] = []
    errors: list[str] = []
    result: dict[str, Any] = {
        "exit_code": -1,
        "stdout": "",
        "stderr": "",
        "status": "FAILED",
        "cleanup_verified": False,
        "streams_drained": False,
        "cleanup_timeout_seconds": 1.0,
    }
    process: subprocess.Popen[str] | None = None
    tracker: DescendantTracker | None = None
    readers: list[threading.Thread] = []
    context = current_run_context()
    execution_deadline, cleanup_allowance, cleanup_ceiling = (
        context.process_deadlines(timeout)
        if context is not None
        else resolve_process_deadlines(timeout)
    )
    result["cleanup_timeout_seconds"] = cleanup_allowance
    result["cleanup_deadline_monotonic"] = cleanup_ceiling
    result["work_deadline_monotonic"] = execution_deadline

    def read_stream(stream: Any, is_stderr: bool) -> None:
        try:
            for line in iter(stream.readline, ""):
                if not line:
                    break
                if is_stderr:
                    if print_stderr:
                        sys.stderr.write(line)
                        sys.stderr.flush()
                    if capture_output:
                        stderr_captured.append(line)
                else:
                    if print_stdout:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                    if capture_output:
                        stdout_captured.append(line)
        except (ValueError, OSError) as error:
            logger.debug("Stream read ended: %s", error)
        finally:
            stream.close()

    try:
        if execution_deadline is not None and time.monotonic() >= execution_deadline:
            raise TimeoutError("Process work budget exhausted before worker creation")
        process = subprocess.Popen(  # nosec B603
            cmd,
            cwd=str(cwd) if cwd else None,
            env=process_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
            start_new_session=True,
        )
        tracker = DescendantTracker(process.pid)
        tracker.start()
        for stream, is_stderr in ((process.stdout, False), (process.stderr, True)):
            reader = threading.Thread(
                target=read_stream, args=(stream, is_stderr), daemon=True
            )
            reader.start()
            readers.append(reader)
        try:
            remaining = (
                None
                if execution_deadline is None
                else max(0.0, execution_deadline - time.monotonic())
            )
            exit_code = process.wait(timeout=remaining)
            result.update(
                exit_code=exit_code, status="SUCCESS" if exit_code == 0 else "FAILED"
            )
        except subprocess.TimeoutExpired:
            result.update(status="TIMEOUT", exit_code=-1)
            result["stop_reason"] = (
                "total_timeout"
                if cleanup_ceiling is not None
                and execution_deadline is not None
                and execution_deadline >= cleanup_ceiling - cleanup_allowance
                else "step_timeout"
            )
    except Exception as error:
        errors.append(str(error))
        result.update(status="FAILED", execution_error_type=type(error).__name__)
    finally:
        if process is not None:
            cleanup_deadline = time.monotonic() + result["cleanup_timeout_seconds"]
            if cleanup_ceiling is not None:
                cleanup_deadline = min(cleanup_deadline, cleanup_ceiling)
            tracked: list[Any] = []
            observation_ok = tracker is not None
            if tracker is not None:
                try:
                    tracked = tracker.stop(
                        timeout=max(0.0, cleanup_deadline - time.monotonic())
                    )
                except Exception as error:
                    observation_ok = False
                    errors.append(str(error))
                    # A failed monitor join must never bypass termination.
                    tracked = tracker.observed_processes
                if tracker.errors:
                    observation_ok = False
                    errors.extend(tracker.errors)
                result["containment"] = tracker.boundary
                result["observed_descendants"] = len(tracked)
            try:
                _terminate_process_tree(
                    process,
                    tracked,
                    cleanup_timeout=max(0.0, cleanup_deadline - time.monotonic()),
                )
                result["cleanup_verified"] = observation_ok
                if result["status"] == "TIMEOUT":
                    result["force_killed"] = True
            except Exception as error:
                errors.append(str(error))
                result["execution_error_type"] = "ProcessCleanupFailure"
            for reader in readers:
                reader.join(timeout=max(0.0, cleanup_deadline - time.monotonic()))
            result["streams_drained"] = bool(readers) and all(
                not reader.is_alive() for reader in readers
            )
            if cleanup_ceiling is not None and time.monotonic() > cleanup_ceiling:
                result["cleanup_verified"] = False
                errors.append("Invocation deadline expired before cleanup verified")
            result["cleanup_completed_monotonic"] = time.monotonic()
            if not result["cleanup_verified"] or not result["streams_drained"]:
                result.update(
                    status="FAILED",
                    exit_code=-1,
                    execution_error_type="ProcessCleanupFailure",
                )
                if not result["streams_drained"]:
                    errors.append(
                        "Worker stream drain did not finish within cleanup budget"
                    )
                result["cleanup_error"] = (
                    "; ".join(errors) or "Process observation was incomplete"
                )
    if capture_output:
        result["stdout"] = "".join(stdout_captured)
        result["stderr"] = "".join(stderr_captured)
    if errors:
        result["stderr"] += ("\n" if result["stderr"] else "") + "; ".join(errors)
        if print_stderr:
            sys.stderr.write(f"Execution error: {'; '.join(errors)}\n")
    return result
