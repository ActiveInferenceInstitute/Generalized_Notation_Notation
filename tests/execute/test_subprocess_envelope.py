#!/usr/bin/env python3
"""Tests for ``gnn.execute.subprocess_envelope.run_subprocess_envelope``.

The envelope is the shared execution contract for every GNN execution
backend; these tests pin its failure-mode conversion semantics.
"""

from __future__ import annotations

import hashlib
import os
import stat
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gnn.execute.subprocess_envelope import (  # noqa: E402
    CancelToken,
    run_subprocess_envelope,
)

PYTHON = sys.executable


def test_success_envelope() -> None:
    result = run_subprocess_envelope([PYTHON, "-c", "print('hello-from-envelope')"])
    assert result["success"] is True
    assert result["return_code"] == 0
    assert "hello-from-envelope" in result["stdout"]
    assert result["stderr"] == ""
    assert isinstance(result["duration_seconds"], float)
    assert result["duration_seconds"] >= 0.0
    assert "error" not in result


def test_nonzero_exit_converted() -> None:
    result = run_subprocess_envelope([PYTHON, "-c", "raise SystemExit(3)"])
    assert result["success"] is False
    assert result["return_code"] == 3
    assert "error" not in result


def test_timeout_converted() -> None:
    result = run_subprocess_envelope(
        [PYTHON, "-c", "import time; time.sleep(30)"], timeout=1
    )
    assert result["success"] is False
    assert result["return_code"] == -1
    assert result["error_type"] == "TimeoutExpired"
    assert "timed out after 1s" in result["error"]


def test_oserror_converted() -> None:
    result = run_subprocess_envelope(["definitely-not-a-real-binary-xyz"])
    assert result["success"] is False
    assert result["return_code"] == -1
    assert result["error_type"] == "FileNotFoundError"
    assert result["stdout"] == ""
    assert result["stderr"] == ""


def test_env_overrides_merge_over_parent() -> None:
    result = run_subprocess_envelope(
        [
            PYTHON,
            "-c",
            "import os; print(os.environ.get('GNN_ENVELOPE_PROBE', 'missing'))",
        ],
        env={"GNN_ENVELOPE_PROBE": "present"},
    )
    assert result["success"] is True
    assert "present" in result["stdout"]


def test_capture_output_false_yields_empty_streams() -> None:
    result = run_subprocess_envelope(
        [PYTHON, "-c", "print('streamed')"], capture_output=False
    )
    assert result["success"] is True
    assert result["stdout"] == ""
    assert result["stderr"] == ""


def test_cwd_is_honored(tmp_path: Path) -> None:
    result = run_subprocess_envelope(
        [PYTHON, "-c", "import os; print(os.getcwd())"], cwd=str(tmp_path)
    )
    assert result["success"] is True
    assert str(tmp_path) in result["stdout"]


def test_command_is_argument_vector_not_shell(tmp_path: Path) -> None:
    result = run_subprocess_envelope(
        [PYTHON, "-c", "print('no shell interpolation $HOME')"]
    )
    assert result["success"] is True


@pytest.mark.parametrize(
    "missing_key",
    [
        "success",
        "return_code",
        "stdout",
        "stderr",
        "duration_seconds",
        "cancelled",
        "child_peak_rss_mb",
        "rss_sample_interval_seconds",
        "rss_samples_count",
    ],
)
def test_envelope_always_carries_core_keys(missing_key: str) -> None:
    ok = run_subprocess_envelope([PYTHON, "-c", "pass"])
    bad = run_subprocess_envelope(["definitely-not-a-real-binary-xyz"])
    assert missing_key in ok
    assert missing_key in bad


def test_timeout_captured_streams_are_str_with_partial_output() -> None:
    """TimeoutExpired streams land as documented ``str`` (CPython delivers
    bytes under ``text=True``); partial pre-kill output is preserved."""
    result = run_subprocess_envelope(
        [PYTHON, "-c", "print('partial-line', flush=True); import time; time.sleep(5)"],
        timeout=1,
    )
    assert result["success"] is False
    assert result["error_type"] == "TimeoutExpired"
    assert "timed out after 1s" in result["error"]
    assert result["stdout"] == "partial-line\n"
    assert result["stderr"] == ""


def test_timeout_without_capture_yields_empty_streams() -> None:
    result = run_subprocess_envelope(
        [PYTHON, "-c", "import time; time.sleep(5)"],
        timeout=1,
        capture_output=False,
    )
    assert result["success"] is False
    assert result["error_type"] == "TimeoutExpired"
    assert result["stdout"] == ""
    assert result["stderr"] == ""


def test_input_support_pipes_stdin_to_child() -> None:
    """Wave-2 MIN-03: the envelope can feed stdin (no raw bypass needed)."""
    envelope = run_subprocess_envelope(
        [sys.executable, "-c", "import sys; print(sys.stdin.read().strip())"],
        timeout=30,
        input="envelope-stdin-ok",
    )
    assert envelope["success"] is True
    assert envelope["return_code"] == 0
    assert "envelope-stdin-ok" in envelope["stdout"]


def test_slow_reader_receives_complete_multimegabyte_utf8_input() -> None:
    """Startup past several poll slices must preserve all bytes and EOF."""
    payload = "Ω scientific matrix\x00\r\n" * 150000
    encoded = payload.encode("utf-8")
    child = (
        "import sys,time,hashlib; time.sleep(0.75); "
        "data=sys.stdin.buffer.read(); "
        "print(len(data)); print(hashlib.sha256(data).hexdigest())"
    )
    result = run_subprocess_envelope([PYTHON, "-c", child], input=payload, timeout=6)
    assert result["success"], result
    assert result["stdout"].splitlines() == [
        str(len(encoded)),
        hashlib.sha256(encoded).hexdigest(),
    ]
    assert result["cleanup_verified"] and result["streams_drained"]


@pytest.mark.parametrize("mode", ["success", "spawn_failure", "timeout", "cancel"])
def test_input_descriptor_is_private_and_closed_on_every_outcome(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A child that never reads input must still time out/cancel and reap."""
    import gnn.execute.subprocess_envelope as module

    opened = []
    original = module.tempfile.TemporaryFile

    def observed_file(*args, **kwargs):
        handle = original(*args, **kwargs)
        opened.append(handle)
        if os.name == "posix":
            metadata = os.fstat(handle.fileno())
            assert stat.S_IMODE(metadata.st_mode) == 0o600
            assert metadata.st_nlink == 0
        return handle

    monkeypatch.setattr(module.tempfile, "TemporaryFile", observed_file)
    token = CancelToken()
    timer = threading.Timer(0.5, token.cancel, args=("large-input-abort",))
    if mode == "cancel":
        timer.start()
    command = (
        ["definitely-not-a-real-binary-xyz"]
        if mode == "spawn_failure"
        else [
            PYTHON,
            "-c",
            "print('started',flush=True); "
            + ("pass" if mode == "success" else "import time; time.sleep(30)"),
        ]
    )
    try:
        result = run_subprocess_envelope(
            command, input="λ" * 2_000_000, timeout=1, cancel_token=token
        )
    finally:
        timer.cancel()
        if mode == "cancel":
            timer.join(timeout=2)
    assert opened and all(handle.closed for handle in opened)
    if mode == "success":
        assert result["success"]
    else:
        assert not result["success"]
        assert (
            result["error_type"]
            == {
                "spawn_failure": "FileNotFoundError",
                "timeout": "TimeoutExpired",
                "cancel": "Cancelled",
            }[mode]
        )
    if mode != "spawn_failure":
        assert result["cleanup_verified"] and result["streams_drained"]
    assert result["duration_seconds"] < 4


def test_stdin_staging_does_not_renew_an_exhausted_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gnn.execute.subprocess_envelope as module

    original = module.tempfile.TemporaryFile
    opened = []

    def delayed_file(*args, **kwargs):
        time.sleep(0.15)
        handle = original(*args, **kwargs)
        opened.append(handle)
        return handle

    def forbidden_spawn(*args, **kwargs):
        raise AssertionError("Input staging exhausted the original launch budget")

    monkeypatch.setattr(module.tempfile, "TemporaryFile", delayed_file)
    monkeypatch.setattr(module.subprocess, "Popen", forbidden_spawn)
    result = run_subprocess_envelope(
        [PYTHON, "-c", "pass"], input="payload", timeout=0.05
    )
    assert not result["success"]
    assert result["error_type"] == "TimeoutExpired"
    assert opened and all(handle.closed for handle in opened)


def test_stdin_close_failure_still_reaps_the_real_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gnn.execute.subprocess_envelope as module

    original = module.tempfile.TemporaryFile
    opened = []

    class FailingClose:
        def __init__(self, handle):
            self.handle = handle

        def __getattr__(self, name):
            return getattr(self.handle, name)

        def close(self):
            self.handle.close()
            raise OSError("injected stdin close failure")

    def failing_file(*args, **kwargs):
        handle = original(*args, **kwargs)
        opened.append(handle)
        return FailingClose(handle)

    monkeypatch.setattr(module.tempfile, "TemporaryFile", failing_file)
    result = run_subprocess_envelope(
        [PYTHON, "-c", "import os,time; print(os.getpid(),flush=True); time.sleep(30)"],
        input="complete input",
        timeout=1,
    )
    assert not result["success"] and result["cleanup_verified"] is False
    assert result["streams_drained"] is True
    assert "injected stdin close failure" in result["error"]
    assert result["execution_error_type"] == "TimeoutExpired"
    assert opened and all(handle.closed for handle in opened)
    import psutil

    assert not psutil.pid_exists(int(result["stdout"].strip()))


def test_sandbox_false_runs_unsandboxed_with_receipt(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """sandbox=False runs the command verbatim and emits the disabled receipt."""
    import logging

    with caplog.at_level(logging.WARNING, logger="gnn.execute.subprocess_envelope"):
        result = run_subprocess_envelope(
            [PYTHON, "-c", "print('unsandboxed-ok')"], sandbox=False
        )
    assert result["success"] is True
    assert result["sandbox_mode"] == "off"
    assert result["sandboxed"] is False
    assert "unsandboxed-ok" in result["stdout"]
    receipts = [
        r
        for r in caplog.records
        if r.__dict__.get("event") == "sandbox_disabled_receipt"
    ]
    assert receipts, "sandbox=False must emit sandbox_disabled_receipt"


def test_sandbox_default_off_mode_runs_unsandboxed_with_receipt(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Default sandbox=True with GNN_SANDBOX=off runs unsandboxed with receipt."""
    import logging

    with caplog.at_level(logging.WARNING, logger="gnn.execute.subprocess_envelope"):
        result = run_subprocess_envelope([PYTHON, "-c", "print('default-off-ok')"])
    assert result["success"] is True
    assert result["sandbox_mode"] == "off"
    assert result["sandboxed"] is False
    receipts = [
        r
        for r in caplog.records
        if r.__dict__.get("event") == "sandbox_disabled_receipt"
    ]
    assert receipts, "GNN_SANDBOX=off default must emit sandbox_disabled_receipt"


def test_sandbox_require_without_backend_refuses_to_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GNN_SANDBOX=require with no backend blocks before any subprocess runs."""
    import subprocess as sp

    from gnn.execute import subprocess_envelope as se

    monkeypatch.setenv("GNN_SANDBOX", "require")
    monkeypatch.setattr(
        "gnn.execute.sandbox.detect_sandbox", lambda: None, raising=False
    )

    def _boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("subprocess must not run when sandbox require is blocked")

    monkeypatch.setattr(sp, "run", _boom)
    result = run_subprocess_envelope([PYTHON, "-c", "print('must-not-run')"])
    assert result["success"] is False
    assert result["return_code"] == se.NEVER_STARTED
    assert result["error_type"] == "SandboxUnavailable"
    assert "require" in result["error"]


def test_sandbox_prefer_wraps_command_when_backend_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GNN_SANDBOX=prefer with a backend prefixes the command vector."""
    from gnn.execute.sandbox import SandboxSpec

    monkeypatch.setenv("GNN_SANDBOX", "prefer")
    monkeypatch.setattr(
        "gnn.execute.sandbox.detect_sandbox",
        lambda: SandboxSpec("echo", ("echo", "SANDBOXED")),
    )
    result = run_subprocess_envelope([PYTHON, "-c", "print('wrapped')"])
    assert result["success"] is True
    assert result["sandboxed"] is True
    assert result["sandbox_mode"] == "prefer"
    assert "SANDBOXED" in result["stdout"]


def test_cancelled_is_false_on_non_cancel_paths() -> None:
    """``cancelled`` is additive: False on success, timeout, and spawn failure."""
    ok = run_subprocess_envelope([PYTHON, "-c", "pass"])
    timed_out = run_subprocess_envelope(
        [PYTHON, "-c", "import time; time.sleep(30)"], timeout=1
    )
    spawn_failure = run_subprocess_envelope(["definitely-not-a-real-binary-xyz"])
    for envelope in (ok, timed_out, spawn_failure):
        assert envelope["cancelled"] is False


def test_cancel_before_spawn_spawns_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A token already cancelled at entry reports cancelled without spawning."""
    import subprocess as sp

    from gnn.execute import subprocess_envelope as se

    token = CancelToken()
    token.cancel("pre-spawn-abort")
    spawn_attempts: list[object] = []

    def _refuse_to_spawn(*args: object, **kwargs: object) -> object:
        spawn_attempts.append(args)
        raise AssertionError("Popen must not run for a pre-spawn cancel")

    monkeypatch.setattr(sp, "Popen", _refuse_to_spawn)

    result = run_subprocess_envelope(
        [PYTHON, "-c", "print('must-not-run')"], timeout=30, cancel_token=token
    )

    assert spawn_attempts == []
    assert result["cancelled"] is True
    assert result["success"] is False
    assert result["error_type"] == "Cancelled"
    assert result["return_code"] == se.NEVER_STARTED
    assert result["error"] == "Execution cancelled: pre-spawn-abort"
    assert result["stdout"] == ""
    assert result["stderr"] == ""


def test_cancel_mid_flight_kills_group_and_preserves_partial_output() -> None:
    """A cancel from a second thread beats a long explicit timeout: the poll
    loop observes the token within one slice, group-kills, and drains the
    partial streams instead of waiting out the 30s deadline."""
    token = CancelToken()

    def _cancel_soon() -> None:
        time.sleep(0.5)
        token.cancel("user-abort")

    canceller = threading.Thread(target=_cancel_soon, daemon=True)
    canceller.start()
    result = run_subprocess_envelope(
        [
            PYTHON,
            "-c",
            "print('partial-line', flush=True); import time; time.sleep(30)",
        ],
        timeout=30,
        cancel_token=token,
    )
    canceller.join(timeout=2)

    assert result["cancelled"] is True
    assert result["success"] is False
    assert result["error_type"] == "Cancelled"
    assert result["return_code"] == -1
    assert result["error"] == "Execution cancelled: user-abort"
    assert "partial-line" in result["stdout"]
    assert result["duration_seconds"] < 10.0  # cancelled at ~0.5s, not 30s


def test_timeout_kills_whole_process_group_including_grandchildren(
    tmp_path: Path,
) -> None:
    """Regression: the timeout kill must reap the whole spawned tree.

    The child spawns a grandchild that writes a sentinel ~3s in; the parent
    is killed at timeout=1. A child-only kill orphans the grandchild, which
    then writes the sentinel — this assertion fails on that regression.
    """
    sentinel = tmp_path / "sentinel"
    grandchild_code = (
        "import pathlib, time\n"
        f"time.sleep(3)\npathlib.Path({str(sentinel)!r}).write_text('x')\n"
    )
    child_code = (
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {grandchild_code!r}])\n"
        "time.sleep(30)\n"
    )
    result = run_subprocess_envelope([PYTHON, "-c", child_code], timeout=1)

    assert result["success"] is False
    assert result["error_type"] == "TimeoutExpired"
    time.sleep(3.5)  # the grandchild would write the sentinel at ~3s if orphaned
    assert not sentinel.exists(), "grandchild survived the timeout group-kill"


@pytest.mark.needs_posix
@pytest.mark.parametrize("mode", ["timeout", "cancel", "normal_exit"])
def test_detached_descendant_cannot_hold_pipes_or_write_after_return(
    tmp_path: Path,
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import contextlib

    import psutil

    from gnn.execute import subprocess_envelope as se

    def unrelated_host_scan(*args: object, **kwargs: object) -> None:
        raise AssertionError("supervision must not scan the entire host process table")

    # All three real completion paths must discover and kill the detached
    # process without psutil's recursive, host-wide process enumeration.
    monkeypatch.setattr(psutil.Process, "children", unrelated_host_scan)
    pid_file = tmp_path / "detached.pid"
    observed_barrier = tmp_path / "observed"
    sentinel = tmp_path / "late-write"
    trackers: list[se.DescendantTracker] = []

    class RecordingTracker(se.DescendantTracker):
        def __init__(self, pid: int) -> None:
            super().__init__(pid)
            trackers.append(self)

    monkeypatch.setattr(se, "DescendantTracker", RecordingTracker)
    child_code = (
        "import os,pathlib,time; "
        f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid())); "
        # The write falls after execution plus its one-second cleanup bound.
        "print('detached-partial', flush=True); time.sleep(4); "
        f"pathlib.Path({str(sentinel)!r}).write_text('escaped'); time.sleep(30)"
    )
    leader_code = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable,'-c',{child_code!r}], start_new_session=True); "
        f"barrier=__import__('pathlib').Path({str(observed_barrier)!r}); "
        "deadline=time.monotonic()+3; "
        "exec('while not barrier.exists() and time.monotonic()<deadline: time.sleep(.01)'); "
        f"time.sleep({0.1 if mode == 'normal_exit' else 30})"
    )
    token = CancelToken()
    stop_observer = threading.Event()

    def release_after_observation() -> None:
        while not stop_observer.wait(0.01):
            if trackers and trackers[0].observed_count:
                observed_barrier.write_text("observed")
                if mode == "cancel":
                    token.cancel("detached cancel")
                return

    observer = threading.Thread(target=release_after_observation)
    observer.start()
    started = time.monotonic()
    try:
        result = run_subprocess_envelope(
            [PYTHON, "-c", leader_code],
            timeout=1.5,
            cancel_token=token,
            capture_output=mode != "normal_exit",
        )
        assert time.monotonic() - started < 2.9, result
        assert (
            result["cleanup_verified"] is True and result["streams_drained"] is True
        ), result
        assert result["containment"] == "observed_descendants"
        assert result["observed_descendant_count"] >= 1
        assert observed_barrier.exists()
        if mode == "normal_exit":
            assert result["success"] and result["return_code"] == 0
        else:
            assert not result["success"]
            assert "detached-partial" in result["stdout"]
            assert result["error_type"] == (
                "Cancelled" if mode == "cancel" else "TimeoutExpired"
            )
        assert pid_file.exists(), result
        child = (
            psutil.Process(int(pid_file.read_text()))
            if psutil.pid_exists(int(pid_file.read_text()))
            else None
        )
        assert child is None or child.status() == psutil.STATUS_ZOMBIE
        assert not sentinel.exists()
    finally:
        stop_observer.set()
        observer.join(timeout=1)
        if pid_file.exists():
            with contextlib.suppress(psutil.NoSuchProcess):
                psutil.Process(int(pid_file.read_text())).kill()


@pytest.mark.parametrize("timeout", [True, 0, -1, float("nan"), float("inf")])
def test_invalid_timeout_does_not_launch(
    timeout: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("invalid timeout must not start a worker")

    monkeypatch.setattr(subprocess, "Popen", fail)
    result = run_subprocess_envelope([PYTHON, "-c", "pass"], timeout=timeout)  # type: ignore[arg-type]
    assert not result["success"] and result["error_type"] == "InvalidExecutionTimeout"


def test_unverified_cleanup_cannot_publish_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def denied(*args: object, **kwargs: object) -> None:
        raise PermissionError("cleanup observation denied")

    monkeypatch.setattr(
        "gnn.execute.subprocess_envelope.terminate_process_tree", denied
    )
    result = run_subprocess_envelope([PYTHON, "-c", "print('completed')"])
    assert not result["success"]
    assert result["error_type"] == "ProcessCleanupFailure"
    assert result["cleanup_verified"] is False
    assert result["streams_drained"] is True
    assert result["stdout"] == "completed\n"
    assert "denied" in result["cleanup_error"]


def test_observer_stop_failure_still_kills_live_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import contextlib

    import psutil

    from gnn.execute import subprocess_envelope as se

    class FailedStop(se.DescendantTracker):
        def stop(self, timeout: float = 0.1) -> list[psutil.Process]:
            super().stop(timeout)
            raise RuntimeError("observer stop could not be certified")

    monkeypatch.setattr(se, "DescendantTracker", FailedStop)
    pid_file = tmp_path / "worker.pid"
    code = (
        "import os,pathlib,time; "
        f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid())); "
        "print('worker-partial',flush=True); time.sleep(30)"
    )
    started = time.monotonic()
    try:
        result = run_subprocess_envelope([PYTHON, "-c", code], timeout=0.4)
        assert time.monotonic() - started < 1.9, result
        assert not result["success"] and result["cleanup_verified"] is False
        assert result["error_type"] == "ProcessCleanupFailure"
        assert result["execution_error_type"] == "TimeoutExpired"
        assert result["streams_drained"] is True
        assert "worker-partial" in result["stdout"]
        assert "observer stop" in result["cleanup_error"]
        assert not psutil.pid_exists(int(pid_file.read_text()))
    finally:
        if pid_file.exists():
            with contextlib.suppress(psutil.NoSuchProcess):
                psutil.Process(int(pid_file.read_text())).kill()


@pytest.mark.needs_posix
def test_denied_process_group_cleanup_cannot_publish_completed_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def denied(*args: object, **kwargs: object) -> None:
        raise PermissionError("process-group signal denied")

    monkeypatch.setattr("gnn.utils.runtime_safety.process_tree.os.killpg", denied)
    result = run_subprocess_envelope([PYTHON, "-c", "print('completed')"])
    assert not result["success"] and result["cleanup_verified"] is False
    assert result["error_type"] == "ProcessCleanupFailure"
    assert "Process-group termination was denied" in result["cleanup_error"]
    assert result["stdout"] == "completed\n" and result["streams_drained"] is True


@pytest.mark.needs_posix
@pytest.mark.parametrize("absolute_budget", [False, True])
def test_unobserved_inherited_pipe_has_bounded_failure_and_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    absolute_budget: bool,
) -> None:
    """A missed detached child cannot turn the final drain into an infinite wait.

    Deliberately suppress descendant observation to exercise its documented OS
    boundary. The test owns and reaps the surviving process independently.
    """
    import contextlib
    import json

    import psutil

    class NoObservation:
        boundary = "process_group_only"
        errors: list[str] = []
        observed_count = 0
        observed_processes: list[psutil.Process] = []

        def __init__(self, pid: int) -> None:
            pass

        def start(self) -> None:
            pass

        def stop(self, timeout: float) -> list[psutil.Process]:
            return []

    monkeypatch.setattr(
        "gnn.execute.subprocess_envelope.DescendantTracker", NoObservation
    )
    identity_file = tmp_path / "escaped.json"
    child_code = (
        "import json,os,pathlib,psutil,time; "
        f"pathlib.Path({str(identity_file)!r}).write_text(json.dumps("
        "{'pid':os.getpid(),'created':psutil.Process().create_time()})); "
        "print('unobserved-partial',flush=True); time.sleep(30)"
    )
    leader_code = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable,'-c',{child_code!r}],start_new_session=True); "
        "time.sleep(0.1)"
    )
    started = time.monotonic()
    try:
        result = run_subprocess_envelope(
            [PYTHON, "-c", leader_code],
            timeout=0.3,
            deadline_monotonic=started + 0.8 if absolute_budget else None,
        )
        assert time.monotonic() - started < (0.95 if absolute_budget else 1.9), result
        assert not result["success"]
        assert result["error_type"] == "ProcessCleanupFailure"
        assert result["execution_error_type"] == "TimeoutExpired"
        assert result["cleanup_verified"] is False
        assert result["streams_drained"] is False
        assert result["containment"] == "process_group_only"
        assert "unobserved-partial" in result["stdout"]
        assert "pipes" in result["cleanup_error"]
    finally:
        if identity_file.exists():
            identity = json.loads(identity_file.read_text())
            with contextlib.suppress(psutil.NoSuchProcess):
                child = psutil.Process(identity["pid"])
                if child.create_time() == identity["created"]:
                    child.kill()
                    with contextlib.suppress(psutil.TimeoutExpired):
                        child.wait(timeout=2)


def test_absolute_request_deadline_reserves_cleanup_inside_short_budget(
    tmp_path: Path,
) -> None:
    import psutil

    pid_file = tmp_path / "worker.pid"
    code = (
        "import os,pathlib,time; "
        f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid())); "
        "print('request-partial',flush=True); time.sleep(30)"
    )
    started = time.monotonic()
    result = run_subprocess_envelope(
        [PYTHON, "-c", code], timeout=30, deadline_monotonic=started + 0.8
    )
    assert time.monotonic() - started < 0.95, result
    assert result["error_type"] == "TimeoutExpired", result
    assert result["cleanup_verified"] is True and result["streams_drained"] is True
    assert result["cleanup_timeout_seconds"] <= 0.2
    assert "request-partial" in result["stdout"]
    assert pid_file.exists(), "short budgets must still execute real work"
    assert not psutil.pid_exists(int(pid_file.read_text()))


@pytest.mark.parametrize("deadline", [True, float("nan"), float("inf"), "bad"])
def test_invalid_absolute_deadline_never_launches(
    deadline: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("invalid deadline must not start a worker")

    monkeypatch.setattr("subprocess.Popen", fail)
    result = run_subprocess_envelope(
        [PYTHON, "-c", "pass"],
        deadline_monotonic=deadline,  # type: ignore[arg-type]
    )
    assert result["error_type"] == "InvalidExecutionDeadline" and not result["success"]


def test_expired_absolute_deadline_never_launches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("expired deadline must not start a worker")

    monkeypatch.setattr("subprocess.Popen", fail)
    result = run_subprocess_envelope(
        [PYTHON, "-c", "pass"], deadline_monotonic=time.monotonic() - 1
    )
    assert result["error_type"] == "TimeoutExpired" and not result["success"]


RSS_KEYS = ("child_peak_rss_mb", "rss_sample_interval_seconds", "rss_samples_count")


def test_child_peak_rss_sampled_on_poll_cadence() -> None:
    """W8-B: a live child reports peak tree RSS sampled on the poll cadence."""
    result = run_subprocess_envelope([PYTHON, "-c", "import time; time.sleep(0.4)"])
    assert result["success"] is True, result.get("error")
    assert result["child_peak_rss_mb"] is not None
    assert result["child_peak_rss_mb"] > 0
    assert result["rss_samples_count"] > 0
    assert result["rss_sample_interval_seconds"] == 0.25


def test_child_rss_keys_null_when_psutil_handle_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A psutil failure degrades to explicit null keys; envelope shape holds."""
    monkeypatch.setattr("gnn.execute.subprocess_envelope._psutil_module", None)
    result = run_subprocess_envelope([PYTHON, "-c", "import time; time.sleep(0.4)"])
    assert result["success"] is True
    assert result["child_peak_rss_mb"] is None
    assert result["rss_samples_count"] == 0
    assert result["rss_sample_interval_seconds"] == 0.25


@pytest.mark.parametrize("rss_key", RSS_KEYS)
def test_child_rss_keys_present_on_every_envelope_path(rss_key: str) -> None:
    """Additive contract: the three keys exist (never absent) on the success,
    spawn-failure, and pre-spawn-cancel envelopes; values may be null."""
    ok = run_subprocess_envelope([PYTHON, "-c", "import time; time.sleep(0.2)"])
    bad = run_subprocess_envelope(["definitely-not-a-real-binary-xyz"])
    token = CancelToken()
    token.cancel(reason="pre-spawn")
    cancelled = run_subprocess_envelope([PYTHON, "-c", "pass"], cancel_token=token)
    for envelope in (ok, bad, cancelled):
        assert rss_key in envelope
