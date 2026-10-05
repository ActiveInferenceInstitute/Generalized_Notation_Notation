"""Behavior tests for the job-cancel race in the async job manager.

A cancelled job must stay ``cancelled`` at every race point:

- cancel mid-flight while ``communicate()`` waits → the terminal state stays
  ``cancelled`` (never re-reported ``failed`` with a stderr error message)
  and whatever partial output the pipeline already wrote is preserved;
- cancel after completion → no-op, the job stays ``completed``;
- double cancel → idempotent, the first cancel wins;
- cancel while pending → the pipeline is never even launched;
- cancel signals the whole process group → grandchildren die with the job.
"""

import asyncio
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Callable

import pytest

from gnn.api import processor as job_mgr
from gnn.api.pipeline_runner import PIPELINE_SUMMARY

#: A pid that cannot exist: os.getpgid raises ProcessLookupError, exercising
#: the direct-child fallback in _terminate_process_tree instead of ever
#: signalling a real process group.
_NO_SUCH_PID = 999_999_999


def _job(job_id: str) -> dict[str, Any]:
    """Fetch a known-created job's client-visible state, non-Optionally."""
    job = job_mgr.get_job(job_id)
    assert job is not None, f"job {job_id} vanished from the store"
    return job


class _BlockingProcess:
    """Fake subprocess that blocks in communicate() until terminated."""

    def __init__(
        self,
        returncode: int,
        stderr: bytes = b"",
        on_release: Callable[[], None] | None = None,
    ) -> None:
        self._final_returncode = returncode
        self.returncode: int | None = None
        self.stderr = stderr
        self.pid = _NO_SUCH_PID
        self.terminate_calls = 0
        self._released = asyncio.Event()
        self._on_release = on_release

    async def communicate(self) -> tuple[bytes, bytes]:
        await self._released.wait()
        if self._on_release is not None:
            self._on_release()
        self.returncode = self._final_returncode
        return b"partial stdout", self.stderr

    def terminate(self) -> None:
        self.terminate_calls += 1
        self._released.set()


class _InstantProcess:
    """Fake subprocess that has already finished successfully."""

    def __init__(self) -> None:
        self.returncode = 0
        self.pid = _NO_SUCH_PID

    async def communicate(self) -> tuple[bytes, bytes]:
        return b"", b""


async def _poll_until(
    predicate: Callable[[], bool], *, timeout: float = 15.0, interval: float = 0.05
) -> None:
    """Await a predicate, failing the test when it never becomes true."""

    async def _loop() -> bool:
        while not predicate():
            await asyncio.sleep(interval)
        return True

    await asyncio.wait_for(_loop(), timeout=timeout)


@pytest.mark.asyncio
async def test_cancel_midflight_keeps_cancelled_state_and_partial_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A job cancelled mid-flight must report 'cancelled', not 'failed'."""
    job_id = job_mgr.create_job(target_dir=".")
    try:
        job_mgr._JOBS[job_id]["output_dir"] = str(tmp_path)
        started = asyncio.Event()

        def write_partial_summary() -> None:
            """Simulate the pipeline having finished one step pre-cancel."""
            summary_path = tmp_path / PIPELINE_SUMMARY
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(
                    {
                        "run_id": job_id,
                        "steps": [
                            {
                                "script_name": "3_gnn.py",
                                "step_num": 3,
                                "status": "SUCCESS",
                                "duration_seconds": 0.1,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            # Automatic filesystem mtimes can use a coarser clock than the
            # invocation freshness gate. Stamp this simulated current-run
            # receipt explicitly; stale-summary rejection is tested separately.
            now_ns = time.time_ns()
            os.utime(summary_path, ns=(now_ns, now_ns))

        proc = _BlockingProcess(
            returncode=-15,
            stderr=b"Traceback (most recent call last): killed by signal",
            on_release=write_partial_summary,
        )

        async def fake_spawn(*_cmd: str, **_kwargs: Any) -> _BlockingProcess:
            started.set()
            return proc

        monkeypatch.setattr(job_mgr.asyncio, "create_subprocess_exec", fake_spawn)

        task = asyncio.create_task(job_mgr.execute_job_async(job_id))
        await asyncio.wait_for(started.wait(), timeout=10.0)

        assert job_mgr.cancel_job(job_id) is True
        cancelled = _job(job_id)
        assert cancelled["status"] == "cancelled"
        assert cancelled["completed_at"] is not None

        await asyncio.wait_for(task, timeout=10.0)

        final = _job(job_id)
        # The terminated process exits nonzero (SIGTERM); execute must not
        # re-report the job as 'failed' with a fabricated stderr message.
        assert final["status"] == "cancelled"
        assert final["error_message"] is None
        assert final["exit_code"] == -15
        # Partial output written before the cancel is preserved.
        assert final["steps_completed"] == [3]
        assert final["steps_failed"] == []
        # cancel_job's own write is terminal: completed_at untouched, and the
        # terminate attempt really happened.
        assert final["completed_at"] == cancelled["completed_at"]
        assert proc.terminate_calls == 1
    finally:
        job_mgr._JOBS.pop(job_id, None)


@pytest.mark.asyncio
async def test_cancel_after_completion_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancel landing after completion must not touch the finished job."""
    job_id = job_mgr.create_job(target_dir=".")
    try:

        async def fake_spawn(*_cmd: str, **_kwargs: Any) -> _InstantProcess:
            return _InstantProcess()

        monkeypatch.setattr(job_mgr.asyncio, "create_subprocess_exec", fake_spawn)

        await job_mgr.execute_job_async(job_id)
        job = _job(job_id)
        assert job["status"] == "completed"
        assert job["error_message"] is None

        assert job_mgr.cancel_job(job_id) is False
        after = _job(job_id)
        assert after["status"] == "completed"
        assert after["completed_at"] == job["completed_at"]
        assert after["error_message"] is None
    finally:
        job_mgr._JOBS.pop(job_id, None)


@pytest.mark.asyncio
async def test_double_cancel_is_idempotent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The second cancel is a no-op; the first cancel's state wins."""
    job_id = job_mgr.create_job(target_dir=".")
    try:
        job_mgr._JOBS[job_id]["output_dir"] = str(tmp_path)
        started = asyncio.Event()
        proc = _BlockingProcess(returncode=-15)

        async def fake_spawn(*_cmd: str, **_kwargs: Any) -> _BlockingProcess:
            started.set()
            return proc

        monkeypatch.setattr(job_mgr.asyncio, "create_subprocess_exec", fake_spawn)

        task = asyncio.create_task(job_mgr.execute_job_async(job_id))
        await asyncio.wait_for(started.wait(), timeout=10.0)

        assert job_mgr.cancel_job(job_id) is True
        first = _job(job_id)
        assert job_mgr.cancel_job(job_id) is False
        second = _job(job_id)
        assert second["status"] == "cancelled"
        assert second["completed_at"] == first["completed_at"]

        await asyncio.wait_for(task, timeout=10.0)
        final = _job(job_id)
        assert final["status"] == "cancelled"
        assert final["error_message"] is None
    finally:
        job_mgr._JOBS.pop(job_id, None)


@pytest.mark.asyncio
async def test_cancel_pending_job_never_launches_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A job cancelled before execution starts must never spawn its pipeline."""
    job_id = job_mgr.create_job(target_dir=".")
    try:
        spawn_calls: list[list[str]] = []

        async def fake_spawn(*_cmd: str, **_kwargs: Any) -> None:
            spawn_calls.append(list(_cmd))
            raise AssertionError("pipeline launched for a cancelled job")

        monkeypatch.setattr(job_mgr.asyncio, "create_subprocess_exec", fake_spawn)

        assert job_mgr.cancel_job(job_id) is True
        assert _job(job_id)["status"] == "cancelled"

        # Simulates the background task racing a fast cancel: the skip guard
        # must short-circuit before touching the subprocess.
        await job_mgr.execute_job_async(job_id)
        assert spawn_calls == []
        final = _job(job_id)
        assert final["status"] == "cancelled"
        assert final["exit_code"] is None
    finally:
        job_mgr._JOBS.pop(job_id, None)


@pytest.mark.asyncio
async def test_cancel_survives_communication_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exception after a cancel keeps the cancelled terminal state."""
    job_id = job_mgr.create_job(target_dir=".")
    try:
        started = asyncio.Event()

        class _RaisingProcess:
            pid = _NO_SUCH_PID
            returncode: int | None = None

            def __init__(self) -> None:
                self._released = asyncio.Event()

            async def communicate(self) -> tuple[bytes, bytes]:
                await self._released.wait()
                raise RuntimeError("boom")

            def terminate(self) -> None:
                self._released.set()

        proc = _RaisingProcess()

        async def fake_spawn(*_cmd: str, **_kwargs: Any) -> _RaisingProcess:
            started.set()
            return proc

        monkeypatch.setattr(job_mgr.asyncio, "create_subprocess_exec", fake_spawn)

        task = asyncio.create_task(job_mgr.execute_job_async(job_id))
        await asyncio.wait_for(started.wait(), timeout=10.0)

        assert job_mgr.cancel_job(job_id) is True
        await asyncio.wait_for(task, timeout=10.0)

        final = _job(job_id)
        assert final["status"] == "cancelled"
        assert final["error_message"] is None
    finally:
        job_mgr._JOBS.pop(job_id, None)


@pytest.mark.needs_posix
@pytest.mark.asyncio
async def test_cancel_kills_grandchild_process_group(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A cancelled job signals its whole process group, grandchildren too.

    Real subprocesses: the job child spawns a grandchild that records its pid
    and exits on SIGTERM. The grandchild inherits the job's stdout/stderr
    pipes, so a direct-child-only terminate would leave ``communicate()``
    hanging until the grandchild's 60s sleep ends — the group kill is what
    makes the cancel prompt.
    """
    job_id = job_mgr.create_job(target_dir=".")
    try:
        job_mgr._JOBS[job_id]["output_dir"] = str(tmp_path)
        gc_exited = tmp_path / "grandchild_exited"
        gc_pid_file = tmp_path / "grandchild.pid"

        grandchild_code = (
            "import atexit, os, signal, sys, time\n"
            "def _mark() -> None:\n"
            f"    with open({str(gc_exited)!r}, 'w', encoding='ascii') as fh:\n"
            "        fh.write('done')\n"
            "atexit.register(_mark)\n"
            "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
            f"with open({str(gc_pid_file)!r}, 'w', encoding='ascii') as fh:\n"
            "    fh.write(str(os.getpid()))\n"
            "time.sleep(60)\n"
        )
        child_code = (
            "import subprocess, sys, time\n"
            f"gc = subprocess.Popen([sys.executable, '-c', {grandchild_code!r}])\n"
            "time.sleep(60)\n"
        )

        def fake_build(*_args: Any, **_kwargs: Any) -> list[str]:
            return [sys.executable, "-c", child_code]

        monkeypatch.setattr(job_mgr, "build_pipeline_command", fake_build)

        real_spawn = job_mgr.asyncio.create_subprocess_exec
        captured: dict[str, Any] = {}

        async def recording_spawn(*_cmd: str, **kwargs: Any) -> Any:
            proc = await real_spawn(*_cmd, **kwargs)
            captured["proc"] = proc
            return proc

        monkeypatch.setattr(job_mgr.asyncio, "create_subprocess_exec", recording_spawn)

        task = asyncio.create_task(job_mgr.execute_job_async(job_id))

        # The grandchild writes its own pid file only AFTER its SIGTERM
        # handler is installed, so file existence guarantees the handler is
        # live and the cancel below can never win that race.
        await _poll_until(lambda: gc_pid_file.is_file())
        assert int(gc_pid_file.read_text(encoding="ascii").strip()) > 0

        assert job_mgr.cancel_job(job_id) is True
        await asyncio.wait_for(task, timeout=15.0)

        final = _job(job_id)
        assert final["status"] == "cancelled"
        assert final["error_message"] is None
        assert final["exit_code"] == -15

        # The grandchild received the group SIGTERM and exited: a pid-only
        # terminate would leave it sleeping for 60 more seconds.
        await _poll_until(lambda: gc_exited.is_file())
    finally:
        proc = job_mgr._JOBS.get(job_id, {}).get("process")
        job_mgr._JOBS.pop(job_id, None)
        if proc is not None and proc.returncode is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
