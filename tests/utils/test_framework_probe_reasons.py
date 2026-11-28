"""Probe causes survive structured readiness and timeout boundaries."""

import json
import logging
import subprocess
import sys
import time

import psutil
import pytest

from gnn.utils.runtime_safety import framework_availability as availability


def _child_is_gone_or_zombie(pid: int, *, timeout: float = 0.0) -> bool:
    """Check one process identity, optionally waiting for its kernel exit."""
    try:
        child = psutil.Process(pid)
        deadline = time.monotonic() + timeout
        while child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(0.01, remaining))
    except psutil.NoSuchProcess:
        return True
    return True


@pytest.mark.parametrize(
    "code",
    ["missing_module", "missing_toolchain", "unsupported_python", "probe_failed"],
)
def test_probe_reason_is_not_conflated_with_missing_module(
    monkeypatch: pytest.MonkeyPatch, code: str
) -> None:
    answer = {"available": False, "reason_code": code, "reason": code}
    if code == "missing_module":
        answer["missing_module"] = "cmdstanpy"
    monkeypatch.setattr(
        availability,
        "_run_probe_envelope",
        lambda *args, **kwargs: {
            "success": True,
            "return_code": 0,
            "stdout": "library chatter\nGNN_FRAMEWORK_STATUS:" + json.dumps(answer),
        },
    )
    status = availability.check_framework("stan")
    assert status.reason_code == code
    assert status.missing_module == ("cmdstanpy" if code == "missing_module" else None)
    if code == "missing_toolchain":
        assert "install_cmdstan" in status.install_hint


def test_actual_child_timeout_is_distinct(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = tmp_path / "slow_probe.py"
    module.write_text("import time\ntime.sleep(5)\n")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setitem(
        availability.FRAMEWORK_IMPORT_CHECK, "test_slow", ("slow_probe", "unused")
    )
    status = availability.check_framework("test_slow", timeout=0.1)
    assert status.reason_code == "probe_timeout"
    assert status.missing_module is None and status.install_hint is None


def test_actual_missing_executor_is_distinct() -> None:
    status = availability.check_framework("stan", executor="/no/such/python")
    assert status.reason_code == "executor_unavailable"


def test_ngclearn_reports_target_python_gate() -> None:
    # The package floor is 3.12, so the host interpreter never trips the
    # ngc-learn Python gate; the gate remains for older external executors.
    status = availability.check_framework("ngclearn", executor=sys.executable)
    assert status.reason_code != "unsupported_python"


@pytest.mark.parametrize("version", ["0.4.4", "0.5.0"])
def test_actual_wrong_cpomdp_version_is_diagnosed(
    tmp_path, monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    (tmp_path / "cpomdp.py").write_text(f'__version__ = "{version}"\n')
    info = tmp_path / f"cpomdp-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: cpomdp\nVersion: {version}\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    status = availability.check_framework("cpomdp")
    assert status.reason_code == "unsupported_version"
    assert "0.5.1" in status.reason and version in status.reason
    assert status.missing_module is None


def test_actual_selected_cpomdp_version_is_admitted_without_native_skip_gate(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "cpomdp.py").write_text('__version__ = "0.5.1"\n')
    info = tmp_path / "cpomdp-0.5.1.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: cpomdp\nVersion: 0.5.1\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    status = availability.check_framework("cpomdp")
    assert status.available is True, status
    assert status.reason_code is None and status.install_hint is None


def test_invocation_cache_never_dispatches_after_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from gnn.pipeline import run_context

    context = SimpleNamespace(
        run_id="probe-run", bounded_timeout=lambda timeout: timeout
    )
    monkeypatch.setattr(run_context, "current_run_context", lambda: context)
    calls = []

    def probe(*args, **kwargs):
        calls.append(kwargs["timeout"])
        return {
            "success": True,
            "return_code": 0,
            "stdout": 'GNN_FRAMEWORK_STATUS:{"available":true}',
        }

    monkeypatch.setattr(availability, "_run_probe_envelope", probe)
    availability._CONTEXT_PROBE_CACHE.clear()
    assert availability.check_framework("jax", timeout=2).available
    assert availability.check_framework("jax", timeout=2).available
    assert calls == [2]
    context.bounded_timeout = lambda timeout: 0
    assert availability.check_framework("jax").reason_code == "probe_timeout"
    assert calls == [2]
    context.run_id = "different-run"
    context.bounded_timeout = lambda timeout: 0.5
    assert availability.check_framework("jax").available
    assert calls == [2, 0.5]


def test_boolean_probe_timeout_is_rejected() -> None:
    with pytest.raises(ValueError):
        availability.check_framework("jax", timeout=True)


def test_import_spawned_child_cannot_hold_probe_pipes_open(
    tmp_path, monkeypatch
) -> None:
    """A real importing module inherits pipes into a child; both are terminated."""
    from gnn.execute.subprocess_envelope import run_subprocess_envelope  # noqa: F401

    pid_file = tmp_path / "child.pid"
    (tmp_path / "child_probe.py").write_text(
        "import pathlib, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(child.pid))\n"
        "time.sleep(30)\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setitem(
        availability.FRAMEWORK_IMPORT_CHECK, "test_child", ("child_probe", "unused")
    )
    started = time.monotonic()
    try:
        status = availability.check_framework("test_child", timeout=1.0)
        assert time.monotonic() - started < 2.5
        assert pid_file.exists(), (
            "The import must actually spawn the pipe-holding child"
        )
        pid = int(pid_file.read_text())
        assert status.reason_code == "probe_timeout"
        assert status.execution_error_type == "TimeoutExpired"
        assert status.cleanup_verified is True and status.streams_drained is True
        # Pipe closure can arrive before the killed child reports its final
        # kernel status. Observe that identity briefly after the probe returns.
        assert _child_is_gone_or_zombie(pid, timeout=0.5), (
            "the spawned child must not survive the probe"
        )

        # The reap can land between the existence check and the status read, so the
        # vanished-pid path is asserted directly: the raw read raises NoSuchProcess
        # and the helper treats that as "gone" instead of letting the race decide
        # the outcome. A sleep here would only make the same race slower.
        reaped = subprocess.Popen([sys.executable, "-c", "pass"])
        reaped.wait()
        with pytest.raises(psutil.NoSuchProcess):
            psutil.Process(reaped.pid).status()
        assert _child_is_gone_or_zombie(reaped.pid)
    finally:
        if pid_file.exists():
            try:
                child = psutil.Process(int(pid_file.read_text()))
                child.kill()
                child.wait(timeout=2)
            except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                pass


def test_cleanup_failure_retains_timeout_cause_in_readiness_receipt(
    monkeypatch,
) -> None:
    from gnn.execute.processor.envelope import _make_skipped_result

    monkeypatch.setattr(
        availability,
        "_run_probe_envelope",
        lambda *args, **kwargs: {
            "success": False,
            "return_code": None,
            "error_type": "ProcessCleanupFailure",
            "execution_error_type": "TimeoutExpired",
            "cleanup_verified": False,
            "streams_drained": False,
        },
    )
    status = availability.check_framework("jax")
    assert status.reason_code == "probe_failed"
    assert status.missing_module is None and status.install_hint is None
    assert "cleanup" in status.reason
    receipt = _make_skipped_result(
        {"name": "example.py", "path": "/tmp/model/jax/example.py"},
        "jax",
        "model",
        sys.executable,
        logging.getLogger(__name__),
        status,
    )
    # Containment failure takes priority over the probe's original timeout;
    # neither uncertainty may become an optional dependency skip.
    assert receipt["error_type"] == "ProcessCleanupFailure"
    assert receipt["status"] == "failed"
    assert receipt["success"] is False and receipt["skipped"] is False
    assert receipt["execution_error_type"] == "TimeoutExpired"
    assert receipt["cleanup_verified"] is False and receipt["streams_drained"] is False
