"""Step12 keeps the supervised subprocess cleanup verdict in persisted receipts."""

import contextlib
import json
import logging
import sys
import time
from pathlib import Path

import psutil
import pytest

from gnn.execute.processor import single
from gnn.utils.runtime_safety.framework_availability import FrameworkStatus


def test_cleanup_failure_survives_execution_and_persisted_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    script = tmp_path / "fixture.py"
    script.write_text('print("fixture")\n')
    cleanup = {
        "success": False,
        "return_code": -9,
        "stdout": "",
        "stderr": "",
        "error_type": "ProcessCleanupFailure",
        "error": "child cleanup unverified",
        "execution_error_type": "TimeoutExpired",
        "containment": "process_group_and_observed_descendants",
        "cleanup_verified": False,
        "streams_drained": False,
        "cleanup_timeout_seconds": 0.2,
        "cleanup_error": "pipe remained open",
        "cancelled": False,
    }
    monkeypatch.setattr(
        single, "run_subprocess_envelope", lambda *args, **kwargs: cleanup
    )
    result = single.execute_single_script(
        {
            "path": script,
            "name": script.name,
            "framework": "python",
            "executor": sys.executable,
        },
        tmp_path / "output",
        False,
        logging.getLogger(__name__),
    )
    assert not result["success"] and result["error_type"] == "ProcessCleanupFailure"
    assert result["status"] == "failed"
    persisted = json.loads(Path(result["structured_result_file"]).read_text())
    for key in (
        "containment",
        "cleanup_verified",
        "streams_drained",
        "cleanup_timeout_seconds",
        "cleanup_error",
        "execution_error_type",
        "error_type",
    ):
        assert persisted[key] == result[key] == cleanup[key]


def _kill_owned_wrapper_child(marker: Path) -> None:
    if not marker.is_file():
        return
    pid, birth = json.loads(marker.read_text())
    with contextlib.suppress(psutil.NoSuchProcess):
        child = psutil.Process(pid)
        if abs(child.create_time() - birth) < 0.01:
            child.kill()
            with contextlib.suppress(psutil.TimeoutExpired, ChildProcessError):
                child.wait(timeout=2)


@pytest.mark.pipeline
@pytest.mark.toolchain
@pytest.mark.needs_posix
@pytest.mark.parametrize("framework", ["python", "pymdp", "jax"])
def test_dispatch_does_not_repeat_raw_version_or_import_probes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, framework: str
) -> None:
    from gnn.execute import processor

    monkeypatch.setattr(
        processor,
        "_check_framework_by_name",
        lambda name, **kwargs: FrameworkStatus(name, True),
    )
    monkeypatch.setattr(single, "_detect_accelerator_type", lambda: "cpu")
    script = tmp_path / "model" / framework / "fixture.py"
    script.parent.mkdir(parents=True)
    script.write_text('print("requested-script")\n')
    marker = tmp_path / "raw-probe-child.json"
    wrapper = tmp_path / "wrapped-python"
    wrapper.write_text(
        "#!" + sys.executable + "\n"
        "import json, os, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "import psutil\n"
        "if sys.argv[1] in ('--version', '-c'):\n"
        "    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'])\n"
        f"    Path({str(marker)!r}).write_text(json.dumps([child.pid, psutil.Process(child.pid).create_time()]))\n"
        "    time.sleep(20)\n"
        "else:\n"
        "    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
    )
    wrapper.chmod(0o700)
    try:
        result = single.execute_single_script(
            {
                "path": script,
                "name": script.name,
                "framework": framework,
                "executor": str(wrapper),
            },
            tmp_path / "output",
            False,
            logging.getLogger(__name__),
            timeout=2,
        )
        assert result["success"] and result["status"] == "success", result
        assert "requested-script" in result["stdout"]
        assert result["cleanup_verified"] and result["streams_drained"]
        assert not marker.exists(), "dispatch repeated a redundant unsupervised probe"
        persisted = json.loads(Path(result["structured_result_file"]).read_text())
        assert persisted["status"] == "success"
    finally:
        _kill_owned_wrapper_child(marker)


@pytest.mark.pipeline
@pytest.mark.toolchain
@pytest.mark.needs_posix
def test_actual_executor_wrapper_timeout_reaps_grandchild_and_records_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(single, "_detect_accelerator_type", lambda: "cpu")
    script = tmp_path / "model" / "python" / "fixture.py"
    script.parent.mkdir(parents=True)
    script.write_text('print("fixture")\n')
    marker = tmp_path / "child.json"
    wrapper = tmp_path / "slow-python"
    wrapper.write_text(
        "#!" + sys.executable + "\n"
        "import json, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "import psutil\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'])\n"
        f"Path({str(marker)!r}).write_text(json.dumps([child.pid, psutil.Process(child.pid).create_time()]))\n"
        "print('partial-executor-output', flush=True)\n"
        "time.sleep(20)\n"
    )
    wrapper.chmod(0o700)
    started = time.monotonic()
    try:
        result = single.execute_single_script(
            {
                "path": script,
                "name": script.name,
                "framework": "python",
                "executor": str(wrapper),
            },
            tmp_path / "output",
            False,
            logging.getLogger(__name__),
            timeout=1,
        )
        assert time.monotonic() - started < 1.4
        assert not result["success"] and result["status"] == "timed_out", result
        assert result["error_type"] == "TimeoutExpired"
        assert result["cleanup_verified"] and result["streams_drained"]
        assert "partial-executor-output" in result["stdout"]
        assert marker.is_file(), "wrapper did not launch the real child"
        pid, birth = json.loads(marker.read_text())
        with contextlib.suppress(psutil.NoSuchProcess):
            child = psutil.Process(pid)
            assert (
                abs(child.create_time() - birth) >= 0.01
                or child.status() == psutil.STATUS_ZOMBIE
            )
        persisted = json.loads(Path(result["structured_result_file"]).read_text())
        assert persisted["status"] == "timed_out" and not persisted["success"]
    finally:
        _kill_owned_wrapper_child(marker)


def test_typed_cancellation_keeps_cancelled_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "fixture.py"
    script.write_text('print("fixture")\n')
    monkeypatch.setattr(single, "_detect_accelerator_type", lambda: "cpu")
    monkeypatch.setattr(
        single,
        "run_subprocess_envelope",
        lambda *args, **kwargs: {
            "success": False,
            "return_code": -1,
            "stdout": "partial",
            "stderr": "",
            "error_type": "Cancelled",
            "error": "operator cancellation",
            "cancelled": True,
            "cleanup_verified": True,
            "streams_drained": True,
        },
    )
    result = single.execute_single_script(
        {
            "path": script,
            "name": script.name,
            "framework": "python",
            "executor": sys.executable,
        },
        tmp_path / "output",
        False,
        logging.getLogger(__name__),
    )
    assert result["status"] == "cancelled" and result["error_type"] == "Cancelled"
    assert result["cancelled"] and not result["success"]
    assert result["stdout"] == "partial" and result["cleanup_verified"]
    persisted = json.loads(Path(result["structured_result_file"]).read_text())
    assert persisted["status"] == "cancelled" and not persisted["success"]
