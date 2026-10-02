"""Real subprocess/artifact boundaries; native inference has separate receipts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import psutil
import pytest

from gnn.execute.subprocess_envelope import CancelToken
from gnn.execute.thrml import execute_thrml_script, run_thrml_records, run_thrml_scripts
from gnn.execute.thrml import thrml_runner as runner
from gnn.pipeline.output_lease import OutputLease
from gnn.utils.runtime_safety import framework_availability as availability
from gnn.utils.runtime_safety.framework_availability import FrameworkStatus
from tests.helpers.thrml_results import write_protocol_script


@pytest.fixture
def protocol_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    # Deliberate orchestration injection; no native inference claim is made.
    monkeypatch.setattr(
        runner,
        "check_framework",
        lambda *args, **kwargs: FrameworkStatus("thrml", True),
    )


def test_new_public_packages_are_cold_and_optional_import_free() -> None:
    code = "import sys; import gnn.execute.thrml, gnn.analysis.thrml; assert 'thrml' not in sys.modules; assert 'jax' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=10)


def test_real_child_protocol_identity_and_scientific_receipt(
    tmp_path: Path, protocol_ready: None
) -> None:
    script = write_protocol_script(tmp_path / "case_thrml.py")
    result = execute_thrml_script(script, output_dir=tmp_path / "out", timeout=5)
    assert result["success"], result
    assert result["script_path"] == str(script.resolve())
    assert (
        result["simulation_data"]["runtime_metadata"]["execution_id"]
        == result["execution_id"]
    )
    assert result["scientific_analysis"]["inference_mode"] == "gibbs_monte_carlo"
    assert result["cleanup_verified"] is True
    assert result["streams_drained"] is True
    assert (
        tmp_path / "out" / "stdout.txt"
    ).read_text().strip() == "protocol-child-completed"
    assert json.loads((tmp_path / "out" / "execution_log.json").read_text())["success"]


def test_stale_artifact_cannot_verify_the_next_run(
    tmp_path: Path, protocol_ready: None
) -> None:
    script = write_protocol_script(tmp_path / "case_thrml.py", once=True)
    first = execute_thrml_script(script, output_dir=tmp_path / "out", timeout=5)
    assert first["success"]
    second = execute_thrml_script(script, output_dir=tmp_path / "out", timeout=5)
    assert not second["success"] and second["error_type"] == "InvalidThrmlResult"
    assert "identity" in second["error"]
    assert second["execution_id"] != first["execution_id"]
    assert second["execution_result_success"] is True
    assert "protocol-child-completed" in second["stdout"]


@pytest.mark.parametrize(
    "mutate",
    [
        "payload['runtime_metadata']['script_sha256'] = 'c' * 64",
        "payload['beliefs'][0] = [0.8, 0.2]",
        "payload['validation']['all_conditionals_valid'] = False",
    ],
)
def test_invalid_identity_or_sampling_witness_never_verifies(
    tmp_path: Path, protocol_ready: None, mutate: str
) -> None:
    script = write_protocol_script(tmp_path / "case_thrml.py", mutate=mutate)
    result = execute_thrml_script(script, output_dir=tmp_path / "out", timeout=5)
    assert not result["success"] and result["error_type"] == "InvalidThrmlResult"


def test_child_created_result_link_cannot_verify_an_external_artifact(
    tmp_path: Path, protocol_ready: None
) -> None:
    script = write_protocol_script(tmp_path / "case_thrml.py")
    external = tmp_path / "external.json"
    script.write_text(
        script.read_text()
        + f"external = Path({str(external)!r})\n"
        + "external.write_text(target.read_text())\n"
        + "target.unlink()\n"
        + "target.symlink_to(external)\n"
    )
    result = execute_thrml_script(script, output_dir=tmp_path / "out", timeout=5)
    assert not result["success"] and result["error_type"] == "InvalidThrmlResult"
    assert result["execution_result_success"] is True
    assert "protocol-child-completed" in result["stdout"]
    assert external.is_file()


@pytest.mark.parametrize(
    "reason,expected",
    [
        ("missing_module", "skipped"),
        ("unsupported_version", "skipped"),
        ("probe_timeout", "timed_out"),
        ("probe_failed", "failed"),
    ],
)
def test_structured_readiness_cause_survives_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reason: str, expected: str
) -> None:
    script = tmp_path / "case_thrml.py"
    script.write_text("print('safe')")
    monkeypatch.setattr(
        runner,
        "check_framework",
        lambda *args, **kwargs: FrameworkStatus(
            "thrml", False, reason_code=reason, reason=reason
        ),
    )
    result = execute_thrml_script(script, output_dir=tmp_path / "out", timeout=5)
    assert not result["success"] and result["status"] == expected
    assert result["reason_code"] == reason
    assert not (tmp_path / "out").exists()


def _slow_dependency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    dependency = tmp_path / "dependency"
    dependency.mkdir()
    pid = dependency / "pid"
    (dependency / "thrml.py").write_text(
        f"import os,time,pathlib\npathlib.Path({str(pid)!r}).write_text(str(os.getpid()))\ntime.sleep(30)\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(dependency))
    return pid


def _assert_reaped(pid: Path) -> None:
    assert pid.exists()
    try:
        process = psutil.Process(int(pid.read_text()))
    except psutil.NoSuchProcess:
        return
    assert process.status() == psutil.STATUS_ZOMBIE


def test_actual_slow_thrml_import_timeout_is_budgeted_and_reaped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid = _slow_dependency(tmp_path, monkeypatch)
    script = tmp_path / "case_thrml.py"
    script.write_text("print('safe')")
    started = time.monotonic()
    result = execute_thrml_script(script, output_dir=tmp_path / "out", timeout=1)
    assert not result["success"] and result["status"] == "timed_out"
    assert time.monotonic() - started < 2
    _assert_reaped(pid)
    assert not (tmp_path / "out").exists()


def test_actual_slow_thrml_import_cancellation_is_reaped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid = _slow_dependency(tmp_path, monkeypatch)
    token = CancelToken()
    script = tmp_path / "case_thrml.py"
    script.write_text("print('safe')")

    def cancel_after_start() -> None:
        until = time.monotonic() + 2
        while not pid.exists() and time.monotonic() < until:
            time.sleep(0.01)
        token.cancel("required cancellation")

    controller = threading.Thread(target=cancel_after_start)
    controller.start()
    try:
        result = execute_thrml_script(
            script, output_dir=tmp_path / "out", timeout=3, cancel_token=token
        )
    finally:
        controller.join(timeout=3)
    assert not controller.is_alive()
    assert not result["success"] and result["status"] == "cancelled"
    _assert_reaped(pid)
    assert not (tmp_path / "out").exists()


def test_known_probe_cleanup_failure_is_not_hidden_by_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "case_thrml.py"
    script.write_text("print('safe')")
    token = CancelToken()

    def failed(*args: Any, **kwargs: Any) -> FrameworkStatus:
        token.cancel("cancelled with cleanup failure")
        return FrameworkStatus(
            "thrml",
            False,
            reason_code="probe_failed",
            execution_error_type="Cancelled",
            cleanup_verified=False,
            streams_drained=False,
        )

    monkeypatch.setattr(runner, "check_framework", failed)
    result = execute_thrml_script(script, timeout=2, cancel_token=token)
    assert not result["success"] and result["status"] == "failed"
    assert result["error_type"] == "ProcessCleanupFailure"
    assert result["execution_error_type"] == "Cancelled"


def test_concurrent_writer_and_link_escape_fail_before_script_execution(
    tmp_path: Path, protocol_ready: None
) -> None:
    script = write_protocol_script(tmp_path / "case_thrml.py")
    output = tmp_path / "out"
    with OutputLease(output, "already-owned"):
        result = execute_thrml_script(script, output_dir=output, timeout=5)
    assert not result["success"] and result["error_type"] == "OutputLeaseError"
    assert not (output / "simulation_data").exists()
    outside = tmp_path / "unrelated"
    outside.mkdir()
    (outside / "keep").write_text("untouched")
    (output / "simulation_data").symlink_to(outside)
    result = execute_thrml_script(script, output_dir=output, timeout=5)
    assert not result["success"] and result["error_type"] == "OutputLeaseError"
    assert list(outside.iterdir()) == [outside / "keep"]


def test_explicit_empty_batch_selection_does_not_discover_or_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_protocol_script(tmp_path / "case_thrml.py")
    monkeypatch.setattr(
        runner,
        "check_framework",
        lambda *args, **kwargs: pytest.fail("empty selection must not probe"),
    )
    assert run_thrml_records(tmp_path, selected_scripts=[]) == []
    assert run_thrml_scripts(tmp_path, selected_scripts=[])


def test_empty_default_discovery_does_not_claim_artifact_success(
    tmp_path: Path,
) -> None:
    assert run_thrml_records(tmp_path) == []
    assert run_thrml_scripts(tmp_path) is False


def test_batch_duplicate_stems_remain_distinct_and_analyzable(
    tmp_path: Path, protocol_ready: None
) -> None:
    from gnn.analysis.thrml import generate_analysis_from_logs

    root = tmp_path / "render"
    for folder in ("one", "two"):
        directory = root / folder
        directory.mkdir(parents=True)
        write_protocol_script(
            directory / "same_thrml.py",
            mutate="payload['source_identity'].pop('artifact_stem'); payload['source_identity'].pop('model_id')",
        )
    output = tmp_path / "results"
    records = run_thrml_records(root, output, timeout=5)
    assert len(records) == 2 and all(record["success"] for record in records)
    assert len({record["output_dir"] for record in records}) == 2
    reports = generate_analysis_from_logs(output, tmp_path / "analysis")
    assert len(reports) == len(set(reports)) == 2
    identities = {
        json.loads(Path(path).read_text())["runtime_metadata"]["execution_id"]
        for path in reports
    }
    assert identities == {record["execution_id"] for record in records}


def test_batch_shares_one_deadline_and_preserves_sibling_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deadlines = []
    for folder in ("one", "two"):
        directory = tmp_path / folder
        directory.mkdir()
        (directory / "same_thrml.py").write_text("print('safe')")

    def child(script: Path, *args: Any, **kwargs: Any) -> dict[str, Any]:
        deadlines.append(kwargs["deadline_monotonic"])
        return {
            "script_path": str(script),
            "success": len(deadlines) == 1,
            "status": "success" if len(deadlines) == 1 else "timed_out",
        }

    monkeypatch.setattr(runner, "execute_thrml_script", child)
    records = run_thrml_records(tmp_path, tmp_path / "out", timeout=1)
    assert len(records) == 2 and records[0]["success"] and not records[1]["success"]
    assert deadlines[0] == deadlines[1]


def test_late_required_receipt_publication_is_not_success(
    tmp_path: Path, protocol_ready: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = write_protocol_script(tmp_path / "case_thrml.py")
    original = runner._persist
    calls = 0

    def slow(result: dict[str, Any], output: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            time.sleep(1.1)
        original(result, output)

    monkeypatch.setattr(runner, "_persist", slow)
    result = execute_thrml_script(script, output_dir=tmp_path / "out", timeout=1)
    assert not result["success"] and result["status"] == "timed_out"
    assert "protocol-child-completed" in result["stdout"]
    assert result["cleanup_verified"] is True
    assert not json.loads((tmp_path / "out" / "execution_log.json").read_text())[
        "success"
    ]


@pytest.mark.parametrize("mode", ["timeout", "cancelled"])
def test_real_dispatch_exhaustion_retains_partial_output_and_reaps(
    tmp_path: Path, protocol_ready: None, mode: str
) -> None:
    script = tmp_path / "case_thrml.py"
    script.write_text(
        "import os,sys,time\nfrom pathlib import Path\n"
        "Path(os.environ['THRML_OUTPUT_DIR']).joinpath('pid').write_text(str(os.getpid()))\n"
        "print('partial-sampling', flush=True)\ntime.sleep(30)\n"
    )
    output = tmp_path / "out"
    token = CancelToken()

    def cancel_after_dispatch() -> None:
        until = time.monotonic() + 2
        while not (output / "pid").exists() and time.monotonic() < until:
            time.sleep(0.01)
        token.cancel("cancel sampling")

    controller = (
        threading.Thread(target=cancel_after_dispatch) if mode == "cancelled" else None
    )
    if controller is not None:
        controller.start()
    try:
        result = execute_thrml_script(
            script,
            output_dir=output,
            timeout=3 if mode == "cancelled" else 1,
            cancel_token=token if mode == "cancelled" else None,
        )
    finally:
        if controller is not None:
            controller.join(timeout=3)
    assert not result["success"]
    assert result["status"] == ("cancelled" if mode == "cancelled" else "timed_out")
    assert "partial-sampling" in result["stdout"]
    assert result["cleanup_verified"] is True and result["streams_drained"] is True
    _assert_reaped(output / "pid")


def test_actual_wrong_released_thrml_version_reports_incompatibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "thrml.py").write_text("VERSION = '0.1.3'\n")
    info = tmp_path / "thrml-0.1.3.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: thrml\nVersion: 0.1.3\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    status = availability.check_framework("thrml", timeout=2)
    assert status.reason_code == "unsupported_version"
    assert "0.1.4" in status.reason and "0.1.3" in status.reason
    assert status.missing_module is None
