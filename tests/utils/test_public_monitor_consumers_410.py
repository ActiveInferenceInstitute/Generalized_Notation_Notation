"""Public monitoring consumers of real, offline supervised worker receipts."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import pytest

from gnn.utils.pipeline_orchestration import pipeline_monitor as monitoring
from gnn.utils.pipeline_orchestration.execution_utils import execute_command_streaming


def test_saved_report_preserves_native_failure_after_notification_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    delivered = []

    def unavailable_notification(alert: monitoring.Alert) -> None:
        raise OSError("notification destination unavailable")

    monitor = monitoring.PipelineMonitor([unavailable_notification, delivered.append])
    monitor.start_monitoring()
    failure_id = monitor.record_step_start("native-failure")
    started = time.monotonic()
    failed = execute_command_streaming(
        [
            sys.executable,
            "-c",
            "import sys; print('worker-started'); "
            "print('critical timeout diagnostic', file=sys.stderr); sys.exit(7)",
        ],
        cwd=tmp_path,
        timeout=3,
    )
    failure_duration = time.monotonic() - started
    assert failed["status"] == "FAILED" and failed["exit_code"] == 7, failed
    assert failed["cleanup_verified"] and failed["streams_drained"], failed
    streams = capsys.readouterr()
    assert "worker-started" in streams.out
    assert "critical timeout diagnostic" in streams.err
    monitor.record_step_warning("native-failure", failure_id, failed["stderr"])
    monitor.record_step_failure(
        "native-failure", failure_id, failure_duration, "NativeExit7", failed["stderr"]
    )

    success_id = monitor.record_step_start("native-success")
    started = time.monotonic()
    succeeded = execute_command_streaming(
        [sys.executable, "-c", "print('completed')"],
        cwd=tmp_path,
        timeout=3,
        print_stdout=False,
        print_stderr=False,
    )
    assert succeeded["status"] == "SUCCESS", succeeded
    assert succeeded["cleanup_verified"] and succeeded["streams_drained"], succeeded
    monitor.record_step_success(
        "native-success", success_id, time.monotonic() - started
    )
    pending_id = monitor.record_step_start("not-completed")
    assert pending_id
    report_path = monitor.save_health_report(tmp_path / "reports")
    report = json.loads(report_path.read_text())

    assert report["overall_health"]["status"] == "critical"
    assert report["overall_health"]["total_executions"] == 3
    assert report["overall_health"]["total_failures"] == 1
    assert report["overall_health"]["healthy_steps"] == 1
    assert report["step_details"]["native-failure"]["error_types"] == {"NativeExit7": 1}
    assert report["step_details"]["native-failure"]["last_failure"]
    assert report["step_details"]["native-success"]["last_success"]
    assert report["step_details"]["not-completed"]["health_status"] == "unknown"
    assert report["step_details"]["not-completed"]["min_duration"] == 0
    assert (
        report["performance_summary"]["least_reliable_step"]["name"] == "native-failure"
    )
    assert (
        report["performance_summary"]["most_reliable_step"]["name"] == "native-success"
    )
    assert len(delivered) == len(report["recent_alerts"]) == 2
    assert "Alert callback failed: notification destination unavailable" in caplog.text
    assert report["recent_alerts"][-1]["details"]["execution_id"] == failure_id
    monitor.resolve_alert(0)
    updated = json.loads(
        monitor.save_health_report(tmp_path / "reports", "resolved.json").read_text()
    )
    assert updated["recent_alerts"][0]["resolved"] is True
    assert updated["overall_health"]["active_alerts"] == 1
    assert updated["overall_health"]["total_failures"] == 1
    monitor.stop_monitoring()


@pytest.mark.parametrize("ratio,level", [(2.5, "warning"), (4.0, "critical")])
def test_native_duration_baseline_alert_is_preserved_in_saved_report(
    tmp_path: Path, ratio: float, level: str
) -> None:
    monitor = monitoring.PipelineMonitor()
    execution_id = monitor.record_step_start("native-worker")
    started = time.monotonic()
    result = execute_command_streaming(
        [sys.executable, "-c", "print('baseline-witness')"],
        cwd=tmp_path,
        timeout=3,
        print_stdout=False,
        print_stderr=False,
    )
    duration = time.monotonic() - started
    assert result["status"] == "SUCCESS", result
    # Calibrate from this real receipt rather than relying on host speed.
    monitor.set_performance_baseline("native-worker", duration / ratio)
    monitor.record_step_success("native-worker", execution_id, duration)
    report = json.loads(
        monitor.save_health_report(tmp_path, "baseline.json").read_text()
    )
    (alert,) = report["recent_alerts"]
    assert alert["level"] == level
    assert alert["step_name"] == "native-worker"
    assert alert["details"]["ratio"] == pytest.approx(ratio)
    assert report["overall_health"]["total_failures"] == 0


def test_public_health_facade_consumes_saved_failure_summary(tmp_path: Path) -> None:
    from gnn.utils import generate_pipeline_health_report

    started = time.monotonic()
    receipt = execute_command_streaming(
        [sys.executable, "-c", "print('failed operation'); raise SystemExit(9)"],
        cwd=tmp_path,
        timeout=3,
        print_stdout=False,
        print_stderr=False,
    )
    assert receipt["status"] == "FAILED" and receipt["exit_code"] == 9, receipt
    summary_path = tmp_path / "external_worker_summary.json"
    # This is a consumer's wire projection of an actual worker, not a GNN
    # current-run receipt and not evidence of pipeline artifact ownership.
    summary_path.write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "script_name": "native-failure",
                        "status": receipt["status"],
                        "duration_seconds": time.monotonic() - started,
                    }
                ],
                "performance_summary": {"successful_steps": 0, "critical_failures": 1},
            }
        )
    )
    health = generate_pipeline_health_report(
        json.loads(summary_path.read_text()), logging.getLogger(__name__)
    )
    (tmp_path / "health.json").write_text(json.dumps(health))
    saved = json.loads((tmp_path / "health.json").read_text())
    assert saved["overall_health"] == "critical"
    assert saved["execution_efficiency"] == 0
    assert saved["step_performance"]["native-failure"]["status"] == "FAILED"
    assert saved["critical_issues"] == ["1 critical step failures detected"]
    assert any("error handling" in item for item in saved["recommendations"])
    assert any("step logs" in item for item in saved["recommendations"])
