"""Observable behavior of the structured logger surface beyond telemetry.

Complements ``test_structured_logging_contract.py``: every record here is
captured from a real ``logging`` handler, and every metric from a real
``LogAggregator``; nothing is mocked except the process probe failure.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import pytest

from gnn.utils.observability import structured_logging as sl

pytestmark = pytest.mark.fast


class Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def messages(self, level: int | None = None) -> list[str]:
        return [
            r.getMessage() for r in self.records if level is None or r.levelno == level
        ]


@pytest.fixture
def captured(request: pytest.FixtureRequest) -> tuple[sl.StructuredLogger, Capture]:
    logger = sl.StructuredLogger(f"gnn.test.surface.{request.node.name}", "cid-1")
    capture = Capture()
    logger.base_logger.addHandler(capture)
    logger.base_logger.propagate = False
    yield logger, capture
    logger.base_logger.removeHandler(capture)


def test_metrics_duration_and_memory_delta() -> None:
    open_metrics = sl.PerformanceMetrics(start_time=time.time() - 5)
    assert open_metrics.duration_seconds >= 5
    assert open_metrics.memory_delta_mb == 0.0
    closed = sl.PerformanceMetrics(
        start_time=10.0, end_time=12.5, memory_start_mb=100.0, memory_end_mb=96.0
    )
    assert closed.duration_seconds == 2.5
    assert closed.memory_delta_mb == -4.0


def test_text_and_json_records_carry_correlation_and_context(
    captured: tuple[sl.StructuredLogger, Capture],
) -> None:
    logger, capture = captured
    logger.set_context(step_name="3_gnn", unknown_field="ignored")
    assert logger.context.step_name == "3_gnn"
    assert not hasattr(logger.context, "unknown_field")
    for method, level in (
        (logger.debug, logging.DEBUG),
        (logger.warning, logging.WARNING),
        (logger.critical, logging.CRITICAL),
    ):
        method("plain")
        assert capture.messages(level)[-1] == "[cid-1] plain"
    logger.info("with extra", rows=3)
    assert capture.messages()[-2:] == [
        "[cid-1] with extra",
        "[cid-1] Extra data: {'rows': 3}",
    ]

    logger.log_format = sl.LogFormat.JSON
    logger.context.operation = "op"
    logger.performance_metrics["op"] = sl.PerformanceMetrics(
        start_time=1.0, end_time=3.0, operation_count=2, error_count=1
    )
    logger.error("as json", code=7)
    record = json.loads(capture.messages(logging.ERROR)[-1])
    assert record["correlation_id"] == "cid-1"
    assert record["step_name"] == "3_gnn" and record["operation"] == "op"
    assert record["level"] == "ERROR" and record["code"] == 7
    assert record["performance"] == {
        "duration_seconds": 2.0,
        "memory_delta_mb": 0.0,
        "operation_count": 2,
        "error_count": 1,
    }


@pytest.mark.parametrize(
    "status, level, phrase",
    [
        ("SUCCESS", logging.INFO, "completed successfully"),
        ("SUCCESS_WITH_WARNINGS", logging.WARNING, "completed with warnings"),
        ("FAILED", logging.ERROR, "failed"),
        ("SKIPPED", logging.INFO, "completed with status: SKIPPED"),
    ],
)
def test_pipeline_step_status_selects_level(
    captured: tuple[sl.StructuredLogger, Capture], status: str, level: int, phrase: str
) -> None:
    logger, capture = captured
    logger.log_format = sl.LogFormat.JSON
    logger.log_pipeline_step("7_export", status, duration=1.23456, rows=1)
    record = json.loads(capture.messages(level)[-1])
    assert phrase in record["message"]
    assert record["event_type"] == "pipeline_step"
    assert record["duration_seconds"] == 1.235
    assert logger.context.step_name == "7_export"


def test_log_error_records_type_traceback_and_context(
    captured: tuple[sl.StructuredLogger, Capture],
) -> None:
    logger, capture = captured
    logger.log_format = sl.LogFormat.JSON
    try:
        raise ValueError("bad matrix")
    except ValueError as exc:
        logger.log_error(exc, step_name="5_type_checker", file="m.md")
    record = json.loads(capture.messages(logging.ERROR)[-1])
    assert record["error_type"] == "ValueError"
    assert record["error_message"] == "bad matrix"
    assert "ValueError: bad matrix" in record["traceback"]
    assert record["file"] == "m.md" and record["event_type"] == "error"
    assert record["message"] == "Error in 5_type_checker: bad matrix"
    logger.log_error(RuntimeError("x"))
    unnamed = json.loads(capture.messages(logging.ERROR)[-1])
    assert unnamed["message"] == "Error in unknown: x"
    assert unnamed["step_name"] == "5_type_checker"  # context is retained


def test_child_logger_shares_correlation_and_context(
    captured: tuple[sl.StructuredLogger, Capture],
) -> None:
    logger, _ = captured
    child = logger.create_child_logger("child")
    assert child.name == f"{logger.name}.child"
    assert child.correlation_id == "cid-1"
    assert child.context is logger.context


def test_end_tracking_failure_is_non_fatal(
    captured: tuple[sl.StructuredLogger, Capture], monkeypatch: pytest.MonkeyPatch
) -> None:
    logger, capture = captured
    logger.performance_metrics["op"] = sl.PerformanceMetrics(start_time=time.time())

    def broken() -> Any:
        raise OSError("probe gone")

    monkeypatch.setattr(sl, "_process_handle", broken)
    logger._end_performance_tracking("op")
    assert logger.performance_metrics["op"].end_time is None
    assert any(
        "Performance tracking end failed (non-fatal): probe gone" in m
        for m in capture.messages(logging.DEBUG)
    )


def test_formatter_prefixes_correlation_and_extra_data() -> None:
    formatter = sl.StructuredFormatter("cid-f")
    record = logging.LogRecord("n", logging.INFO, __file__, 1, "hello", None, None)
    assert formatter.format(record).startswith("[cid-f] ")
    assert record.correlation_id == "cid-f"
    record.extra_data = {"k": 1}
    assert formatter.format(record).endswith("[INFO] n: hello | {'k': 1}")


def test_aggregator_counts_success_rates_and_saves(tmp_path: Path) -> None:
    aggregator = sl.LogAggregator()
    for status, duration in (
        ("SUCCESS", 2),
        ("SUCCESS_WITH_WARNINGS", 4),
        ("FAILED", 6),
    ):
        aggregator.add_log(
            {
                "event_type": "pipeline_step",
                "step_name": "3_gnn",
                "status": status,
                "duration_seconds": duration,
            }
        )
    aggregator.add_log({"event_type": "pipeline_step", "status": "FAILED"})
    aggregator.add_log({"event_type": "other"})
    summary = aggregator.generate_summary()
    assert summary["total_logs"] == 5
    step = summary["step_metrics"]["3_gnn"]
    assert (step["count"], step["successes"], step["failures"]) == (3, 2, 1)
    assert step["success_rate"] == pytest.approx(2 / 3)
    assert step["average_duration"] == 4
    assert summary["step_metrics"]["unknown"]["failures"] == 1
    saved = aggregator.save_summary(tmp_path)
    assert saved.parent == tmp_path
    assert json.loads(saved.read_text())["step_metrics"]["3_gnn"]["count"] == 3


def test_thread_correlation_context_and_system_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sl.set_correlation_context("thread-cid")
    try:
        assert sl.get_pipeline_logger("gnn.test.thread").correlation_id == "thread-cid"
    finally:
        del sl._correlation_context.correlation_id
    assert sl.get_pipeline_logger("gnn.test.thread").correlation_id.startswith("gnn_")
    info = sl.get_system_info()
    assert set(info) == {"platform", "python_version", "cpu_count", "hostname"}

    def fail() -> str:
        raise OSError("no platform")

    monkeypatch.setattr(sl.platform, "platform", fail)
    assert sl.get_system_info() == {"error": "Could not retrieve system info"}


def test_convenience_helpers_accept_structured_and_standard_loggers(
    captured: tuple[sl.StructuredLogger, Capture],
) -> None:
    logger, capture = captured
    logger.log_format = sl.LogFormat.JSON
    sl.log_pipeline_start(logger, "gnn", run="r1")
    sl.log_pipeline_complete(logger, "gnn", "SUCCESS", 1.23456)
    sl.log_step_start(logger, "3_gnn")
    sl.log_step_success(logger, "ok")
    sl.log_step_warning(logger, "careful")
    sl.log_step_error(logger, "broken")
    events = [json.loads(m) for m in capture.messages() if m.startswith("{")]
    assert [e["event_type"] for e in events] == [
        "pipeline_start",
        "pipeline_complete",
        "step_start",
        "step_success",
        "step_warning",
        "step_error",
    ]
    assert events[0]["run"] == "r1" and "platform" in events[0]["system_info"]
    assert events[1]["total_duration_seconds"] == 1.235

    plain = logging.getLogger("gnn.test.surface.plain")
    plain_capture = Capture()
    plain.addHandler(plain_capture)
    plain.setLevel(logging.DEBUG)
    plain.propagate = False
    try:
        sl.log_pipeline_start(plain, "gnn")
        sl.log_pipeline_complete(plain, "gnn", "FAILED", 2.0)
        sl.log_step_start(plain, "3_gnn")
        sl.log_step_success(plain, "ok")
        sl.log_step_warning(plain, "careful")
        sl.log_step_error(plain, "broken")
    finally:
        plain.removeHandler(plain_capture)
    assert plain_capture.messages() == [
        "🚀 Starting pipeline: gnn",
        "🏁 Pipeline gnn completed with status: FAILED in 2.00s",
        "🚀 Starting step: 3_gnn",
        "✅ ok",
        "⚠️ careful",
        "❌ broken",
    ]
