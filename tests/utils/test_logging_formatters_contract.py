"""Pins for the logging subpackage formatters and rotation (batch 8)."""

from __future__ import annotations

import gzip
import logging

from gnn.utils.logging.logging_utils import (
    JSONFormatter,
    StructuredFormatter,
)


def _record() -> logging.LogRecord:
    return logging.LogRecord(
        name="w2probe",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="structured body",
        args=(),
        exc_info=None,
    )


def test_structured_formatter_appends_structured_data() -> None:
    record = _record()
    record.structured_data = {"event_type": "step", "step": "3_gnn", "ok": 1}  # type: ignore[attr-defined]

    formatted = StructuredFormatter("%(message)s").format(record)

    assert formatted.startswith("structured body [")
    assert "step=3_gnn" in formatted
    assert "ok=1" in formatted
    assert "event_type" not in formatted  # reserved key excluded


def test_structured_formatter_passthrough_without_data() -> None:
    formatted = StructuredFormatter("%(message)s").format(_record())

    assert formatted == "structured body"


def test_json_formatter_emits_parseable_envelope() -> None:
    import json

    formatted = JSONFormatter("%(message)s").format(_record())
    entry = json.loads(formatted)

    assert entry["message"] == "structured body"
    assert entry["level"] == "INFO"
    assert entry["logger"] == "w2probe"
    assert entry["correlation_id"]
    assert entry["step_name"]
    assert "timestamp" in entry


def test_json_formatter_includes_structured_and_performance_context() -> None:
    import json

    record = _record()
    record.structured_data = {"step": "3_gnn"}  # type: ignore[attr-defined]
    record.performance_context = {"duration": 1.5}  # type: ignore[attr-defined]

    entry = json.loads(JSONFormatter("%(message)s").format(record))

    assert entry["data"] == {"step": "3_gnn"}
    assert entry["performance"] == {"duration": 1.5}


def test_json_formatter_includes_exception_info() -> None:
    import json

    try:
        raise RuntimeError("boom")
    except RuntimeError:
        import sys

        record = _record()
        record.exc_info = sys.exc_info()

    entry = json.loads(JSONFormatter("%(message)s").format(record))

    assert "boom" in entry["exception"]


def test_rotate_logs_compresses_and_prunes(tmp_path: object) -> None:
    from gnn.utils.logging_utils import rotate_logs

    log_dir = tmp_path / "logs"  # type: ignore[attr-defined]
    log_dir.mkdir()

    # Seed max_files rotated archives; the current log forces one rotation.
    for index in range(3):
        old = log_dir / f"pipeline_2026010{index}_000000.log"
        old.write_text(f"old {index}", encoding="utf-8")
    (log_dir / "pipeline.log").write_text("current", encoding="utf-8")

    rotate_logs(log_dir, max_files=2, compress=True)

    assert not (log_dir / "pipeline.log").exists()  # rotated away
    rotated = sorted(log_dir.glob("pipeline_*.log.gz"))
    assert rotated, "current log should be compressed"
    # 3 seeded + 1 rotated - pruning keeps max_files - 1 = 1 archive.
    assert len(rotated) == 1
    assert gzip.open(rotated[0], "rb").read() == b"current"


def test_rotate_logs_noop_without_current_log(tmp_path: object) -> None:
    from gnn.utils.logging_utils import rotate_logs

    rotate_logs(tmp_path, max_files=2, compress=True)  # type: ignore[attr-defined]

    assert list(tmp_path.iterdir()) == []  # type: ignore[attr-defined]
