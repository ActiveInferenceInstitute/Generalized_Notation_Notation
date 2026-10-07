"""Consumer guarantees for concurrent text and JSON logging handlers."""

from __future__ import annotations

import io
import json
import logging
from concurrent.futures import ThreadPoolExecutor

import pytest

from gnn.utils.logging.logging_utils import (
    JSONFormatter,
    StructuredFormatter,
)
from gnn.utils.logging_utils import PipelineLogger


@pytest.mark.parametrize("json_first", [False, True])
def test_metadata_is_rendered_once_without_mutating_other_handlers(
    json_first: bool,
) -> None:
    streams = [io.StringIO() for _ in range(3)]
    formatters = [
        StructuredFormatter("%(message)s"),
        JSONFormatter(),
        StructuredFormatter("%(message)s"),
    ]
    logger = logging.Logger("handler-isolation", level=logging.INFO)
    order = [1, 0, 2] if json_first else [0, 1, 2]
    for index in order:
        handler = logging.StreamHandler(streams[index])
        handler.setFormatter(formatters[index])
        logger.addHandler(handler)

    PipelineLogger.log_structured(
        logger, logging.INFO, "Processed model", model="asymmetric.md", count=3
    )

    assert streams[0].getvalue() == streams[2].getvalue()
    assert streams[0].getvalue().count("model=asymmetric.md") == 1
    entry = json.loads(streams[1].getvalue())
    assert entry["message"] == "Processed model"
    assert entry["data"] == {"model": "asymmetric.md", "count": 3}


def test_correlation_is_shared_by_formats_and_isolated_between_threads() -> None:
    def log_thread(index: int) -> tuple[str, dict]:
        PipelineLogger.set_correlation_context(f"step-{index}", f"run-{index}")
        record = logging.LogRecord(
            "parallel", logging.INFO, __file__, 1, "source %s", (index,), None
        )
        try:
            text = StructuredFormatter(
                "%(correlation_id)s:%(step_name)s %(message)s"
            ).format(record)
            return text, json.loads(JSONFormatter().format(record))
        finally:
            PipelineLogger.clear_correlation_context()

    with ThreadPoolExecutor(max_workers=4) as executor:
        emitted = list(executor.map(log_thread, range(12)))

    for index, (text, entry) in enumerate(emitted):
        assert text == f"run-{index}:step-{index} source {index}"
        assert entry["correlation_id"] == f"run-{index}"
        assert entry["step_name"] == f"step-{index}"
        assert entry["message"] == f"source {index}"


def test_timed_operation_reports_and_preserves_the_original_exception() -> None:
    stream = io.StringIO()
    logger = logging.Logger("failed-operation", level=logging.INFO)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    cause = ValueError("matrix row 2 has invalid mass")

    with pytest.raises(ValueError) as raised:
        with PipelineLogger.timed_operation("validate matrix", logger):
            raise cause

    assert raised.value is cause
    entries = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert entries[-1]["level"] == "ERROR"
    assert entries[-1]["data"]["event_type"] == "operation_error"
    assert entries[-1]["data"]["error"] == str(cause)
