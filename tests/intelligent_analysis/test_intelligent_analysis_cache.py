"""Tests for the step-24 LLM analysis cache wiring.

``_run_llm_analysis`` consults the content-addressed ``LLMCache`` when a
``cache_dir`` is supplied: cache hits skip the LLM call entirely, entries
persist on disk across processor instances, the model name participates in
the cache key, invalid LLM output is never cached, and
``process_intelligent_analysis`` wires the step-24 output directory's
``.cache`` through to the LLM analysis.
"""

import asyncio
import json
import logging
from collections.abc import Coroutine
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gnn.intelligent_analysis.processor import (
    _run_llm_analysis,
    process_intelligent_analysis,
)

VALID_REPORT = (
    "### Executive Summary\n"
    "The pipeline completed successfully with a perfect health score.\n\n"
    "### Red Flags (Critical Issues)\n"
    "None. All steps finished with exit code zero.\n\n"
    "### Yellow Flags (Warnings)\n"
    "None. No concerning patterns were recorded.\n\n"
    "### Root Cause Analysis\n"
    "No failures occurred, so there are no root causes to report.\n\n"
    "### Optimization Opportunities\n"
    "Consider trimming per-step durations and peak memory further.\n\n"
    "### Action Items\n"
    "- Keep the current configuration."
)

INVALID_REPORT = (
    "### Executive Summary\n"
    "The pipeline completed successfully.\n\n"
    "### Red Flags (Critical Issues)\n"
    "None.\n\n"
    "### Yellow Flags (Warnings)\n"
    "None.\n\n"
    "### Root Cause Analysis\n"
    "No failures occurred.\n\n"
    "### Optimization Opportunities\n"
    "None.\n\n"
    "Action Items heading intentionally omitted."
)

ANALYSIS_CONTEXT: dict[str, Any] = {
    "overall_status": "SUCCESS",
    "total_duration": 1.5,
    "health_score": 100.0,
    "failures": [],
    "warnings": [],
    "performance_metrics": {"peak_memory_mb": 10.0},
}


class _FakeProcessor:
    """Counting fake standing in for the LLM boundary."""

    def __init__(
        self,
        content: str,
        *,
        raise_if_called: bool = False,
        finish_reason: str | None = "stop",
    ) -> None:
        self._content = content
        self._finish_reason = finish_reason
        self._raise_if_called = raise_if_called
        self.calls = 0

    async def get_response(
        self,
        *,
        messages: Any,
        model_name: Any,
        max_tokens: Any,
        **kwargs: Any,
    ) -> Any:
        if self._raise_if_called:
            raise AssertionError("get_response called; expected a cache hit")
        self.calls += 1
        return SimpleNamespace(content=self._content, finish_reason=self._finish_reason)


def _install_fake_processor(
    monkeypatch: pytest.MonkeyPatch, processor: _FakeProcessor
) -> None:
    """Route initialize_global_processor through ``processor``."""

    async def fake_initialize() -> Any:
        return processor

    monkeypatch.setattr(
        "gnn.llm.llm_processor.initialize_global_processor", fake_initialize
    )


def _llm_call(
    log: logging.Logger,
    model: str,
    cache_dir: Path,
    context: dict[str, Any],
) -> Coroutine[Any, Any, tuple[str, str]]:
    """Build the ``_run_llm_analysis`` coroutine for one step-24 call."""
    return _run_llm_analysis(
        context, [], {}, log, analysis_model=model, cache_dir=cache_dir
    )


def _cache_json_entries(cache_dir: Path) -> list[Path]:
    """List the on-disk cache entries under ``cache_dir``."""
    return sorted(cache_dir.glob("*.json"))


@pytest.mark.unit
def test_second_identical_run_hits_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An identical second run is served from the cache without an LLM call."""
    cache_dir = tmp_path / ".cache"
    processor = _FakeProcessor(VALID_REPORT)
    _install_fake_processor(monkeypatch, processor)
    log = logging.getLogger("test-step24-cache-hit")

    async def run_twice() -> tuple[tuple[str, str], tuple[str, str]]:
        first = await _llm_call(log, "test-model", cache_dir, ANALYSIS_CONTEXT)
        second = await _llm_call(log, "test-model", cache_dir, ANALYSIS_CONTEXT)
        return first, second

    first, second = asyncio.run(run_twice())

    assert first == (VALID_REPORT, "llm")
    assert second == first
    assert processor.calls == 1


@pytest.mark.unit
def test_cache_entry_persists_across_processor_instances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh processor instance is served the persisted entry from disk."""
    cache_dir = tmp_path / ".cache"
    log = logging.getLogger("test-step24-cache-persist")

    first_processor = _FakeProcessor(VALID_REPORT)
    _install_fake_processor(monkeypatch, first_processor)
    content, source = asyncio.run(
        _llm_call(log, "test-model", cache_dir, ANALYSIS_CONTEXT)
    )
    assert source == "llm"
    assert content == VALID_REPORT
    assert len(_cache_json_entries(cache_dir)) == 1

    second_processor = _FakeProcessor(VALID_REPORT, raise_if_called=True)
    _install_fake_processor(monkeypatch, second_processor)
    cached_content, cached_source = asyncio.run(
        _llm_call(log, "test-model", cache_dir, ANALYSIS_CONTEXT)
    )
    assert cached_source == "llm"
    assert cached_content == content
    assert second_processor.calls == 0


@pytest.mark.unit
def test_model_name_participates_in_cache_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Distinct models over one cache dir produce distinct entries and calls."""
    cache_dir = tmp_path / ".cache"
    processor = _FakeProcessor(VALID_REPORT)
    _install_fake_processor(monkeypatch, processor)
    log = logging.getLogger("test-step24-cache-model")

    first = asyncio.run(_llm_call(log, "test-model", cache_dir, ANALYSIS_CONTEXT))
    second = asyncio.run(_llm_call(log, "other-model", cache_dir, ANALYSIS_CONTEXT))

    assert first == (VALID_REPORT, "llm")
    assert second == (VALID_REPORT, "llm")
    assert processor.calls == 2
    assert len(_cache_json_entries(cache_dir)) == 2


@pytest.mark.unit
def test_invalid_llm_output_is_not_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Output failing the report contract falls back and stores nothing."""
    cache_dir = tmp_path / ".cache"
    processor = _FakeProcessor(INVALID_REPORT)
    _install_fake_processor(monkeypatch, processor)
    log = logging.getLogger("test-step24-cache-invalid")

    content, source = asyncio.run(
        _llm_call(log, "test-model", cache_dir, ANALYSIS_CONTEXT)
    )

    assert source == "rule_based"
    assert content != INVALID_REPORT
    assert _cache_json_entries(cache_dir) == []


@pytest.mark.unit
def test_process_intelligent_analysis_wires_step24_cache_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """process_intelligent_analysis stores and reuses the step-24 cache."""
    summary_dir = tmp_path / "00_pipeline_summary"
    summary_dir.mkdir()
    summary = {
        "overall_status": "SUCCESS",
        "total_duration_seconds": 1.5,
        "steps": [
            {
                "script_name": "01_setup",
                "status": "SUCCESS",
                "duration_seconds": 1.0,
                "exit_code": 0,
                "memory_mb": 10.0,
            }
        ],
        "performance_summary": {"peak_memory_mb": 10.0},
    }
    (summary_dir / "pipeline_execution_summary.json").write_text(json.dumps(summary))

    processor = _FakeProcessor(VALID_REPORT)
    _install_fake_processor(monkeypatch, processor)
    log = logging.getLogger("test-step24-cache-wiring")

    ok = process_intelligent_analysis(
        target_dir=tmp_path,
        output_dir=tmp_path,
        logger=log,
        analysis_model="test-model",
    )
    assert ok is True

    cache_dir = tmp_path / "24_intelligent_analysis_output" / ".cache"
    assert cache_dir.is_dir()
    assert len(_cache_json_entries(cache_dir)) >= 1

    data_path = tmp_path / "24_intelligent_analysis_output" / "analysis_data.json"
    analysis_data = json.loads(data_path.read_text())
    assert analysis_data["analysis_source"] == "llm"
    assert processor.calls == 1

    ok = process_intelligent_analysis(
        target_dir=tmp_path,
        output_dir=tmp_path,
        logger=log,
        analysis_model="test-model",
    )
    assert ok is True
    assert processor.calls == 1


@pytest.mark.parametrize("finish_reason", [None, "length"])
def test_valid_report_text_with_unfinished_native_response_is_not_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, finish_reason: str | None
) -> None:
    processor = _FakeProcessor(VALID_REPORT, finish_reason=finish_reason)
    _install_fake_processor(monkeypatch, processor)
    content, source = asyncio.run(
        _llm_call(
            logging.getLogger("test-stop"), "test-model", tmp_path, ANALYSIS_CONTEXT
        )
    )
    assert source == "rule_based"
    assert _cache_json_entries(tmp_path) == []
