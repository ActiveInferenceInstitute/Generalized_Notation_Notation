"""Telemetry-free import closure for ``gnn.ontology`` (issue #400, item 2).

``gnn.ontology`` logs through ``gnn.utils.observability.structured_logging``.
That module used to import ``psutil`` at module scope, so a minimal install
without the process-telemetry dependency could not import the ontology
package at all. ``psutil`` is now imported lazily by the performance probe.

The closure checks run in a fresh interpreter: the pytest process has already
imported ``psutil`` through the wider test stack, and ``sys.modules["psutil"]
= None`` must be in place before the first ``gnn`` import to model an
environment where the distribution is absent.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.fast

SRC = Path(__file__).resolve().parents[2] / "src"

_BLOCKED_CLOSURE = """
import sys
sys.modules["psutil"] = None  # model a closure without the distribution

import gnn
import gnn.ontology
from gnn.ontology import (
    DECLARED_STATUS_RESOLVED,
    pinned_declared_index,
    resolve_declared_label,
    validate_annotations,
    verify_ontology_pin,
)

assert sys.modules["psutil"] is None, "psutil was imported despite the block"
index, error = pinned_declared_index()
assert error is None, error
for label, identifier in (
    ("HiddenState", "term-hidden-state"),
    ("LikelihoodMapping", "term-likelihood"),
    ("TransitionMatrix", "term-transition"),
    ("Preferences", "term-preference"),
):
    resolution = resolve_declared_label(label, index)
    assert resolution.status == DECLARED_STATUS_RESOLVED, (label, resolution)
    assert resolution.identifier == identifier, (label, resolution)
result = validate_annotations(["A=LikelihoodMapping", "B=TransitionMatrix"])
assert result["invalid_annotations"] == [], result
assert verify_ontology_pin()["verified"] is True

from gnn.utils.observability.structured_logging import StructuredLogger

logger = StructuredLogger("gnn.ontology.closure-check")
with logger.operation_context("probe"):
    pass
assert "probe" not in logger.performance_metrics
print("CLOSURE-OK")
"""

_UNBLOCKED_IMPORT = """
import sys
import gnn.ontology
print("PSUTIL-LOADED" if "psutil" in sys.modules else "PSUTIL-DEFERRED")
"""


def _run_fresh(code: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(SRC), env.get("PYTHONPATH", "")) if part
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env=env,
    )


def test_ontology_imports_and_resolves_terms_without_psutil() -> None:
    result = _run_fresh(_BLOCKED_CLOSURE)
    assert result.returncode == 0, (
        f"rc={result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "CLOSURE-OK" in result.stdout


def test_importing_ontology_defers_psutil_even_when_installed() -> None:
    # ``psutil`` is a declared core dependency, so the test closure has it.
    result = _run_fresh(_UNBLOCKED_IMPORT)
    assert result.returncode == 0, result.stderr
    assert "PSUTIL-DEFERRED" in result.stdout


def test_telemetry_is_recorded_when_psutil_is_installed() -> None:
    import psutil

    from gnn.utils.observability import structured_logging

    logger = structured_logging.StructuredLogger("gnn.ontology.telemetry-check")
    logger._start_performance_tracking("op")
    metrics = logger.performance_metrics["op"]
    expected_rss = psutil.Process().memory_info().rss / 1024 / 1024
    assert metrics.memory_start_mb > 0
    # Same process, same probe: the lazily obtained handle reads this process.
    assert abs(metrics.memory_start_mb - expected_rss) < 512
    logger._end_performance_tracking("op")
    assert metrics.end_time is not None
    assert metrics.memory_end_mb > 0


def test_missing_psutil_degrades_tracking_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gnn.utils.observability import structured_logging

    monkeypatch.setitem(sys.modules, "psutil", None)
    with pytest.raises(ImportError):
        structured_logging._process_handle()

    logger = structured_logging.StructuredLogger("gnn.ontology.degraded-check")
    seen: list[Any] = []
    monkeypatch.setattr(logger.base_logger, "debug", seen.append)
    with logger.operation_context("degraded"):
        pass
    assert "degraded" not in logger.performance_metrics
    assert any("Performance tracking start failed" in str(m) for m in seen)
    # An end call for an operation that never started is a no-op.
    logger._end_performance_tracking("degraded")
